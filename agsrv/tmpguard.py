"""Per-turn temporary storage and free-space checks.

Design and threat model
-----------------------
Each turn gets a private directory ``.agsrv-tmp-<random>`` inside its working
directory. Other writers may share that parent directory and can rename or
replace entries in it, so no pathname is trusted after it has been used once:

* The parent descriptor is obtained by walking from the workspace-root
  descriptor one component at a time (``O_DIRECTORY | O_NOFOLLOW``). A symlink
  swapped into any ancestor makes the walk fail before anything is created, and
  the walk can never leave the workspace root.
* The directory is created with a ``dir_fd``-relative ``mkdir`` under a random
  128-bit name (atomic; an existing entry is never reused), then opened at once
  and its ``(st_dev, st_ino)`` recorded.
* Removal renames the entry through the parent descriptor to a fresh tombstone
  name, verifies the tombstone still has the recorded identity, and only then
  deletes the tree with ``dir_fd``-relative operations anchored on the held
  descriptor. A mismatch deletes nothing.

Residual risks: children receive the directory as a path string (unavoidable),
so a hostile writer can redirect *children*; it cannot redirect server-side
creation or deletion. A writer that enumerates the parent and swaps the entry
in the instant between ``mkdir`` and the following ``open`` is not detectable by
identity alone; the opened directory must therefore also be empty and owned by
the server user, otherwise nothing is deleted. The workspace root path itself
(and everything above it) is administrator-controlled and trusted.
"""

import asyncio
import logging
import os
import secrets
import shutil
import stat
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from agsrv.config import Settings

logger = logging.getLogger(__name__)

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_PREFIX = ".agsrv-tmp-"
_TOMBSTONE = ".agsrv-del-"


def check_space(settings: "Settings", cwd: Path | None = None) -> None:
    """Reject a turn if either enabled free-space threshold is unmet.

    Raises:
        HTTPException: If free space cannot be established or is insufficient.
    """
    paths = [
        (Path("/tmp"), settings.tmp_min_free_bytes),  # ruff: ignore[hardcoded-temp-file]
        (settings.workspace_root, settings.workspace_min_free_bytes),
    ]
    if cwd is not None and cwd != settings.workspace_root:
        paths.append((cwd, settings.workspace_min_free_bytes))
    for path, required in paths:
        if required:
            try:
                free = shutil.disk_usage(path).free
            except OSError as exc:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    f"Cannot check free space on {path}: {exc}",
                ) from exc
            if free < required:
                raise HTTPException(
                    status.HTTP_503_SERVICE_UNAVAILABLE,
                    f"Insufficient free space on {path}: {free} bytes free, "
                    f"{required} bytes required",
                )


class TmpdirError(Exception):
    """Per-turn directory setup failed."""


@dataclass
class _Turn:
    parent_fd: int
    child_fd: int
    name: str
    identity: tuple[int, int]
    path: Path


def _walk(root: Path, cwd: Path) -> tuple[int, Path]:
    """Open ``cwd`` by descending from the workspace root without pathnames.

    Returns:
        The open directory descriptor and the canonical path of the directory.

    Raises:
        TmpdirError: If ``cwd`` is not inside the root or any component fails.
    """
    root = Path(os.path.realpath(root))
    cwd = Path(os.path.realpath(cwd))
    try:
        parts = cwd.relative_to(root).parts
    except ValueError as exc:
        msg = f"working directory {cwd} is outside the workspace root"
        raise TmpdirError(msg) from exc
    fd = -1
    try:
        fd = os.open(root, _DIR_FLAGS)
        for part in parts:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except OSError as exc:
        if fd >= 0:
            os.close(fd)
        msg = f"cannot open working directory {cwd} safely: {exc}"
        raise TmpdirError(msg) from exc
    return fd, root.joinpath(*parts)


def _purge(root_fd: int) -> None:
    """Delete everything below ``root_fd`` using only descriptor-relative calls.

    Every directory is opened with ``O_NOFOLLOW`` relative to its parent's
    descriptor, so symlinks are unlinked, never followed.
    """
    top = os.listdir(root_fd)
    stack: list[tuple[int, str | None, list[str]]] = [(root_fd, None, top)]
    try:
        while stack:
            fd, name, pending = stack[-1]
            if pending:
                entry = pending.pop()
                if stat.S_ISDIR(
                    os.stat(entry, dir_fd=fd, follow_symlinks=False).st_mode
                ):
                    child = os.open(entry, _DIR_FLAGS, dir_fd=fd)
                    with suppress(PermissionError):
                        os.fchmod(child, 0o700)  # read-only trees must stay removable
                    listing: list[str] = []
                    stack.append((child, entry, listing))  # closed by `finally`
                    listing.extend(os.listdir(child))  # ruff: ignore[os-listdir]
                else:
                    os.unlink(entry, dir_fd=fd)
            else:
                stack.pop()
                if name is not None:
                    os.close(fd)
                    os.rmdir(name, dir_fd=stack[-1][0])
    finally:
        for fd, name, _ in stack:
            if name is not None:
                os.close(fd)


def _same(turn: _Turn, tomb: str) -> bool:
    found = os.stat(tomb, dir_fd=turn.parent_fd, follow_symlinks=False)
    return stat.S_ISDIR(found.st_mode) and (found.st_dev, found.st_ino) == turn.identity


def _retire(turn: _Turn, tomb: str) -> bool:
    """Move the entry to ``tomb`` and confirm it is the directory we created.

    Returns:
        Whether the tombstone is verified to be ours and may be deleted.
    """
    try:
        os.rename(turn.name, tomb, src_dir_fd=turn.parent_fd, dst_dir_fd=turn.parent_fd)
    except FileNotFoundError:
        if os.fstat(turn.child_fd).st_nlink:
            logger.warning(
                "Turn temporary directory was moved away; skipping cleanup: %s",
                turn.path,
            )
        return False
    if _same(turn, tomb):
        return True
    _restore_foreign(turn.parent_fd, tomb, turn.name, turn.path)
    return False


def _delete_verified(turn: _Turn) -> None:
    tomb = f"{_TOMBSTONE}{secrets.token_hex(16)}"
    if _retire(turn, tomb):
        _purge(turn.child_fd)
        if _same(turn, tomb):  # rmdir only ever removes an empty directory
            os.rmdir(tomb, dir_fd=turn.parent_fd)


def _remove(turn: _Turn) -> None:
    """Remove a turn directory if, and only if, it is still the one created.

    Closes both descriptors. Never raises; problems are logged.
    """
    try:
        _delete_verified(turn)
    except Exception:  # cleanup must never mask the turn result
        logger.exception("Failed to remove turn temporary directory %s", turn.path)
    finally:
        os.close(turn.child_fd)
        os.close(turn.parent_fd)


def _restore_foreign(parent_fd: int, tomb: str, name: str, path: Path) -> None:
    """Leave a foreign entry untouched; put it back if its name is free."""
    logger.error(
        "Turn temporary directory was replaced; deleting nothing and leaving the "
        "foreign entry in place: %s",
        path,
    )
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        os.rename(tomb, name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    else:
        logger.error("Foreign entry remains under tombstone name %s", tomb)


def _open_fresh(parent_fd: int, name: str, path: Path) -> _Turn:
    """Open the just-created directory and record its identity.

    Returns:
        A handle owning ``parent_fd`` and the new child descriptor.

    Raises:
        TmpdirError: If the entry is not an empty directory we own, which means
            it was swapped after ``mkdir``. Nothing is deleted in that case.
    """
    # A restrictive umask can leave the new directory unopenable (mode 000).
    os.chmod(name, 0o700, dir_fd=parent_fd, follow_symlinks=False)
    child_fd = os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    try:
        info = os.fstat(child_fd)
        empty = not os.listdir(child_fd)  # ruff: ignore[os-listdir]
        fresh = info.st_uid == os.geteuid() and empty
    except BaseException:
        os.close(child_fd)
        raise
    if not fresh:
        os.close(child_fd)
        logger.error("Unexpected entry at new turn directory %s; leaving it", path)
        msg = f"{path} was replaced during creation"
        raise TmpdirError(msg)
    return _Turn(parent_fd, child_fd, name, (info.st_dev, info.st_ino), path)


def _create(root: Path, cwd: Path) -> _Turn:
    """Create the directory and return its descriptor-anchored handle.

    Returns:
        The handle owning both descriptors.

    Raises:
        TmpdirError: If any step fails; partial state is rolled back first.
    """
    parent_fd, parent = _walk(root, cwd)
    name = f"{_PREFIX}{secrets.token_hex(16)}"
    path = parent / name
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        msg = f"cannot create {path}: {exc}"
        raise TmpdirError(msg) from exc
    try:
        turn = _open_fresh(parent_fd, name, path)
    except BaseException as exc:
        if not isinstance(exc, TmpdirError):
            try:  # no identity yet: rmdir can only ever remove an empty directory
                os.rmdir(name, dir_fd=parent_fd)
            except OSError:
                logger.exception("Failed to roll back %s", path)
        os.close(parent_fd)
        if isinstance(exc, OSError | NotImplementedError):
            msg = f"cannot open {path}: {exc}"
            raise TmpdirError(msg) from exc
        raise
    try:
        os.fchmod(turn.child_fd, 0o700)
    except BaseException as exc:
        _remove(turn)  # identity is known: the full anchored removal applies
        if isinstance(exc, OSError):
            msg = f"cannot set mode of {path}: {exc}"
            raise TmpdirError(msg) from exc
        raise
    return turn


@asynccontextmanager
async def turn_tmpdir(root: Path, cwd: Path) -> "AsyncGenerator[Path, None]":
    """Create a private directory for a turn and remove it afterward.

    Args:
        root: Workspace root that contains ``cwd``.
        cwd: Canonical working directory of the turn.

    Yields:
        The directory inside the working directory (a canonical path string
        source for children; server-side removal never uses it).

    Raises:
        HTTPException: 503 if the directory cannot be created safely.
    """
    try:
        turn = _create(root, cwd)
    except TmpdirError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Cannot prepare turn temporary directory: {exc}",
        ) from exc
    try:
        yield turn.path
    finally:
        # The worker closes the descriptors itself, so they are released even
        # if this await is cancelled again.
        await asyncio.to_thread(_remove, turn)
