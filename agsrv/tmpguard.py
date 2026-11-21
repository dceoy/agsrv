"""Per-turn temporary storage and free-space checks.

Design and threat model
-----------------------
Each turn gets a private directory ``.agsrv-tmp-<random>`` inside its working
directory (the *cwd*). Writers that share the cwd can list it, rename or remove
entries, substitute real directories or symlinks, and change ancestors. Neither
random names nor the workspace lease establish ownership of a directory entry,
so ownership is decided by an open descriptor, never by a name:

* **Admission.** The workspace root is opened once at startup. An accepted cwd
  is converted to relative components and opened by descending from the root
  descriptor (``O_DIRECTORY | O_NOFOLLOW`` per component). The resulting cwd
  descriptor is held by the request/job (``Admitted``). After the last wait the
  same walk is repeated and compared with ``os.path.samestat``; all later
  server mutations use the held descriptor, so a later symlink swap cannot
  redirect them.
* **Creation in staging.** A service-controlled staging directory (one per
  workspace mount, opened and validated at startup: directory, owned by the
  service, mode ``0700``) is outside the writers' reach. The turn directory is
  created there with ``mkdir`` and a random name, made ``0700``, opened, and
  verified through the descriptor while still private. It is then *published*
  into the cwd with ``renameat2(RENAME_NOREPLACE)``, which never overwrites even
  an empty directory.
* **Reclaim.** Cleanup renames the public entry back into staging (no-replace),
  checks it against the held descriptor with ``os.path.samestat``, and only on a
  match empties the tree *through the held child descriptor* (descriptor-safe
  ``shutil.rmtree(entry, dir_fd=child_fd)`` per entry). The only name-based
  destructive step is the final ``rmdir`` of the now-empty directory under the
  protected staging parent, so even a reachable staging directory can cost at
  most one empty directory. A mismatching entry is never traversed or deleted;
  it is renamed back to the public name if that is free, otherwise it is kept
  in staging and its exact location is logged. Cleanup runs in one worker thread
  and is shielded from cancellation.

Residual risks (see also the README): children receive a ``TMPDIR`` path string
and resolve it themselves, so a writer can redirect *children* by replacing the
leaf or an ancestor; server-side operations stay attached to the held
descriptors. A descriptor identifies the object even if an ancestor is moved.
Mounting filesystems, service compromise and a same-UID child that can reach the
staging directory (e.g. via ``/proc``) are outside the model. A service crash
can leave a public or staged entry; there is deliberately no name-based
scavenger. The staging directory, workspace root and their ancestors are trusted.

Startup is fail-fast: the workspace root's mount must have a staging directory,
every staging directory must pass a rename round trip into its mount root (which
proves the same *mount*, not merely the same ``st_dev``), and staging below
``/tmp`` and other child-writable roots is rejected.
"""

import asyncio
import ctypes
import errno
import functools
import logging
import os
import secrets
import shutil
import stat
import tempfile
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from agsrv.config import Settings

logger = logging.getLogger(__name__)

_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_PUBLIC = ".agsrv-tmp-"
_STAGED = ".agsrv-stg-"
_RECOVERY = ".agsrv-rec-"
_RENAME_NOREPLACE = 1
_PRIVATE_MODE = 0o700
_NAME_ATTEMPTS = 8
_DRAIN_TIMEOUT = 60.0
_PROBE = ".agsrv-probe-"


def _forbidden_stage_roots() -> tuple[Path, ...]:
    """Return roots that agent children can normally write to.

    Returns:
        Directories under which a staging directory is refused.
    """
    roots = [
        Path("/tmp"),  # ruff: ignore[hardcoded-temp-file]
        Path("/var/tmp"),  # ruff: ignore[hardcoded-temp-file]
        Path("/dev/shm"),  # ruff: ignore[hardcoded-temp-file]
        Path(tempfile.gettempdir()),
    ]
    return tuple(dict.fromkeys(r.resolve() for r in roots))


class TmpdirError(Exception):
    """Per-turn directory setup failed."""


@functools.cache
def _renameat2_fn() -> "Callable[..., int]":
    """Load libc's ``renameat2`` with a declared signature.

    Returns:
        The foreign function.

    Raises:
        OSError: ``ENOSYS`` if libc has no ``renameat2``.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        fn = libc.renameat2
    except AttributeError as exc:
        raise OSError(errno.ENOSYS, "renameat2 is unavailable in libc") from exc
    fn.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    fn.restype = ctypes.c_int
    return fn


def _rename_noreplace(src_fd: int, src: str, dst_fd: int, dst: str) -> None:
    """Move an entry between directories without ever replacing the target.

    Raises:
        OSError: On any failure (``EEXIST`` if the destination exists, ``EXDEV``
            across mounts, ``ENOSYS``/``EINVAL`` if unsupported).
    """
    fn = _renameat2_fn()
    if fn(src_fd, os.fsencode(src), dst_fd, os.fsencode(dst), _RENAME_NOREPLACE) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), dst)


def _walk(root_fd: int, parts: tuple[str, ...]) -> int:
    """Open a descendant by descending one no-follow component at a time.

    Returns:
        A new descriptor owned by the caller (``dup`` of the root for no parts).

    Raises:
        TmpdirError: If a component is unsafe or cannot be opened.
    """
    if any(p in {"", ".", ".."} or "/" in p for p in parts):
        msg = f"unsafe path components {parts!r}"
        raise TmpdirError(msg)
    fd = os.dup(root_fd)
    part = ""
    try:
        for part in parts:
            next_fd = os.open(part, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except OSError as exc:
        os.close(fd)
        msg = f"cannot open working directory component {part!r} safely: {exc}"
        raise TmpdirError(msg) from exc
    return fd


@dataclass
class Admitted:
    """A request's working directory bound to an open descriptor.

    Attributes:
        path: Canonical absolute path (for policy, leases and child strings).
        parts: Components below the workspace root.
        fd: Held cwd descriptor, valid until :meth:`close`.
        stage_fd: Borrowed staging descriptor for this cwd's mount.
    """

    anchors: "Anchors"
    path: Path
    parts: tuple[str, ...]
    fd: int
    stage_fd: int
    _closed: bool = field(default=False, repr=False)

    def revalidate(self) -> None:
        """Re-walk from the root and require the same directory object.

        Raises:
            TmpdirError: If the path no longer leads to the admitted directory.
        """
        fd = _walk(self.anchors.root_fd, self.parts)
        try:
            same = os.path.samestat(os.fstat(fd), os.fstat(self.fd))
        finally:
            os.close(fd)
        if not same:
            msg = f"working directory {self.path} changed after admission"
            raise TmpdirError(msg)

    def close(self) -> None:
        """Release the descriptor (idempotent)."""
        if not self._closed:
            self._closed = True
            os.close(self.fd)
            self.anchors.release()


def _open_trusted(
    settings: "Settings", root_path: Path
) -> tuple[int, list[tuple[Path, int]]]:
    """Open the workspace root and every staging directory (all or nothing).

    Returns:
        The root descriptor and ``(mount root, staging descriptor)`` pairs.

    Raises:
        TmpdirError: If a trusted location is unusable.
        OSError: Never; ``OSError`` is converted to ``TmpdirError``.
    """
    opened: list[int] = []
    stages: list[tuple[Path, int]] = []
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        root_fd = os.open(root_path, _DIR_FLAGS)
        opened.append(root_fd)
        for mount, stage in settings.tmp_staging_dirs:
            fd = _open_stage(mount.resolve(), stage, root_path)
            opened.append(fd)
            stages.append((mount.resolve(), fd))
        _renameat2_fn()
        if not any(root_path.is_relative_to(mount) for mount, _ in stages):
            msg = (
                f"no staging directory is configured for the workspace root "
                f"{root_path}: set AGSRV_TMP_STAGING_DIRS="
                f"<mount root>=<staging directory> (see the README)"
            )
            raise TmpdirError(msg)  # ruff: ignore[raise-within-try]
    except (OSError, TmpdirError) as exc:
        for fd in opened:
            os.close(fd)
        if isinstance(exc, TmpdirError):
            raise
        msg = f"cannot open trusted location: {exc}"
        raise TmpdirError(msg) from exc
    return root_fd, stages


class Anchors:
    """Trusted descriptors held for the life of the server."""

    def __init__(self, settings: "Settings") -> None:
        """Remember configuration; descriptors are opened by :meth:`open`."""
        self.settings = settings
        self.root_fd = -1
        self.root_path = settings.workspace_root.resolve()
        self.stages: list[tuple[Path, int]] = []
        self.is_open = False
        self._live = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self.cleanups: set[asyncio.Task[None]] = set()

    def open(self) -> None:
        """Open and validate the workspace root and every staging directory.

        Idempotent. Called at startup and lazily by :meth:`admit`.

        Raises:
            TmpdirError: If a trusted location is unusable.
        """
        if self.is_open:
            return
        if not shutil.rmtree.avoids_symlink_attacks:
            msg = "this platform's shutil.rmtree is not symlink-attack resistant"
            raise TmpdirError(msg)
        self.root_fd, stages = _open_trusted(self.settings, self.root_path)
        self.stages = sorted(stages, key=lambda s: len(s[0].parts), reverse=True)
        self.is_open = True

    def _stage_for(self, cwd: Path) -> int:
        # Startup guarantees a mount root covering the workspace root, hence
        # every cwd below it.
        return next(fd for mount, fd in self.stages if cwd.is_relative_to(mount))

    def admit(self, cwd: Path) -> Admitted:
        """Bind an accepted canonical ``cwd`` to a held descriptor.

        Args:
            cwd: Canonical path already accepted by ``security.resolve_cwd``.

        Returns:
            The admitted directory; the caller must close it.

        Raises:
            HTTPException: 503 if the directory cannot be opened safely.
        """
        try:
            self.open()
            parts = cwd.relative_to(self.root_path).parts
            stage_fd = self._stage_for(cwd)
            fd = _walk(self.root_fd, parts)
        except (TmpdirError, ValueError) as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Cannot prepare turn temporary directory: {exc}",
            ) from exc
        self._live += 1
        self._idle.clear()
        return Admitted(self, cwd, parts, fd, stage_fd)

    def release(self) -> None:
        """Record that one admitted descriptor was closed."""
        self._live -= 1
        if self._live == 0:
            self._idle.set()

    async def aclose(self) -> None:
        """Wait for every turn owner, then close the shared descriptors."""
        try:
            async with asyncio.timeout(_DRAIN_TIMEOUT):
                while self.cleanups:
                    await asyncio.gather(*self.cleanups, return_exceptions=True)
                await self._idle.wait()
        except TimeoutError:
            logger.exception("Turn owners did not finish; keeping anchors open")
            return
        if self.is_open:
            for fd in (self.root_fd, *(fd for _, fd in self.stages)):
                os.close(fd)
            self.root_fd = -1
            self.stages = []
            self.is_open = False


def _check_stage(stage: Path, mount: Path, info: os.stat_result) -> None:
    """Validate an opened staging directory's ownership, mode and filesystem.

    Raises:
        TmpdirError: If it is not service-owned ``0700`` on the mount's filesystem.
    """
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != _PRIVATE_MODE:
        msg = f"staging directory {stage} must be owned by the service, mode 0700"
        raise TmpdirError(msg)
    if mount.stat().st_dev != info.st_dev:
        msg = f"staging directory {stage} is not on the filesystem of {mount}"
        raise TmpdirError(msg)


def _probe_mount(stage: Path, mount: Path, stage_fd: int) -> None:
    """Prove staging and ``mount`` share a mount with a rename round trip.

    ``st_dev`` is equal across bind mounts of one filesystem, yet ``renameat2``
    between them fails with ``EXDEV``. A probe directory is therefore moved from
    staging into the mount root and back, using the very syscall publication uses.

    Raises:
        TmpdirError: If the round trip fails (for example ``EXDEV``).
    """
    name = f"{_PROBE}{os.urandom(16).hex()}"
    mount_fd = os.open(mount, _DIR_FLAGS)  # OSError is reported by the caller
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        os.mkdir(name, _PRIVATE_MODE, dir_fd=stage_fd)
        in_mount = False
        try:
            _rename_noreplace(stage_fd, name, mount_fd, name)
            in_mount = True
            _rename_noreplace(mount_fd, name, stage_fd, name)
            in_mount = False
        finally:  # leave nothing behind, wherever the probe ended up
            with suppress(OSError):
                os.rmdir(name, dir_fd=mount_fd if in_mount else stage_fd)
    except OSError as exc:
        msg = (
            f"staging directory {stage} is not on the same mount as {mount} "
            f"(rename round trip failed: {exc}); a bind mount or a different "
            "mount needs its own staging directory"
        )
        raise TmpdirError(msg) from exc
    finally:
        os.close(mount_fd)


def _open_stage(mount: Path, stage: Path, root: Path) -> int:
    """Open and validate a staging directory.

    Returns:
        The held descriptor.

    Raises:
        TmpdirError: If it is unsafe: inside the workspace or a child-writable
            root, not a ``0700`` directory owned by the service, or not on the
            same mount as its mount root.
    """
    stage = stage.resolve()
    if stage.is_relative_to(root):
        msg = f"staging directory {stage} must be outside the workspace root"
        raise TmpdirError(msg)
    for shared in _forbidden_stage_roots():
        if stage.is_relative_to(shared):
            msg = (
                f"staging directory {stage} is below {shared}, which agent "
                "children can write to; choose a service-private location"
            )
            raise TmpdirError(msg)
    try:
        fd = os.open(stage, _DIR_FLAGS)
    except OSError as exc:
        msg = f"cannot open staging directory {stage}: {exc}"
        raise TmpdirError(msg) from exc
    try:
        _check_stage(stage, mount, os.fstat(fd))
        _probe_mount(stage, mount, fd)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _free(fd: int) -> int:
    info = os.fstatvfs(fd)
    return info.f_bavail * info.f_frsize


def check_space(settings: "Settings", anchors: Anchors, cwd: Admitted) -> None:
    """Reject a turn if either enabled free-space threshold is unmet.

    The filesystems actually used are measured: ``/tmp``, the held workspace
    root and the held cwd (skipped when it is the root).

    Raises:
        HTTPException: If free space cannot be established or is insufficient.
    """
    targets: list[tuple[str, int | None, int]] = [
        ("/tmp", None, settings.tmp_min_free_bytes),  # ruff: ignore[hardcoded-temp-file]
        (str(anchors.root_path), anchors.root_fd, settings.workspace_min_free_bytes),
    ]
    if cwd.path != anchors.root_path:
        targets.append((str(cwd.path), cwd.fd, settings.workspace_min_free_bytes))
    for label, fd, required in targets:
        if not required:
            continue
        try:
            if fd is None:
                info = os.statvfs(label)
                free = info.f_bavail * info.f_frsize
            else:
                free = _free(fd)
        except OSError as exc:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Cannot check free space on {label}: {exc}",
            ) from exc
        if free < required:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                f"Insufficient free space on {label}: {free} bytes free, "
                f"{required} bytes required",
            )


@dataclass
class _Turn:
    """Ownership record for one published turn directory.

    ``stage_fd``, ``cwd_fd`` and ``child_fd`` are private duplicates owned by the
    record (closed by :func:`_reclaim`), so a cleanup worker that outlives its
    owner never operates on a descriptor number that was closed and reused.
    """

    stage_fd: int
    cwd_fd: int
    child_fd: int
    public: str
    path: Path
    where: str


def _private_rollback(stage_fd: int, name: str, child_fd: int) -> None:
    """Remove a still-private directory by its staged name."""
    if child_fd >= 0:
        os.close(child_fd)
    try:
        os.rmdir(name, dir_fd=stage_fd)
    except OSError:
        logger.exception("Failed to roll back private staged directory %s", name)


def _mkdir_private(stage_fd: int) -> str:
    """Create an empty ``0700`` directory under an unused random staging name.

    Returns:
        The staged name (ownership is proven by the private parent).

    Raises:
        TmpdirError: If no directory could be created.
    """
    for _ in range(_NAME_ATTEMPTS):
        name = f"{_STAGED}{secrets.token_hex(16)}"
        try:
            os.mkdir(name, _PRIVATE_MODE, dir_fd=stage_fd)
        except FileExistsError:
            continue  # never touch an existing entry
        except OSError as exc:
            msg = f"cannot create staged directory: {exc}"
            raise TmpdirError(msg) from exc
        return name
    msg = "could not pick an unused staging name"
    raise TmpdirError(msg)


def _open_private(stage_fd: int, name: str) -> int:
    """Fix the mode, open and verify a staged directory.

    Returns:
        The verified directory descriptor.

    Raises:
        TmpdirError: If the opened object is not a ``0700`` directory.
    """
    # A restrictive umask can leave mode 000; fix it while still private.
    os.chmod(name, _PRIVATE_MODE, dir_fd=stage_fd)
    child_fd = os.open(name, _DIR_FLAGS, dir_fd=stage_fd)
    info = os.fstat(child_fd)
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) != _PRIVATE_MODE:
        os.close(child_fd)
        msg = f"staged directory {name} has unexpected type or mode"
        raise TmpdirError(msg)
    return child_fd


def _setup(adm: Admitted) -> _Turn:
    """Create the directory privately in staging and publish it atomically.

    Returns:
        The ownership record; the caller must reclaim it.

    Raises:
        TmpdirError: If any step fails; partial state is rolled back first.
    """
    stage_fd = adm.stage_fd
    if os.fstat(adm.fd).st_dev != os.fstat(stage_fd).st_dev:
        msg = f"staging is on a different filesystem than {adm.path}"
        raise TmpdirError(msg)
    name = _mkdir_private(stage_fd)
    child_fd = -1
    owned: list[int] = []
    public = f"{_PUBLIC}{secrets.token_hex(16)}"
    try:
        # One at a time: if the second dup fails the first must still be closed.
        owned.append(os.dup(stage_fd))  # ruff: ignore[repeated-append]
        owned.append(os.dup(adm.fd))
        child_fd = _open_private(stage_fd, name)
        _rename_noreplace(stage_fd, name, adm.fd, public)
    except BaseException as exc:
        _private_rollback(stage_fd, name, child_fd)
        for fd in owned:
            os.close(fd)
        if isinstance(exc, OSError):
            msg = f"cannot publish into {adm.path}: {exc}"
            raise TmpdirError(msg) from exc
        raise
    path = adm.path / public
    return _Turn(owned[0], owned[1], child_fd, public, path, str(path))


def _capture(turn: _Turn) -> str | None:
    """Move the public entry into staging under a fresh recovery name.

    Returns:
        The recovery name, or ``None`` if nothing was captured.
    """
    for _ in range(_NAME_ATTEMPTS):
        recovery = f"{_RECOVERY}{secrets.token_hex(16)}"
        try:
            _rename_noreplace(turn.cwd_fd, turn.public, turn.stage_fd, recovery)
        except FileExistsError:
            continue
        except FileNotFoundError:
            logger.warning(
                "Turn temporary directory was moved away; skipping cleanup: %s",
                turn.path,
            )
            return None
        except OSError:
            logger.exception("Could not reclaim turn directory %s", turn.path)
            return None
        return recovery
    logger.error("Could not pick a recovery name; %s is retained", turn.path)
    return None


def _make_removable(child_fd: int) -> None:
    """Let read-only subdirectories be emptied (descriptor-based chmod only)."""
    os.fchmod(child_fd, _PRIVATE_MODE)
    for _top, _dirs, _files, dirfd in os.fwalk(".", dir_fd=child_fd):
        with suppress(OSError):
            os.fchmod(dirfd, _PRIVATE_MODE)


def _empty_through(child_fd: int) -> None:
    """Delete everything inside the held directory, never following names out."""
    _make_removable(child_fd)
    with os.scandir(child_fd) as entries:
        children = [(e.name, e.is_dir(follow_symlinks=False)) for e in entries]
    for name, is_dir in children:
        if is_dir:
            shutil.rmtree(name, dir_fd=child_fd)
        else:
            os.unlink(name, dir_fd=child_fd)


def _reclaim_verified(turn: _Turn) -> None:
    recovery = _capture(turn)
    if recovery is None:
        return
    turn.where = f"staging entry {recovery}"
    try:
        captured = os.stat(recovery, dir_fd=turn.stage_fd, follow_symlinks=False)
        ours = stat.S_ISDIR(captured.st_mode) and os.path.samestat(
            captured, os.fstat(turn.child_fd)
        )
    except OSError:
        logger.exception(
            "Could not inspect %s; retaining the unverified entry as %s",
            turn.path,
            turn.where,
        )
        return
    if not ours:
        try:
            _rename_noreplace(turn.stage_fd, recovery, turn.cwd_fd, turn.public)
        except OSError:
            logger.exception(
                "Turn directory %s was replaced by a foreign entry; deleting "
                "nothing. It could not be restored and is retained as %s",
                turn.path,
                turn.where,
            )
        else:
            logger.error(
                "Turn directory %s was replaced by a foreign entry; deleting "
                "nothing and restored it",
                turn.path,
            )
        return
    # Contents go through the held descriptor; the sole name-based removal is
    # the final rmdir of the (now empty) verified directory under staging.
    _empty_through(turn.child_fd)
    os.rmdir(recovery, dir_fd=turn.stage_fd)


def _reclaim(turn: _Turn) -> None:
    """Reclaim a turn directory; closes the child descriptor. Never raises."""
    try:
        _reclaim_verified(turn)
    except Exception:  # cleanup must never mask the turn result
        logger.exception(
            "Failed to remove turn temporary directory %s; last known location: %s",
            turn.path,
            turn.where,
        )
    finally:
        for fd in (turn.child_fd, turn.cwd_fd, turn.stage_fd):
            os.close(fd)


async def _finish(anchors: Anchors, turn: _Turn) -> None:
    """Run cleanup in a worker; survive repeated cancellation of the owner.

    Raises:
        asyncio.CancelledError: After cleanup finished, if the owner was cancelled.
    """
    task = asyncio.ensure_future(asyncio.to_thread(_reclaim, turn))
    anchors.cleanups.add(task)
    task.add_done_callback(anchors.cleanups.discard)
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            if task.done():  # the cleanup itself was cancelled (loop shutdown)
                break
        else:
            break
    if cancelled:
        raise asyncio.CancelledError


@asynccontextmanager
async def turn_tmpdir(
    settings: "Settings", anchors: Anchors, adm: Admitted
) -> "AsyncGenerator[Path, None]":
    """Create a private per-turn directory in the cwd and reclaim it afterward.

    Call after the last workspace/capacity wait. Revalidates the admitted cwd,
    checks free space on the filesystems used, then creates and publishes.

    Args:
        settings: Server settings.
        anchors: Trusted descriptors.
        adm: Admitted working directory (held by the caller).

    Yields:
        The canonical absolute path to hand to the child as ``TMPDIR``.

    Raises:
        HTTPException: 503 if the directory cannot be created safely or free
            space is insufficient.
    """
    try:
        adm.revalidate()
    except TmpdirError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Cannot prepare turn temporary directory: {exc}",
        ) from exc
    check_space(settings, anchors, adm)
    try:
        turn = _setup(adm)  # publication and ownership: no await in between
    except TmpdirError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Cannot prepare turn temporary directory: {exc}",
        ) from exc
    try:
        yield turn.path
    finally:
        await _finish(anchors, turn)
