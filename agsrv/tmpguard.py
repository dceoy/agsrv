"""Per-turn temporary storage and free-space checks."""

import logging
import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from collections.abc import Generator

    from agsrv.config import Settings

logger = logging.getLogger(__name__)


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


def _cleanup(parent_fd: int, child_fd: int, name: str, path: Path) -> None:
    try:  # ruff: ignore[too-many-statements-in-try-clause]
        identity = os.fstat(child_fd)
        current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
            logger.warning(
                "Turn temporary directory changed; skipping cleanup: %s", path
            )
            return
        for entry in os.scandir(child_fd):
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.name, dir_fd=child_fd)
            else:
                os.unlink(entry.name, dir_fd=child_fd)
        os.rmdir(name, dir_fd=parent_fd)
    except Exception:  # cleanup must never mask the turn result
        logger.exception("Failed to remove turn temporary directory %s", path)


@contextmanager
def turn_tmpdir(cwd: Path) -> "Generator[Path, None, None]":
    """Create a private directory for a turn and remove it afterward.

    Yields:
        The directory inside the working directory.
    """
    name = f".agsrv-tmp-{uuid4().hex}"
    path = cwd / name
    parent_fd = os.open(cwd, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        os.chmod(name, 0o700, dir_fd=parent_fd)
        child_fd = os.open(
            name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        try:
            yield path
        finally:
            _cleanup(parent_fd, child_fd, name, path)
            os.close(child_fd)
    finally:
        os.close(parent_fd)
