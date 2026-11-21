"""Per-turn temporary storage and free-space checks."""

import logging
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


def check_space(settings: "Settings") -> None:
    """Reject a turn if either enabled free-space threshold is unmet.

    Raises:
        HTTPException: If free space cannot be established or is insufficient.
    """
    for path, required in (
        (Path("/tmp"), settings.tmp_min_free_bytes),  # noqa: S108
        (settings.workspace_root, settings.workspace_min_free_bytes),
    ):
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


@contextmanager
def turn_tmpdir(cwd: Path) -> "Generator[Path, None, None]":
    """Create a private directory for a turn and remove it afterward.

    Yields:
        The directory inside the working directory.
    """
    path = cwd / f".agsrv-tmp-{uuid4().hex}"
    path.mkdir(mode=0o700)
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except OSError:
            logger.exception("Failed to remove turn temporary directory %s", path)
