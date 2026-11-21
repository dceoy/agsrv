"""Race and failure tests for the staging/no-replace per-turn directory design.

Everything here uses real directories and real ``renameat2`` calls. Writers act
inside hooks placed at exact syscall boundaries, so no test depends on timing.
"""

from __future__ import annotations

import asyncio
import ctypes
import errno
import logging
import os
import shutil
import stat
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi import HTTPException

from agsrv import tmpguard
from agsrv.config import Settings
from agsrv.tmpguard import Admitted, Anchors, TmpdirError, check_space, turn_tmpdir
from tests.conftest import statvfs_stub, vfs

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable


def stg(n: int) -> str:
    """Return the n-th deterministic staged name."""
    return f"{tmpguard._STAGED}{n:032x}"  # pyright: ignore[reportPrivateUsage]


def pub(n: int) -> str:
    """Return the n-th deterministic public name."""
    return f"{tmpguard._PUBLIC}{n:032x}"  # pyright: ignore[reportPrivateUsage]


def rec(n: int) -> str:
    """Return the n-th deterministic recovery name."""
    return f"{tmpguard._RECOVERY}{n:032x}"  # pyright: ignore[reportPrivateUsage]


@dataclass
class Env:
    """A workspace root with a held staging directory."""

    root: Path
    stage: Path
    settings: Settings
    anchors: Anchors

    def admit(self, cwd: Path | None = None) -> Admitted:
        """Admit ``cwd`` (default: the root)."""
        return self.anchors.admit((cwd or self.root).resolve())

    def listing(self, directory: Path) -> list[str]:
        """Return sorted entry names."""
        return sorted(p.name for p in directory.iterdir())


def _make_settings(root: Path, stage: Path, **kwargs: Any) -> Settings:  # ruff: ignore[any-type]
    return Settings(
        workspace_root=root,
        tmp_staging_dirs=((root, stage),),
        tmp_min_free_bytes=0,
        workspace_min_free_bytes=0,
        **kwargs,
    )


@pytest_asyncio.fixture
async def env(tmp_path: Path) -> AsyncIterator[Env]:
    """Workspace root, private staging directory and open anchors.

    Yields:
        The environment.
    """
    root = tmp_path / "ws"
    stage = tmp_path / "stage"
    root.mkdir()
    stage.mkdir()
    stage.chmod(0o700)
    settings = _make_settings(root, stage)
    anchors = Anchors(settings)
    anchors.open()
    yield Env(root.resolve(), stage.resolve(), settings, anchors)
    await anchors.aclose()


@pytest.fixture
def names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make random names predictable: 1 = staged, 2 = public, 3.. = recovery."""
    counter = iter(range(1, 10_000))

    def token_hex(_n: int) -> str:
        return f"{next(counter):032x}"

    monkeypatch.setattr(tmpguard, "secrets", SimpleNamespace(token_hex=token_hex))


@pytest.fixture(autouse=True)  # ruff: ignore[pytest-fixture-autouse]
def clean_renameat2() -> Any:  # ruff: ignore[any-type]
    """Reset the cached libc handle around tests that replace it.

    Yields:
        Nothing.
    """
    tmpguard._renameat2_fn.cache_clear()  # pyright: ignore[reportPrivateUsage, reportFunctionMemberAccess]
    yield
    tmpguard._renameat2_fn.cache_clear()  # pyright: ignore[reportPrivateUsage, reportFunctionMemberAccess]


def fds() -> int:
    """Count open descriptors."""
    return len(os.listdir("/proc/self/fd"))  # ruff: ignore[os-listdir]


async def run_turn(
    env: Env, cwd: Path, body: Callable[[Path], None] | None = None
) -> Path:
    """Run one turn against a fresh admission, closing it afterwards.

    Returns:
        The directory handed to the child.
    """
    adm = env.admit(cwd)
    try:
        async with turn_tmpdir(env.settings, env.anchors, adm) as directory:
            if body is not None:
                body(directory)
    finally:
        adm.close()
    return directory


def wrap(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, hook: Any) -> None:  # ruff: ignore[any-type]
    """Replace ``module.name`` by ``hook(real, *args, **kwargs)``."""
    real = getattr(module, name)

    def replacement(*args: object, **kwargs: object) -> object:
        return hook(real, *args, **kwargs)

    monkeypatch.setattr(module, name, replacement)


def staged_chmod_hook(action: Callable[[], None]) -> Callable[..., None]:
    """Build an ``os.chmod`` wrapper running ``action`` before the private chmod."""

    def hook(real: Any, path: Any, *args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        if kwargs.get("dir_fd") is not None and str(path).startswith(stg(0)[:-32]):
            action()
        real(path, *args, **kwargs)

    return hook


# --- success paths -----------------------------------------------------------


@pytest.mark.asyncio
async def test_lifecycle_and_nested_cwd(env: Env, tmp_path: Path) -> None:
    """The directory is private, descendants (incl. symlinks) go, nothing leaks."""
    cwd = env.root / "a" / "b"
    cwd.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("k")
    base = fds()
    seen: list[Path] = []

    def body(directory: Path) -> None:
        seen.append(directory)
        assert directory.parent == cwd
        assert directory.name.startswith(".agsrv-tmp-")
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        (directory / "sub" / "deep").mkdir(parents=True)
        (directory / "sub" / "deep" / "f").write_text("x")
        (directory / "link").symlink_to(outside)
        locked = directory / "ro"
        locked.mkdir()
        (locked / "f").write_text("x")
        locked.chmod(0o500)

    await run_turn(env, cwd, body)
    assert not seen[0].exists()
    assert env.listing(cwd) == []
    assert env.listing(env.stage) == []
    assert (outside / "keep").read_text() == "k"
    assert fds() == base
    assert env.anchors._live == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_error_in_turn_still_cleans(env: Env) -> None:
    """A failing turn body keeps its own error and leaves nothing behind."""
    msg = "turn"
    with pytest.raises(ValueError, match="turn"):
        await run_turn(env, env.root, lambda _d: (_ for _ in ()).throw(ValueError(msg)))
    assert env.listing(env.root) == []
    assert env.listing(env.stage) == []


@pytest.mark.asyncio
async def test_restrictive_umask(env: Env) -> None:
    """Mode 000 from the umask is repaired while the directory is private."""
    old = os.umask(0o777)
    try:
        modes: list[int] = []
        await run_turn(
            env, env.root, lambda d: modes.append(stat.S_IMODE(d.stat().st_mode))
        )
    finally:
        os.umask(old)
    assert modes == [0o700]
    assert env.listing(env.stage) == []


@pytest.mark.asyncio
async def test_child_removing_its_own_directory(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """If the public name is gone, cleanup logs and stops."""
    await run_turn(env, env.root, shutil.rmtree)
    assert "moved away" in caplog.text
    assert env.listing(env.stage) == []


# --- setup substitution --------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
@pytest.mark.parametrize("kind", ["empty-dir", "symlink"])
async def test_setup_substitution_is_not_replaced(
    env: Env, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """A writer taking the public name between mkdir and chmod is never replaced."""
    outside = tmp_path / "outside"
    outside.mkdir(mode=0o750)
    outside.chmod(0o750)
    (outside / "marker").write_text("m")
    public = env.root / pub(2)
    info: list[os.stat_result] = []

    def place() -> None:
        if kind == "empty-dir":
            public.mkdir(mode=0o755)
            public.chmod(0o755)
        else:
            public.symlink_to(outside)
        info.append(public.lstat())

    wrap(monkeypatch, os, "chmod", staged_chmod_hook(place))
    base = fds()
    with pytest.raises(HTTPException) as error:
        await run_turn(env, env.root)
    assert error.value.status_code == 503
    after = public.lstat()
    assert (after.st_ino, after.st_mode) == (info[0].st_ino, info[0].st_mode)
    assert env.listing(env.stage) == []
    assert (outside / "marker").read_text() == "m"
    assert stat.S_IMODE(outside.stat().st_mode) == 0o750
    assert fds() == base


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_umask_with_foreign_name_still_fails_closed(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restrictive umask plus a squatter on the public name: no overwrite."""
    public = env.root / pub(2)

    def place() -> None:
        public.mkdir()

    wrap(monkeypatch, os, "chmod", staged_chmod_hook(place))
    old = os.umask(0o777)
    try:
        with pytest.raises(HTTPException):
            await run_turn(env, env.root)
    finally:
        os.umask(old)
    assert public.is_dir()
    assert env.listing(env.stage) == []


# --- substitution while the turn runs --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["populated", "empty", "symlink"])
async def test_real_directory_substitution_survives(
    env: Env, tmp_path: Path, caplog: pytest.LogCaptureFixture, kind: str
) -> None:
    """A swapped-in entry is restored untouched; the moved genuine tree is kept."""
    external = tmp_path / "ext"
    external.mkdir()
    (external / "m.txt").write_text("m")
    aside = env.root / "aside"
    seen: list[os.stat_result] = []

    def body(directory: Path) -> None:
        directory.rename(aside)
        (aside / "genuine.txt").write_text("g")
        if kind == "symlink":
            directory.symlink_to(external)
        else:
            directory.mkdir()
            if kind == "populated":
                (directory / "foreign.txt").write_text("f")
        seen.append(directory.lstat())

    directory = await run_turn(env, env.root, body)
    assert (aside / "genuine.txt").read_text() == "g"
    after = directory.lstat()
    assert (after.st_ino, after.st_mode) == (seen[0].st_ino, seen[0].st_mode)
    if kind == "populated":
        assert (directory / "foreign.txt").read_text() == "f"
    if kind == "symlink":
        assert directory.is_symlink()
        assert (external / "m.txt").read_text() == "m"
    assert env.listing(env.stage) == []
    assert "foreign entry" in caplog.text


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_empty_dir_installed_before_removal_survives(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory placed at the freed public name after verification survives."""
    public = env.root / pub(2)
    inodes: list[int] = []

    def hook(real: Callable[..., None], *args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        public.mkdir()
        inodes.append(public.stat().st_ino)
        real(*args, **kwargs)

    wrap(monkeypatch, tmpguard, "_make_removable", hook)
    await run_turn(env, env.root)
    assert public.stat().st_ino == inodes[0]
    assert env.listing(env.stage) == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_name_retaken_before_restoration(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the public name is retaken, the foreign tree stays quarantined and logged."""
    public = env.root / pub(2)
    adm = env.admit()
    taken: list[int] = []

    def hook(
        real: Callable[..., None], src_fd: int, src: str, dst_fd: int, dst: str
    ) -> None:
        if src_fd == adm.stage_fd and dst == pub(2):
            public.mkdir()
            taken.append(public.stat().st_ino)
        real(src_fd, src, dst_fd, dst)

    async with turn_tmpdir(env.settings, env.anchors, adm) as directory:
        directory.rename(env.root / "aside")
        directory.mkdir()
        (directory / "foreign.txt").write_text("f")
        wrap(monkeypatch, tmpguard, "_rename_noreplace", hook)
    adm.close()
    assert public.stat().st_ino == taken[0]
    assert (env.stage / rec(3) / "foreign.txt").read_text() == "f"
    assert (env.root / "aside").is_dir()
    assert rec(3) in caplog.text


# --- ancestors ---------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["outside", "in-root", "real-replacement"])
async def test_ancestor_changed_during_wait(
    env: Env, tmp_path: Path, target: str
) -> None:
    """A swapped ancestor fails the turn before publication or launch."""
    cwd = env.root / "a" / "b"
    cwd.mkdir(parents=True)
    other = {"outside": tmp_path / "outside", "in-root": env.root / "other"}.get(
        target, env.root / "a"
    )
    adm = env.admit(cwd)
    (env.root / "a").rename(env.root / "a-aside")
    if target == "real-replacement":
        (other / "b").mkdir(parents=True)
    else:
        (other / "b").mkdir(parents=True)
        (env.root / "a").symlink_to(other)
    launched = False
    with pytest.raises(HTTPException) as error:
        async with turn_tmpdir(env.settings, env.anchors, adm):
            launched = True  # pragma: no cover
    adm.close()
    assert error.value.status_code == 503
    assert not launched
    assert not list(tmp_path.rglob(".agsrv-tmp-*"))  # ruff: ignore[blocking-path-method-in-async-function]
    assert env.listing(env.stage) == []


def test_ancestor_swapped_while_walking_uses_held_descriptor(
    env: Env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """After opening ``a`` the walk continues from the held fd, not the new name."""
    original = env.root / "a" / "b"
    original.mkdir(parents=True)
    decoy = tmp_path / "decoy"
    (decoy / "b").mkdir(parents=True)
    expected = original.stat().st_ino
    swapped: list[bool] = []

    def hook(
        real: Callable[..., int],
        path: str,
        flags: int,
        *args: object,
        **kwargs: Any,  # ruff: ignore[any-type]
    ) -> int:
        fd = real(path, flags, *args, **kwargs)
        if path == "a" and kwargs.get("dir_fd") is not None and not swapped:
            swapped.append(True)
            (env.root / "a").rename(env.root / "a-aside")
            (env.root / "a").symlink_to(decoy)
        return fd

    wrap(monkeypatch, os, "open", hook)
    adm = env.admit(original)
    monkeypatch.undo()
    assert os.fstat(adm.fd).st_ino == expected
    adm.close()
    assert swapped


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_ancestor_swapped_after_publication(env: Env, tmp_path: Path) -> None:
    """Cleanup reaches only the admitted directory, never a same-named decoy."""
    cwd = env.root / "a" / "b"
    cwd.mkdir(parents=True)
    decoy = tmp_path / "decoy"
    marker = decoy / "b" / pub(2) / "marker"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep")

    def body(_directory: Path) -> None:
        (env.root / "a").rename(env.root / "a-aside")
        (env.root / "a").symlink_to(decoy)

    await run_turn(env, cwd, body)
    assert marker.read_text() == "keep"
    assert env.listing(env.root / "a-aside" / "b") == []
    assert env.listing(env.stage) == []


@pytest.mark.parametrize("parts", [("..",), (".",), ("",), ("a/b",)])
def test_walk_rejects_unsafe_components(env: Env, parts: tuple[str, ...]) -> None:
    """Absolute, ``.`` and ``..`` components are rejected before any open."""
    with pytest.raises(TmpdirError, match="unsafe"):
        tmpguard._walk(env.anchors.root_fd, parts)  # pyright: ignore[reportPrivateUsage]


def test_admit_outside_root_missing_stage_and_missing_root(
    env: Env, tmp_path: Path
) -> None:
    """Admission fails with 503 for each unusable configuration."""
    with pytest.raises(HTTPException) as error:
        env.anchors.admit(tmp_path)  # not below the root
    assert error.value.status_code == 503
    bare = Anchors(Settings(workspace_root=env.root, tmp_staging_dirs=()))
    with pytest.raises(HTTPException, match="no staging"):
        bare.admit(env.root)
    gone = tmp_path / "gone"
    lost = Anchors(_make_settings(gone, env.stage))
    with pytest.raises(HTTPException, match="trusted location"):
        lost.admit(gone)
    assert not lost.is_open


def test_most_specific_staging_wins(tmp_path: Path) -> None:
    """The deepest applicable mount root selects the staging directory."""
    root = tmp_path / "ws"
    (root / "mnt").mkdir(parents=True)
    stages: list[Path] = []
    for name in ("outer", "inner"):
        stage = tmp_path / name
        stage.mkdir()
        stage.chmod(0o700)
        stages.append(stage)
    settings = Settings(
        workspace_root=root,
        tmp_staging_dirs=((root, stages[0]), (root / "mnt", stages[1])),
    )
    anchors = Anchors(settings)
    anchors.open()
    anchors.open()  # idempotent
    inner = anchors.admit(root / "mnt")
    outer = anchors.admit(root)
    assert os.path.samestat(os.fstat(inner.stage_fd), stages[1].stat())
    assert os.path.samestat(os.fstat(outer.stage_fd), stages[0].stat())
    inner.close()
    outer.close()
    inner.close()  # idempotent
    asyncio.run(anchors.aclose())
    asyncio.run(anchors.aclose())  # already closed


# --- trusted anchors -----------------------------------------------------------


def test_staging_validation(tmp_path: Path, env: Env) -> None:
    """Unsafe staging directories are refused at startup, closing what was opened."""
    root = env.root
    loose = tmp_path / "loose"
    loose.mkdir()
    loose.chmod(0o755)
    inside = root / "stage-inside"
    inside.mkdir()
    inside.chmod(0o700)
    for bad, text in [
        (loose, "mode 0700"),
        (inside, "outside the workspace"),
        (tmp_path / "missing", "cannot open staging"),
    ]:
        anchors = Anchors(
            Settings(
                workspace_root=root, tmp_staging_dirs=((root, env.stage), (root, bad))
            )
        )
        base = fds()
        with pytest.raises(TmpdirError, match=text):
            anchors.open()
        assert fds() == base
        assert not anchors.is_open


def test_check_stage_ownership_and_filesystem(env: Env) -> None:
    """Foreign ownership and another filesystem are refused."""
    real = env.stage.stat()
    check = tmpguard._check_stage  # pyright: ignore[reportPrivateUsage]
    check(env.stage, env.root, real)
    values = list(real)
    values[4] += 1  # st_uid
    with pytest.raises(TmpdirError, match="owned by the service"):
        check(env.stage, env.root, os.stat_result(values))
    values = list(real)
    values[2] += 1  # st_dev
    with pytest.raises(TmpdirError, match="not on the filesystem"):
        check(env.stage, env.root, os.stat_result(values))


def test_unsafe_rmtree_and_missing_renameat2(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Platforms without safe rmtree or renameat2 fail closed."""
    fresh = Anchors(env.settings)
    monkeypatch.setattr(shutil.rmtree, "avoids_symlink_attacks", False)
    with pytest.raises(TmpdirError, match="symlink-attack"):
        fresh.open()
    monkeypatch.undo()

    class NoSymbol:
        def __init__(self, *_a: object, **_k: object) -> None: ...

        def __getattr__(self, name: str) -> object:
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "CDLL", NoSymbol)
    tmpguard._renameat2_fn.cache_clear()  # pyright: ignore[reportPrivateUsage, reportFunctionMemberAccess]
    with pytest.raises(TmpdirError, match="renameat2"):
        fresh.open()
    assert not fresh.is_open


def test_rename_noreplace_errors(env: Env) -> None:
    """The wrapper reports EEXIST without replacing, and normal OSError subclasses."""
    (env.root / "src").mkdir()
    (env.root / "dst").mkdir()
    fd = env.anchors.root_fd
    with pytest.raises(FileExistsError):
        tmpguard._rename_noreplace(fd, "src", fd, "dst")  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(FileNotFoundError):
        tmpguard._rename_noreplace(fd, "nope", fd, "other")  # pyright: ignore[reportPrivateUsage]
    assert (env.root / "src").is_dir()


# --- failure injection ---------------------------------------------------------


async def expect_setup_failure(env: Env, *, cwd: Path | None = None) -> None:
    """Run a turn that must fail setup and verify nothing was left anywhere."""
    base = fds()
    launched: list[bool] = []
    with pytest.raises(HTTPException) as error:
        await run_turn(env, cwd or env.root, lambda _d: launched.append(True))
    assert error.value.status_code == 503
    assert not launched
    assert env.listing(env.stage) == []
    assert [n for n in env.listing(cwd or env.root) if n.startswith(".agsrv")] == []
    assert fds() == base
    assert env.anchors._live == 0  # pyright: ignore[reportPrivateUsage]


def raise_for_staged(error: BaseException) -> Callable[..., Any]:
    """Build an ``os.chmod``/``os.open`` hook failing only on staged names."""

    def hook(real: Callable[..., Any], path: Any, *args: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
        if kwargs.get("dir_fd") is not None and str(path).startswith(".agsrv-stg-"):
            raise error
        return real(path, *args, **kwargs)

    return hook


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["chmod", "open"])
async def test_failure_after_private_mkdir_rolls_back(
    env: Env, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    """chmod/open failure removes the private directory by its staged name."""
    wrap(monkeypatch, os, target, raise_for_staged(OSError(errno.EMFILE, "boom")))
    await expect_setup_failure(env)


@pytest.mark.asyncio
async def test_non_oserror_during_setup_propagates_after_rollback(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any other exception still triggers rollback and keeps its type."""
    wrap(monkeypatch, os, "chmod", raise_for_staged(RuntimeError("bug")))
    with pytest.raises(RuntimeError, match="bug"):
        await run_turn(env, env.root)
    assert env.listing(env.stage) == []
    assert env.anchors._live == 0  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_wrong_mode_after_chmod_rolls_back(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact-mode verification fails closed."""

    def hook(real: Callable[..., None], path: Any, mode: int, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        real(path, 0o755 if kwargs.get("dir_fd") is not None else mode, **kwargs)

    wrap(monkeypatch, os, "chmod", hook)
    await expect_setup_failure(env)


@pytest.mark.asyncio
async def test_rollback_failure_is_logged(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If even the rollback rmdir fails the staged name is reported."""
    wrap(monkeypatch, os, "chmod", raise_for_staged(OSError(errno.EIO, "io")))
    wrap(monkeypatch, os, "rmdir", raise_for_staged(OSError(errno.EBUSY, "busy")))
    with pytest.raises(HTTPException):
        await run_turn(env, env.root)
    monkeypatch.undo()
    assert "Failed to roll back private staged directory" in caplog.text
    assert env.listing(env.stage)  # retained, not deleted blindly


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        OSError(errno.EXDEV, "xdev"),
        OSError(errno.ENOSYS, "nosys"),
        OSError(errno.EINVAL, "inval"),
    ],
)
async def test_publication_failure_removes_private_directory(
    env: Env, monkeypatch: pytest.MonkeyPatch, error: OSError
) -> None:
    """EXDEV/unsupported publication fails closed and cleans the staged entry."""

    def fail(*_a: object) -> None:
        raise error

    monkeypatch.setattr(tmpguard, "_rename_noreplace", fail)
    await expect_setup_failure(env)


@pytest.mark.asyncio
async def test_publication_non_oserror_propagates(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-OS error at publication rolls back and propagates unchanged."""

    def fail(*_a: object) -> None:
        msg = "bug"
        raise RuntimeError(msg)

    monkeypatch.setattr(tmpguard, "_rename_noreplace", fail)
    with pytest.raises(RuntimeError, match="bug"):
        await run_turn(env, env.root)
    assert env.listing(env.stage) == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_staged_name_collision_is_skipped_untouched(env: Env) -> None:
    """An existing staged entry is never reused or modified."""
    (env.stage / stg(1)).mkdir()
    (env.stage / stg(1) / "mine").write_text("m")
    await run_turn(env, env.root)
    assert (env.stage / stg(1) / "mine").read_text() == "m"
    assert env.listing(env.stage) == [stg(1)]


@pytest.mark.asyncio
async def test_staged_names_exhausted(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If every candidate name exists the turn fails and entries are untouched."""

    def token_hex(_n: int) -> str:
        return f"{1:032x}"

    monkeypatch.setattr(tmpguard, "secrets", SimpleNamespace(token_hex=token_hex))
    (env.stage / stg(1)).mkdir()
    with pytest.raises(HTTPException, match="unused staging name"):
        await run_turn(env, env.root)
    assert env.listing(env.stage) == [stg(1)]


@pytest.mark.asyncio
async def test_mkdir_error_fails_the_turn(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real mkdir error (disk full) fails setup with nothing left behind."""

    def full(
        real: Callable[..., None],
        path: str,
        *args: object,
        **kwargs: Any,  # ruff: ignore[any-type]
    ) -> None:
        if kwargs.get("dir_fd") is not None and path.startswith(".agsrv-stg-"):
            raise OSError(errno.ENOSPC, "full")
        real(path, *args, **kwargs)

    wrap(monkeypatch, os, "mkdir", full)
    await expect_setup_failure(env)


@pytest.mark.asyncio
async def test_staging_on_another_filesystem_is_refused(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A device mismatch between cwd and staging fails before mkdir."""
    adm = env.admit()
    real = os.fstat

    def fake(fd: int) -> os.stat_result:
        info = real(fd)
        if fd != adm.stage_fd:
            return info
        values = list(info)
        values[2] += 1
        return os.stat_result(values)

    monkeypatch.setattr(os, "fstat", fake)
    with pytest.raises(HTTPException, match="different filesystem"):
        async with turn_tmpdir(env.settings, env.anchors, adm):
            pytest.fail("must not launch")  # pragma: no cover
    monkeypatch.undo()
    adm.close()
    assert env.listing(env.stage) == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_capture_collision_retries_with_fresh_name(env: Env) -> None:
    """A pre-existing recovery name is left alone and another is used."""
    (env.stage / rec(3)).mkdir()
    (env.stage / rec(3) / "mine").write_text("m")
    await run_turn(env, env.root)
    assert (env.stage / rec(3) / "mine").read_text() == "m"
    assert env.listing(env.stage) == [rec(3)]


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_capture_names_exhausted_retains_public(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """If no recovery name is free, the entry is retained, not deleted."""
    for n in range(3, 11):
        (env.stage / rec(n)).mkdir()
    directory = await run_turn(env, env.root)
    assert directory.is_dir()
    assert "Could not pick a recovery name" in caplog.text


@pytest.mark.asyncio
async def test_capture_failure_retains_entry(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed reclaim rename leaves the public entry and reports it."""
    adm = env.admit()

    def fail(real: Callable[..., None], src_fd: int, *rest: Any) -> None:  # ruff: ignore[any-type]
        if src_fd == adm.fd:
            raise OSError(errno.EACCES, "denied")
        real(src_fd, *rest)

    async with turn_tmpdir(env.settings, env.anchors, adm) as directory:
        wrap(monkeypatch, tmpguard, "_rename_noreplace", fail)
    monkeypatch.undo()
    adm.close()
    assert directory.is_dir()
    assert "Could not reclaim turn directory" in caplog.text
    shutil.rmtree(directory)


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_inspection_failure_retains_unverified_entry(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the captured entry cannot be inspected it is retained in staging."""
    adm = env.admit()

    def fail(real: Callable[..., Any], path: Any, **kwargs: Any) -> Any:  # ruff: ignore[any-type]
        if kwargs.get("dir_fd") == adm.stage_fd and str(path) == rec(3):
            raise OSError(errno.EIO, "io")
        return real(path, **kwargs)

    async with turn_tmpdir(env.settings, env.anchors, adm):
        wrap(monkeypatch, os, "stat", fail)
    monkeypatch.undo()
    adm.close()
    assert env.listing(env.stage) == [rec(3)]
    assert "retaining the unverified entry" in caplog.text


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_recursive_removal_failure_is_retained_and_logged(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed rmtree keeps the entry, logs where it is, and closes descriptors."""
    base = fds()
    monkeypatch.setattr(
        tmpguard.shutil, "rmtree", MagicMock(side_effect=OSError(errno.EIO, "io"))
    )
    monkeypatch.setattr(
        tmpguard.shutil.rmtree, "avoids_symlink_attacks", True, raising=False
    )
    await run_turn(env, env.root)
    monkeypatch.undo()
    assert env.listing(env.stage) == [rec(3)]
    assert rec(3) in caplog.text
    assert fds() == base


@pytest.mark.asyncio
async def test_restoration_succeeds_when_name_is_free(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """Covered via the substitution tests; assert the log wording once more."""

    def body(directory: Path) -> None:
        directory.rename(env.root / "aside")
        directory.mkdir()

    await run_turn(env, env.root, body)
    assert "restored it" in caplog.text


# --- admission/resource accounting -----------------------------------------------


@pytest.mark.asyncio
async def test_aclose_drains_owners_and_times_out(
    env: Env, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Shared descriptors close only after every owner is released."""
    adm = env.admit()
    closer = asyncio.create_task(env.anchors.aclose())
    await asyncio.sleep(0.05)
    assert not closer.done()
    assert env.anchors.is_open
    adm.close()
    await asyncio.wait_for(closer, 1)
    assert not env.anchors.is_open
    env.anchors.open()
    held = env.admit()
    monkeypatch.setattr(tmpguard, "_DRAIN_TIMEOUT", 0.01)
    await env.anchors.aclose()
    assert env.anchors.is_open  # kept open: an owner never finished
    assert "did not finish" in caplog.text
    held.close()


# --- event loop and cancellation ------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.usefixtures("names")
async def test_cleanup_is_off_loop_and_survives_repeated_cancellation(  # ruff: ignore[too-many-statements]
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removal blocks a worker, not the loop; cancellation cannot interrupt it."""
    entered, release = threading.Event(), threading.Event()
    real = cast("Callable[..., None]", shutil.rmtree)
    threads: list[int] = []

    def blocking(*args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        real(*args, **kwargs)

    blocking.avoids_symlink_attacks = True  # pyright: ignore[reportFunctionMemberAccess]
    monkeypatch.setattr(tmpguard.shutil, "rmtree", blocking)
    jobs_lock = asyncio.Lock()  # stands in for the workspace lease
    first_adm = env.admit()
    second_started = asyncio.Event()
    work_done = asyncio.Event()

    async def first() -> None:
        async with jobs_lock, turn_tmpdir(env.settings, env.anchors, first_adm):
            await asyncio.Event().wait()

    async def second() -> None:
        async with jobs_lock:
            second_started.set()
            adm = env.admit()
            async with turn_tmpdir(env.settings, env.anchors, adm):
                pass
            adm.close()
            work_done.set()

    owner = asyncio.create_task(first())
    await asyncio.sleep(0.05)
    owner.cancel()
    await asyncio.to_thread(entered.wait, 5)
    other = asyncio.create_task(second())
    ticks = 0

    async def unrelated() -> None:
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1

    await unrelated()  # runs while removal is blocked
    assert ticks == 5
    owner.cancel()  # second cancellation
    await asyncio.sleep(0.02)
    assert not owner.done()
    assert not second_started.is_set()
    assert threads[0] != threading.get_ident()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await owner
    first_adm.close()
    await asyncio.wait_for(other, 2)
    assert work_done.is_set()
    assert env.listing(env.stage) == []
    assert env.listing(env.root) == []


@pytest.mark.asyncio
async def test_cancelled_cleanup_task_does_not_spin(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the cleanup task itself is cancelled (loop shutdown) the owner exits."""
    entered, release = threading.Event(), threading.Event()
    real = cast("Callable[..., None]", shutil.rmtree)

    def blocking(*args: Any, **kwargs: Any) -> None:  # ruff: ignore[any-type]
        entered.set()
        assert release.wait(5)
        real(*args, **kwargs)

    blocking.avoids_symlink_attacks = True  # pyright: ignore[reportFunctionMemberAccess]
    monkeypatch.setattr(tmpguard.shutil, "rmtree", blocking)
    adm = env.admit()

    async def owner() -> None:
        async with turn_tmpdir(env.settings, env.anchors, adm):
            pass

    task = asyncio.create_task(owner())
    await asyncio.to_thread(entered.wait, 5)
    (cleanup,) = env.anchors.cleanups
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    release.set()
    adm.close()


# --- free space ------------------------------------------------------------------


def test_check_space_semantics(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Equality passes, disabled checks are skipped, errors fail closed."""
    adm = env.admit()
    settings = _make_settings(env.root, env.stage)
    mock = MagicMock(side_effect=OSError("unavailable"))
    monkeypatch.setattr(os, "statvfs", mock)
    monkeypatch.setattr(os, "fstatvfs", mock)
    check_space(settings, env.anchors, adm)
    mock.assert_not_called()
    with pytest.raises(HTTPException, match="Cannot check free space on /tmp"):
        check_space(replace(settings, tmp_min_free_bytes=1), env.anchors, adm)
    monkeypatch.setattr(os, "statvfs", statvfs_stub(lambda: 100))
    equal = replace(settings, tmp_min_free_bytes=100)  # equality is accepted
    check_space(equal, env.anchors, adm)
    with pytest.raises(HTTPException, match="100 bytes free"):
        check_space(replace(settings, tmp_min_free_bytes=101), env.anchors, adm)
    with pytest.raises(HTTPException, match="Cannot check free space on"):
        check_space(replace(settings, workspace_min_free_bytes=10), env.anchors, adm)
    adm.close()


def test_check_space_uses_each_held_filesystem(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cwd's own filesystem (held fd) must pass; the root is not counted twice."""
    sub = env.root / "mounted"
    sub.mkdir()
    adm = env.admit(sub)
    settings = replace(
        _make_settings(env.root, env.stage), workspace_min_free_bytes=100
    )
    calls: list[int] = []

    def per_fd(fd: int) -> os.statvfs_result:
        calls.append(fd)
        return vfs(1 if fd == adm.fd else 1000)

    monkeypatch.setattr(os, "fstatvfs", per_fd)
    with pytest.raises(HTTPException, match=str(sub)):
        check_space(settings, env.anchors, adm)
    assert calls == [env.anchors.root_fd, adm.fd]
    adm.close()
    root_adm = env.admit()
    calls.clear()
    check_space(settings, env.anchors, root_adm)
    assert calls == [env.anchors.root_fd]
    root_adm.close()


# --- settings ----------------------------------------------------------------------


def test_staging_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Staging pairs parse from the environment and are validated."""
    monkeypatch.setenv(
        "AGSRV_TMP_STAGING_DIRS", f"{tmp_path}=/var/stage, ,{tmp_path}/m={tmp_path}/s"
    )
    pairs = Settings.from_env().tmp_staging_dirs
    assert pairs == (
        (tmp_path.resolve(), Path("/var/stage").resolve()),
        ((tmp_path / "m").resolve(), (tmp_path / "s").resolve()),
    )
    monkeypatch.setenv("AGSRV_TMP_STAGING_DIRS", "nonsense")
    with pytest.raises(ValueError, match="invalid AGSRV_TMP_STAGING_DIRS"):
        Settings.from_env()
    monkeypatch.delenv("AGSRV_TMP_STAGING_DIRS")
    assert Settings.from_env().tmp_staging_dirs == ()
    with pytest.raises(ValueError, match="must be absolute"):
        Settings(tmp_staging_dirs=((Path("rel"), Path("/x")),))


@pytest.mark.parametrize("name", ["tmp_min_free_bytes", "workspace_min_free_bytes"])
def test_negative_threshold(name: str) -> None:
    """Negative thresholds name the offending setting."""
    with pytest.raises(ValueError, match=f"AGSRV_{name.upper()}"):
        Settings(tmp_min_free_bytes=-1) if name == "tmp_min_free_bytes" else Settings(
            workspace_min_free_bytes=-1
        )


def test_env_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thresholds parse their explicit values including disabled checks."""
    monkeypatch.setenv("AGSRV_TMP_MIN_FREE_BYTES", "0")
    monkeypatch.setenv("AGSRV_WORKSPACE_MIN_FREE_BYTES", "123")
    settings = Settings.from_env()
    assert settings.tmp_min_free_bytes == 0
    assert settings.workspace_min_free_bytes == 123


def test_logging_is_quiet_when_everything_is_fine(
    env: Env, caplog: pytest.LogCaptureFixture
) -> None:
    """Normal turns log nothing at warning level or above."""
    caplog.set_level(logging.WARNING)
    asyncio.run(run_turn(env, env.root))
    assert not [r for r in caplog.records if r.name == tmpguard.logger.name]
