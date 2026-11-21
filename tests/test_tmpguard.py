"""Disk admission and per-turn temporary directory tests."""

from __future__ import annotations

import os
import shutil
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from agsrv import tmpguard
from agsrv.claude import build_options
from agsrv.config import Settings
from agsrv.main import create_app
from agsrv.sandbox import SandboxSetupError
from agsrv.schemas import QueryRequest
from agsrv.tmpguard import check_space, turn_tmpdir
from tests.test_main import FakeRunner


@pytest.mark.asyncio
async def test_directory_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Private directories are removed on success and error; cleanup errors log."""
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        assert directory.parent == tmp_path
        assert directory.name.startswith(".agsrv-tmp-")
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        (directory / "file").write_text("test")
        (directory / "sub" / "deep").mkdir(parents=True)
        (directory / "sub" / "deep" / "f").write_text("x")
        (directory / "link").symlink_to(tmp_path)
    assert not directory.exists()
    assert _found(tmp_path.parent, tmp_path.name)
    msg = "turn"
    with pytest.raises(ValueError, match="turn"):
        async with turn_tmpdir(tmp_path, tmp_path) as directory:
            raise ValueError(msg)
    assert not directory.exists()
    monkeypatch.setattr(tmpguard, "_purge", MagicMock(side_effect=OSError("cleanup")))
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        pass
    assert "Failed to remove" in caplog.text
    monkeypatch.undo()
    shutil.rmtree(_found(tmp_path, ".agsrv-del-*")[0])


@pytest.mark.asyncio
async def test_readonly_tree_and_nested_cwd(tmp_path: Path) -> None:
    """Read-only trees are removable, and a nested cwd is reached from the root."""
    cwd = tmp_path / "a" / "b"
    cwd.mkdir(parents=True)
    async with turn_tmpdir(tmp_path, cwd) as directory:
        assert directory.parent == cwd
        locked = directory / "ro"
        locked.mkdir()
        (locked / "f").write_text("x")
        locked.chmod(0o500)
    assert not directory.exists()
    assert list(cwd.iterdir()) == []


@pytest.mark.asyncio
async def test_removal_runs_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recursive deletion executes in a worker thread."""
    import threading  # ruff: ignore[import-outside-top-level]

    threads: list[int] = []
    real = tmpguard._purge  # pyright: ignore[reportPrivateUsage]

    def purge(fd: int) -> None:
        threads.append(threading.get_ident())
        real(fd)

    monkeypatch.setattr(tmpguard, "_purge", purge)
    async with turn_tmpdir(tmp_path, tmp_path):
        pass
    assert threads
    assert threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_missing_entry_is_quiet_and_moved_entry_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn that deletes its own directory is fine; a moved one is reported."""
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        shutil.rmtree(directory)
    assert "moved away" not in caplog.text
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        moved = tmp_path / "moved"
        directory.rename(moved)
    assert "moved away" in caplog.text
    assert moved.exists()


@pytest.mark.parametrize("sandbox", [False, True])
def test_options_tmpdir(
    tmp_path: Path, sandbox: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both Claude configurations carry the turn directory."""
    monkeypatch.setattr(
        "agsrv.claude.prepare_sandbox_cache", MagicMock(return_value=tmp_path)
    )
    monkeypatch.setattr("agsrv.claude.bundled_cli_path", lambda: tmp_path)
    request = QueryRequest(agent="claude", prompt="test", sandbox=sandbox)
    options = build_options(
        request,
        Settings(workspace_root=tmp_path),
        tmp_path,
        tmp_path,
        tmp_path / "private",
    )
    assert options.env["TMPDIR"] == str(tmp_path / "private")
    if sandbox:
        assert options.env["XDG_CACHE_HOME"] == str(tmp_path / ".cache/agsrv-sandbox")


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize("low", ["tmp", "workspace"])
def test_reject_low_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str, agent: str, low: str
) -> None:
    """Both APIs and agents reject either low filesystem before starting."""
    settings = Settings(
        api_key="secret",
        workspace_root=tmp_path,
        jobs_db_path=tmp_path / "jobs.db",
        tmp_min_free_bytes=100,
        workspace_min_free_bytes=200,
    )

    def usage(path: Path) -> shutil._ntuple_diskusage:  # pyright: ignore[reportPrivateUsage]
        free = 1 if (str(path) == "/tmp") == (low == "tmp") else 1000  # ruff: ignore[hardcoded-temp-file]
        return shutil._ntuple_diskusage(1000, 1000 - free, free)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(shutil, "disk_usage", usage)
    runner = FakeRunner()
    with TestClient(create_app(settings, runner)) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": "Bearer secret"},
            json={
                "agent": agent,
                "prompt": "test",
                **({"sandbox": False} if agent == "claude" else {}),
            },
        )
        assert response.status_code == 503
        expected = "/tmp" if low == "tmp" else str(tmp_path)  # ruff: ignore[hardcoded-temp-file]
        assert expected in response.json()["detail"]
        assert "1 bytes free" in response.json()["detail"]
        assert runner.stream_calls == 0
        assert not _found(tmp_path, ".agsrv-tmp-*")


def _identity(path: str | Path) -> str | Path:
    return path


def _fds() -> int:
    return len(os.listdir("/proc/self/fd"))  # ruff: ignore[os-listdir]


def _found(root: Path, pattern: str) -> list[Path]:
    return list(root.glob(pattern))


def test_check_disabled_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero disables checks, and an unavailable filesystem fails closed."""
    settings = Settings(
        workspace_root=tmp_path, tmp_min_free_bytes=0, workspace_min_free_bytes=0
    )
    mock = MagicMock(side_effect=OSError("unavailable"))
    monkeypatch.setattr(shutil, "disk_usage", mock)
    check_space(settings)
    mock.assert_not_called()
    with pytest.raises(HTTPException) as exc:
        check_space(replace(settings, tmp_min_free_bytes=1))
    assert exc.value.status_code == 503


def test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Thresholds parse their explicit values including disabled checks."""
    monkeypatch.setenv("AGSRV_TMP_MIN_FREE_BYTES", "0")
    monkeypatch.setenv("AGSRV_WORKSPACE_MIN_FREE_BYTES", "123")
    settings = Settings.from_env()
    assert settings.tmp_min_free_bytes == 0
    assert settings.workspace_min_free_bytes == 123


@pytest.mark.parametrize("endpoint", ["/v1/query", "/v1/jobs"])
@pytest.mark.parametrize("fail", [False, True])
def test_turn_directory_lifecycle(tmp_path: Path, endpoint: str, fail: bool) -> None:
    """Claude receives a private directory that is removed after either outcome."""
    from tests.test_jobs import _wait  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        api_key="secret", workspace_root=tmp_path, jobs_db_path=tmp_path / "jobs.db"
    )
    runner = FakeRunner(fail_after=0 if fail else None)
    with TestClient(create_app(settings, runner)) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": "Bearer secret"},
            json={"agent": "claude", "prompt": "test", "sandbox": False},
        )
        if endpoint == "/v1/jobs":
            outcome = _wait(client, response.json()["job_id"])
            assert outcome["status"] == ("failed" if fail else "succeeded")
        else:
            assert response.status_code == (502 if fail else 200)
        assert runner.options is not None
        directory = Path(runner.options.env["TMPDIR"])
        assert directory.parent == tmp_path
        assert not directory.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("original", ["moved", "deleted"])
async def test_substitution_attack_deletes_nothing_foreign(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, original: str
) -> None:
    """A substitute directory, and everything in it, survives cleanup."""
    victim = tmp_path / "victim"
    (victim / "nested").mkdir(parents=True)
    (victim / "nested" / "precious").write_text("safe")
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        if original == "moved":
            directory.rename(tmp_path / "moved")
        else:
            shutil.rmtree(directory)
        victim.rename(directory)
    assert (directory / "nested" / "precious").read_text() == "safe"
    assert "was replaced" in caplog.text
    assert not _found(tmp_path, ".agsrv-del-*")


@pytest.mark.asyncio
async def test_substitution_when_name_is_retaken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the original name is busy again, the foreign entry stays at its tombstone."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious").write_text("safe")
    real = tmpguard._restore_foreign  # pyright: ignore[reportPrivateUsage]

    def busy(parent_fd: int, tomb: str, name: str, path: Path) -> None:
        (tmp_path / name).mkdir()
        real(parent_fd, tomb, name, path)

    monkeypatch.setattr(tmpguard, "_restore_foreign", busy)
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        directory.rename(tmp_path / "moved")
        victim.rename(directory)
    (tomb,) = _found(tmp_path, ".agsrv-del-*")
    assert (tomb / "precious").read_text() == "safe"
    assert "remains under tombstone name" in caplog.text


@pytest.mark.asyncio
async def test_identity_mismatch_skips_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A tombstone with a different identity is never purged."""
    purge = MagicMock()
    monkeypatch.setattr(tmpguard, "_purge", purge)
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        (directory / "mine").write_text("x")
        original = directory.stat()
        foreign = tmp_path / "foreign"
        foreign.mkdir()
        assert foreign.stat().st_ino != original.st_ino
        directory.rename(tmp_path / "moved")
        foreign.rename(directory)
    purge.assert_not_called()
    assert directory.is_dir()
    assert "was replaced" in caplog.text


@pytest.mark.asyncio
async def test_symlink_replaced_by_non_directory(tmp_path: Path) -> None:
    """A non-directory substitute is also left alone."""
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        directory.rename(tmp_path / "moved")
        directory.write_text("file")
    assert directory.read_text() == "file"


@pytest.mark.asyncio
@pytest.mark.parametrize("stale", [False, True])
async def test_ancestor_symlink_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    """An ancestor swapped for a symlink fails the turn before creating anything.

    With ``stale`` the swap lands after canonicalisation (the race): only the
    ``O_NOFOLLOW`` descriptor walk can catch it.
    """
    outside = tmp_path / "outside"
    (outside / "b").mkdir(parents=True)
    root = tmp_path / "root"
    (root / "a" / "b").mkdir(parents=True)
    cwd = (root / "a" / "b").resolve()
    (root / "a").rename(tmp_path / "a-moved")
    (root / "a").symlink_to(outside)
    if stale:
        monkeypatch.setattr(os.path, "realpath", _identity)
    with pytest.raises(HTTPException) as error:
        async with turn_tmpdir(root, cwd):
            pytest.fail("must not start")
    assert error.value.status_code == 503
    assert ("safely" if stale else "outside the workspace root") in error.value.detail
    assert list(outside.rglob("*")) == [outside / "b"]
    assert not _found(tmp_path, "**/.agsrv-*")


@pytest.mark.asyncio
async def test_cwd_outside_root_is_refused(tmp_path: Path) -> None:
    """The cwd must be inside the root that anchors the walk."""
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(HTTPException, match="outside the workspace root"):
        async with turn_tmpdir(root, tmp_path):
            pytest.fail("must not start")


@pytest.mark.asyncio
async def test_missing_cwd_and_mkdir_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failures before or at mkdir create nothing and leak no descriptors."""
    with pytest.raises(HTTPException, match="cannot open working directory"):
        async with turn_tmpdir(tmp_path, tmp_path / "missing"):
            pytest.fail("must not start")
    before = _fds()
    monkeypatch.setattr(os, "mkdir", MagicMock(side_effect=OSError("full")))
    with pytest.raises(HTTPException, match="cannot create"):
        async with turn_tmpdir(tmp_path, tmp_path):
            pytest.fail("must not start")
    monkeypatch.undo()
    assert _fds() == before
    assert not _found(tmp_path, ".agsrv-*")


@pytest.mark.asyncio
async def test_rollback_after_identity_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after the directory is opened removes it via the anchored path."""
    before = _fds()
    monkeypatch.setattr(os, "fchmod", MagicMock(side_effect=OSError("mode")))
    with pytest.raises(HTTPException, match="cannot set mode"):
        async with turn_tmpdir(tmp_path, tmp_path):
            pytest.fail("must not start")
    monkeypatch.undo()
    assert _fds() == before
    assert not _found(tmp_path, ".agsrv-*")


@pytest.mark.asyncio
async def test_rollback_non_oserror_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-OS failures still roll back and propagate unchanged."""
    monkeypatch.setattr(os, "fchmod", MagicMock(side_effect=KeyError("boom")))
    with pytest.raises(KeyError):
        async with turn_tmpdir(tmp_path, tmp_path):
            pytest.fail("must not start")
    monkeypatch.undo()
    assert not _found(tmp_path, ".agsrv-*")


@pytest.mark.asyncio
async def test_rollback_before_identity_uses_rmdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If the open fails, only an empty directory can be removed by name."""
    real = os.open

    def failing(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if kwargs.get("dir_fd") is not None and str(path).startswith(".agsrv-tmp-"):
            msg = "open"
            raise OSError(msg)
        return real(path, flags, *args, **kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    monkeypatch.setattr(os, "open", failing)
    with pytest.raises(HTTPException, match="cannot open"):
        async with turn_tmpdir(tmp_path, tmp_path):
            pytest.fail("must not start")
    assert not _found(tmp_path, ".agsrv-*")
    monkeypatch.setattr(os, "rmdir", MagicMock(side_effect=OSError("busy")))
    with pytest.raises(HTTPException):
        async with turn_tmpdir(tmp_path, tmp_path):
            pytest.fail("must not start")
    assert "Failed to roll back" in caplog.text
    monkeypatch.undo()
    shutil.rmtree(_found(tmp_path, ".agsrv-tmp-*")[0])


@pytest.mark.asyncio
async def test_rollback_on_interrupt_before_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fstat failures close descriptors and roll back, propagating as-is."""
    before = _fds()
    monkeypatch.setattr(os, "fstat", MagicMock(side_effect=KeyError("boom")))
    with pytest.raises(KeyError):
        async with turn_tmpdir(tmp_path, tmp_path):
            pytest.fail("must not start")
    monkeypatch.undo()
    assert _fds() == before
    assert not _found(tmp_path, ".agsrv-*")


@pytest.mark.asyncio
async def test_swap_between_mkdir_and_open_is_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A populated directory swapped in right after mkdir is never adopted."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "precious").write_text("safe")
    real = os.mkdir

    def swapping(path: str, mode: int = 0o777, *, dir_fd: int | None = None) -> None:
        real(path, mode, dir_fd=dir_fd)
        (tmp_path / path).rename(tmp_path / "ours")
        victim.rename(tmp_path / path)

    monkeypatch.setattr(os, "mkdir", swapping)
    with pytest.raises(HTTPException, match="replaced during creation"):
        async with turn_tmpdir(tmp_path, tmp_path):
            pytest.fail("must not start")
    monkeypatch.undo()
    (kept,) = _found(tmp_path, ".agsrv-tmp-*")
    assert (kept / "precious").read_text() == "safe"


@pytest.mark.asyncio
async def test_restrictive_umask(tmp_path: Path) -> None:
    """Creation succeeds and is usable even with every mode bit masked."""
    previous = os.umask(0o777)
    try:
        async with turn_tmpdir(tmp_path, tmp_path) as directory:
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
            (directory / "file").write_text("usable")
    finally:
        os.umask(previous)
    assert not directory.exists()


@pytest.mark.parametrize("name", ["tmp_min_free_bytes", "workspace_min_free_bytes"])
def test_negative_threshold(name: str) -> None:
    """Negative thresholds name the offending setting."""
    with pytest.raises(ValueError, match=f"AGSRV_{name.upper()}"):
        Settings(tmp_min_free_bytes=-1) if name == "tmp_min_free_bytes" else Settings(
            workspace_min_free_bytes=-1
        )


def test_cwd_separate_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Admission checks the actual cwd even if its mount differs from the root."""
    cwd = tmp_path / "mounted"
    cwd.mkdir()
    settings = Settings(
        workspace_root=tmp_path, tmp_min_free_bytes=0, workspace_min_free_bytes=100
    )

    def usage(path: Path) -> shutil._ntuple_diskusage:  # pyright: ignore[reportPrivateUsage]
        free = 1 if path == cwd else 1000
        return shutil._ntuple_diskusage(1000, 1000 - free, free)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(shutil, "disk_usage", usage)
    with pytest.raises(HTTPException, match=str(cwd)):
        check_space(settings, cwd)


def test_job_replay_and_low_space_leave_no_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing keys replay under low space; new jobs leave no rows or capacity held."""
    import sqlite3  # ruff: ignore[import-outside-top-level]

    from tests.test_jobs import _wait  # pyright: ignore[reportPrivateUsage]  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        api_key="secret",
        workspace_root=tmp_path,
        jobs_db_path=tmp_path / "jobs.db",
        tmp_min_free_bytes=1,
    )
    app = create_app(settings, FakeRunner())
    headers = {"Authorization": "Bearer secret", "Idempotency-Key": "replay"}
    body = {"agent": "claude", "prompt": "test", "sandbox": False}
    with TestClient(app) as client:
        first = client.post("/v1/jobs", headers=headers, json=body)
        assert _wait(client, first.json()["job_id"])["status"] == "succeeded"
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            MagicMock(return_value=shutil.disk_usage(tmp_path)._replace(free=0)),
        )
        assert (
            client.post("/v1/jobs", headers=headers, json=body).json()["job_id"]
            == first.json()["job_id"]
        )
        assert (
            client.post(
                "/v1/jobs", headers=headers, json={**body, "prompt": "other"}
            ).status_code
            == 409
        )
        assert (
            client.post(
                "/v1/jobs", headers={"Authorization": "Bearer secret"}, json=body
            ).status_code
            == 503
        )
        with sqlite3.connect(settings.jobs_db_path) as db:
            assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
        assert app.state.semaphore._value == settings.max_concurrency
        assert app.state.codex_semaphore._value == settings.codex_max_concurrency
        assert not app.state.jobs.workspaces


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude", "codex"])
async def test_query_rechecks_after_workspace_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """Queued queries create no directory and reject newly exhausted space."""
    import asyncio  # ruff: ignore[import-outside-top-level]

    import httpx  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        api_key="secret",
        workspace_root=tmp_path,
        tmp_min_free_bytes=1,
        workspace_min_free_bytes=1,
        max_concurrency=1,
    )
    app = create_app(settings, FakeRunner())
    monkeypatch.setattr("agsrv.main._codex_available", AsyncMock(return_value=True))
    low = False

    def usage(_path: Path) -> shutil._ntuple_diskusage:  # pyright: ignore[reportPrivateUsage]
        return shutil._ntuple_diskusage(100, 100 if low else 0, 0 if low else 100)  # pyright: ignore[reportPrivateUsage]

    monkeypatch.setattr(shutil, "disk_usage", usage)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with app.state.jobs.workspace(tmp_path):
            task = asyncio.create_task(
                client.post(
                    "/v1/query",
                    headers={"Authorization": "Bearer secret"},
                    json={
                        "agent": agent,
                        "prompt": "test",
                        **({"sandbox": False} if agent == "claude" else {}),
                    },
                )
            )
            for _ in range(100):
                if (
                    app.state.semaphore.locked()
                    or app.state.codex_semaphore._value < settings.codex_max_concurrency
                ):
                    break
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.02)
            assert not task.done()
            assert not await asyncio.to_thread(lambda: _found(tmp_path, ".agsrv-tmp-*"))
            low = True
        assert (await asyncio.wait_for(task, 1)).status_code == 503
        assert not app.state.jobs.workspaces
        assert not await asyncio.to_thread(lambda: _found(tmp_path, ".agsrv-tmp-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("agent", ["claude", "codex"])
async def test_job_rechecks_after_slot_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agent: str
) -> None:
    """A job waiting for an agent slot rechecks free space before execution."""
    import asyncio  # ruff: ignore[import-outside-top-level]

    from agsrv.schemas import CodexQueryRequest  # ruff: ignore[import-outside-top-level]

    settings = Settings(
        workspace_root=tmp_path, tmp_min_free_bytes=1, workspace_min_free_bytes=0
    )
    app = create_app(settings, FakeRunner())
    semaphore = app.state.semaphore if agent == "claude" else app.state.codex_semaphore
    for _ in range(semaphore._value):
        await semaphore.acquire()
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        MagicMock(return_value=shutil.disk_usage(tmp_path)._replace(free=0)),
    )
    body = (
        QueryRequest(agent="claude", prompt="test", sandbox=False)
        if agent == "claude"
        else CodexQueryRequest(agent="codex", prompt="test")
    )
    task = asyncio.create_task(app.state.jobs._execute(body, tmp_path))
    await asyncio.sleep(0)
    assert not task.done()
    semaphore.release()
    with pytest.raises(HTTPException) as error:
        await task
    assert error.value.status_code == 503
    assert not await asyncio.to_thread(lambda: _found(tmp_path, ".agsrv-tmp-*"))


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [False, True])
async def test_codex_detached_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disconnect: bool
) -> None:
    """A cancelled Codex turn retains its directory until detached cleanup ends."""
    import asyncio  # ruff: ignore[import-outside-top-level]

    from starlette.requests import Request  # ruff: ignore[import-outside-top-level]

    from agsrv import main  # ruff: ignore[import-outside-top-level]
    from agsrv.schemas import CodexQueryRequest  # ruff: ignore[import-outside-top-level]

    app = create_app(Settings(workspace_root=tmp_path))
    started, closing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    gone = False
    directory: Path | None = None

    async def receive() -> dict[str, str]:
        if gone:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    async def run(**kwargs: object) -> dict[str, object]:
        nonlocal directory
        directory = Path(str(kwargs["tmpdir"]))
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            closing.set()
            await release.wait()
        return {}

    monkeypatch.setattr(main, "_codex_available", AsyncMock(return_value=True))
    monkeypatch.setattr(main.codex, "run_codex", run)
    monkeypatch.setattr(main, "DISCONNECT_POLL_INTERVAL", 0.01)
    request = Request({"type": "http", "app": app}, receive)
    body = CodexQueryRequest(
        agent="codex", prompt="test", timeout_seconds=0.05 if not disconnect else 10
    )
    task = asyncio.create_task(main._codex_query_with_disconnect(body, request))  # pyright: ignore[reportPrivateUsage]
    await asyncio.wait_for(started.wait(), 1)
    assert directory is not None
    assert await asyncio.to_thread(directory.exists)
    gone = disconnect
    if disconnect:
        assert (await asyncio.wait_for(task, 1)).status_code == 499
    else:
        with pytest.raises(HTTPException) as error:
            await asyncio.wait_for(task, 1)
        assert error.value.status_code == 504
    await asyncio.wait_for(closing.wait(), 1)
    assert await asyncio.to_thread(directory.exists)
    release.set()
    for _ in range(100):
        if not main._pump_tasks:
            break
        await asyncio.sleep(0.01)
    assert not main._pump_tasks
    assert not await asyncio.to_thread(directory.exists)
    assert not app.state.jobs.workspaces


@pytest.mark.asyncio
async def test_missing_root_leaks_nothing(tmp_path: Path) -> None:
    """A missing workspace root fails closed."""
    with pytest.raises(HTTPException, match="safely"):
        async with turn_tmpdir(tmp_path / "gone", tmp_path / "gone"):
            pytest.fail("must not start")


@pytest.mark.asyncio
async def test_failed_purge_closes_descriptors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure deep in a tree is logged and every descriptor is released."""
    before = _fds()
    monkeypatch.setattr(os, "unlink", MagicMock(side_effect=OSError("busy")))
    async with turn_tmpdir(tmp_path, tmp_path) as directory:
        (directory / "a" / "b").mkdir(parents=True)
        (directory / "a" / "b" / "f").write_text("x")
    monkeypatch.undo()
    assert "Failed to remove" in caplog.text
    assert _fds() == before
    shutil.rmtree(_found(tmp_path, ".agsrv-del-*")[0])


@pytest.mark.asyncio
async def test_tombstone_swapped_after_purge_is_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tombstone replaced after the purge is not removed."""
    real = tmpguard._purge  # pyright: ignore[reportPrivateUsage]
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    def purge(fd: int) -> None:
        real(fd)
        (tomb,) = _found(tmp_path, ".agsrv-del-*")
        tomb.rename(tmp_path / "ours")
        foreign.rename(tomb)

    monkeypatch.setattr(tmpguard, "_purge", purge)
    async with turn_tmpdir(tmp_path, tmp_path):
        pass
    assert len(_found(tmp_path, ".agsrv-del-*")) == 1


def _rate_client(tmp_path: Path, runner: FakeRunner) -> TestClient:
    settings = Settings(api_key="k", workspace_root=tmp_path)
    return TestClient(create_app(settings, runner), raise_server_exceptions=False)


def test_rate_limits_use_private_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The /usage child gets a private TMPDIR that is removed, and is gated."""
    from claude_agent_sdk import ResultMessage  # ruff: ignore[import-outside-top-level]

    message = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=0,
        session_id="s",
        result="Current session: 5% used",
        total_cost_usd=0.0,
    )
    runner = FakeRunner(messages=[message])
    client = _rate_client(tmp_path, runner)
    headers = {"Authorization": "Bearer k"}
    path = "/v1/agents/claude/rate-limits"
    assert client.get(path, headers=headers).status_code == 200
    assert runner.options is not None
    directory = Path(runner.options.env["TMPDIR"])
    assert directory.parent == tmp_path
    assert not directory.exists()
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        MagicMock(return_value=shutil.disk_usage(tmp_path)._replace(free=0)),
    )
    runner.stream_calls = 0
    assert client.get(path, headers=headers).status_code == 503
    assert runner.stream_calls == 0
    assert not _found(tmp_path, ".agsrv-*")


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (HTTPException(503, "tmp"), 503),
        (SandboxSetupError("sandbox"), 503),
        (RuntimeError("bug"), 500),
    ],
)
def test_rate_limits_setup_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, code: int
) -> None:
    """Errors before the agent starts keep their status and release the slot."""
    from agsrv import main  # ruff: ignore[import-outside-top-level]

    monkeypatch.setattr(main, "build_options", MagicMock(side_effect=error))
    runner = FakeRunner()
    client = _rate_client(tmp_path, runner)
    response = client.get(
        "/v1/agents/claude/rate-limits", headers={"Authorization": "Bearer k"}
    )
    assert response.status_code == code
    assert runner.stream_calls == 0
    assert not _found(tmp_path, ".agsrv-*")
