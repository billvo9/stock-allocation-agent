"""
Where the Stock Lab's Model Health inputs go (since 2026-10-21): the real
daily job writes its monitor files under ~/stock-agent-private/health_inputs,
outside every checkout, so the unattended evening can run from a runner clone
outside ~/Desktop (macOS privacy protection keeps launchd jobs out of it)
while Model Health runs from the owner's own checkout. The synthetic demo
replay keeps writing to its checkout's reports/health_inputs. Folders are
created owner-only (0700) whatever the umask, files 0600, and an existing
folder is never changed. Model Health switches to the shared folder only once
it holds a monitor file, so an empty folder never hides the last one. Every
test runs under a temporary HOME or the session's temporary folder
(tests/conftest.py): nothing real is read or written.
"""

from __future__ import annotations

import argparse
import io
import os
import stat
from pathlib import Path

import pandas as pd
import pytest
from test_stock_lab_ledger import daily

from stock_agent.health import stock_lab as health_stock_lab
from stock_agent.health.model import HealthState
from stock_agent.stock_lab import monitoring
from stock_agent.stock_lab.job import NOT_AUTHORIZED
from stock_agent.stock_lab.store import DB_NAME, StockLabStore

ARTIFACT = monitoring.ARTIFACT


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    """A temporary home folder (Path.home() reads HOME), without the session's override."""

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(monitoring.HOME_ENV)
    assert Path.home() == home
    return home


def test_no_test_can_write_into_the_owners_shared_folder():
    # tests/conftest.py points the default at a temporary folder for the whole session, so
    # a test that forgets --health-inputs never reaches ~/stock-agent-private/health_inputs.
    # (The fixture also fails the session if anything was written there.)
    default = monitoring.shared_inputs()
    assert default == Path(os.environ[monitoring.HOME_ENV])
    assert not default.is_relative_to(Path.home() / "stock-agent-private")


def test_the_variable_moves_only_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv(monitoring.HOME_ENV, "~/elsewhere")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert monitoring.shared_inputs() == tmp_path / "elsewhere"
    given = Path("/h")  # an explicit home folder is never moved
    assert monitoring.shared_inputs(given) == Path("/h/stock-agent-private/health_inputs")
    monkeypatch.setenv(monitoring.HOME_ENV, "  ")  # blank: the home folder's
    assert monitoring.shared_inputs() == tmp_path / "stock-agent-private" / "health_inputs"


def test_model_health_never_follows_the_writers_variable(tmp_path, monkeypatch):
    # The variable keeps tests out of the owner's folder; Model Health reads the shared
    # folder under the home folder (or --stock-lab-inputs), never the variable's.
    moved = tmp_path / "moved"
    monkeypatch.setenv(monitoring.HOME_ENV, str(moved))
    monitoring.write_monitor(moved, {"run_id": "job-1", "artifact": ARTIFACT})
    home, checkout = tmp_path / "home", tmp_path / "checkout"
    assert health_stock_lab.inputs_root(home, checkout) == checkout


@pytest.fixture
def loose_umask():
    """A shell's usual umask (022), under which a plain mkdir makes a folder others can read."""

    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


def test_every_folder_the_writer_creates_is_the_owners_only_whatever_the_umask(
    tmp_path, loose_umask
):
    root = tmp_path / "home" / "stock-agent-private" / "health_inputs"  # none of it exists
    path = monitoring.write_monitor(root, {"run_id": "job-1", "artifact": ARTIFACT})
    assert path == root / ARTIFACT / "job-1.json"
    for folder in (root.parent.parent, root.parent, root, root / ARTIFACT):
        assert _mode(folder) == 0o700, folder
    assert _mode(path) == 0o600
    with pytest.raises(FileExistsError):  # write-once, as before
        monitoring.write_monitor(root, {"run_id": "job-1", "artifact": ARTIFACT})


def test_the_folders_are_owner_writable_even_under_a_umask_that_removes_it(tmp_path):
    # mkdir's mode is masked by the umask: without the chmod, a 0277 umask makes each new
    # folder 0500, and the next one inside it cannot be created.
    previous = os.umask(0o277)
    try:
        root = tmp_path / "a" / "b"
        path = monitoring.write_monitor(root, {"run_id": "job-1", "artifact": ARTIFACT})
    finally:
        os.umask(previous)
    for folder in (root.parent, root, root / ARTIFACT):
        assert _mode(folder) == 0o700, folder
    assert _mode(path) & 0o077 == 0  # the file stays the owner's only


def test_an_existing_folder_is_never_changed(tmp_path, loose_umask):
    root = tmp_path / "health_inputs"
    root.mkdir()
    root.chmod(0o750)  # the owner's own choice: left as it is
    monitoring.write_monitor(root, {"run_id": "job-1", "artifact": ARTIFACT})
    assert _mode(root) == 0o750
    assert _mode(root / ARTIFACT) == 0o700 and _mode(root / ARTIFACT / "job-1.json") == 0o600


def test_the_shared_folder_is_in_the_home_folder_outside_every_checkout(home):
    shared = monitoring.shared_inputs()
    assert shared == home / "stock-agent-private" / "health_inputs"
    assert monitoring.shared_inputs(Path("/h")) == Path("/h/stock-agent-private/health_inputs")
    assert not shared.is_relative_to(daily.PROJECT)


def test_the_real_job_writes_its_monitor_file_to_the_shared_folder_by_default(
    home, tmp_path, loose_umask
):
    store = StockLabStore(tmp_path / "research" / DB_NAME)
    out = io.StringIO()
    code = daily.main(
        [],  # no --health-inputs: the default
        store_factory=lambda: store,
        clock=lambda: pd.Timestamp("2026-10-17T01:00:00Z"),
        out=out,
    )
    assert code == 0
    folder = home / "stock-agent-private" / "health_inputs" / ARTIFACT
    (written,) = folder.glob("*.json")
    assert f"health input: {written}" in out.getvalue()
    for created in (folder.parent.parent, folder.parent, folder):
        assert _mode(created) == 0o700, created
    assert _mode(written) == 0o600
    # Model Health reads it from there, whichever checkout it runs from.
    assert health_stock_lab.inputs_root(home, tmp_path / "checkout") == folder.parent
    checks = {c.check_id: c for c in health_stock_lab.stock_lab_checks(folder)}
    job = checks["forecast.stock_lab_job"]
    assert (job.state, job.reason) == (HealthState.INSUFFICIENT_EVIDENCE, NOT_AUTHORIZED)


@pytest.mark.parametrize(
    "demo, given, expected",
    [
        (None, None, "shared"),
        (5, None, "demo"),  # synthetic files never reach the real job's folder
        (None, "given", "given"),
        (5, "given", "given"),
    ],
)
def test_where_each_mode_writes_its_monitor_file(home, tmp_path, demo, given, expected):
    places = {
        "shared": home / "stock-agent-private" / "health_inputs",
        "demo": daily.PROJECT / "reports" / "health_inputs",
        "given": tmp_path / "given",
    }
    args = argparse.Namespace(
        demo_replay=demo, health_inputs=None if given is None else places["given"]
    )
    assert daily.health_inputs(args) == places[expected]


def test_model_health_reads_the_shared_folder_once_it_holds_monitor_files(tmp_path):
    home, checkout = tmp_path / "home", tmp_path / "checkout"
    shared = home / "stock-agent-private" / "health_inputs"
    assert health_stock_lab.inputs_root(home, checkout) == checkout  # nothing shared yet
    assert health_stock_lab.inputs_root(home, None) is None
    shared.mkdir(parents=True)  # the root alone, from a mkdir: still the checkout's files
    assert health_stock_lab.inputs_root(home, checkout) == checkout
    (shared / ARTIFACT).mkdir()  # an empty monitor folder: still the checkout's files
    assert health_stock_lab.inputs_root(home, checkout) == checkout
    (shared / ARTIFACT / "notes.txt").write_text("not a monitor file\n", encoding="utf-8")
    assert health_stock_lab.inputs_root(home, checkout) == checkout
    (shared / ARTIFACT / "job-1.json").write_text("{}", encoding="utf-8")
    assert health_stock_lab.inputs_root(home, checkout) == shared
    assert health_stock_lab.inputs_root(home, None) == shared


def test_an_empty_shared_folder_never_hides_an_overdue_job(tmp_path):
    # The owner created the shared folder by hand, and no evening has written to it since:
    # the checkout's last file is still judged, so the stopped job still shows as overdue.
    home, checkout = tmp_path / "home", tmp_path / "checkout"
    (home / "stock-agent-private" / "health_inputs" / ARTIFACT).mkdir(parents=True)
    store = StockLabStore(tmp_path / "research" / DB_NAME)
    daily.main(
        ["--health-inputs", str(checkout)],
        store_factory=lambda: store,
        clock=lambda: pd.Timestamp("2026-10-17T01:00:00Z"),
        out=io.StringIO(),
    )
    root = health_stock_lab.inputs_root(home, checkout)
    assert root == checkout
    later = pd.Timestamp("2026-10-30T12:00:00Z").to_pydatetime()  # days past the next run
    checks = {c.check_id: c for c in health_stock_lab.stock_lab_checks(root / ARTIFACT, now=later)}
    job = checks["forecast.stock_lab_job"]
    assert (job.state, job.reason) == (HealthState.FAILED, "job_overdue")
