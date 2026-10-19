"""
Delivering the briefing: the private preview directory, duplicate-send
protection per market date with a fake adapter, failures recorded without
their message, no credential read or written anywhere, and the owner-run
script (closed days skipped, flags exact, the synthetic demo labelled).
Synthetic fixtures; nothing leaves the machine.
"""

from __future__ import annotations

import ast
import importlib.util
import io
import os
import sqlite3
import stat
import threading
from pathlib import Path

import pytest
from stock_lab_briefing_helpers import Lab

from stock_agent.stock_lab.briefing import delivery
from stock_agent.stock_lab.briefing.delivery import (
    DUPLICATE,
    FAILED,
    IN_PROGRESS,
    SENT,
    DeliveryLog,
    DeliveryRefused,
    PreviewAdapter,
    RenderedBriefing,
    deliver_once,
    private_directory,
)
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords
from stock_agent.stock_lab.store import DB_NAME

ROOT = Path(__file__).resolve().parents[1]
BRIEFING = ROOT / "src" / "stock_agent" / "stock_lab" / "briefing"
SCRIPT = ROOT / "scripts" / "stock_lab_briefing.py"
SECRETS = {
    "TIINGO_API_KEY": "tiingo-secret-0f9e8d7c",
    "SMTP_PASSWORD": "smtp-secret-1a2b3c4d",
    "STOCK_LAB_EMAIL_TOKEN": "token-secret-5e6f7a8b",
}


def _script():
    spec = importlib.util.spec_from_file_location("stock_lab_briefing", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _script()


def _private(tmp_path: Path, name: str = "briefings") -> Path:
    path = tmp_path / name
    path.mkdir(mode=0o700)
    return path


def _briefing(market_date="2026-10-19", data_class="synthetic") -> RenderedBriefing:
    return RenderedBriefing(market_date, data_class, "subject", "text", "<p>html</p>", "{}")


class FakeAdapter:
    """Counts deliveries; can be told to fail with a message that holds a secret."""

    channel = "fake"

    def __init__(self, error: Exception | None = None) -> None:
        self.sent: list[str] = []
        self.error = error

    def deliver(self, briefing: RenderedBriefing) -> str:
        if self.error is not None:
            raise self.error
        self.sent.append(briefing.market_date)
        return "delivered"


# --- the private directory ---


def test_the_directory_is_refused_inside_a_git_working_tree(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    with pytest.raises(DeliveryRefused, match="Git working tree"):
        private_directory(tmp_path / "repo" / "out")


def test_the_directory_is_refused_beside_the_research_store_or_open_to_others(tmp_path):
    shared = _private(tmp_path, "shared")
    (shared / DB_NAME).write_bytes(b"")
    with pytest.raises(DeliveryRefused, match="research store"):
        private_directory(shared)
    open_dir = tmp_path / "open"
    open_dir.mkdir(mode=0o755)
    open_dir.chmod(0o755)
    with pytest.raises(DeliveryRefused, match="other users"):
        private_directory(open_dir)


def test_a_missing_directory_is_created_private(tmp_path):
    made = private_directory(tmp_path / "new" / "briefings")
    assert stat.S_IMODE(made.stat().st_mode) == 0o700


def test_the_default_directory_is_private_and_outside_the_repository(monkeypatch):
    monkeypatch.delenv(delivery.HOME_ENV, raising=False)
    default = delivery.default_directory()
    assert default == Path.home() / "stock-agent-private" / "stock_lab_briefings"
    assert ROOT not in default.parents


def test_the_preview_adapter_writes_private_files_once(tmp_path):
    adapter = PreviewAdapter(_private(tmp_path))
    receipt = adapter.deliver(_briefing())
    paths = adapter.paths(_briefing())
    assert set(paths) == {"html", "txt", "json"} and all(p.name in receipt for p in paths.values())
    for path in paths.values():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert paths["html"].read_text() == "<p>html</p>"
    with pytest.raises(FileExistsError):
        adapter.deliver(_briefing())


# --- duplicate-send protection ---


def test_a_market_date_is_delivered_once(tmp_path):
    directory = _private(tmp_path)
    adapter, log = FakeAdapter(), DeliveryLog(directory)
    first = deliver_once(_briefing(), adapter, log)
    second = deliver_once(_briefing(), adapter, log)
    assert (first.status, first.attempt) == (SENT, 1)
    assert second.status == DUPLICATE and second.attempt is None
    assert adapter.sent == ["2026-10-19"]  # the adapter was not called again
    assert [row["status"] for row in log.history()] == ["claimed", "sent"]
    assert all(row["at"].endswith("Z") for row in log.history())


def test_other_market_dates_and_the_other_data_class_are_separate(tmp_path):
    adapter, log = FakeAdapter(), DeliveryLog(_private(tmp_path))
    assert deliver_once(_briefing("2026-10-19"), adapter, log).status == SENT
    assert deliver_once(_briefing("2026-10-20"), adapter, log).status == SENT
    assert deliver_once(_briefing("2026-10-19", "real"), adapter, log).status == SENT
    assert adapter.sent == ["2026-10-19", "2026-10-20", "2026-10-19"]


def test_a_failure_is_recorded_by_type_only_and_may_be_retried(tmp_path):
    log = DeliveryLog(_private(tmp_path))
    failing = FakeAdapter(RuntimeError(f"auth failed for user with {SECRETS['SMTP_PASSWORD']}"))
    result = deliver_once(_briefing(), failing, log)
    assert result.status == FAILED and result.detail == "failed: RuntimeError"
    assert SECRETS["SMTP_PASSWORD"] not in log.path.read_bytes().decode("latin-1")
    retry = deliver_once(_briefing(), FakeAdapter(), log)
    assert (retry.status, retry.attempt) == (SENT, 2)


def test_an_attempt_that_never_finished_blocks_a_second_send(tmp_path):
    log = DeliveryLog(_private(tmp_path))
    assert log.claim("2026-10-19", "fake", "synthetic") == ("claimed", 1)
    adapter = FakeAdapter()
    result = deliver_once(_briefing(), adapter, log)
    assert result.status == IN_PROGRESS and adapter.sent == []


def test_two_runs_at_the_same_moment_claim_at_most_once(tmp_path):
    directory = _private(tmp_path)
    barrier, results = threading.Barrier(4), []

    def run():
        barrier.wait()
        results.append(DeliveryLog(directory).claim("2026-10-19", "fake", "synthetic")[0])

    threads = [threading.Thread(target=run) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results) == ["claimed", "in_progress", "in_progress", "in_progress"]


def test_the_log_is_append_only(tmp_path):
    log = DeliveryLog(_private(tmp_path))
    deliver_once(_briefing(), FakeAdapter(), log)
    connection = sqlite3.connect(log.path)
    for statement in ("UPDATE deliveries SET status = 'failed'", "DELETE FROM deliveries"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(statement)
    connection.close()


# --- credentials: nothing read, nothing written ---


def _python_files():
    return [*sorted(BRIEFING.glob("*.py")), SCRIPT]


def test_no_briefing_code_imports_mail_network_or_credential_modules():
    forbidden = {"smtplib", "imaplib", "getpass", "keyring", "subprocess", "socket", "ssl",
                 "urllib", "http", "requests", "httpx", "email"}  # fmt: skip
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.split(".")[0])
        assert not names & forbidden, (path.name, names & forbidden)


def test_the_briefing_reads_only_directory_variables_never_a_credential():
    # Its own directory variable here; the store's location variable through
    # store.default_path (also a path). Nothing else, and no credential, is read.
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv"):
                assert path.name == "delivery.py", path.name
    source = (BRIEFING / "delivery.py").read_text(encoding="utf-8")
    assert source.count("os.environ") == 1 and "os.environ.get(HOME_ENV" in source
    for path in _python_files():
        code = path.read_text(encoding="utf-8")
        assert "from os import" not in code and "dotenv" not in code, path.name


def test_secrets_in_the_environment_never_reach_any_output(tmp_path, monkeypatch):
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    lab = Lab(tmp_path)
    lab.watch("AMD", "MU")
    directory = _private(tmp_path)
    out = io.StringIO()
    code = script.main(
        ["--preview", "--at", "2026-10-19T08:30:00-04:00", "--directory", str(directory)],
        records_factory=lambda: ReadOnlyRecords(lab.path),
        out=out,
    )
    assert code == 0
    written = [out.getvalue()] + [
        path.read_bytes().decode("latin-1") for path in directory.iterdir() if path.is_file()
    ]
    for value in SECRETS.values():
        assert not any(value in text for text in written)


# --- the script ---


def _run(lab, *args, directory=None):
    out = io.StringIO()
    extra = ["--directory", str(directory)] if directory else []
    code = script.main([*args, *extra], records_factory=lambda: ReadOnlyRecords(lab.path), out=out)
    return code, out.getvalue()


def test_a_closed_day_is_skipped_unless_asked_for(tmp_path):
    lab = Lab(tmp_path)
    directory = _private(tmp_path)
    code, text = _run(lab, "--preview", "--at", "2026-11-26T08:30:00-05:00", directory=directory)
    assert code == 0 and text.startswith("skipped: the market is closed on Thursday 26 November")
    assert list(directory.iterdir()) == []
    code, text = _run(lab, "--at", "2026-10-18T09:00:00-04:00", "--include-closed-days")
    assert code == 0 and "For Monday 19 October 2026" in text
    assert "On a day the market is closed: a preview" in text


def test_the_script_previews_once_per_market_date(tmp_path):
    lab = Lab(tmp_path)
    lab.watch("AMD", "MRVL", "MU", "NVDA", "SKHY", "SNDK")
    directory = _private(tmp_path)
    at = ("--at", "2026-10-19T08:30:00-04:00")
    code, text = _run(lab, "--preview", *at, directory=directory)
    assert code == 0 and "preview: sent" in text
    assert "== AMD · Unavailable ==" in text and "SKHY ·" not in text
    assert "By the owner's choice: SKHY, SNDK" in text
    assert {p.suffix for p in directory.iterdir()} == {".html", ".txt", ".json", ".sqlite3"}
    code, text = _run(lab, "--preview", *at, directory=directory)
    assert code == 0 and "preview: duplicate" in text


def test_flags_are_exact_and_times_need_a_zone(tmp_path):
    lab = Lab(tmp_path)
    with pytest.raises(SystemExit):
        _run(lab, "--prev")
    code, text = _run(lab, "--at", "2026-10-19T08:30")
    assert code == 2 and "time zone" in text


def test_the_script_never_installs_a_schedule_or_writes_the_store():
    source = SCRIPT.read_text(encoding="utf-8")
    for needle in ("launchctl", "LaunchAgents", "crontab", "record_forecast", "record_scores",
                   "stock_lab_daily"):  # fmt: skip
        assert needle not in source, needle


def test_the_demo_is_synthetic_from_its_first_line():
    out = io.StringIO()
    assert script.main(["--demo", "2"], out=out) == 0
    text = out.getvalue()
    assert text.startswith("*** SYNTHETIC DEMO")
    assert text.count("· Fresh ==") == 3
    assert "Last successful update" in text and "synthetic_weekdays" in text


def test_preview_files_stay_out_of_the_repository_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv(delivery.HOME_ENV, str(ROOT / "reports" / "briefings"))
    with pytest.raises(DeliveryRefused):
        private_directory()
    assert not (ROOT / "reports" / "briefings").exists()


def test_a_preview_outside_the_premarket_window_never_uses_the_days_delivery(tmp_path):
    lab = Lab(tmp_path)
    lab.watch("AMD")
    directory = _private(tmp_path)
    code, text = _run(lab, "--preview", "--at", "2026-10-19T20:30:00-04:00", directory=directory)
    assert code == 0 and "preview: written" in text and "a preview, not a delivery" in text
    assert text.startswith("BRIEFING PREVIEW")
    code, text = _run(lab, "--preview", "--at", "2026-10-20T08:30:00-04:00", directory=directory)
    assert code == 0 and "preview: sent" in text  # Tuesday's premarket delivery is still due
    names = sorted(p.name for p in directory.iterdir() if p.suffix == ".html")
    assert names == [
        "2026-10-20_real_advance-preview-003000_briefing.html",
        "2026-10-20_real_premarket_briefing.html",
    ]


def test_a_claim_waits_for_a_concurrent_claim_and_then_sees_it(tmp_path):
    directory = _private(tmp_path)
    log = DeliveryLog(directory)
    assert log.claim("2026-10-16", "fake", "synthetic")[0] == "claimed"  # creates the table
    holder = sqlite3.connect(log.path, isolation_level=None, timeout=10)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute(
        "INSERT INTO deliveries (market_date, channel, data_class, attempt, status, at) "
        "VALUES ('2026-10-19', 'fake', 'synthetic', 1, 'claimed', 'x')"
    )
    results = []
    thread = threading.Thread(
        target=lambda: results.append(log.claim("2026-10-19", "fake", "synthetic")[0])
    )
    thread.start()
    thread.join(timeout=0.5)
    assert thread.is_alive()  # blocked behind the open immediate transaction
    holder.execute("COMMIT")
    holder.close()
    thread.join(timeout=10)
    assert results == ["in_progress"]


def test_a_failure_part_way_leaves_no_preview_files_and_a_retry_is_sent(tmp_path):
    directory = _private(tmp_path)
    adapter, log = PreviewAdapter(directory), DeliveryLog(directory)
    blocking = adapter.paths(_briefing())["txt"]
    blocking.write_text("an earlier file")  # the second link will fail
    assert deliver_once(_briefing(), adapter, log).status == FAILED
    assert not adapter.paths(_briefing())["html"].exists()  # the first link was undone
    assert blocking.read_text() == "an earlier file"  # a file it did not create is untouched
    assert not [p for p in directory.iterdir() if p.name.endswith(".partial")]
    blocking.unlink()
    assert deliver_once(_briefing(), adapter, log).status == SENT


def test_a_receipt_that_is_not_plain_file_names_is_stored_as_sent():
    assert delivery.safe_receipt("a.html, a.txt") == "a.html, a.txt"
    assert delivery.safe_receipt("sent to someone@example.com") == "sent"
    assert delivery.safe_receipt(None) == "sent"


def test_a_failure_that_cannot_be_recorded_never_chains_the_secret(tmp_path):
    import traceback

    class BrokenLog(DeliveryLog):
        def finish(self, *args, **kwargs):
            raise sqlite3.OperationalError("database is locked")

    log = BrokenLog(_private(tmp_path))
    failing = FakeAdapter(RuntimeError(f"auth failed: {SECRETS['SMTP_PASSWORD']}"))
    with pytest.raises(DeliveryRefused) as raised:
        deliver_once(_briefing(), failing, log)
    printed = "".join(traceback.format_exception(raised.value))
    assert SECRETS["SMTP_PASSWORD"] not in printed and "OperationalError" in printed


def test_the_directory_and_log_must_belong_to_this_user_and_be_real_files(tmp_path, monkeypatch):
    directory = _private(tmp_path)
    (directory / delivery.LOG_NAME).symlink_to(tmp_path / "elsewhere.sqlite3")
    with pytest.raises(DeliveryRefused, match="regular file"):
        DeliveryLog(directory).claim("2026-10-19", "fake", "synthetic")
    other = os.getuid() + 1
    monkeypatch.setattr(delivery.os, "getuid", lambda: other)
    with pytest.raises(DeliveryRefused, match="another user"):
        private_directory(_private(tmp_path, "mine"))


def test_the_research_stores_home_is_never_the_briefing_directory(tmp_path, monkeypatch):
    home = _private(tmp_path, "lab_home")
    monkeypatch.setenv("STOCK_AGENT_STOCK_LAB_HOME", str(home))
    with pytest.raises(DeliveryRefused, match="research store"):
        private_directory(home)
