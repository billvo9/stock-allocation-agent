"""
Reading back the briefings already written (`stock_lab.briefing.stored`): a
written JSON copy reads back field for field, anything malformed or out of
place is refused with a short reason (never skipped in silence, never
repeated at length), the latest briefing comes first, the two data classes
are never mixed, the delivery log is read without writing, and nothing in the
folder is ever created or changed. Synthetic fixtures and the made-up ticker
SYNTH; nothing leaves the machine.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sqlite3
import threading
from pathlib import Path

import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, build, ny

from stock_agent.stock_lab.briefing import stored
from stock_agent.stock_lab.briefing.delivery import (
    DeliveryLog,
    PreviewAdapter,
    RenderedBriefing,
    deliver_once,
)
from stock_agent.stock_lab.briefing.report import (
    SCHEMA_VERSION,
    ReadOnlyRecords,
    briefing_document,
    briefing_from_document,
)
from stock_agent.stock_lab.briefing.stored import BriefingUnreadable, list_briefings, read_briefing

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stock_lab_briefing.py"
MONDAY = "2026-10-19_real_premarket_briefing.json"


def _script():
    spec = importlib.util.spec_from_file_location("stock_lab_briefing_for_stored", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _script()


def private(tmp_path: Path, name: str = "briefings") -> Path:
    directory = tmp_path / name
    directory.mkdir(mode=0o700, parents=True)
    directory.chmod(0o700)
    return directory


def snapshot(directory: Path) -> dict[str, tuple]:
    """The folder's own mode and time, then every entry's bytes, mode and time."""

    info = directory.lstat()
    out = {".": (info.st_mode, info.st_mtime_ns)}
    for path in sorted(directory.iterdir()):
        info = path.lstat()
        out[path.name] = (path.read_bytes(), info.st_mode, info.st_mtime_ns)
    return out


def real_lab(tmp_path: Path) -> Lab:
    """A real-class store: fresh SYNTH forecasts as of 2026-10-15 and 10-16, AMD unforecast."""

    lab = Lab(tmp_path)
    lab.watch(REAL_TEST_TICKER, "AMD", at=ny("2026-10-01", "12:00"))
    lab.forecast(REAL_TEST_TICKER, "2026-10-15", p=(0.25, 0.45, 0.30))
    lab.forecast(REAL_TEST_TICKER, "2026-10-16", p=(0.2, 0.5, 0.3))
    lab.job(at=ny("2026-10-16", "21:10"), expected="2026-10-16", status="ok", activation=None)
    return lab


def write_real(lab: Lab, directory: Path, at: str) -> str:
    out = io.StringIO()
    code = script.main(
        ["--preview", "--at", at, "--directory", str(directory), "--tickers", REAL_TEST_TICKER, "AMD"],
        records_factory=lambda: ReadOnlyRecords(lab.path),
        out=out,
    )  # fmt: skip
    assert code == 0, out.getvalue()
    return out.getvalue()


def put(directory: Path, name: str, text: str) -> Path:
    path = directory / name
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)
    return path


@pytest.fixture(scope="module")
def demo_folder(tmp_path_factory) -> Path:
    directory = private(tmp_path_factory.mktemp("demo"))
    out = io.StringIO()
    assert script.main(["--demo", "4", "--preview", "--directory", str(directory)], out=out) == 0
    return directory


def _demo_path(demo_folder: Path) -> Path:
    (path,) = sorted(demo_folder.glob("*_synthetic_premarket_briefing.json"))
    return path


def _demo_document(demo_folder: Path) -> dict:
    return json.loads(_demo_path(demo_folder).read_text(encoding="utf-8"))


# --- reading one briefing back ---


def test_a_written_demo_briefing_reads_back_field_for_field(demo_folder):
    path = _demo_path(demo_folder)
    briefing = read_briefing(path)
    assert briefing.data_class == "synthetic" and briefing.state == "premarket"
    assert briefing.stocks and all(stock.status == "fresh" for stock in briefing.stocks)
    assert briefing.reliability.prospective.groups  # buckets, rolling and groups carried
    assert briefing_document(briefing) == json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("moment", "state", "tickers"),
    [
        (("2026-10-19", "08:30"), "premarket", [REAL_TEST_TICKER, "AMD"]),
        (("2026-10-19", "10:00"), "after_open", [REAL_TEST_TICKER, "AMD"]),
        (("2026-10-16", "20:30"), "advance", [REAL_TEST_TICKER, "AMD"]),
        (("2026-10-17", "12:00"), "closed_day", [REAL_TEST_TICKER]),
        (("2026-10-19", "08:30"), "premarket", []),  # an empty watchlist
    ],
)
def test_real_briefings_in_every_state_round_trip_exactly(tmp_path, moment, state, tickers):
    briefing = build(real_lab(tmp_path), ny(*moment), data_class="real", tickers=tickers)
    assert briefing.state == state and len(briefing.stocks) == len(tickers)
    document = json.loads(json.dumps(briefing_document(briefing), allow_nan=False))
    assert briefing_from_document(document) == briefing


def test_a_fresh_real_stock_carries_its_change_track_and_lineage(tmp_path):
    lab = real_lab(tmp_path)
    briefing = build(lab, ny("2026-10-19", "08:30"), data_class="real",
                     tickers=[REAL_TEST_TICKER, "AMD"])  # fmt: skip
    assert [s.status for s in briefing.stocks] == ["fresh", "unavailable"]
    assert briefing.stocks[0].change.comparable and briefing.stocks[0].lineage is not None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(extra=1), "other fields"),
        (lambda d: d.pop("warnings"), "other fields"),
        (lambda d: d["stocks"][0].update(symbol=7), "is not a str"),
        (lambda d: d["ledger"].update(entries=True), "is not a int"),
        (lambda d: d["ledger"].update(entries=1.0), "is not a int"),
        (lambda d: d["ledger"].update(verified=1), "is not a bool"),
        (lambda d: d["stocks"][0]["lineage"].update(band="0.005"), "is not a number"),
        (lambda d: d["stocks"][0]["lineage"].update(band=True), "is not a number"),
        (lambda d: d["stocks"][0]["lineage"].update(band=None), "is not a number"),
        (
            lambda d: d["stocks"][0]["next_session"]["probabilities"].update(stable=None),
            "is not a number",
        ),
        (lambda d: d.update(stocks={}), "is not a list"),
        (lambda d: d.update(schema_version="0.9"), "not stock_lab_premarket_briefing"),
    ],
)
def test_a_document_that_does_not_fit_exactly_is_refused(demo_folder, change, message):
    document = _demo_document(demo_folder)
    change(document)
    with pytest.raises((TypeError, ValueError), match=message):
        briefing_from_document(document)


@pytest.mark.parametrize(
    ("name", "text", "reason"),
    [
        ("2025-01-02_synthetic_premarket_briefing.json", None, "name and its contents"),
        ("2025-01-01_real_premarket_briefing.json", None, "name and its contents"),
        ("2025-01-01_synthetic_advance-preview-003000_briefing.json", None, "different states"),
        ("2025-01-01_synthetic_premarket_briefing.json", "{not json", "not valid JSON"),
        ("2025-01-01_synthetic_premarket_briefing.json", '{"a": NaN}', "strict JSON"),
        ("2025-01-01_synthetic_premarket_briefing.json", '{"a": 1, "a": 2}', "repeats a field"),
        ("2025-01-01_synthetic_premarket_briefing.json", "[" * 100_000, "not valid JSON"),
        ("2025-01-01_synthetic_premarket_briefing.json", "9" * 5000, "not valid JSON"),
    ],
)
def test_a_file_out_of_place_is_refused_with_its_reason(demo_folder, tmp_path, name, text, reason):
    source = _demo_path(demo_folder)
    target = put(private(tmp_path), name, text if text is not None else source.read_text())
    with pytest.raises(BriefingUnreadable, match=reason):
        read_briefing(target)


def test_a_reason_never_repeats_the_files_contents_at_length(demo_folder, tmp_path):
    document = _demo_document(demo_folder)
    document["schema"] = "x" * 5000
    document.update({f"extra_{i}": i for i in range(50)})
    target = put(private(tmp_path), _demo_path(demo_folder).name, json.dumps(document))
    listing = list_briefings("synthetic", target.parent)
    (entry,) = listing.briefings
    assert entry.briefing is None and len(entry.problem) <= stored.MAX_REASON
    assert "x" * 40 not in entry.problem


def test_a_briefing_the_renderer_cannot_write_is_refused(demo_folder, tmp_path):
    document = _demo_document(demo_folder)
    document["generated_at"] = "not a time"
    target = put(private(tmp_path), _demo_path(demo_folder).name, json.dumps(document))
    with pytest.raises(BriefingUnreadable, match="cannot be shown"):
        read_briefing(target)


def test_a_file_others_can_open_another_users_a_link_or_a_large_one_is_never_read(
    demo_folder, tmp_path, monkeypatch
):
    source = _demo_path(demo_folder)
    directory = private(tmp_path)
    shared = put(directory, source.name, source.read_text())
    shared.chmod(0o644)
    with pytest.raises(BriefingUnreadable, match="open to other users"):
        read_briefing(shared)
    shared.chmod(0o600)
    link = private(tmp_path, "other") / source.name
    link.symlink_to(source)
    with pytest.raises(BriefingUnreadable, match="could not be opened"):
        read_briefing(link)
    monkeypatch.setattr(stored, "MAX_BYTES", 100)
    with pytest.raises(BriefingUnreadable, match="larger than 100 bytes"):
        read_briefing(shared)
    monkeypatch.setattr(stored, "MAX_BYTES", 2_000_000)
    owner = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    with pytest.raises(BriefingUnreadable, match="belongs to another user"):
        read_briefing(shared)


def test_a_named_pipe_is_refused_without_waiting(tmp_path):
    directory = private(tmp_path)
    os.mkfifo(directory / MONDAY, 0o600)
    found = []
    reader = threading.Thread(target=lambda: found.append(list_briefings("real", directory)))
    reader.daemon = True
    reader.start()
    reader.join(5)
    assert not reader.is_alive(), "list_briefings waited on a named pipe"
    (entry,) = found[0].briefings
    assert "not a regular file" in entry.problem


def test_a_file_swapped_for_a_link_after_listing_is_not_followed(
    demo_folder, tmp_path, monkeypatch
):
    source = _demo_path(demo_folder)
    directory = private(tmp_path)
    target = put(directory, source.name, source.read_text())
    listed = stored._names

    def swap(folder, data_class):
        names = listed(folder, data_class)
        target.unlink()
        target.symlink_to(source)
        return names

    monkeypatch.setattr(stored, "_names", swap)
    (entry,) = list_briefings("synthetic", directory).briefings
    assert entry.briefing is None and "could not be opened" in entry.problem


# --- the folder ---


def test_a_missing_folder_lists_nothing_and_is_not_created(tmp_path):
    missing = tmp_path / "nothing" / "briefings"
    listing = list_briefings("real", missing)
    assert (listing.state, listing.briefings, listing.older) == (stored.NONE, (), 0)
    assert stored.delivery_status("2026-10-19", "real", missing)[0].status == stored.NOT_RECORDED
    assert not missing.exists() and not missing.parent.exists()


def test_a_folder_others_can_open_or_inside_a_working_tree_is_refused(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    listing = list_briefings("real", shared)
    assert listing.state == stored.REFUSED and "chmod 700" in listing.reason
    tree = tmp_path / "checkout"
    (tree / ".git").mkdir(parents=True)
    inside = private(tree, "briefings")
    listing = list_briefings("synthetic", inside)
    assert listing.state == stored.REFUSED and "Git working tree" in listing.reason
    status = stored.delivery_status("2026-10-19", "real", inside)[0]
    assert status.status == stored.UNREADABLE and "Git working tree" in status.detail
    link = tmp_path / "linked-briefings"  # a link from outside into the working tree
    link.symlink_to(inside)
    assert list_briefings("real", link).state == stored.REFUSED
    owned = private(tmp_path, "owned")
    owner = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    listing = list_briefings("real", owned)
    assert listing.state == stored.REFUSED and "another user" in listing.reason


@pytest.mark.skipif(os.getuid() == 0, reason="root reads a folder of mode 000")
def test_a_folder_that_cannot_be_listed_is_refused_with_the_error_type_only(tmp_path):
    closed = private(tmp_path, "closed")
    closed.chmod(0o000)
    try:
        listing = list_briefings("real", closed)
    finally:
        closed.chmod(0o700)
    assert listing.state == stored.REFUSED
    assert listing.reason.startswith("the briefing directory could not be")
    assert listing.reason.endswith("(PermissionError)") and str(tmp_path) not in listing.reason


def test_names_that_are_not_a_briefings_are_not_listed(demo_folder, tmp_path):
    directory = private(tmp_path)
    source = _demo_path(demo_folder).read_text()
    for name in ("2026-13-45_synthetic_premarket_briefing.json",
                 "２０２５-01-01_synthetic_premarket_briefing.json",  # full-width digits
                 "2025-01-01_synthetic_premarket_briefing.html",
                 ".2025-01-01_synthetic_premarket_briefing.json.partial"):  # fmt: skip
        put(directory, name, source)
    assert list_briefings("synthetic", directory).state == stored.NONE


def test_the_latest_briefing_comes_first_and_classes_are_never_mixed(tmp_path, demo_folder):
    lab = real_lab(tmp_path)
    directory = private(tmp_path)
    write_real(lab, directory, "2026-10-16T08:30:00-04:00")  # Friday's premarket briefing
    write_real(lab, directory, "2026-10-16T20:30:00-04:00")  # Friday evening, for Monday
    write_real(lab, directory, "2026-10-19T08:30:00-04:00")  # Monday's premarket briefing
    write_real(lab, directory, "2026-10-19T10:00:00-04:00")  # Monday, after the open
    for path in demo_folder.glob("*.json"):  # a synthetic demo in the same folder
        put(directory, path.name, path.read_text())
    listing = list_briefings("real", directory)
    # By preparation time, not by name: the after-open preview sorts before "premarket"
    # by name but was prepared later.
    assert [entry.name for entry in listing.briefings] == [
        "2026-10-19_real_after-open-preview-140000_briefing.json",
        MONDAY,
        "2026-10-19_real_advance-preview-003000_briefing.json",
        "2026-10-16_real_premarket_briefing.json",
    ]
    assert [entry.briefing.state for entry in listing.briefings] == [
        "after_open", "premarket", "advance", "premarket",
    ]  # fmt: skip
    assert all(entry.briefing.data_class == "real" for entry in listing.briefings)
    synthetic = list_briefings("synthetic", directory)
    assert [entry.data_class for entry in synthetic.briefings] == ["synthetic"]
    assert synthetic.briefings[0].briefing.schema_version == SCHEMA_VERSION


def test_an_unreadable_file_is_listed_with_its_reason_after_the_readable_ones(tmp_path):
    lab = real_lab(tmp_path)
    directory = private(tmp_path)
    write_real(lab, directory, "2026-10-19T08:30:00-04:00")
    put(directory, "2026-10-20_real_premarket_briefing.json", "{}")
    listing = list_briefings("real", directory)
    names = [(entry.name, entry.problem is None) for entry in listing.briefings]
    assert names == [(MONDAY, True), ("2026-10-20_real_premarket_briefing.json", False)]
    assert "not stock_lab_premarket_briefing" in listing.briefings[1].problem


@pytest.mark.parametrize(("limit", "older"), [(1, 2), (2, 2), (3, 1), (4, 0), (5, 0)])
def test_whole_market_dates_are_read_and_older_files_are_counted_never_opened(
    tmp_path, demo_folder, monkeypatch, limit, older
):
    lab = real_lab(tmp_path)
    directory = private(tmp_path)
    for at in ("2026-10-15T08:30:00-04:00", "2026-10-16T08:30:00-04:00",
               "2026-10-19T08:30:00-04:00", "2026-10-19T10:00:00-04:00"):  # fmt: skip
        write_real(lab, directory, at)
    for path in demo_folder.glob("*.json"):  # never counted with the real ones
        put(directory, path.name, path.read_text())
    opened = []
    reader = stored.read_briefing
    monkeypatch.setattr(
        stored, "read_briefing", lambda path: opened.append(path.name) or reader(path)
    )
    monkeypatch.setattr(stored, "MAX_LISTED", limit)
    listing = list_briefings("real", directory)
    dates = [entry.market_date for entry in listing.briefings]
    expected = ["2026-10-19", "2026-10-19", "2026-10-16", "2026-10-15"][: 4 - older]
    assert dates == expected and listing.older == older
    assert sorted(opened) == sorted(entry.name for entry in listing.briefings)


def test_listing_and_reading_the_log_change_nothing_in_the_folder(tmp_path, demo_folder):
    lab = real_lab(tmp_path)
    directory = private(tmp_path)
    write_real(lab, directory, "2026-10-19T08:30:00-04:00")
    before = snapshot(directory)
    for kind in ("real", "synthetic"):
        list_briefings(kind, directory)
    stored.delivery_status("2026-10-19", "real", directory)
    assert snapshot(directory) == before
    os.remove(directory / stored.LOG_NAME)
    stored.delivery_status("2026-10-19", "real", directory)
    assert not (directory / stored.LOG_NAME).exists()  # read-only: never created


# --- the delivery log, read only ---


def _rendered(market_date: str = "2026-10-19") -> RenderedBriefing:
    return RenderedBriefing(market_date, "real", "subject", "text", "<html></html>", "{}")


class Failing:
    channel = "fake"

    def deliver(self, briefing):
        raise RuntimeError("smtp://user:secret@example.invalid refused")


def test_the_log_reports_sent_failed_and_unfinished_per_channel(tmp_path):
    directory = private(tmp_path)
    log = DeliveryLog(directory)
    deliver_once(_rendered(), PreviewAdapter(directory), log)
    deliver_once(_rendered(), Failing(), log)
    log.claim("2026-10-20", "preview", "real")  # a run that died mid-delivery
    statuses = {s.channel: s for s in stored.delivery_status("2026-10-19", "real", directory)}
    assert statuses["preview"].status == stored.SENT and statuses["preview"].attempts == 1
    assert statuses["fake"].status == stored.FAILED
    assert statuses["fake"].detail == "failed: RuntimeError"  # never the message
    (unfinished,) = stored.delivery_status("2026-10-20", "real", directory)
    assert (unfinished.channel, unfinished.status) == ("preview", stored.IN_PROGRESS)
    (other,) = stored.delivery_status("2026-10-19", "synthetic", directory)
    assert other.status == stored.NOT_RECORDED


def test_a_later_success_after_a_failure_reads_as_sent(tmp_path):
    directory = private(tmp_path)
    log = DeliveryLog(directory)

    class FailsOnce:
        channel = "preview"
        calls = 0

        def deliver(self, briefing):
            FailsOnce.calls += 1
            if FailsOnce.calls == 1:
                raise OSError("disk")
            return "files"

    adapter = FailsOnce()
    assert deliver_once(_rendered(), adapter, log).status == "failed"
    assert deliver_once(_rendered(), adapter, log).status == "sent"
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert (status.status, status.attempts) == (stored.SENT, 2)


def test_the_writer_keeps_the_rollback_journal_the_reader_needs(tmp_path):
    directory = private(tmp_path)
    log = DeliveryLog(directory)
    log.claim("2026-10-19", "preview", "real")
    header = log.path.read_bytes()[:100]
    assert header.startswith(stored.SQLITE_HEADER) and header[18:20] == b"\x01\x01"


def test_a_write_ahead_log_is_refused_without_creating_its_files(tmp_path):
    directory = private(tmp_path)
    log = DeliveryLog(directory)
    log.claim("2026-10-19", "preview", "real")
    connection = sqlite3.connect(log.path)
    assert connection.execute("PRAGMA journal_mode=WAL").fetchone() == ("wal",)
    connection.close()
    before = snapshot(directory)
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert status.status == stored.UNREADABLE and "rollback-journal" in status.detail
    assert snapshot(directory) == before
    assert not list(directory.glob("*-wal")) and not list(directory.glob("*-shm"))


def test_a_log_that_cannot_be_read_says_so_without_its_contents(tmp_path, monkeypatch):
    directory = private(tmp_path)
    log = directory / stored.LOG_NAME
    log.write_bytes(b"not a database, secret-0f9e8d7c" * 10)
    log.chmod(0o600)
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert status.status == stored.UNREADABLE and "not an SQLite database" in status.detail
    assert "secret" not in status.detail
    log.chmod(0o644)
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert status.status == stored.UNREADABLE and "chmod 600" in status.detail
    assert log.read_bytes() == b"not a database, secret-0f9e8d7c" * 10
    log.unlink()
    elsewhere = private(tmp_path, "elsewhere")
    DeliveryLog(elsewhere).claim("2026-10-19", "preview", "real")
    log.symlink_to(elsewhere / stored.LOG_NAME)
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert status.status == stored.UNREADABLE and "could not be opened" in status.detail
    log.unlink()
    DeliveryLog(directory).claim("2026-10-19", "preview", "real")
    owner = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: owner + 1)
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert status.status == stored.UNREADABLE and "another user" in status.detail


@pytest.mark.parametrize(
    "row",
    [
        ("preview", 1, "sent", "2026-10-19 08:30", None),  # no UTC stamp
        ("preview", 1, "bogus", "2026-10-19T12:30:00.000000Z", None),
        ("preview", 0, "sent", "2026-10-19T12:30:00.000000Z", None),
        ("bad channel!", 1, "sent", "2026-10-19T12:30:00.000000Z", None),
    ],
)
def test_a_log_entry_the_writer_never_writes_is_reported_not_read(tmp_path, row):
    directory = private(tmp_path)
    log = DeliveryLog(directory)
    log.claim("2026-10-18", "preview", "real")  # creates the table
    connection = sqlite3.connect(log.path)
    connection.execute("PRAGMA ignore_check_constraints = ON")
    connection.execute(
        "INSERT INTO deliveries (channel, attempt, status, at, detail, market_date, data_class) "
        "VALUES (?, ?, ?, ?, ?, '2026-10-19', 'real')",
        row,
    )
    connection.commit()
    connection.close()
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert status.status == stored.UNREADABLE and "not one the log writes" in status.detail


def test_a_detail_that_is_not_plain_text_is_not_passed_on(tmp_path):
    directory = private(tmp_path)
    log = DeliveryLog(directory)
    _, attempt = log.claim("2026-10-19", "preview", "real")
    log.finish("2026-10-19", "preview", "real", attempt, "failed",
               "failed: ![x](http://example.invalid/a.png)")  # fmt: skip
    (status,) = stored.delivery_status("2026-10-19", "real", directory)
    assert status.status == stored.FAILED and status.detail == stored.HIDDEN_DETAIL


def test_the_html_bucket_tables_hold_exactly_the_shared_rows(demo_folder):
    from html import escape

    from stock_agent.stock_lab.briefing.render import bucket_rows, render_html

    briefing = read_briefing(_demo_path(demo_folder))
    page = render_html(briefing, script.briefing_theme())
    groups = [g for g in briefing.reliability.prospective.groups if g.target == "1"]
    assert groups and all(group.buckets for group in groups)
    for group in groups:
        for row in bucket_rows(group.buckets):
            cells = "".join(f'<td class="num">{escape(cell)}</td>' for cell in row[1:])
            assert f'<tr><th scope="row">{escape(row[0])}</th>{cells}</tr>' in page


def test_the_reader_never_imports_the_writers():
    source = (ROOT / "src" / "stock_agent" / "stock_lab" / "briefing" / "stored.py").read_text()
    assert "delivery" not in "".join(
        line for line in source.splitlines() if line.startswith(("import", "from"))
    )
    for needle in ("mkdir", "write_text", "write_bytes", "O_CREAT", "O_WRONLY", "INSERT", "unlink"):
        assert needle not in source, needle
    sqlite_open = [line for line in source.splitlines() if "sqlite3.connect(" in line]
    assert sqlite_open and all("mode=ro" in line for line in sqlite_open)
