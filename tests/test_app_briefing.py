"""
The Lab's Premarket Briefing page under AppTest: the latest written briefing
is shown as written, real or synthetic demo and never both, labelled; every
category is written in words beside its marker; previews, unreadable files,
refused folders, a briefing that is not today's and the delivery log's states
are said in words; the comparison with B1 carries its persistence caveat only
when its flag is on; text from a file is shown literally; the page writes and
creates nothing; and every state passes the advice scan (with the trading
words of the briefing's own renderer tests). Synthetic fixtures and the
made-up ticker SYNTH; nothing leaves the machine.
"""

from __future__ import annotations

import ast
import functools
import io
import json
import re
from datetime import date
from pathlib import Path

import pytest
from stock_lab_briefing_helpers import REAL_TEST_TICKER, Lab, build, ny
from streamlit.testing.v1 import AppTest
from test_app_destinations import text_of
from test_stock_lab_briefing_render import ADVICE_PATTERNS
from test_stock_lab_briefing_report import _shifted
from test_stock_lab_briefing_stored import private, put, real_lab, script, snapshot, write_real

from stock_agent.app import briefing as page_module
from stock_agent.app import briefing_wording as words
from stock_agent.app import navigation, wording
from stock_agent.app.prose import literal
from stock_agent.stock_lab.briefing import render, report, stored
from stock_agent.stock_lab.briefing.delivery import DeliveryLog, deliver_once
from stock_agent.stock_lab.briefing.delivery import RenderedBriefing as Rendered
from stock_agent.stock_lab.briefing.report import briefing_document
from stock_agent.stock_lab.store import HOME_ENV as STORE_HOME_ENV

RUN = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "app" / "run.py"
TIMEOUT = 120
GLANCE = words.GLANCE_COLUMNS
MONDAY = "2026-10-19_real_premarket_briefing.json"


@pytest.fixture(scope="module")
def demo_files(tmp_path_factory) -> Path:
    """A synthetic demo briefing and its delivery log, written by the briefing script."""

    directory = private(tmp_path_factory.mktemp("demo"))
    assert script.main(["--demo", "4", "--preview", "--directory", str(directory)],
                       out=io.StringIO()) == 0  # fmt: skip
    return directory


@pytest.fixture(scope="module")
def compared(tmp_path_factory) -> dict:
    """
    One synthetic matched comparison with B1 whose difference shifts mid-record,
    built with its persistence flag on (as it is) and forced off (an infinite
    threshold), by flag.
    """

    flagged = _shifted(Lab(tmp_path_factory.mktemp("flagged")))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            report.rel,
            "matched_comparison",
            functools.partial(report.rel.matched_comparison, threshold=float("inf")),
        )
        plain = _shifted(Lab(tmp_path_factory.mktemp("plain")))
    return {True: flagged, False: plain}


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    directory = private(tmp_path, "briefings")
    monkeypatch.setenv(stored.HOME_ENV, str(directory))
    monkeypatch.setattr(page_module, "new_york_today", lambda: date(2026, 10, 19))
    return directory


def page() -> AppTest:
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    app.switch_page(navigation.PAGES["briefing"]).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def read(app: AppTest) -> str:
    """Every rendered text (scanned for advice), also scanned for trading words."""

    text = text_of(app)
    for pattern in ADVICE_PATTERNS:
        assert not re.search(pattern, text.lower()), pattern
    return text


def copy_demo(demo_files: Path, directory: Path) -> None:
    for path in demo_files.iterdir():
        put(directory, path.name, "")
        (directory / path.name).write_bytes(path.read_bytes())


def glance(app: AppTest):
    return next(t.value for t in app.table if GLANCE[1] in t.value.columns)


def write_document(directory: Path, briefing, name: str | None = None, change=None) -> Path:
    document = briefing_document(briefing)
    if change is not None:
        change(document)
    label = "premarket" if briefing.state == "premarket" else "x"
    name = name or f"{briefing.market_date}_{briefing.data_class}_{label}_briefing.json"
    return put(directory, name, json.dumps(document, allow_nan=False))


def test_with_nothing_written_the_page_says_how_to_write_one(tmp_path, monkeypatch):
    missing = tmp_path / "no" / "briefings"
    monkeypatch.setenv(stored.HOME_ENV, str(missing))
    app = page()
    text = read(app)
    assert words.TITLE in text and words.NONE_TITLE in text and words.READ_ONLY_BADGE in text
    assert words.NONE["real"].format(directory=missing) in text
    assert [code.proto.code_text for code in app.get("code")] == [words.COMMANDS["real"]]
    assert wording.CANNOT_TELL in text
    assert not missing.exists() and not missing.parent.exists()
    app.radio[0].set_value("synthetic").run()
    assert words.COMMANDS["synthetic"] in read(app)


def test_a_synthetic_demo_is_shown_labelled_exactly_as_written(home, demo_files):
    copy_demo(demo_files, home)
    app = page()
    assert app.radio[0].value == "synthetic"  # the only kind written
    text = read(app)
    for label in (words.SYNTHETIC_BADGE, words.SYNTHETIC_NOTICE, words.PREMARKET_BADGE,
                  words.DELIVERED, words.EXPERIMENTAL_BADGE):  # fmt: skip
        assert label in text, label
    assert words.NOT_TODAY[:20] not in text  # a synthetic date is never "not today"
    (path,) = home.glob("*_synthetic_premarket_briefing.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    table = glance(app)
    assert list(table.index) == [stock["symbol"] for stock in document["stocks"]]
    for stock in document["stocks"]:
        stored_p = stock["next_session"]["probabilities"]
        for category, column in (("increase", "▲ Increase"), ("stable", "● Stable"),
                                 ("decrease", "▼ Decrease")):  # fmt: skip
            assert table.loc[stock["symbol"], column] == render.pct(stored_p[category])
    assert "For Wednesday 1 January 2025 · target 08:30 New York time" in text
    assert len(app.expander) == len(document["stocks"])  # one per stock


def test_real_comes_first_and_the_two_kinds_are_never_mixed(home, demo_files, tmp_path):
    copy_demo(demo_files, home)
    write_real(real_lab(tmp_path), home, "2026-10-19T08:30:00-04:00")
    app = page()
    assert app.radio[0].value == "real"
    text = read(app)
    assert list(glance(app).index) == [REAL_TEST_TICKER, "AMD"]
    assert words.REAL_BADGE in text and words.SYNTHETIC_BADGE not in text and "DEMO_" not in text
    assert glance(app).loc["AMD", GLANCE[1]] == "Unavailable: no recorded forecast"
    assert glance(app).loc[REAL_TEST_TICKER, GLANCE[2]] == "Increase 0 · Stable +5 · Decrease −5"
    app.radio[0].set_value("synthetic").run()
    text = read(app)
    assert all(symbol.startswith("DEMO_") for symbol in glance(app).index)
    assert words.SYNTHETIC_BADGE in text and words.REAL_BADGE not in text
    assert REAL_TEST_TICKER not in " ".join(glance(app).index)


def test_every_category_is_written_in_words_beside_its_marker(home, demo_files, tmp_path):
    copy_demo(demo_files, home)
    write_real(real_lab(tmp_path), home, "2026-10-19T08:30:00-04:00")
    for kind in ("real", "synthetic"):
        app = page()
        app.radio[0].set_value(kind).run()
        text = read(app)
        assert "▲" in text and "●" in text and "▼" in text, kind
        for marker, word in (("▲", "Increase"), ("●", "Stable"), ("▼", "Decrease")):
            assert not re.search(f"{marker}(?! {word})", text), (kind, marker)


def test_the_latest_is_chosen_first_and_earlier_ones_stay_reachable(home, tmp_path):
    lab = real_lab(tmp_path)
    write_real(lab, home, "2026-10-16T08:30:00-04:00")
    write_real(lab, home, "2026-10-19T08:30:00-04:00")
    app = page()
    choice = app.selectbox[0]
    assert choice.options == [
        "Monday 19 October 2026 · premarket briefing, prepared 08:30 New York",
        "Friday 16 October 2026 · premarket briefing, prepared 08:30 New York",
    ]
    assert choice.value == MONDAY
    choice.set_value("2026-10-16_real_premarket_briefing.json").run()
    text = read(app)
    assert "For Friday 16 October 2026" in text
    assert words.NOT_TODAY.format(market="Friday 16 October 2026",
                                  today="Monday 19 October 2026") in text  # fmt: skip


def test_a_briefing_from_an_earlier_day_says_so(home, tmp_path, monkeypatch):
    write_real(real_lab(tmp_path), home, "2026-10-19T08:30:00-04:00")
    assert words.NOT_TODAY[:20] not in read(page())
    monkeypatch.setattr(page_module, "new_york_today", lambda: date(2026, 10, 20))
    text = read(page())
    assert words.NOT_TODAY.format(market="Monday 19 October 2026",
                                  today="Tuesday 20 October 2026") in text  # fmt: skip


def test_a_preview_is_labelled_and_never_read_as_the_delivery(home, tmp_path):
    write_real(real_lab(tmp_path), home, "2026-10-16T20:30:00-04:00")  # Friday evening
    app = page()
    text = read(app)
    assert words.PREVIEW_BADGE in text and words.PREVIEW_TITLE in text
    assert words.PREMARKET_BADGE not in text and "Delivered once" not in text
    assert words.PREVIEW_TEXT + words.PREVIEW_STATES["not_recorded"] in text
    assert app.selectbox[0].options == [
        "Monday 19 October 2026 · preview, prepared Fri 16 Oct, 20:30 New York"
    ]
    assert "Stale: newest forecast as of 2026-10-15" in glance(app).loc[REAL_TEST_TICKER, "Status"]
    assert "stale: stale" not in text.lower()


def test_a_preview_names_its_dates_delivery_once_across_channels(home, tmp_path):
    lab = real_lab(tmp_path)
    write_real(lab, home, "2026-10-19T08:30:00-04:00")  # the delivery
    write_real(lab, home, "2026-10-19T10:00:00-04:00")  # a late preview, the latest
    deliver_once(Rendered("2026-10-19", "real", "s", "t", "<p></p>", "{}"), FailingOther(),
                 DeliveryLog(home))  # fmt: skip
    app = page()
    assert app.selectbox[0].value == "2026-10-19_real_after-open-preview-140000_briefing.json"
    text = read(app)
    assert text.count(words.PREVIEW_TITLE) == 1
    assert "This market date's premarket briefing was delivered" in text


class FailingOther:
    channel = "other"

    def deliver(self, briefing):
        raise OSError("disk full at /Users/someone")


@pytest.mark.parametrize("state", ["failed", "unfinished", "not_recorded", "unreadable", "other"])
def test_the_delivery_logs_states_are_said_in_words(home, tmp_path, state):
    written = private(tmp_path, "written")
    write_real(real_lab(tmp_path), written, "2026-10-19T08:30:00-04:00")
    put(home, MONDAY, (written / MONDAY).read_text())
    rendered = Rendered("2026-10-19", "real", "s", "t", "<p></p>", "{}")
    if state == "failed":

        class Failing:
            channel = "preview"

            def deliver(self, briefing):
                raise OSError("disk full at /Users/someone")

        deliver_once(rendered, Failing(), DeliveryLog(home))
    elif state == "unfinished":
        DeliveryLog(home).claim("2026-10-19", "preview", "real")
    elif state == "unreadable":
        put(home, stored.LOG_NAME, "not a database")
    elif state == "other":  # only the synthetic class was delivered on this date
        synthetic = Rendered("2026-10-19", "synthetic", "s", "t", "<p></p>", "{}")
        deliver_once(synthetic, type("Ok", (), {"channel": "preview",
                                                "deliver": lambda self, b: "ok"})(),
                     DeliveryLog(home))  # fmt: skip
    text = read(page())
    expected = {
        "failed": (words.FAILED_TITLE, "(attempt 1): Failed: OSError."),
        "unfinished": (words.UNFINISHED_TITLE, "never sent twice"),
        "not_recorded": (words.NOT_RECORDED_TITLE, "moved or removed"),
        "unreadable": (words.LOG_UNREADABLE_TITLE, "The delivery log is not an SQLite database."),
        "other": (words.NOT_RECORDED_TITLE,),
    }[state]
    assert all(part in text for part in expected), state
    assert "/Users/someone" not in text
    assert ("Delivered once" in text) is False


def test_a_newer_unreadable_file_is_announced_above_the_older_readable_one(home, tmp_path):
    write_real(real_lab(tmp_path), home, "2026-10-16T08:30:00-04:00")
    put(home, MONDAY, '{"schema": "something else"}')
    app = page()
    assert app.selectbox[0].value == "2026-10-16_real_premarket_briefing.json"
    warning = next(w.value for w in app.warning if "newest briefing file" in w.value)
    assert "for Monday 19 October 2026, could not be read" in warning
    assert "the latest readable one, for Friday 16 October 2026" in warning
    text = read(app)
    assert words.OTHER_UNREADABLE[0].format(names=MONDAY) in text


def test_an_unreadable_briefing_is_named_and_left_alone(home):
    put(home, MONDAY, '{"schema": "something else"}')
    before = snapshot(home)
    text = read(page())
    assert words.UNREADABLE_TITLE in text and MONDAY in text
    assert "not stock_lab_premarket_briefing" in text
    assert snapshot(home) == before


def test_a_folder_other_users_can_open_is_not_read(home):
    home.chmod(0o755)
    text = read(page())
    assert words.REFUSED_TITLE in text and "chmod 700" in text


def test_older_files_are_counted_in_a_caption(home, tmp_path, monkeypatch):
    lab = real_lab(tmp_path)
    for at in ("2026-10-15T08:30:00-04:00", "2026-10-16T08:30:00-04:00",
               "2026-10-19T08:30:00-04:00"):  # fmt: skip
        write_real(lab, home, at)
    monkeypatch.setattr(stored, "MAX_LISTED", 1)
    text = read(page())
    assert words.OLDER[1].format(count=2) in text


def test_a_failed_ledger_check_is_a_blocking_notice(home, tmp_path):
    briefing = build(real_lab(tmp_path), ny("2026-10-19", "08:30"), data_class="real",
                     tickers=[REAL_TEST_TICKER])  # fmt: skip

    def fail(document):
        document["ledger"].update(verified=False, problem="entry 3 does not chain")
        document["warnings"].append(
            "The ledger check failed (entry 3 does not chain): every stored probability is "
            "withheld."
        )

    write_document(home, briefing, change=fail)
    app = page()
    (blocking,) = [e.value for e in app.error]
    assert blocking.startswith("**Blocking.**") and "entry 3 does not chain" in blocking
    assert all("ledger" not in i.value for i in app.info)


def test_an_empty_watchlist_says_so(home, tmp_path):
    briefing = build(real_lab(tmp_path), ny("2026-10-19", "08:30"), data_class="real", tickers=[])
    write_document(home, briefing)
    app = page()
    assert words.EMPTY_WATCHLIST in read(app) and not app.expander


def _texts(app: AppTest) -> list[tuple[str, str]]:
    """The page's markdown and captions as (type, source), in the order shown."""

    return [(node.type, node.value) for node in app.main if node.type in ("markdown", "caption")]


@pytest.mark.parametrize("flag", [True, False])
def test_the_persistence_caveat_follows_the_comparison_only_when_flagged(home, compared, flag):
    briefing = compared[flag]
    prospective = briefing.reliability.prospective
    assert prospective.comparison.persistence_suspected is flag
    assert (prospective.comparison_caveat is not None) is flag
    write_document(home, briefing)
    app = page()
    text = read(app)  # the advice scan
    assert prospective.comparison_words in text
    shown = _texts(app)
    at = shown.index(("markdown", f"**{words.COMPARED}** {literal(prospective.comparison_words)}"))
    first_note = ("caption", literal((*briefing.reliability.notes, render.INTERVAL_NOTE)[0]))
    if flag:
        assert shown[at + 1 : at + 3] == [
            ("markdown", literal(prospective.comparison_caveat)),
            first_note,
        ]
        assert text.count(prospective.comparison_caveat) == 1  # as stored, never reworded
    else:
        assert shown[at + 1] == first_note
        assert "Caveat (reported only)" not in text


def test_text_from_a_file_is_shown_literally(home, tmp_path):
    briefing = build(real_lab(tmp_path), ny("2026-10-19", "08:30"), data_class="real",
                     tickers=[REAL_TEST_TICKER])  # fmt: skip
    image = "![x](http://example.invalid/a.png) **bold**"

    def inject(document):
        document["stocks"][0]["words"] = image
        document["limitations"].append(image)

    write_document(home, briefing, change=inject)
    app = page()
    sources = [m.value for m in app.markdown]
    assert not [s for s in sources if "![x](" in s or "**bold**" in s]
    assert any(r"\!\[x\]\(http://example\.invalid/a\.png\) \*\*bold\*\*" in s for s in sources)
    assert image in read(app)  # read back as written, escapes removed


def test_the_page_writes_nothing_and_runs_nothing(home, demo_files, tmp_path, monkeypatch):
    copy_demo(demo_files, home)
    lab = real_lab(tmp_path)
    write_real(lab, home, "2026-10-16T08:30:00-04:00")
    write_real(lab, home, "2026-10-19T08:30:00-04:00")
    monkeypatch.setenv(STORE_HOME_ENV, str(lab.path.parent))  # a real store beside the page
    before, store_before = snapshot(home), snapshot(lab.path.parent)
    app = page()
    app.selectbox[0].set_value("2026-10-16_real_premarket_briefing.json").run()
    app.radio[0].set_value("synthetic").run()
    assert not app.exception
    assert snapshot(home) == before and snapshot(lab.path.parent) == store_before
    (home / stored.LOG_NAME).unlink()
    page()
    assert not (home / stored.LOG_NAME).exists()


WRITING_NAMES = {
    "deliver", "deliver_once", "PreviewAdapter", "DeliveryLog", "private_directory", "claim",
    "finish", "build_briefing", "ReadOnlyRecords", "run_replay", "record_run", "record_forecast",
    "record_scores", "write", "write_text", "write_bytes", "mkdir", "unlink", "chmod", "rename",
    "replace", "subprocess", "system", "connect", "open",
}  # fmt: skip


def test_the_page_never_imports_the_writers_or_runs_a_job():
    for path in (RUN.parent / "briefing.py", RUN.parent / "destinations" / "briefing.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                names.add(node.id)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.ImportFrom):
                assert "delivery" not in (node.module or ""), path.name
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
        assert not names & WRITING_NAMES, (path.name, sorted(names & WRITING_NAMES))
