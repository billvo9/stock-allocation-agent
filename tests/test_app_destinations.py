"""
The Portfolio Lab's navigation shell and its destinations under AppTest:
the eight pages in order with Overview as the landing page, every destination
rendered through the running app (run.py) with the advice-wording scan over
all rendered text, and the Overview, Experiments, Model Health and Evidence
pages' states. The Explore page has its own tests (test_app_explore.py) and
the Builder too (test_app_builder.py).
"""

from __future__ import annotations

import html
import importlib.util
import json
import os
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from stock_agent.app import (
    evidence,
    health_summary,
    labels,
    navigation,
    synthetic,
    wording,
)
from stock_agent.health.context import CodeState
from stock_agent.health.model import Check, HealthState, Scope
from stock_agent.health.summary import build_summary, write_summary
from stock_agent.health.view import STATE_LEVELS, STATE_WORDS, build_view
from stock_agent.ui import theme

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "stock_agent" / "app"
RUN = PACKAGE / "run.py"
TIMEOUT = 120
ADVICE_WORDS = (" buy", " sell", "recommend", "should invest", "outperform")
# Wider than ADVICE_WORDS: word starts ("buy", "sell") or whole words.
ADVICE_PATTERNS = (
    r"\bbuy",
    r"\bsell",
    r"recommend",
    r"should invest",
    r"should hold",
    r"outperform",
    r"\bconsider\b",
    r"\ballocate\b",
    r"\bbest\b",
)
# The Explore page's approved disclaimer negates advice ("it does not say which
# fund anyone should hold"); it is removed, verbatim, before the scan.
DISCLAIMERS = (labels.CANNOT_TELL,)
KEYS = list(wording.NAV)


def scan_for_advice(text: str) -> None:
    lowered = text
    for disclaimer in DISCLAIMERS:
        lowered = lowered.replace(disclaimer, "")
    lowered = lowered.lower()
    for word in ADVICE_WORDS:
        assert word not in lowered, word
    for pattern in ADVICE_PATTERNS:
        assert not re.search(pattern, lowered), pattern


def widget_text(app: AppTest) -> list[str]:
    """Labels, options and help of every widget, expander labels and code blocks."""

    parts = []
    widgets = (
        *app.radio,
        *app.multiselect,
        *app.number_input,
        *app.select_slider,
        *app.selectbox,
        *app.button,
    )
    for widget in widgets:
        parts.append(str(getattr(widget, "label", "") or ""))
        parts.append(str(getattr(widget, "help", "") or ""))
        parts.extend(str(option) for option in (getattr(widget, "options", None) or ()))
    parts += [expander.label for expander in app.expander]
    parts += [code.proto.code_text for code in app.get("code")]
    return parts


def text_of(app: AppTest) -> str:
    """
    Every rendered text as read (markdown escapes and bold markers removed):
    tables, page links, widget labels, options and help, expander labels and
    code included. Every state test reads the page through here, so every
    rendered state is scanned for trading or advice wording.
    """

    parts = [
        e.value
        for e in (
            *app.title,
            *app.subheader,
            *app.markdown,
            *app.caption,
            *app.info,
            *app.warning,
            *app.error,
            *app.success,
        )
    ]
    parts += [html.unescape(h.proto.body) for h in app.get("html") if "<style>" not in h.proto.body]
    parts += [frame.value.to_csv(index=False) for frame in app.dataframe]
    parts += [table.value.to_csv() for table in app.table]
    parts += [f"{metric.label} {metric.value}" for metric in app.metric]
    parts += [link.proto.label for link in app.get("page_link")]
    parts += [button.proto.label for button in app.get("link_button")]
    parts += widget_text(app)
    text = " ".join(parts).replace("**", "")
    # Markdown escapes removed: "\\$" and the literal text of health summaries.
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|<>~$&:/=^%@?'\"])", r"\1", text)
    scan_for_advice(text)
    return text


def running(key: str = navigation.DEFAULT) -> AppTest:
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    if key != navigation.DEFAULT:
        app.switch_page(navigation.PAGES[key]).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


# --- the shell ---


def test_the_eight_destinations_are_registered_in_order_with_overview_first():
    assert [title for title, _ in wording.NAV.values()] == [
        "Overview",
        "Explore",
        "Portfolio Builder",
        "Stock Probability",
        "Premarket Briefing",
        "Experiments",
        "Model Health",
        "Evidence",
    ]
    for key, path in navigation.PAGES.items():
        assert (PACKAGE / path).is_file(), key
    source = RUN.read_text(encoding="utf-8")
    assert 'position="top"' in source and 'apply_theme("lab")' in source
    app = running()
    assert [t.value for t in app.title] == [wording.OVERVIEW_TITLE]  # the landing page


@pytest.mark.parametrize("key", KEYS)
def test_every_destination_renders_through_the_app_without_advice_wording(key):
    app = running(key)
    text = text_of(app)
    assert app.title, key
    assert wording.CANNOT_TELL in text or labels.CANNOT_TELL in text
    styles = [h for h in app.get("html") if "<style>" in h.proto.body]
    assert len(styles) == 1 and "--sa-serif" in styles[0].proto.body  # the Lab's layer, once


@pytest.mark.parametrize("key", ["overview", "explore", "builder", "evidence"])
def test_every_destination_showing_data_labels_it_synthetic(key):
    text = text_of(running(key))
    assert labels.SYNTHETIC_BADGE in text and labels.SYNTHETIC_NOTICE in text


def test_copy_has_no_trading_advice_or_decision_codes():
    shown = []
    for name, value in vars(wording).items():
        if name == "PENDING":  # the decision tags (last field) are never shown
            shown += [" ".join(item[:3]) for item in value]
        elif isinstance(value, str):
            shown.append(value)
        elif isinstance(value, tuple | dict):
            shown.append(repr(value))
    text = " ".join(shown)
    scan_for_advice(text)
    for word in ("optimal", "guarantee", "safe"):
        assert word not in text.lower(), word
    assert not re.search(r"\bOD\d", text, re.IGNORECASE)
    # The rendered Overview names no decision code either.
    assert not re.search(r"\bOD\d", text_of(running()), re.IGNORECASE)


# --- Overview ---


def test_the_overview_summarizes_the_data_and_what_needs_attention():
    app = running()
    text = text_of(app)
    assert "all 6 funds come from a seeded generator" in text
    assert "5 of 6 funds have data, together covering 2007-01-01 to 2024-12-31." in text
    # Needs attention: every item comes from the records, each status in words.
    stale = synthetic.demo_histories()[3]
    assert f"{stale.name}" in text and stale.stale_reason in text
    assert "Synthetic international equity fund" in text and "not set to zero" in text
    assert "From 2019-03-08, the hypothetical growth cannot be continued" in text
    assert "5 sessions without a close (first 2015-08-17 to 2015-08-21)" in text
    for word in ("Stale", "No data", "Break", "Gap"):
        assert f">{word}</span>" in text, word
    # Waiting on a decision: dated, each item named as pending.
    assert f"As recorded on {wording.PENDING_AS_OF}." in text
    for _, title, sentence, _ in wording.PENDING:
        assert title in text and sentence in text
    # The exact counts sit in the expander.
    counts = next(f.value for f in app.dataframe if "Sessions with a close" in f.value.columns)
    assert counts["Fund"].tolist() == [h.name for h in synthetic.demo_histories()]
    assert counts.loc[0, "Sessions without a close"] == "5"


def test_the_overview_offers_its_next_actions_as_links():
    links = running().get("page_link")
    assert [(link.proto.label, link.proto.page) for link in links] == [
        ("Explore fund history", "explore"),
        ("Build a mix", "builder"),
        ("See where the numbers come from", "evidence"),
    ]


def test_the_overview_without_links_still_renders():
    def page():
        from stock_agent.app import overview, synthetic

        overview.render(synthetic.demo_histories)

    app = AppTest.from_function(page, default_timeout=TIMEOUT).run()
    assert not app.exception and not app.get("page_link")


def test_the_overview_says_so_when_nothing_needs_attention():
    def page():
        from stock_agent.app import overview, synthetic

        clean = [
            h
            for h in synthetic.demo_histories()
            if h.fund_id in ("synthetic-bond", "synthetic-cash")
        ]
        overview.render(lambda: clean)

    app = AppTest.from_function(page, default_timeout=TIMEOUT).run()
    text = text_of(app)
    # Synthetic data always needs attention; nothing else does for these two funds.
    assert ">Synthetic</span>" in text
    for word in ("Stale", "No data", "Break", "Gap"):
        assert f">{word}</span>" not in text, word


def test_the_overview_with_only_clean_real_funds_needs_no_attention():
    def page():
        import dataclasses

        from stock_agent.app import overview, synthetic

        real = [
            dataclasses.replace(h, limitations=())
            for h in synthetic.demo_histories()
            if h.fund_id in ("synthetic-bond", "synthetic-cash")
        ]
        overview.render(lambda: real)

    app = AppTest.from_function(page, default_timeout=TIMEOUT).run()
    text = text_of(app)
    assert wording.NOTHING_NEEDS_ATTENTION in text
    assert labels.SYNTHETIC_BADGE not in text
    assert "The lab holds 2 funds, 0 of them synthetic." in text


# --- Experiments ---


def test_experiments_is_an_honest_empty_state_with_a_way_to_the_dashboard():
    app = running("experiments")
    text = text_of(app)
    assert wording.EXPERIMENTS_EMPTY_TITLE in text and wording.EXPERIMENTS_EMPTY in text
    assert not app.get("plotly_chart") and not app.dataframe and not app.metric
    assert [c.proto.code_text for c in app.get("code")] == [wording.DASHBOARD_COMMAND]
    assert [b.proto.url for b in app.get("link_button")] == [wording.DASHBOARD_URL]
    assert (REPO / "scripts" / "run_dashboard.py").is_file()
    assert "--port 8501" in (REPO / "scripts" / "run_dashboard.py").read_text(encoding="utf-8")


# --- Model Health ---

HEALTH_NOW = datetime(2026, 10, 15, 12, tzinfo=UTC)
STAGE_LINE = (
    "etf.ridge@v1 (etf:synthetic:h20, synthetic): engineering_only; development_evaluated "
    "blocked by: data_class_engineering_only (data_class='synthetic'); "
    "universe_not_point_in_time"
)
HEAD = "ab" * 32


@pytest.fixture(autouse=True)
def _no_health_summary_paths(monkeypatch):
    # The Lab reads summaries only from these variables; tests never see the owner's.
    monkeypatch.delenv(health_summary.LOCAL_VARIABLE, raising=False)
    monkeypatch.delenv(health_summary.CI_VARIABLE, raising=False)


def _health_check(check_id: str, state: HealthState, **fields) -> Check:
    measured = state is not HealthState.INSUFFICIENT_EVIDENCE
    defaults = {
        "title": f"Planted {check_id}",
        "measured_at": HEALTH_NOW if measured else None,
        "value": 1 if measured else None,
        "unit": "planted units" if measured else None,
        "threshold": "planted threshold",
        "summary": f"Planted sentence for {check_id} ({state.value}).",
        "detail": f"planted detail for {check_id}",
        "evidence": "fixture:lab",
        "reason": None if state is HealthState.HEALTHY else "planted_reason",
    }
    return Check(
        check_id=check_id, area=check_id.split(".")[0], state=state, **{**defaults, **fields}
    )


def _health_summary(checks, *, scope=Scope.PRIVATE, generated_at=HEALTH_NOW):
    return build_summary(
        checks,
        generated_at=generated_at,
        code=CodeState("c" * 40, False),
        scope=scope,
        run_context={"environment": "local"},
    )


def _write_summary(path: Path, checks, **fields) -> str:
    write_summary(path, _health_summary(checks, **fields))
    return str(path)


def _health_page(local, ci):
    from datetime import UTC, datetime
    from pathlib import Path

    from stock_agent.app import health_summary, model_health

    now = datetime(2026, 10, 15, 12, tzinfo=UTC)
    page = health_summary.HealthPage(
        local=health_summary.load_summary(None if local is None else Path(local), now=now),
        ci=health_summary.load_summary(None if ci is None else Path(ci), now=now),
    )
    model_health.render(lambda: page)


def _render_health(local: str | None, ci: str | None = None) -> AppTest:
    app = AppTest.from_function(_health_page, default_timeout=TIMEOUT, args=(local, ci)).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _pill_words(app: AppTest) -> list[str]:
    html = " ".join(h.proto.body for h in app.get("html"))
    return re.findall(r'class="sa-pill sa-pill--([a-z-]+)"[^>]*>([^<]+)</span>', html)


def test_model_health_is_not_yet_collected_and_shows_no_value():
    app = running("health")
    text = text_of(app)
    assert wording.HEALTH_EMPTY_TITLE in text and wording.HEALTH_EMPTY in text
    assert wording.HEALTH_NOT_SET in text
    assert wording.HEALTH_CREATE_COMMAND in [c.proto.code_text for c in app.get("code")]
    assert not app.get("plotly_chart") and not app.dataframe and not app.metric
    assert "valid" not in {level for level, _ in _pill_words(app)}  # nothing is green


def test_a_configured_summary_that_does_not_exist_is_not_yet_collected(tmp_path, monkeypatch):
    missing = tmp_path / "nowhere" / "latest.json"
    monkeypatch.setenv(health_summary.LOCAL_VARIABLE, str(missing))
    text = text_of(running("health"))
    assert wording.HEALTH_EMPTY_TITLE in text
    assert wording.HEALTH_LOOKED.format(path=missing) in text


# One summary per overall state: the planted state is the worst check's.
STATE_CASES = {
    HealthState.HEALTHY: [HealthState.HEALTHY, HealthState.HEALTHY],
    HealthState.INSUFFICIENT_EVIDENCE: [HealthState.HEALTHY, HealthState.INSUFFICIENT_EVIDENCE],
    HealthState.DEGRADED: [HealthState.INSUFFICIENT_EVIDENCE, HealthState.DEGRADED],
    HealthState.UNAVAILABLE: [HealthState.DEGRADED, HealthState.UNAVAILABLE],
    HealthState.FAILED: [HealthState.UNAVAILABLE, HealthState.FAILED],
}


@pytest.mark.parametrize("overall", list(STATE_CASES), ids=lambda state: state.value)
def test_model_health_shows_each_overall_state_in_words_with_its_next_step(tmp_path, overall):
    first, second = STATE_CASES[overall]
    checks = [_health_check("ci.verify", first), _health_check("forecast.ledger", second)]
    app = _render_health(_write_summary(tmp_path / "latest.json", checks))
    text = text_of(app)
    view = build_view(_health_summary(checks), now=HEALTH_NOW)
    assert view.state is overall
    pills = _pill_words(app)
    assert pills[0] == (theme.status_slug(STATE_LEVELS[overall]), STATE_WORDS[overall])
    assert view.headline in text
    assert view.next_action in text and (pills[1][1], wording.HEALTH_NEXT_TITLE) == (
        wording.HEALTH_NEXT_WORD,
        wording.HEALTH_NEXT_TITLE,
    )
    assert view.caveat in text  # informational only, always shown
    # Each area in words, then each check with its own sentence.
    assert (theme.status_slug(STATE_LEVELS[first]), STATE_WORDS[first]) in pills
    assert (theme.status_slug(STATE_LEVELS[second]), STATE_WORDS[second]) in pills
    for check in checks:
        assert check.summary in text and check.title in text
        assert check.detail in [c.proto.code_text for c in app.get("code")]
    assert "Code and tests" in text and "Prospective forecast ledger" in text
    labels = [expander.label for expander in app.expander]
    assert wording.HEALTH_MEASUREMENTS.format(title="Prospective forecast ledger") in labels


def test_the_five_states_render_with_five_distinct_words(tmp_path):
    checks = [
        _health_check(check_id, state)
        for check_id, state in zip(
            ("ci.verify", "ci.tests", "deps.pins", "data.anchors", "forecast.ledger"),
            HealthState,
            strict=True,
        )
    ]
    app = _render_health(_write_summary(tmp_path / "latest.json", checks))
    shown = {word for _, word in _pill_words(app)}
    assert {STATE_WORDS[state] for state in HealthState} <= shown
    assert len({STATE_WORDS[state] for state in HealthState}) == 5


def test_no_evidence_not_activated_stale_and_failed_render_with_distinct_words(tmp_path):
    checks = [
        _health_check("ci.verify", HealthState.HEALTHY),
        _health_check("data.freshness", HealthState.DEGRADED, reason="stale_data"),
        _health_check("forecast.ledger", HealthState.INSUFFICIENT_EVIDENCE, reason="input_missing"),
        _health_check(
            "forecast.stock_lab_job", HealthState.INSUFFICIENT_EVIDENCE, reason="not_authorized"
        ),
        _health_check("data.stock_lab_tickers", HealthState.FAILED, reason="planted_reason"),
    ]
    app = _render_health(_write_summary(tmp_path / "latest.json", checks))
    pills = _pill_words(app)
    words = {word for _, word in pills}
    assert {"Healthy", "Stale", "Insufficient evidence", "Not activated", "Failed"} <= words
    slug = {word: level for level, word in pills}
    assert slug["Stale"] == theme.status_slug("WARNING")
    assert slug["Not activated"] == theme.status_slug("NOTICE") != slug["Failed"]
    text = text_of(app)
    # The Stock Lab's checks have their own area, not "Other checks".
    assert "Stock Lab daily job (experimental)" in text and "Other checks" not in text
    assert "1 stale" in text and "1 not activated" in text and "1 insufficient evidence" in text


def test_a_stale_healthy_summary_shows_stale_never_healthy(tmp_path):
    from stock_agent.health.model import STALE_AFTER

    checks = [_health_check("ci.verify", HealthState.HEALTHY)]
    old = HEALTH_NOW - STALE_AFTER - timedelta(hours=1)
    app = _render_health(_write_summary(tmp_path / "latest.json", checks, generated_at=old))
    pills = _pill_words(app)
    assert pills[0] == (theme.status_slug("WARNING"), "Stale")
    text = text_of(app)
    assert "The latest summary is stale" in text and "This summary is 27 hours old" in text
    assert "Everything measured is healthy" not in text


def test_a_summary_with_nothing_measured_shows_nothing_green(tmp_path):
    checks = [
        _health_check(check_id, HealthState.INSUFFICIENT_EVIDENCE)
        for check_id in ("ci.verify", "data.anchors", "forecast.ledger", "portfolio.diagnostics")
    ]
    app = _render_health(_write_summary(tmp_path / "latest.json", checks))
    shown = {wording.HEALTH_EMPTY_BADGE, wording.HEALTH_NEXT_WORD}  # the page's own badges
    pills = [pill for pill in _pill_words(app) if pill[1] not in shown]
    assert pills and all(word == "Insufficient evidence" for _, word in pills)
    assert "valid" not in {level for level, _ in _pill_words(app)}
    assert "Healthy" not in text_of(app)


def test_a_summary_without_checks_is_insufficient_evidence_and_shows_nothing_green(tmp_path):
    app = _render_health(_write_summary(tmp_path / "latest.json", []))
    text = text_of(app)
    assert "This summary has no checks." in text
    assert "Run scripts/health_check.py to measure something" in text
    assert not app.expander  # no area, no check
    pills = _pill_words(app)
    assert pills[0] == (theme.status_slug("NOTICE"), "Insufficient evidence")
    assert "valid" not in {level for level, _ in pills}


def test_a_refused_summary_shows_why_and_nothing_from_it(tmp_path):
    path = tmp_path / "latest.json"
    _write_summary(path, [_health_check("ci.verify", HealthState.FAILED)])
    document = json.loads(path.read_text(encoding="utf-8"))
    document["overall_state"] = "healthy"  # an edited summary
    path.write_text(json.dumps(document), encoding="utf-8")
    app = _render_health(str(path))
    text = text_of(app)
    assert "was refused (state_mismatch)" in text and "Nothing from it is shown." in text
    assert "Planted sentence" not in text and not app.expander
    assert "valid" not in {level for level, _ in _pill_words(app)}


def test_every_models_stage_and_its_blockers_are_shown(tmp_path):
    stages = _health_check(
        "registry.stages",
        HealthState.HEALTHY,
        title="Computed stages",
        value=0,
        detail=f"by stage: engineering_only 1\nblockers to the next stage: none\n{STAGE_LINE}",
    )
    app = _render_health(_write_summary(tmp_path / "latest.json", [stages]))
    text = text_of(app)
    assert "Models, versions and evidence stage" in text
    assert STAGE_LINE in text
    assert "blocked by: data_class_engineering_only" in text


def test_ledger_facts_and_a_broken_chain_are_shown(tmp_path):
    ledger_ok = _health_check(
        "forecast.ledger",
        HealthState.FAILED,
        title="Forecast ledger integrity",
        value=None,
        reason="ledger_refused",
        summary="The forecast ledger failed its integrity check, so none of its records can be "
        "trusted.",
        detail="ledger.jsonl: Line 2 was edited (its hash does not match).",
    )
    backup = _health_check(
        "forecast.ledger_backup",
        HealthState.HEALTHY,
        title="Kept copy of the ledger head",
        value=0,
        detail=f"kept {HEAD}\nhead {HEAD}\n3 entries in the ledger",
    )
    app = _render_health(_write_summary(tmp_path / "latest.json", [ledger_ok, backup]))
    text = text_of(app)
    assert ledger_ok.summary in text and "Line 2 was edited" in text
    assert f"head {HEAD}" in text
    assert (theme.status_slug("INVALID/BLOCKED"), "Failed") in _pill_words(app)
    assert "Start with “Forecast ledger integrity”" in text


def test_the_daily_github_summary_is_shown_or_its_download_command(tmp_path):
    local = _write_summary(
        tmp_path / "latest.json", [_health_check("ci.verify", HealthState.HEALTHY)]
    )
    app = _render_health(local)
    text = text_of(app)
    assert wording.HEALTH_CI_TITLE in text and wording.HEALTH_CI_MISSING in text
    assert wording.HEALTH_CI_DOWNLOAD in [c.proto.code_text for c in app.get("code")]

    ci = _write_summary(
        tmp_path / "ci.json",
        [_health_check("ci.tests", HealthState.FAILED)],
        scope=Scope.SYNTHETIC,
        generated_at=HEALTH_NOW - timedelta(days=3),
    )
    text = text_of(_render_health(local, ci))
    assert "Generated 2026-10-12 12:00 UTC." in text
    assert "Something is broken" in text
    assert "This summary is 3 days old" in text


PLANTED = "See [a link](http://evil.example/a) or http://evil.example/b and <b>bold</b>."
MARKDOWN_ELEMENTS = ("markdown", "caption", "info", "warning", "error", "success", "subheader")


def _all_html(app: AppTest) -> str:
    return " ".join(h.proto.body for h in app.get("html") if "<style>" not in h.proto.body)


def test_producer_text_is_escaped_html_and_never_a_link(tmp_path):
    planted = _health_check(
        "ci.verify",
        HealthState.DEGRADED,  # degraded: it is quoted in the next step and an alert too
        title=f"Title {PLANTED}",
        summary=PLANTED,
        threshold=PLANTED,
        unit=None,
        value=PLANTED,
        evidence=PLANTED,
        detail=PLANTED,
    )
    app = _render_health(_write_summary(tmp_path / "latest.json", [planted]))
    # Markdown would autolink a URL even with its punctuation escaped, so producer text
    # never reaches a markdown element at all.
    for kind in MARKDOWN_ELEMENTS:
        for element in getattr(app, kind):
            assert "evil.example" not in element.value, (kind, element.value)
    rendered = _all_html(app)
    assert rendered.count("http://evil.example/b") >= 5  # summary, next step, alert, ...
    assert not re.search(r"<a\b[^>]*evil", rendered)  # no anchor anywhere points there
    assert "<b>bold</b>" not in rendered and "&lt;b&gt;bold&lt;/b&gt;" in rendered
    anchors = re.findall(r'<a\b[^>]*href="([^"]+)"', rendered)
    assert anchors == [view_runbook_url("ci.verify")]  # the runbook link only
    assert PLANTED in [code.proto.code_text for code in app.get("code")]  # detail, literal


def view_runbook_url(check_id: str) -> str:
    from stock_agent.health.model import runbook_for
    from stock_agent.health.view import RUNBOOK_BASE_URL

    return html.escape(RUNBOOK_BASE_URL + runbook_for(check_id), quote=True)


def test_a_refused_summary_path_and_reason_are_escaped_html(tmp_path):
    directory = tmp_path / "http:" / "evil.example"
    directory.mkdir(parents=True)
    path = directory / "latest.json"
    path.write_text("{not json", encoding="utf-8")
    app = _render_health(str(path))
    for kind in MARKDOWN_ELEMENTS:
        for element in getattr(app, kind):
            assert "evil.example" not in element.value, (kind, element.value)
    assert not re.search(r"<a\b[^>]*evil", _all_html(app))
    assert "was refused (invalid_json)" in text_of(app)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable files")
def test_an_unreadable_summary_is_shown_as_refused_not_a_crash(tmp_path):
    path = Path(
        _write_summary(tmp_path / "latest.json", [_health_check("ci.verify", HealthState.HEALTHY)])
    )
    path.chmod(0)
    try:
        app = _render_health(str(path))
    finally:
        path.chmod(0o600)
    text = text_of(app)
    assert "was refused (unreadable)" in text and "PermissionError" in text
    assert "valid" not in {level for level, _ in _pill_words(app)}


def test_degraded_and_unavailable_have_distinct_colours(tmp_path):
    levels = {state: STATE_LEVELS[state] for state in HealthState}
    assert len(set(levels.values())) == 5  # one treatment per state, the word beside it
    for level in levels.values():
        tones = theme.STATUS[level]
        assert theme.contrast_ratio(tones["fg"], tones["bg"]) >= 4.5, level
    checks = [
        _health_check("ci.verify", HealthState.DEGRADED),
        _health_check("ci.tests", HealthState.UNAVAILABLE),
    ]
    pills = _pill_words(_render_health(_write_summary(tmp_path / "latest.json", checks)))
    slug = {word: level for level, word in pills}
    assert slug["Degraded"] != slug["Unavailable"]
    assert slug["Unavailable"] == theme.status_slug("UNAVAILABLE")


def test_the_lab_page_reads_a_real_local_monitor_summary(tmp_path):
    from stock_agent.health import collect, fixtures

    root = tmp_path / "repo"
    root.mkdir()
    (root / "requirements.txt").write_text("pytest==9.1.1\n", encoding="utf-8")
    inputs = tmp_path / "health_inputs"
    for document in (fixtures.training_health(), fixtures.backtest_summary()):
        fixtures.write_input(inputs, document)

    def runner(argv, **kwargs):
        argv = [str(part) for part in argv]
        if argv[:2] == ["git", "check-ignore"]:
            return subprocess.CompletedProcess(argv, 0)
        if argv[0] == "git":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, "No broken requirements found.\n", "")

    environment = collect.Environment(
        root=root,
        env={},
        now=lambda: HEALTH_NOW,
        runner=runner,
        installed={"pytest": "9.1.1"}.get,
        python="/repo/.venv/bin/python",
        home=Path("/Users/someone"),
    )
    _, latest = collect.run_local(
        environment,
        tmp_path / "private",
        verify=None,
        inputs=collect.PrivateInputs(
            health_inputs=inputs,
            registry=fixtures.build_registry(tmp_path / "ml-fixture"),
        ),
    )
    app = _render_health(str(latest))
    text = text_of(app)
    for _, title, _ in health_summary.SECTIONS:
        assert title in text, title
    assert "Every registered version is synthetic" in text
    assert "blocked by: " in text  # each version's blockers, from the registry
    assert "No copy of the ledger head is configured" in text
    assert "risk, turnover and drawdown diagnostics" in text.lower()


def test_every_policys_limits_and_fallbacks_reach_the_page(tmp_path):
    from stock_agent.health import backtest, fixtures, policy

    for name, costs in (("equal_weight", 12.5), ("inverse_volatility", 12.5), ("momentum", 75.0)):
        document = fixtures.backtest_summary()
        document |= {
            "synthetic": False,
            "transition_status": "approved",
            "policy": name,
            "run_id": f"run-1-{name}",
            "constraint_violations": 0,
            "fallback_events": 0,
        }
        document["measures"]["costs_bps"] = costs
        fixtures.write_input(tmp_path / "inputs", document)
    directory = tmp_path / "inputs" / backtest.ARTIFACT
    checks = [*backtest.backtest_checks(directory), *policy.policy_checks(directory, None)]
    app = _render_health(_write_summary(tmp_path / "latest.json", checks))
    text = text_of(app)
    assert "momentum: Estimated costs (75) breaches its limit (50)." in text
    for name in ("equal_weight", "inverse_volatility", "momentum"):
        assert f"policy {name}" in text, name  # each policy's own line, in the details
    assert ("blocked", "Failed") in _pill_words(app)


def test_checks_are_grouped_into_the_labs_areas_and_nothing_is_dropped():
    assert health_summary.section_of("ci.daily_health") == "code"
    assert health_summary.section_of("forecast.coverage") == "runs"
    assert health_summary.section_of("forecast.ledger_backup") == "ledger"
    assert health_summary.section_of("registry.stages") == "models"
    assert health_summary.section_of("data.window_coverage") == "data"
    assert health_summary.section_of("portfolio.fallbacks") == "policy"
    assert health_summary.section_of("portfolio.diagnostics") == "diagnostics"
    assert health_summary.section_of("training.future_check") == "runs"
    checks = [
        _health_check("portfolio.turnover", HealthState.HEALTHY),
        _health_check("ci.tests", HealthState.DEGRADED),
        _health_check("ci.verify", HealthState.HEALTHY),
    ]
    grouped = health_summary.sections(_health_summary(checks))
    assert [item.key for item in grouped] == ["code", "policy"]
    assert [check.check_id for check in grouped[0].checks] == ["ci.tests", "ci.verify"]
    assert grouped[0].state is HealthState.DEGRADED
    assert grouped[0].summary == f"{checks[1].summary} 1 of 2 checks healthy."
    assert grouped[1].summary == "All 1 check healthy."
    assert sum(len(item.checks) for item in grouped) == len(checks)


def test_the_launcher_passes_the_health_summary_paths(tmp_path):
    path = REPO / "scripts" / "run_portfolio_lab.py"
    spec = importlib.util.spec_from_file_location("stock_agent_run_portfolio_lab_health", path)
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    assert launcher.LOCAL_SUMMARY_VARIABLE == health_summary.LOCAL_VARIABLE
    assert launcher.CI_SUMMARY_VARIABLE == health_summary.CI_VARIABLE
    env = launcher.environment(tmp_path / "latest.json", None)
    assert env[health_summary.LOCAL_VARIABLE] == str((tmp_path / "latest.json").resolve())
    assert health_summary.CI_VARIABLE not in env
    old, new = tmp_path / "1" / "health_summary.json", tmp_path / "2" / "health_summary.json"
    for index, item in enumerate((old, new)):
        item.parent.mkdir()
        item.write_text("{}", encoding="utf-8")
        os.utime(item, (1_000_000 + index, 1_000_000 + index))
    assert launcher.newest_ci_summary(tmp_path) == new


# --- Evidence ---


def test_evidence_explains_why_nothing_is_investment_ready():
    text = text_of(running("evidence"))
    assert wording.NOT_READY_TITLE in text
    for _, word, title, sentence in (wording.SYNTHETIC_REASON, *wording.NOT_READY):
        assert f">{word}</span>" in text and title in text and sentence in text


def test_evidence_lists_data_versions_calculations_and_provenance_from_metadata():
    app = running("evidence")
    text = text_of(app)
    sources = next(f.value for f in app.dataframe if "Vintage" in f.value.columns)
    assert sources["Fund"].tolist() == [h.name for h in synthetic.demo_histories()]
    assert sources["Vintage"].str.contains(str(synthetic.SEED)).all()
    for line in (*wording.CALCULATIONS, wording.BUILDER_STEP, wording.MODELS):
        assert line in text, line
    assert f"version {synthetic.GENERATOR_VERSION}, seed {synthetic.SEED}" in text
    assert "Data from 2025-01-01 on is reserved for one final test" in text
    assert "the demo data ends 2024-12-31" in text
    for assumption in (*labels.HONEST_LABELS, labels.FRACTIONAL_SHARES, labels.DAILY_CLOSES):
        assert assumption in text, assumption
    for limitation in labels.LIMITATION_TEXT.values():
        assert limitation in text


def test_evidence_reports_the_running_software_versions():
    versions = evidence.software_versions().set_index("Software")["Version"]
    assert versions["streamlit"] == streamlit.__version__
    assert versions["pandas"] == pd.__version__
    assert set(versions.index) == {"Python", *evidence.PACKAGES}


# --- choices kept across navigation ---


def _session_state_warnings(app: AppTest) -> list[str]:
    alerts = [e.value for e in (*app.warning, *app.error, *app.exception)]
    return [text for text in alerts if "Session State" in str(text)]


def test_explore_choices_are_kept_when_visiting_another_page():
    from stock_agent.app import explore

    bond, cash = synthetic.demo_histories()[1:3]
    app = running("explore")
    app = app.multiselect(key=explore.FUNDS_KEY).set_value([bond.fund_id, cash.fund_id]).run()
    app = app.radio(key=explore.PERIOD_KEY).set_value("custom").run()
    years = next(s for s in app.select_slider if s.key.startswith(explore.YEARS_KEY))
    app = years.set_range(2010, 2015).run()
    app.switch_page(navigation.PAGES["overview"]).run()
    app.switch_page(navigation.PAGES["explore"]).run()
    assert not app.exception and not _session_state_warnings(app)
    assert app.multiselect(key=explore.FUNDS_KEY).value == [bond.fund_id, cash.fund_id]
    assert app.radio(key=explore.PERIOD_KEY).value == "custom"
    assert "Shown: 2010-01-01 to 2015-12-31." in text_of(app)


# --- the advice scan over states other than the defaults ---


def _builder_entered(app):
    from stock_agent.app import builder

    app = app.radio(key=builder.RULE_KEY).set_value(builder.ENTERED).run()
    for fund, points in (("synthetic-equity", 60), ("synthetic-bond", 40), ("synthetic-cash", 0)):
        app = app.number_input(key=f"{builder.WEIGHT_PREFIX}{fund}").set_value(points).run()
    return app


def _builder_each_alternative(app):
    from stock_agent.app import builder

    for option in app.radio(key=builder.VARIANT_KEY).options:
        key = next(k for k, v in _variant_labels(app).items() if v == option)
        app = app.radio(key=builder.VARIANT_KEY).set_value(key).run()
        text_of(app)
    return app


def _variant_labels(app):
    from stock_agent.app import builder, mix

    weights = mix.equal_weights(app.multiselect(key=builder.FUNDS_KEY).value)
    held = [h for h in synthetic.demo_histories() if h.fund_id in weights]
    return {v.key: v.label for v in mix.variants(held, weights)}


def _builder_over_100(app):
    app = _builder_entered(app)
    return app.number_input(key="builder_weight_synthetic-equity").set_value(70).run()


def _builder_empty(app):
    from stock_agent.app import builder

    return app.multiselect(key=builder.FUNDS_KEY).set_value([]).run()


def _explore_custom_years(app):
    from stock_agent.app import explore

    app = app.radio(key=explore.PERIOD_KEY).set_value("custom").run()
    years = next(s for s in app.select_slider if s.key.startswith(explore.YEARS_KEY))
    return years.set_range(2008, 2009).run()


def _explore_gap_and_stale(app):
    from stock_agent.app import explore

    funds = ["synthetic-bond-distribution-gap", "synthetic-bond-stale"]
    return app.multiselect(key=explore.FUNDS_KEY).set_value(funds).run()


def _explore_empty(app):
    from stock_agent.app import explore

    return app.multiselect(key=explore.FUNDS_KEY).set_value([]).run()


STATES = {
    "builder-entered": ("builder", _builder_entered),
    "builder-alternatives": ("builder", _builder_each_alternative),
    "builder-over-100": ("builder", _builder_over_100),
    "builder-empty": ("builder", _builder_empty),
    "explore-custom-years": ("explore", _explore_custom_years),
    "explore-gap-and-stale": ("explore", _explore_gap_and_stale),
    "explore-empty": ("explore", _explore_empty),
}


@pytest.mark.parametrize("state", sorted(STATES))
def test_non_default_states_carry_no_advice_wording(state):
    key, drive = STATES[state]
    app = drive(running(key))
    assert not app.exception, [e.value for e in app.exception]
    assert text_of(app)


def test_the_scan_catches_advice_and_spares_the_approved_disclaimer():
    scan_for_advice(labels.CANNOT_TELL + " " + wording.CANNOT_TELL)
    for phrase in ("You should hold more bonds.", "Consider this mix.", "The best fund.", "buy"):
        with pytest.raises(AssertionError):
            scan_for_advice(phrase)


# --- pending items and the sealed date stay tied to the project's records ---

STATE_FILE = REPO / "docs" / "PROJECT_STATE.md"


def _pending_paragraph() -> str:
    text = STATE_FILE.read_text(encoding="utf-8")
    start = text.index("- Pending:")
    end = text.find("\n\n", start)
    return " ".join(text[start:end].split())


def _waits_for(decision: str, pending: str) -> bool:
    """The whole decision tag is listed: "D9" is not found inside "OD9's"."""

    pattern = rf"(?<![A-Za-z0-9]){re.escape(decision)}(?![A-Za-z0-9])"
    return re.search(pattern, pending) is not None


def test_every_pending_item_still_waits_for_a_pending_decision():
    import datetime

    datetime.date.fromisoformat(wording.PENDING_AS_OF)
    pending = _pending_paragraph()
    for _, title, _, decision in wording.PENDING:
        if decision is not None:
            assert _waits_for(decision, pending), (
                f"'{title}' waits for {decision}, which docs/PROJECT_STATE.md no longer lists "
                "as pending: update wording.PENDING and PENDING_AS_OF."
            )


def test_a_decision_taken_off_the_pending_list_is_caught():
    pending = _pending_paragraph()
    for _, title, _, decision in wording.PENDING:
        if decision is None:
            continue
        # The bullet that names the decision, removed as it would be once decided.
        bullets = [b for b in pending.split(" - ") if _waits_for(decision, b)]
        assert bullets, (title, decision)
        without = pending
        for bullet in bullets:
            without = without.replace(bullet, "")
        assert not _waits_for(decision, without), (title, decision)
    assert not _waits_for("D9", "OD9's signal_cutoff_at for decisions")
    assert _waits_for("D9", "- D9 (the Builder's buy-and-hold mix)")


def test_the_sealed_date_is_the_owners_lockbox_and_no_demo_data_reaches_it():
    from stock_agent.app import status
    from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

    assert status.SEALED_FROM == MODEL_HOLDOUT_START
    for history in synthetic.demo_histories():
        assert (history.records["date"] < status.SEALED_FROM).all(), history.fund_id
    assert pd.Timestamp(synthetic.CALENDAR_END, tz="UTC") < status.SEALED_FROM
