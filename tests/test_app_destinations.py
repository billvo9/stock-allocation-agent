"""
The Portfolio Lab's navigation shell and its destinations under AppTest:
the six pages in order with Overview as the landing page, every destination
rendered through the running app (run.py) with the advice-wording scan over
all rendered text, and the Overview, Experiments, Model Health and Evidence
pages' states. The Explore page has its own tests (test_app_explore.py) and
the Builder too (test_app_builder.py).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from stock_agent.app import (
    evidence,
    labels,
    navigation,
    synthetic,
    wording,
)
from stock_agent.app.model_health import HealthCheck, HealthReport

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "stock_agent" / "app"
RUN = PACKAGE / "run.py"
TIMEOUT = 120
ADVICE_WORDS = (" buy", " sell", "recommend", "should invest", "outperform")
KEYS = list(wording.NAV)


def text_of(app: AppTest) -> str:
    """
    Every rendered text as read (markdown escapes and bold markers removed),
    tables and page links included. Every state test reads the page through
    here, so every rendered state is scanned for trading or advice wording.
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
        )
    ]
    parts += [h.proto.body for h in app.get("html") if "<style>" not in h.proto.body]
    parts += [frame.value.to_csv(index=False) for frame in app.dataframe]
    parts += [table.value.to_csv() for table in app.table]
    parts += [f"{metric.label} {metric.value}" for metric in app.metric]
    parts += [link.proto.label for link in app.get("page_link")]
    parts += [button.proto.label for button in app.get("link_button")]
    text = " ".join(parts).replace("\\$", "$").replace("**", "")
    lowered = text.lower()
    for word in ADVICE_WORDS:
        assert word not in lowered, word
    return text


def running(key: str = navigation.DEFAULT) -> AppTest:
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    if key != navigation.DEFAULT:
        app.switch_page(navigation.PAGES[key]).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


# --- the shell ---


def test_the_six_destinations_are_registered_in_order_with_overview_first():
    assert [title for title, _ in wording.NAV.values()] == [
        "Overview",
        "Explore",
        "Portfolio Builder",
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
    for value in vars(wording).values():
        if isinstance(value, str):
            shown.append(value)
        elif isinstance(value, tuple | dict):
            shown.append(repr(value))
    text = " ".join(shown).lower()
    for word in (*ADVICE_WORDS, "best", "optimal", "guarantee", "safe"):
        assert word not in text, word
    assert "od8" not in text and "od5" not in text


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
    for _, title, sentence in wording.PENDING:
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


def test_model_health_is_not_yet_collected_and_shows_no_value():
    app = running("health")
    text = text_of(app)
    assert wording.HEALTH_EMPTY_TITLE in text and wording.HEALTH_EMPTY in text
    assert not app.get("plotly_chart") and not app.dataframe and not app.metric


REPORT = HealthReport(
    collected_at="2026-10-14 18:00 UTC",
    source="a test collector",
    checks=(
        HealthCheck("Data freshness", "VALID", "Fresh", "Prices end at the expected session."),
        HealthCheck(
            "Feature drift",
            "WARNING",
            "Drift",
            "Two features moved beyond their reference range.",
            detail="PSI 0.31 and 0.27 against the training window",
            measured_at="2026-10-14",
        ),
    ),
)


def _health_page(report):
    from stock_agent.app import model_health

    model_health.render(lambda: report)


def test_model_health_lays_out_a_supplied_report_with_statuses_in_words():
    app = AppTest.from_function(_health_page, default_timeout=TIMEOUT, args=(REPORT,)).run()
    assert not app.exception
    text = text_of(app)
    assert "Collected 2026-10-14 18:00 UTC from a test collector." in text
    for check in REPORT.checks:
        assert check.name in text and check.summary in text and f">{check.word}</span>" in text
    details = app.dataframe[0].value
    assert details["Detail"].tolist() == ["—", "PSI 0.31 and 0.27 against the training window"]
    assert wording.HEALTH_EMPTY_TITLE not in text


def test_an_empty_report_is_still_not_yet_collected():
    empty = HealthReport("2026-10-14", "a test collector", ())
    app = AppTest.from_function(_health_page, default_timeout=TIMEOUT, args=(empty,)).run()
    assert wording.HEALTH_EMPTY_TITLE in text_of(app)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"level": "GREEN"}, ValueError),
        ({"word": ""}, ValueError),
        ({"summary": ""}, ValueError),
    ],
)
def test_a_health_check_with_an_unknown_level_or_missing_words_is_refused(change, error):
    fields = {"name": "Check", "level": "VALID", "word": "Fresh", "summary": "Fine.", **change}
    with pytest.raises(error):
        HealthCheck(**fields)
    with pytest.raises(ValueError):
        HealthReport("", "source", ())
    with pytest.raises(TypeError):
        HealthReport("now", "source", [])  # a list is not the contract


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
    assert "Data after 2024-12-31 is never shown" in text
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
