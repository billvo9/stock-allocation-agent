"""
The Portfolio Builder under AppTest, one test per first-class state: the
running app (equal weights over the first three funds), concentration,
entered weights and their constraint feedback, a fund at 0% left out, no
fund chosen, one fund, funds without data, the what-if weight comparison,
choices kept across navigation, and the mix history shown only as awaiting
approval: no chart, no value and no performance figure anywhere. Every
state is read through `text_of`, which scans for trading or advice wording.
"""

from __future__ import annotations

import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from stock_agent.app import builder, labels, mix, navigation, synthetic, wording

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "src" / "stock_agent" / "app" / "run.py"
TIMEOUT = 120
ADVICE_WORDS = (" buy", " sell", "recommend", "should invest", "outperform")
EQUITY, BOND, CASH, STALE, DISTRIBUTION_GAP, MISSING = (
    h.fund_id for h in synthetic.demo_histories()
)


def text_of(app: AppTest) -> str:
    """Every rendered text as read, tables included; scanned for advice wording."""

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
    text = " ".join(parts).replace("\\$", "$").replace("**", "")
    lowered = text.lower()
    for word in ADVICE_WORDS:
        assert word not in lowered, word
    return text


def assert_no_mix_history(app: AppTest, text: str) -> None:
    """The mix's history awaits approval: no chart, no dollar value, no decline or return."""

    assert not app.get("plotly_chart")
    assert "$" not in text
    for phrase in ("drawdown", "decline", "lost money", "was worth", "growth of"):
        assert phrase not in text.lower(), phrase
    assert wording.HISTORY_PENDING in text and wording.NOT_MODELLED in text


def running() -> AppTest:
    app = AppTest.from_file(str(RUN), default_timeout=TIMEOUT).run()
    app.switch_page(navigation.PAGES["builder"]).run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def choose(app: AppTest, *fund_ids: str) -> AppTest:
    return app.multiselect(key=builder.FUNDS_KEY).set_value(list(fund_ids)).run()


def enter(app: AppTest, **points: int) -> AppTest:
    """Switch to entered weights and set them, by fund id with dashes as underscores."""

    if app.radio(key=builder.RULE_KEY).value != builder.ENTERED:
        app = app.radio(key=builder.RULE_KEY).set_value(builder.ENTERED).run()
    for fund, value in points.items():
        key = f"{builder.WEIGHT_PREFIX}{fund.replace('_', '-')}"
        app = app.number_input(key=key).set_value(value).run()
    return app


# --- the running app ---


def test_the_running_builder_shows_equal_weights_over_three_funds_and_their_spread():
    app = running()
    text = text_of(app)
    assert app.multiselect(key=builder.FUNDS_KEY).value == [EQUITY, BOND, CASH]
    assert "Each fund: 33.3%." in text
    assert (
        "Weights: Synthetic equity fund 33.3% · Synthetic bond fund 33.3% · Synthetic cash "
        "fund 33.3%." in text
    )
    values = {metric.label: metric.value for metric in app.metric}
    assert values == {
        wording.FUNDS_HELD_LABEL: "3",
        wording.LARGEST_LABEL: "33.3%",
        wording.ROLES_LABEL: "3",
    }
    assert "Equity 33.3%; Bonds 33.3%; Cash proxy 33.3%" in text
    weights = next(t.value for t in app.table if "Weight" in t.value.columns)
    assert weights["Weight"].tolist() == ["33.3%"] * 3
    for label in (
        wording.ILLUSTRATIVE_BADGE,
        wording.NO_HISTORY_BADGE,
        labels.SYNTHETIC_BADGE,
        labels.SYNTHETIC_NOTICE,
        wording.CONCENTRATION_SCOPE,
        wording.CANNOT_TELL,
    ):
        assert label in text, label
    assert "Not listed, because they have no available data: Synthetic international" in text
    assert_no_mix_history(app, text)


def test_the_pending_mix_history_points_to_each_funds_own_history():
    links = running().get("page_link")
    assert [(link.proto.label, link.proto.page) for link in links] == [
        (wording.EXPLORE_LINK, "explore")
    ]


# --- entered weights ---


def test_entered_weights_start_at_even_steps_and_must_add_to_100():
    app = enter(running())
    inputs = {n.key: n.value for n in app.number_input}
    assert inputs == {
        f"{builder.WEIGHT_PREFIX}{EQUITY}": 35,
        f"{builder.WEIGHT_PREFIX}{BOND}": 35,
        f"{builder.WEIGHT_PREFIX}{CASH}": 30,
    }
    over = enter(app, synthetic_equity=40)
    text = text_of(over)
    assert "The weights add to 105%. Remove 5 points so they add to 100%." in text
    assert not over.metric and wording.HISTORY_PENDING in text
    short = enter(over, synthetic_equity=30)
    assert "The weights add to 95%. Add 5 points so they reach 100%." in text_of(short)
    assert not short.metric


def test_a_concentrated_mix_says_so_and_a_fund_at_zero_is_left_out():
    app = enter(running(), synthetic_equity=60, synthetic_bond=40, synthetic_cash=0)
    text = text_of(app)
    assert "Weights: Synthetic equity fund 60.0% · Synthetic bond fund 40.0%." in text
    assert "More than half of the mix sits in one fund: Synthetic equity fund, 60.0%." in text
    assert {m.label: m.value for m in app.metric}[wording.FUNDS_HELD_LABEL] == "2"
    assert_no_mix_history(app, text)


# --- states ---


def test_no_fund_chosen_shows_the_empty_state():
    app = choose(running())
    text = text_of(app)
    assert wording.EMPTY in text
    assert not app.metric
    assert_no_mix_history(app, text)


def test_one_fund_has_no_alternative_and_sits_whole_in_one_fund():
    app = choose(running(), BOND)
    text = text_of(app)
    assert wording.SINGLE_FUND.format(fund="Synthetic bond fund") in text
    assert wording.NO_VARIANT in text
    assert [r.key for r in app.radio] == [builder.RULE_KEY]


def test_a_stale_fund_is_named_stale_and_a_fund_without_data_is_not_listed():
    widget = running().multiselect(key=builder.FUNDS_KEY)
    assert "Synthetic bond fund, stale copy · stale" in widget.options
    assert not [option for option in widget.options if "international" in option]


def _page(histories):
    from stock_agent.app import builder
    from stock_agent.ui.components import apply_theme

    apply_theme("lab")
    builder.render(lambda: histories)


def test_without_navigation_the_page_draws_no_link():
    app = AppTest.from_function(
        _page, default_timeout=TIMEOUT, kwargs={"histories": synthetic.demo_histories()}
    ).run()
    assert not app.exception and not app.get("page_link")
    assert wording.HISTORY_PENDING in text_of(app)


def test_real_funds_are_not_labelled_synthetic():
    import dataclasses

    real = tuple(dataclasses.replace(h, limitations=()) for h in synthetic.demo_histories()[:3])
    app = AppTest.from_function(_page, default_timeout=TIMEOUT, kwargs={"histories": real}).run()
    text = text_of(app)
    assert labels.SYNTHETIC_BADGE not in text and labels.SYNTHETIC_NOTICE not in text


# --- what-if ---


def test_the_what_if_compares_weights_and_their_spread_only():
    app = running()
    options = app.radio(key=builder.VARIANT_KEY).options
    assert options == [
        "Ten points moved from equity funds to bond and cash funds",
        "Ten points moved from bond and cash funds to equity funds",
    ]
    table = next(t.value for t in app.table if wording.MIX_NAME in t.value.columns)
    assert list(table.columns) == [wording.MIX_NAME, options[0]]
    assert list(table.index) == [
        "Weights",
        "Funds held",
        "Largest weight",
        "Equity",
        "Bonds",
        "Cash proxy",
    ]
    assert table.loc["Equity", options[0]] == "23.3%"
    assert table.loc["Bonds", options[0]] == "38.3%"
    app = app.radio(key=builder.VARIANT_KEY).set_value(mix.TOWARD_EQUITY).run()
    table = next(t.value for t in app.table if wording.MIX_NAME in t.value.columns)
    assert table.loc["Equity", options[1]] == "43.3%"
    assert_no_mix_history(app, text_of(app))


def test_equal_weights_are_offered_as_an_alternative_to_entered_weights():
    app = enter(running(), synthetic_equity=60, synthetic_bond=30, synthetic_cash=10)
    assert "Equal weights across the same funds" in app.radio(key=builder.VARIANT_KEY).options


# --- navigation, structure ---


def test_choices_are_kept_when_visiting_another_page():
    app = enter(choose(running(), EQUITY, BOND), synthetic_equity=60, synthetic_bond=40)
    app = app.radio(key=builder.VARIANT_KEY).set_value(mix.EQUAL).run()
    app.switch_page(navigation.PAGES["overview"]).run()
    app.switch_page(navigation.PAGES["builder"]).run()
    assert not app.exception
    assert app.multiselect(key=builder.FUNDS_KEY).value == [EQUITY, BOND]
    assert app.radio(key=builder.RULE_KEY).value == builder.ENTERED
    assert app.radio(key=builder.VARIANT_KEY).value == mix.EQUAL
    assert {n.key: n.value for n in app.number_input} == {
        f"{builder.WEIGHT_PREFIX}{EQUITY}": 60,
        f"{builder.WEIGHT_PREFIX}{BOND}": 40,
    }


def test_the_page_builds_from_the_shared_components_and_calls_no_calculation():
    source = (REPO / "src" / "stock_agent" / "app" / "builder.py").read_text(encoding="utf-8")
    for banned in ("st.title(", "unsafe_allow_html", "st.html(", "st.error(", "st.warning("):
        assert banned not in source, banned
    assert not re.search(r"#[0-9A-Fa-f]{6}\b", source)
    for banned in ("stock_agent.portfolio", "DescriptiveCalculator", "plotly", "figures"):
        assert banned not in source, banned
    page = (REPO / "src" / "stock_agent" / "app" / "destinations" / "builder.py").read_text()
    assert "DescriptiveCalculator" not in page
