"""
The Portfolio Builder: user-selected synthetic funds and weights, how those
weights are spread (concentration by fund and by role), and discrete what-if
alternatives for the same funds.

Layout only. Weights, concentration and alternatives come from `mix`, every
caveat from `labels` and `wording`, and every surface from the shared
components. Nothing here describes performance: how a fixed mix would have
grown is a reference-portfolio formula awaiting approval, and transaction
costs, rebalancing, execution and forecasts are not modelled. The page says
both plainly and shows them as pending, never as a value.

First-class states, each in words: no fund chosen, entered weights that do
not add to 100, a fund at 0% left out, one fund (no alternative), and funds
without data (not listed, and named). Choices survive a visit to another
page in the same session (`persist`).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pandas as pd
import streamlit as st

from stock_agent.app import labels, mix, persist, wording
from stock_agent.app.history import FundHistory
from stock_agent.app.prose import say, section
from stock_agent.app.view import percent
from stock_agent.ui import components as ui

FUNDS_KEY = "builder_funds"
RULE_KEY = "builder_rule"
WEIGHT_PREFIX = "builder_weight_"
VARIANT_KEY = "builder_variant"
SAVED_KEY = "builder_saved"
_SAVED_PREFIXES = (FUNDS_KEY, RULE_KEY, WEIGHT_PREFIX, VARIANT_KEY)
EQUAL, ENTERED = tuple(wording.RULES)
DEFAULT_FUNDS = 3
INPUTS_PER_ROW = 3


# --- inputs ---


def _pick_funds(usable: Sequence[FundHistory], where) -> tuple[FundHistory, ...]:
    by_id = {history.fund_id: history for history in usable}
    if FUNDS_KEY not in st.session_state:
        fresh = [h.fund_id for h in usable if not h.stale]
        st.session_state[FUNDS_KEY] = fresh[:DEFAULT_FUNDS]

    def option(fund_id: str) -> str:
        return by_id[fund_id].name + (" · stale" if by_id[fund_id].stale else "")

    chosen = where.multiselect(
        wording.FUNDS_LABEL,
        list(by_id),
        format_func=option,
        key=FUNDS_KEY,
        help=wording.FUNDS_HELP,
    )
    return tuple(history for history in usable if history.fund_id in chosen)


def _pick_weights(
    selected: Sequence[FundHistory], rule: str
) -> tuple[dict[str, float] | None, str | None]:
    """The weights as fractions, or None and the constraint to fix, in words."""

    ids = [history.fund_id for history in selected]
    if rule == EQUAL:
        say(f"Each fund: {percent(1 / len(ids))}.", "caption")
        return mix.equal_weights(ids), None
    keys = {fund_id: f"{WEIGHT_PREFIX}{fund_id}" for fund_id in ids}
    if not any(key in st.session_state for key in keys.values()):
        for fund_id, points in mix.starting_points(ids).items():
            st.session_state[keys[fund_id]] = points
    for key in keys.values():
        st.session_state.setdefault(key, 0)
    points = {}
    for first in range(0, len(selected), INPUTS_PER_ROW):
        row = selected[first : first + INPUTS_PER_ROW]
        for column, history in zip(st.columns(INPUTS_PER_ROW), row, strict=False):
            points[history.fund_id] = int(
                column.number_input(
                    f"{history.name} (%)",
                    min_value=0,
                    max_value=mix.TOTAL_POINTS,
                    step=mix.WEIGHT_STEP,
                    key=keys[history.fund_id],
                    help=wording.WEIGHT_HELP,
                )
            )
    gap = mix.points_gap(points)
    total = mix.TOTAL_POINTS - gap
    if gap > 0:
        return None, wording.POINTS_SHORT.format(total=total, gap=gap)
    if gap < 0:
        return None, wording.POINTS_OVER.format(total=total, gap=-gap)
    return mix.from_points(points), None


# --- the page ---


def render(
    load_histories: Callable[[], Sequence[FundHistory]],
    *,
    links: Mapping[str, object] | None = None,
) -> None:
    """`links` maps "explore" to its page for the next action; without it no link is drawn."""

    persist.restore(SAVED_KEY)
    ui.hero(
        wording.BUILDER_TITLE,
        eyebrow=wording.BUILDER_EYEBROW,
        lede=wording.BUILDER_LEDE,
        key="builder",
    )
    histories = tuple(load_histories())
    usable = tuple(history for history in histories if history.coverage_start is not None)
    without = [history.name for history in histories if history.coverage_start is None]
    badges = [("NOTICE", wording.ILLUSTRATIVE_BADGE), ("FUTURE", wording.NO_HISTORY_BADGE)]
    if any(history.synthetic for history in histories):
        badges.append(("WARNING", labels.SYNTHETIC_BADGE))
    ui.pills(badges)
    if any(history.synthetic for history in histories):
        say(labels.SYNTHETIC_NOTICE, "warning")

    with ui.card("builder-inputs"):
        section(wording.INPUTS_TITLE, wording.INPUTS_SCOPE)
        funds_column, rule_column = st.columns([3, 2], gap="large")
        selected = _pick_funds(usable, funds_column)
        if without:
            say(wording.NO_DATA_FUNDS.format(names=", ".join(without)), "caption", funds_column)
        rule = rule_column.radio(
            wording.RULE_LABEL,
            list(wording.RULES),
            format_func=wording.RULES.get,
            key=RULE_KEY,
            horizontal=True,
            help=wording.RULE_HELP,
        )
        weights, problem = _pick_weights(selected, rule) if selected else (None, None)

    if not selected:
        say(wording.EMPTY, "notice")
    elif problem:
        say(problem, "notice")
    else:
        _concentration(selected, weights)
        _what_if(selected, weights)
    persist.save(SAVED_KEY, _SAVED_PREFIXES)
    _pending(links)
    section(wording.CANNOT_TELL_TITLE)
    say(wording.CANNOT_TELL)


def _concentration(selected: Sequence[FundHistory], weights: Mapping[str, float]) -> None:
    spread = mix.concentration(selected, weights)
    held = [history for history in selected if weights.get(history.fund_id, 0) > 0]
    with ui.card("builder-concentration"):
        section(wording.CONCENTRATION_TITLE, wording.CONCENTRATION_SCOPE)
        say(wording.WEIGHTS_SUMMARY.format(weights=mix.weights_text(held, weights)))
        if spread.holdings == 1:
            say(wording.SINGLE_FUND.format(fund=spread.largest_fund))
        elif spread.largest_weight > 0.5:
            say(
                wording.CONCENTRATED.format(
                    fund=spread.largest_fund, weight=percent(spread.largest_weight)
                )
            )
        roles = "; ".join(
            f"{labels.ROLE_TEXT[role]} {percent(weight)}" for role, weight in spread.by_role.items()
        )
        ui.stat_row(
            [
                ui.Stat(wording.FUNDS_HELD_LABEL, str(spread.holdings)),
                ui.Stat(
                    wording.LARGEST_LABEL,
                    percent(spread.largest_weight),
                    caption=spread.largest_fund,
                ),
                ui.Stat(wording.ROLES_LABEL, str(len(spread.by_role)), caption=roles),
            ]
        )
        rows = [
            {
                "Fund": history.name,
                "Role": labels.ROLE_TEXT[history.role],
                "Weight": percent(weights[history.fund_id]),
            }
            for history in held
        ]
        st.table(pd.DataFrame(rows).set_index("Fund"))


def comparison_table(rows: Sequence[dict[str, str]]) -> pd.DataFrame:
    """Rows turned so each mix is a column and each measure a row (narrow screens)."""

    return pd.DataFrame(rows).set_index("Mix").T.rename_axis("Measure")


def _what_if(selected: Sequence[FundHistory], weights: Mapping[str, float]) -> None:
    section(wording.WHAT_IF_TITLE, wording.WHAT_IF_SCOPE)
    held = [history for history in selected if weights.get(history.fund_id, 0) > 0]
    options = {variant.key: variant for variant in mix.variants(held, weights)}
    if not options:
        say(wording.NO_VARIANT, "caption")
        return
    if st.session_state.get(VARIANT_KEY) not in options:
        st.session_state[VARIANT_KEY] = next(iter(options))
    choice = st.radio(
        wording.WHAT_IF_LABEL,
        list(options),
        format_func=lambda key: options[key].label,
        key=VARIANT_KEY,
    )
    variant = options[choice]
    rows = [
        mix.concentration_row(wording.MIX_NAME, held, weights),
        mix.concentration_row(variant.label, held, variant.weights),
    ]
    st.table(comparison_table(rows))


def _pending(links: Mapping[str, object] | None) -> None:
    with ui.state_card("builder-history-pending"):
        ui.pill("FUTURE", wording.NOT_MODELLED_BADGE)
        say(f"**{wording.HISTORY_PENDING_TITLE}**")
        say(wording.HISTORY_PENDING)
        say(wording.NOT_MODELLED, "caption")
        if links and "explore" in links:
            ui.page_links([(links["explore"], wording.EXPLORE_LINK)], key="builder")
