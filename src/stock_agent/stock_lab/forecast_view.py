"""
Pure view logic for the Stock Probability page's modelled views: the next
five sessions, reversals, bull traps and the model diagnostics. No
Streamlit, no Plotly; numpy, pandas and the Stock Lab's descriptive modules
only (the fitted models and the evaluation code are never imported here:
their results arrive as stored records).

Uncertainty shown beside a forecast is the evaluation's own track record:
for the estimator in use and the same target, the share of development-test
forecasts in the same probability range whose category happened, with its
Jeffreys interval and count ("forecasts like this came true 29-37% of the
time, 1,234 cases"). It describes calibration in past tests; it is not a
confidence interval for the stock's true probability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_agent.stock_lab.contracts import CATEGORIES, CATEGORY_CODES, CATEGORY_WORDS, StableBand
from stock_agent.stock_lab.reversal import (
    INSUFFICIENT,
    LOW,
    after_move,
    after_streak,
    next_session,
    reversal_durations,
    streak_events,
)
from stock_agent.stock_lab.runs import RunRule
from stock_agent.stock_lab.traps import (
    CONFIRMED,
    HELD,
    UNRESOLVED,
    TrapRule,
    evaluate_traps,
    trap_events,
)

DASH = "—"
MINUS = "−"
KEY_OF = {
    "decrease": "p_decrease",
    "stable": "p_stable",
    "increase": "p_increase",
}
MODEL_WORDS = {
    "B0": "Long-run frequencies",
    "B1": "Recent-weighted frequencies",
    "B2": "Streak-aware frequencies",
    "B3": "Volatility-scaled baseline",
    "M1": "Regularized logistic model",
}
STATUS_WORDS = {INSUFFICIENT: "Insufficient observations", LOW: "Low sample", "ok": ""}
TRAP_STATUS_WORDS = {
    UNRESOLVED: "Unresolved: its window has not closed",
    CONFIRMED: "Confirmed: the breakout failed",
    HELD: "Held: the breakout did not fail",
}


def pct(value: float | None, digits: int = 0) -> str:
    if value is None or not np.isfinite(value):
        return DASH
    return f"{value:.{digits}%}".replace("-", MINUS)


def model_words(model: str) -> str:
    return f"{MODEL_WORDS.get(model, model)} ({model})"


# --- the next five sessions ---


@dataclass(frozen=True)
class TrackRecord:
    """Development-test forecasts in the same range: how often the category happened."""

    observed: float
    low: float
    high: float
    n: int
    dates: int
    bin_low: float
    bin_high: float


def track_record(
    evaluation: Mapping | None, model: str, target: str, category: str, probability: float
) -> TrackRecord | None:
    if evaluation is None:
        return None
    code = CATEGORY_CODES[next(c for c in CATEGORIES if c.value == category)]
    for row in evaluation.get("reliability", ()):
        if row["model"] != model or row["target"] != target or row["category"] != code:
            continue
        last_bin = row["bin_high"] >= 1.0
        if row["bin_low"] <= probability < row["bin_high"] or (last_bin and probability == 1.0):
            return TrackRecord(
                row["observed"],
                row["ci_low"],
                row["ci_high"],
                row["n"],
                row.get("dates", row["n"]),
                row["bin_low"],
                row["bin_high"],
            )
    return None


@dataclass(frozen=True)
class ProbabilityCard:
    label: str
    value: str
    caption: str
    status: str


def next_session_cards(record: Mapping, evaluation: Mapping | None) -> tuple[ProbabilityCard, ...]:
    """Three cards for the first session ahead: Increase, Stable, Decrease."""

    first = record["daily"][0]
    usual = record["signals"].get("usual_mix") or {}
    cards = []
    for category in CATEGORIES:
        key = KEY_OF[category.value]
        p = first[key]
        track = track_record(evaluation, record["estimator"], "target_1", category.value, p)
        parts = []
        if track is not None:
            parts.append(f"tests: {pct(track.low)}–{pct(track.high)} came true")
        if key in usual:
            parts.append(f"usual {pct(usual[key])}")
        status = ""
        cards.append(
            ProbabilityCard(CATEGORY_WORDS[category], pct(p), " · ".join(parts) or DASH, status)
        )
    return tuple(cards)


def plain_sentence(record: Mapping) -> str:
    """One sentence for the first session ahead, in "3 in 10" form."""

    first = record["daily"][0]
    tenths = {k: round(first[KEY_OF[k]] * 10) for k in ("increase", "stable", "decrease")}
    usual = record["signals"].get("usual_mix") or {}
    usual_text = ""
    if usual:
        usual_tenths = [round(usual[KEY_OF[k]] * 10) for k in ("increase", "stable", "decrease")]
        usual_text = (
            f" This stock's usual mix is about {usual_tenths[0]} / {usual_tenths[1]} / "
            f"{usual_tenths[2]}."
        )
    band = StableBand(record["band"]).words()
    return (
        f"For {record['target_sessions'][0]}, the model gives about {tenths['increase']} in 10 "
        f"for a rise beyond {band}, {tenths['stable']} in 10 for staying within it and "
        f"{tenths['decrease']} in 10 for a fall beyond it.{usual_text}"
    )


def five_session_frame(record: Mapping, evaluation: Mapping | None) -> pd.DataFrame:
    """Long form for the chart: session, horizon, category, probability, range, usual."""

    usual = record["signals"].get("usual_mix") or {}
    rows = []
    for day in record["daily"]:
        target = f"target_{day['horizon']}"
        for category in CATEGORIES:
            key = KEY_OF[category.value]
            track = track_record(evaluation, record["estimator"], target, category.value, day[key])
            rows.append(
                {
                    "horizon": day["horizon"],
                    "session": day["session"],
                    "category": category.value,
                    "probability": day[key],
                    "range_low": None if track is None else track.low,
                    "range_high": None if track is None else track.high,
                    "cases": None if track is None else track.n,
                    "usual": usual.get(key),
                }
            )
    return pd.DataFrame(rows)


def five_session_table(record: Mapping) -> pd.DataFrame:
    rows = []
    for day in record["daily"]:
        rows.append(
            {
                "Session": day["session"],
                "Sessions ahead": day["horizon"],
                "Increase": pct(day["p_increase"]),
                "Stable": pct(day["p_stable"]),
                "Decrease": pct(day["p_decrease"]),
            }
        )
    return pd.DataFrame(rows).set_index("Session")


def cumulative_cards(record: Mapping) -> tuple[ProbabilityCard, ...] | None:
    total = record.get("cumulative")
    if total is None:
        return None
    band = StableBand(record["cumulative_band"]).words()
    return tuple(
        ProbabilityCard(
            CATEGORY_WORDS[category],
            pct(total[KEY_OF[category.value]]),
            f"five-session total, band {band}",
            "",
        )
        for category in CATEGORIES
    )


# --- reversals ---


def streak_profile(
    frame: pd.DataFrame,
    direction: str,
    band: StableBand,
    *,
    lengths: Sequence[int] = range(1, 8),
    outcome_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    For each streak length k: how often the next session rose, stayed or fell
    after exactly k sessions in `direction`, with counts and intervals.
    """

    rows = []
    for k in lengths:
        positions = streak_events(frame, k, direction)
        shares = next_session(frame, positions, outcome_end)
        rise, stay, fall = shares["increase"], shares["stable"], shares["decrease"]
        rows.append(
            {
                "length": k,
                "events": rise.n,
                "rose": rise.estimate,
                "rose_low": rise.ci_low,
                "rose_high": rise.ci_high,
                "stayed": stay.estimate,
                "fell": fall.estimate,
                "fell_low": fall.ci_low,
                "fell_high": fall.ci_high,
                "status": rise.status,
            }
        )
    return pd.DataFrame(rows)


def streak_table(profile: pd.DataFrame, direction: str) -> pd.DataFrame:
    noun = "rising" if direction == "increase" else "falling"
    rows = []
    for row in profile.itertuples():
        status = STATUS_WORDS[row.status]
        rows.append(
            {
                "After": f"{row.length} {noun} session{'s' if row.length > 1 else ''} in a row",
                "Cases": f"{row.events:,}",
                "Next rose": pct(row.rose),
                "Range": DASH
                if row.status == INSUFFICIENT
                else f"{pct(row.rose_low)}–{pct(row.rose_high)}",
                "Next stayed": pct(row.stayed),
                "Next fell": pct(row.fell),
                "Evidence": status or "Enough cases",
            }
        )
    return pd.DataFrame(rows).set_index("After")


def current_streak_sentence(profile: pd.DataFrame, direction: str, length: int) -> str | None:
    """'After 3 rising sessions in a row (41 times), the next session fell 15 times: 37% ...'."""

    rows = profile[profile["length"] == length]
    if rows.empty:
        return None
    row = rows.iloc[0]
    noun = "rising" if direction == "increase" else "falling"
    opposite, low, high, word = (
        ("fell", row["fell_low"], row["fell_high"], row["fell"])
        if direction == "increase"
        else ("rose", row["rose_low"], row["rose_high"], row["rose"])
    )
    lead = f"After {length} {noun} session{'s' if length > 1 else ''} in a row ({row['events']:,} times)"
    if row["status"] == INSUFFICIENT:
        return f"{lead}, there are too few cases to say how often the next session {opposite}."
    return (
        f"{lead}, the next session {opposite} {pct(word)} of the time "
        f"(likely {pct(low)}–{pct(high)})."
    )


def move_table(
    frame: pd.DataFrame,
    threshold: float,
    sign: int,
    band: StableBand,
    *,
    outcome_end: pd.Timestamp | None = None,
) -> tuple[str, pd.DataFrame, int, int]:
    """(definition, outcome table, events, pending) after a one-session move beyond X."""

    summary = after_move(frame, threshold, sign, band, outcome_end=outcome_end)
    table = summary.table[summary.table["regime"] == "all"]
    rows = []
    for horizon, group in table.groupby("horizon"):
        row = {"Over the next": f"{horizon} session{'s' if horizon > 1 else ''}"}
        for outcome in ("continuation", "stabilization", "reversal"):
            match = group[group["outcome"] == outcome].iloc[0]
            row[outcome.capitalize()] = (
                STATUS_WORDS[INSUFFICIENT]
                if match["status"] == INSUFFICIENT
                else pct(match["estimate"])
            )
        row["Cases"] = f"{int(group['n'].iloc[0]):,}"
        rows.append(row)
    return (
        summary.definition,
        pd.DataFrame(rows).set_index("Over the next"),
        summary.events,
        summary.censored,
    )


def regime_table(frame: pd.DataFrame, k: int, direction: str, band: StableBand) -> pd.DataFrame:
    """The next session after exactly k sessions in `direction`, by volatility regime."""

    summary = after_streak(frame, k, direction, band)
    rows = []
    for regime in ("calm", "middle", "volatile"):
        part = summary.table[(summary.table["regime"] == regime) & (summary.table["horizon"] == 1)]
        reversal = part[part["outcome"] == "reversal"].iloc[0]
        rows.append(
            {
                "Volatility": regime.capitalize(),
                "Cases": f"{int(reversal['n']):,}",
                "Reversed next session": STATUS_WORDS[INSUFFICIENT]
                if reversal["status"] == INSUFFICIENT
                else pct(reversal["estimate"]),
            }
        )
    return pd.DataFrame(rows).set_index("Volatility")


def duration_summary(frame: pd.DataFrame, k: int, direction: str, rule: RunRule) -> dict:
    survival = reversal_durations(frame, k, direction, rule)
    table = pd.DataFrame(
        {
            "Lasted more than": [
                f"{int(n)} session{'s' if n > 1 else ''}" for n in survival.lengths
            ],
            "Share of reversals": [pct(value) for value in survival.survival],
            "Still running": [f"{int(n):,}" for n in survival.at_risk],
        }
    )
    return {
        "runs": survival.runs,
        "censored": survival.censored,
        "median": survival.median,
        "table": table.set_index("Lasted more than") if len(table) else table,
    }


# --- bull traps ---


@dataclass(frozen=True, eq=False)
class TrapView:
    kind: str
    rule: TrapRule
    setups_resolved: int
    failure_rate: str
    base_rate: str
    lift: str
    precision: str
    recall: str
    false_alarm_rate: str
    unresolved: int
    latest: Mapping | None
    by_regime: pd.DataFrame


def trap_view(
    frame: pd.DataFrame, rule: TrapRule | None = None, *, kind: str = "bull", outcome_end=None
) -> TrapView:
    rule = rule or TrapRule()
    events = trap_events(frame, rule, kind=kind, outcome_end=outcome_end)
    evaluation = evaluate_traps(events, rule, kind)
    latest = None
    if evaluation.latest_setup is not None:
        latest = {
            "session": pd.Timestamp(evaluation.latest_setup["session"]).strftime("%Y-%m-%d"),
            "status": TRAP_STATUS_WORDS[evaluation.latest_setup["status"]],
            "status_code": evaluation.latest_setup["status"],
        }
    regime = evaluation.by_regime.copy()
    regime["base_rate"] = regime["base_rate"].map(pct)
    regime["setup_rate"] = regime["setup_rate"].map(pct)
    regime = regime.rename(
        columns={
            "regime": "Volatility",
            "breakouts": "Breakouts",
            "base_rate": "Failed (all breakouts)",
            "setups": "Setups",
            "setup_rate": "Failed (setups)",
            "status": "Evidence",
        }
    )
    regime["Volatility"] = regime["Volatility"].str.capitalize()
    regime["Evidence"] = regime["Evidence"].map(lambda s: STATUS_WORDS[s] or "Enough cases")
    lift = evaluation.lift
    return TrapView(
        kind=kind,
        rule=rule,
        setups_resolved=evaluation.setup_rate.n,
        failure_rate=STATUS_WORDS[INSUFFICIENT]
        if evaluation.setup_rate.status == INSUFFICIENT
        else pct(evaluation.setup_rate.estimate),
        base_rate=pct(evaluation.base_rate.estimate),
        lift=DASH if not np.isfinite(lift) else f"{lift:.2f}×",
        precision=pct(evaluation.precision),
        recall=pct(evaluation.recall),
        false_alarm_rate=pct(evaluation.false_alarm_rate),
        unresolved=evaluation.unresolved_setups,
        latest=latest,
        by_regime=regime.set_index("Volatility"),
    )


# --- diagnostics ---


def selection_sentence(evaluation: Mapping) -> str:
    selection = evaluation["selection"]
    chosen = selection["selected"]
    if chosen == "B1":
        return (
            f"No challenger beat the recent-weighted frequencies (B1) clearly enough, so B1 is "
            f"kept. Rule: {selection['rule']}."
        )
    return (
        f"{model_words(chosen)} beat the recent-weighted frequencies (B1) by the pre-registered "
        f"rule, so it is used. Rule: {selection['rule']}."
    )


def scores_table(evaluation: Mapping) -> pd.DataFrame:
    metrics = pd.DataFrame(evaluation["metrics"])
    daily = metrics[metrics["target"].str.match(r"target_\d")]
    grouped = daily.groupby("model")[
        ["rps", "brier", "log_loss", "moving_brier", "direction_brier"]
    ]
    table = grouped.mean()
    table.insert(0, "Estimator", [model_words(m) for m in table.index])
    return (
        table.reset_index(drop=True)
        .rename(
            columns={
                "rps": "RPS (primary)",
                "brier": "Brier",
                "log_loss": "Log loss",
                "moving_brier": "Moving vs Stable",
                "direction_brier": "Up vs down",
            }
        )
        .set_index("Estimator")
        .map(lambda v: f"{v:.4f}")
    )


def comparison_table(evaluation: Mapping) -> pd.DataFrame:
    rows = [r for r in evaluation["comparisons"] if r["target"] == "daily_pooled"]
    out = []
    for row in rows:
        out.append(
            {
                "Estimator": model_words(row["model"]),
                "RPS minus B1": f"{row['mean_diff']:+.5f}".replace("-", MINUS),
                "Standard error (Newey-West)": DASH
                if row["nw_se"] is None
                else f"{row['nw_se']:.5f}",
                "One-sided p": DASH if row["p_one_sided"] is None else f"{row['p_one_sided']:.3f}",
                "Holm-adjusted p": DASH if row.get("p_holm") is None else f"{row['p_holm']:.3f}",
                "Dates": f"{row['dates']:,}",
            }
        )
    return pd.DataFrame(out).set_index("Estimator") if out else pd.DataFrame()


def reliability_frame(
    evaluation: Mapping, models: Sequence[str], target: str = "target_1"
) -> pd.DataFrame:
    rows = [r for r in evaluation["reliability"] if r["model"] in models and r["target"] == target]
    return pd.DataFrame(rows)


def calibration_status(frame: pd.DataFrame, model: str, category: int, minimum: int = 200) -> str:
    """
    'Close to calibrated', 'Over-confident', 'Under-confident' or 'Too few
    forecasts', from the bins whose observed interval excludes the forecast:
    over-confident when the outcomes sit closer to the category's base rate
    (its overall frequency) than the forecasts did, under-confident when
    further away.
    """

    rows = frame[(frame["model"] == model) & (frame["category"] == category)]
    if rows["n"].sum() < minimum:
        return "Too few forecasts"
    base = float((rows["observed"] * rows["n"]).sum() / rows["n"].sum())
    outside = (rows["mean_predicted"] < rows["ci_low"]) | (rows["mean_predicted"] > rows["ci_high"])
    misses = rows[outside]
    if misses.empty:
        return "Close to calibrated"
    too_far = (misses["mean_predicted"] - base).abs() > (misses["observed"] - base).abs()
    weight = misses["n"]
    over = (weight * too_far).sum() >= (weight * ~too_far).sum()
    return "Over-confident" if over else "Under-confident"


def fold_table(evaluation: Mapping) -> pd.DataFrame:
    rows = []
    for fold in evaluation["folds"]:
        rows.append(
            {
                "Test window": f"{fold['test_start']} to {fold['test_end']}",
                "Training rows": f"{fold['train_rows']:,}",
                "Test rows": f"{fold['test_rows']:,}",
                "Purged rows": f"{fold['purged_rows']:,}",
                "Half-life chosen": f"{fold['half_life']:.0f}",
                "Logistic C (next session)": DASH
                if "target_1" not in fold["m1_c"]
                else f"{fold['m1_c']['target_1']:g}",
            }
        )
    return pd.DataFrame(rows).set_index("Test window")


# --- the forecast record (PR 3) ---


def staleness(record: Mapping, job_run: Mapping | None) -> str | None:
    """
    The words for a stale forecast, or None: stale when the latest job run of
    the same kind of data expected a later session than the forecast's
    as-of session (its newest close never arrived, so no new forecast was
    made and none is invented).
    """

    if job_run is None or job_run.get("data_class") != record["data_class"]:
        return None
    expected = job_run.get("expected_session")
    if expected and expected > record["as_of_session"]:
        return (
            f"Stale: made as of {record['as_of_session']}; the latest update expected "
            f"{expected} ({job_run['status']}). It stays the last valid forecast until a new close "
            "arrives."
        )
    return None


OUTCOME_WORDS = {0: "Decrease", 1: "Stable", 2: "Increase"}


def history_table(
    forecasts: Sequence[Mapping], scores: Sequence[Mapping], limit: int = 30
) -> pd.DataFrame:
    """
    The latest forecasts, newest first: as-of session, estimator, the next
    session's three probabilities, and what happened when it has closed
    (its category and the ranked probability score of the stored
    probabilities); "pending" until then. Stored probabilities are shown as
    recorded: nothing is recomputed or rewritten.
    """

    settled = {(s["forecast_id"], s["target"]): s for s in scores}
    rows = []
    for record in list(forecasts)[::-1][:limit]:
        if record["refusal"] is not None or not record["daily"]:
            continue
        first = record["daily"][0]
        score = settled.get((record["forecast_id"], "1"))
        if score is None:
            outcome, rps_text = "pending", DASH
        elif score["status"] != "scored":
            outcome, rps_text = f"not scored ({score['reason']})", DASH
        else:
            outcome = OUTCOME_WORDS[score["outcome"]]
            rps_text = f"{score['rps']:.3f}"
        rows.append(
            {
                "As of": record["as_of_session"],
                "For": first["session"],
                "Estimator": record["estimator"],
                "Increase": pct(first["p_increase"]),
                "Stable": pct(first["p_stable"]),
                "Decrease": pct(first["p_decrease"]),
                "What happened": outcome,
                "Score (RPS)": rps_text,
            }
        )
    return pd.DataFrame(rows).set_index("As of") if rows else pd.DataFrame()


def scored_summary(scores: Sequence[Mapping]) -> dict | None:
    """Counts and mean scores of the scored daily targets (None before any)."""

    daily = [s for s in scores if s["status"] == "scored" and s["target"] != "cum"]
    if not daily:
        return None
    return {
        "targets": len(daily),
        "rps": float(np.mean([s["rps"] for s in daily])),
        "brier": float(np.mean([s["brier"] for s in daily])),
    }
