"""
The sealed holdout record in plain words: the answer to the one question,
then the descriptive evidence, then the limits. Reads a record only (no
store, no bars); lower scores are better throughout.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

SCORE_WORDS = {"rps": "RPS", "brier": "Brier", "log_loss": "log loss"}
TARGET_WORDS = {
    "daily_pooled": "the five daily targets, pooled",
    "target_cum": "the five-session total",
}
CATEGORIES = ("decrease", "stable", "increase")


def _number(value: object) -> str:
    if not isinstance(value, int | float) or not math.isfinite(value):
        return "n/a"
    return f"{value:+.4f}"


def _p(value: object) -> str:
    if not isinstance(value, int | float) or not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


def _header(record: Mapping) -> list[str]:
    model, counts, protocol = record["model"], record["counts"], record["protocol"]
    out = [
        (
            f"Sealed holdout evaluation of frozen model {model['model_id'][:16]} "
            f"({model['estimator']}, {model['state_rule']}) against {model['comparator']}"
        ),
        (
            f"  record {record['record_id'][:16]}, protocol {record['protocol_sha256'][:16]}, "
            f"run {record['created_at']}"
        ),
        (
            f"  decisions {counts['first_decision']} to {counts['last_decision']} "
            f"({counts['decisions']}), outcomes {protocol['holdout_start']} to "
            f"{protocol['holdout_end']}; {counts['forecasts']} stock forecasts"
        ),
    ]
    skipped = {reason: n for reason, n in counts["skipped"].items() if n}
    if skipped:
        out.append(f"  not forecast: {', '.join(f'{n} ({r})' for r, n in skipped.items())}")
    return out


def _verdict(decision: Mapping) -> str:
    if decision["status"] != "ok":
        reason = decision.get("reason") or decision["status"]
        return f"undetermined ({reason})"
    return "yes" if decision["beats_comparator"] else "no"


def _answer(record: Mapping) -> list[str]:
    decision, comparator = record["decision"], record["model"]["comparator"]
    return [
        f"The question: {decision['question']}",
        (
            f"  answer: {_verdict(decision)} (mean of the block means "
            f"{_number(decision['block_mean_diff'])}, one-sided p {_p(decision['p_one_sided'])} "
            f"against {decision['alpha']:.1%}, t with {decision['df']} degrees of freedom)"
        ),
        (
            f"  a negative difference means the model scored better than {comparator}. 'no' "
            "means the evidence did not reach the pre-registered 2.5% level; the sign and the "
            "interval below are descriptive (a positive difference with an interval above zero "
            "describes a worse score in this period)"
        ),
    ]


def _comparisons(record: Mapping) -> list[str]:
    estimator, comparator = record["model"]["estimator"], record["model"]["comparator"]
    out = [
        (
            f"Every comparison ({estimator} minus {comparator}, the mean of the block means and "
            "its 95% interval; only one decides):"
        )
    ]
    for row in record["comparisons"]:
        mark = "deciding" if row["deciding"] else "reported"
        out.append(
            f"  {SCORE_WORDS[row['score']]:8} {TARGET_WORDS[row['target']]:31} "
            f"{_number(row['block_mean_diff'])} "
            f"[{_number(row['ci_low'])}, {_number(row['ci_high'])}] "
            f"p {_p(row['p_one_sided'])} ({mark})"
        )
    deciding = next(row for row in record["comparisons"] if row["deciding"])
    blocks = ", ".join(f"{b['block']} {_number(b['mean_diff'])}" for b in deciding["by_block"])
    out.append(f"  by block (the deciding comparison): {blocks}")
    return out


def _breakdowns(record: Mapping) -> list[str]:
    estimator, comparator = record["model"]["estimator"], record["model"]["comparator"]
    out = [f"By stock (RPS, {TARGET_WORDS['daily_pooled']}):"]
    for row in record["by_stock"]:
        if row["score"] == "rps" and row["target"] == "daily_pooled":
            out.append(
                f"  {row['stock']:5} {estimator} {row[estimator]:.4f}  {comparator} "
                f"{row[comparator]:.4f}  difference {_number(row['diff'])}  (n {row['n']})"
            )
    out.append(f"By year (RPS, {TARGET_WORDS['daily_pooled']}):")
    for row in record["by_year"]:
        if row["score"] == "rps" and row["target"] == "daily_pooled":
            out.append(f"  {row['year']}  difference {_number(row['diff'])}  (n {row['n']})")
    out.append(f"Base rates, each session's category (D / S / I), against the frozen {comparator}:")
    for row in record["base_rates"]:
        held = " / ".join(f"{row[f'holdout_{c}']:.0%}" for c in CATEGORIES)
        frozen = " / ".join(f"{row[f'b0_{c}']:.0%}" for c in CATEGORIES)
        out.append(f"  {row['stock']:5} holdout {held}  frozen {frozen}  ({row['sessions']})")
    return out


def lines(record: Mapping) -> list[str]:
    """The record's report, one printed line per entry."""

    limits = ["Limits:", *(f"  - {text}" for text in record["limitations"])]
    sections = (_header, _answer, _comparisons, _breakdowns)
    out: list[str] = []
    for section in sections:
        out += [*section(record), ""]
    return out + limits
