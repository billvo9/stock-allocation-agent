"""
The prospective forecast ledger as the Experiments page reads it (pure; no
Streamlit).

The ledger is written by the prospective forecasting job
(`stock_agent.forecasting.prospective.ledger`, which the app may not
import). This module re-verifies its hash chain with the same rule: each
line's `entry_sha256` is the SHA-256 of the canonical JSON (sorted keys, no
whitespace) of its other fields, and each line names the previous line's
hash. Any break refuses the whole ledger; nothing is repaired or skipped.

It computes no forecast and no statistic. Every number shown comes from a
record: the forecast rows, their scores, and the cumulative coverage the
latest scoring run recorded. Formatting is the only work done here.

Location: the environment variable STOCK_AGENT_PROSPECTIVE_LEDGER, or the
private default data/raw/prospective_forecasts/ledger.jsonl (ignored by Git).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.app import wording

LEDGER_ENV = "STOCK_AGENT_PROSPECTIVE_LEDGER"
DEFAULT_LEDGER = (
    Path(__file__).resolve().parents[3] / "data" / "raw" / "prospective_forecasts" / "ledger.jsonl"
)
GENESIS = "0" * 64
FIELDS = {
    "ledger_schema_version",
    "sequence",
    "kind",
    "recorded_at",
    "payload",
    "prev_entry_sha256",
    "entry_sha256",
}
DASH = "—"


class LedgerUnreadable(ValueError):
    """The ledger failed its integrity check; nothing from it is shown."""


@dataclass(frozen=True)
class ForecastRecords:
    forecasts: tuple[Mapping[str, object], ...]  # newest first, each with its entry hash
    scores: Mapping[tuple, Mapping[str, object]]
    cumulative: tuple[Mapping[str, object], ...]


def ledger_path() -> Path:
    configured = os.environ.get(LEDGER_ENV)
    return Path(configured) if configured else DEFAULT_LEDGER


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _refuse_constant(token: str) -> None:
    raise LedgerUnreadable(f"a non-finite number ({token})")


def verified_entries(text: str) -> list[dict]:
    """Every entry of a ledger's text, after verifying the chain (LedgerUnreadable if not)."""

    if text and not text.endswith("\n"):
        raise LedgerUnreadable("its last line is incomplete")
    entries, previous = [], GENESIS
    for number, line in enumerate(text.splitlines()):
        try:
            raw = json.loads(line, parse_constant=_refuse_constant)
        except json.JSONDecodeError:
            raise LedgerUnreadable(f"line {number + 1} is not JSON") from None
        if not isinstance(raw, dict) or set(raw) != FIELDS:
            raise LedgerUnreadable(f"line {number + 1} does not hold the ledger fields")
        claimed = raw.pop("entry_sha256")
        body = _canonical(raw)
        if raw["sequence"] != number or raw["prev_entry_sha256"] != previous:
            raise LedgerUnreadable(f"line {number + 1} breaks the chain")
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != claimed:
            raise LedgerUnreadable(f"line {number + 1} was edited")
        entries.append({**raw, "entry_sha256": claimed})
        previous = claimed
    return entries


def read_records(path: Path | None = None) -> ForecastRecords | None:
    """The ledger's records, or None when no ledger exists yet."""

    path = ledger_path() if path is None else Path(path)
    if path.is_symlink():
        raise LedgerUnreadable("it is a symbolic link")
    if not path.exists():
        return None
    entries = verified_entries(path.read_text(encoding="utf-8"))
    forecasts = [
        {**e["payload"], "entry_sha256": e["entry_sha256"]}
        for e in entries
        if e["kind"] == "forecast"
    ]
    scores, cumulative = {}, ()
    for entry in entries:
        if entry["kind"] != "score":
            continue
        for score in entry["payload"]["scores"]:
            key = (
                score["forecast_entry_sha256"],
                score["asset"],
                score["horizon"],
                score["target"],
            )
            scores[key] = score
        cumulative = tuple(entry["payload"].get("cumulative", ()))
    if not forecasts:
        return None
    return ForecastRecords(tuple(reversed(forecasts)), scores, cumulative)


def _percent(value: object, target: str) -> str:
    if value is None:
        return DASH
    return f"{value:+.2f}%" if target == "total_log_return" else f"{value:.2f}%"


def _share(value: object) -> str:
    return DASH if value is None else f"{value:.0%}"


def forecast_table(forecast: Mapping[str, object], scores: Mapping[tuple, Mapping]) -> pd.DataFrame:
    """One display row per fund, horizon and measure of one forecast record."""

    rows = []
    level = None
    for row in forecast["rows"]:
        target = row["target"]
        base = {
            wording.PROSPECTIVE_COLUMNS["fund"]: row["asset"],
            wording.PROSPECTIVE_COLUMNS["horizon"]: f"{row['horizon']} sessions",
            wording.PROSPECTIVE_COLUMNS["measure"]: wording.PROSPECTIVE_MEASURES[target],
        }
        if row["status"] != "ok":
            rows.append(
                {**base, wording.PROSPECTIVE_COLUMNS["forecast"]: wording.PROSPECTIVE_NO_CLOSE}
            )
            continue
        shown = row["display_percent"]
        level = round(100 * row["band_level"])
        key = (forecast["entry_sha256"], row["asset"], row["horizon"], target)
        score = scores.get(key)
        development = row["development"].get(f"band_coverage_{level}")
        rows.append(
            {
                **base,
                wording.PROSPECTIVE_COLUMNS["forecast"]: _percent(shown["point"], target),
                wording.PROSPECTIVE_COLUMNS["low"]: _percent(shown["band_low"], target),
                wording.PROSPECTIVE_COLUMNS["high"]: _percent(shown["band_high"], target),
                wording.PROSPECTIVE_COLUMNS["null"]: _percent(shown["null"], target),
                wording.PROSPECTIVE_COLUMNS["development"]: _share(development),
                wording.PROSPECTIVE_COLUMNS["stage"]: wording.PROSPECTIVE_STAGES.get(
                    row["stage"], row["stage"]
                ),
                wording.PROSPECTIVE_COLUMNS["outcome"]: (
                    wording.PROSPECTIVE_PENDING
                    if score is None
                    else _percent(score["outcome_display_percent"], target)
                ),
                wording.PROSPECTIVE_COLUMNS["inside"]: (
                    DASH if score is None else ("Yes" if score["inside_band"] else "No")
                ),
            }
        )
    columns = list(wording.PROSPECTIVE_COLUMNS.values())
    return pd.DataFrame(rows).reindex(columns=columns).fillna(DASH)


def cumulative_table(rows) -> pd.DataFrame:
    """The latest recorded cumulative coverage, one row per model version, fund, horizon and measure."""

    shown = [
        {
            "Model": f"{row['model_id']} v{row['model_version']}",
            "Fund": row["asset"],
            "Horizon": f"{row['horizon']} sessions",
            "Measure": wording.PROSPECTIVE_MEASURES[row["target"]],
            "Scored": str(row["n_scored"]),
            "Inside the band": str(row["n_inside"]),
            "Share inside": _share(row["coverage"]),
            "95% interval": f"{_share(row['coverage_low'])} to {_share(row['coverage_high'])}",
        }
        for row in rows
    ]
    return pd.DataFrame(shown)
