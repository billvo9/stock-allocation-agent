"""
The registered synthetic strategy comparison as the Experiments page reads it
(pure; no Streamlit).

The comparison is produced by `scripts/run_ppo_comparison.py`
(`stock_agent.comparison`, which the app may not import) and written once to
`reports/comparison/<artifact_id>/comparison.json`. This module re-checks
each file before anything from it is shown, and withholds any file that is
not a result of the registered comparison protocol:

- the schema, the protocol and the `content_sha256` over the canonical JSON
  of every other field (an edited file is withheld);
- synthetic data only, labelled "Synthetic scenarios only — not evidence of
  real investment performance" and "engineering evaluation";
- the approved transition, the fairness contract passed, and the test fold
  with its look recorded in the evaluation log;
- every strategy's registry stage at most `development_evaluated`;
- its look is in the configuration's private evaluation log
  (`<looks dir>/<config_sha256>.jsonl`; STOCK_AGENT_COMPARISON_LOOKS or
  ~/stock-agent-private/comparison/looks), whose hash chain is re-verified
  here with the producer's rule: the entry names this artifact, the test
  fold, its look number and its entry hash. A missing or broken log
  withholds the artifact.

It computes no measure and no statistic: every number shown comes from the
artifact, and formatting is the only work done here. Strategies keep their
registered order; nothing is sorted by a result.

Location: the environment variable STOCK_AGENT_COMPARISON_DIR, or the default
reports/comparison/ (ignored by Git).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from stock_agent.app import wording

COMPARISON_ENV = "STOCK_AGENT_COMPARISON_DIR"
DEFAULT_DIR = Path(__file__).resolve().parents[3] / "reports" / "comparison"
LOOKS_ENV = "STOCK_AGENT_COMPARISON_LOOKS"
DEFAULT_LOOKS_DIR = Path.home() / "stock-agent-private" / "comparison" / "looks"
GENESIS = "0" * 64
LOG_FIELDS = {
    "log_schema_version",
    "sequence",
    "recorded_at",
    "config_name",
    "config_sha256",
    "fold",
    "look_number",
    "repeat",
    "artifact_id",
    "selection_sha256",
    "code_sha",
    "git_dirty",
    "prev_entry_sha256",
    "entry_sha256",
}
ARTIFACT_FILE = "comparison.json"
# The producer's contract (`stock_agent.comparison.report`); a test checks they agree.
SCHEMA = "ppo_comparison"
SCHEMA_VERSION = "1.1.0"
PROTOCOL = "registered_synthetic_comparison"
SCOPE_LABEL = "Synthetic scenarios only — not evidence of real investment performance"
EVALUATION_LABEL = "engineering evaluation"
EVIDENCE_STATUS = "synthetic_engineering_only"
SHOWN_STAGES = ("engineering_only", "development_evaluated")
DASH = "—"


class ComparisonWithheld(ValueError):
    """A comparison file that is not a result of the registered protocol; nothing is shown."""


@dataclass(frozen=True)
class Comparisons:
    shown: Mapping[str, object] | None
    withheld: tuple[tuple[str, str], ...] = field(default=())  # (file, reason)
    looks: Mapping[str, object] | None = None  # the shown artifact's log facts


def comparison_dir() -> Path:
    configured = os.environ.get(COMPARISON_ENV)
    return Path(configured) if configured else DEFAULT_DIR


def looks_dir() -> Path:
    configured = os.environ.get(LOOKS_ENV)
    return Path(configured) if configured else DEFAULT_LOOKS_DIR


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _refuse_constant(token: str) -> None:
    raise ComparisonWithheld(f"a non-finite number ({token})")


def verified(text: str) -> dict:
    """The artifact in `text` after every check (module docstring), or `ComparisonWithheld`."""

    try:
        document = json.loads(text, parse_constant=_refuse_constant)
    except json.JSONDecodeError as error:
        raise ComparisonWithheld("it is not JSON") from error
    if not isinstance(document, dict):
        raise ComparisonWithheld("it is not a comparison artifact")
    if (document.get("schema"), document.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        raise ComparisonWithheld("it is not a ppo_comparison 1.0.0 artifact")
    if document.get("protocol") != PROTOCOL:
        raise ComparisonWithheld("it was not produced by the registered comparison protocol")
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    digest = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
    if document.get("content_sha256") != digest:
        raise ComparisonWithheld("its content does not match its recorded hash (edited)")
    labels = document.get("labels") or {}
    if document.get("synthetic") is not True or document.get("evidence_status") != EVIDENCE_STATUS:
        raise ComparisonWithheld("it is not labelled as synthetic engineering evidence")
    if labels.get("scope") != SCOPE_LABEL or labels.get("evaluation") != EVALUATION_LABEL:
        raise ComparisonWithheld("it lacks the synthetic-scope and engineering labels")
    if (document.get("transition") or {}).get("status") != "approved":
        raise ComparisonWithheld("its simulator transition is not the approved one")
    if (document.get("fairness") or {}).get("contract") != "passed":
        raise ComparisonWithheld("its fairness contract did not pass")
    if (document.get("fold") or {}).get("role") != "test":
        raise ComparisonWithheld("it is not the registered test-fold evaluation")
    look = document.get("look") or {}
    if not isinstance(look.get("number"), int) or look["number"] < 1:
        raise ComparisonWithheld("its look at the test fold is not on record")
    for strategy in document.get("strategies") or []:
        if strategy.get("registry_stage") not in SHOWN_STAGES:
            # Nothing from a withheld file is echoed, not even a strategy's name.
            raise ComparisonWithheld(
                "a strategy claims a registry stage above what a synthetic comparison can show"
            )
    return document


def log_entries(text: str, config_sha256: str) -> list[dict]:
    """A configuration's looks after verifying the chain (`ComparisonWithheld` if not)."""

    if text and not text.endswith("\n"):
        raise ComparisonWithheld("its evaluation log's last line is incomplete")
    entries, previous = [], GENESIS
    for number, line in enumerate(text.splitlines()):
        try:
            entry = json.loads(line, parse_constant=_refuse_constant)
        except json.JSONDecodeError as error:
            raise ComparisonWithheld("its evaluation log is not JSON") from error
        if not isinstance(entry, dict) or set(entry) != LOG_FIELDS:
            raise ComparisonWithheld("its evaluation log has unexpected fields")
        body = {key: value for key, value in entry.items() if key != "entry_sha256"}
        digest = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
        if (
            entry["sequence"] != number
            or entry["prev_entry_sha256"] != previous
            or entry["entry_sha256"] != digest
            or entry["config_sha256"] != config_sha256
        ):
            raise ComparisonWithheld("its evaluation log failed its integrity check")
        entries.append(entry)
        previous = entry["entry_sha256"]
    return entries


def look_on_record(document: Mapping[str, object], looks: Path) -> dict[str, object]:
    """The artifact's look as the private log records it, or `ComparisonWithheld`."""

    path = Path(looks) / f"{document['config_sha256']}.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ComparisonWithheld("its look is not in the evaluation log") from error
    entries = log_entries(text, str(document["config_sha256"]))
    look = document["look"]
    matches = [e for e in entries if e["artifact_id"] == document["artifact_id"]]
    if len(matches) != 1 or (
        matches[0]["fold"],
        matches[0]["look_number"],
        matches[0]["entry_sha256"],
    ) != ("test", look["number"], look["log_entry_sha256"]):
        raise ComparisonWithheld("its look does not match the evaluation log")
    return {
        "test_looks": sum(e["fold"] == "test" for e in entries),
        "dev_looks": sum(e["fold"] == "dev" for e in entries),
        "head": entries[-1]["entry_sha256"],
    }


def read_comparisons(
    directory: Path | None = None, looks: Path | None = None
) -> Comparisons | None:
    """
    The newest verified comparison (by `generated_at`) with its log facts,
    and every withheld file with its reason; None when no comparison file
    exists.
    """

    root = comparison_dir() if directory is None else Path(directory)
    looks = looks_dir() if looks is None else Path(looks)
    files = sorted(root.glob(f"*/{ARTIFACT_FILE}")) if root.is_dir() else []
    if not files:
        return None
    shown, withheld = [], []
    for path in files:
        try:
            document = verified(path.read_text(encoding="utf-8"))
            shown.append((document, look_on_record(document, looks)))
        except (ComparisonWithheld, OSError, UnicodeDecodeError) as error:
            withheld.append((path.parent.name, str(error)))
    if not shown:
        return Comparisons(shown=None, withheld=tuple(withheld))
    document, facts = max(shown, key=lambda item: (item[0]["generated_at"], item[0]["artifact_id"]))
    return Comparisons(shown=document, withheld=tuple(withheld), looks=facts)


# --- formatting (no computation beyond picking fields) ---


def _wealth(value: float | None) -> str:
    return DASH if value is None else f"{value:.4f}"


def _percent(value: float | None) -> str:
    return DASH if value is None else f"{value * 100:+.2f}%"


def _fraction(value: float | None) -> str:
    return DASH if value is None else f"{value * 100:.1f}%"


def _median(spread: Mapping[str, float] | None, fmt) -> str:
    return DASH if spread is None else fmt(spread["median"])


def _range(spread: Mapping[str, float] | None, fmt) -> str:
    if spread is None:
        return DASH
    return f"{fmt(spread['median'])} ({fmt(spread['min'])} to {fmt(spread['max'])})"


def labels_of(document: Mapping[str, object]) -> dict[str, str]:
    return {s["key"]: s["label"] for s in document["strategies"]}


def constraint_set_label(constraint_set: Mapping[str, object]) -> str:
    limit = constraint_set["max_turnover"]
    turnover = (
        wording.COMPARISON_NO_TURNOVER_LIMIT
        if limit is None
        else wording.COMPARISON_TURNOVER_LIMIT.format(limit=limit)
    )
    return wording.COMPARISON_SET.format(
        name=constraint_set["name"], cap=constraint_set["max_weight"], turnover=turnover
    )


def facts(document: Mapping[str, object], looks: Mapping[str, object]) -> list[str]:
    look = document["look"]
    code = document["code"]
    return [
        wording.COMPARISON_FACT_PROTOCOL.format(
            name=document["config"]["name"], sha=document["config_sha256"][:12]
        ),
        wording.COMPARISON_FACT_LOOK.format(
            number=look["number"],
            repeat=wording.COMPARISON_REPEAT if look["repeat"] else "",
            recorded=looks["test_looks"],
            generated=document["generated_at"],
        ),
        wording.COMPARISON_FACT_LOG.format(head=str(looks["head"])[:12], dev=looks["dev_looks"]),
        wording.COMPARISON_FACT_CODE.format(
            sha=(code.get("git_sha") or "unknown")[:12],
            dirty=wording.COMPARISON_DIRTY if code.get("git_dirty") else "",
        ),
        wording.COMPARISON_FACT_SIMULATOR.format(
            lag=document["execution_lag"],
            every=document["schedule"]["rebalance_every"],
            offset=document["schedule"]["rebalance_offset"],
        ),
        _sentence(document["labels"]["cash"]),
        _sentence(document["labels"]["costs"]),
    ]


def _sentence(text: str) -> str:
    return f"{text[:1].upper()}{text[1:]}." if text else text


def strategy_table(document: Mapping[str, object]) -> pd.DataFrame:
    columns = wording.COMPARISON_STRATEGY_COLUMNS
    rows = []
    for strategy in document["strategies"]:
        seeds = strategy["member_seeds"]
        rows.append(
            {
                columns["strategy"]: strategy["label"],
                columns["kind"]: wording.COMPARISON_KINDS.get(strategy["kind"], strategy["kind"]),
                columns["rule"]: strategy["rule"],
                columns["seeds"]: DASH
                if seeds is None
                else ", ".join(str(s) for s in next(iter(seeds.values()))),
                columns["stage"]: wording.PROSPECTIVE_STAGES[strategy["registry_stage"]],
            }
        )
    return pd.DataFrame(rows, columns=list(columns.values()))


def _summaries(document, constraint_set: str, scenario: str) -> list[Mapping[str, object]]:
    by_key = {
        s["strategy"]: s
        for s in document["summary"]
        if (s["constraint_set"], s["scenario"]) == (constraint_set, scenario)
    }
    return [by_key[s["key"]] for s in document["strategies"] if s["key"] in by_key]


def results_table(
    document: Mapping[str, object], constraint_set: str, scenario: str
) -> pd.DataFrame:
    columns = wording.COMPARISON_RESULT_COLUMNS
    labels = labels_of(document)
    rows = []
    for summary in _summaries(document, constraint_set, scenario):
        drawdown = summary["drawdown"]
        incomplete = drawdown["incomplete_histories"]
        rows.append(
            {
                columns["strategy"]: labels[summary["strategy"]],
                columns["runs"]: str(summary["episodes"]),
                columns["wealth"]: _range(summary["terminal_wealth"], _wealth),
                columns["return"]: _median(summary["total_return"], _percent),
                columns["drawdown"]: _median(drawdown["value"], _fraction)
                + (wording.COMPARISON_INCOMPLETE.format(n=incomplete) if incomplete else ""),
                columns["duration"]: _median(drawdown["duration_sessions"], lambda v: f"{v:.0f}"),
                columns["construction"]: _median(
                    summary["construction_turnover_one_way"], lambda v: f"{v:.2f}"
                ),
                columns["rebalance"]: _median(
                    summary["rebalance_turnover_one_way"], lambda v: f"{v:.2f}"
                ),
                columns["per_rebalance"]: _median(
                    summary["turnover_per_rebalance"], lambda v: f"{v:.3f}"
                ),
                columns["sessions"]: _median(summary["span_sessions"], lambda v: f"{v:.0f}"),
                columns["notional"]: _median(summary["traded_notional"], lambda v: f"{v:.2f}"),
                columns["zero"]: _median(summary["cost_sensitivity"]["0x"], _wealth),
                columns["one"]: _median(summary["cost_sensitivity"]["1x"], _wealth),
                columns["two"]: _median(summary["cost_sensitivity"]["2x"], _wealth),
                columns["cap"]: str(summary["interventions"]["cap_hits"]),
                columns["limit"]: str(summary["interventions"]["turnover_limit_hits"]),
                columns["fallbacks"]: str(summary["fallback_events"]),
            }
        )
    return pd.DataFrame(rows, columns=list(columns.values()))


def seed_table(document: Mapping[str, object], constraint_set: str, scenario: str) -> pd.DataFrame:
    columns = wording.COMPARISON_SEED_COLUMNS
    labels = labels_of(document)
    rows = []
    for entry in document["seed_variability"]:
        if (entry["constraint_set"], entry["scenario"]) != (constraint_set, scenario):
            continue
        spread = entry["distribution"]
        rows.append(
            {
                columns["strategy"]: labels[entry["strategy"]],
                columns["seeds"]: str(spread["n"]),
                columns["min"]: _wealth(spread["min"]),
                columns["q1"]: _wealth(spread["q1"]),
                columns["median"]: _wealth(spread["median"]),
                columns["q3"]: _wealth(spread["q3"]),
                columns["max"]: _wealth(spread["max"]),
                columns["values"]: ", ".join(
                    f"{seed}: {_wealth(value)}"
                    for seed, value in zip(
                        entry["seeds"], entry["per_seed_mean_terminal_wealth"], strict=True
                    )
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(columns.values()))


def build_table(document: Mapping[str, object], constraint_set: str) -> pd.DataFrame:
    columns = wording.COMPARISON_BUILD_COLUMNS
    labels = labels_of(document)
    by_key = {b["strategy"]: b for b in document["build"] if b["constraint_set"] == constraint_set}
    rows = []
    for strategy in document["strategies"]:
        build = by_key.get(strategy["key"])
        if build is None:
            continue
        weights = build["executed_weights"]["median"]
        episodes = build["episodes"]
        rows.append(
            {
                columns["strategy"]: labels[strategy["key"]],
                columns["weights"]: DASH
                if weights is None
                else " / ".join(_fraction(w) for w in weights),
                columns["cash"]: _median(build["uninvested_cash"], _fraction),
                columns["incomplete"]: f"{build['incomplete_construction']} of {episodes}",
                columns["never"]: f"{build['never_fully_built']} of {episodes}",
                columns["sessions"]: _median(
                    build["sessions_until_fully_built"], lambda v: f"{v:.0f}"
                ),
                columns["end_cash"]: _median(build["terminal_cash_weight"], _fraction),
            }
        )
    return pd.DataFrame(rows, columns=list(columns.values()))


def scenarios_of(document: Mapping[str, object]) -> Sequence[str]:
    return list(document["fold"]["scenario_ids"])
