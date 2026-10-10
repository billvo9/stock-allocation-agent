"""
Run status and research warnings, derived only from stored T2 outputs.

T2 defines check severities (leakage, instrument, research_warning, info)
and stores each check's `passed` value. It defines no run-level status, so
the mapping below is the dashboard's, published on the overview page. It
never re-derives pass/fail from observed values and thresholds, and never
escalates: info rows, warning-status metric rows and unavailable metrics
do not change the run status.

Reading `passed` by severity (it does not always mean "good"):

    leakage / instrument   True = check held; False = failed; missing = undetermined
    research_warning       True = the warning is ACTIVE; False = not active
    info                   recorded fact (e.g. a corroborating method's size)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from stock_agent.model_diagnostics import contract

VALID = "VALID"
WARNING = "WARNING"
BLOCKED = "INVALID/BLOCKED"

STATUS_RULES = (
    (
        BLOCKED,
        (
            "A stored leakage or instrument check failed or has no stored result; or the "
            "stored checks_summary is missing, disagrees with the check rows, or breaks "
            "passed + failed + undetermined = total for a family; or the recorded decision "
            "method disagrees with the stored checks."
        ),
    ),
    (WARNING, "No blocking check; a stored research warning is active."),
    (VALID, "No T2 check failures. This certifies the pipeline, not the results."),
)

BLOCKING_SEVERITIES = ("leakage", "instrument")
UNSAFE_ROLE = contract.UNSAFE_REFERENCE_ROLE  # never a candidate strategy


def check_outcomes(checks: pd.DataFrame) -> pd.Series:
    """Outcome label per stored check row, read from (severity, passed)."""

    def outcome(row) -> str:
        passed = row["passed"]
        known = not pd.isna(passed)
        if row["severity"] in BLOCKING_SEVERITIES:
            return ("pass" if bool(passed) else "fail") if known else "undetermined"
        if row["severity"] == "research_warning":
            return (
                ("warning_active" if bool(passed) else "warning_inactive")
                if known
                else ("warning_undetermined")
            )
        return "info"

    return checks.apply(outcome, axis=1).rename("outcome")


@dataclass(frozen=True)
class RunStatus:
    level: str
    reasons: list[str]
    counts: dict[str, int]
    summary_matches: bool
    families: dict[str, dict[str, int]] = field(default_factory=dict)
    summary_source: str = "recorded"


def family_counts(checks: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Per check family (severity): total, passed, failed, undetermined, from stored rows."""

    families = {}
    for severity, rows in checks.groupby("severity", sort=True):
        passed = rows["passed"]
        families[str(severity)] = {
            "total": len(rows),
            "passed": int(passed.eq(True).fillna(False).sum()),
            "failed": int(passed.eq(False).fillna(False).sum()),
            "undetermined": int(passed.isna().sum()),
        }
    return families


FAMILY_KEYS = ("total", "passed", "failed", "undetermined")


def _summary_problems(checks: pd.DataFrame, checks_summary: dict | None, counts: dict) -> list[str]:
    """
    Every disagreement between the stored summary and the stored rows. Any
    problem blocks the run (fail closed): the summary is what other readers
    see, so it must describe the rows exactly.
    """

    if not isinstance(checks_summary, dict):
        return ["stored checks_summary is missing"]
    problems = []
    expected = {
        "leakage_checks_failed": counts["leakage_failed"],
        "instrument_checks_failed": counts["instrument_failed"],
        "research_warnings": counts["research_warnings"],
    }
    for key, value in expected.items():
        if checks_summary.get(key) != value:
            problems.append(
                f"checks_summary.{key} = {checks_summary.get(key)} but rows give {value}"
            )
    families = checks_summary.get("families")
    if families is None:
        return problems  # legacy (1.0) summary: families are derived from the rows
    if not isinstance(families, dict):
        return [*problems, "checks_summary.families is not a mapping"]
    rows = family_counts(checks)
    if set(families) != set(rows):
        problems.append(
            f"checks_summary.families covers {sorted(families)} but rows have {sorted(rows)}"
        )
    for family, stored in families.items():
        try:
            values = {key: int(stored[key]) for key in FAMILY_KEYS}
        except (KeyError, TypeError, ValueError):
            problems.append(f"checks_summary.families.{family} lacks {list(FAMILY_KEYS)}")
            continue
        if values["passed"] + values["failed"] + values["undetermined"] != values["total"]:
            problems.append(
                f"checks_summary.families.{family}: passed + failed + undetermined != total"
            )
        if family in rows and values != rows[family]:
            problems.append(
                f"checks_summary.families.{family} = {values} but rows give {rows[family]}"
            )
    return problems


def run_status(
    checks: pd.DataFrame, checks_summary: dict | None, *, record_issues: tuple[str, ...] = ()
) -> RunStatus:
    """VALID / WARNING / INVALID-BLOCKED from the stored check rows."""

    outcomes = check_outcomes(checks)
    severity = checks["severity"]
    counts = {
        "leakage_failed": int(((outcomes == "fail") & (severity == "leakage")).sum()),
        "instrument_failed": int(((outcomes == "fail") & (severity == "instrument")).sum()),
        "undetermined": int((outcomes == "undetermined").sum()),
        "research_warnings": int((outcomes == "warning_active").sum()),
    }
    problems = _summary_problems(checks, checks_summary, counts)
    summary_matches = not problems
    source = (
        "recorded"
        if isinstance(checks_summary, dict) and "families" in checks_summary
        else "legacy: families derived from the stored check rows"
    )
    families = family_counts(checks)
    reasons = []
    for name, outcome in zip(checks["check"], outcomes, strict=True):
        if outcome in ("fail", "undetermined"):
            reasons.append(f"{name}: {outcome}")
    reasons += problems
    reasons += list(record_issues)
    if reasons:
        return RunStatus(BLOCKED, reasons, counts, summary_matches, families, source)
    active = checks.loc[outcomes == "warning_active", ["check", "model"]]
    warnings = [f"{row.check}: {row.model}" for row in active.itertuples(index=False)]
    level = WARNING if warnings else VALID
    return RunStatus(level, warnings, counts, summary_matches, families, source)


@dataclass(frozen=True)
class Notice:
    """One research warning or notice, quoted from stored outputs."""

    category: str
    level: str  # "blocking" | "warning" | "notice" (never escalated by the dashboard)
    title: str
    detail: str
    source: str
    rows: pd.DataFrame | None = field(default=None, compare=False)


def research_notices(view) -> list[Notice]:
    """Warnings and notices surfaced from a loaded RunView, without new judgments."""

    record = view.record
    tables = view.tables
    checks = tables["checks"]
    outcomes = check_outcomes(checks)
    notices: list[Notice] = []

    issues = view.record_issues
    if issues:
        notices.append(
            Notice(
                "record",
                "blocking",
                "The run record disagrees with its stored rows",
                " ".join(issues),
                "record.spec, checks",
            )
        )
    gaps = view.legacy_gaps
    if gaps:
        notices.append(
            Notice(
                "legacy",
                "notice",
                f"Output schema {view.schema_version}: some fields were not recorded",
                "This run predates these fields. They are shown as not recorded (or as "
                "derived from stored rows, where labelled), never filled with assumed values: "
                + "; ".join(gaps)
                + ".",
                "manifest.output_schema_version",
            )
        )

    blocking = checks[outcomes.isin(["fail", "undetermined"])]
    if len(blocking):
        notices.append(
            Notice(
                "checks",
                "blocking",
                f"{len(blocking)} leakage or instrument check(s) failed or undetermined",
                "T2 classifies these severities as failures. Review them before reading "
                "results (T2 notes that a purged-memorizer failure can have a legitimate "
                "cause; see docs/research/model_diagnostics.md).",
                "checks",
                blocking,
            )
        )

    universe = record["spec"]["universe"]
    excluded = universe.get("excluded_symbols") or {}
    notices.append(
        Notice(
            "universe",
            "notice",
            "Universe note (as recorded)",
            (universe.get("note") or "No universe note recorded.")
            + (f" Excluded: {excluded}." if excluded else ""),
            "record.spec.universe",
        )
    )
    active = checks[outcomes == "warning_active"]
    for row in active.itertuples(index=False):
        notices.append(
            Notice(
                "selection",
                "warning",
                f"Research warning: {row.check} ({row.model})",
                f"{row.metric} = {row.observed:.3f} (rule: {row.comparison} {row.threshold:g}). "
                f"{row.detail if isinstance(row.detail, str) else ''}",
                "checks",
            )
        )
    undetermined = checks[outcomes == "warning_undetermined"]
    if len(undetermined):
        notices.append(
            Notice(
                "selection",
                "notice",
                f"{len(undetermined)} research-warning check(s) have no stored result",
                "T2 could not evaluate these warnings (for example an undefined estimate); "
                "they neither fire nor clear.",
                "checks",
                undetermined,
            )
        )

    # The unsafe canary reference is a positive control, not a model: it is
    # discussed only in the canary panel.
    metrics = tables["metrics"][tables["metrics"]["role"] != UNSAFE_ROLE]
    mde = metrics[
        (metrics["metric"] == "mean_rank_ic_minimum_detectable")
        & metrics["fold_id"].isna()
        & (metrics["role"] != UNSAFE_ROLE)
        & (metrics["status"] == "ok")
    ]
    if len(mde):
        notices.append(
            Notice(
                "sample",
                "notice",
                "Minimum detectable effects (stored)",
                "Minimum detectable mean rank IC (80% power, decision method): "
                + ", ".join(f"{r.model} {r.estimate:.3f}" for r in mde.itertuples(index=False))
                + ". 'No evidence' below these sizes is not 'no effect'.",
                "metrics",
                mde,
            )
        )

    unavailable = metrics[metrics["status"] == "unavailable"]
    if len(unavailable):
        inventory = (
            unavailable.groupby(["model", "metric", "reason"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )
        notices.append(
            Notice(
                "undefined",
                "notice",
                f"{len(unavailable)} stored metric rows are unavailable",
                "Shown with their stored reasons; never replaced by zeros.",
                "metrics",
                inventory,
            )
        )
    flagged = metrics[metrics["status"] == "warning"]
    curves = tables["curves"][tables["curves"]["role"] != UNSAFE_ROLE]
    flagged_curves = curves[curves["status"] == "warning"]
    if len(flagged) or len(flagged_curves):
        parts = pd.concat(
            [
                flagged[["model", "metric", "reason"]].rename(columns={"metric": "item"}),
                flagged_curves[["model", "curve", "reason"]].rename(columns={"curve": "item"}),
            ]
        ).drop_duplicates()
        notices.append(
            Notice(
                "metric_warnings",
                "notice",
                "Metrics flagged by T2 (status 'warning')",
                "Values exist but are degenerate by construction; read them with the reason.",
                "metrics, curves",
                parts.reset_index(drop=True),
            )
        )

    drift = tables["feature_stats"]
    psi = drift[(drift["scope"] == "evaluation") & (drift["statistic"] == "psi")]
    if len(psi.dropna(subset=["value"])):
        top = psi.loc[psi["value"].idxmax()]
        notices.append(
            Notice(
                "drift",
                "notice",
                "Evaluation-only drift distances",
                f"Largest stored PSI {top['value']:.2f} ({top['feature']}, fold {top['fold_id']}). "
                "Distances only: T2 makes no drift judgment, the 0.10/0.25 guide lines are "
                "uncalibrated, and empty test bins dominate PSI through its floor. Never a "
                "reason to remove a feature.",
                "feature_stats (scope=evaluation)",
            )
        )

    info = checks[(checks["severity"] == "info") & checks["passed"].eq(False).fillna(False)]
    if len(info):
        notices.append(
            Notice(
                "inference",
                "notice",
                "Corroborating methods over-reject in simulation",
                "Simulated false-positive rates above the stored threshold for methods that are "
                "not the decision method (recorded as info by T2).",
                "checks (severity=info)",
                info[["metric", "observed", "threshold"]].reset_index(drop=True),
            )
        )

    evidence = record["lockbox_evidence"]
    notices.append(
        Notice(
            "lockbox",
            "notice",
            "Lockbox evidence (as recorded)",
            f"holdout_start {evidence.get('holdout_start')}; holdout_values_used "
            f"{evidence.get('holdout_values_used')}; holdout_rows_loaded "
            f"{evidence.get('holdout_rows_loaded')}; max target end "
            f"{evidence.get('max_target_end_date')}. {evidence.get('adjusted_price_disclosure') or ''}",
            "record.lockbox_evidence",
        )
    )
    if record["spec"]["code"].get("git_dirty"):
        notices.append(
            Notice(
                "provenance",
                "notice",
                "Run produced from a modified working tree",
                f"git_dirty=True at {record['spec']['code'].get('git_sha')}; the diff hash is "
                f"{record['spec']['code'].get('git_diff_sha256')}.",
                "record.spec.code",
            )
        )
    return notices
