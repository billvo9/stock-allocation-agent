"""
Policy safety and diagnostics availability (local monitor only), from the
newest backtest summary (`reports/health_inputs/backtest_summary/`) and the
newest RL evaluation report (`models/rl/<run_id>/evaluation_<fold>_<n>.json`,
schema `rl_engineering_report`, written by scripts/evaluate_rl.py).

Optional backtest-summary fields read here (additions to the agreed v1.0
format; a producer that omits them leaves the check insufficient_evidence):

    constraint_violations  int >= 0: decisions whose requested target broke a
                           hard constraint
    fallback_events        int >= 0: decisions the registered fallback took
    fallback_reasons       {reason: int}
    measures.volatility    the risk measure (as defined by the producer)

portfolio.violations
    Constraint violations: the backtest's `constraint_violations` plus the RL
    report's `invalid_targets` over every method. Any is failed with reason
    `constraint_breached` (an informational constraint-violation alert, never
    an action); none is healthy; neither source reporting is
    insufficient_evidence.
portfolio.fallbacks
    Fallback events: the backtest's `fallback_events` plus the guarded
    policies' `fallback_decisions` in the RL report (the registered fallback,
    fixed balanced weights, or holding positions on stale inputs, decided
    instead of the learned policy), with their reasons. Any is degraded: the
    safety rule worked, and the policy behind it needs a look; none is
    healthy; nothing reported is insufficient_evidence.
portfolio.diagnostics
    Which of the risk (`volatility`), turnover and drawdown (`max_drawdown`)
    measures the newest backtest summary reports. All three is healthy;
    otherwise, or with no backtest summary yet (until the T5a engine writes
    them), insufficient_evidence. Values are compared with limits by the
    `portfolio.*` limit checks, never here.

Every check takes its worst finding; an engineering-only transition is
degraded (`unapproved_transition`), and in the private scope a synthetic
input is at most insufficient_evidence (`synthetic_input`). RL evaluation
reports come from synthetic scenarios, so they are synthetic inputs. Nothing
is recomputed: these are the producers' own counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stock_agent.health import backtest
from stock_agent.health.inputs import (
    SYNTHETIC_FINDING,
    TRANSITION_UNAPPROVED,
    Artifact,
    Finding,
    FormatError,
    NotRead,
    common_findings,
    load_latest_by,
    require,
    worst_finding,
)
from stock_agent.health.model import (
    CONSTRAINT_BREACHED,
    Area,
    Better,
    Check,
    HealthState,
    SummaryRefused,
    loads_strict,
    parse_timestamp,
)

TITLES = {
    "portfolio.violations": "Policy constraint violations",
    "portfolio.fallbacks": "Fallback decisions",
    "portfolio.diagnostics": "Risk, turnover and drawdown diagnostics",
}
REPORT_SCHEMA = "rl_engineering_report"
APPROVED_LABEL = "approved_transition"
GUARDED_KIND = "ppo_guarded"
DIAGNOSTICS = {"risk": "volatility", "turnover": "turnover", "drawdown": "max_drawdown"}


@dataclass(frozen=True)
class Evaluation:
    """The newest RL evaluation report: its own counts, never recomputed."""

    path: Path
    run_id: str
    evaluated_at: datetime
    label: str
    methods: dict[str, dict[str, Any]]
    unreadable: tuple[str, ...] = ()

    def describe(self) -> str:
        text = (
            f"RL evaluation {self.path.parent.name}/{self.path.name}, run {self.run_id}, "
            f"evaluated {self.evaluated_at:%Y-%m-%d %H:%M} UTC, {self.label}, synthetic scenarios"
        )
        if self.unreadable:
            text += f"; unreadable reports ignored: {', '.join(self.unreadable[:3])}"
        return text


def _evaluation(path: Path) -> Evaluation:
    try:
        document = loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SummaryRefused) as error:
        raise FormatError(str(error)) from error
    if not isinstance(document, dict) or document.get("schema") != REPORT_SCHEMA:
        raise FormatError(f"schema is not {REPORT_SCHEMA!r}")
    try:
        evaluated_at = parse_timestamp("evaluated_at", document.get("evaluated_at"))
    except SummaryRefused as error:
        raise FormatError(error.detail) from error
    methods = require(document, "methods", "object")
    for name, method in methods.items():
        if not isinstance(method, dict):
            raise FormatError(f"method {name!r} is not an object")
        for key in ("decisions", "invalid_targets", "fallback_decisions"):
            require(method, key, "int")
        reasons = require(method, "fallback_reasons", "object")
        for reason in reasons:
            require(reasons, reason, "int")
    return Evaluation(
        path=path,
        run_id=require(document, "run_id", "text"),
        evaluated_at=evaluated_at,
        label=require(document, "label", "text"),
        methods=methods,
    )


def load_evaluation(root: Path | None) -> Evaluation | NotRead:
    """The newest readable `<root>/*/evaluation_*.json`, by `evaluated_at`."""

    if root is None:
        return NotRead(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_not_configured",
            "No RL runs directory was given to this run.",
        )
    files = sorted(root.glob("*/evaluation_*.json")) if root.is_dir() else []
    if not files:
        return NotRead(
            HealthState.INSUFFICIENT_EVIDENCE,
            "input_missing",
            f"No RL evaluation report under {root} yet.",
        )
    readable, unreadable = [], []
    for path in files:
        try:
            readable.append(_evaluation(path))
        except FormatError as error:
            unreadable.append(f"{path.parent.name}/{path.name} ({error})")
    if not readable:
        return NotRead(
            HealthState.UNAVAILABLE,
            "unrecognized_format",
            f"No RL evaluation report under {root} could be read: {'; '.join(unreadable[:3])}",
        )
    newest = max(readable, key=lambda item: (item.evaluated_at, str(item.path)))
    return Evaluation(**{**newest.__dict__, "unreadable": tuple(unreadable)})


def _evaluation_findings(evaluation: Evaluation, *, allow_synthetic: bool) -> list[Finding]:
    findings = []
    if not allow_synthetic:
        findings.append(SYNTHETIC_FINDING)
    if evaluation.label != APPROVED_LABEL:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "unapproved_transition",
                "The RL evaluation uses an engineering-only portfolio transition that is not yet "
                "approved, so its counts are a visible limitation, not evidence.",
            )
        )
    if evaluation.unreadable:
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "unreadable_inputs",
                f"{len(evaluation.unreadable)} RL evaluation reports could not be read; the "
                "newest readable one was used.",
            )
        )
    return findings


def _backtests(directory: Path | None) -> dict[str, Artifact] | NotRead:
    """The newest backtest summary per policy, so no policy of a run is left out."""

    loaded = load_latest_by(
        directory, backtest.ARTIFACT, "policy", lambda item: _parse_optional(item.document)
    )
    if isinstance(loaded, NotRead):
        return loaded
    return {policy: artifact for policy, (artifact, _) in loaded.items()}


def _parse_optional(document: dict[str, Any]) -> None:
    for key in ("constraint_violations", "fallback_events"):
        if key in document:
            require(document, key, "int", optional=True)
    reasons = document.get("fallback_reasons")
    if reasons is not None:
        require(document, "fallback_reasons", "object")
        for reason in reasons:
            require(reasons, reason, "int")
    measures = require(document, "measures", "object")
    for key in DIAGNOSTICS.values():
        if key in measures:
            require(measures, key, "nonnegative", optional=True)
    require(document, "transition_status", "text")


@dataclass(frozen=True)
class _Sources:
    backtests: dict[str, Artifact] | NotRead
    evaluation: Evaluation | NotRead

    @property
    def policies(self) -> dict[str, Artifact]:
        return {} if isinstance(self.backtests, NotRead) else self.backtests

    def measured_at(self) -> datetime | None:
        times = [artifact.generated_at for artifact in self.policies.values()]
        if isinstance(self.evaluation, Evaluation):
            times.append(self.evaluation.evaluated_at)
        return max(times) if times else None

    def evidence(self, backtest_dir: Path | None, rl_runs: Path | None) -> str | None:
        parts = []
        if len(self.policies) == 1:
            parts.append(str(next(iter(self.policies.values())).path))
        elif backtest_dir is not None:
            parts.append(str(backtest_dir))
        if isinstance(self.evaluation, Evaluation):
            parts.append(str(self.evaluation.path))
        elif rl_runs is not None:
            parts.append(str(rl_runs))
        return "; ".join(parts) or None


def _not_read_lines(sources: _Sources) -> list[str]:
    lines = []
    if isinstance(sources.backtests, NotRead):
        lines.append(f"backtest summary: {sources.backtests.reason}: {sources.backtests.detail}")
    if isinstance(sources.evaluation, NotRead):
        lines.append(f"RL evaluation: {sources.evaluation.reason}: {sources.evaluation.detail}")
    return lines


def _unavailable(sources: _Sources) -> list[Finding]:
    return [
        Finding(
            HealthState.UNAVAILABLE,
            source.reason,
            "A policy input exists but could not be read, so this is not fully judged.",
        )
        for source in (sources.backtests, sources.evaluation)
        if isinstance(source, NotRead) and source.state is HealthState.UNAVAILABLE
    ]


def _context(sources: _Sources, *, allow_synthetic: bool) -> list[Finding]:
    findings = _unavailable(sources)
    for artifact in sources.policies.values():
        findings += common_findings(artifact, allow_synthetic=allow_synthetic)
    if isinstance(sources.evaluation, Evaluation):
        findings += _evaluation_findings(sources.evaluation, allow_synthetic=allow_synthetic)
    return findings


def _make(check_id: str, finding: Finding, value, unit, threshold, lines, sources, evidence):
    return Check(
        check_id=check_id,
        area=Area.PORTFOLIO,
        title=TITLES[check_id],
        state=finding.state,
        reason=finding.reason,
        measured_at=sources.measured_at() if value is not None else None,
        value=value,
        unit=unit if value is not None else None,
        threshold=threshold,
        better=Better.LOWER if value is not None and check_id != "portfolio.diagnostics" else None,
        tolerance=0 if value is not None and check_id != "portfolio.diagnostics" else None,
        summary=finding.summary,
        detail="\n".join(lines),
        evidence=evidence,
    )


def _violations(sources: _Sources, evidence, *, allow_synthetic: bool) -> Check:
    counts, lines, broken = [], [], []
    for policy, artifact in sources.policies.items():
        reported = artifact.document.get("constraint_violations")
        lines.append(
            f"backtest {artifact.run_id}, policy {policy}: constraint violations "
            + ("not reported" if reported is None else str(reported))
        )
        if reported is not None:
            counts.append(reported)
            if reported:
                broken.append(policy)
    if isinstance(sources.evaluation, Evaluation):
        invalid = {
            name: method["invalid_targets"] for name, method in sources.evaluation.methods.items()
        }
        counts.append(sum(invalid.values()))
        broken += [name for name, n in sorted(invalid.items()) if n]
        flagged = ", ".join(f"{name} {n}" for name, n in sorted(invalid.items()) if n) or "none"
        lines.append(f"{sources.evaluation.describe()}: invalid targets by method: {flagged}")
    lines += _not_read_lines(sources)
    threshold = "no decision breaks a hard constraint"
    if not counts:
        finding = worst_finding(
            [
                *_context(sources, allow_synthetic=allow_synthetic),
                Finding(
                    HealthState.INSUFFICIENT_EVIDENCE,
                    "not_reported",
                    "No backtest or RL evaluation has reported constraint violations yet.",
                ),
            ]
        )
        return _make(
            "portfolio.violations", finding, None, None, threshold, lines, sources, evidence
        )
    total = sum(counts)
    measured = (
        Finding(
            HealthState.FAILED,
            CONSTRAINT_BREACHED,
            f"{total} policy decisions broke a hard constraint ({', '.join(broken)}). Simulated, "
            "not a track record; nothing acts on this alert.",
        )
        if total
        else Finding(HealthState.HEALTHY, None, "No policy decision broke a hard constraint.")
    )
    finding = worst_finding([measured, *_context(sources, allow_synthetic=allow_synthetic)])
    return _make(
        "portfolio.violations",
        finding,
        total,
        "decisions that broke a constraint",
        threshold,
        lines,
        sources,
        evidence,
    )


def _fallbacks(sources: _Sources, evidence, *, allow_synthetic: bool) -> Check:
    counts, decisions, reasons, lines = [], 0, Counter(), []
    for policy, artifact in sources.policies.items():
        reported = artifact.document.get("fallback_events")
        lines.append(
            f"backtest {artifact.run_id}, policy {policy}: fallback events "
            + ("not reported" if reported is None else str(reported))
        )
        if reported is not None:
            counts.append(reported)
            reasons.update(artifact.document.get("fallback_reasons") or {})
    if isinstance(sources.evaluation, Evaluation):
        guarded = {
            name: method
            for name, method in sources.evaluation.methods.items()
            if method.get("kind") == GUARDED_KIND or name.startswith("guarded:")
        }
        fallen = sum(method["fallback_decisions"] for method in guarded.values())
        decisions += sum(method["decisions"] for method in guarded.values())
        for method in guarded.values():
            reasons.update(method["fallback_reasons"])
        if guarded:
            counts.append(fallen)
        per_method = (
            ", ".join(
                f"{name} {method['fallback_decisions']} of {method['decisions']}"
                for name, method in sorted(guarded.items())
            )
            or "no guarded policy evaluated"
        )
        lines.append(f"{sources.evaluation.describe()}: fallback decisions: {per_method}")
    lines.append(
        "reasons: " + (", ".join(f"{r} {n}" for r, n in sorted(reasons.items())) or "none")
    )
    lines += _not_read_lines(sources)
    threshold = "the registered fallback never had to decide"
    if not counts:
        finding = worst_finding(
            [
                *_context(sources, allow_synthetic=allow_synthetic),
                Finding(
                    HealthState.INSUFFICIENT_EVIDENCE,
                    "not_reported",
                    "No backtest or guarded policy evaluation has reported fallback decisions yet.",
                ),
            ]
        )
        return _make(
            "portfolio.fallbacks", finding, None, None, threshold, lines, sources, evidence
        )
    total = sum(counts)
    share = f" of {decisions} guarded decisions" if decisions else ""
    measured = (
        Finding(
            HealthState.DEGRADED,
            "fallbacks_recorded",
            f"The registered fallback decided {total}{share} instead of the policy "
            f"({', '.join(sorted(reasons)) or 'reasons not reported'}).",
        )
        if total
        else Finding(HealthState.HEALTHY, None, "The registered fallback never had to decide.")
    )
    finding = worst_finding([measured, *_context(sources, allow_synthetic=allow_synthetic)])
    return _make(
        "portfolio.fallbacks",
        finding,
        total,
        "fallback decisions",
        threshold,
        lines,
        sources,
        evidence,
    )


def _diagnostics(sources: _Sources, evidence, *, allow_synthetic: bool) -> Check:
    threshold = "the newest backtest summary of every policy reports risk, turnover and drawdown"
    if isinstance(sources.backtests, NotRead):
        source = sources.backtests
        finding = Finding(
            source.state,
            source.reason,
            "No backtest summary yet: risk, turnover and drawdown diagnostics appear once the "
            "approved T5a engine writes them."
            if source.state is HealthState.INSUFFICIENT_EVIDENCE
            else "The backtest summary could not be read.",
        )
        return _make(
            "portfolio.diagnostics",
            finding,
            None,
            None,
            threshold,
            [source.detail],
            sources,
            evidence,
        )
    lines, findings, fewest = [], [], len(DIAGNOSTICS)
    for policy, artifact in sources.policies.items():
        measures = artifact.document["measures"]
        units = artifact.document.get("units") or {}
        shown = [
            f"{name} ({key}): "
            + (
                "not reported"
                if measures.get(key) is None
                else f"{measures[key]} {units.get(key, '(unit as defined by the producer)')}"
            )
            for name, key in DIAGNOSTICS.items()
        ]
        missing = [name for name, key in DIAGNOSTICS.items() if measures.get(key) is None]
        fewest = min(fewest, len(DIAGNOSTICS) - len(missing))
        lines.append(f"{policy}: " + "; ".join(shown) + f" ({artifact.describe()})")
        if missing:
            findings.append(
                Finding(
                    HealthState.INSUFFICIENT_EVIDENCE,
                    "diagnostics_missing",
                    f"The newest backtest of {policy} does not report {' or '.join(missing)}.",
                )
            )
        findings += common_findings(artifact, allow_synthetic=allow_synthetic)
        if artifact.document.get("transition_status") == TRANSITION_UNAPPROVED:
            lines.append(f"{policy}: transition engineering only, not yet approved")
    count = len(sources.policies)
    findings.append(
        Finding(
            HealthState.HEALTHY,
            None,
            "The newest backtest reports risk, turnover and drawdown."
            if count == 1
            else f"The newest backtests of all {count} policies report risk, turnover and drawdown.",
        )
    )
    return _make(
        "portfolio.diagnostics",
        worst_finding(findings),
        fewest,
        f"of {len(DIAGNOSTICS)} diagnostics reported (risk, turnover, drawdown), fewest "
        "across policies",
        threshold,
        lines,
        sources,
        evidence,
    )


def policy_checks(
    backtest_dir: Path | None, rl_runs: Path | None, *, allow_synthetic: bool = False
) -> list[Check]:
    """`portfolio.violations`, `portfolio.fallbacks` and `portfolio.diagnostics`."""

    sources = _Sources(_backtests(backtest_dir), load_evaluation(rl_runs))
    evidence = sources.evidence(backtest_dir, rl_runs)
    common = {"allow_synthetic": allow_synthetic}
    return [
        _violations(sources, evidence, **common),
        _fallbacks(sources, evidence, **common),
        _diagnostics(sources, evidence, **common),
    ]


__all__ = ["TITLES", "load_evaluation", "policy_checks"]
