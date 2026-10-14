"""
Training health from the RL track's training-health files (artifact
`rl_training_health`, agreed format v1.0), read from
`<inputs>/rl_training_health/` (the newest file by `generated_at`).

Fields besides the envelope (`inputs`):

    algorithm          text (for example "ppo")
    status             "completed" | "running" | "failed"
    seeds              number of seeds in the run (>= 1)
    scenario_version   text
    transition_status  "approved" | "engineering_only_unapproved_transition"
    checks             [{check_id, status: pass|warn|fail|not_run, value, unit,
                         threshold, detail}]

The listed check ids are `LISTED_CHECKS`; a listed id the file omits counts
as not_run, and other ids are shown too. Health takes the producer's own
statuses and recomputes no training metric. The check takes its worst
finding: a failed run or any failed check is failed; any warning, an
unapproved transition or unreadable input files are degraded; a run in
progress or a listed check that did not run is insufficient_evidence; only a
completed run whose every listed check passed is healthy.
"""

from __future__ import annotations

from pathlib import Path

from stock_agent.health.inputs import (
    TRANSITION_STATUSES,
    Artifact,
    Finding,
    FormatError,
    NotRead,
    choice,
    common_findings,
    load_latest,
    require,
    worst_finding,
)
from stock_agent.health.model import Area, Better, Check, HealthState

ARTIFACT = "rl_training_health"
RUN_STATUSES = frozenset({"completed", "running", "failed"})
CHECK_STATUSES = frozenset({"pass", "warn", "fail", "not_run"})
LISTED_CHECKS = (
    "nan_or_inf",
    "exploding_values",
    "approx_kl",
    "clip_fraction",
    "entropy_collapse",
    "value_loss",
    "reward_trend",
    "seed_spread",
)
TITLE = "Training health"
THRESHOLD = "the run completed and every listed producer check passed"


def _parse(artifact: Artifact) -> dict:
    document = artifact.document
    parsed = {
        "algorithm": require(document, "algorithm", "text"),
        "status": choice(document, "status", RUN_STATUSES),
        "seeds": require(document, "seeds", "positive"),
        "scenario_version": require(document, "scenario_version", "text"),
        "transition_status": choice(document, "transition_status", TRANSITION_STATUSES),
        "checks": require(document, "checks", "list"),
    }
    ids = []
    for item in parsed["checks"]:
        if not isinstance(item, dict):
            raise FormatError("every check must be an object")
        ids.append(require(item, "check_id", "text"))
        choice(item, "status", CHECK_STATUSES)
        require(item, "value", "number", optional=True)
        for key in ("unit", "threshold", "detail"):
            require(item, key, "text", optional=True)
    if len(ids) != len(set(ids)):
        raise FormatError("check ids repeat")
    return parsed


def _describe(item: dict) -> str:
    text = f"{item['check_id']}: {item['status']}"
    if item.get("value") is not None:
        text += f" ({item['value']}{' ' + item['unit'] if item.get('unit') else ''})"
    if item.get("threshold"):
        text += f" [{item['threshold']}]"
    return text


def _findings(parsed: dict, statuses: dict[str, str], artifact: Artifact) -> list[Finding]:
    def named(status: str) -> list[str]:
        return [check_id for check_id, value in statuses.items() if value == status]

    findings = []
    if parsed["status"] == "failed":
        findings.append(Finding(HealthState.FAILED, "training_failed", "The training run failed."))
    if named("fail"):
        findings.append(
            Finding(
                HealthState.FAILED,
                "training_check_failed",
                f"{len(named('fail'))} training checks failed: {', '.join(named('fail'))}.",
            )
        )
    if named("warn"):
        findings.append(
            Finding(
                HealthState.DEGRADED,
                "training_check_warning",
                f"{len(named('warn'))} training checks warn: {', '.join(named('warn'))}.",
            )
        )
    findings += common_findings(artifact)
    if parsed["status"] == "running":
        findings.append(
            Finding(
                HealthState.INSUFFICIENT_EVIDENCE,
                "training_in_progress",
                "The training run is still in progress; it is judged when it completes.",
            )
        )
    not_run = [check_id for check_id in LISTED_CHECKS if statuses[check_id] == "not_run"]
    if not_run:
        findings.append(
            Finding(
                HealthState.INSUFFICIENT_EVIDENCE,
                "training_checks_not_run",
                f"{len(not_run)} of the {len(LISTED_CHECKS)} listed training checks did not "
                f"run: {', '.join(not_run)}.",
            )
        )
    findings.append(
        Finding(
            HealthState.HEALTHY,
            None,
            f"The run completed and all {len(LISTED_CHECKS)} listed training checks passed.",
        )
    )
    return findings


def training_checks(directory: Path | None) -> list[Check]:
    common = {
        "check_id": "training.health",
        "area": Area.TRAINING,
        "title": TITLE,
        "threshold": THRESHOLD,
        "evidence": None if directory is None else str(directory),
    }
    loaded = load_latest(directory, ARTIFACT)
    parsed = None
    if isinstance(loaded, Artifact):
        try:
            parsed = _parse(loaded)
        except FormatError as error:
            loaded = NotRead(
                HealthState.UNAVAILABLE, "unrecognized_format", f"{loaded.path}: {error}"
            )
    if parsed is None:
        return [
            Check(
                state=loaded.state,
                reason=loaded.reason,
                measured_at=None,
                value=None,
                unit=None,
                summary="No training-health record could be read, so training is not judged."
                if loaded.state is HealthState.UNAVAILABLE
                else "No training run has reported its health yet.",
                detail=loaded.detail,
                **common,
            )
        ]

    statuses = {check_id: "not_run" for check_id in LISTED_CHECKS}
    statuses.update({item["check_id"]: item["status"] for item in parsed["checks"]})
    finding = worst_finding(_findings(parsed, statuses, loaded))
    extra = [item for item in parsed["checks"] if item["check_id"] not in LISTED_CHECKS]
    listed = {item["check_id"]: item for item in parsed["checks"]}
    shown = [
        _describe(listed.get(check_id, {"check_id": check_id, "status": "not_run (absent)"}))
        for check_id in LISTED_CHECKS
    ]
    detail = (
        f"{parsed['algorithm']}, status {parsed['status']}, {parsed['seeds']} seeds, scenario "
        f"{parsed['scenario_version']}, transition {parsed['transition_status']}; listed: "
        + "; ".join(shown)
        + (f"; other checks: {'; '.join(_describe(item) for item in extra)}" if extra else "")
        + f"; {loaded.describe()}"
    )
    return [
        Check(
            state=finding.state,
            reason=finding.reason,
            measured_at=loaded.generated_at,
            value=sum(statuses[check_id] == "pass" for check_id in LISTED_CHECKS),
            unit=f"of {len(LISTED_CHECKS)} listed training checks pass",
            better=Better.HIGHER,
            tolerance=0,
            summary=finding.summary,
            detail=detail,
            **common,
        )
    ]
