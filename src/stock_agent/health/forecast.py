"""
Forecast health from the ML track's registry export (artifact
`model_registry`, agreed format v1.0), read from `<inputs>/model_registry/`
(the newest file by `generated_at`). The registry itself stays the source of
truth for entries and stages (`health.registry` reads it through its API);
this module reads only the export's `forecast_evaluation` rows:

    model_id, version (>= 1), horizon (>= 1)
    forecast_kind       "point" | "interval" | "probabilistic"
    interval_level      fraction, or null
    observed_coverage   fraction, or null
    n                   number of evaluated forecasts
    calibration_status  "pass" | "warn" | "fail" | "not_run"
    calibration_detail  text, or null

The export's `models`, `selection_events` and `experiments` must be present
with the agreed types; they are not judged here.

Point forecasts carry null interval fields and `not_run`. Coverage and
calibration do not apply to them: they are named in the detail and never
count towards the overall state, and when every row is a point forecast both
checks are listed as not applicable rather than given a state.

forecast.coverage     the largest |observed_coverage - interval_level| over
                      interval and probabilistic rows with n >= MIN_COVERAGE_N;
                      healthy <= 0.05, failed > 0.10. A screening band, not a
                      test: overlapping horizons make forecasts dependent, so
                      a binomial standard error would be too small
forecast.calibration  the producer's own calibration status per row

Health computes no statistic beyond comparing these stored values. In the
private scope a synthetic export is at most insufficient_evidence
(`synthetic_input`).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from stock_agent.health.inputs import (
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
from stock_agent.health.model import Area, Better, Check, HealthState, NotApplicable

ARTIFACT = "model_registry"
KINDS = frozenset({"point", "interval", "probabilistic"})
STATUSES = frozenset({"pass", "warn", "fail", "not_run"})
MIN_COVERAGE_N = 50
COVERAGE_DEGRADED = 0.05
COVERAGE_FAILED = 0.10
CHECKS = {
    "forecast.coverage": "Forecast interval coverage",
    "forecast.calibration": "Forecast calibration",
}
POINT_NOTE = "not applicable: point forecast"


def _parse(artifact: Artifact) -> list[dict]:
    document = artifact.document
    require(document, "models", "list")
    require(document, "selection_events", "list")
    require(document, "experiments", "object")
    rows = require(document, "forecast_evaluation", "list")
    for row in rows:
        if not isinstance(row, dict):
            raise FormatError("every forecast evaluation must be an object")
        require(row, "model_id", "text")
        require(row, "version", "positive")
        require(row, "horizon", "positive")
        kind = choice(row, "forecast_kind", KINDS)
        level = require(row, "interval_level", "fraction", optional=True)
        observed = require(row, "observed_coverage", "fraction", optional=True)
        require(row, "n", "int")
        status = choice(row, "calibration_status", STATUSES)
        require(row, "calibration_detail", "text", optional=True)
        if kind == "point" and (level is not None or observed is not None or status != "not_run"):
            raise FormatError("a point forecast carries null interval fields and not_run")
    return rows


def _label(row: dict) -> str:
    return f"{row['model_id']}@v{row['version']} h{row['horizon']}"


def _coverage(
    rows: list[dict], points: int, artifact: Artifact, common: dict, allow_synthetic: bool
) -> Check:
    threshold = (
        f"|observed - nominal| <= {COVERAGE_DEGRADED} healthy, > {COVERAGE_FAILED} failed; "
        f"rows with n < {MIN_COVERAGE_N} or no reported coverage are not judged"
    )
    unit = "largest |observed - nominal| interval coverage"
    eligible = [
        row
        for row in rows
        if row["n"] >= MIN_COVERAGE_N
        and row["interval_level"] is not None
        and row["observed_coverage"] is not None
    ]
    point_note = f"; {points} point forecasts: {POINT_NOTE}" if points else ""
    if not eligible:
        finding = worst_finding(
            [
                *common_findings(artifact, allow_synthetic=allow_synthetic),
                Finding(
                    HealthState.INSUFFICIENT_EVIDENCE,
                    "too_few_observations",
                    "No interval forecast has enough reported observations to judge coverage.",
                ),
            ]
        )
        return Check(
            state=finding.state,
            reason=finding.reason,
            measured_at=artifact.generated_at,
            value=None,
            unit=unit,
            threshold=threshold,
            summary=finding.summary,
            detail=f"{len(rows)} interval or probabilistic rows, none judged{point_note}; "
            f"{artifact.describe()}",
            **common,
        )
    gaps = [(abs(row["observed_coverage"] - row["interval_level"]), row) for row in eligible]
    gap, row = max(gaps, key=lambda pair: pair[0])
    if gap > COVERAGE_FAILED:
        band = Finding(
            HealthState.FAILED,
            "coverage_off",
            f"{_label(row)} intervals cover {row['observed_coverage']:.0%} against a stated "
            f"{row['interval_level']:.0%}.",
        )
    elif gap > COVERAGE_DEGRADED:
        band = Finding(
            HealthState.DEGRADED,
            "coverage_off",
            f"{_label(row)} intervals cover {row['observed_coverage']:.0%} against a stated "
            f"{row['interval_level']:.0%}.",
        )
    else:
        band = Finding(
            HealthState.HEALTHY,
            None,
            f"Forecast intervals cover close to their stated level (widest gap {gap:.1%}).",
        )
    finding = worst_finding([band, *common_findings(artifact, allow_synthetic=allow_synthetic)])
    return Check(
        state=finding.state,
        reason=finding.reason,
        measured_at=artifact.generated_at,
        value=round(gap, 4),
        unit=unit,
        threshold=threshold,
        better=Better.LOWER,
        tolerance=0.02,
        summary=finding.summary,
        detail=(
            f"{len(eligible)} of {len(rows)} interval or probabilistic rows judged; widest gap "
            f"{_label(row)}: observed {row['observed_coverage']:.1%} vs nominal "
            f"{row['interval_level']:.0%} over n={row['n']}{point_note}; {artifact.describe()}"
        ),
        **common,
    )


def _calibration(
    rows: list[dict], points: int, artifact: Artifact, common: dict, allow_synthetic: bool
) -> Check:
    statuses = Counter(row["calibration_status"] for row in rows)
    flagged = [
        f"{_label(row)}: {row['calibration_detail'] or row['calibration_status']}"
        for row in rows
        if row["calibration_status"] in {"fail", "warn"}
    ]
    if statuses["fail"]:
        band = Finding(
            HealthState.FAILED,
            "calibration_failed",
            f"{statuses['fail']} forecast evaluations fail their calibration check.",
        )
    elif statuses["warn"]:
        band = Finding(
            HealthState.DEGRADED,
            "calibration_warning",
            f"{statuses['warn']} forecast evaluations warn on calibration.",
        )
    elif not statuses["pass"]:
        band = Finding(
            HealthState.INSUFFICIENT_EVIDENCE,
            "no_calibration_result",
            "No forecast calibration check has run yet.",
        )
    else:
        band = Finding(
            HealthState.HEALTHY,
            None,
            f"All {statuses['pass']} evaluated interval forecasts pass their calibration check.",
        )
    finding = worst_finding([band, *common_findings(artifact, allow_synthetic=allow_synthetic)])
    detail = ", ".join(f"{status} {count}" for status, count in sorted(statuses.items()))
    if flagged:
        detail += "; " + "; ".join(flagged[:5])
    if points:
        detail += f"; {points} point forecasts: {POINT_NOTE}"
    return Check(
        state=finding.state,
        reason=finding.reason,
        measured_at=artifact.generated_at,
        value=statuses["pass"],
        unit=f"of {len(rows)} interval evaluations pass the producer's calibration check",
        threshold="no evaluation fails or warns",
        better=Better.HIGHER,
        tolerance=0,
        summary=finding.summary,
        detail=f"{detail}; {artifact.describe()}",
        **common,
    )


def forecast_checks(
    directory: Path | None, *, allow_synthetic: bool = False
) -> list[Check | NotApplicable]:
    evidence = None if directory is None else str(directory)
    loaded = load_latest(directory, ARTIFACT)
    rows = None
    if isinstance(loaded, Artifact):
        try:
            rows = _parse(loaded)
        except FormatError as error:
            loaded = NotRead(
                HealthState.UNAVAILABLE, "unrecognized_format", f"{loaded.path}: {error}"
            )
    if rows is None:
        return [
            Check(
                check_id=check_id,
                area=Area.FORECAST,
                title=title,
                state=loaded.state,
                reason=loaded.reason,
                measured_at=None,
                value=None,
                unit=None,
                threshold=None,
                summary="The model-registry export could not be read, so this is not judged."
                if loaded.state is HealthState.UNAVAILABLE
                else "No forecast evaluation has been exported yet.",
                detail=loaded.detail,
                evidence=evidence,
            )
            for check_id, title in CHECKS.items()
        ]
    applicable = [row for row in rows if row["forecast_kind"] != "point"]
    points = len(rows) - len(applicable)
    if points and not applicable:
        return [
            NotApplicable(
                check_id=check_id,
                area=Area.FORECAST,
                title=title,
                reason="point_forecast",
                detail=f"All {points} evaluated forecasts are point forecasts; {POINT_NOTE}. "
                f"{loaded.describe()}",
            )
            for check_id, title in CHECKS.items()
        ]
    if not rows:
        return [
            Check(
                check_id=check_id,
                area=Area.FORECAST,
                title=title,
                state=HealthState.INSUFFICIENT_EVIDENCE,
                reason="no_forecast_evaluation",
                measured_at=None,
                value=None,
                unit=None,
                threshold=None,
                summary="The registry export lists no forecast evaluation yet.",
                detail=loaded.describe(),
                evidence=evidence,
            )
            for check_id, title in CHECKS.items()
        ]
    common = {"area": Area.FORECAST, "evidence": evidence}
    return [
        _coverage(
            applicable,
            points,
            loaded,
            {**common, "check_id": "forecast.coverage", "title": CHECKS["forecast.coverage"]},
            allow_synthetic,
        ),
        _calibration(
            applicable,
            points,
            loaded,
            {
                **common,
                "check_id": "forecast.calibration",
                "title": CHECKS["forecast.calibration"],
            },
            allow_synthetic,
        ),
    ]
