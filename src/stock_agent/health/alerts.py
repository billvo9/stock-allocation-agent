"""
Informational alerts derived from checks, and from the previous summary when
one is given.

    failure               a check failed (a test, contract or pin is broken)
    constraint_violation  a check failed because a stored measure breached its
                          stored limit (reason `constraint_breached`)
    unavailable_data      a measurement that should exist could not be taken
    degraded              a check is outside its warning threshold
    deterioration         a numeric value moved in its worse direction by more
                          than its tolerance since the previous summary, while
                          its state did not get worse

An alert is a message. Nothing reads an alert to trade, retrain, promote,
roll back or change configuration; every alert carries
`automatic_action: "none"` and the summary reader refuses anything else.
insufficient_evidence raises no alert: it is the expected state of a check
whose producer does not exist yet.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from stock_agent.health.model import (
    CONSTRAINT_BREACHED,
    Alert,
    AlertKind,
    Better,
    Check,
    HealthState,
    HealthSummary,
    format_timestamp,
    worse,
)

_STATE_KINDS = {
    HealthState.FAILED: AlertKind.FAILURE,
    HealthState.UNAVAILABLE: AlertKind.UNAVAILABLE_DATA,
    HealthState.DEGRADED: AlertKind.DEGRADED,
}


def _since(check: Check, previous: HealthSummary | None) -> str:
    if previous is None:
        return ""
    before = previous.check(check.check_id)
    stamp = format_timestamp(previous.generated_at)
    if before is None:
        return f" (not in the previous summary of {stamp})"
    if before.state is check.state:
        return f" (also {check.state.value} in the previous summary of {stamp})"
    return f" (new: {before.state.value} in the previous summary of {stamp})"


def _worsened(check: Check, before: Check) -> float | None:
    """How far the value moved in its worse direction, when that is defined."""

    numeric = (int, float)
    if (
        check.better is None
        or check.tolerance is None
        or isinstance(check.value, bool)
        or isinstance(before.value, bool)
        or not isinstance(check.value, numeric)
        or not isinstance(before.value, numeric)
        or check.unit != before.unit
    ):
        return None
    if check.better is Better.HIGHER:
        return before.value - check.value
    return check.value - before.value


def derive_alerts(
    checks: Sequence[Check], *, raised_at: datetime, previous: HealthSummary | None = None
) -> tuple[Alert, ...]:
    alerts: list[Alert] = []
    for check in checks:
        kind = _STATE_KINDS.get(check.state)
        if kind is AlertKind.FAILURE and check.reason == CONSTRAINT_BREACHED:
            kind = AlertKind.CONSTRAINT_VIOLATION
        if kind is not None:
            message = f"{check.title} is {check.state.value}: {check.summary}"
            alerts.append(
                Alert(
                    kind=kind,
                    check_id=check.check_id,
                    state=check.state,
                    message=message + _since(check, previous),
                    raised_at=raised_at,
                )
            )
        before = None if previous is None else previous.check(check.check_id)
        if before is None or worse(check.state, before.state):
            continue
        moved = _worsened(check, before)
        if moved is not None and moved > check.tolerance:
            alerts.append(
                Alert(
                    kind=AlertKind.DETERIORATION,
                    check_id=check.check_id,
                    state=check.state,
                    message=(
                        f"{check.title} moved from {before.value} to {check.value} "
                        f"{check.unit or ''} since {format_timestamp(previous.generated_at)} "
                        f"(tolerance {check.tolerance:g}); state still {check.state.value}."
                    ).replace("  ", " "),
                    raised_at=raised_at,
                )
            )
    return tuple(alerts)
