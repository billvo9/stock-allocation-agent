"""
Build a summary from checks and write it.

`build_summary` derives the alerts (comparing with the previous summary when
one is given) and leaves the overall state to the summary itself, which
recomputes it from the checks. `write_summary` writes through a temporary
file in the same directory and an atomic rename, then reads the file back
with the strict reader, so a written summary is always a readable one.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path

from stock_agent.health.alerts import derive_alerts
from stock_agent.health.context import CodeState
from stock_agent.health.model import (
    Check,
    HealthSummary,
    NotApplicable,
    Scope,
    dumps,
    read_summary,
)


def build_summary(
    checks: Sequence[Check],
    *,
    generated_at: datetime,
    code: CodeState,
    scope: Scope,
    run_context: Mapping[str, str | None],
    previous: HealthSummary | None = None,
    not_applicable: Sequence[NotApplicable] = (),
) -> HealthSummary:
    checks = tuple(checks)
    return HealthSummary(
        not_applicable=tuple(not_applicable),
        generated_at=generated_at,
        code_sha=code.sha,
        code_dirty=code.dirty,
        scope=scope,
        run_context=run_context,
        checks=checks,
        alerts=derive_alerts(checks, raised_at=generated_at, previous=previous),
    )


def write_summary(path: str | Path, summary: HealthSummary) -> Path:
    """Write `summary` to `path` atomically (replacing it) and verify it reads back."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".health-", suffix=".json", dir=target.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(dumps(summary.to_json_dict()))
        os.replace(temporary, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    read_summary(target)
    return target
