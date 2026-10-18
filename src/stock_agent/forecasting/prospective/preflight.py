"""
The registration preflight: a read-only report, made before the registration
pull request is merged, of whether the system is ready to register. Each
check is pass or fail with a machine-readable reason and its evidence; what
cannot be machine-checked is `owner_attestation_required`, and what is not
built yet is `not_built`. The report carries the SHA-256 of its canonical
JSON. Nothing here registers, records, fetches or writes; the job's script
gathers the inputs (the store's reader, the research store, Git, the Model
Health summary, the kept head's path) and passes them in.

    data         window          the prospective store holds the complete,
                                 spot-checked 64-close window ending at the
                                 latest available close, for every fund
                 fingerprint     the bundle's training-data fingerprints equal
                                 those recomputed from the research store
    software     code_identity   the running code's registered paths are
                                 committed and equal the bundle's freeze commit
                 record          the registration record matches the bundle
                 environment     the environment manifest (not built yet)
    monitoring   model_health    Model Health's latest summary is at most
                                 26 hours old and not failed
                 calendar        the exchange calendar covers the 24-month
                                 checkpoint (session 524) from the latest close
    backups      head_backup     the kept head's file is configured, outside
                                 every repository, a regular file of mode 600
                                 and writable, in a directory of mode 700 that
                                 also holds ledger_head_history.log (mode 600)
                 second_medium   owner attestation required
    not built    forecast_job_schedule, checkpoint_analysis

The report names its subject (the bundle and the record's SHA-256) and
passes when no check fails; `owner_attestation_required` and `not_built`
lines do not fail it. Its SHA-256 shows integrity, not authenticity: anyone
can recompute it, and `generated_at` is the local clock.

`gate(report, ...)` would refuse a registration whose preflight does not
match its SHA-256, names another bundle or record, has a failing check, lacks
or adds a check, has a check other than `pass` (the second medium excepted,
whose attestation the owner records), was made after the merge or is older
than MAX_AGE at the merge. It is NOT wired into `register`: that is part of
ETF-R1, which the owner has not approved.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.data.private_paths import repository_of
from stock_agent.forecasting.prospective import registration, registration_record
from stock_agent.forecasting.prospective.readiness import PreMerge
from stock_agent.forecasting.prospective.refusal import ForecastRefused
from stock_agent.model_diagnostics.record import canonical_json, sha256_json

PASS, FAIL = "pass", "fail"
OWNER_ATTESTATION, NOT_BUILT = "owner_attestation_required", "not_built"
REPORT_TYPE = "prospective_registration_preflight"
REPORT_SCHEMA_VERSION = "1.0"
HEALTH_MAX_AGE = pd.Timedelta(hours=26)  # Model Health's own staleness rule (model.STALE_AFTER)
MAX_AGE = pd.Timedelta(hours=26)  # proposed for ETF-R1: a preflight this old at the merge is stale
SESSIONS_NEEDED = registration.TRIAL_SESSIONS + registration.CHECKPOINTS["24_months"]  # 524
HISTORY_FILE = "ledger_head_history.log"
# Every check a complete report holds, in order (ETF-R1's gate requires exactly these).
REQUIRED_CHECKS = (
    ("data", "window"),
    ("data", "fingerprint"),
    ("software", "code_identity"),
    ("software", "record"),
    ("software", "environment"),
    ("monitoring", "model_health"),
    ("monitoring", "calendar"),
    ("backups", "head_backup"),
    ("backups", "second_medium"),
    ("operations", "forecast_job_schedule"),
    ("analysis", "checkpoint_analysis"),
)


@dataclass(frozen=True)
class Item:
    area: str
    check: str
    status: str
    reason: str | None = None
    evidence: Mapping[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "area": self.area,
            "check": self.check,
            "status": self.status,
            "reason": self.reason,
            "evidence": dict(self.evidence),
        }

    def line(self) -> str:
        return f"{self.status:>28}  {self.area}.{self.check}  {self.reason or ''}".rstrip()


def _item(area, check, failed: str | None, evidence) -> Item:
    return Item(area, check, FAIL if failed else PASS, failed, evidence)


# --- data -----------------------------------------------------------------------------------


def data_window(pre: PreMerge) -> Item:
    """The verified window ending at the latest available close (readiness.pre_merge)."""

    evidence = {"close": pre.close, **pre.evidence}
    return _item("data", "window", None if pre.ready else "window_not_ready", evidence)


def training_data_summary(data, config) -> dict[str, object]:
    """The bundle's `training_data` block, recomputed with freeze's own function."""

    from stock_agent.forecasting.prospective.training import training_data_block

    return training_data_block(data, config)


def data_fingerprint(
    frozen: Mapping[str, object], recomputed: Mapping[str, object] | None, error: str | None = None
) -> Item:
    if recomputed is None:
        return _item("data", "fingerprint", "research_store_unreadable", {"error": error})
    differ = sorted(
        key for key in set(frozen) | set(recomputed) if frozen.get(key) != recomputed.get(key)
    )
    evidence = {"training_data_sha256": sha256_json(dict(frozen)), "differ": differ}
    return _item("data", "fingerprint", "fingerprint_mismatch" if differ else None, evidence)


# --- software -------------------------------------------------------------------------------


def code_identity(
    running: Mapping[str, object], frozen_trees: Mapping[str, str], frozen_commit: str
) -> Item:
    trees = running.get("trees") if isinstance(running.get("trees"), Mapping) else {}
    differ = sorted(
        p for p in registration_record.CODE_PATHS if trees.get(p) != frozen_trees.get(p)
    )
    evidence = {
        "freeze_commit": frozen_commit,
        "running_commit": running.get("git_sha"),
        "differ": differ,
    }
    if differ:
        return _item("software", "code_identity", "code_differs_from_freeze", evidence)
    if running.get("trees_dirty") is not False:
        return _item("software", "code_identity", "uncommitted_changes", evidence)
    return _item("software", "code_identity", None, evidence)


def record_match(record: Mapping[str, object] | None, models) -> Item:
    if record is None:
        return _item("software", "record", "no_record", {})
    try:
        body = registration_record.validate_record(record)
        registration_record.check_record(body, models)
    except ForecastRefused as refused:
        return _item("software", "record", refused.reason, {"detail": refused.detail})
    return _item(
        "software", "record", None, {"record_sha256": registration_record.record_sha256(body)}
    )


def environment_manifest() -> Item:
    return Item(
        "software",
        "environment",
        NOT_BUILT,
        "environment_manifest_not_built",
        {"note": "the environment manifest is built in a separate pull request"},
    )


# --- monitoring -------------------------------------------------------------------------------


def model_health(
    generated_at: datetime | None,
    overall_state: str | None,
    *,
    now: pd.Timestamp,
    error: str | None = None,
) -> Item:
    if generated_at is None:
        return _item("monitoring", "model_health", "health_unreadable", {"error": error})
    stamp = pd.Timestamp(generated_at).tz_convert("UTC")
    age = pd.Timestamp(now).tz_convert("UTC") - stamp
    evidence = {
        "generated_at": stamp.isoformat(),
        "age_hours": round(age / pd.Timedelta(hours=1), 2),
        "overall_state": overall_state,
    }
    if age < pd.Timedelta(0):
        return _item("monitoring", "model_health", "health_from_the_future", evidence)
    if age > HEALTH_MAX_AGE:
        return _item("monitoring", "model_health", "health_stale", evidence)
    if overall_state in ("failed", "unavailable"):  # the two worst states
        return _item("monitoring", "model_health", "health_failed", evidence)
    return _item("monitoring", "model_health", None, evidence)


def calendar_coverage(calendar: Callable, start: pd.Timestamp) -> Item:
    """The exchange calendar must reach session 524 (C24) from `start`, about session 1."""

    first = pd.Timestamp(start).tz_convert("UTC").normalize()
    evidence: dict[str, object] = {"from": str(first.date()), "sessions_needed": SESSIONS_NEEDED}
    try:
        sessions = calendar(first, first + pd.Timedelta(days=int(SESSIONS_NEEDED * 1.6)))
    except (CalendarRefused, ValueError) as error:
        return _item(
            "monitoring", "calendar", "calendar_short", {**evidence, "refusal": str(error)}
        )
    evidence["sessions_available"] = len(sessions)
    return _item(
        "monitoring",
        "calendar",
        None if len(sessions) >= SESSIONS_NEEDED else "calendar_short",
        evidence,
    )


# --- backups --------------------------------------------------------------------------------


def _shown(path: Path) -> str:
    """A path with the home directory as `~`: the report never names the account."""

    try:
        return "~/" + path.relative_to(Path.home()).as_posix()
    except ValueError:
        return str(path)


def head_backup(path: Path | None, repository: Path) -> Item:
    """
    The kept head's file (section 6): configured, outside every repository, a
    regular file of mode 600 and writable, in a directory of mode 700 that
    also holds the history log (mode 600).
    """

    if path is None:
        return _item("backups", "head_backup", "not_configured", {})
    file = Path(path).expanduser()
    evidence: dict[str, object] = {"path": _shown(file)}
    if repository_of(file, Path(repository)) is not None:
        return _item("backups", "head_backup", "inside_repository", evidence)
    if file.is_symlink() or not file.exists():
        return _item("backups", "head_backup", "missing", evidence)
    if not file.is_file():
        return _item("backups", "head_backup", "not_a_file", evidence)
    mode = file.stat().st_mode & 0o777
    directory_mode = file.parent.stat().st_mode & 0o777
    evidence |= {"mode": oct(mode), "directory_mode": oct(directory_mode)}
    if mode != 0o600:
        return _item("backups", "head_backup", "mode_not_600", evidence)
    if directory_mode != 0o700:
        return _item("backups", "head_backup", "directory_not_700", evidence)
    history = file.parent / HISTORY_FILE
    if history.is_symlink() or not history.is_file():
        return _item("backups", "head_backup", "history_missing", evidence)
    if history.stat().st_mode & 0o777 != 0o600:
        return _item("backups", "head_backup", "history_mode_not_600", evidence)
    if not os.access(file, os.W_OK):
        return _item("backups", "head_backup", "not_writable", evidence)
    return _item("backups", "head_backup", None, evidence)


def second_medium() -> Item:
    return Item(
        "backups",
        "second_medium",
        OWNER_ATTESTATION,
        "cannot_be_machine_checked",
        {"note": "a copy of the kept head on a medium only the owner controls (section 6)"},
    )


def not_built() -> list[Item]:
    return [
        Item("operations", "forecast_job_schedule", NOT_BUILT, "forecast_job_not_scheduled"),
        Item("analysis", "checkpoint_analysis", NOT_BUILT, "checkpoint_analysis_not_built"),
    ]


# --- the report and the gate ----------------------------------------------------------------


def report(
    items: Sequence[Item], *, generated_at: pd.Timestamp, subject: Mapping[str, object]
) -> dict[str, object]:
    """The report: its subject (bundle and record SHA-256), the checks and its SHA-256."""

    body = {
        "report_type": REPORT_TYPE,
        "report_schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": pd.Timestamp(generated_at).tz_convert("UTC").isoformat(),
        "subject": dict(subject),
        "passed": not any(item.status == FAIL for item in items),
        "items": [item.to_json() for item in items],
    }
    body = json.loads(canonical_json(body))
    return {**body, "sha256": sha256_json(body)}


def verify_report(document: Mapping[str, object]) -> dict[str, object]:
    """The report, refused (`preflight_tampered`) unless it matches its SHA-256 and verdict."""

    body = {key: value for key, value in document.items() if key != "sha256"}
    if sha256_json(body) != document.get("sha256"):
        raise ForecastRefused("preflight_tampered", "the report does not match its SHA-256")
    if (body.get("report_type"), body.get("report_schema_version")) != (
        REPORT_TYPE,
        REPORT_SCHEMA_VERSION,
    ):
        raise ForecastRefused("preflight_tampered", "not a preflight report of this schema")
    items = body.get("items")
    if not isinstance(items, list) or body.get("passed") != (
        not any(isinstance(i, Mapping) and i.get("status") == FAIL for i in items)
    ):
        raise ForecastRefused("preflight_tampered", "the verdict does not follow from the checks")
    return dict(document)


def gate(
    document: Mapping[str, object],
    *,
    merged_at: pd.Timestamp,
    bundle_id: str,
    record_sha256: str,
    max_age: pd.Timedelta = MAX_AGE,
) -> None:
    """The module docstring's gate. Built for ETF-R1; not called by `register`."""

    checked = verify_report(document)
    if checked.get("subject") != {"bundle_id": bundle_id, "record_sha256": record_sha256}:
        raise ForecastRefused(
            "preflight_not_for_this_registration",
            "the preflight names another bundle or registration record",
        )
    items = checked["items"]
    if not checked["passed"]:
        failed = [f"{i['area']}.{i['check']}" for i in items if i.get("status") == FAIL]
        raise ForecastRefused("preflight_failed", f"failing checks: {failed}")
    present = [(i.get("area"), i.get("check")) for i in items]
    expected = {key: PASS for key in REQUIRED_CHECKS} | {
        ("backups", "second_medium"): OWNER_ATTESTATION
    }
    actual = {(i.get("area"), i.get("check")): i.get("status") for i in items}
    if sorted(present) != sorted(REQUIRED_CHECKS) or actual != expected:
        raise ForecastRefused(
            "preflight_incomplete",
            "the preflight must hold every required check once, each passing (the second "
            "medium as an owner attestation); not-built checks leave the system unready",
        )
    made = pd.Timestamp(checked["generated_at"])
    merged = pd.Timestamp(merged_at).tz_convert("UTC")
    if made > merged:
        raise ForecastRefused(
            "preflight_after_merge",
            f"the preflight was made {made.isoformat()}, after the merge {merged.isoformat()}",
        )
    if merged - made > max_age:
        raise ForecastRefused(
            "preflight_stale", f"the preflight was made {merged - made} before the merge"
        )


__all__ = [
    "FAIL",
    "MAX_AGE",
    "NOT_BUILT",
    "OWNER_ATTESTATION",
    "PASS",
    "REQUIRED_CHECKS",
    "Item",
    "calendar_coverage",
    "code_identity",
    "data_fingerprint",
    "data_window",
    "environment_manifest",
    "gate",
    "head_backup",
    "model_health",
    "not_built",
    "record_match",
    "report",
    "second_medium",
    "training_data_summary",
    "verify_report",
]
