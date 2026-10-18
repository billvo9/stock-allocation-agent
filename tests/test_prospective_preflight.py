"""
The registration preflight (read-only), run before the registration pull
request is merged: each check is pass or fail with a reason and evidence, the
report carries its SHA-256, and what cannot be machine-checked or is not
built says so. The gate that would refuse a registration without a passing,
fresh preflight made before the merge is built and tested here, but it is not
wired into `register`: that enforcement is part of ETF-R1, not approved.
"""

from __future__ import annotations

import inspect
import os

import pandas as pd
import pytest
from forecast_prospective_helpers import (
    ASSETS,
    CODE_TREES,
    RUN_CODE,
    exchange_calendar,
    prospective_store,
    registration,
    without_close,
)

from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.forecasting.prospective import (
    preflight,
    readiness,
    registration_record,
    training,
)
from stock_agent.forecasting.prospective.refusal import ForecastRefused

NOW = pd.Timestamp("2026-10-15T23:00:00Z")
MERGED = pd.Timestamp("2026-10-16T14:00:00Z")


def _pre_merge(store, now=NOW):
    return readiness.pre_merge(
        load_window=load_inference_window,
        store_root=store,
        assets=ASSETS,
        calendar=exchange_calendar,
        now=now,
        lookback=63,
    )


# --- data -----------------------------------------------------------------------------------


def test_the_data_window_check_passes_with_every_funds_verified_window(prospective_store_root):
    item = preflight.data_window(_pre_merge(prospective_store_root))
    assert (item.area, item.status, item.reason) == ("data", "pass", None)
    assert item.evidence["close"] == "2026-10-15" and item.evidence["batches"]


def test_the_data_window_check_fails_with_its_gaps(tmp_path):
    gap = prospective_store(tmp_path / "gap", edit=without_close("VTI", ["2026-09-15"]))
    item = preflight.data_window(_pre_merge(gap))
    assert (item.status, item.reason) == ("fail", "window_not_ready")
    assert item.evidence["incomplete"] == {"VTI": ["2026-09-15"]}


def test_the_training_data_fingerprint_is_recomputed_from_the_research_store(prospective_frozen):
    bundle = prospective_frozen.result.bundle.document
    data = training.load_training_data(prospective_frozen.store, prospective_frozen.config)
    recomputed = preflight.training_data_summary(data, prospective_frozen.config)
    item = preflight.data_fingerprint(bundle["training_data"], recomputed)
    assert (item.status, item.reason) == ("pass", None)
    changed = {**recomputed, "prices_sha256": "f" * 64}
    item = preflight.data_fingerprint(bundle["training_data"], changed)
    assert (item.status, item.reason) == ("fail", "fingerprint_mismatch")
    assert item.evidence["differ"] == ["prices_sha256"]
    unread = preflight.data_fingerprint(bundle["training_data"], None, error="not_f1_store")
    assert (unread.status, unread.reason) == ("fail", "research_store_unreadable")


# --- software -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("running", "reason"),
    [
        (RUN_CODE, None),
        ({**RUN_CODE, "trees": {**CODE_TREES, "src": "f" * 40}}, "code_differs_from_freeze"),
        ({**RUN_CODE, "trees_dirty": True}, "uncommitted_changes"),
    ],
)
def test_the_code_identity_must_be_the_freeze_commits(running, reason):
    item = preflight.code_identity(running, CODE_TREES, "0" * 40)
    assert item.status == ("pass" if reason is None else "fail") and item.reason == reason


def test_the_record_must_match_the_bundle(prospective_frozen):
    models = prospective_frozen.models()
    assert preflight.record_match(registration(models), models).status == "pass"
    other = {**registration(models), "config_sha256": "f" * 64}
    item = preflight.record_match(other, models)
    assert (item.status, item.reason) == ("fail", "registration_mismatch")
    missing = preflight.record_match(None, models)
    assert (missing.status, missing.reason) == ("fail", "no_record")


def test_the_environment_manifest_is_reported_as_not_built():
    item = preflight.environment_manifest()
    assert (item.status, item.reason) == ("not_built", "environment_manifest_not_built")


# --- monitoring -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("generated", "state", "status", "reason"),
    [
        ("2026-10-15T12:00:00Z", "healthy", "pass", None),
        ("2026-10-15T12:00:00Z", "degraded", "pass", None),
        ("2026-10-14T20:59:59Z", "healthy", "fail", "health_stale"),  # 26 h and 1 s old
        ("2026-10-15T12:00:00Z", "failed", "fail", "health_failed"),
        ("2026-10-15T12:00:00Z", "unavailable", "fail", "health_failed"),
        ("2026-10-16T00:00:00Z", "healthy", "fail", "health_from_the_future"),
    ],
)
def test_model_health_must_be_fresh_and_not_failed(generated, state, status, reason):
    item = preflight.model_health(pd.Timestamp(generated), state, now=NOW)
    assert (item.status, item.reason) == (status, reason)


def test_the_freshness_rule_is_model_healths_own():
    from stock_agent.health.model import STALE_AFTER

    assert preflight.HEALTH_MAX_AGE == pd.Timedelta(STALE_AFTER)


def test_a_missing_health_summary_fails():
    item = preflight.model_health(None, None, now=NOW, error="not_found")
    assert (item.status, item.reason) == ("fail", "health_unreadable")


# --- backups --------------------------------------------------------------------------------


def _kept(path, mode=0o600, *, directory_mode=0o700, history=True):
    """The kept head's file and its history log, in a private directory (section 6)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(directory_mode)
    path.write_text("0" * 64 + "\n", encoding="utf-8")
    path.chmod(mode)
    if history:
        log = path.parent / "ledger_head_history.log"
        log.write_text("", encoding="utf-8")
        log.chmod(0o600)
    return path


def test_the_head_backup_must_be_outside_the_repository_mode_600_and_writable(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    good = _kept(tmp_path / "private" / "ledger_head.txt")
    assert preflight.head_backup(good, repo).status == "pass"
    other = tmp_path / "another_checkout"
    (other / ".git").mkdir(parents=True)  # any repository, not only this checkout
    cases = {
        None: "not_configured",
        tmp_path / "absent.txt": "missing",
        _kept(repo / "kept" / "ledger_head.txt"): "inside_repository",
        _kept(other / "kept" / "ledger_head.txt"): "inside_repository",
        _kept(tmp_path / "loose" / "ledger_head.txt", 0o644): "mode_not_600",
        _kept(tmp_path / "open" / "ledger_head.txt", directory_mode=0o755): "directory_not_700",
        _kept(tmp_path / "alone" / "ledger_head.txt", history=False): "history_missing",
    }
    for path, reason in cases.items():
        item = preflight.head_backup(path, repo)
        assert (item.status, item.reason) == ("fail", reason), path


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anything")
def test_a_read_only_head_backup_fails(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = _kept(tmp_path / "private" / "ledger_head.txt")
    asked = []

    def access(file, mode):
        asked.append(mode)
        return mode != os.W_OK

    monkeypatch.setattr(preflight.os, "access", access)
    assert preflight.head_backup(path, repo).reason == "not_writable"
    assert asked == [os.W_OK]


def test_the_second_medium_needs_the_owners_attestation():
    item = preflight.second_medium()
    assert item.status == "owner_attestation_required"


# --- coverage and what is not built ---------------------------------------------------------


def test_the_calendar_must_cover_the_24_month_checkpoint():
    item = preflight.calendar_coverage(exchange_calendar, pd.Timestamp("2026-10-15", tz="UTC"))
    assert item.evidence["sessions_needed"] == 524

    def refusing(start, end):
        raise CalendarRefused("outside_coverage", "the rules calendar ends 2027-12-31")

    refused = preflight.calendar_coverage(refusing, pd.Timestamp("2026-10-15", tz="UTC"))
    assert (refused.status, refused.reason) == ("fail", "calendar_short")


def test_what_is_not_built_is_listed():
    checks = {item.check for item in preflight.not_built()}
    assert {"forecast_job_schedule", "checkpoint_analysis"} <= checks
    assert all(item.status == "not_built" for item in preflight.not_built())


# --- the report and the (unwired) gate -------------------------------------------------------


SUBJECT = {"bundle_id": "etf_prospective_test-fixture", "record_sha256": "a" * 64}


def _items(*, fail=False, drop=None, status=None):
    """Every required check: passing, except as changed."""

    items = []
    for area, check in preflight.REQUIRED_CHECKS:
        state = preflight.PASS
        if (area, check) == ("backups", "second_medium"):
            state = preflight.OWNER_ATTESTATION
        if status and (area, check) == status[0]:
            state = status[1]
        if (area, check) == drop:
            continue
        evidence = {"record_sha256": SUBJECT["record_sha256"]} if check == "record" else {}
        items.append(preflight.Item(area, check, state, None, evidence))
    if fail:
        items[0] = preflight.Item("data", "window", "fail", "window_not_ready", {})
    return items


def _document(generated=NOW, **changes):
    return preflight.report(_items(**changes), generated_at=generated, subject=SUBJECT)


def test_the_report_has_a_sha256_and_passes_only_without_a_failure():
    passing = _document()
    assert passing["passed"] is True and passing["subject"] == SUBJECT
    assert preflight.verify_report(passing) == passing
    failing = _document(fail=True)
    assert failing["passed"] is False
    assert len(passing["sha256"]) == 64 and passing["sha256"] != failing["sha256"]


def test_a_report_whose_hash_does_not_match_is_refused():
    edited = _document(fail=True)
    edited["passed"] = True
    with pytest.raises(ForecastRefused) as refused:
        preflight.verify_report(edited)
    assert refused.value.reason == "preflight_tampered"


def _gate(document, **changes):
    options = {
        "merged_at": MERGED,
        "bundle_id": SUBJECT["bundle_id"],
        "record_sha256": SUBJECT["record_sha256"],
        **changes,
    }
    preflight.gate(document, **options)


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        (_document(), None),
        (_document(fail=True), "preflight_failed"),
        (_document(MERGED - pd.Timedelta(hours=26, seconds=1)), "preflight_stale"),
        (_document(MERGED + pd.Timedelta(seconds=1)), "preflight_after_merge"),
        # A not-built check (the forecast job's schedule, say) leaves the system unready.
        (
            _document(status=(("operations", "forecast_job_schedule"), "not_built")),
            "preflight_incomplete",
        ),
        # A report rebuilt without its failing check, its hash recomputed: incomplete.
        (_document(drop=("data", "window")), "preflight_incomplete"),
        (_document(status=(("backups", "second_medium"), "pass")), "preflight_incomplete"),
    ],
    ids=["passing", "failed", "stale", "after_merge", "not_built", "dropped", "attestation"],
)
def test_the_gate_would_block_a_failing_stale_late_or_incomplete_preflight(document, reason):
    if reason is None:
        _gate(document)
        return
    with pytest.raises(ForecastRefused) as refused:
        _gate(document)
    assert refused.value.reason == reason


@pytest.mark.parametrize("changes", [{"bundle_id": "another-bundle"}, {"record_sha256": "b" * 64}])
def test_the_gate_would_block_a_preflight_made_for_another_registration(changes):
    with pytest.raises(ForecastRefused) as refused:
        _gate(_document(), **changes)
    assert refused.value.reason == "preflight_not_for_this_registration"


def test_registration_does_not_yet_require_a_preflight():
    # ETF-R1 (registration requires a passing preflight) is not approved: `register` takes
    # no preflight, and the gate is not called anywhere in the job.
    assert "preflight" not in inspect.signature(registration_record.register).parameters
    from stock_agent.forecasting.prospective import forecast

    assert "preflight" not in inspect.getsource(registration_record)
    assert "preflight" not in inspect.getsource(forecast)


# --- review follow-ups: boundaries and each condition on its own -----------------------------


def test_the_gate_refuses_a_tampered_report():
    document = _document()
    document["generated_at"] = (MERGED - pd.Timedelta(hours=1)).isoformat()  # not re-hashed
    with pytest.raises(ForecastRefused) as refused:
        _gate(document)
    assert refused.value.reason == "preflight_tampered"


@pytest.mark.parametrize("before", [pd.Timedelta(0), pd.Timedelta(hours=26)])
def test_the_gate_accepts_a_preflight_at_its_time_limits(before):
    _gate(_document(MERGED - before))  # made at the merge itself, or exactly 26 h before


def _rehashed(document, **changes):
    from stock_agent.model_diagnostics.record import sha256_json

    body = {k: v for k, v in document.items() if k != "sha256"} | changes
    return {**body, "sha256": sha256_json(body)}


@pytest.mark.parametrize(
    "edit",
    [
        lambda d: {**d, "items": [{**d["items"][0], "evidence": {"x": 1}}, *d["items"][1:]]},
        lambda d: _rehashed(d, passed=False),  # the verdict no longer follows from the checks
        lambda d: _rehashed(d, report_type="another_report"),
        lambda d: _rehashed(d, report_schema_version="0.9"),
    ],
    ids=["hash", "verdict", "type", "schema"],
)
def test_each_report_condition_is_checked_on_its_own(edit):
    with pytest.raises(ForecastRefused) as refused:
        preflight.verify_report(edit(_document()))
    assert refused.value.reason == "preflight_tampered"


@pytest.mark.parametrize(("count", "status"), [(524, "pass"), (523, "fail")])
def test_the_calendar_check_counts_sessions(count, status):
    def calendar(start, end):
        return pd.bdate_range(start, periods=count, tz="UTC")

    item = preflight.calendar_coverage(calendar, pd.Timestamp("2026-10-15", tz="UTC"))
    assert item.status == status and item.evidence["sessions_available"] == count


def test_model_health_exactly_26_hours_old_is_fresh():
    item = preflight.model_health(NOW - pd.Timedelta(hours=26), "healthy", now=NOW)
    assert item.status == "pass"


@pytest.mark.parametrize(("mode", "reason"), [(0o700, "mode_not_600"), (0o400, "mode_not_600")])
def test_the_head_backup_mode_is_exactly_600(tmp_path, mode, reason):
    repo = tmp_path / "repo"
    repo.mkdir()
    path = _kept(tmp_path / "private" / "ledger_head.txt", mode)
    assert preflight.head_backup(path, repo).reason == reason


def test_a_symlinked_or_directory_head_backup_fails(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    real = _kept(tmp_path / "private" / "ledger_head.txt")
    link = tmp_path / "private" / "link.txt"
    link.symlink_to(real)
    assert preflight.head_backup(link, repo).reason == "missing"
    folder = tmp_path / "private" / "folder"
    folder.mkdir()
    assert preflight.head_backup(folder, repo).reason == "not_a_file"


def test_the_report_never_names_the_home_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight.Path, "home", classmethod(lambda cls: tmp_path))
    repo = tmp_path / "repo"
    repo.mkdir()
    path = _kept(tmp_path / "private" / "ledger_head.txt")
    assert preflight.head_backup(path, repo).evidence["path"] == "~/private/ledger_head.txt"
