"""
Prospective data-quality checks with the real store, reader, rules calendar
and anchors loader passed in as the private script passes them: every state
of `data.anchors`, `data.store_batches` and `data.window_coverage`, and no
amount or close ever copied into a check.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from prospective_helpers import FUNDS, batch, stored

from stock_agent.calendars import nyse_rules
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.data.prospective.window import H_END, WINDOW_START
from stock_agent.data.vendors.anchors import COLUMNS, AnchorScope, load_anchors
from stock_agent.health.ledger import RulesCalendar
from stock_agent.health.model import HealthState
from stock_agent.health.prospective import (
    ProspectiveSources,
    last_complete_session,
    prospective_checks,
)

CALENDAR = RulesCalendar(nyse_rules.sessions, nyse_rules.CALENDAR_ID)
AFTER_H_END_CLOSE = datetime(2026, 10, 14, 23, tzinfo=UTC)  # 19:00 New York
INSUFFICIENT = HealthState.INSUFFICIENT_EVIDENCE
AMOUNT = "0.2718"
ANCHOR_ROWS = {
    "VTI": "2026-09-28",
    "AGG": "2026-08-03",
    "SHV": "2026-10-01",
}


def _anchors_file(tmp_path: Path, funds=tuple(ANCHOR_ROWS)) -> Path:
    path = tmp_path / "anchors.csv"
    rows = [",".join(COLUMNS)]
    for fund in funds:
        rows.append(
            f"{fund},{ANCHOR_ROWS[fund]},{AMOUNT},4,per_share_as_paid,anchor,Issuer,"
            "https://example.com/distributions,2026-10-14,"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _sources(store_root, *, anchors: Path | None = None, loader=True) -> ProspectiveSources:
    project = Path(anchors).parent / "project" if anchors is not None else None
    if project is not None:
        project.mkdir(exist_ok=True)

    def load():
        return load_anchors(
            anchors,
            AnchorScope.PROSPECTIVE,
            project_root=project,
            today=pd.Timestamp("2026-10-15", tz="UTC"),
        )

    return ProspectiveSources(
        store_root=store_root,
        assets=FUNDS,
        window_start=WINDOW_START,
        h_end=H_END,
        calendar=CALENDAR,
        load_window=load_inference_window,
        load_anchors=load if loader and anchors is not None else None,
    )


def _checks(sources, now=AFTER_H_END_CLOSE):
    return {check.check_id: check for check in prospective_checks(sources, now=now)}


def _all_text(checks) -> str:
    return " ".join(f"{c.summary} {c.detail} {c.value}" for c in checks.values())


def test_without_sources_every_check_is_insufficient_evidence():
    checks = _checks(None)
    assert list(checks) == ["data.anchors", "data.store_batches", "data.window_coverage"]
    for check in checks.values():
        assert (check.state, check.reason, check.value) == (
            INSUFFICIENT,
            "input_not_configured",
            None,
        )


def test_a_store_not_created_yet_is_insufficient_evidence(tmp_path):
    checks = _checks(_sources(tmp_path / "prospective"))
    for check_id in ("data.store_batches", "data.window_coverage"):
        assert (checks[check_id].state, checks[check_id].reason) == (INSUFFICIENT, "store_missing")
        assert "awaits the owner's approval" in checks[check_id].summary


def test_a_complete_window_with_valid_anchors_is_healthy(tmp_path):
    root = stored(tmp_path / "prospective", batch())
    checks = _checks(_sources(root, anchors=_anchors_file(tmp_path)))
    assert {c.state for c in checks.values()} == {HealthState.HEALTHY}
    anchors = checks["data.anchors"]
    assert anchors.value == 3 and "VTI 1, AGG 1, SHV 1" in anchors.detail
    assert "sha256 " in anchors.detail
    batches = checks["data.store_batches"]
    assert batches.value == 1 and "spot-checked against an earlier anchors file" in batches.detail
    window = checks["data.window_coverage"]
    assert window.value == 1.0
    assert "window 2026-07-16 to 2026-10-14: 64 sessions x 3 funds = 192" in window.detail
    assert AMOUNT not in _all_text(checks)  # never an amount
    assert "300." not in _all_text(checks)  # never a close


def test_missing_closes_are_degraded_with_their_reason_and_dates(tmp_path):
    root = stored(tmp_path / "prospective", batch(skip={("SHV", "2026-09-15")}))
    window = _checks(_sources(root))["data.window_coverage"]
    assert (window.state, window.reason) == (HealthState.DEGRADED, "closes_missing")
    assert window.value == pytest.approx(191 / 192, abs=1e-6)
    assert "SHV: 63 of 64 closes (missing_session 1): 2026-09-15" in window.detail
    assert "would be refused, never filled" in window.summary


def test_sessions_after_the_last_batch_are_missing_not_filled(tmp_path):
    root = stored(tmp_path / "prospective", batch())
    later = datetime(2026, 10, 16, 23, tzinfo=UTC)  # 10-15 and 10-16 are complete, not stored
    window = _checks(_sources(root), now=later)["data.window_coverage"]
    assert window.state is HealthState.DEGRADED
    assert "no_batch 2" in window.detail and "2026-10-16" in window.detail


def test_values_revised_between_batches_are_a_vendor_basis_warning(tmp_path):
    root = stored(
        tmp_path / "prospective",
        batch(),
        batch(
            retrieved_at=pd.Timestamp("2026-10-15T01:00:00Z"),
            offset=0.5,
            payload=b"a later invented payload",
        ),
    )
    batches = _checks(_sources(root), now=datetime(2026, 10, 15, 2, tzinfo=UTC))[
        "data.store_batches"
    ]
    # The reader uses the newest batch for every session and lists where the older differs.
    assert (batches.state, batches.reason, batches.value) == (
        HealthState.DEGRADED,
        "vendor_revisions",
        1,
    )
    assert "revisions between batches: changed" in batches.detail
    assert "values are never shown" in batches.detail


def test_a_store_the_reader_refuses_is_unavailable(tmp_path):
    root = stored(tmp_path / "prospective", batch())
    (root / "stray.txt").write_text("not a batch", encoding="utf-8")
    checks = _checks(_sources(root))
    for check_id in ("data.store_batches", "data.window_coverage"):
        assert (checks[check_id].state, checks[check_id].reason) == (
            HealthState.UNAVAILABLE,
            "store_refused",
        )
        assert "foreign_entry" in checks[check_id].detail


def test_before_the_window_opens_nothing_is_judged(tmp_path):
    root = stored(tmp_path / "prospective", batch())
    early = datetime(2026, 10, 14, 20, tzinfo=UTC)  # 16:00 New York: H_END not yet complete
    checks = _checks(_sources(root), now=early)
    assert checks["data.window_coverage"].reason == "window_not_open"
    assert checks["data.window_coverage"].state is INSUFFICIENT


def test_an_empty_store_has_no_batches_yet(tmp_path):
    from prospective_helpers import new_store

    root = new_store(tmp_path / "prospective")
    checks = _checks(_sources(root))
    for check_id in ("data.store_batches", "data.window_coverage"):
        assert (checks[check_id].state, checks[check_id].reason) == (INSUFFICIENT, "no_batches")


def test_a_missing_anchors_file_is_an_empty_allowlist(tmp_path):
    anchors = _checks(_sources(None, anchors=tmp_path / "absent.csv"))["data.anchors"]
    assert (anchors.state, anchors.reason, anchors.value) == (INSUFFICIENT, "anchors_missing", None)
    assert "nothing can be stored" in anchors.summary


def test_a_fund_without_an_anchor_is_degraded(tmp_path):
    anchors = _checks(_sources(None, anchors=_anchors_file(tmp_path, ("AGG", "SHV"))))[
        "data.anchors"
    ]
    assert (anchors.state, anchors.reason, anchors.value) == (
        HealthState.DEGRADED,
        "fund_without_anchors",
        2,
    )
    assert "VTI has no anchor" in anchors.summary


def test_a_refused_anchors_file_fails_without_naming_a_value(tmp_path):
    path = _anchors_file(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").replace(",4,", ",3,"), encoding="utf-8")
    anchors = _checks(_sources(None, anchors=path))["data.anchors"]
    assert (anchors.state, anchors.reason) == (HealthState.FAILED, "anchors_refused")
    assert AMOUNT not in anchors.detail and "row" in anchors.detail


def test_the_last_complete_session_waits_for_the_close():
    assert last_complete_session(CALENDAR, WINDOW_START, AFTER_H_END_CLOSE) == H_END
    before = datetime(2026, 10, 14, 21, 59, tzinfo=UTC)  # 17:59 New York
    assert last_complete_session(CALENDAR, WINDOW_START, before) < H_END
