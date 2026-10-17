"""
Smoke checks of the monitors themselves, on synthetic inputs with planted
defects. They answer "would the health center notice?", not "is the data
healthy?": in CI there is no real data to judge.

smoke.data_monitors
    Builds an in-memory synthetic fund store (the F1 synthetic adapter's
    funds for the last quarter of 2024, plus a second synthetic source for
    SYN_EQ). A clean store must give every data check healthy with value 0
    (coverage 1.0). A planted store, with SYN_CASH's last 3 sessions removed
    and 2 of the second source's closes moved by 1 %, must give freshness 3,
    coverage 1 - 3/192, missing sessions 3 and discrepancies 2, each degraded.
smoke.input_readers
    Writes the example producer inputs (`fixtures`, agreed formats v1.0) where
    producers write them and reads them back: a training file with an
    unapproved transition is degraded (`unapproved_transition`) and with an
    approved one healthy; a registry built through the ML track's write API is
    healthy, and fails its entry and reference-version checks once a run record
    is edited; the registry export's interval forecast is judged and a
    point-only export is listed as not applicable; a backtest with a planted cap
    breach fails its concentration check while the others show the unapproved
    transition; absent inputs are insufficient_evidence and an unreadable file
    is unavailable.
"""

from __future__ import annotations

import dataclasses
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_agent.data.funds.adapter import SYNTHETIC_FUND_IDS, SyntheticFundAdapter
from stock_agent.data.funds.ingest import build_record
from stock_agent.data.funds.reader import FundStore
from stock_agent.data.funds.schema import (
    DISTRIBUTIONS,
    MISSING_SESSIONS,
    SPLITS,
    batches_frame,
    empty,
)
from stock_agent.health import (
    backtest,
    data_quality,
    fixtures,
    forecast,
    registry,
    stock_lab,
    training,
)
from stock_agent.health.contracts import FUND_END, FUND_RETRIEVED_AT, FUND_SESSIONS, FUND_START
from stock_agent.health.model import CONSTRAINT_BREACHED, Area, Better, Check, HealthState

SHADOW_SOURCE = "synthetic_shadow"
SHADOW_RETRIEVED_AT = FUND_RETRIEVED_AT + pd.Timedelta(hours=1)
SMOKE_AS_OF = FUND_RETRIEVED_AT + pd.Timedelta(hours=2)
DROPPED_TAIL = 3
BUMPED = 2
BUMP = 1.01


def synthetic_store(*, drop_tail: int = 0, bumped: int = 0) -> FundStore:
    """The synthetic fund store, optionally with planted defects."""

    raw = SyntheticFundAdapter(retrieved_at=FUND_RETRIEVED_AT).fetch(
        list(SYNTHETIC_FUND_IDS), FUND_START, FUND_END
    )
    record = build_record(raw)
    prices = raw.prices.assign(batch_id=record.batch_id)
    if drop_tail:
        cash = prices.index[prices["fund_id"] == "SYN_CASH"]
        prices = prices.drop(cash[-drop_tail:])
    equity = raw.prices[raw.prices["fund_id"] == "SYN_EQ"].copy()
    shadow_raw = dataclasses.replace(
        raw,
        adapter_id=SHADOW_SOURCE,
        fund_ids=("SYN_EQ",),
        instruments=tuple(item for item in raw.instruments if item.fund_id == "SYN_EQ"),
        retrieved_at=SHADOW_RETRIEVED_AT,
        raw_payload=b"synthetic shadow source",
    )
    shadow = build_record(shadow_raw)
    equity["source_id"] = SHADOW_SOURCE
    equity["batch_id"] = shadow.batch_id
    equity.iloc[:bumped, equity.columns.get_loc("close")] *= BUMP
    return FundStore(
        batches=batches_frame([record, shadow]),
        prices=pd.concat([prices, equity], ignore_index=True),
        distributions=empty(DISTRIBUTIONS),
        splits=empty(SPLITS),
        missing_sessions=empty(MISSING_SESSIONS),
    )


def _expected_data() -> list[tuple[str, dict[str, tuple[HealthState, float]]]]:
    total = FUND_SESSIONS * len(SYNTHETIC_FUND_IDS)
    return [
        (
            "clean",
            {
                "data.freshness": (HealthState.HEALTHY, 0),
                "data.coverage": (HealthState.HEALTHY, 1.0),
                "data.missing_sessions": (HealthState.HEALTHY, 0),
                "data.vendor_discrepancy": (HealthState.HEALTHY, 0),
            },
        ),
        (
            "planted",
            {
                "data.freshness": (HealthState.DEGRADED, DROPPED_TAIL),
                "data.coverage": (HealthState.DEGRADED, round(1 - DROPPED_TAIL / total, 6)),
                "data.missing_sessions": (HealthState.DEGRADED, DROPPED_TAIL),
                "data.vendor_discrepancy": (HealthState.DEGRADED, BUMPED),
            },
        ),
    ]


def data_monitor_results() -> list[tuple[str, bool, str]]:
    stores = {
        "clean": synthetic_store(),
        "planted": synthetic_store(drop_tail=DROPPED_TAIL, bumped=BUMPED),
    }
    results = []
    measured = SMOKE_AS_OF.to_pydatetime()
    for scenario, expected in _expected_data():
        checks = data_quality.data_quality_checks(
            stores[scenario],
            as_of=SMOKE_AS_OF,
            measured_at=measured,
            evidence="fixture:stock_agent.health.smoke.synthetic_store",
        )
        for check in checks:
            want = expected[check.check_id]
            held = (check.state, check.value) == want
            results.append(
                (
                    f"{scenario} {check.check_id}",
                    held,
                    f"{check.state.value} {check.value} (expected {want[0].value} {want[1]})",
                )
            )
    return results


def _states(checks: list) -> dict[str, tuple[str, str | None]]:
    return {
        item.check_id: (
            item.state.value if isinstance(item, Check) else "not_applicable",
            item.reason,
        )
        for item in checks
    }


def input_reader_results(directory: Path) -> list[tuple[str, bool, str]]:
    def inputs(name: str, *documents: dict) -> Path:
        root = directory / name
        for document in documents:
            fixtures.write_input(root, document)
        return root

    def registry_states(name: str, *, edit: bool) -> list[Check]:
        root = fixtures.build_registry(directory / name)
        if edit:
            record = next((root.parent / "runs").iterdir()) / "record.json"
            record.write_text(record.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return registry.registry_checks(root, now=SMOKE_AS_OF.to_pydatetime, allow_synthetic=True)

    approved = fixtures.training_health()
    approved["transition_status"] = "approved"
    point_only = fixtures.registry_export()
    point_only["forecast_evaluation"] = point_only["forecast_evaluation"][1:]
    stock_lab_broken = fixtures.stock_lab_monitor()
    stock_lab_broken["run_id"] = "synthetic-stock-lab-broken"
    stock_lab_broken["job"].update(status="failed", reason="the price update timed out")
    stock_lab_broken["freshness"][1]["sessions_behind"] = 2
    stock_lab_broken["scores"]["n"] = 12
    stock_lab_broken["inputs_drift"][0]["psi"] = 0.4
    stock_lab_broken["ledger"].update(verified=False, reason="entry 7 does not match its hash")
    breach = fixtures.backtest_summary()
    breach["measures"]["max_weight"] = 0.55
    healthy = ("healthy", None)
    unapproved = ("degraded", "unapproved_transition")
    missing = ("insufficient_evidence", "input_missing")
    scenarios: list[tuple[str, Callable[[], list], dict]] = [
        (
            "training, unapproved transition",
            lambda: training.training_checks(
                inputs("t1", fixtures.training_health()) / training.ARTIFACT, allow_synthetic=True
            ),
            {"training.health": unapproved},
        ),
        (
            "training, approved transition",
            lambda: training.training_checks(
                inputs("t2", approved) / training.ARTIFACT, allow_synthetic=True
            ),
            {"training.health": healthy},
        ),
        (
            "registry",
            lambda: registry_states("r1", edit=False),
            {check_id: healthy for check_id in registry.CHECKS},
        ),
        (
            "registry with an edited run record",
            lambda: registry_states("r2", edit=True),
            {
                "registry.entries": ("failed", "artifacts_unverified"),
                "registry.stages": healthy,
                "registry.selection_history": ("failed", "active_unverified"),
                "registry.experiments": healthy,
            },
        ),
        (
            "registry export",
            lambda: forecast.forecast_checks(
                inputs("f1", fixtures.registry_export()) / forecast.ARTIFACT, allow_synthetic=True
            ),
            {check_id: healthy for check_id in forecast.CHECKS},
        ),
        (
            "point-only registry export",
            lambda: forecast.forecast_checks(
                inputs("f2", point_only) / forecast.ARTIFACT, allow_synthetic=True
            ),
            {check_id: ("not_applicable", "point_forecast") for check_id in forecast.CHECKS},
        ),
        (
            "backtest with a cap breach",
            lambda: backtest.backtest_checks(
                inputs("b1", breach) / backtest.ARTIFACT, allow_synthetic=True
            ),
            {
                **{check_id: unapproved for check_id in backtest.MEASURES},
                "portfolio.concentration": ("failed", CONSTRAINT_BREACHED),
            },
        ),
        (
            "stock lab monitor",
            lambda: stock_lab.stock_lab_checks(
                inputs("s1", fixtures.stock_lab_monitor()) / stock_lab.ARTIFACT,
                allow_synthetic=True,
            ),
            {check_id: healthy for check_id in stock_lab.TITLES},
        ),
        (
            "stock lab with a failed update, a stale stock and a broken ledger",
            lambda: stock_lab.stock_lab_checks(
                inputs("s2", stock_lab_broken) / stock_lab.ARTIFACT, allow_synthetic=True
            ),
            {
                "data.stock_lab_freshness": ("degraded", "stale_prices"),
                "forecast.stock_lab_job": ("failed", "job_failed"),
                "forecast.stock_lab_scores": ("insufficient_evidence", "too_few_scored"),
                "forecast.stock_lab_drift": ("degraded", "input_drift"),
                "forecast.stock_lab_ledger": ("failed", "ledger_broken"),
            },
        ),
        (
            "absent inputs",
            lambda: (
                stock_lab.stock_lab_checks(directory / "absent" / stock_lab.ARTIFACT)
                + training.training_checks(directory / "absent" / training.ARTIFACT)
                + registry.registry_checks(directory / "absent", now=SMOKE_AS_OF.to_pydatetime)
                + forecast.forecast_checks(directory / "absent" / forecast.ARTIFACT)
                + backtest.backtest_checks(directory / "absent" / backtest.ARTIFACT)
            ),
            {
                check_id: missing
                for check_id in (
                    "training.health",
                    *registry.CHECKS,
                    *forecast.CHECKS,
                    *backtest.MEASURES,
                    *stock_lab.TITLES,
                )
            },
        ),
        (
            "unreadable file",
            lambda: training.training_checks(
                inputs("u1", {"artifact": "other", "run_id": "broken"}) / "other"
            ),
            {"training.health": ("unavailable", "unrecognized_format")},
        ),
    ]
    results = []
    for name, run, expected in scenarios:
        got = _states(run())
        held = got == expected
        wrong = {key: value for key, value in got.items() if expected.get(key) != value}
        results.append((name, held, "as expected" if held else f"unexpected: {wrong}"))
    return results


def _smoke_check(
    check_id: str, title: str, subject: str, results: list, measured_at: datetime
) -> Check:
    held = sum(ok for _, ok, _ in results)
    missed = [name for name, ok, _ in results if not ok]
    if missed:
        state, reason = HealthState.FAILED, "monitor_misreported"
        summary = f"{len(missed)} of {len(results)} {subject} gave the wrong result."
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"All {len(results)} {subject} gave the expected result."
    return Check(
        check_id=check_id,
        area=Area.SMOKE,
        title=title,
        state=state,
        reason=reason,
        measured_at=measured_at,
        value=held,
        unit=f"of {len(results)} expectations met",
        threshold="every expectation met",
        better=Better.HIGHER,
        tolerance=0,
        summary=summary,
        detail="; ".join(f"[{'ok' if ok else 'WRONG'}] {n}: {d}" for n, ok, d in results),
        evidence="fixture:stock_agent.health.smoke",
    )


def smoke_checks(*, now: Callable[[], datetime]) -> list[Check]:
    data = _smoke_check(
        "smoke.data_monitors",
        "Data monitors on planted defects",
        "data-monitor expectations (clean and planted synthetic stores)",
        data_monitor_results(),
        now(),
    )
    with tempfile.TemporaryDirectory(prefix="health-smoke-") as directory:
        readers = input_reader_results(Path(directory))
    inputs = _smoke_check(
        "smoke.input_readers",
        "Producer-file readers on fixtures",
        "producer-file reader expectations",
        readers,
        now(),
    )
    return [data, inputs]
