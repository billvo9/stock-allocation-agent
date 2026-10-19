"""
Fixtures for the premarket briefing: a Stock Lab store on a simulated clock
holding synthetic forecasts, scores, job runs and prices. Demo identifiers
contain "_", which no US ticker can, and every such record is synthetic.
The few "real"-class records that some tests need, to prove a rule that
applies only to real data, use the made-up ticker SYNTH and are never
rendered as an example. No network, no private store.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_agent.calendars.nyse_rules import sessions as nyse_sessions
from stock_agent.stock_lab import forecast as artifact
from stock_agent.stock_lab.briefing.report import ReadOnlyRecords, build_briefing
from stock_agent.stock_lab.briefing.timing import briefing_timing
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.replay import SimulatedClock
from stock_agent.stock_lab.scores import brier, log_loss, rps
from stock_agent.stock_lab.store import DB_NAME, StockLabStore, stamp
from stock_agent.stock_lab.symbols import parse_tickers

NEW_YORK = ZoneInfo("America/New_York")
CALENDAR_ID = "test_nyse_rules"
REAL_TEST_TICKER = "SYNTH"


def ny(day: str, hhmm: str) -> pd.Timestamp:
    """A New York wall-clock time, in UTC."""

    hour, minute = (int(part) for part in hhmm.split(":"))
    local = datetime.combine(date.fromisoformat(day), datetime.min.time(), tzinfo=NEW_YORK)
    return pd.Timestamp(local.replace(hour=hour, minute=minute)).tz_convert("UTC")


def next_sessions(as_of: str, count: int = 5, calendar=nyse_sessions) -> list[str]:
    start = date.fromisoformat(as_of)
    days = [pd.Timestamp(d).date() for d in calendar(start, start + pd.Timedelta(days=20))]
    return [d.isoformat() for d in days if d > start][:count]


class Lab:
    """A store under `tmp_path`, written in time order on a simulated clock."""

    def __init__(self, tmp_path: Path) -> None:
        self.clock = SimulatedClock(pd.Timestamp("2000-01-03", tz="UTC"))
        self.path = tmp_path / "lab" / DB_NAME
        self.store = StockLabStore(self.path, clock=self.clock)

    def at(self, when: pd.Timestamp) -> None:
        self.clock.advance(pd.Timestamp(when))

    def watch(self, *tickers: str, at: pd.Timestamp | None = None) -> None:
        if at is not None:
            self.at(at)
        self.store.add(parse_tickers(" ".join(tickers)))

    def prices(self, symbol: str, bars: pd.DataFrame, *, at: pd.Timestamp) -> None:
        self.at(at)
        kind = DataClass.SYNTHETIC if "_" in symbol else DataClass.REAL
        self.store.record_prices(
            symbol, bars, source="fixture", retrieved_at=at, data_class=kind, vintage_note="test"
        )

    def forecast(
        self,
        symbol: str,
        as_of: str,
        *,
        created_at: pd.Timestamp | None = None,
        recorded_at: pd.Timestamp | None = None,  # the store's clock (default: created_at)
        targets: list[str] | None = None,
        p=(0.3, 0.3, 0.4),  # Decrease, Stable, Increase for every session, or five triples
        cumulative=(0.35, 0.2, 0.45),
        estimator: str = "B1",
        band: float = 0.005,
        signals: dict | None = None,
        refusal: dict | None = None,
        supersedes: str | None = None,
        evaluation_run_id: str | None = None,
        model_version: str = "fixture-model",
    ) -> dict:
        created_at = created_at if created_at is not None else ny(as_of, "21:05")
        targets = targets or next_sessions(as_of)
        kind = "synthetic" if "_" in symbol else "real"
        record = artifact.build(
            symbol=symbol,
            data_class=kind,
            as_of_session=as_of,
            created_at=stamp(created_at),
            target_sessions=targets,
            daily=None if refusal else ([list(p)] * 5 if len(p) == 3 else [list(row) for row in p]),
            cumulative=None if refusal else list(cumulative),
            band=band,
            cumulative_band=band * 5**0.5,
            estimator=estimator,
            model_version=model_version,
            feature_version="stock_lab_features_v1",
            evaluation_run_id=evaluation_run_id,
            data={"source": "fixture", "retrieved_at": stamp(created_at - pd.Timedelta(minutes=5))},
            evidence_status="synthetic_demo" if kind == "synthetic" else "experimental_unvalidated",
            calibration_status="fixture",
            signals=signals if signals is not None else {},
            limitations=["Synthetic fixture."] if kind == "synthetic" else ["Test fixture."],
            refusal=refusal,
            supersedes=supersedes,
        )
        self.at(recorded_at if recorded_at is not None else created_at)
        self.store.record_forecast(record, real_authorized=kind == "real")
        return record

    def score(self, record: dict, target: str, outcome: int, *, at: pd.Timestamp) -> None:
        row = record["cumulative"] if target == "cum" else record["daily"][int(target) - 1]
        p = np.array([[row["p_decrease"], row["p_stable"], row["p_increase"]]])
        y = np.array([outcome])
        session = record["target_sessions"][-1 if target == "cum" else int(target) - 1]
        self.at(at)
        self.store.record_scores(
            [
                {
                    "forecast_id": record["forecast_id"],
                    "target": target,
                    "symbol": record["symbol"],
                    "session": session,
                    "status": "scored",
                    "reason": None,
                    "outcome": outcome,
                    "realized_return": 0.01 if outcome == 2 else -0.01 if outcome == 0 else 0.0,
                    "brier": float(brier(p, y)[0]),
                    "log_loss": float(log_loss(p, y)[0]),
                    "rps": float(rps(p, y)[0]),
                    "batch_id": None,
                }
            ]
        )

    def job(
        self,
        *,
        at: pd.Timestamp,
        expected: str,
        status: str = "skipped",
        activation: str | None = "not_authorized",
        data_class: str = "real",
    ) -> None:
        self.at(at)
        self.store.record_job_run(
            {
                "run_id": f"job-{stamp(at)}",
                "started_at": stamp(at - pd.Timedelta(minutes=1)),
                "finished_at": stamp(at),
                "status": status,
                "reason": None,
                "expected_session": expected,
                "data_class": data_class,
                "activation": activation,
            }
        )


def build(
    lab: Lab,
    now: pd.Timestamp,
    *,
    data_class: str = "synthetic",
    tickers=None,
    calendar=nyse_sessions,
    **options,
):
    plan = briefing_timing(now.to_pydatetime(), calendar)
    return build_briefing(
        ReadOnlyRecords(lab.path),
        timing=plan,
        calendar=calendar,
        calendar_id=CALENDAR_ID,
        data_class=data_class,
        tickers=tickers,
        **options,
    )
