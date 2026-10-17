"""
A synthetic replay of the daily job (research side; no network, synthetic
data only): it shows the whole forecast-record-score loop working before any
real stock is involved, without ever letting a model see the sessions it is
scored on.

The demo stocks' last `sessions` synthetic sessions play the "future":

    1. Before the replay, only the history through the replay's first as-of
       session is stored (a synthetic batch "retrieved" just after that
       close), and the estimators are evaluated and frozen with every
       outcome censored at that session (`EvaluationConfig.outcome_end`).
    2. The first forecasts are made as of that session. Each replay day then
       reveals the next session as a new synthetic batch, scores every
       forecast whose target sessions have now closed (only on batches
       retrieved after the forecast was created; scoring.py), and makes the
       day's forecasts from the new close, as a real day would.
    3. The job run, the ledger and the Model Health monitor input are
       recorded like a real day's.

Every record is labelled synthetic; the simulated clock is synthetic too
(each session's 21:00 UTC). The store itself runs on that clock
(`SimulatedClock`), so every record carries the simulated time at which
the store wrote it, exactly as a real day's records carry the real time.
No record is backdated: the store refuses that. Each day's forecast is a
new forecast for a new as-of session, so it supersedes nothing. Nothing
here is evidence about any real stock.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.evaluation import EvaluationConfig, build_panel, evaluate
from stock_agent.stock_lab.predict import fit_final, forecast_stock
from stock_agent.stock_lab.store import StockLabStore, stamp

SOURCE = "synthetic_replay"
NOTE = "synthetic replay: generated sessions revealed one per simulated day"


@dataclass(frozen=True)
class ReplayDay:
    as_of: str
    forecasts: int
    scored: int


class ReplayRefused(ValueError):
    """The store already holds replay records; a replay never writes over or beside them."""


class SimulatedClock:
    """The replay's store clock: a synthetic time the replay moves forward, never back."""

    def __init__(self, start: pd.Timestamp) -> None:
        self.now = pd.Timestamp(start)

    def __call__(self) -> pd.Timestamp:
        return self.now

    def advance(self, to: pd.Timestamp) -> None:
        if to < self.now:
            raise ValueError("the simulated clock never moves back")
        self.now = pd.Timestamp(to)


def _weekdays(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(first, last, tz="UTC").as_unit("us")


def _clock(session: pd.Timestamp, minutes: int) -> pd.Timestamp:
    return session + pd.Timedelta(hours=21, minutes=minutes)


def run_replay(
    store: StockLabStore,
    *,
    sessions: int = 20,
    record_day: Callable[..., None] | None = None,
) -> tuple[ReplayDay, ...]:
    """
    Replay the last `sessions` synthetic sessions (module docstring).
    `record_day` receives the replay's store, which runs on the simulated clock.
    """

    from stock_agent.stock_lab import scoring  # the scorer reads only stored records

    if any(store.batches(stock_id) for stock_id in synthetic.DEMO_IDS):
        raise ReplayRefused(
            "this store already holds synthetic demo prices; records are write-once, so a "
            "second replay would collide with the first: use a new STOCK_AGENT_STOCK_LAB_HOME"
        )
    full = dict(synthetic.demo_histories())
    any_bars = next(iter(full.values()))
    first = len(any_bars) - sessions - 1
    start = any_bars["session"].iloc[first]
    clock = SimulatedClock(_clock(start, 0))
    store = StockLabStore(store.path, clock=clock)
    for stock_id, bars in full.items():
        store.record_prices(
            stock_id,
            bars.iloc[: first + 1],
            source=SOURCE,
            retrieved_at=_clock(start, 0),
            data_class=DataClass.SYNTHETIC,
            vintage_note=NOTE,
        )
    clock.advance(_clock(start, 1))
    config = EvaluationConfig(outcome_end=start)
    history = {stock_id: bars.iloc[: first + 1] for stock_id, bars in full.items()}
    record = evaluate(
        history,
        config,
        data_class=DataClass.SYNTHETIC.value,
        run_id=f"S-replay-{start.strftime('%Y%m%d')}",
        created_at=_clock(start, 1),
    )
    store.record_evaluation(record)
    model = fit_final(build_panel(history, config), config, record)

    def forecast_all(index: int) -> int:
        as_of = any_bars["session"].iloc[index]
        clock.advance(_clock(as_of, 5))
        visible = {stock_id: bars.iloc[: index + 1] for stock_id, bars in full.items()}
        panel = build_panel(visible, config)
        for stock_id in full:
            targets = pd.bdate_range(as_of + pd.Timedelta(days=1), periods=5)
            forecast = forecast_stock(
                panel,
                stock_id,
                model,
                record,
                target_sessions=[d.strftime("%Y-%m-%d") for d in targets],
                created_at=_clock(as_of, 5),
                data={
                    "source": SOURCE,
                    "batch_ids": [b.batch_id for b in store.batches(stock_id)][-1:],
                    "retrieved_at": stamp(_clock(as_of, 0)),
                    "last_close_session": as_of.strftime("%Y-%m-%d"),
                },
                data_class=DataClass.SYNTHETIC.value,
            )
            store.record_forecast(forecast)
        return len(full)

    days = [ReplayDay(start.strftime("%Y-%m-%d"), forecast_all(first), 0)]
    for step in range(1, sessions + 1):
        index = first + step
        session = any_bars["session"].iloc[index]
        clock.advance(_clock(session, 0))
        for stock_id, bars in full.items():  # the day's close arrives
            store.record_prices(
                stock_id,
                bars.iloc[index : index + 1],
                source=SOURCE,
                retrieved_at=_clock(session, 0),
                data_class=DataClass.SYNTHETIC,
                vintage_note=NOTE,
            )
        clock.advance(_clock(session, 2))
        scored = scoring.score_store(store, list(full), calendar=_weekdays)  # closed only
        made = forecast_all(index)  # then today's forecasts, from today's close
        days.append(ReplayDay(session.strftime("%Y-%m-%d"), made, scored))
        if record_day is not None:
            clock.advance(_clock(session, 11))
            record_day(as_of=session, store=store, record=record, model=model)
    return tuple(days)
