"""
Evaluate the Stock Lab's estimators walk-forward and record the result
(local, no network).

    --demo   the three synthetic demo stocks: the evaluation, then one
             synthetic demo forecast per stock as of the last synthetic
             session (target sessions are the next synthetic weekdays). Both
             are stored, labelled synthetic, in the private Stock Lab store,
             where the Lab's Stock Probability page reads them. They are
             recorded on a simulated store clock just after that session's
             close (21:01 and 21:05 UTC), because a forecast must be made
             before its first target session opens, and the store refuses a
             backdated one. A second run is a recalculation: each new demo
             forecast supersedes the live one.
    --real   the saved watchlist's stored real prices through 2024-12-31,
             with every outcome censored there. This is a real-stock scoring
             experiment on a hindsight-selected list, so it needs the
             owner's decision A3 (docs/research/stock_probability_lab.md):
             the script refuses unless --authorized-evaluation is also
             passed. It records the evaluation only: no real forecast is made
             (that needs decisions A2 and A4).

The evaluation is written once (a second run is a new record); nothing
earlier is overwritten. Prints the selection and the main scores only.

Usage:
    .venv/bin/python scripts/stock_lab_evaluate.py --demo
    .venv/bin/python scripts/stock_lab_evaluate.py --real --authorized-evaluation
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import TextIO

import pandas as pd

from stock_agent.stock_lab.contracts import DEVELOPMENT_END, DataClass
from stock_agent.stock_lab.evaluation import EvaluationConfig, build_panel, evaluate
from stock_agent.stock_lab.predict import fit_final, forecast_stock
from stock_agent.stock_lab.store import StockLabStore, stamp, utc_now
from stock_agent.stock_lab.synthetic import demo_histories
from stock_agent.stock_lab.view import real_bars_shown


def run_id(prefix: str, now: pd.Timestamp) -> str:
    return f"{prefix}-{now.strftime('%Y%m%dT%H%M%S%fZ')}"


def _print_summary(record: dict, out: TextIO) -> None:
    metrics = pd.DataFrame(record["metrics"])
    daily = metrics[metrics["target"].str.match(r"target_\d")]
    rps = daily.groupby("model")["rps"].mean()
    print(f"evaluation {record['run_id']}: {len(record['folds'])} folds", file=out)
    for model, value in rps.items():
        print(f"  {model}: mean RPS over the five daily targets {value:.5f}", file=out)
    selection = record["selection"]
    print(f"selected: {selection['selected']} ({selection['rule']})", file=out)


def main(
    argv: Sequence[str] | None = None,
    *,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    clock: Callable[[], pd.Timestamp] = utc_now,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--demo", action="store_true")
    which.add_argument("--real", action="store_true")
    parser.add_argument("--authorized-evaluation", action="store_true")
    args = parser.parse_args(argv)
    store = store_factory()
    now = clock()

    if args.real:
        if not args.authorized_evaluation:
            print(
                "refused: evaluating real stocks needs the owner's decision A3 "
                "(pass --authorized-evaluation once it is recorded)",
                file=out,
            )
            return 2
        histories = {}
        for ticker in store.watched():
            bars = real_bars_shown(store.bars(ticker))
            if bars is not None and len(bars) > 1000:
                histories[ticker] = bars
        if not histories:
            print("nothing to evaluate: no watched stock has enough stored history", file=out)
            return 1
        config = EvaluationConfig(outcome_end=pd.Timestamp(DEVELOPMENT_END, tz="UTC"))
        record = evaluate(
            histories,
            config,
            data_class=DataClass.REAL.value,
            run_id=run_id("R", now),
            created_at=now,
        )
        store.record_evaluation(record)
        _print_summary(record, out)
        print("no forecast recorded: real forecasts need decisions A2 and A4", file=out)
        return 0

    histories = dict(demo_histories())
    final = max(bars["session"].iloc[-1] for bars in histories.values())
    made = final + pd.Timedelta(hours=21, minutes=5)  # synthetic: just after the last close
    store = StockLabStore(store.path, clock=lambda: made)
    config = EvaluationConfig()
    record = evaluate(
        histories,
        config,
        data_class=DataClass.SYNTHETIC.value,
        run_id=run_id("S", now),
        created_at=final + pd.Timedelta(hours=21, minutes=1),
    )
    store.record_evaluation(record)
    _print_summary(record, out)
    panel = build_panel(histories, config)
    model = fit_final(panel, config, record)
    for stock_id, bars in histories.items():
        last = bars["session"].iloc[-1]
        sessions = pd.bdate_range(last + pd.Timedelta(days=1), periods=5)  # synthetic weekdays
        previous = store.live_forecast(stock_id, last.strftime("%Y-%m-%d"))
        forecast = forecast_stock(
            panel,
            stock_id,
            model,
            record,
            target_sessions=[day.strftime("%Y-%m-%d") for day in sessions],
            created_at=made,
            data={
                "source": "synthetic_demo",
                "batch_ids": [],
                "retrieved_at": stamp(made),
                "last_close_session": last.strftime("%Y-%m-%d"),
            },
            data_class=DataClass.SYNTHETIC.value,
            supersedes=None if previous is None else previous["forecast_id"],
        )
        store.record_forecast(forecast)
        print(f"{stock_id}: synthetic demo forecast {forecast['forecast_id'][:12]}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
