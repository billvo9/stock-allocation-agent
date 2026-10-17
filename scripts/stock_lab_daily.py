"""
The Stock Lab's daily job (owner-run, private, local): score the forecasts
whose sessions have closed, record the run and its Model Health input, and,
once the owner's decisions allow it, update prices and make the day's
forecasts. It never fabricates a forecast: a stock whose newest close is
not the expected session keeps its last valid forecast, labelled stale.

Modes
    --demo-replay N   a synthetic replay of N sessions (no network): the
                      whole forecast-record-score loop on the demo stocks
                      (stock_lab.replay), each day recorded like a real one.
    (no mode)         the real job, step by step:
        1. the expected session: the last NYSE session whose close is at
           least AVAILABILITY (20:00 America/New_York) old, from the rules
           calendar the script passes in;
        2. prices for the recent window (decisions A1 and A2);
        3. scoring of stored forecasts (outcomes retrieved after each
           forecast only);
        4. forecasts for stocks whose newest close is the expected session
           (decision A4), never for a stale one;
        5. the job run, the ledger and the `stock_lab_monitor` health input
           (written to --health-inputs, default reports/health_inputs/).
       Until the owner records A1, A2 and A4, steps 2 and 4 are not built: the
       job does steps 1, 3 and 5, records itself as skipped with the reason,
       and refuses the flags --authorized, --inference-window and
       --record-real.

Scheduling: run it once a day after the availability time, for example with
the launchd agent printed by `scripts/stock_lab_schedule.py` (the owner
installs it; nothing here installs anything).

Usage:
    .venv/bin/python scripts/stock_lab_daily.py --demo-replay 20
    .venv/bin/python scripts/stock_lab_daily.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.stock_lab import scoring
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.job import record_run
from stock_agent.stock_lab.replay import run_replay
from stock_agent.stock_lab.store import StockLabStore, stamp, utc_now
from stock_agent.stock_lab.synthetic import DEMO_IDS

REPLAY_CALENDAR = "synthetic_weekdays"

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_HEALTH_INPUTS = PROJECT / "reports" / "health_inputs"
AVAILABILITY = "20:00"  # America/New_York: the job's earliest run time for a session's close


def expected_session(now: pd.Timestamp) -> pd.Timestamp:
    """The last session whose close is at least AVAILABILITY old in New York at `now`."""

    local = now.tz_convert("America/New_York")
    hour, minute = (int(part) for part in AVAILABILITY.split(":"))
    today = pd.Timestamp(local.date())
    candidates = sessions(today - pd.Timedelta(days=10), today)
    if len(candidates) and candidates[-1].date() == local.date():
        cutoff = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if local < cutoff:
            candidates = candidates[:-1]
    return candidates[-1]


def next_expected(now: pd.Timestamp) -> str:
    local = now.tz_convert("America/New_York")
    upcoming = sessions(pd.Timestamp(local.date()) + pd.Timedelta(days=1),
                        pd.Timestamp(local.date()) + pd.Timedelta(days=10))  # fmt: skip
    hour, minute = (int(part) for part in AVAILABILITY.split(":"))
    nxt = pd.Timestamp(upcoming[0].date(), tz="America/New_York").replace(hour=hour, minute=minute)
    return stamp(nxt.tz_convert("UTC"))


def code_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = out.stdout.strip()
    return text if out.returncode == 0 and len(text) == 40 else None


def main(
    argv: Sequence[str] | None = None,
    *,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    clock: Callable[[], pd.Timestamp] = utc_now,
    out: TextIO = sys.stdout,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--demo-replay", type=int, metavar="N")
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--inference-window", action="store_true")
    parser.add_argument("--record-real", action="store_true")
    parser.add_argument("--health-inputs", type=Path, default=DEFAULT_HEALTH_INPUTS)
    args = parser.parse_args(argv)
    store = store_factory()
    started = clock()

    if args.demo_replay is not None:
        if not 1 <= args.demo_replay <= 60:
            print("refused: --demo-replay takes 1 to 60 sessions", file=out)
            return 2

        def weekdays(start, end):
            return pd.bdate_range(start, end, tz="UTC")

        def record_day(*, as_of, store, record, model):
            day = pd.Timestamp(as_of)
            record_run(
                store,
                run_id=f"replay-{day.strftime('%Y%m%d')}",
                started=day + pd.Timedelta(hours=21, minutes=10),
                finished=day + pd.Timedelta(hours=21, minutes=11),
                status="ok",
                reason=None,
                expected=day,
                tickers=list(DEMO_IDS),
                data_class=DataClass.SYNTHETIC,
                health_inputs=args.health_inputs,
                sessions=weekdays,
                calendar_id=REPLAY_CALENDAR,
                code_sha=code_sha(),
            )

        days = run_replay(store, sessions=args.demo_replay, record_day=record_day)
        scored = sum(day.scored for day in days)
        print(f"synthetic replay: {len(days)} sessions, {scored} targets scored", file=out)
        return 0

    if args.authorized or args.inference_window or args.record_real:
        print(
            "refused: the real price update and forecasts are built after the owner records "
            "decisions A1, A2 and A4 (docs/research/stock_probability_lab.md, section 7)",
            file=out,
        )
        return 2
    tickers = store.watched()
    expected = expected_session(started)
    skipped = [
        "prices: the recent-window update needs decisions A1 and A2",
        "forecasts: recording real forecasts needs decision A4",
    ]
    # Step 3: scoring runs on whatever is stored (real forecasts exist only after A4).
    scored = scoring.score_store(store, tickers)
    status, reason = "skipped", "; ".join(skipped)
    if args.authorized or args.inference_window or args.record_real:
        print(
            "refused: the real price update and forecasts are not implemented until the owner "
            "records decisions A1, A2 and A4",
            file=out,
        )
        return 2
    finished = clock()
    path = record_run(
        store,
        run_id=f"job-{started.strftime('%Y%m%dT%H%M%S%fZ')}",
        started=started,
        finished=finished,
        status=status,
        reason=reason,
        expected=expected,
        tickers=tickers,
        data_class=DataClass.REAL,
        health_inputs=args.health_inputs,
        sessions=lambda start, end: sessions(start.date(), end.date()),
        calendar_id=CALENDAR_ID,
        code_sha=code_sha(),
        next_expected_at=next_expected(finished),
        skipped=skipped,
    )
    print(f"expected session {expected.date()}; {scored} targets scored; {status}", file=out)
    for line in skipped:
        print(f"  skipped {line}", file=out)
    print(f"health input: {path}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
