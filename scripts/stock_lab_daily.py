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
           (written to --health-inputs, default
           ~/stock-agent-private/health_inputs/: outside every checkout, so
           Model Health reads it whichever checkout runs; the demo replay's
           synthetic files go to reports/health_inputs/), with the kept
           ledger head check (stock_lab.job): the head hash is kept in
           --ledger-head-backup, by default stock_lab_ledger_head beside the
           store (a copy kept elsewhere is stronger).
       Without --record-real the job does steps 1, 3 and 5, records itself as
       skipped and not yet authorized (Model Health reads that as not
       activated, never as a degraded job; a run that stops still shows as
       overdue). The price update of step 2 is the separate, owner-run
       scripts/stock_lab_fetch_recent.py --authorized --store, run first; this
       job refuses --authorized and --inference-window.
    --dry-run --model ID
                      the real path, recording nothing: it loads and verifies
                      the registered frozen model ID (its id or an unambiguous
                      prefix of at least 16 characters), reads each watched
                      stock's 252-session inference window as of the expected
                      session at this moment (only between that close's 20:00
                      and the next 09:30 New York), computes the forecasts with
                      B1 beside them and what scoring would write, and prints
                      them. The end-to-end dry run that activation needs.
    --record-real --model ID
                      the same, recorded: the scores (outcomes from the
                      inference store, stock_lab.evening), the forecasts
                      (write-once; a stale stock keeps its last forecast), then
                      every missed evening reconstructed apart from them (as
                      --reconstruct-missed), the job run and its Model Health
                      input, with the prospective record's missed evenings
                      and those reconstructed (stock_lab.job.evening_record). Refused unless the
                      owner activated that model (decision A4:
                      evening.ACTIVATED_MODELS, only the models the owner
                      explicitly activated).
    --reconstruct-missed --model ID
                      the manual one-shot reconstruction (owner decision of
                      2026-10-30, replacing decision A4's "missed sessions are
                      never backfilled" for the activated model): every missed
                      (stock, session) since the first live forecast,
                      reconstructed from prices dated up to that close and
                      recorded apart from the live record (stock_lab.evening,
                      "Reconstructing missed evenings"), then the
                      reconstructions scored (live forecasts are scored by the
                      evening run only). It records no job run. --record-real does the same after each
                      evening's live forecasts, so this is only for an evening
                      that cannot wait for the next run. Exit 0 nothing refused,
                      1 a reconstruction refused (tried again next time), 2 the
                      model.
    --check-environment --model ID
                      read-only: this Python's minor version, numpy, pandas
                      and scikit-learn beside the frozen model ID's recorded
                      environment, which the real path requires exactly (it
                      refuses the model otherwise). Run it with the venv that
                      runs the evening, before the evening (docs/operations/
                      runbook.md, "The runner checkout"). Exit 0 the same,
                      1 different, 2 no such model.

Scheduling: the unattended evening (`scripts/stock_lab_schedule.py`, which
prints the launchd agent and its run.sh) runs
`scripts/stock_lab_fetch_recent.py --authorized --store --keychain --unattended`
and, only when that succeeds, `--record-real --model ID`. The owner installs it;
nothing here installs anything. A forecast that the store refuses as
backdated (the Mac slept between the decision and the write) is made again,
once, at the time of its write, from its window as of then; no write starts
at or after the next open (stock_lab.evening, "Timing").

Usage:
    .venv/bin/python scripts/stock_lab_daily.py --demo-replay 20
    .venv/bin/python scripts/stock_lab_daily.py
    .venv/bin/python scripts/stock_lab_daily.py --dry-run --model ID
    .venv/bin/python scripts/stock_lab_daily.py --record-real --model ID   # after A4
    .venv/bin/python scripts/stock_lab_daily.py --check-environment --model ID
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TextIO

import pandas as pd

from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.stock_lab import scoring
from stock_agent.stock_lab.contracts import DataClass
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.job import NOT_AUTHORIZED, evening_record, record_run
from stock_agent.stock_lab.monitoring import shared_inputs
from stock_agent.stock_lab.replay import ReplayRefused, run_replay
from stock_agent.stock_lab.store import StockLabStore, StoreRefused, stamp, utc_now
from stock_agent.stock_lab.synthetic import DEMO_IDS

REPLAY_CALENDAR = "synthetic_weekdays"
KEPT_HEAD_NAME = "stock_lab_ledger_head"

PROJECT = Path(__file__).resolve().parents[1]
# The synthetic demo's health inputs stay in this checkout; the real job's go to
# monitoring.shared_inputs() (module docstring).
DEMO_HEALTH_INPUTS = PROJECT / "reports" / "health_inputs"
AVAILABILITY = "20:00"  # America/New_York: the job's earliest run time for a session's close


def expected_session(now: pd.Timestamp, calendar=sessions) -> pd.Timestamp:
    """The last session whose close is at least AVAILABILITY old in New York at `now`."""

    local = now.tz_convert("America/New_York")
    hour, minute = (int(part) for part in AVAILABILITY.split(":"))
    today = pd.Timestamp(local.date())
    candidates = calendar(today - pd.Timedelta(days=10), today)
    if len(candidates) and candidates[-1].date() == local.date():
        cutoff = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if local < cutoff:
            candidates = candidates[:-1]
    return candidates[-1]


def next_expected(now: pd.Timestamp, calendar=sessions) -> str:
    """The next time a new close should be available: today's AVAILABILITY on a session
    day before it, else the next session's."""

    local = now.tz_convert("America/New_York")
    hour, minute = (int(part) for part in AVAILABILITY.split(":"))
    today = pd.Timestamp(local.date())
    upcoming = calendar(today, today + pd.Timedelta(days=10))
    for day in upcoming:
        due = pd.Timestamp(day.date(), tz="America/New_York").replace(hour=hour, minute=minute)
        if due > local:
            return stamp(due.tz_convert("UTC"))
    raise CalendarRefused("outside_coverage", "no session in the next ten days")


def calendar_sessions(first: pd.Timestamp, last: pd.Timestamp) -> pd.DatetimeIndex:
    """NYSE sessions between two sessions (for scoring: a missing session is a gap)."""

    return sessions(first.date(), last.date())


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


MODEL_ID = re.compile(r"[0-9a-f]{16,64}")


def _resolve_model(store: StockLabStore, text: str | None) -> dict | None:
    """The one registered real frozen model whose id starts with `text` (>= 16 hex)."""

    if text is None or not MODEL_ID.fullmatch(text):
        return None
    found = [m for m in store.frozen_models(DataClass.REAL) if m["model_id"].startswith(text)]
    return found[0] if len(found) == 1 else None


def _percent(row: dict) -> str:
    return " ".join(
        f"{label} {row[key]:.0%}"
        for label, key in (("Incr.", "p_increase"), ("Stab.", "p_stable"), ("Decr.", "p_decrease"))
    )


def health_inputs(args: argparse.Namespace) -> Path:
    """--health-inputs when given, else the demo's folder in this checkout or the shared one."""

    if args.health_inputs is not None:
        return args.health_inputs
    return DEMO_HEALTH_INPUTS if args.demo_replay is not None else shared_inputs()


def check_environment(store: StockLabStore, text: str | None, out: TextIO) -> int:
    """--check-environment (module docstring): reads the store's frozen model only."""

    record = _resolve_model(store, text)
    if record is None:
        print(
            "refused: --model must name one registered real frozen model (its id, or an "
            "unambiguous prefix of at least 16 characters)",
            file=out,
        )
        return 2
    try:
        from stock_agent.stock_lab import registry  # the reviewed real path (decision A4)

        here = registry.environment()
    except ImportError as error:  # a package missing, or one that cannot load (metadata too)
        print(f"differs: this Python ({sys.executable}) cannot load the model: {error}", file=out)
        return 1
    frozen = record["environment"]  # always present: the store re-checks every record
    print(
        f"frozen model {record['model_id'][:16]}: its environment beside this Python "
        f"({sys.executable})",
        file=out,
    )
    differ = []
    for key in sorted(set(frozen) | set(here), key=lambda name: (name != "python", name)):
        same = frozen.get(key) == here.get(key)
        if not same:
            differ.append(key)
        print(
            f"  {key:<13} frozen {frozen.get(key, 'not recorded'):<10} here "
            f"{here.get(key, 'not checked'):<10} {'same' if same else 'DIFFERS'}",
            file=out,
        )
    if not differ:
        print("same: the evening can load this model with this Python", file=out)
        return 0
    needs = []
    if "python" in differ:
        needs.append(f"a venv made from Python {frozen.get('python', '(not recorded)')}")
    pins = " ".join(f"{key}=={frozen[key]}" for key in differ if key != "python" and key in frozen)
    if pins:
        needs.append(f"exactly {pins}")
    print(
        f"differs: {', '.join(differ)}; the evening would refuse this model "
        f"(environment_mismatch). It needs {' with '.join(needs) or 'the frozen environment'}, "
        "never newer versions; or re-freeze the model",
        file=out,
    )
    return 1


def _real(args, *, store, clock, started, out, inference_path, calendar, calendar_id, kept_head):
    """--dry-run and --record-real (module docstring)."""

    from stock_agent.stock_lab import evening, registry  # the reviewed real path (decision A4)

    record = _resolve_model(store, args.model)
    if record is None:
        print(
            "refused: --model must name one registered real frozen model (its id, or an "
            "unambiguous prefix of at least 16 characters)",
            file=out,
        )
        return 2
    if args.record_real or args.reconstruct_missed:
        try:
            evening.check_activated(record["model_id"])
        except evening.NotActivated as error:
            print(f"refused: {error}", file=out)
            return 2
    try:
        model = registry.load(record)
    except registry.RegistryRefused as error:
        print(f"refused: the frozen model does not load ({error.reason}: {error.detail})", file=out)
        return 2
    try:
        expected = expected_session(started, calendar)
    except CalendarRefused as error:
        print(f"refused: the NYSE calendar refused ({error})", file=out)
        return 2
    path = inference_path or evening.inference_path()
    tickers = store.watched()
    if args.reconstruct_missed:
        return _reconstruct_only(evening, store, model, tickers=tickers, clock=clock,
                                 expected=expected, calendar=calendar, calendar_id=calendar_id,
                                 path=path, out=out)  # fmt: skip
    write = bool(args.record_real)
    _, problem = store.verify_ledger()
    if problem is not None:  # a record changed outside the store: nothing more is written
        print(f"refused: the ledger does not verify ({problem})", file=out)
        if write:
            finished = clock()
            record_run(
                store, run_id=f"job-{started.strftime('%Y%m%dT%H%M%S%fZ')}", started=started,
                finished=finished, status="failed", reason=f"ledger: {problem}",
                expected=expected, tickers=tickers, data_class=DataClass.REAL,
                health_inputs=args.health_inputs,
                sessions=lambda first, last: calendar(first.date(), last.date()),
                calendar_id=calendar_id, code_sha=code_sha(),
                next_expected_at=next_expected(finished, calendar), kept_head=kept_head,
            )  # fmt: skip
        return 1
    symbols = sorted(set(tickers) | set(store.forecast_symbols(DataClass.REAL)))
    outcomes = evening.score_real(
        store, symbols=symbols, now=started, expected=expected, sessions=calendar,
        calendar_id=calendar_id, path=path, write=write, model_id=model.model_id,
        watched=tickers,
    )  # fmt: skip
    # The decision time is now; the clock remakes a forecast refused as backdated (a sleep).
    found = evening.forecast_watchlist(
        store, model, tickers=tickers, now=clock(), expected=expected, sessions=calendar,
        calendar_id=calendar_id, path=path, record=write, clock=clock,
    )  # fmt: skip
    mode = "recorded" if write else "dry run: nothing recorded"
    print(
        f"model {model.version} ({model.estimator}, B1 beside it); expected session "
        f"{expected.date()}; {mode}",
        file=out,
    )
    print(
        f"scores: {outcomes.scored} scored, {outcomes.unscorable} unscorable, {outcomes.pending} "
        f"pending ({outcomes.missing} past their session), {len(outcomes.refused)} refused",
        file=out,
    )
    for result in found.results:
        line = f"  {result.ticker}: {result.status}"
        if result.reason:
            line += f" ({result.reason}: {result.detail})"
        if result.forecast is not None and result.forecast["daily"]:
            first = result.forecast["daily"][0]
            b1 = result.forecast["comparator"]["daily"][0]
            line += f"; next session {first['session']}: {_percent(first)}; B1 {_percent(b1)}"
        print(line, file=out)
    if write:  # after the live forecasts: every missed evening, reconstructed apart from them
        _reconstruct(evening, store, model, clock=clock, expected=expected,
                     calendar=calendar, calendar_id=calendar_id, path=path, out=out)  # fmt: skip
    finished = clock()
    evenings = evening_record(
        store, now=finished, expected=expected,
        sessions=lambda first, last: calendar(first.date(), last.date()),
    )  # fmt: skip
    if evenings is not None:
        missed = ", ".join(evenings["missed"][-5:]) or "none"
        print(
            f"evenings since {evenings['since']}: {evenings['recorded']} of {evenings['due']} "
            f"closed evenings recorded; missed (latest 5): {missed}",
            file=out,
        )
    if not write:
        return 0
    done = [
        r for r in found.results if r.status == evening.RECORDED or r.reason == "already_recorded"
    ]
    trouble = [r for r in found.results if r not in done]
    refused = [r.ticker for r in trouble if r.status == evening.REFUSED]
    status = "ok"
    if trouble:
        status = "failed" if not done and refused else "partial"
    path_written = record_run(
        store,
        run_id=f"job-{started.strftime('%Y%m%dT%H%M%S%fZ')}",
        started=started,
        finished=finished,
        status=status,
        reason="; ".join(f"{r.ticker}: {r.reason}" for r in trouble) or None,
        expected=expected,
        tickers=tickers,
        data_class=DataClass.REAL,
        health_inputs=args.health_inputs,
        sessions=lambda first, last: calendar(first.date(), last.date()),
        calendar_id=calendar_id,
        code_sha=code_sha(),
        next_expected_at=next_expected(finished, calendar),
        failed=refused,
        skipped=[r.ticker for r in trouble if r.status != evening.REFUSED],
        kept_head=kept_head,
        measured_freshness=found.freshness,
        measured_drift=evening.input_drift(found.panel, model),
        results=[r.row() for r in found.results],
        outcomes=outcomes.as_dict(),
        evenings=evenings,
    )
    print(f"{status}; health input: {path_written}", file=out)
    return 0 if status != "failed" else 1


def _reconstruct(evening, store, model, *, clock, expected, calendar, calendar_id,
                 path, out):  # fmt: skip
    """
    Reconstruct the missed evenings and print them; True when none was refused. `evening` is
    the real path's module, which only _real loads (the isolation tests check it).
    """

    try:
        found = evening.reconstruct_missed(
            store, model, clock=clock, expected=expected, sessions=calendar,
            calendar_id=calendar_id, path=path,
        )  # fmt: skip
    except (ForecastRefused, StoreRefused) as error:  # a stored record failed its check
        print(f"reconstruction skipped: a stored record does not verify ({error})", file=out)
        return False
    if not found:
        print("missed evenings: nothing to reconstruct", file=out)
        return True
    for day in sorted({r.session for r in found}):
        made = [r.ticker for r in found if r.session == day and r.status == evening.RECORDED]
        if made:
            print(
                f"reconstructed {day}: {', '.join(made)} (made after the fact, kept apart from "
                "the live record; the evening stays missed on time)",
                file=out,
            )
        for r in found:
            if r.session == day and r.status != evening.RECORDED:
                print(
                    f"  not reconstructed {r.ticker} as of {day} ({r.reason}: {r.detail}); "
                    "tried again at the next run",
                    file=out,
                )
    return all(r.status == evening.RECORDED for r in found)


def _reconstruct_only(evening, store, model, *, tickers, clock, expected, calendar, calendar_id,
                      path, out) -> int:  # fmt: skip
    """--reconstruct-missed (module docstring): reconstruct, then score; no job run."""

    _, problem = store.verify_ledger()
    if problem is not None:  # a record changed outside the store: nothing is written
        print(f"refused: the ledger does not verify ({problem})", file=out)
        return 1
    clean = _reconstruct(evening, store, model, clock=clock, expected=expected,
                         calendar=calendar, calendar_id=calendar_id, path=path, out=out)  # fmt: skip
    symbols = sorted(set(tickers) | set(store.forecast_symbols(DataClass.REAL)))
    outcomes = evening.score_real(
        store, symbols=symbols, now=clock(), expected=expected, sessions=calendar,
        calendar_id=calendar_id, path=path, write=True, model_id=model.model_id,
        watched=tickers, reconstructed_only=True,
    )  # fmt: skip
    counts = outcomes.reconstructed
    print(
        f"reconstructed scores: {counts['scored']} scored, {counts['unscorable']} unscorable, "
        f"{counts['pending']} pending (live forecasts are scored by the evening run only)",
        file=out,
    )
    return 0 if clean else 1


def main(
    argv: Sequence[str] | None = None,
    *,
    store_factory: Callable[[], StockLabStore] = StockLabStore,
    clock: Callable[[], pd.Timestamp] = utc_now,
    out: TextIO = sys.stdout,
    inference_path: Path | None = None,
    calendar=sessions,
    calendar_id: str = CALENDAR_ID,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1], allow_abbrev=False)
    parser.add_argument("--demo-replay", type=int, metavar="N")
    parser.add_argument("--authorized", action="store_true")
    parser.add_argument("--inference-window", action="store_true")
    real = parser.add_mutually_exclusive_group()
    real.add_argument("--record-real", action="store_true")
    real.add_argument("--dry-run", action="store_true")
    real.add_argument("--check-environment", action="store_true")
    real.add_argument("--reconstruct-missed", action="store_true")
    parser.add_argument("--model", metavar="ID")
    parser.add_argument("--health-inputs", type=Path, default=None)
    parser.add_argument("--ledger-head-backup", type=Path, default=None)
    args = parser.parse_args(argv)
    store = store_factory()
    if args.check_environment:
        if args.demo_replay is not None or args.authorized or args.inference_window:
            print("refused: --check-environment takes --model ID only", file=out)
            return 2
        return check_environment(store, args.model, out)
    args.health_inputs = health_inputs(args)
    started = clock()
    kept_head = args.ledger_head_backup or store.path.with_name(KEPT_HEAD_NAME)

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
                kept_head=kept_head,
            )

        try:
            days = run_replay(store, sessions=args.demo_replay, record_day=record_day)
        except ReplayRefused as error:
            print(f"refused: {error}", file=out)
            return 2
        scored = sum(day.scored for day in days)
        print(f"synthetic replay: {len(days)} sessions, {scored} targets scored", file=out)
        return 0

    if args.authorized or args.inference_window:
        print(
            "refused: the recent-window price update is scripts/stock_lab_fetch_recent.py "
            "--authorized --store (decisions A1 and A2); real forecasts are --dry-run or "
            "--record-real --model ID (decision A4)",
            file=out,
        )
        return 2
    if args.dry_run or args.record_real or args.reconstruct_missed:
        return _real(
            args, store=store, clock=clock, started=started, out=out,
            inference_path=inference_path, calendar=calendar, calendar_id=calendar_id,
            kept_head=kept_head,
        )  # fmt: skip
    if args.model is not None:
        print("refused: --model is for --dry-run or --record-real", file=out)
        return 2
    tickers = store.watched()
    try:
        expected = expected_session(started, calendar)
    except CalendarRefused as error:  # e.g. past the calendar's coverage: record the failure
        store.record_job_run(
            {
                "run_id": f"job-{started.strftime('%Y%m%dT%H%M%S%fZ')}",
                "started_at": stamp(started),
                "finished_at": stamp(clock()),
                "status": "failed",
                "reason": f"calendar: {error}",
                "expected_session": None,
                "data_class": DataClass.REAL.value,
            }
        )
        print(f"failed: the NYSE calendar refused ({error}); the run is recorded", file=out)
        return 1
    skipped = [
        "prices: the recent-window update needs decisions A1 and A2",
        "forecasts: recording real forecasts needs decision A4",
    ]
    # Step 3: scoring runs on whatever is stored (real forecasts exist only after A4).
    scored = scoring.score_store(store, tickers, calendar=calendar_sessions)
    status, reason = "skipped", "; ".join(skipped)
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
        sessions=lambda first, last: calendar(first.date(), last.date()),
        calendar_id=calendar_id,
        code_sha=code_sha(),
        next_expected_at=next_expected(finished, calendar),
        skipped=skipped,
        activation=NOT_AUTHORIZED,
        kept_head=kept_head,
    )
    print(f"expected session {expected.date()}; {scored} targets scored; {status}", file=out)
    for line in skipped:
        print(f"  skipped {line}", file=out)
    print(f"health input: {path}", file=out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
