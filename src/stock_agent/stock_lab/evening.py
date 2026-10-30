"""
The real evening job's frozen-model side (decision A4: built, NOT activated).

It is the one reviewed module of the Stock Lab that reads the inference store
for forecasts and scores (tests/test_stock_lab_inference_boundaries.py): the
frozen-model reader for a forecast's inputs, and the outcome reader for its
scores. The owner-run daily script calls it only on its real path.

Activation (decision A4)
    A real forecast is recorded only for a registered frozen model whose id
    is in ACTIVATED_MODELS. Each id is added by a reviewed change that records
    the owner's explicit go, after validated data, the frozen registered
    model, verified decision timestamps and a successful dry run, with the
    owner's words in docs/PROJECT_STATE.md. The owner activated model
    2b6cd0c1a52f5ecf (B1, half-life 252, frozen_long_run) for AMD, MRVL, MU
    and NVDA on 2026-10-20. A dry run computes everything and records
    nothing.

One evening, for each watched stock (`forecast_watchlist`)
    1. the expected session: the caller's (the last session whose close is at
       least 20:00 New York old);
    2. skipped, with the reason, when the stock is not in the frozen model, or
       already has a live forecast for that session (a second run never
       recalculates), or its newest stored close is older than the expected
       session (stale: its last valid forecast stays, labelled stale; nothing
       is invented or backfilled);
    3. refused, with the reader's reason, when the 252-session window cannot
       be read as of the decision time (a short window, a missing close, a
       decision before the close was available or at or after the next open);
    4. otherwise the frozen model's forecast from that window alone, its B1
       comparator on the same inputs, band and sessions, and the recorded
       state (frozen_forecast.forecast_record), recorded write-once by the research
       store, which refuses a created_at away from its clock or after the
       first target session opens.

Timing (2026-10-27: the Mac slept between the decision and the writes, and
the store refused every forecast as backdated)
    A forecast is made at its decision time: its window is read as of that
    time, which is both its created_at and its data.decision_time, so no
    batch retrieved later reaches it. The evening's decision time is `now`.
    When the store refuses a forecast as backdated (the Mac slept before the
    write) and the job passes its clock, that stock's forecast is made again,
    once, at clock(): its window read as of then (refused at or after the next
    open, like any decision), on its own panel (every feature is the stock's
    own), and written once more. Any other refusal, and a second one, is
    final. With its clock the job also never writes at or after the first
    target session's open (after_next_open), even a forecast made before it
    (the store judges the open by created_at, within CLOCK_SKEW). The store's
    checks are unchanged.

Reconstructing missed evenings (`reconstruct_missed`; the owner's decision of
2026-10-30 replaces decision A4's "missed sessions are never backfilled" for
the activated model)
    After the evening's live forecasts, every due evening since the first live
    forecast (evenings.missing_forecasts) is reconstructed, per stock (owner,
    2026-10-30), for each stock of the model that was watched when that
    evening began (the append-only watchlist, never today's) and has no
    forecast of it (live or reconstructed): a missed evening's stocks and a
    partly recorded evening's missing ones, all of them,
    session by session, chosen by no outcome. Each session's windows are read
    at the job's clock with load_reconstruction_window: the 252 sessions
    dated up to that close only, from every batch retrieved by then. The
    panel and the frozen model's forecast are the live evening's, so on the
    same data a reconstruction equals the forecast the evening would have
    recorded. It is recorded write-once through the store's own path,
    reconstructed true, at its true created_at (never backdated), one per
    (stock, session) and never beside a live one. A session whose window
    cannot be read is reported and tried again at the next run. A window_only
    model is never reconstructed (its prior pools the evening's panel, which
    can no longer be read as it was): each gap is reported.

Scoring (`score_real`): every real forecast not yet fully scored, on outcomes
read from the inference store (read_outcomes): a target is scored only when
every session it needs was absent from the store when the forecast was known
(scoring.score_forecast). A watched stock's session that has closed by the
expected session but has no stored outcome yet is counted as missing, once per
(stock, session), for Model Health; a stock no longer watched is still scored
when its outcomes arrive, but its prices are no longer fetched, so it never
counts as missing.

Reconstructions are scored by session dates (scoring.score_forecast) and
counted apart (ScoreSummary.reconstructed); the live counts are live alone.

Model Health (`monitor_fields`): per-stock results (recorded, skipped, stale,
refused), the freshness of each stock's newest stored close, input drift of
the windows against the frozen development reference, and the outcome
counts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.stock_lab import evenings, monitoring, registry, scoring
from stock_agent.stock_lab.evaluation import build_panel
from stock_agent.stock_lab.forecast import ForecastRefused, is_reconstructed, session_open
from stock_agent.stock_lab.frozen_forecast import forecast_record
from stock_agent.stock_lab.frozen_model import RegistryRefused
from stock_agent.stock_lab.inference.outcomes import latest_close, read_outcomes
from stock_agent.stock_lab.inference.schema import InferenceRefused, default_path
from stock_agent.stock_lab.inference.window import (
    load_inference_window,
    load_reconstruction_window,
)
from stock_agent.stock_lab.store import BackdatingRefused, StockLabStore, StoreRefused, stamp

Sessions = Callable[[pd.Timestamp, pd.Timestamp], pd.DatetimeIndex]

# Decision A4: the registered frozen models the owner has activated (2026-10-20, recorded in
# docs/PROJECT_STATE.md). Only these ids may record real forecasts.
ACTIVATED_MODELS: frozenset[str] = frozenset(
    {"2b6cd0c1a52f5ecfc16d6c8e221956d6fd23093826618e895949ac62e7fa65e7"}
)

RECORDED = "recorded"
DRY_RUN = "dry_run"
SKIPPED = "skipped"
STALE = "stale"
REFUSED = "refused"
TARGET_COUNT = 5
NEXT_SPAN = pd.Timedelta(days=21)


class NotActivated(ValueError):
    """The model is not activated for real forecasts (decision A4)."""


class AfterOpen(ValueError):
    """The job's clock reached the first target session's open before a write."""


def inference_path() -> Path:
    """The inference store's file (its own environment variable or default location)."""

    return default_path()


def check_activated(model_id: str) -> None:
    if model_id not in ACTIVATED_MODELS:
        raise NotActivated(
            f"model {model_id[:16]} is not activated: recording real forecasts needs the owner's "
            "explicit go (decision A4), added to evening.ACTIVATED_MODELS in a reviewed change"
        )


@dataclass(frozen=True)
class TickerResult:
    ticker: str
    status: str
    reason: str | None
    detail: str
    forecast: Mapping | None = None
    last_session: str | None = None

    def row(self) -> dict:
        return {"ticker": self.ticker, "status": self.status, "reason": self.reason}


@dataclass(frozen=True)
class Evening:
    results: tuple[TickerResult, ...]
    panel: pd.DataFrame | None
    freshness: tuple[dict, ...]


@dataclass(frozen=True)
class Reconstruction:
    """One missed (stock, session): recorded, or refused with the reason (tried again later)."""

    ticker: str
    session: str
    status: str  # RECORDED or REFUSED
    reason: str | None
    detail: str
    forecast_id: str | None = None


def _counts() -> dict:
    return {"scored": 0, "unscorable": 0, "pending": 0, "refused": 0}


@dataclass
class ScoreSummary:
    """The live track's counts; `reconstructed` holds the reconstructions' apart."""

    scored: int = 0
    unscorable: int = 0
    pending: int = 0
    missing: int = 0
    refused: list[dict] = field(default_factory=list)
    reconstructed: dict = field(default_factory=_counts)

    def as_dict(self) -> dict:
        found = {
            "scored": self.scored,
            "unscorable": self.unscorable,
            "pending": self.pending,
            "missing": self.missing,
            "refused": list(self.refused),
        }
        if any(self.reconstructed.values()):  # additive: only once a reconstruction exists
            found["reconstructed"] = dict(self.reconstructed)
        return found


def target_sessions(sessions: Sessions, as_of: pd.Timestamp) -> list[str]:
    """The five calendar sessions after `as_of`."""

    found = sessions(as_of + pd.Timedelta(days=1), as_of + NEXT_SPAN)
    days = [pd.Timestamp(day).strftime("%Y-%m-%d") for day in found[:TARGET_COUNT]]
    if len(days) != TARGET_COUNT:
        raise ValueError("the calendar has fewer than five sessions after the as-of session")
    return days


def _behind(sessions: Sessions, last: str | None, expected: pd.Timestamp) -> int | None:
    if last is None:
        return None
    day = pd.Timestamp(last, tz="UTC")
    return 0 if day >= expected else len(sessions(day + pd.Timedelta(days=1), expected))


def forecast_watchlist(
    store: StockLabStore,
    model: registry.FrozenModel,
    *,
    tickers: Sequence[str],
    now: pd.Timestamp,
    expected: pd.Timestamp,
    sessions: Sessions,
    calendar_id: str,
    path: Path,
    record: bool,
    clock: Callable[[], pd.Timestamp] | None = None,
) -> Evening:
    """
    One evening's forecasts (module docstring); `record` False is the dry run. Recording
    refuses a model the owner has not activated (NotActivated, decision A4). `now` is the
    decision time; with `clock`, a forecast refused as backdated is made again at `clock()`
    (module docstring, "Timing").
    """

    if record:
        check_activated(model.model_id)

    results: dict[str, TickerResult] = {}
    windows = {}
    freshness = []
    day = expected.strftime("%Y-%m-%d")
    for ticker in tickers:
        try:
            newest = latest_close(path, ticker, retrieved_by=now, calendar_id=calendar_id)
        except InferenceRefused as error:
            freshness.append({"ticker": ticker, "last_session": None, "sessions_behind": None})
            results[ticker] = TickerResult(ticker, REFUSED, error.reason.value, error.detail)
            continue
        last = None if newest is None else newest.session
        freshness.append(
            {
                "ticker": ticker,
                "last_session": last,
                "sessions_behind": _behind(sessions, last, expected),
            }
        )
        if ticker not in model.stocks:
            results[ticker] = TickerResult(ticker, SKIPPED, "not_in_model",
                                           f"{ticker} is not in the frozen model", last_session=last)  # fmt: skip
            continue
        try:
            recorded_already = store.live_forecast(ticker, day) is not None
        except (ForecastRefused, StoreRefused) as error:  # a stored record failed its check
            results[ticker] = TickerResult(ticker, REFUSED, "record_check_failed", str(error),
                                           last_session=last)  # fmt: skip
            continue
        if recorded_already:
            results[ticker] = TickerResult(ticker, SKIPPED, "already_recorded",
                                           f"{ticker} already has a forecast as of {day}", last_session=last)  # fmt: skip
        if last is None or last < day:
            if not recorded_already:
                words = (
                    "no close is stored" if last is None else f"the newest stored close is {last}"
                )
                results[ticker] = TickerResult(ticker, STALE, "stale_prices",
                                               f"{words}, not {day}: no forecast is made",
                                               last_session=last)  # fmt: skip
            continue
        try:
            windows[ticker] = load_inference_window(
                path, ticker, day, retrieved_by=now, sessions=sessions, calendar_id=calendar_id
            )
        except InferenceRefused as error:
            if not recorded_already:
                results[ticker] = TickerResult(ticker, REFUSED, error.reason.value, error.detail,
                                               last_session=last)  # fmt: skip
    panel = None
    if windows:  # every window read, including those of stocks already recorded (for drift)
        bars = {t: w.bars.drop(columns=["batch_id"]) for t, w in windows.items()}
        panel = build_panel(bars, model.final.config)
        targets = target_sessions(sessions, expected)
        for ticker, window in windows.items():
            if ticker in results:  # already recorded: its window counts for drift only
                continue
            try:
                artifact = _forecast(panel, ticker, model, window, day=day, targets=targets,
                                     decided=now)  # fmt: skip
            except (RegistryRefused, ForecastRefused) as error:
                reason = getattr(error, "reason", "forecast_refused")
                results[ticker] = TickerResult(ticker, REFUSED, reason, str(error), None, day)
                continue
            status, reason, detail = DRY_RUN, None, "computed; nothing recorded (dry run)"
            if record:
                # Made again only on a panel of its own: under window_only, B0 pools every
                # model stock's window, so a backdating refusal stays final there.
                remake = None
                if clock is not None and model.state_rule != registry.WINDOW_ONLY:
                    remake = partial(
                        _remade, model, ticker, path=path, day=day, targets=targets,
                        sessions=sessions, calendar_id=calendar_id, clock=clock,
                    )  # fmt: skip
                try:
                    artifact = _record(store, artifact, remake, clock)
                    status, detail = RECORDED, f"recorded {artifact['forecast_id'][:12]}"
                    if artifact["created_at"] != stamp(now):
                        detail += (
                            f", made again at {artifact['created_at']} after a backdating refusal"
                        )
                except StoreRefused as error:
                    status, reason, detail = REFUSED, "store_refused", str(error)
                except AfterOpen as error:
                    status, reason, detail = REFUSED, "after_next_open", str(error)
                except InferenceRefused as error:  # made again: its window refused (the open)
                    status, reason, detail = REFUSED, error.reason.value, error.detail
                except (RegistryRefused, ForecastRefused) as error:  # made again: refused
                    status, reason = REFUSED, getattr(error, "reason", "forecast_refused")
                    detail = str(error)
            results[ticker] = TickerResult(ticker, status, reason, detail, artifact, day)
    ordered = tuple(results[ticker] for ticker in tickers)
    return Evening(ordered, panel, tuple(freshness))


def _forecast(panel, ticker, model, window, *, day, targets, decided, reconstructed=False) -> dict:
    """`ticker`'s forecast from `window`, read as of `decided`: its decision time and created_at."""

    newest = window.batches[-1]
    data = {
        "source": newest.source,
        "batch_ids": [batch.batch_id for batch in window.batches],
        "retrieved_at": max(batch.retrieved_at for batch in window.batches),
        "last_close_session": day,
        "decision_time": stamp(decided),
        "window": {
            "first_session": window.sessions[0].strftime("%Y-%m-%d"),
            "sessions": len(window.sessions),
            "gaps": list(window.gaps),
            "revised_sessions": list(window.revised_sessions),
            "calendar": window.calendar_id,
        },
    }
    return forecast_record(
        panel, ticker, model, target_sessions=targets, created_at=decided, data=data,
        data_class=window.data_class, reconstructed=reconstructed,
    )  # fmt: skip


def _remade(model, ticker, *, path, day, targets, sessions, calendar_id, clock) -> dict:
    """`ticker`'s forecast made again at `clock()`, from its window read as of then."""

    decided = clock()
    window = load_inference_window(
        path, ticker, day, retrieved_by=decided, sessions=sessions, calendar_id=calendar_id
    )
    panel = build_panel({ticker: window.bars.drop(columns=["batch_id"])}, model.final.config)
    return _forecast(panel, ticker, model, window, day=day, targets=targets, decided=decided)


def _record(
    store: StockLabStore,
    artifact: dict,
    remake: Callable[[], dict] | None,
    clock: Callable[[], pd.Timestamp] | None,
) -> dict:
    """
    Record `artifact` and return what was recorded. A backdating refusal (made before the Mac
    slept) with `remake` makes the forecast again now and writes that once; any other refusal,
    and a second one, propagates (StoreRefused, or the remade window's InferenceRefused). No
    write starts at or after the first target session's open on `clock` (AfterOpen).
    """

    def write(record: dict) -> None:
        opens = pd.Timestamp(session_open(record["target_sessions"][0]))
        if clock is not None and clock() >= opens:
            raise AfterOpen(f"the {record['target_sessions'][0]} open has passed: nothing written")
        store.record_forecast(record, real_authorized=True)

    try:
        write(artifact)
    except BackdatingRefused:
        if remake is None:
            raise
        artifact = remake()
        write(artifact)
    return artifact


def reconstruct_missed(
    store: StockLabStore,
    model: registry.FrozenModel,
    *,
    clock: Callable[[], pd.Timestamp],
    expected: pd.Timestamp,
    sessions: Sessions,
    calendar_id: str,
    path: Path,
) -> tuple[Reconstruction, ...]:
    """
    Every missed (stock, session) since the first live forecast, reconstructed and recorded
    (module docstring). Only for a model the owner activated (NotActivated). ForecastRefused
    or StoreRefused when a stored record no longer verifies, as for every reader of the store.
    """

    check_activated(model.model_id)
    gaps = evenings.missing_forecasts(
        store, stocks=model.stocks, now=clock(), expected=expected, sessions=sessions
    )
    if model.state_rule == registry.WINDOW_ONLY:
        detail = "a window_only model's evening panel can no longer be read as it was"
        return tuple(Reconstruction(t, day, REFUSED, "window_only", detail) for day, t in gaps)
    out: list[Reconstruction] = []
    for day in sorted({session for session, _ in gaps}):
        decided = clock()  # the reconstruction's own time: its reads and its created_at
        windows, refused = {}, {}
        stocks = [t for t in store.watched_at(evenings.evening_opens(day)) if t in model.stocks]
        for ticker in stocks:  # that evening's panel: every stock watched then
            try:
                windows[ticker] = load_reconstruction_window(
                    path, ticker, day, retrieved_by=decided, sessions=sessions,
                    calendar_id=calendar_id,
                )  # fmt: skip
            except InferenceRefused as error:
                refused[ticker] = error
        bars = {t: w.bars.drop(columns=["batch_id"]) for t, w in windows.items()}
        panel = build_panel(bars, model.final.config) if bars else None
        targets = target_sessions(sessions, pd.Timestamp(day, tz="UTC"))
        for ticker in [t for session, t in gaps if session == day]:
            if ticker in refused:
                error = refused[ticker]
                out.append(Reconstruction(ticker, day, REFUSED, error.reason.value, error.detail))
                continue
            try:
                artifact = _forecast(panel, ticker, model, windows[ticker], day=day,
                                     targets=targets, decided=decided, reconstructed=True)  # fmt: skip
                store.record_reconstruction(artifact, real_authorized=True)
            except StoreRefused as error:
                out.append(Reconstruction(ticker, day, REFUSED, "store_refused", str(error)))
                continue
            except (RegistryRefused, ForecastRefused) as error:
                reason = getattr(error, "reason", "forecast_refused")
                out.append(Reconstruction(ticker, day, REFUSED, reason, str(error)))
                continue
            made = f"reconstructed {artifact['forecast_id'][:12]} at {artifact['created_at']}"
            out.append(Reconstruction(ticker, day, RECORDED, None, made, artifact["forecast_id"]))
    return tuple(out)


def score_real(
    store: StockLabStore,
    *,
    symbols: Sequence[str],
    now: pd.Timestamp,
    expected: pd.Timestamp,
    sessions: Sessions,
    calendar_id: str,
    path: Path,
    write: bool,
    model_id: str | None = None,
    watched: Sequence[str] | None = None,
    reconstructed_only: bool = False,
) -> ScoreSummary:
    """
    Score the real forecasts whose targets have closed (module docstring). Writing the scores
    is part of the activated job: `write` refuses unless `model_id`, the job's model, is
    activated (NotActivated, decision A4); without `write` nothing is written (the dry run).
    """

    if write:
        check_activated(model_id or "")
    summary = ScoreSummary()
    counted = set(symbols if watched is None else watched)
    missing: set[tuple[str, str]] = set()
    day = expected.strftime("%Y-%m-%d")
    for symbol in symbols:
        done = store.scored_keys(symbol)
        recorded = store.forecast_recorded_at(symbol)
        try:
            stored = [*store.forecasts(symbol), *store.forecasts(symbol, reconstructed=True)]
        except (ForecastRefused, StoreRefused):  # a stored record failed its check: none scored
            summary.refused.append(
                {"ticker": symbol, "forecast_id": None, "reason": "record_check_failed"}
            )
            continue
        for record in stored:
            if record["data_class"] != "real" or record["refusal"] is not None:
                continue
            rebuilt = is_reconstructed(record)  # counted apart from the live track
            if reconstructed_only and not rebuilt:  # the manual reconstruction: no live score
                continue
            targets = [str(h) for h in range(1, 6)] + (["cum"] if record["cumulative"] else [])
            if all((record["forecast_id"], t) in done for t in targets):
                continue
            if record["forecast_id"] not in recorded:
                continue  # no ledger entry: the chain check reports it; never scored
            known = max(pd.Timestamp(record["created_at"]),
                        pd.Timestamp(recorded[record["forecast_id"]]))  # fmt: skip
            try:
                outcomes = read_outcomes(
                    path, symbol, as_of_session=record["as_of_session"],
                    target_sessions=record["target_sessions"], known_at=known, scored_by=now,
                    sessions=sessions, calendar_id=calendar_id,
                )  # fmt: skip
            except InferenceRefused as error:
                if rebuilt:
                    summary.reconstructed["refused"] += 1
                    continue
                summary.refused.append({"ticker": symbol, "forecast_id": record["forecast_id"],
                                        "reason": error.reason.value})  # fmt: skip
                continue
            found = scoring.score_forecast(
                record, outcomes.bars, outcomes.batch_of,
                seen_before=outcomes.seen_before, sessions=outcomes.span,
            )  # fmt: skip
            fresh = []
            for score in found:
                if (score.forecast_id, score.target) in done:
                    continue
                if rebuilt:
                    summary.reconstructed[score.status] += 1
                    if score.status != scoring.PENDING:
                        fresh.append(score)
                    continue
                if score.status == scoring.PENDING:
                    summary.pending += 1
                    # Missing: a closed session with no stored outcome, once per (stock,
                    # session), for watched stocks only (the evening fetch updates those).
                    if score.session <= day and symbol in counted:
                        missing.add((symbol, score.session))
                    continue
                fresh.append(score)
                if score.status == scoring.SCORED:
                    summary.scored += 1
                else:
                    summary.unscorable += 1
            if write and fresh:
                store.record_scores(fresh)
    summary.missing = len(missing)
    return summary


def input_drift(panel: pd.DataFrame | None, model: registry.FrozenModel) -> list[dict]:
    """
    PSI of the windows' latest sessions (monitoring.RECENT per stock) against the frozen
    development reference, pooled over the stocks read; each row names how many stocks and
    finite values it pooled, since PSI moves with that mix as well as with the inputs.
    """

    rows = []
    for feature in monitoring.DRIFT_FEATURES:
        recent, stocks = np.array([]), 0
        if panel is not None:
            parts = [
                group[feature].to_numpy(dtype="float64")[-monitoring.RECENT :]
                for _, group in panel.groupby("symbol", sort=False)
            ]
            stocks = len(parts)
            recent = np.concatenate(parts) if parts else recent
        value = monitoring.psi_against(model.drift_reference.get(feature), recent)
        rows.append(
            {
                "feature": feature,
                "psi": value,
                "stocks": stocks,
                "values": int(np.isfinite(recent).sum()),
            }
        )
    return rows
