"""
The premarket briefing, assembled from the Stock Lab's stored records
(read-only; numpy, pandas and the Stock Lab only).

What it reads
    Only records the store has already checked: the watchlist, the recorded
    forecasts (each re-checked against its id when read), their scores, the
    evaluation runs, the daily job's runs and the ledger's verification.
    The store is opened read-only (`ReadOnlyRecords`: SQLite `mode=ro`, and
    no write method exists on it). Nothing is retrained, no old forecast is
    recomputed, and the evening job is never run.

As known when it is prepared
    A record counts only if it existed when the briefing was prepared: a
    forecast created and recorded (its ledger entry) by then, a score
    written by then, a job run finished by then. A recalculation replaces a
    forecast only if the recalculation itself existed by then.

Per stock (the forecast for the market date)
    fresh        the live forecast is as of the prior session and its first
                 target session is the market date: its stored probabilities
                 are shown exactly as recorded
    stale        the newest live forecast is older: it is named, its
                 probabilities are not shown as today's
    unavailable  no forecast, a refusal, a failed ledger check, or a target
                 session that is not the market date; always with a reason
    A probability is never filled in, carried forward or recomputed.

Descriptive context and the point-in-time rules
    - The current run, the reversal statistics after it and the bull-trap
      signal are read from the forecast record (computed by the forecast
      job from data through its as-of close).
    - Momentum, volatility, the volatility regime and the bear-trap check
      are computed from stored prices through the as-of session only, and
      only for synthetic stocks or sessions through 2024-12-31. For a real
      stock after that, the prices are protected-period observations: under
      decision A6 nothing computes a feature on them outside the frozen
      forecast job, so the briefing shows the state that job recorded with
      the forecast (momentum, volatility and the regime against its window),
      and the latest observed close, its session and its retrieval time from
      the isolated descriptive reader (decision A6; the script passes it in),
      as stored by the preparation time. Research-store prices of 2025 or
      later are never shown for a real stock.
    - Trap flags on real data are withheld until the owner confirms the
      failure margin f (decision A5).

Change since the prior forecast
    Only between identical target definitions: the next session's three
    probabilities against the prior forecast's next session, at the same
    stable band; the five-session total against the prior total at the
    same band. A changed band is reported as not comparable.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import types
import typing
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from stock_agent.stock_lab import forecast_view
from stock_agent.stock_lab.briefing import reliability as rel
from stock_agent.stock_lab.briefing.timing import (
    PREMARKET,
    STATE_WORDS,
    Sessions,
    Timing,
    gap_note,
    long_date,
)
from stock_agent.stock_lab.contracts import DEVELOPMENT_END, DataClass, StableBand
from stock_agent.stock_lab.features import build_features
from stock_agent.stock_lab.forecast import ForecastRefused
from stock_agent.stock_lab.reversal import INSUFFICIENT, volatility_regime
from stock_agent.stock_lab.scores import jeffreys
from stock_agent.stock_lab.store import StockLabStore, StoreRefused, stamp
from stock_agent.stock_lab.traps import UNRESOLVED, TrapRule, evaluate_traps, trap_events
from stock_agent.stock_lab.view import run_words

SCHEMA = "stock_lab_premarket_briefing"
SCHEMA_VERSION = "1.1"  # 1.1 adds the comparison's persistence check and its caveat
FRESH = "fresh"
STALE = "stale"
UNAVAILABLE = "unavailable"
CATEGORIES = ("increase", "stable", "decrease")
SETUP_RECENT_DAYS = 21  # a setup counts as current for this many calendar days (as predict.py)
LEVEL = 0.9
LEDGER_FAILED = "The ledger check failed"  # the notice that withholds every probability


# --- the read-only view of the store ---


class ReadOnlyRecords:
    """
    The store's read methods only. Every read opens SQLite with `mode=ro`
    (StockLabStore._read); a missing file reads as empty and is not created.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._store = StockLabStore(path)

    @property
    def path(self) -> Path:
        return self._store.path

    @property
    def exists(self) -> bool:
        return self._store.exists

    def watched(self) -> tuple[str, ...]:
        return self._store.watched()

    def forecasts(self, symbol: str) -> tuple[dict, ...]:
        return self._store.forecasts(symbol)

    def forecast_recorded_at(self, symbol: str) -> dict[str, str]:
        return self._store.forecast_recorded_at(symbol)

    def scores(self, symbol: str) -> tuple[dict, ...]:
        return self._store.scores(symbol)

    def job_runs(self) -> tuple[dict, ...]:
        return self._store.job_runs()

    def evaluation(self, run_id: str) -> dict | None:
        return self._store.evaluation(run_id)

    def evaluations(self, data_class: DataClass) -> tuple[dict, ...]:
        return self._store.evaluations(data_class)

    def scored_symbols(self) -> tuple[str, ...]:
        """Every symbol with a stored score (any data class), including former watchlist ones."""

        return tuple(sorted({score["symbol"] for score in self._store.scores()}))

    def bars(self, symbol: str, *, as_of: pd.Timestamp | None = None):
        return self._store.bars(symbol, as_of=as_of)

    def verify_ledger(self) -> tuple[int, str | None]:
        return self._store.verify_ledger()


# --- the briefing's parts ---


@dataclass(frozen=True)
class Probabilities:
    increase: float
    stable: float
    decrease: float


@dataclass(frozen=True)
class SessionForecast:
    horizon: int
    session: str
    probabilities: Probabilities


@dataclass(frozen=True)
class TrackRecord:
    """Development-test forecasts in the same probability range: how often it happened."""

    category: str
    bin_low: float
    bin_high: float
    observed: float
    low: float
    high: float
    n: int
    dates: int
    status: str  # insufficient when the range held fewer than reliability.MIN_CLUSTERS dates


@dataclass(frozen=True)
class Lineage:
    forecast_id: str
    as_of_session: str
    created_at: str
    estimator: str
    model_version: str
    feature_version: str
    evaluation_run_id: str | None
    evidence_status: str
    calibration_status: str
    band: float
    cumulative_band: float
    source: str | None
    retrieved_at: str | None


@dataclass(frozen=True)
class Close:
    available: bool
    session: str | None
    value: float | None
    retrieved_at: str | None
    reason: str | None
    words: str
    source: str | None = None  # "research store" or "observed" (decision A6)


@dataclass(frozen=True)
class Context:
    available: bool
    reason: str | None
    words: str
    ret_5: float | None = None
    ret_20: float | None = None
    vol_20: float | None = None
    ewma_vol: float | None = None
    regime: str | None = None
    regime_basis: str | None = None  # "history" (its own history) or "window" (the forecast's)


@dataclass(frozen=True)
class Streak:
    available: bool
    direction: str | None
    length: int | None
    words: str


@dataclass(frozen=True)
class ReversalOutcome:
    outcome: str
    estimate: float | None
    low: float | None
    high: float | None
    n_eff: float | None
    status: str


@dataclass(frozen=True)
class Reversal:
    available: bool
    reason: str | None
    words: str
    definition: str | None = None
    events: int | None = None
    outcomes: tuple[ReversalOutcome, ...] = ()


@dataclass(frozen=True)
class Trap:
    kind: str  # bull or bear
    available: bool
    reason: str | None
    words: str
    source: str | None = None
    setup_now: bool | None = None  # an unresolved setup within SETUP_RECENT_DAYS
    setup_session: str | None = None  # the latest setup, any status
    historical_failure_rate: float | None = None
    base_rate: float | None = None
    setups_resolved: int | None = None


@dataclass(frozen=True)
class Change:
    comparable: bool
    reason: str | None
    words: str
    prior_forecast_id: str | None = None
    prior_as_of: str | None = None
    next_session: Probabilities | None = None  # differences, in probability
    cumulative: Probabilities | None = None
    model_changed: bool = False


@dataclass(frozen=True)
class StockBriefing:
    symbol: str
    data_class: str
    status: str
    reason: str | None
    words: str
    lineage: Lineage | None
    described_as_of: str | None  # the close the descriptive notes describe (the forecast's)
    sessions_behind: int | None
    next_session: SessionForecast | None
    track: tuple[TrackRecord, ...]
    five_sessions: tuple[SessionForecast, ...]
    cumulative: Probabilities | None
    cumulative_end: str | None
    close: Close
    context: Context
    streak: Streak
    change: Change
    reversal: Reversal
    bull_trap: Trap
    bear_trap: Trap
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class LatestUpdate:
    last_success_at: str | None
    last_run_at: str | None
    last_status: str | None
    activation: str | None
    expected_session: str | None
    words: str


@dataclass(frozen=True)
class LedgerCheck:
    entries: int
    verified: bool
    problem: str | None


@dataclass(frozen=True)
class RollingGroup:
    target: str
    band: float
    estimator: str
    model_version: str
    rolling: rel.Rolling


@dataclass(frozen=True)
class ProspectiveEvidence:
    data_class: str
    status: str  # none, insufficient or ok
    words: str
    scored_targets: int  # daily targets (sessions 1-5), counted together for the status only
    settled_sessions: int
    symbols: tuple[str, ...]  # every symbol with a scored record, on today's list or not
    former: tuple[str, ...]  # those of them not on this briefing (removed or excluded)
    groups: tuple[rel.CalibrationGroup, ...]
    rolling: tuple[RollingGroup, ...]
    comparison: rel.MatchedComparison | None
    comparison_words: str
    comparison_caveat: str | None  # the persistence caveat, only when the flag is on


@dataclass(frozen=True)
class ReliabilitySection:
    development: rel.DevelopmentEvidence | None
    development_words: str
    prospective: ProspectiveEvidence
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Briefing:
    schema: str
    schema_version: str
    data_class: str
    market_date: str
    prior_session: str
    scheduled_at: str
    generated_at: str
    state: str
    state_words: str
    closed_days: tuple[str, ...]
    gap_note: str | None
    calendar_id: str
    data_as_of: str | None
    freshness: Mapping[str, int]
    model_versions: tuple[str, ...]
    latest_update: LatestUpdate
    ledger: LedgerCheck
    stocks: tuple[StockBriefing, ...]
    excluded: tuple[str, ...]
    warnings: tuple[str, ...]
    limitations: tuple[str, ...]
    reliability: ReliabilitySection


def briefing_document(briefing: Briefing) -> dict:
    """The briefing as plain JSON-ready data (non-finite numbers become None)."""

    def clean(value):
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, dict):
            return {str(key): clean(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [clean(item) for item in value]
        return value

    return clean(asdict(briefing))


@functools.cache
def _hints(kind: type) -> dict[str, object]:
    return typing.get_type_hints(kind)


def _some(values) -> str:
    """At most three values, each cut short, for an error message."""

    shown = sorted(repr(value)[:30] for value in values)
    return ", ".join(shown[:3]) + (f" and {len(shown) - 3} more" if len(shown) > 3 else "")


def _load(kind, value, where: str):
    """`value` (parsed JSON) as `kind`, refusing anything that does not fit exactly."""

    origin, args = typing.get_origin(kind), typing.get_args(kind)
    if origin in (types.UnionType, typing.Union):
        if value is None and type(None) in args:
            return None
        (other,) = (arg for arg in args if arg is not type(None))
        return _load(other, value, where)
    if dataclasses.is_dataclass(kind):
        if not isinstance(value, dict):
            raise TypeError(f"{where} is not an object")
        names = [field.name for field in dataclasses.fields(kind)]
        if set(value) != set(names):
            raise ValueError(f"{where} has other fields: {_some(set(value) ^ set(names))}")
        hints = _hints(kind)
        return kind(**{name: _load(hints[name], value[name], f"{where}.{name}") for name in names})
    if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
        if not isinstance(value, list):
            raise TypeError(f"{where} is not a list")
        return tuple(_load(args[0], item, f"{where}[{i}]") for i, item in enumerate(value))
    if origin in (Mapping, dict) and args and args[0] is str:
        if not isinstance(value, dict):
            raise TypeError(f"{where} is not an object")
        return {key: _load(args[1], item, f"{where}.{key}") for key, item in value.items()}
    if kind is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"{where} is not a number")
        return float(value)
    if kind in (str, int, bool):
        if type(value) is not kind:
            raise TypeError(f"{where} is not a {kind.__name__}")
        return value
    raise TypeError(f"{where} has a type a briefing never holds: {kind!r}")


def briefing_from_document(document: Mapping) -> Briefing:
    """
    The briefing read back from its JSON copy (`briefing_document`), field by
    field and type by type: a missing or extra field, or a value of the wrong
    type, is refused (ValueError or TypeError), and so is another schema or version.
    A null where a number is required is refused too: `briefing_document` writes
    a non-finite number as null, and no briefing a page can show holds one there.
    Error messages name fields, never repeat the document's values at length.
    """

    if not isinstance(document, dict):
        raise TypeError("the briefing is not an object")
    if (document.get("schema"), document.get("schema_version")) != (SCHEMA, SCHEMA_VERSION):
        written = _some([document.get("schema"), document.get("schema_version")])
        raise ValueError(f"written as {written}, not {SCHEMA} {SCHEMA_VERSION}")
    return _load(Briefing, document, "briefing")


# --- helpers ---


def _finite(value) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _probabilities(row: Mapping) -> Probabilities:
    return Probabilities(
        increase=float(row["p_increase"]),
        stable=float(row["p_stable"]),
        decrease=float(row["p_decrease"]),
    )


def _difference(new: Probabilities, old: Probabilities) -> Probabilities:
    return Probabilities(
        increase=new.increase - old.increase,
        stable=new.stable - old.stable,
        decrease=new.decrease - old.decrease,
    )


def _pct(value: float | None, digits: int = 0) -> str:
    return forecast_view.pct(value, digits)


def _session_ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def _sessions_between(calendar: Sessions, after: str, through: str) -> int:
    """Sessions in (after, through]."""

    start = date.fromisoformat(after)
    end = date.fromisoformat(through)
    if end <= start:
        return 0
    days = {pd.Timestamp(d).date() for d in calendar(start, end)}
    return len([d for d in days if start < d <= end])


@dataclass(frozen=True)
class _Known:
    """One stock's records as known when the briefing was prepared."""

    forecasts: tuple[dict, ...]  # created and recorded by then, oldest first
    live: tuple[dict, ...]  # those not replaced by a recalculation known by then
    scores: tuple[dict, ...]
    problem: str | None  # a record that failed its check when read


def _known(records: ReadOnlyRecords, symbol: str, cutoff: str, data_class: str) -> _Known:
    try:
        stored = records.forecasts(symbol)
        recorded = records.forecast_recorded_at(symbol)
        scores = records.scores(symbol)
    except (ForecastRefused, StoreRefused) as error:
        return _Known((), (), (), f"{type(error).__name__}: {error}")
    forecasts = tuple(
        record
        for record in stored
        if record["data_class"] == data_class
        and record["created_at"] <= cutoff
        and record["forecast_id"]
        in recorded  # no ledger entry: never read (the chain check reports it)
        and recorded[record["forecast_id"]] <= cutoff
    )
    replaced = {record["supersedes"] for record in forecasts if record["supersedes"]}
    live = tuple(record for record in forecasts if record["forecast_id"] not in replaced)
    ids = {record["forecast_id"] for record in forecasts}
    known_scores = tuple(
        score for score in scores if score["forecast_id"] in ids and score["scored_at"] <= cutoff
    )
    return _Known(forecasts, live, known_scores, None)


def _lineage(record: Mapping) -> Lineage:
    data = record.get("data") or {}
    return Lineage(
        forecast_id=record["forecast_id"],
        as_of_session=record["as_of_session"],
        created_at=record["created_at"],
        estimator=record["estimator"],
        model_version=record["model_version"],
        feature_version=record["feature_version"],
        evaluation_run_id=record["evaluation_run_id"],
        evidence_status=record["evidence_status"],
        calibration_status=record["calibration_status"],
        band=float(record["band"]),
        cumulative_band=float(record["cumulative_band"]),
        source=data.get("source"),
        retrieved_at=data.get("retrieved_at"),
    )


def _track(evaluation: Mapping | None, record: Mapping) -> tuple[TrackRecord, ...]:
    if evaluation is None or not record["daily"]:
        return ()
    first = record["daily"][0]
    out = []
    for category in CATEGORIES:
        found = forecast_view.track_record(
            evaluation, record["estimator"], "target_1", category, first[f"p_{category}"]
        )
        if found is not None:
            out.append(
                TrackRecord(
                    category,
                    found.bin_low,
                    found.bin_high,
                    found.observed,
                    found.low,
                    found.high,
                    found.n,
                    found.dates,
                    INSUFFICIENT if found.dates < rel.MIN_CLUSTERS else rel.OK,
                )
            )
    return tuple(out)


def _streak(record: Mapping | None) -> Streak:
    run = None if record is None else record["signals"].get("current_run")
    if not run:
        return Streak(False, None, None, "Not recorded with a forecast.")
    direction, length = run.get("direction"), run.get("length")
    if direction is None or not length:
        return Streak(True, None, None, "No run: the latest session follows a gap.")
    return Streak(True, direction, int(length), f"{run_words(direction, int(length))}.")


def _reversal(record: Mapping | None) -> Reversal:
    if record is None:
        return Reversal(False, "no_forecast", "Not recorded: no forecast.")
    found = record["signals"].get("after_this_streak")
    if found is None:
        run = record["signals"].get("current_run") or {}
        if run.get("direction") in (None, "stable"):
            return Reversal(
                False,
                "no_streak",
                "No rising or falling streak is running, so no reversal statistic applies.",
            )
        return Reversal(False, "not_recorded", "Not recorded with the forecast.")
    outcomes = []
    for name in ("continuation", "stabilization", "reversal"):
        row = found["outcomes"].get(name) or {}
        estimate, n_eff = _finite(row.get("estimate")), _finite(row.get("n_eff"))
        status = row.get("status", INSUFFICIENT)
        low = high = None
        if estimate is not None and n_eff and status != INSUFFICIENT:
            low, high = jeffreys(estimate * n_eff, n_eff, LEVEL)
        else:
            estimate = None
        outcomes.append(ReversalOutcome(name, estimate, low, high, n_eff, status))
    if found.get("source"):  # a frozen model's statistics: development history, not the window
        where = f"past cases in {found['source']}"
        if found.get("table_length") and found.get("run_length", 0) > found["table_length"]:
            where += (
                f"; the current run is {found['run_length']} sessions, so the "
                f"{found['table_length']}-session figures are shown"
            )
    else:
        where = "past cases, through the as-of close"
    return Reversal(
        True,
        None,
        f"After {found['definition']} ({found['events']:,} {where}; intervals are 90% "
        "Jeffreys intervals on the effective count).",
        definition=found["definition"],
        events=int(found["events"]),
        outcomes=tuple(outcomes),
    )


def _trap_words(kind: str, setup_now: bool, setup_session: str | None, rate, base, n) -> str:
    noun = "bull-trap" if kind == "bull" else "bear-trap"
    if setup_now:
        lead = (
            f"A {noun} setup (the registered rule) formed on {setup_session}. It is unresolved: "
            "its 10-session window has not closed, so it is not confirmed either way."
        )
    elif setup_session:
        lead = f"No unresolved {noun} setup. The latest setup was on {setup_session}."
    else:
        lead = f"No {noun} setup in the stored history."
    if rate is None:
        tail = f" Too few resolved past setups ({n or 0}) to say how often they failed."
    else:
        tail = (
            f" Past setups failed {_pct(rate)} of the time ({n:,} resolved; all breakouts: "
            f"{_pct(base)})."
        )
    return lead + tail


def _withheld_trap(kind: str) -> Trap:
    return Trap(
        kind,
        False,
        "trap_margin_unconfirmed",
        "Withheld for real data until the owner confirms the trap rule's failure margin f "
        "(decision A5).",
    )


def _bull_trap(record: Mapping | None, data_class: str, margin_confirmed: bool) -> Trap:
    if data_class == DataClass.REAL.value and not margin_confirmed:
        return _withheld_trap("bull")
    signal = None if record is None else record["signals"].get("bull_trap")
    if signal is None:
        return Trap("bull", False, "not_recorded", "Not recorded with a forecast.")
    rate = _finite(signal.get("historical_failure_rate"))
    base = _finite(signal.get("base_rate"))
    n = signal.get("setups_resolved")
    return Trap(
        "bull",
        True,
        None,
        _trap_words(
            "bull", bool(signal.get("setup_now")), signal.get("setup_session"), rate, base, n
        ),
        source="forecast record",
        setup_now=bool(signal.get("setup_now")),
        setup_session=signal.get("setup_session"),
        historical_failure_rate=rate,
        base_rate=base,
        setups_resolved=n,
    )


def _prices(
    records: ReadOnlyRecords,
    symbol: str,
    session: str | None,
    *,
    data_class: str,
    cutoff: pd.Timestamp,
    margin_confirmed: bool,
) -> tuple[Close, Context, Trap]:
    """The stored close of `session`, and the context computed through it (module docstring)."""

    def missing(reason: str, words: str) -> tuple[Close, Context, Trap]:
        trap = (
            _withheld_trap("bear")
            if data_class == DataClass.REAL.value and not margin_confirmed
            else Trap("bear", False, reason, words)
        )
        return Close(False, session, None, None, reason, words), Context(False, reason, words), trap

    if session is None:
        return missing("no_session", "No session to describe: no forecast is recorded.")
    try:
        stored = records.bars(symbol, as_of=cutoff)
    except StoreRefused as error:
        return missing("prices_refused", f"The stored prices failed a check ({error}).")
    if stored is None:
        return missing("no_stored_prices", "No prices are stored for this stock.")
    bars = stored.bars[stored.bars["session"] <= _session_ts(session)]
    row = bars[bars["session"] == _session_ts(session)]
    if row.empty or pd.isna(row["close"].iloc[0]):
        newest = bars.loc[bars["close"].notna(), "session"]
        latest = newest.max().strftime("%Y-%m-%d") if len(newest) else "none"
        return missing(
            "close_not_stored",
            f"The research store holds no close for {session} (its newest stored close: "
            f"{latest}). Recent closes are kept apart and are not read by the briefing.",
        )
    batch = row["batch_id"].iloc[0]
    retrieved = next((b.retrieved_at for b in stored.batches if b.batch_id == batch), None)
    close = Close(
        True,
        session,
        float(row["close"].iloc[0]),
        retrieved,
        None,
        f"Close of {session}, from a stored batch that passed its count and hash checks.",
        source="research store",
    )
    allowed = (
        data_class == DataClass.SYNTHETIC.value or date.fromisoformat(session) <= DEVELOPMENT_END
    )
    if not allowed:
        words = (
            f"Not computed: {session} is in the protected period (2025 and later), and under "
            "decision A6 nothing outside the frozen forecast job computes features on those "
            "prices. These values appear once that job records them with each forecast."
        )
        return close, Context(False, "recent_prices_not_read", words), _withheld_trap("bear")
    frame = build_features(bars.drop(columns=["batch_id"]))
    last = frame.iloc[-1]
    regime = volatility_regime(frame).iloc[-1]
    context = Context(
        True,
        None,
        "From stored prices through the as-of close.",
        ret_5=_finite(last["ret_5"]),
        ret_20=_finite(last["ret_20"]),
        vol_20=_finite(last["vol_20"]),
        ewma_vol=_finite(last["ewma_vol"]),
        regime=regime if isinstance(regime, str) else None,
        regime_basis="history",
    )
    if data_class == DataClass.REAL.value and not margin_confirmed:
        return close, context, _withheld_trap("bear")
    rule = TrapRule()
    evaluation = evaluate_traps(
        trap_events(frame, rule, kind="bear", outcome_end=_session_ts(session)), rule, "bear"
    )
    latest = evaluation.latest_setup
    setup_session = None if latest is None else pd.Timestamp(latest["session"]).strftime("%Y-%m-%d")
    recent = (
        latest is not None
        and (_session_ts(session) - pd.Timestamp(latest["session"])).days <= SETUP_RECENT_DAYS
    )
    setup_now = bool(recent and latest["status"] == UNRESOLVED)
    rate = _finite(evaluation.setup_rate.estimate)
    base = _finite(evaluation.base_rate.estimate)
    n = evaluation.setup_rate.n
    trap = Trap(
        "bear",
        True,
        None,
        _trap_words("bear", setup_now, setup_session, rate, base, n),
        source="stored prices",
        setup_now=setup_now,
        setup_session=setup_session,
        historical_failure_rate=rate,
        base_rate=base,
        setups_resolved=n,
    )
    return close, context, trap


RECENT_WORDS = {
    "no_store": "No recent prices are stored yet: the recent-window fetch has not stored any.",
    "no_data": "No recent prices are stored for this stock by the time the briefing was prepared.",
}


def _observed_close(recent, symbol: str, described: str | None, cutoff: pd.Timestamp) -> Close:
    """
    The latest observed close of a real stock, its session and its retrieval time, from the
    isolated descriptive reader (decision A6), as stored by the preparation time.
    """

    if recent is None:
        return Close(False, None, None, None, "recent_not_read",
                     "Recent observed prices are not read for this briefing.")  # fmt: skip
    observed = recent(symbol, retrieved_by=cutoff)
    if observed.state != "observed":
        words = RECENT_WORDS.get(
            observed.state, f"The recent prices could not be read ({observed.reason})."
        )
        return Close(False, None, None, None, observed.state, words)
    served = [row for row in observed.sessions if row.close is not None]
    if not served:
        return Close(False, None, None, None, "no_close_served",
                     "No recent session was served with a close.")  # fmt: skip
    latest = served[-1]
    words = (
        "Latest observed close, for reference: recent prices kept apart from the research data, "
        "not research evidence."
    )
    if latest.session != observed.last_session:
        words += f" The newest stored session, {observed.last_session}, was served without a close."
    if described is not None and latest.session != described:
        words += f" The newest forecast is as of the close of {described}."
    return Close(True, latest.session, latest.close, observed.retrieved_at, None, words,
                 source="observed")  # fmt: skip


def _recorded_context(record: Mapping | None) -> Context:
    """The state the frozen job recorded with the forecast, from its window (never recomputed)."""

    state = None if record is None else record["signals"].get("market_state")
    if not state:
        return Context(
            False,
            "not_recorded",
            "Not recorded with this forecast. The frozen forecast job records momentum, "
            "volatility and the regime with each real forecast it makes; the briefing never "
            "computes them itself (decision A6).",
        )
    return Context(
        True,
        None,
        "Recorded with the forecast by the frozen job, from its inference window through the "
        "as-of close.",
        ret_5=_finite(state.get("ret_5")),
        ret_20=_finite(state.get("ret_20")),
        vol_20=_finite(state.get("vol_20")),
        ewma_vol=_finite(state.get("ewma_vol")),
        regime=state.get("regime") if isinstance(state.get("regime"), str) else None,
        regime_basis="window",
    )


def _change(record: Mapping | None, live: Sequence[Mapping]) -> Change:
    if record is None or record["refusal"] is not None:
        return Change(False, "no_forecast", "No current forecast to compare.")
    earlier = [f for f in live if f["as_of_session"] < record["as_of_session"]]
    if not earlier:
        return Change(False, "no_prior", "No earlier forecast is recorded.")
    prior = max(earlier, key=lambda f: (f["as_of_session"], f["created_at"]))
    common = {"prior_forecast_id": prior["forecast_id"], "prior_as_of": prior["as_of_session"]}
    if prior["refusal"] is not None:
        return Change(False, "prior_refused", "The prior forecast gave no probability.", **common)
    if float(prior["band"]) != float(record["band"]):
        before, after = StableBand(prior["band"]).words(), StableBand(record["band"]).words()
        return Change(
            False,
            "band_changed",
            f"The stable band changed ({before} to {after}), so the probabilities measure "
            "different things and are not compared.",
            **common,
        )
    next_change = _difference(_probabilities(record["daily"][0]), _probabilities(prior["daily"][0]))
    cumulative = None
    if (
        record["cumulative"] is not None
        and prior["cumulative"] is not None
        and float(prior["cumulative_band"]) == float(record["cumulative_band"])
    ):
        cumulative = _difference(
            _probabilities(record["cumulative"]), _probabilities(prior["cumulative"])
        )
    model_changed = (prior["estimator"], prior["model_version"]) != (
        record["estimator"],
        record["model_version"],
    )
    words = (
        f"Against the forecast as of {prior['as_of_session']}: each one's next session, at the "
        f"same band ({StableBand(record['band']).words()})."
    )
    if model_changed:
        words += " The model changed between the two forecasts."
    return Change(
        True,
        None,
        words,
        next_session=next_change,
        cumulative=cumulative,
        model_changed=model_changed,
        **common,
    )


def _no_forecast_words(symbol: str, data_class: str, not_activated: bool, prior: str) -> str:
    if data_class == DataClass.REAL.value and not_activated:
        return (
            f"No forecast is recorded for {symbol}. Real forecasts are recorded only after the "
            "owner activates the forecast protocol (decision A4); nothing is shown in their place."
        )
    return f"No forecast is recorded for {symbol} as of {prior} or earlier."


def _stock(
    records: ReadOnlyRecords,
    symbol: str,
    *,
    plan: Timing,
    calendar: Sessions,
    data_class: str,
    cutoff: str,
    cutoff_ts: pd.Timestamp,
    ledger_problem: str | None,
    not_activated: bool,
    evaluations: dict[str, Mapping | None],
    margin_confirmed: bool,
    recent=None,
) -> tuple[StockBriefing, _Known]:
    known = _known(records, symbol, cutoff, data_class)
    market = plan.market_date.isoformat()
    prior = plan.prior_session.isoformat()
    candidates = [f for f in known.live if f["as_of_session"] < market]
    record = max(candidates, key=lambda f: (f["as_of_session"], f["created_at"]), default=None)
    status, reason, words, behind = UNAVAILABLE, None, "", None
    shown = None  # the record whose probabilities are shown (fresh only)
    if known.problem is not None:
        reason, words = (
            "record_check_failed",
            "Withheld: a stored record failed its integrity check when read.",
        )
        record = None
    elif record is None:
        reason = "no_recorded_forecast"
        words = _no_forecast_words(symbol, data_class, not_activated, prior)
    elif ledger_problem is not None:
        reason = "ledger_unverified"
        words = (
            f"Withheld: the ledger check failed ({ledger_problem}), so no stored probability can "
            "be shown as verified."
        )
    elif record["as_of_session"] < prior:
        status, reason = STALE, "stale"
        behind = _sessions_between(calendar, record["as_of_session"], prior)
        words = (
            f"Stale: the newest forecast is as of {record['as_of_session']}, {behind} "
            f"session{'s' if behind != 1 else ''} before {prior}. No forecast exists for "
            f"{market}; none is invented, and the stale one's probabilities are not shown as "
            "today's."
        )
    elif record["as_of_session"] != prior:
        reason = "as_of_not_prior_session"
        words = (
            f"Withheld: the forecast is as of {record['as_of_session']}, which is not the prior "
            f"session ({prior}) on the exchange calendar."
        )
    elif record["refusal"] is not None:
        refusal = record["refusal"]
        reason = "refused"
        words = (
            f"The model gave no probability as of {record['as_of_session']}: "
            f"{refusal.get('detail') or refusal.get('reason')}"
        )
    elif record["target_sessions"][0] != market:
        reason = "target_mismatch"
        words = (
            f"Withheld: the forecast's first target session is {record['target_sessions'][0]}, "
            f"not {market}, so its sessions do not match the exchange calendar."
        )
    else:
        status, words, shown = FRESH, f"Forecast as of the close of {prior}.", record
    run_id = None if record is None else record["evaluation_run_id"]
    if run_id is not None and run_id not in evaluations:
        found = records.evaluation(run_id)
        usable = (
            found is not None
            and found["data_class"] == data_class
            and found["created_at"] <= cutoff
        )
        evaluations[run_id] = found if usable else None
    evaluation = None if run_id is None else evaluations.get(run_id)
    described = None if record is None else record["as_of_session"]
    signals_record = record if (record is not None and status != UNAVAILABLE) else None
    if record is not None and record["refusal"] is not None and status == UNAVAILABLE:
        signals_record = record  # a refusal still carries its descriptive signals
    if reason in ("ledger_unverified", "target_mismatch", "as_of_not_prior_session"):
        signals_record = None
    protected = data_class == DataClass.REAL.value and (
        described is None or date.fromisoformat(described) > DEVELOPMENT_END
    )
    if protected:  # 2025 and later: the observed close and the recorded state (decision A6)
        close = _observed_close(recent, symbol, described, cutoff_ts)
        context = _recorded_context(signals_record)
        bear = (
            _withheld_trap("bear")
            if not margin_confirmed
            else Trap("bear", False, "not_recorded", "Not recorded with the forecast.")
        )
    else:
        close, context, bear = _prices(
            records,
            symbol,
            described,
            data_class=data_class,
            cutoff=cutoff_ts,
            margin_confirmed=margin_confirmed,
        )
    stock = StockBriefing(
        symbol=symbol,
        data_class=data_class,
        status=status,
        reason=reason,
        words=words,
        lineage=None if record is None else _lineage(record),
        described_as_of=described,
        sessions_behind=behind,
        next_session=None
        if shown is None
        else SessionForecast(1, shown["daily"][0]["session"], _probabilities(shown["daily"][0])),
        track=() if shown is None else _track(evaluation, shown),
        five_sessions=()
        if shown is None
        else tuple(
            SessionForecast(row["horizon"], row["session"], _probabilities(row))
            for row in shown["daily"]
        ),
        cumulative=None
        if shown is None or shown["cumulative"] is None
        else _probabilities(shown["cumulative"]),
        cumulative_end=None if shown is None else shown["target_sessions"][-1],
        close=close,
        context=context,
        streak=_streak(signals_record),
        change=_change(shown, known.live)
        if shown is not None
        else Change(False, "no_forecast", "No current forecast to compare."),
        reversal=_reversal(signals_record),
        bull_trap=_bull_trap(signals_record, data_class, margin_confirmed),
        bear_trap=bear,
        limitations=() if record is None else tuple(record["limitations"]),
    )
    return stock, known


def new_york_time(text: str | None) -> str:
    """A stored UTC stamp in New York time: 'Tue 31 Dec 2024, 16:11 New York'."""

    if not text:
        return "none yet"
    local = pd.Timestamp(text).tz_convert("America/New_York")
    return f"{local:%a} {local.day} {local:%b %Y, %H:%M} New York"


def _latest_update(records: ReadOnlyRecords, data_class: str, cutoff: str) -> LatestUpdate:
    runs = [
        run
        for run in records.job_runs()
        if run.get("data_class") == data_class and run["finished_at"] <= cutoff
    ]
    if not runs:
        return LatestUpdate(None, None, None, None, None, "No daily-job run is recorded.")
    last = runs[-1]
    ok = [run for run in runs if run["status"] == "ok"]
    success = ok[-1]["finished_at"] if ok else None
    success_words = new_york_time(success)
    if last.get("activation") == "not_authorized":
        words = (
            f"The daily job last ran {new_york_time(last['finished_at'])} and recorded itself as not "
            "activated: real forecasts start only after decision A4. Last successful update: "
            f"{success_words}."
        )
    else:
        words = (
            f"Last successful update: {success_words}. Latest run: {last['status']}, "
            f"{new_york_time(last['finished_at'])} (expected session "
            f"{last.get('expected_session')})."
        )
    return LatestUpdate(
        success,
        last["finished_at"],
        last["status"],
        last.get("activation"),
        last.get("expected_session"),
        words,
    )


TARGET_ORDER = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "cum": 6}


def _prospective(
    known: Mapping[str, _Known], data_class: str, shown: Collection[str]
) -> ProspectiveEvidence:
    forecasts = [record for k in known.values() for record in k.forecasts]
    scores = [score for k in known.values() for score in k.scores]
    replaced = {record["supersedes"] for record in forecasts if record["supersedes"]}
    found = rel.observations(forecasts, scores, superseded=replaced)
    daily = [o for o in found if o.target != "cum"]
    sessions = len({o.session for o in daily})
    synthetic = data_class == DataClass.SYNTHETIC.value
    if not found:
        status = rel.NONE
        words = (
            "No scored synthetic forecasts yet."
            if synthetic
            else "No prospective record yet: no real forecast has been recorded and scored, so "
            "nothing can be said about its accuracy."
        )
    elif len(daily) < rel.MIN_SCORED or sessions < rel.MIN_SESSIONS:
        status = INSUFFICIENT
        words = (
            f"Insufficient evidence: {len(daily):,} scored daily targets over {sessions} settled "
            f"sessions. At least {rel.MIN_SCORED} targets over {rel.MIN_SESSIONS} sessions are "
            "needed before a rolling score means anything (Model Health's threshold); "
            "calibration needs more."
        )
    else:
        status = rel.OK
        words = (
            f"{len(daily):,} scored daily targets over {sessions} settled sessions (sessions 1 to "
            "5 counted together for this status only; each horizon is shown on its own)."
        )
    if found:
        words += (
            " Synthetic replay: it shows the machinery working and is not evidence about any "
            "real stock."
            if synthetic
            else " Prospective monitoring of recorded forecasts only: these outcomes are from "
            "the protected period and are never used to select, tune or claim research "
            "performance."
        )
    symbols = tuple(sorted({o.symbol for o in found}))
    former = tuple(symbol for symbol in symbols if symbol not in set(shown))
    if former:
        words += (
            f" The record covers every stock with scored forecasts, including {', '.join(former)}, "
            "no longer on this briefing, so dropping a stock never removes its track record."
        )
    groups, rolling = [], []
    keys = sorted({rel.definition(o) for o in found}, key=lambda k: (TARGET_ORDER[k[0]], *k[1:]))
    for key in keys:
        members = [o for o in found if rel.definition(o) == key]
        for category in CATEGORIES:
            groups.append(rel.calibration_group(members, category))
        if key[0] == "1":
            rolling.append(RollingGroup(key[0], key[1], key[2], key[3], rel.rolling(members)))
    live = [f for f in forecasts if f["forecast_id"] not in replaced and f["refusal"] is None]
    current = max(live, key=lambda f: (f["as_of_session"], f["created_at"], f["forecast_id"]),
                  default=None)  # fmt: skip
    comparison, comparison_words = _comparison(found, forecasts, scores, replaced, current)
    caveat = None if comparison is None else _persistence_caveat(comparison)
    return ProspectiveEvidence(
        data_class=data_class,
        status=status,
        words=words,
        scored_targets=len(daily),
        settled_sessions=sessions,
        symbols=symbols,
        former=former,
        groups=tuple(groups),
        rolling=tuple(rolling),
        comparison=comparison,
        comparison_words=comparison_words,
        comparison_caveat=caveat,
    )


def _signed(value: float) -> str:
    if round(value, 4) == 0:
        return "0.0000"
    return f"{value:+.4f}".replace("-", "\u2212")


def _comparison(found, forecasts, scores, replaced, current):
    """
    The next session's matched comparison with B1 (reliability.matched_comparison) for the
    current model, the newest live forecast's (fresh or stale), on the B1 forecasts recorded
    beside it (forecast schema 1.1). Reported, never a decision: whether a challenger beats B1
    is decided on development data only (A7).
    """

    if current is None:
        return None, "No forecast is recorded, so there is nothing to compare with B1."
    if current["estimator"] == rel.REFERENCE:
        return None, "The forecasts are B1's own, so there is no challenger to compare with B1."
    members = [
        o
        for o in found
        if o.target == "1"
        and (o.estimator, o.model_version, o.band)
        == (current["estimator"], current["model_version"], float(current["band"]))
    ]
    if not members:
        return None, (
            f"No scored next-session forecast of the current model ({current['estimator']}, "
            f"model {current['model_version']}) yet."
        )
    key = rel.definition(members[0])
    baseline = [
        o
        for o in rel.comparator_observations(forecasts, scores, superseded=replaced)
        if (o.target, o.band, o.model_version) == (key[0], key[1], key[3])
    ]
    if not baseline:
        return None, (
            f"B1 is not recorded beside these {key[2]} forecasts (records from before forecast "
            "schema 1.1), so no matched comparison is possible."
        )
    found_comparison = rel.matched_comparison(members, baseline)
    head = (
        f"Next session, {key[2]} (model {key[3]}) against the B1 recorded beside each forecast: "
        f"{found_comparison.n:,} matched forecasts over {found_comparison.dates:,} dates."
    )
    tail = " Reported only; it never promotes a model (decision A7)."
    if found_comparison.status == rel.OK:
        parts = []
        for difference in found_comparison.differences:
            label = {"rps": "RPS", "brier": "Brier", "log_loss": "log loss"}[difference.score]
            parts.append(
                f"{label} {_signed(difference.difference)} "
                f"(90% interval {_signed(difference.low)} to {_signed(difference.high)})"
            )
        words = (
            f"{head} {key[2]} minus B1 (negative means {key[2]} scored better, since lower is "
            f"better; intervals: t on {found_comparison.blocks} blocks of 20 or more dates): "
            f"{'; '.join(parts)}."
        )
    elif found_comparison.status == rel.OUTCOME_MISMATCH:
        words = f"{head} The two sides disagree on an outcome, so nothing is compared."
    else:
        need = rel.COMPARISON_BLOCK * rel.MIN_COMPARISON_BLOCKS
        words = (
            f"{head} Insufficient evidence: this comparison needs at least {need} matched dates "
            "of its own before a difference is reported."
        )
    return found_comparison, words + tail


def _persistence_caveat(comparison: rel.MatchedComparison) -> str | None:
    """
    The caveat line when the comparison's persistence flag is on (the owner's
    wording, 2026-10-20). It is a reading aid only: no number, interval or
    decision depends on it.
    """

    if not comparison.persistence_suspected:
        return None
    (ratio,) = (d.se_ratio for d in comparison.differences if d.score == rel.PERSISTENCE_SCORE)
    return (
        "Caveat (reported only): for RPS, the standard error from blocks of "
        f"{rel.PERSISTENCE_BLOCK} dates is {ratio:.2f} times the one from blocks of "
        f"{rel.COMPARISON_BLOCK}. The differences may stay alike for longer than "
        f"{rel.COMPARISON_BLOCK} dates, or may have shifted during the record, so the "
        f"{rel.LEVEL:.0%} intervals may be too narrow. This changes no number above and no "
        "decision."
    )


def _development(
    records: ReadOnlyRecords,
    evaluations: Mapping[str, Mapping | None],
    stocks: Sequence[StockBriefing],
    data_class: str,
    cutoff: str,
) -> tuple[rel.DevelopmentEvidence | None, str]:
    fresh = [s for s in stocks if s.status == FRESH and s.lineage is not None]
    evaluation, estimator = None, None
    for stock in sorted(fresh, key=lambda s: s.lineage.created_at, reverse=True):
        found = evaluations.get(stock.lineage.evaluation_run_id or "")
        if found is not None:
            evaluation, estimator = found, stock.lineage.estimator
            break
    if evaluation is None:
        earlier = [
            e for e in records.evaluations(DataClass(data_class)) if e["created_at"] <= cutoff
        ]
        if earlier:
            evaluation = earlier[-1]  # the newest that existed when the briefing was prepared
            estimator = evaluation["selection"]["selected"]
    if evaluation is None:
        return None, (
            "No development evaluation is stored for "
            f"{'synthetic' if data_class == DataClass.SYNTHETIC.value else 'real'} data."
        )
    found = rel.development_evidence(evaluation, estimator)
    if data_class == DataClass.SYNTHETIC.value:
        words = "Development test on synthetic data: it checks the machinery and is not evidence."
    else:
        words = (
            "Development test on pre-2025 history of a watchlist chosen with hindsight "
            "(exploratory, decision A3): not a confirmatory result."
        )
    return found, words


def build_briefing(
    records: ReadOnlyRecords,
    *,
    timing: Timing,
    calendar: Sessions,
    calendar_id: str,
    data_class: DataClass | str,
    tickers: Sequence[str] | None = None,
    excluded: Collection[str] = (),
    trap_margin_confirmed: bool = False,
    recent=None,
) -> Briefing:
    """
    The briefing for `timing`'s market date from `records` (read-only), for
    `tickers` (the store's watchlist when None) minus `excluded`. Only
    records of `data_class` are read: a briefing is wholly real or wholly
    synthetic. `recent(symbol, retrieved_by=...)` is the isolated descriptive
    reader of recent observed prices (decision A6), for real stocks after
    2024; the owner-run script passes it in, so this module never reads that
    store itself.
    """

    kind = DataClass(data_class).value
    generated = pd.Timestamp(timing.generated_at).tz_convert("UTC")
    cutoff = stamp(generated)
    listed = tuple(tickers) if tickers is not None else records.watched()
    removed = tuple(t for t in listed if t in set(excluded))
    chosen = tuple(t for t in listed if t not in set(excluded))
    entries, problem = records.verify_ledger()
    update = _latest_update(records, kind, cutoff)
    not_activated = update.activation == "not_authorized" or update.last_run_at is None
    evaluations: dict[str, Mapping | None] = {}
    stocks, known = [], {}
    for symbol in chosen:
        stock, records_known = _stock(
            records,
            symbol,
            plan=timing,
            calendar=calendar,
            data_class=kind,
            cutoff=cutoff,
            cutoff_ts=generated,
            ledger_problem=problem,
            not_activated=not_activated,
            evaluations=evaluations,
            margin_confirmed=trap_margin_confirmed,
            recent=recent,
        )
        stocks.append(stock)
        known[symbol] = records_known
    for symbol in records.scored_symbols():  # the track record keeps stocks no longer listed
        if symbol not in known:
            known[symbol] = _known(records, symbol, cutoff, kind)
    counts = {
        state: sum(1 for s in stocks if s.status == state) for state in (FRESH, STALE, UNAVAILABLE)
    }
    fresh = [s for s in stocks if s.status == FRESH]
    versions = tuple(
        sorted({f"{s.lineage.estimator} {s.lineage.model_version}" for s in fresh if s.lineage})
    )
    development, development_words = _development(records, evaluations, stocks, kind, cutoff)
    prospective = _prospective(known, kind, chosen)
    synthetic = kind == DataClass.SYNTHETIC.value
    prior = timing.prior_session.isoformat()
    market = timing.market_date.isoformat()
    warnings = []  # the synthetic label is the renderers' banner, from data_class
    note = gap_note(timing)
    if note:
        warnings.append(note)
    if timing.state != PREMARKET:
        warnings.append(STATE_WORDS[timing.state] + ".")
    if problem is not None:
        warnings.append(f"{LEDGER_FAILED} ({problem}): every stored probability is withheld.")
    if not synthetic and not_activated:
        warnings.append(
            "Real forecasts are not activated yet (decision A4), so no real stock has a "
            "probability."
        )
    if counts[STALE]:
        warnings.append(
            f"{counts[STALE]} of {len(stocks)} stocks are stale: their newest forecast is older "
            f"than the close of {prior}."
        )
    if stocks and not fresh:
        warnings.append(f"No stock has a forecast for {market}.")
    if not stocks:
        warnings.append("The watchlist is empty: there is nothing to brief.")
    limitations = [
        (
            "Experimental and unvalidated. Daily direction is close to unpredictable; the most "
            "these models can show is how likely a quiet (Stable) session is."
        ),
        (
            "Each session's probabilities are for that day's own move; the five-session total is "
            "a separate forecast, never read as a path."
        ),
        (
            "Momentum, volatility, regime, streak and trap notes describe the past; they are not "
            "signals."
        ),
        (
            "Each forecast is made after the prior session's close from end-of-day prices: it "
            "reflects no after-hours move, overnight news or premarket move."
        ),
        (
            "This briefing describes probabilities and how reliable they have been; it does not "
            "say what anyone should do with a stock."
        ),
    ]
    if not synthetic:
        limitations.append("Private, local use: vendor data is never redistributed.")
    notes = (
        (
            "Whether a challenger beats B1 is decided in the development evaluation only: since "
            "decision A7 (2026-10-19), by the fold-block t with Holm at a family level of 0.025; "
            "an evaluation stored before then states its own rule. A comparison with B1 in this "
            "briefing is reported, never a decision."
        ),
        (
            "Horizons and stable bands are never pooled; forecasts on one date, and correlated "
            "stocks, are not counted as independent trials."
        ),
    )
    return Briefing(
        schema=SCHEMA,
        schema_version=SCHEMA_VERSION,
        data_class=kind,
        market_date=market,
        prior_session=prior,
        scheduled_at=timing.scheduled_at.isoformat(),
        generated_at=cutoff,
        state=timing.state,
        state_words=STATE_WORDS[timing.state],
        closed_days=tuple(day.isoformat() for day in timing.closed_days),
        gap_note=note,
        calendar_id=calendar_id,
        data_as_of=prior if fresh else None,
        freshness=counts,
        model_versions=versions,
        latest_update=update,
        ledger=LedgerCheck(entries, problem is None, problem),
        stocks=tuple(stocks),
        excluded=removed,
        warnings=tuple(warnings),
        limitations=tuple(limitations),
        reliability=ReliabilitySection(development, development_words, prospective, notes),
    )


def subject(briefing: Briefing) -> str:
    """A one-line title: 'Premarket briefing for Monday 19 October 2026'."""

    lead = "[Synthetic demo] " if briefing.data_class == DataClass.SYNTHETIC.value else ""
    market = long_date(date.fromisoformat(briefing.market_date))
    kind = "Premarket briefing" if briefing.state == PREMARKET else "Briefing preview"
    return f"{lead}{kind} for {market}"


def generated_local(briefing: Briefing) -> datetime:
    return pd.Timestamp(briefing.generated_at).tz_convert("America/New_York").to_pydatetime()


__all__ = [
    "Briefing",
    "ReadOnlyRecords",
    "StockBriefing",
    "briefing_document",
    "briefing_from_document",
    "build_briefing",
    "subject",
]
