"""
Pure view logic for the Lab's Stock Probability page: the watchlist summary
rows and one stock's detail. No Streamlit, no Plotly, no network.

Point in time and the sealed period. For a real stock, the page shows bars
only through DEVELOPMENT_END (2024-12-31) until the owner decides how the
sealed period may be used; later bars stay in the private store, unshown.
Synthetic demo stocks are shown whole and always labelled synthetic.

Nothing here is a forecast: every number describes stored history (counts,
shares and runs of past sessions).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stock_agent.stock_lab import features as feature_module
from stock_agent.stock_lab import synthetic
from stock_agent.stock_lab.contracts import (
    CATEGORIES,
    CATEGORY_WORDS,
    DATA_STATUS_WORDS,
    DEFAULT_BAND,
    DEVELOPMENT_END,
    EVIDENCE_WORDS,
    SYMBOL_STATUS_WORDS,
    Category,
    DataClass,
    DataStatus,
    EvidenceStatus,
    StableBand,
)
from stock_agent.stock_lab.prices import categorize, stable_share
from stock_agent.stock_lab.runs import DEFAULT_RULE, RunRule, run_table
from stock_agent.stock_lab.store import StockLabStore, StoredBars
from stock_agent.stock_lab.symbols import Classification, unverified

MINUS = "−"
DASH = "—"
COMPARISON_BANDS = (StableBand(0.0025), StableBand(0.005), StableBand(0.01))
LONGEST_SHOWN = 6  # run lengths 1..5 and "6 or more"
RECENT_SESSIONS = 60


def percent(fraction: float | None, *, signed: bool = False, digits: int = 1) -> str:
    if fraction is None or not np.isfinite(fraction):
        return DASH
    fraction = float(fraction) + 0.0
    text = f"{fraction:+.{digits}%}" if signed else f"{fraction:.{digits}%}"
    return text.replace("-", MINUS)


def day(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def run_words(direction: str | None, length: int) -> str:
    """'3 rising sessions in a row', '1 stable session', or a dash after a gap."""

    if direction is None or (isinstance(direction, float) and np.isnan(direction)) or not length:
        return DASH
    noun = "session" if length == 1 else "sessions"
    adjective = {
        Category.INCREASE.value: "rising",
        Category.DECREASE.value: "falling",
        Category.STABLE.value: "stable",
    }[direction]
    tail = " in a row" if length > 1 else ""
    return f"{length} {adjective} {noun}{tail}"


# --- the watchlist summary ---


@dataclass(frozen=True)
class WatchRow:
    ticker: str
    name: str
    entered_as: str
    added_at: str
    synthetic: bool
    classification: Classification
    data_status: DataStatus
    data_detail: str
    last_session: str
    current_run: str
    stable_share: str
    evidence: EvidenceStatus


def real_bars_shown(stored: StoredBars | None) -> pd.DataFrame | None:
    """
    Stored bars as the page may show them (prices.BAR_COLUMNS): real bars only
    through DEVELOPMENT_END; None without a stored batch.
    """

    if stored is None:
        return None
    bars = stored.bars.drop(columns=["batch_id"])
    if stored.data_class == DataClass.REAL.value:
        end = pd.Timestamp(DEVELOPMENT_END, tz="UTC")
        bars = bars[bars["session"] <= end].reset_index(drop=True)
    return bars


def _summary(
    ticker: str,
    bars: pd.DataFrame | None,
    *,
    band: StableBand,
    rule: RunRule,
) -> tuple[str, str, str]:
    """(last session, current run, stable share) of shown bars, or dashes."""

    if bars is None or bars.empty:
        return DASH, DASH, DASH
    frame = feature_module.build_features(bars, band=band, rule=rule)
    last = feature_module.latest_row(frame)
    if last is None:
        return DASH, DASH, DASH
    return (
        day(last["session"]),
        run_words(last["run_direction"], int(last["run_length"])),
        percent(stable_share(frame["category"])),
    )


def watch_rows(
    store: StockLabStore,
    *,
    band: StableBand = DEFAULT_BAND,
    rule: RunRule = DEFAULT_RULE,
) -> tuple[WatchRow, ...]:
    """One row per watched ticker, alphabetical (nothing is ranked)."""

    checks = store.latest_checks()
    rows = []
    for item in store.watchlist():
        classification = checks.get(item.ticker, unverified(item.ticker))
        stored = store.bars(item.ticker)
        if stored is None:
            status, detail = DataStatus.NO_DATA, "No prices stored yet."
            summary = (DASH, DASH, DASH)
        else:
            bars = real_bars_shown(stored)
            summary = _summary(item.ticker, bars, band=band, rule=rule)
            revised = [
                day_text
                for day_text in stored.revised_sessions
                if stored.data_class != DataClass.REAL.value
                or day_text <= DEVELOPMENT_END.isoformat()
            ]
            if revised:
                status = DataStatus.REVISED
                detail = f"{len(revised)} sessions were revised between downloads."
            else:
                status = DataStatus.UNCHECKED
                detail = (
                    "Shown through 2024-12-31 only; freshness is checked by the daily update, "
                    "which is not running yet."
                )
        rows.append(
            WatchRow(
                ticker=item.ticker,
                name=item.ticker,
                entered_as=item.entered_as,
                added_at=item.added_at[:10],
                synthetic=False,
                classification=classification,
                data_status=status,
                data_detail=detail,
                last_session=summary[0],
                current_run=summary[1],
                stable_share=summary[2],
                evidence=EvidenceStatus.DESCRIPTIVE,
            )
        )
    return tuple(rows)


def demo_rows(
    histories: Mapping[str, pd.DataFrame],
    *,
    band: StableBand = DEFAULT_BAND,
    rule: RunRule = DEFAULT_RULE,
) -> tuple[WatchRow, ...]:
    """One row per synthetic demo stock, labelled synthetic."""

    rows = []
    for stock_id, bars in histories.items():
        spec = synthetic.demo_spec(stock_id)
        last, run, share = _summary(stock_id, bars, band=band, rule=rule)
        rows.append(
            WatchRow(
                ticker=stock_id,
                name=spec.name,
                entered_as=stock_id,
                added_at=DASH,
                synthetic=True,
                classification=unverified(stock_id),
                data_status=DataStatus.SYNTHETIC,
                data_detail="Synthetic demo data, generated from a fixed seed.",
                last_session=last,
                current_run=run,
                stable_share=share,
                evidence=EvidenceStatus.SYNTHETIC_DEMO,
            )
        )
    return tuple(rows)


def summary_table(rows: Sequence[WatchRow], *, band: StableBand) -> pd.DataFrame:
    """The watchlist summary as text: one row per stock, in the given order."""

    def why(row: WatchRow) -> str:
        if row.synthetic:
            return row.data_detail
        parts = [row.classification.detail, row.data_detail]
        return " ".join(part for part in parts if part)

    return pd.DataFrame(
        {
            "Stock": [row.name if row.synthetic else row.ticker for row in rows],
            "Status": [
                "Synthetic demo"
                if row.synthetic
                else SYMBOL_STATUS_WORDS[row.classification.status]
                for row in rows
            ],
            "Data": [DATA_STATUS_WORDS[row.data_status] for row in rows],
            "Why": [why(row) for row in rows],
            "Last session": [row.last_session for row in rows],
            "Current streak": [row.current_run for row in rows],
            f"Stable sessions at {band.words()}": [row.stable_share for row in rows],
            "Evidence": [EVIDENCE_WORDS[row.evidence] for row in rows],
        }
    )


def search(rows: Sequence[WatchRow], query: str) -> tuple[WatchRow, ...]:
    """Rows whose ticker or name contains `query` (case-insensitive); all when empty."""

    text = query.strip().upper()
    if not text:
        return tuple(rows)
    return tuple(row for row in rows if text in row.ticker.upper() or text in row.name.upper())


# --- one stock's detail ---


@dataclass(frozen=True, eq=False)
class StockDetail:
    stock_id: str
    name: str
    synthetic: bool
    band: StableBand
    rule: RunRule
    frame: pd.DataFrame  # features.build_features output for the shown bars
    runs: pd.DataFrame  # runs.run_table
    last: pd.Series | None
    shown_through: str
    limitations: tuple[str, ...] = field(default_factory=tuple)


def stock_detail(
    stock_id: str,
    bars: pd.DataFrame,
    *,
    name: str,
    synthetic_data: bool,
    band: StableBand = DEFAULT_BAND,
    rule: RunRule = DEFAULT_RULE,
    limitations: Sequence[str] = (),
) -> StockDetail:
    frame = feature_module.build_features(bars, band=band, rule=rule)
    # For display: prices on the last shown session's share basis (uses no later split).
    basis = float(np.prod(pd.to_numeric(bars["split_factor"]).fillna(1.0).to_numpy()))
    frame["display_level"] = frame["level"] / basis
    runs = run_table(frame["session"], frame["category"], rule)
    last = feature_module.latest_row(frame)
    notes = list(limitations)
    if frame["ex_dividend"].any():
        notes.append(
            f"{int(frame['ex_dividend'].sum())} sessions are ex-dividend dates: their price "
            "return includes the dividend's mechanical drop."
        )
    if frame["split_applied"].any():
        notes.append(
            f"{int(frame['split_applied'].sum())} split(s) applied: prices are compared on a "
            "split-consistent basis, so a split never shows as a fall."
        )
    if frame["large_move"].any():
        notes.append(
            f"{int(frame['large_move'].sum())} session(s) rose 90% or more, or fell 47% or "
            "more, without a recorded split: flagged for review, not corrected."
        )
    unavailable = frame["reason"].ne("ok") & frame["reason"].ne("first_close")
    if unavailable.any():
        notes.append(f"{int(unavailable.sum())} session(s) have no usable return (gaps).")
    return StockDetail(
        stock_id=stock_id,
        name=name,
        synthetic=synthetic_data,
        band=band,
        rule=rule,
        frame=frame,
        runs=runs,
        last=last,
        shown_through=DASH if last is None else day(last["session"]),
        limitations=tuple(notes),
    )


@dataclass(frozen=True)
class Card:
    """A metric card: a short value, with the rest in its caption and help."""

    label: str
    value: str
    caption: str | None = None
    help: str | None = None


TYPICAL_LABEL = "Typical daily move"
TYPICAL_HELP = "The median size of a daily price move, up or down."
STREAK_WORDS = {
    Category.INCREASE.value: "rising",
    Category.DECREASE.value: "falling",
    Category.STABLE.value: "stable",
}


def headline(detail: StockDetail) -> tuple[Card, ...]:
    """The detail page's metric cards (short values; dates and words in captions)."""

    frame, last = detail.frame, detail.last
    moves = frame["price_return"].abs().dropna()
    categories = frame["category"].dropna()
    if last is None:
        close = Card("Last close", DASH)
        streak = Card("Current streak", DASH)
    else:
        close = Card("Last close", f"{last['close']:,.2f}", caption=f"on {day(last['session'])}")
        direction = last["run_direction"]
        length = int(last["run_length"])
        if isinstance(direction, str) and length:
            noun = "session" if length == 1 else "sessions"
            streak = Card(
                "Current streak",
                f"{length} {STREAK_WORDS[direction]}",
                caption=f"{noun} in a row" if length > 1 else noun,
            )
        else:
            streak = Card("Current streak", DASH, caption="after a missing session")
    return (
        close,
        streak,
        Card(
            TYPICAL_LABEL,
            percent(float(moves.median()) if len(moves) else None, digits=2),
            caption=f"over {len(moves):,} sessions",
            help=TYPICAL_HELP,
        ),
        Card(
            f"Stable sessions at {detail.band.words()}",
            percent(stable_share(frame["category"])),
            caption=f"of {len(categories):,} sessions",
        ),
    )


def price_table(detail: StockDetail) -> pd.DataFrame:
    """The price chart as a table: the last session of each year and the last shown session."""

    frame = detail.frame[detail.frame["display_level"].notna()]
    if frame.empty:
        return pd.DataFrame({"Session date": [], "Price": []})
    years = frame.groupby(frame["session"].dt.year).tail(1)
    return pd.DataFrame(
        {
            "Session date": [day(value) for value in years["session"]],
            "Price": [f"{value:,.2f}" for value in years["display_level"]],
        }
    ).set_index("Session date")


def recent_table(detail: StockDetail, count: int = RECENT_SESSIONS) -> pd.DataFrame:
    """The recent-sessions chart as a table."""

    recent = recent_sessions(detail, count)
    words = {category.value: CATEGORY_WORDS[category] for category in CATEGORIES}
    return pd.DataFrame(
        {
            "Session date": [day(value) for value in recent["session"]],
            "Price return": [
                percent(value, signed=True, digits=2) for value in recent["price_return"]
            ],
            "Direction": [words.get(value, DASH) for value in recent["category"]],
        }
    ).set_index("Session date")


def threshold_table(detail: StockDetail) -> pd.DataFrame:
    """
    The share of sessions in each category at nearby stable bands: how the
    same band means different things for calm and volatile stocks.
    """

    returns = detail.frame["price_return"]
    rows = []
    for band in COMPARISON_BANDS:
        categories = categorize(returns, band).dropna()
        row = {"Stable band": band.words(), "Sessions": f"{len(categories):,}"}
        for category in CATEGORIES:
            share = float((categories == category.value).mean()) if len(categories) else None
            row[CATEGORY_WORDS[category]] = percent(share)
        rows.append(row)
    return pd.DataFrame(rows)


def run_length_table(detail: StockDetail) -> pd.DataFrame:
    """
    Completed rising and falling runs by length (1..5, 6 or more), and the
    runs still open or cut by a gap (censored), counted separately.
    """

    runs = detail.runs
    rows = []
    for category in (Category.INCREASE, Category.DECREASE):
        of_kind = runs[runs["direction"].eq(category.value)]
        complete = of_kind[~of_kind["censored"]]
        row = {"Direction": "Rising" if category is Category.INCREASE else "Falling"}
        for length in range(1, LONGEST_SHOWN):
            label = "1 session" if length == 1 else f"{length} sessions"
            row[label] = int(complete["length"].eq(length).sum())
        row[f"{LONGEST_SHOWN} or more"] = int(complete["length"].ge(LONGEST_SHOWN).sum())
        row["Still open or cut by a gap"] = int(of_kind["censored"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


# Plain-language names for the features shown on the detail page.
FEATURE_WORDS = {
    "ret_1": ("Return, last session", "percent_signed"),
    "ret_5": ("Return, last 5 sessions", "percent_signed"),
    "ret_20": ("Return, last 20 sessions", "percent_signed"),
    "ret_60": ("Return, last 60 sessions", "percent_signed"),
    "vol_20": ("Daily volatility, last 20 sessions", "percent"),
    "ewma_vol": ("Daily volatility, recent-weighted", "percent"),
    "vol_ratio": ("Volatility now vs last 60 sessions", "ratio"),
    "ma_gap_20": ("Distance from 20-session average", "percent_signed"),
    "ma_slope_20": ("Slope of 20-session average (5 sessions)", "percent_signed"),
    "drawdown_60": ("Below the 60-session high", "percent_signed"),
    "bounce_20": ("Above the 20-session low", "percent_signed"),
    "momentum_accel": ("Change in 5-session return", "percent_signed"),
    "rel_dollar_volume": ("Trading value vs 20-session median", "ratio"),
    "band_over_vol": ("Stable band ÷ recent volatility", "ratio"),
    "up_count_20": ("Rising sessions, last 20", "count"),
    "down_count_20": ("Falling sessions, last 20", "count"),
    "stable_count_20": ("Stable sessions, last 20", "count"),
}


def _format(value: float, kind: str) -> str:
    if value is None or not np.isfinite(value):
        return DASH
    if kind == "percent_signed":
        return percent(value, signed=True)
    if kind == "percent":
        return percent(value, digits=2)
    if kind == "count":
        return f"{value:.0f}"
    return f"{value:.2f}×".replace("-", MINUS)


def feature_table(detail: StockDetail) -> pd.DataFrame:
    """The latest session's features with plain names (past data only)."""

    last = detail.last
    rows = []
    for column, (words, kind) in FEATURE_WORDS.items():
        value = None if last is None else last[column]
        rows.append({"Measure": words, "Value": _format(value, kind)})
    return pd.DataFrame(rows)


def recent_sessions(detail: StockDetail, count: int = RECENT_SESSIONS) -> pd.DataFrame:
    """The last `count` sessions: date, price return and category (for the strip chart)."""

    frame = detail.frame[["session", "price_return", "category", "level"]].tail(count)
    return frame.reset_index(drop=True)
