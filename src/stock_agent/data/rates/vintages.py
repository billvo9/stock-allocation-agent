"""
The vintage-aware rates store (OD8 group C, C2; approved by the owner on 2026-10-26:
"Approve all C1–C9 (Recommended)"; docs/research/group_c_decisions.md, C2 (a)).

Every value a provider ever published for a date is kept as its own row: a vintage.
Later revisions are new rows with their own availability time and never overwrite an
earlier one (AGENTS.md).

Schema (`VINTAGE_COLUMNS`): the legacy rate columns, plus
- `vintage_date`: the day the provider first showed this value (ALFRED `realtime_start`);
- `available_at`: when the value may first be used as a decision input (UTC);
- `availability_rule`: which rule set `available_at`;
- `retrieved_at`: when this project downloaded it (UTC).

The key is (date, rate_id, vintage_date), replacing the legacy (date, rate_id): a
breaking change to the store's contract, approved as part of C2. `annual_yield` is NaN
for a vintage that published no value for the date (FRED's "."), so the first release
of a bond-market holiday is recorded as missing rather than skipped.

Availability (`h15_or_alfred_later`), for every vintage, first releases included: the
later of
- 16:15 New York on the next Federal Reserve business day after the date (the H.15
  posting), and
- the end of the vintage's `vintage_date` (midnight New York at the end of that day),
because `realtime_start` is a date without a time of day. It is read as a New York date;
FRED is in St. Louis (Central time), an hour behind, so New York midnight is the later
end of that day for every decision cutoff here (none falls between midnight and 1:00).

The Federal Reserve calendar here treats every US federal holiday, with the observed
days on both sides of a weekend holiday, as closed. If it marks a working day as a
holiday, only the H.15 term moves later, and the rule takes the later of the two
times, so an error can delay a value but never make it count sooner.

For dates before ALFRED's first DGS3MO vintage (`ALFRED_START`, 2005-06-28), that
first vintage is labelled `pre_alfred_first_vintage`: the same rule makes it a
decision input only from the end of 2005-06-28. The label is checked row by row.

Reading:
- `first_release(frame)`: the earliest vintage per (date, rate_id). Booked wealth uses
  it (C2.2, C9), so a later revision never rewrites booked history.
  `legacy_rates(frame)` gives it in the legacy column contract, without missing values.
- `as_of(frame, cutoff)`: per (date, rate_id), the latest vintage whose `available_at`
  is at or before `cutoff`. For decision inputs; a revision published later never
  reaches it.

A merge never adds a vintage earlier than a stored date's first release: that would
rewrite booked history.

Storage (`save_vintages`, `load_vintages`): parquet files in a private folder outside
every Git working tree (default ~/stock-agent-private/rates/, or STOCK_AGENT_RATES_HOME),
one new file per retrieval and never edited. A load combines them with `merge_vintages`,
which refuses a key that two retrievals report with different values.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stock_agent.data.private_paths import make_private_directories
from stock_agent.data.rates.schema import RATE_COLUMNS, validate_rate_frame

NEW_YORK = ZoneInfo("America/New_York")
H15_POSTING = time(16, 15)  # H.15 posting, New York, on the next Fed business day
ALFRED_START = date(2005, 6, 28)  # ALFRED's first DGS3MO vintage
LAST_VINTAGE = date(2024, 12, 31)  # vintages published after this are in the holdout
H15_OR_ALFRED_LATER = "h15_or_alfred_later"
PRE_ALFRED = "pre_alfred_first_vintage"
RULES = frozenset({H15_OR_ALFRED_LATER, PRE_ALFRED})
VINTAGE_COLUMNS = [
    *RATE_COLUMNS,
    "vintage_date",
    "available_at",
    "availability_rule",
    "retrieved_at",
]
KEY = ["date", "rate_id", "vintage_date"]
HOME_ENV = "STOCK_AGENT_RATES_HOME"


class VintageRefused(ValueError):
    """Vintage data that breaks the store's contract; nothing from it is kept."""


# --- the Federal Reserve calendar (conservative) ---


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    shift = (weekday - first.weekday()) % 7
    return first + timedelta(days=shift + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    following = date(year + (month == 12), month % 12 + 1, 1)
    last = following - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


@lru_cache(maxsize=128)
def fed_holidays(year: int) -> frozenset[date]:
    """US federal holidays of `year`, a weekend holiday's Friday and Monday both included."""

    fixed = [date(year, 1, 1), date(year, 7, 4), date(year, 11, 11), date(year, 12, 25)]
    if year >= 2021:
        fixed.append(date(year, 6, 19))  # Juneteenth, from 2021
    days = set()
    for holiday in fixed:
        days.add(holiday)
        if holiday.weekday() == 5:  # Saturday: the Friday too (conservative)
            days.add(holiday - timedelta(days=1))
        if holiday.weekday() in (5, 6):  # a weekend: the Monday too
            days.add(holiday + timedelta(days=7 - holiday.weekday()))
    days.update(
        {
            _nth_weekday(year, 1, 0, 3),  # Martin Luther King Jr. Day
            _nth_weekday(year, 2, 0, 3),  # Washington's Birthday
            _last_weekday(year, 5, 0),  # Memorial Day
            _nth_weekday(year, 9, 0, 1),  # Labor Day
            _nth_weekday(year, 10, 0, 2),  # Columbus Day
            _nth_weekday(year, 11, 3, 4),  # Thanksgiving
        }
    )
    return frozenset(days)


def is_fed_holiday(day: date) -> bool:
    """A holiday of `day`'s year, or of the next year's (a Saturday January 1 closes December 31)."""

    return day in fed_holidays(day.year) or day in fed_holidays(day.year + 1)


def next_fed_business_day(day: date) -> date:
    following = day + timedelta(days=1)
    while following.weekday() >= 5 or is_fed_holiday(following):
        following += timedelta(days=1)
    return following


def available_at(observation: date, vintage: date) -> pd.Timestamp:
    """The `h15_or_alfred_later` availability of one vintage of one date (UTC)."""

    posting = datetime.combine(next_fed_business_day(observation), H15_POSTING, NEW_YORK)
    day_end = datetime.combine(vintage + timedelta(days=1), time(0, 0), NEW_YORK)
    return pd.Timestamp(max(posting, day_end)).tz_convert("UTC")


# --- building and checking ---


def _dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="raise").dt.normalize()


def vintages_from_alfred(
    observations: pd.DataFrame,
    *,
    rate_id: str,
    provider_series_id: str,
    currency: str,
    tenor: str,
    quote_convention: str,
    retrieved_at: pd.Timestamp,
) -> pd.DataFrame:
    """
    Vintage rows from FRED/ALFRED series observations (`date`, `value`,
    `realtime_start` columns; "." is no value). Percent yields become decimals.
    """

    if observations.empty:
        raise VintageRefused("no observations")
    raw = observations["value"].astype(str)
    numbers = pd.to_numeric(raw.where(raw != "."), errors="raise") / 100.0
    frame = pd.DataFrame(
        {
            "date": _dates(observations["date"]),
            "rate_id": rate_id,
            "provider_series_id": provider_series_id,
            "currency": currency,
            "tenor": tenor,
            "annual_yield": numbers.astype("float64").to_numpy(),
            "quote_convention": quote_convention,
            "source": "FRED/ALFRED",
            "vintage_date": _dates(observations["realtime_start"]),
        }
    )
    frame["available_at"] = [
        available_at(d.date(), v.date())
        for d, v in zip(frame["date"], frame["vintage_date"], strict=True)
    ]
    frame["availability_rule"] = _rules(frame)
    frame["retrieved_at"] = pd.Timestamp(retrieved_at).tz_convert("UTC")
    frame = frame.sort_values(KEY).reset_index(drop=True)
    validate_vintages(frame)
    return frame


def _rules(frame: pd.DataFrame) -> np.ndarray:
    """
    `pre_alfred_first_vintage` for a date before ALFRED whose vintage is ALFRED's first
    (2005-06-28); every other row `h15_or_alfred_later`. Decided by the row itself, so a
    partial retrieval of a later revision is never mislabelled.
    """

    pre = (frame["date"].dt.date < ALFRED_START) & (frame["vintage_date"].dt.date == ALFRED_START)
    return np.where(pre, PRE_ALFRED, H15_OR_ALFRED_LATER)


def validate_vintages(frame: pd.DataFrame) -> None:
    """The store's contract (module docstring); raises `VintageRefused`."""

    missing = set(VINTAGE_COLUMNS) - set(frame.columns)
    if missing:
        raise VintageRefused(f"missing columns {sorted(missing)}")
    if frame.empty:
        raise VintageRefused("no rows")
    required = [c for c in VINTAGE_COLUMNS if c != "annual_yield"]
    if frame[required].isna().any(axis=None):
        raise VintageRefused("a required field is missing")
    yields = frame["annual_yield"].to_numpy(dtype="float64")
    if np.isinf(yields).any():
        raise VintageRefused("a yield is infinite")
    if frame.duplicated(subset=KEY).any():
        raise VintageRefused("a (date, rate_id, vintage_date) key appears twice")
    if (frame["vintage_date"] < frame["date"]).any():
        raise VintageRefused("a vintage cannot predate its observation")
    if (frame["vintage_date"].dt.date > LAST_VINTAGE).any():
        raise VintageRefused(f"a vintage published after {LAST_VINTAGE} (the holdout)")
    if not set(frame["availability_rule"]) <= RULES:
        raise VintageRefused("an unknown availability rule")
    if not (frame["availability_rule"].to_numpy() == _rules(frame)).all():
        raise VintageRefused("an availability_rule label does not match its row")
    expected = [
        available_at(d.date(), v.date())
        for d, v in zip(frame["date"], frame["vintage_date"], strict=True)
    ]
    if not (pd.DatetimeIndex(frame["available_at"]) == pd.DatetimeIndex(expected)).all():
        raise VintageRefused("available_at does not follow the h15_or_alfred_later rule")


def merge_vintages(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    """
    Append-only: new keys are added; a key already stored with the same yield keeps the
    earlier row; the same key with a different yield is refused (nothing is overwritten).
    """

    validate_vintages(incoming)
    if existing is None or existing.empty:
        return incoming.sort_values(KEY).reset_index(drop=True)
    validate_vintages(existing)
    joined = existing.merge(incoming, on=KEY, how="inner", suffixes=("_old", "_new"))
    old, new = joined["annual_yield_old"], joined["annual_yield_new"]
    differ = ~((old == new) | (old.isna() & new.isna()))
    if differ.any():
        first = joined.loc[differ].iloc[0]
        raise VintageRefused(
            f"{first['rate_id']} {first['date'].date()} vintage {first['vintage_date'].date()} "
            "was stored with another value; vintages are never overwritten"
        )
    known = pd.MultiIndex.from_frame(existing[KEY])
    fresh = incoming.loc[~pd.MultiIndex.from_frame(incoming[KEY]).isin(known)]
    # A vintage earlier than a stored date's first release would replace that first
    # release, and so rewrite booked history (C2.2): refused.
    firsts = existing.groupby(["date", "rate_id"])["vintage_date"].min().rename("stored_first")
    early = fresh.join(firsts, on=["date", "rate_id"], how="inner")
    early = early.loc[early["vintage_date"] < early["stored_first"]]
    if not early.empty:
        row = early.iloc[0]
        raise VintageRefused(
            f"{row['rate_id']} {row['date'].date()}: vintage {row['vintage_date'].date()} predates "
            f"the stored first release ({row['stored_first'].date()}); booked history is never "
            "rewritten"
        )
    merged = pd.concat([existing, fresh], ignore_index=True)
    merged = merged.sort_values(KEY).reset_index(drop=True)
    validate_vintages(merged)
    return merged


# --- reading ---


def first_release(frame: pd.DataFrame) -> pd.DataFrame:
    """The earliest vintage per (date, rate_id), its yield NaN if it published none."""

    validate_vintages(frame)
    ordered = frame.sort_values(KEY)
    return ordered.groupby(["date", "rate_id"], as_index=False).head(1).reset_index(drop=True)


def legacy_rates(frame: pd.DataFrame) -> pd.DataFrame:
    """The first release in the legacy rate contract (no missing values; `validate_rate_frame`)."""

    found = first_release(frame)
    found = found.loc[found["annual_yield"].notna(), RATE_COLUMNS].reset_index(drop=True)
    validate_rate_frame(found)
    return found


def as_of(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """
    Per (date, rate_id), the latest vintage available at or before `cutoff`. Its yield
    is NaN when that vintage published no value for the date (a bond-market holiday):
    the reader decides how to carry, never this function.
    """

    validate_vintages(frame)
    cutoff = pd.Timestamp(cutoff)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    known = frame.loc[frame["available_at"] <= cutoff.tz_convert("UTC")]
    ordered = known.sort_values(KEY)
    return ordered.groupby(["date", "rate_id"], as_index=False).tail(1).reset_index(drop=True)


def diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    """Deterministic data-quality counts (C2.2): coverage, vintages, revisions."""

    validate_vintages(frame)
    first = first_release(frame)
    joined = frame.merge(
        first[["date", "rate_id", "vintage_date", "annual_yield"]],
        on=["date", "rate_id"],
        suffixes=("", "_first"),
    )
    later = joined.loc[joined["vintage_date"] > joined["vintage_date_first"]]
    value, original = later["annual_yield"], later["annual_yield_first"]
    changed = ~((value == original) | (value.isna() & original.isna()))
    revised = later.loc[changed]
    moves = (revised["annual_yield"] - revised["annual_yield_first"]).abs().dropna()
    posting_decides = [
        a != pd.Timestamp(datetime.combine(v.date() + timedelta(days=1), time(0, 0), NEW_YORK))
        for a, v in zip(frame["available_at"], frame["vintage_date"], strict=True)
    ]
    return {
        "rows": len(frame),
        "dates": int(first["date"].nunique()),
        "first_date": first["date"].min().date().isoformat(),
        "last_date": first["date"].max().date().isoformat(),
        "vintages": int(frame["vintage_date"].nunique()),
        "first_release_missing": int(first["annual_yield"].isna().sum()),
        "revised_observations": int(revised[["date", "rate_id"]].drop_duplicates().shape[0]),
        "largest_revision": float(moves.max()) if len(moves) else 0.0,
        "revisions_from_missing": int((revised["annual_yield_first"].isna()).sum()),
        "availability_set_by_the_calendar": int(sum(posting_decides)),
        "pre_alfred_first_vintages": int((frame["availability_rule"] == PRE_ALFRED).sum()),
        "latest_vintage": frame["vintage_date"].max().date().isoformat(),
    }


# --- storage ---


def default_home() -> Path:
    configured = os.environ.get(HOME_ENV, "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "stock-agent-private" / "rates"


def _check_home(home: Path) -> Path:
    resolved = Path(home).expanduser().resolve()
    if any((folder / ".git").exists() for folder in (resolved, *resolved.parents)):
        raise VintageRefused("the rates store must be outside every Git working tree")
    return resolved


def load_vintages(rate_id: str, home: Path | None = None) -> pd.DataFrame | None:
    """Every stored retrieval of `rate_id`, merged append-only; None when none is stored."""

    folder = _check_home(home or default_home())
    files = sorted(folder.glob(f"{rate_id}-vintages-*.parquet"))
    merged = None
    for path in files:
        part = pd.read_parquet(path)
        merged = part if merged is None else merge_vintages(merged, part)
    if merged is not None:
        validate_vintages(merged)
    return merged


def save_vintages(frame: pd.DataFrame, rate_id: str, home: Path | None = None) -> Path | None:
    """
    Store the vintages of one retrieval that are not stored yet, as a new read-only file
    (written to a temporary name, then linked into place, so a failed write leaves no
    partial file). None when everything is already stored. It must merge with what is
    stored, or nothing is written.
    """

    validate_vintages(frame)
    if set(frame["rate_id"]) != {rate_id}:
        raise VintageRefused(f"the frame must hold {rate_id} only")
    folder = _check_home(home or default_home())
    stored = load_vintages(rate_id, folder)
    if stored is not None:
        merge_vintages(stored, frame)  # refuses before anything is written
        known = pd.MultiIndex.from_frame(stored[KEY])
        frame = frame.loc[~pd.MultiIndex.from_frame(frame[KEY]).isin(known)]
        if frame.empty:
            return None
        frame = frame.reset_index(drop=True)
    stamp = pd.Timestamp(frame["retrieved_at"].max()).strftime("%Y%m%dT%H%M%S%fZ")
    path = folder / f"{rate_id}-vintages-{stamp}.parquet"
    make_private_directories(folder)
    temporary = folder / f".{path.name}.partial-{os.getpid()}"
    try:
        with temporary.open("xb") as handle:
            frame.to_parquet(handle, index=False)
        os.chmod(temporary, 0o444)
        os.link(temporary, path)  # refuses an existing name: never an overwrite
    finally:
        temporary.unlink(missing_ok=True)
    return path
