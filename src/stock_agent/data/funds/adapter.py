"""
The replaceable fund-data adapter seam and a deterministic synthetic adapter.

An adapter does retrieval and parsing only. It returns the raw provider bytes
(stored as-is beside the tables, so they can be re-parsed later) and the
parsed rows without storage ids. Everything else (validation, the lockbox
cap, batch ids, writing) belongs to `ingest`, so swapping a provider changes
one adapter and nothing downstream. An adapter calls
`validation.validate_request` before retrieving, so a request reaching the
lockbox is refused before any provider is contacted.

There is no vendor adapter here. `SyntheticFundAdapter` is the only
implementation: three invented funds (`SYN_EQ`, `SYN_BOND`, `SYN_CASH`),
integer-cent closes, a 2-for-1 split and dated distributions on NYSE sessions.
It never touches the network and is not market data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.funds.schema import (
    DISTRIBUTIONS,
    PRICES,
    SPLITS,
    AdjustmentBasis,
    AmountBasis,
    DistributionType,
    RefusalReason,
    dumps_canonical,
    loads_strict,
    refuse,
    utc_timestamp,
)
from stock_agent.data.funds.validation import validate_request


@dataclass(frozen=True, eq=False)
class RawBatch:
    """
    One retrieval as the adapter saw it. The tables carry every schema column
    except `batch_id`, which `ingest` assigns. `retrieved_at` is the actual
    retrieval time in UTC, from an injected clock.
    """

    adapter_id: str
    adapter_version: str
    fund_ids: tuple[str, ...]
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    retrieved_at: pd.Timestamp
    raw_payload: bytes
    terms_note: str
    vintage_note: str
    prices: pd.DataFrame
    distributions: pd.DataFrame
    splits: pd.DataFrame


@runtime_checkable
class FundDataAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def fetch(
        self, fund_ids: Sequence[str], start: pd.Timestamp, end_exclusive: pd.Timestamp
    ) -> RawBatch:
        """Retrieve [start, end_exclusive) for `fund_ids`; validate the request first."""


# --- the synthetic adapter -----------------------------------------------------

SYNTHETIC_FUND_IDS = ("SYN_BOND", "SYN_CASH", "SYN_EQ")
SYNTHETIC_SOURCE_ID = "synthetic_fixture"
SYN_EQ_SPLIT_DATE = pd.Timestamp("2014-06-09", tz="UTC").as_unit("us")  # a Monday session
SYN_EQ_SPLIT_RATIO = 2.0  # new shares per old share


def _month_sessions() -> dict[tuple[int, int], list[pd.Timestamp]]:
    every = nyse.sessions(nyse.COVERAGE.first, nyse.COVERAGE.last)
    by_month = pd.Series(every, index=every).groupby([every.year, every.month])
    return {key: list(group) for key, group in by_month}


def _price_cents(fund_id: str, ordinal: int, session: pd.Timestamp) -> int:
    """Integer cents from the session's position in the whole calendar (prefix-stable)."""

    if fund_id == "SYN_EQ":
        cents = 5000 + 3 * ordinal + 37 * ((ordinal * 7) % 11)
        return cents // 2 if session >= SYN_EQ_SPLIT_DATE else cents
    if fund_id == "SYN_BOND":
        return 10000 + (ordinal * 37) % 200 - 100
    return 10000  # SYN_CASH: a stable-price cash fund


def _volume(fund_id: str, ordinal: int) -> int | None:
    if fund_id == "SYN_EQ":
        return 1_000_000 + 1_000 * (ordinal % 10)
    if fund_id == "SYN_BOND":
        return 200_000 + 500 * (ordinal % 7)
    return None  # SYN_CASH reports no volume


def _distribution_records(
    fund_id: str, months: dict[tuple[int, int], list[pd.Timestamp]]
) -> list[dict]:
    """Ex, record and pay dates are the n-th sessions of a month, so all stay in that month."""

    records = []
    for (_, month), days in sorted(months.items()):
        day = [session.strftime("%Y-%m-%d") for session in days]
        if fund_id == "SYN_EQ" and month in (3, 6, 9, 12):
            post_split = days[2] >= SYN_EQ_SPLIT_DATE
            dividend = [("income_dividend", "0.25" if not post_split else "0.125")]
            if month == 12:
                dividend.append(("capital_gain_long", "0.5" if not post_split else "0.25"))
            records += [(kind, amount, day[2], day[3], day[7]) for kind, amount in dividend]
        elif fund_id == "SYN_BOND":
            records.append(("income_dividend", "0.21", day[0], None, day[3]))
            if month == 12:
                records.append(("capital_gain_short", "0.05", day[0], None, day[3]))
        elif fund_id == "SYN_CASH":
            records.append(("income_dividend", "0.03", day[0], day[0], day[1]))
    return [
        {
            "fund_id": fund_id,
            "type": kind,
            "amount": amount,
            "ex_date": ex,
            "record_date": record,
            "pay_date": pay,
        }
        for kind, amount, ex, record, pay in records
    ]


def _dates(values: list[str | None]) -> pd.Series:
    text = pd.Series(values, dtype=object)
    parsed = pd.to_datetime(text, format="%Y-%m-%d", utc=True)
    return parsed.astype("datetime64[us, UTC]")


def parse_synthetic_payload(
    payload: bytes,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The transformation step: raw payload bytes to the schema's rows (no batch_id)."""

    data = loads_strict(payload.decode("utf-8"))
    source = data["source_id"]
    prices = pd.DataFrame(
        {
            "fund_id": pd.Series([row["fund_id"] for row in data["prices"]], dtype=object),
            "session_date": _dates([row["date"] for row in data["prices"]]),
            "close": pd.Series([int(row["close_cents"]) / 100 for row in data["prices"]]),
            "adjustment_basis": AdjustmentBasis.UNADJUSTED.value,
            "volume": pd.Series([row["volume"] for row in data["prices"]], dtype="Int64"),
            "source_id": source,
        },
        columns=[name for name in PRICES.names if name != "batch_id"],
    )
    rows = data["distributions"]
    distributions = pd.DataFrame(
        {
            "fund_id": pd.Series([row["fund_id"] for row in rows], dtype=object),
            "ex_date": _dates([row["ex_date"] for row in rows]),
            "pay_date": _dates([row["pay_date"] for row in rows]),
            "record_date": _dates([row["record_date"] for row in rows]),
            "amount": pd.Series([float(row["amount"]) for row in rows], dtype="float64"),
            "amount_basis": AmountBasis.PER_SHARE_AS_PAID.value,
            "distribution_type": pd.Series(
                [DistributionType(row["type"]).value for row in rows], dtype=object
            ),
            "source_id": source,
        },
        columns=[name for name in DISTRIBUTIONS.names if name != "batch_id"],
    )
    splits = pd.DataFrame(
        {
            "fund_id": pd.Series([row["fund_id"] for row in data["splits"]], dtype=object),
            "effective_date": _dates([row["effective_date"] for row in data["splits"]]),
            "ratio_new_per_old": pd.Series(
                [float(row["ratio"]) for row in data["splits"]], dtype="float64"
            ),
            "source_id": source,
        },
        columns=[name for name in SPLITS.names if name != "batch_id"],
    )
    return prices, distributions, splits


class SyntheticFundAdapter:
    """
    Deterministic synthetic retrieval. The clock is injected (`retrieved_at`);
    identical requests give identical payloads, and a value never depends on
    the request window (prefix stability).
    """

    adapter_id = "synthetic_funds"
    adapter_version = "1"
    terms_note = "Synthetic fixture generated in this repository; no vendor terms apply."
    vintage_note = (
        "Deterministic generator v1: unadjusted closes, distributions per share as paid, "
        "no revisions."
    )

    def __init__(self, *, retrieved_at: pd.Timestamp) -> None:
        self.retrieved_at = utc_timestamp("retrieved_at", retrieved_at)

    def _payload(self, fund_ids: tuple[str, ...], start: pd.Timestamp, end: pd.Timestamp) -> bytes:
        every = nyse.sessions(nyse.COVERAGE.first, nyse.COVERAGE.last)
        inside = (every >= start) & (every < end)
        months = _month_sessions()
        prices, distributions, splits = [], [], []
        for fund_id in fund_ids:
            for ordinal in inside.nonzero()[0].tolist():
                session = every[ordinal]
                prices.append(
                    {
                        "fund_id": fund_id,
                        "date": session.strftime("%Y-%m-%d"),
                        "close_cents": _price_cents(fund_id, ordinal, session),
                        "volume": _volume(fund_id, ordinal),
                    }
                )
            distributions += [
                row
                for row in _distribution_records(fund_id, months)
                if start <= pd.Timestamp(row["ex_date"], tz="UTC") < end
            ]
            if fund_id == "SYN_EQ" and start <= SYN_EQ_SPLIT_DATE < end:
                splits.append(
                    {
                        "fund_id": fund_id,
                        "effective_date": SYN_EQ_SPLIT_DATE.strftime("%Y-%m-%d"),
                        "ratio": str(SYN_EQ_SPLIT_RATIO),
                    }
                )
        document = {
            "source_id": SYNTHETIC_SOURCE_ID,
            "prices": prices,
            "distributions": distributions,
            "splits": splits,
        }
        return dumps_canonical(document).encode("utf-8")

    def fetch(
        self, fund_ids: Sequence[str], start: pd.Timestamp, end_exclusive: pd.Timestamp
    ) -> RawBatch:
        ids, first, end = validate_request(fund_ids, start, end_exclusive)
        unknown = sorted(set(ids) - set(SYNTHETIC_FUND_IDS))
        if unknown:
            raise refuse(RefusalReason.INVALID_REQUEST, f"unknown synthetic funds {unknown}")
        payload = self._payload(ids, first, end)
        prices, distributions, splits = parse_synthetic_payload(payload)
        return RawBatch(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            fund_ids=ids,
            start=first,
            end_exclusive=end,
            retrieved_at=self.retrieved_at,
            raw_payload=payload,
            terms_note=self.terms_note,
            vintage_note=self.vintage_note,
            prices=prices,
            distributions=distributions,
            splits=splits,
        )
