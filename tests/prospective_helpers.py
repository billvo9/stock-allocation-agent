"""
Invented prospective batches for the store, inference and boundary tests.
Every close and amount is a fixture value, never market data.
"""

from pathlib import Path

import pandas as pd

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.schema import DISTRIBUTIONS, PRICES, SPLITS, Instrument
from stock_agent.data.prospective.store import (
    ProspectiveBatch,
    ingest_prospective,
    prepare_store,
)
from stock_agent.data.prospective.window import H_END, WINDOW_START

FUNDS = ("VTI", "AGG", "SHV")
# After 18:00 America/New_York on H_END: H_END is complete.
AFTER_H_END = pd.Timestamp("2026-10-14T23:00:00.000000Z")
# A decision time after every fixture batch: `retrieved_by` for reading all of them.
LATER = pd.Timestamp("2026-12-31T00:00:00.000000Z")
DIVIDENDS = {"AGG": ["2026-08-03", "2026-09-01", "2026-10-01"], "VTI": ["2026-09-28"]}


def day(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC").as_unit("us")


def close_for(fund: str, session: pd.Timestamp, offset: float = 0.0) -> float:
    """An invented close that identifies its fund and session."""

    base = {"VTI": 300.0, "AGG": 100.0, "SHV": 110.0}.get(fund, 50.0)
    return base + (session.dayofyear % 17) / 4 + offset


def tables(
    funds=FUNDS,
    start=WINDOW_START,
    end=H_END,
    *,
    skip=(),
    offset: float = 0.0,
    dividends=None,
    splits=(),
):
    """F1 rows (no batch_id) for the sessions in [start, end], minus `skip` (fund, date) pairs."""

    dividends = DIVIDENDS if dividends is None else dividends
    sessions = nyse_rules.sessions(start, end)
    rows = [
        (fund, session)
        for fund in funds
        for session in sessions
        if (fund, f"{session:%Y-%m-%d}") not in skip
    ]
    prices = pd.DataFrame(
        {
            "fund_id": pd.Series([fund for fund, _ in rows], dtype=object),
            "session_date": pd.Series([s for _, s in rows], dtype="datetime64[us, UTC]"),
            "close": pd.Series([close_for(f, s, offset) for f, s in rows], dtype="float64"),
            "adjustment_basis": pd.Series(["unadjusted"] * len(rows), dtype=object),
            "volume": pd.Series([pd.NA] * len(rows), dtype="Int64"),
            "source_id": pd.Series(["fixture"] * len(rows), dtype=object),
        }
    )
    paid = [
        (fund, day(date))
        for fund in funds
        for date in dividends.get(fund, [])
        if start <= day(date) <= end
    ]
    none = pd.Series([pd.NaT] * len(paid), dtype="datetime64[us, UTC]")
    distributions = pd.DataFrame(
        {
            "fund_id": pd.Series([fund for fund, _ in paid], dtype=object),
            "ex_date": pd.Series([d for _, d in paid], dtype="datetime64[us, UTC]"),
            "declaration_date": none,
            "record_date": none,
            "pay_date": none,
            "amount": pd.Series([0.25 + offset / 100] * len(paid), dtype="float64"),
            "amount_basis": pd.Series(["per_share_as_paid"] * len(paid), dtype=object),
            "distribution_type": pd.Series(["unknown"] * len(paid), dtype=object),
            "source_sequence": pd.Series([0] * len(paid), dtype="int64"),
            "source_id": pd.Series(["fixture"] * len(paid), dtype=object),
        }
    )
    split_frame = pd.DataFrame(
        {
            "fund_id": pd.Series([fund for fund, _, _ in splits], dtype=object),
            "effective_date": pd.Series(
                [day(date) for _, date, _ in splits], dtype="datetime64[us, UTC]"
            ),
            "ratio_new_per_old": pd.Series([r for _, _, r in splits], dtype="float64"),
            "source_id": pd.Series(["fixture"] * len(splits), dtype=object),
        }
    )
    return (
        prices[_columns(PRICES)],
        distributions[_columns(DISTRIBUTIONS)],
        split_frame[_columns(SPLITS)],
    )


def _columns(spec) -> list[str]:
    return [name for name in spec.names if name != "batch_id"]


def batch(
    funds=FUNDS,
    start=WINDOW_START,
    end=H_END,
    retrieved_at=AFTER_H_END,
    *,
    skip=(),
    offset: float = 0.0,
    dividends=None,
    splits=(),
    payload: bytes = b"invented prospective fixture payload",
) -> ProspectiveBatch:
    prices, distributions, split_frame = tables(
        funds, start, end, skip=skip, offset=offset, dividends=dividends, splits=splits
    )
    return ProspectiveBatch(
        adapter_id="fixture_prospective",
        adapter_version="1",
        fund_ids=tuple(funds),
        start=start,
        end=end,
        instruments=tuple(Instrument(fund, fund, None) for fund in funds),
        retrieved_at=retrieved_at,
        raw_payload=payload,
        anchors_sha256="0" * 64,
        terms_note="invented fixture; no vendor terms",
        vintage_note="invented fixture",
        prices=prices,
        distributions=distributions,
        splits=split_frame,
    )


def new_store(path: Path) -> Path:
    assert prepare_store(path) is None
    return path


def stored(root: Path, *batches: ProspectiveBatch) -> Path:
    new_store(root)
    for item in batches:
        ingest_prospective(root, item)
    return root
