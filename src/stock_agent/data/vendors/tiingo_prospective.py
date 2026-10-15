"""
Tiingo prospective mode: real ETF closes, distributions and splits for the
prospective store, from the session 63 sessions before H_end to the last
complete session.

Owner decisions of 2026-10-14 (`docs/PROJECT_STATE.md`): live Tiingo requests
are authorized but run only by the owner in the owner's terminal
(`scripts/collect_prospective.py --authorized`), and storing anything still
needs the owner's OK at that step (`--store`). The first collection requests
only the 63-session lookback before H_end and later sessions.

Module constants: `H_END` (2026-10-14) and `LOOKBACK_SESSIONS` (63), from
`stock_agent.data.prospective.window`, which also defines `WINDOW_START`
(2026-07-16).

Request window (checked before the key is read or any request is made)
    [start, end], both inclusive session dates. `start` defaults to
    `WINDOW_START`; an earlier start is refused (`sealed_holdout`, or
    `before_window` before 2025-01-01). `end` defaults to the last complete
    session at the adapter clock's time (18:00 America/New_York, the OD9
    descriptive convention, assumed and not proof of vendor publication
    time); a later end is refused (`incomplete_session`).

The same as the research path (`tiingo.py`, unchanged)
    one prices call per fund for allowlisted columns only, each response
    checked before the next request, no retry; the field allowlist and value
    checks; the verified-split allowlist and raw-close check (on the rules
    calendar here); the distribution-basis gate with `MIN_VERIFIED_AMOUNTS`
    and `AMOUNT_TOLERANCE` unchanged, against prospective-scope anchors
    (`anchors.py`: ex-dates on or after WINDOW_START); and the tables rebuilt
    from the raw payload before anything can be stored.

The differences
    - every element dated before WINDOW_START refuses the batch
      (`sealed_holdout`); the research path's lockbox refusal is not used;
    - the envelope is version "p1" and names the store, H_end and the window
      start, so neither path's parser accepts the other's payload;
    - the result is a `ProspectiveBatch` for `store.ingest_prospective`, never
      an F1 `RawBatch`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.schema import (
    Instrument,
    RefusalReason,
    dumps_canonical,
    loads_strict,
    parse_date,
    refuse,
    utc_timestamp,
)
from stock_agent.data.prospective.store import ProspectiveBatch
from stock_agent.data.prospective.window import (
    H_END,
    LOOKBACK_SESSIONS,
    REQUEST_CUTOFF_RULE,
    WINDOW_START,
    ProspectiveRefusal,
    ProspectiveRefused,
    check_request,
    check_window,
    last_complete_session,
)
from stock_agent.data.vendors import tiingo
from stock_agent.data.vendors.anchors import AnchorScope, DistributionAnchors
from stock_agent.data.vendors.tiingo import (
    ALLOWED_FIELDS,
    MAX_TICKERS,
    SOURCE_ID,
    TERMS_NOTE,
    BasisCheck,
    CheckedSession,
    ParsedPayload,
    TiingoAdapter,
    TiingoDataRefused,
    TiingoRefusal,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

ADAPTER_ID = "tiingo_daily_prospective"
ADAPTER_VERSION = "1"
ENVELOPE_VERSION = "p1"
STORE = "prospective"
_ENVELOPE_FIELDS = (
    "envelope_version",
    "source_id",
    "store",
    "h_end",
    "window_start",
    "start_date",
    "end_date",
    "fields",
    "prices",
)
VINTAGE_NOTE = (
    f"{tiingo.VINTAGE_NOTE}; collected for the prospective store (H_end {H_END:%Y-%m-%d}, "
    f"window from {WINDOW_START:%Y-%m-%d}), inputs to frozen models only; {REQUEST_CUTOFF_RULE}"
)


@dataclass(frozen=True, eq=False)
class ProspectiveRetrieval:
    """
    One prospective retrieval, in memory only, checked for the field
    allowlist, the window and the verified splits, NOT for distribution
    basis. `verified_prospective_batch` is the only way to a storable batch.
    `start` and `end` are the requested sessions, both inclusive.
    """

    fund_ids: tuple[str, ...]
    start: pd.Timestamp
    end: pd.Timestamp
    retrieved_at: pd.Timestamp
    raw_payload: bytes = field(repr=False)
    prices: pd.DataFrame = field(repr=False)
    distributions: pd.DataFrame = field(repr=False)
    splits: pd.DataFrame = field(repr=False)


def _guard(label: str, session: pd.Timestamp) -> None:
    if session < MODEL_HOLDOUT_START:
        raise ProspectiveRefused(
            ProspectiveRefusal.BEFORE_WINDOW,
            f"{label}.date {session.date()} is before the holdout start; batch refused",
        )
    if session < WINDOW_START:
        raise ProspectiveRefused(
            ProspectiveRefusal.SEALED_HOLDOUT,
            f"{label}.date {session.date()} is in the sealed holdout before the prospective "
            f"window ({WINDOW_START.date()}); batch refused",
        )


def _previous(session: pd.Timestamp) -> pd.Timestamp:
    return nyse_rules.session_offset(session, -1)


def check_prospective_elements(
    ticker: str, data: object, first: pd.Timestamp, last: pd.Timestamp
) -> list[CheckedSession]:
    """`tiingo.check_elements` with the window guard and the rules calendar."""

    return tiingo.checked_sessions(
        ticker, data, first, last, guard=_guard, previous_session=_previous
    )


def check_tickers(ids: Sequence[str]) -> None:
    """At most `MAX_TICKERS` funds, each an upper-case Tiingo ticker (as the research path)."""

    if len(ids) > MAX_TICKERS:
        raise refuse(
            RefusalReason.INVALID_REQUEST, f"{len(ids)} funds; at most {MAX_TICKERS} a batch"
        )
    invalid = [fund_id for fund_id in ids if not tiingo._TICKER.fullmatch(fund_id)]
    if invalid:
        raise refuse(
            RefusalReason.INVALID_REQUEST, f"fund ids must be upper-case tickers: {invalid}"
        )


def request_window(
    now: pd.Timestamp, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The defaults: WINDOW_START and the last complete session at `now`."""

    return (
        WINDOW_START if start is None else start,
        last_complete_session(now) if end is None else end,
    )


def parse_prospective_payload(payload: bytes) -> ParsedPayload:
    """
    The stored envelope bytes to F1 rows, re-running every element check, the
    envelope's identity (version, store, H_end, window start) and the window.
    """

    envelope = loads_strict(tiingo._utf8("the payload", payload))
    if not isinstance(envelope, dict) or set(envelope) != set(_ENVELOPE_FIELDS):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "the payload is not a prospective envelope")
    if (
        envelope["envelope_version"] != ENVELOPE_VERSION
        or envelope["source_id"] != SOURCE_ID
        or envelope["store"] != STORE
        or envelope["h_end"] != f"{H_END:%Y-%m-%d}"
        or envelope["window_start"] != f"{WINDOW_START:%Y-%m-%d}"
        or envelope["fields"] != list(ALLOWED_FIELDS)
    ):
        raise refuse(RefusalReason.INVALID_FIELD, "the payload is not this store's envelope p1")
    first = parse_date("payload start_date", envelope["start_date"])
    last = parse_date("payload end_date", envelope["end_date"])
    check_window(first, last)
    prices = envelope["prices"]
    if not isinstance(prices, dict) or not prices:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "payload prices must be a non-empty object")
    checked = {}
    for ticker, elements in prices.items():
        if not tiingo._TICKER.fullmatch(ticker):
            raise refuse(RefusalReason.INVALID_FIELD, "a payload key is not a Tiingo ticker")
        checked[ticker] = check_prospective_elements(ticker, elements, first, last)
    return ParsedPayload(first, last, tuple(checked), *tiingo.frames(checked))


def collect_prospective(
    fund_ids: Sequence[str],
    *,
    adapter: TiingoAdapter,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> ProspectiveRetrieval:
    """
    Request [start, end] for every fund through `adapter` (its token
    provider, transport and clock) and check it in memory. The window is
    checked before the key is read or any request is made.
    """

    now = adapter.now()
    first, last = request_window(now, start, end)
    ids, first, last = check_request(fund_ids, first, last, now)
    check_tickers(ids)
    elements = adapter.request_elements(
        ids,
        first,
        last,
        lambda ticker, data: check_prospective_elements(ticker, data, first, last),
    )
    retrieved_at = utc_timestamp("retrieved_at", adapter.now())
    envelope = {
        "envelope_version": ENVELOPE_VERSION,
        "source_id": SOURCE_ID,
        "store": STORE,
        "h_end": f"{H_END:%Y-%m-%d}",
        "window_start": f"{WINDOW_START:%Y-%m-%d}",
        "start_date": f"{first:%Y-%m-%d}",
        "end_date": f"{last:%Y-%m-%d}",
        "fields": list(ALLOWED_FIELDS),
        "prices": elements,
    }
    payload = dumps_canonical(envelope).encode("utf-8")
    parsed = parse_prospective_payload(payload)
    return ProspectiveRetrieval(
        ids, first, last, retrieved_at, payload, parsed.prices, parsed.distributions, parsed.splits
    )


def _rebuilt(retrieval: ProspectiveRetrieval) -> ParsedPayload:
    parsed = parse_prospective_payload(retrieval.raw_payload)
    if (
        sorted(parsed.fund_ids) != sorted(retrieval.fund_ids)
        or parsed.start != retrieval.start
        or parsed.end_date != retrieval.end
    ):
        raise TiingoDataRefused(
            TiingoRefusal.PAYLOAD_MISMATCH,
            "the raw payload's tickers or dates differ from the retrieval's request",
        )
    for name in ("prices", "distributions", "splits"):
        if not getattr(parsed, name).equals(getattr(retrieval, name)):
            raise TiingoDataRefused(
                TiingoRefusal.PAYLOAD_MISMATCH,
                f"the in-memory {name} table differs from the raw payload",
            )
    return parsed


def verify_prospective_basis(
    retrieval: ProspectiveRetrieval, *, anchors: DistributionAnchors
) -> tuple[BasisCheck, ...]:
    """Rules (i) to (iv) per fund on the rebuilt tables, against prospective-scope anchors."""

    anchors = tiingo.require_scope(anchors, AnchorScope.PROSPECTIVE)
    return tiingo.basis_checks(retrieval.fund_ids, _rebuilt(retrieval), anchors)


def verified_prospective_batch(
    retrieval: ProspectiveRetrieval, *, anchors: DistributionAnchors
) -> ProspectiveBatch:
    """
    The storable batch, from the tables rebuilt from the raw payload, only
    when every fund's distribution basis passes; the vintage note records the
    anchors file's sha256 and the request rule.
    """

    anchors = tiingo.require_scope(anchors, AnchorScope.PROSPECTIVE)
    parsed = _rebuilt(retrieval)
    tiingo.refuse_failing(tiingo.basis_checks(retrieval.fund_ids, parsed, anchors))
    funds = ", ".join(retrieval.fund_ids)
    return ProspectiveBatch(
        adapter_id=ADAPTER_ID,
        adapter_version=ADAPTER_VERSION,
        fund_ids=retrieval.fund_ids,
        start=retrieval.start,
        end=retrieval.end,
        instruments=tuple(Instrument(ticker, ticker, None) for ticker in retrieval.fund_ids),
        retrieved_at=retrieval.retrieved_at,
        raw_payload=retrieval.raw_payload,
        anchors_sha256=anchors.sha256,
        terms_note=TERMS_NOTE,
        vintage_note=(
            f"{VINTAGE_NOTE}; {anchors.provenance_note()}; distribution basis spot-checked "
            f"against primary-source amounts (a sample, not a validation of every "
            f"distribution): {funds}"
        ),
        prices=parsed.prices,
        distributions=parsed.distributions,
        splits=parsed.splits,
    )


__all__ = [
    "ADAPTER_ID",
    "ENVELOPE_VERSION",
    "H_END",
    "LOOKBACK_SESSIONS",
    "WINDOW_START",
    "ProspectiveRetrieval",
    "check_prospective_elements",
    "check_tickers",
    "collect_prospective",
    "parse_prospective_payload",
    "request_window",
    "verified_prospective_batch",
    "verify_prospective_basis",
]
