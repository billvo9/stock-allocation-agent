"""
Integration tests: cross-filing fiscal-flow reconciliation in EDGAR history.

These tests run the real pipeline (build_edgar_quarter, observation
extraction, prior selection, reconcile_fiscal_flow). Only the SEC I/O
boundary is faked: the filing lister, the filing loader, and the clock.

Fixture: Micron (MU), 52/53-week fiscal year ending the Thursday nearest
Aug 31. Period ends are real; revenue values (USD millions) and
acceptance times are illustrative.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pytest

import stock_agent.data.fundamentals.edgar_history as history_module
from stock_agent.data.fundamentals.edgar_history import (
    RECONCILIATION_COLUMNS,
    EdgarHistoryResult,
    build_edgar_history,
)
from stock_agent.data.fundamentals.edgar_quarterly import FLOW_METRIC_STATEMENTS
from stock_agent.data.fundamentals.point_in_time import align_quarterly_fundamentals_asof
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
    validate_quarterly_fundamental_frame,
)

REVENUE_CONCEPT = "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"
OCF_CONCEPT = "us-gaap_NetCashProvidedByUsedInOperatingActivities"

FLOW_METRICS = sorted(FLOW_METRIC_STATEMENTS)

CLOCK = pd.Timestamp("2027-01-31", tz="UTC")


# ---------------------------------------------------------------------------
# Fakes for the SEC I/O boundary
# ---------------------------------------------------------------------------


class _FakeStatement:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()


class _FakeStatements:
    def __init__(
        self,
        *,
        income: pd.DataFrame | None,
        cash_flow: pd.DataFrame | None,
    ) -> None:
        self._income = income
        self._cash_flow = cash_flow

    def income_statement(self):
        return None if self._income is None else _FakeStatement(self._income)

    def balance_sheet(self):
        return None

    def cash_flow_statement(self):
        return None if self._cash_flow is None else _FakeStatement(self._cash_flow)


class _FakeXbrl:
    def __init__(
        self,
        statements: _FakeStatements,
        reporting_periods: list[dict[str, Any]],
    ) -> None:
        self.statements = statements
        self.reporting_periods = reporting_periods


class _FakeFiling:
    def __init__(self, xbrl: _FakeXbrl) -> None:
        self._xbrl = xbrl

    def xbrl(self):
        return self._xbrl


@dataclass(frozen=True)
class MuFiling:
    accession: str
    # Expected knowledge time, computed here from the fixture itself
    # (acceptance time, else filing date + 1 day) independently of the
    # pipeline, so tests can check the pipeline's available_at against it.
    available_at: pd.Timestamp
    metadata: dict[str, object]
    filing: _FakeFiling


def _statement_row(
    concept: str,
    period_end: str,
    values: dict[str, float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "concept": concept,
                "dimension": False,
                **{f"{period_end} ({label})": value for label, value in values.items()},
            }
        ]
    )


def _mu_filing(
    *,
    accession: str,
    form: str,
    period_end: str,
    accepted_at: str | None,
    fiscal_year: int,
    fiscal_period: str,
    revenue: dict[str, float],
    operating_cash_flow: dict[str, float] | None = None,
    filing_date: str | None = None,
    without_statements: bool = False,
) -> MuFiling:
    """
    revenue / operating_cash_flow map a period label (Q1..Q4, YTD, FY)
    to the value reported for this filing's own period end.

    accepted_at=None models a filing without an SEC acceptance time;
    filing_date is then required.
    """

    if accepted_at is None:
        assert filing_date is not None
        accepted = pd.NaT
        filed = pd.Timestamp(filing_date, tz="UTC")
        expected_available_at = filed + pd.Timedelta(days=1)
    else:
        accepted = pd.Timestamp(accepted_at, tz="UTC")
        filed = accepted.normalize() if filing_date is None else pd.Timestamp(filing_date, tz="UTC")
        expected_available_at = accepted

    income = None if without_statements else _statement_row(REVENUE_CONCEPT, period_end, revenue)

    cash_flow = (
        None
        if operating_cash_flow is None or without_statements
        else _statement_row(OCF_CONCEPT, period_end, operating_cash_flow)
    )

    xbrl = _FakeXbrl(
        _FakeStatements(income=income, cash_flow=cash_flow),
        reporting_periods=[
            {
                "type": "duration",
                "end_date": period_end,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
            }
        ],
    )

    metadata = {
        "form": form,
        "period_end": pd.Timestamp(period_end, tz="UTC"),
        "filing_date": filed,
        "accepted_at": accepted,
        "accession_number": accession,
        "is_xbrl": True,
    }

    return MuFiling(
        accession=accession,
        available_at=expected_available_at,
        metadata=metadata,
        filing=_FakeFiling(xbrl),
    )


def _run(
    filings: list[MuFiling],
    *,
    clock: pd.Timestamp = CLOCK,
    strict: bool = True,
    listing_rows: list[dict[str, object]] | None = None,
) -> EdgarHistoryResult:
    """
    listing_rows, when given, replaces the listing (e.g. to inject
    duplicate or malformed accession numbers); filings still supply the
    loadable filings.
    """

    rows = listing_rows if listing_rows is not None else [f.metadata for f in filings]
    listing = pd.DataFrame(rows)
    by_accession = {filing.accession: filing.filing for filing in filings}

    return build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: listing,
        filing_loader=lambda accession: by_accession[accession],
        clock=lambda: clock,
        strict=strict,
    )


# ---------------------------------------------------------------------------
# MU FY2025 / FY2026 filings
# ---------------------------------------------------------------------------


def q1_fy25(**revenue: float) -> MuFiling:
    # Fiscal Q1 ends in calendar NOVEMBER (calendar Q4).
    return _mu_filing(
        accession="Q1",
        form="10-Q",
        period_end="2024-11-28",
        accepted_at="2024-12-19 21:00",
        fiscal_year=2025,
        fiscal_period="Q1",
        revenue=revenue or {"YTD": 8700.0},
        operating_cash_flow={"YTD": 3000.0},
    )


def q2_fy25() -> MuFiling:
    return _mu_filing(
        accession="Q2",
        form="10-Q",
        period_end="2025-02-27",
        accepted_at="2025-03-26 20:00",
        fiscal_year=2025,
        fiscal_period="Q2",
        revenue={"YTD": 16000.0},
        operating_cash_flow={"YTD": 6000.0},
    )


def q2a_fy25_early() -> MuFiling:
    return _mu_filing(
        accession="Q2A-EARLY",
        form="10-Q/A",
        period_end="2025-02-27",
        accepted_at="2025-05-15 20:00",
        fiscal_year=2025,
        fiscal_period="Q2",
        revenue={"YTD": 15800.0},
    )


def q3_fy25() -> MuFiling:
    return _mu_filing(
        accession="Q3",
        form="10-Q",
        period_end="2025-05-29",
        accepted_at="2025-06-27 20:00",
        fiscal_year=2025,
        fiscal_period="Q3",
        revenue={"YTD": 25000.0},
        operating_cash_flow={"YTD": 9500.0},
    )


def q2a_fy25_late() -> MuFiling:
    return _mu_filing(
        accession="Q2A-LATE",
        form="10-Q/A",
        period_end="2025-02-27",
        accepted_at="2025-07-15 20:00",
        fiscal_year=2025,
        fiscal_period="Q2",
        revenue={"YTD": 15800.0},
    )


def q3a_fy25() -> MuFiling:
    return _mu_filing(
        accession="Q3A",
        form="10-Q/A",
        period_end="2025-05-29",
        accepted_at="2025-08-01 20:00",
        fiscal_year=2025,
        fiscal_period="Q3",
        revenue={"YTD": 25500.0},
    )


def k_fy25() -> MuFiling:
    # Fiscal year ends in calendar AUGUST (calendar Q3).
    return _mu_filing(
        accession="K",
        form="10-K",
        period_end="2025-08-28",
        accepted_at="2025-10-03 20:00",
        fiscal_year=2025,
        fiscal_period="FY",
        revenue={"FY": 37000.0},
        operating_cash_flow={"FY": 13000.0},
    )


def q1_fy26() -> MuFiling:
    return _mu_filing(
        accession="Q1-FY26",
        form="10-Q",
        period_end="2025-11-27",
        accepted_at="2025-12-17 21:00",
        fiscal_year=2026,
        fiscal_period="Q1",
        revenue={"YTD": 8900.0},
    )


def q2_fy26() -> MuFiling:
    return _mu_filing(
        accession="Q2-FY26",
        form="10-Q",
        period_end="2026-02-26",
        accepted_at="2026-03-18 21:00",
        fiscal_year=2026,
        fiscal_period="Q2",
        revenue={"YTD": 17000.0},
    )


def k_fy25_without_acceptance() -> MuFiling:
    # No SEC acceptance time: available_at falls back to filing date + 1 day.
    return _mu_filing(
        accession="K",
        form="10-K",
        period_end="2025-08-28",
        accepted_at=None,
        filing_date="2025-10-03",
        fiscal_year=2025,
        fiscal_period="FY",
        revenue={"FY": 37000.0},
        operating_cash_flow={"FY": 13000.0},
    )


def broken_filing() -> MuFiling:
    # Loads, but carries no usable statements -> quarter_build_error.
    return _mu_filing(
        accession="BROKEN",
        form="10-Q",
        period_end="2025-05-29",
        accepted_at="2025-09-01 20:00",
        fiscal_year=2025,
        fiscal_period="Q3",
        revenue={},
        without_statements=True,
    )


def _full_timeline() -> list[MuFiling]:
    return [
        q1_fy25(),
        q2_fy25(),
        q3_fy25(),
        q2a_fy25_late(),
        q3a_fy25(),
        k_fy25(),
    ]


def _at_version(
    frame: pd.DataFrame,
    accession: str,
    at: pd.Timestamp | None,
) -> pd.DataFrame:
    """
    Rows of the version anchored by accession at knowledge time at.

    Default (at=None): the filing's OWN version, i.e. the version emitted
    at the filing's own available_at. A filing is the anchor of versions
    only from its own available_at onward, so this is its earliest one.
    Later re-versions with the same anchor are selected with at.
    """

    rows = frame[frame["sec_accession_number"].eq(accession)]
    available_at = pd.to_datetime(rows["available_at"], utc=True)
    target = available_at.min() if at is None else pd.Timestamp(at)
    return rows[available_at.eq(target)]


def _row(
    result: EdgarHistoryResult,
    accession: str,
    at: pd.Timestamp | None = None,
) -> pd.Series:
    rows = _at_version(result.frame, accession, at)
    assert len(rows) == 1
    return rows.iloc[0]


def _lineage(
    result: EdgarHistoryResult,
    accession: str,
    metric_name: str = "revenue",
    at: pd.Timestamp | None = None,
) -> pd.Series:
    lineage = result.reconciliation
    rows = _at_version(lineage[lineage["metric_name"].eq(metric_name)], accession, at)
    assert len(rows) == 1
    return rows.iloc[0]


def _versions(
    result: EdgarHistoryResult,
    period_end: str,
    columns: tuple[str, ...] = ("sec_accession_number", "available_at", "revenue"),
) -> list[tuple[object, ...]]:
    """Every version of one fiscal period, in knowledge-time order."""

    frame = result.frame
    rows = frame[
        pd.to_datetime(frame["period_end"], utc=True).eq(pd.Timestamp(period_end, tz="UTC"))
    ].sort_values("available_at", kind="mergesort")

    return [
        tuple(
            pd.Timestamp(row[column]) if column == "available_at" else row[column]
            for column in columns
        )
        for _, row in rows.iterrows()
    ]


# ---------------------------------------------------------------------------
# R1-R7: required reconciliation scenarios
# ---------------------------------------------------------------------------


def test_r1_q3_is_q3_ytd_minus_prior_q2_ytd():
    result = _run([q2_fy25(), q3_fy25()])

    assert _row(result, "Q3")["revenue"] == pytest.approx(9000.0)

    lineage = _lineage(result, "Q3")
    assert lineage["method"] == "ytd_minus_prior_ytd"
    assert lineage["reason"] == "ok"
    assert lineage["sec_accession_number"] == "Q3"
    assert lineage["prior_accession_number"] == "Q2"
    assert lineage["prior_value_source"] == "ytd"


def test_r1_amendment_available_before_current_is_the_prior():
    result = _run([q2_fy25(), q2a_fy25_early(), q3_fy25()])

    assert _row(result, "Q3")["revenue"] == pytest.approx(9200.0)

    lineage = _lineage(result, "Q3")
    assert lineage["prior_accession_number"] == "Q2A-EARLY"
    assert lineage["equivalent_prior_accessions"] == ("Q2A-EARLY",)


def test_r2_q4_is_fy_minus_prior_q3_ytd():
    result = _run([q3_fy25(), k_fy25()])

    assert _row(result, "K")["revenue"] == pytest.approx(12000.0)

    lineage = _lineage(result, "K")
    assert lineage["method"] == "fy_minus_q3_ytd"
    assert lineage["fiscal_quarter"] == 4
    assert lineage["prior_accession_number"] == "Q3"


def test_r3_amendment_accepted_after_current_is_not_used():
    result = _run([q2_fy25(), q3_fy25(), q2a_fy25_late()])

    # The Q3 version knowable at Q3 time never uses the later amendment.
    assert _row(result, "Q3")["revenue"] == pytest.approx(9000.0)
    assert _lineage(result, "Q3")["prior_accession_number"] == "Q2"
    assert pd.Timestamp(_row(result, "Q3")["available_at"]) == q3_fy25().available_at

    flagged = result.diagnostics[
        result.diagnostics["reason"].eq("derived_quarter_input_superseded")
    ]
    assert len(flagged) == 1

    # The diagnostic belongs to the filing that creates the knowledge
    # (Q2A-LATE), with that filing's timestamps; it must not be knowable
    # at the earlier Q3 acceptance time.
    diagnostic = flagged.iloc[0]
    assert diagnostic["status"] == "info"
    assert diagnostic["accession_number"] == "Q2A-LATE"
    assert pd.Timestamp(diagnostic["accepted_at"]) == q2a_fy25_late().available_at
    assert "metric=revenue" in diagnostic["detail"]
    assert "derived_accession=Q3" in diagnostic["detail"]
    assert "used_prior=Q2" in diagnostic["detail"]


def test_r4_fiscal_year_boundary_is_never_crossed():
    # FY2026 Q1 stands alone even though the prior fiscal year's 10-K
    # (the "preceding quarter" in calendar terms) is available.
    result = _run([k_fy25(), q1_fy26()])

    new_year_q1 = _lineage(result, "Q1-FY26")
    assert new_year_q1["method"] == "q1_ytd"
    assert pd.isna(new_year_q1["prior_accession_number"])
    assert _row(result, "Q1-FY26")["revenue"] == pytest.approx(8900.0)

    # The only fiscal-Q1 candidate for FY2026 Q2 belongs to FY2025.
    result = _run([q1_fy25(), q2_fy26()])

    new_year_q2 = _lineage(result, "Q2-FY26")
    assert new_year_q2["method"] == "unavailable"
    assert new_year_q2["reason"] == "missing_prior_fiscal_observation"
    assert pd.isna(_row(result, "Q2-FY26")["revenue"])


def test_r5_quarters_come_from_fiscal_metadata_not_calendar_month():
    result = _run([q1_fy25(), q2_fy25(), q3_fy25(), k_fy25()])

    identity = (
        result.reconciliation[result.reconciliation["metric_name"].eq("revenue")]
        .set_index("sec_accession_number")[["fiscal_year", "fiscal_quarter"]]
        .to_dict("index")
    )

    # November -> fiscal Q1, February -> Q2, May -> Q3, August -> Q4,
    # all inside fiscal 2025 although they span calendar 2024 and 2025.
    assert identity == {
        "Q1": {"fiscal_year": 2025, "fiscal_quarter": 1},
        "Q2": {"fiscal_year": 2025, "fiscal_quarter": 2},
        "Q3": {"fiscal_year": 2025, "fiscal_quarter": 3},
        "K": {"fiscal_year": 2025, "fiscal_quarter": 4},
    }

    assert result.frame.set_index("sec_accession_number")["revenue"].to_dict() == {
        "Q1": pytest.approx(8700.0),
        "Q2": pytest.approx(7300.0),
        "Q3": pytest.approx(9000.0),
        "K": pytest.approx(12000.0),
    }


@pytest.mark.parametrize(
    "filing",
    [q2_fy25, q3_fy25, k_fy25],
    ids=["q2", "q3", "q4"],
)
def test_r6_missing_prior_is_unavailable_not_guessed(filing):
    only = filing()
    result = _run([only])

    assert pd.isna(_row(result, only.accession)["revenue"])

    lineage = _lineage(result, only.accession)
    assert lineage["method"] == "unavailable"
    assert lineage["reason"] == "missing_prior_fiscal_observation"
    assert pd.isna(lineage["value"])


def test_r6_metric_not_reported_is_unavailable_with_reason():
    result = _run([q3_fy25()])

    lineage = _lineage(result, "Q3", "net_income")
    assert lineage["method"] == "unavailable"
    assert lineage["reason"] == "no_reported_value"


def test_r7_q1_direct_only_filing_feeds_q2_derivation():
    result = _run([q1_fy25(Q1=8700.0), q2_fy25()])

    q1_lineage = _lineage(result, "Q1")
    assert q1_lineage["method"] == "direct_quarter"
    assert _row(result, "Q1")["revenue"] == pytest.approx(8700.0)

    q2_lineage = _lineage(result, "Q2")
    assert q2_lineage["method"] == "ytd_minus_prior_ytd"
    assert q2_lineage["prior_value_source"] == "q1_direct"
    assert _row(result, "Q2")["revenue"] == pytest.approx(7300.0)


# ---------------------------------------------------------------------------
# P1-P4: point-in-time safety
# ---------------------------------------------------------------------------

# retrieved_at records when OUR process fetched the data (the injected
# clock), not when the market could know it. The prefix run below is built
# at a different clock than the full run, exactly as a real earlier run
# would have been, so retrieved_at legitimately differs. It is excluded
# because it is not knowledge-time information; every other column,
# including available_at and every accounting value, must match exactly.
_CLOCK_DEPENDENT_COLUMNS = ["retrieved_at"]


def _known_by(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    known = frame[pd.to_datetime(frame["available_at"], utc=True) <= cutoff]

    return (
        known.drop(columns=_CLOCK_DEPENDENT_COLUMNS, errors="ignore")
        .sort_values(["available_at", "sec_accession_number"], kind="mergesort")
        .reset_index(drop=True)
    )


def _lineage_known_by(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    known = frame[pd.to_datetime(frame["available_at"], utc=True) <= cutoff]

    return known.sort_values(
        ["available_at", "sec_accession_number", "metric_name"],
        kind="mergesort",
    ).reset_index(drop=True)


def _diagnostics_known_by(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    knowable_at: dict[str, pd.Timestamp],
) -> pd.DataFrame:
    # A diagnostic is knowable once the filing it is attached to is.
    known = frame[frame["accession_number"].map(knowable_at) <= cutoff]

    return known.sort_values(
        ["accession_number", "status", "reason", "detail"],
        kind="mergesort",
    ).reset_index(drop=True)


def test_p1_history_is_prefix_stable_for_every_knowledge_time():
    """
    Prefix stability:

        history(all filings) restricted to knowledge time <= t
        ==
        history(only filings knowable by t)

    checked at every knowledge time t in the timeline, for the canonical
    frame, the reconciliation lineage, and the diagnostics.

    The timeline covers: a late Q2 amendment (after Q3), a Q3 amendment
    (before the 10-K), a 10-K without an SEC acceptance time (filing
    date + 1 day fallback), and a filing that fails to build (non-strict).

    "Knowable by t" is computed from the fixture (MuFiling.available_at),
    not from pipeline output. The pipeline's own available_at is pinned to
    the same fixture value in test_lineage_and_canonical_available_at_match
    _fixture_knowledge_time, so a wrong availability rule cannot cancel out
    on both sides of this comparison.
    """

    timeline = [*_full_timeline()[:-1], broken_filing(), k_fy25_without_acceptance()]
    knowable_at = {filing.accession: filing.available_at for filing in timeline}

    full = _run(timeline, strict=False)

    for cutoff in sorted(set(knowable_at.values())):
        known_filings = [filing for filing in timeline if filing.available_at <= cutoff]

        # Built "at the time": one day after the cutoff, not at CLOCK.
        prefix = _run(known_filings, clock=cutoff + pd.Timedelta(days=1), strict=False)

        pd.testing.assert_frame_equal(
            _known_by(full.frame, cutoff),
            _known_by(prefix.frame, cutoff),
            obj=f"canonical history known by {cutoff}",
        )

        pd.testing.assert_frame_equal(
            _lineage_known_by(full.reconciliation, cutoff),
            _lineage_known_by(prefix.reconciliation, cutoff),
            obj=f"reconciliation lineage known by {cutoff}",
        )

        pd.testing.assert_frame_equal(
            _diagnostics_known_by(full.diagnostics, cutoff, knowable_at),
            _diagnostics_known_by(prefix.diagnostics, cutoff, knowable_at),
            obj=f"diagnostics known by {cutoff}",
        )

    q3_time = q3_fy25().available_at
    q2a_late_time = q2a_fy25_late().available_at
    k_time = k_fy25_without_acceptance().available_at

    # Exact versions: (anchor accession, available_at, revenue, OCF).
    # Q3 keeps 9000 at its own time (25000 - original Q2 16000); the late
    # Q2 amendment adds a NEW Q3 version at the amendment's time (9200),
    # it never rewrites the earlier one. Q3A's version carries Q3's OCF
    # forward (the amendment omits OCF). BROKEN never appears.
    frame = full.frame
    assert [
        (
            row["sec_accession_number"],
            pd.Timestamp(row["available_at"]),
            row["revenue"],
            row["operating_cash_flow"],
        )
        for _, row in frame.iterrows()
    ] == [
        ("Q1", q1_fy25().available_at, 8700.0, 3000.0),
        ("Q2", q2_fy25().available_at, 7300.0, 3000.0),
        ("Q3", q3_time, 9000.0, 3500.0),
        ("Q2A-LATE", q2a_late_time, 7100.0, 3000.0),
        ("Q3", q2a_late_time, 9200.0, 3500.0),
        ("Q3A", q3a_fy25().available_at, 9700.0, 3500.0),
        ("K", k_time, 11500.0, 3500.0),
    ]

    assert _lineage(full, "Q3")["prior_accession_number"] == "Q2"
    assert _lineage(full, "Q3", at=q2a_late_time)["prior_accession_number"] == "Q2A-LATE"

    # Nothing known at Q3 time mentions the later amendment.
    assert "Q2A-LATE" not in set(_known_by(full.frame, q3_time)["sec_accession_number"])
    assert (
        _diagnostics_known_by(full.diagnostics, q3_time, knowable_at)["reason"]
        .ne("derived_quarter_input_superseded")
        .all()
    )
    assert "BROKEN" not in set(full.frame["sec_accession_number"])


def test_p2_current_amendment_creates_new_later_version():
    without = _run([q2_fy25(), q3_fy25(), q2a_fy25_late()])
    with_amendment = _run([q2_fy25(), q3_fy25(), q2a_fy25_late(), q3a_fy25()])

    # Both Q3-anchored versions (own time and the Q2A-LATE re-version)
    # are unchanged by the later Q3 amendment.
    for at in (q3_fy25().available_at, q2a_fy25_late().available_at):
        pd.testing.assert_series_equal(
            _row(without, "Q3", at=at).drop(labels=_CLOCK_DEPENDENT_COLUMNS),
            _row(with_amendment, "Q3", at=at).drop(labels=_CLOCK_DEPENDENT_COLUMNS),
        )

    original_after = _row(with_amendment, "Q3")

    amended = _row(with_amendment, "Q3A")
    assert pd.Timestamp(amended["available_at"]) > pd.Timestamp(original_after["available_at"])

    # Q3A runs its own selection at its own available_at: by then the
    # late Q2 amendment is public, so Q3A = 25500 - 15800.
    assert amended["revenue"] == pytest.approx(9700.0)
    assert _lineage(with_amendment, "Q3A")["prior_accession_number"] == "Q2A-LATE"


def test_p3_late_prior_amendment_rederives_current_quarter_as_new_version():
    result = _run([q2_fy25(), q3_fy25(), q2a_fy25_late()])

    q3_time = q3_fy25().available_at
    q2a_late_time = q2a_fy25_late().available_at

    # As-of complete: once Q2A-LATE is public, a NEW Q3 version (anchor
    # Q3, available at the amendment's time) carries 25000 - 15800. The
    # version knowable at Q3 time is untouched.
    assert _versions(result, "2025-05-29") == [
        ("Q3", q3_time, 9000.0),
        ("Q3", q2a_late_time, 9200.0),
    ]

    reversion = _row(result, "Q3", at=q2a_late_time)
    assert reversion["availability_source"] == "sec_acceptance_datetime"
    assert reversion["sec_form_type"] == "10-Q"

    lineage = _lineage(result, "Q3", at=q2a_late_time)
    assert lineage["prior_accession_number"] == "Q2A-LATE"
    assert lineage["prior_available_at"] == q2a_late_time
    assert lineage["source_accession_number"] == "Q3"
    assert lineage["source_available_at"] == q3_time
    assert lineage["version_trigger_accessions"] == ("Q2A-LATE",)


def test_p4_no_duplicate_symbol_period_end_available_at():
    result = _run([*_full_timeline(), q2a_fy25_early(), q1_fy26(), q2_fy26()])

    assert not result.frame.duplicated(subset=["symbol", "period_end", "available_at"]).any()

    # Versions of different periods may share an instant (Q2A-LATE and
    # the Q3 re-version it triggers), so (symbol, available_at) is not a key.
    reversioned = _run(_full_timeline()).frame
    assert reversioned.duplicated(subset=["symbol", "available_at"]).any()
    assert not reversioned.duplicated(subset=["symbol", "period_end", "available_at"]).any()


# ---------------------------------------------------------------------------
# L1-L5: lineage
# ---------------------------------------------------------------------------


def _lineage_fixture() -> EdgarHistoryResult:
    return _run(
        [
            q1_fy25(),
            _mu_filing(
                accession="Q2-OCF",
                form="10-Q",
                period_end="2025-02-27",
                accepted_at="2025-03-26 20:00",
                fiscal_year=2025,
                fiscal_period="Q2",
                revenue={"YTD": 16000.0, "Q2": 7300.0},
                operating_cash_flow={"YTD": 6000.0},
            ),
            q3_fy25(),
            q2a_fy25_late(),
            k_fy25(),
        ]
    )


def test_l1_lineage_key_is_symbol_accession_available_at_metric():
    lineage = _lineage_fixture().reconciliation

    key = ["symbol", "sec_accession_number", "available_at", "metric_name"]
    assert not lineage.duplicated(subset=key).any()

    # The anchor alone is not a key: Q3 anchors two versions.
    assert lineage.duplicated(subset=["symbol", "sec_accession_number", "metric_name"]).any()


def test_l2_every_canonical_flow_value_has_matching_lineage():
    result = _lineage_fixture()

    lineage = result.reconciliation.set_index(
        ["symbol", "sec_accession_number", "available_at", "metric_name"]
    )

    assert len(lineage) == len(result.frame) * len(FLOW_METRICS)

    for _, row in result.frame.iterrows():
        for metric_name in FLOW_METRICS:
            entry = lineage.loc[
                (row["symbol"], row["sec_accession_number"], row["available_at"], metric_name)
            ]
            canonical_value = row[metric_name]

            if pd.isna(canonical_value):
                assert entry["method"] == "unavailable"
                assert isinstance(entry["reason"], str) and entry["reason"] != "ok"
                assert pd.isna(entry["value"])
            else:
                assert entry["method"] != "unavailable"
                assert entry["reason"] == "ok"
                assert entry["value"] == pytest.approx(canonical_value)


def test_l3_lineage_priors_are_earlier_and_from_other_filings():
    lineage = _lineage_fixture().reconciliation

    with_prior = lineage[lineage["prior_accession_number"].notna()]
    assert not with_prior.empty

    assert (
        pd.to_datetime(with_prior["prior_available_at"], utc=True)
        <= pd.to_datetime(with_prior["available_at"], utc=True)
    ).all()

    assert (with_prior["prior_accession_number"] != with_prior["sec_accession_number"]).all()


def test_l4_empty_filings_give_empty_lineage_with_columns():
    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: pd.DataFrame(),
    )

    assert result.reconciliation.empty
    assert result.reconciliation.columns.tolist() == RECONCILIATION_COLUMNS


def test_l5_canonical_schema_is_unchanged():
    result = _lineage_fixture()

    assert result.frame.columns.tolist() == QUARTERLY_FUNDAMENTAL_COLUMNS
    validate_quarterly_fundamental_frame(result.frame)


# ---------------------------------------------------------------------------
# A1-A2: agreement and order independence
# ---------------------------------------------------------------------------


def test_a1_reconciliation_agrees_with_build_edgar_quarter(monkeypatch):
    """
    build_edgar_quarter still emits direct and Q1-from-YTD values on its
    own. Wherever it does, history reconciliation must give the same
    number. The real builder runs; the spy only records its output.
    """

    real_builder = history_module.build_edgar_quarter
    emitted: dict[str, pd.Series] = {}

    def spy(**kwargs: Any) -> pd.DataFrame:
        frame = real_builder(**kwargs)
        emitted[kwargs["sec_accession_number"]] = frame.iloc[0]
        return frame

    monkeypatch.setattr(history_module, "build_edgar_quarter", spy)

    result = _lineage_fixture()

    compared = 0

    for accession, builder_row in emitted.items():
        history_row = _row(result, accession)

        for metric_name in FLOW_METRICS:
            if pd.isna(builder_row[metric_name]):
                continue

            assert history_row[metric_name] == pytest.approx(builder_row[metric_name])
            compared += 1

    # Q1 revenue (Q1 YTD) and Q2-OCF revenue (direct Q2).
    assert compared >= 2


def test_a2_output_is_independent_of_filing_order():
    timeline = [*_full_timeline(), q2a_fy25_early()]
    baseline = _run(timeline)

    rng = random.Random(20251004)

    for _ in range(10):
        shuffled = timeline[:]
        rng.shuffle(shuffled)

        result = _run(shuffled)

        pd.testing.assert_frame_equal(result.frame, baseline.frame)
        pd.testing.assert_frame_equal(result.reconciliation, baseline.reconciliation)

        # Per-filing diagnostics are emitted in listing order, which is
        # presentation, not content; compare them as a sorted set.
        pd.testing.assert_frame_equal(
            _sorted_diagnostics(result.diagnostics),
            _sorted_diagnostics(baseline.diagnostics),
        )


def _sorted_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(
        ["accession_number", "status", "reason", "detail"],
        kind="mergesort",
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Review fixes: reason codes at history level (item 4)
# ---------------------------------------------------------------------------


def _filing_variant(
    base: MuFiling,
    *,
    accession: str,
    accepted_at: str | None = None,
    revenue: dict[str, float] | None = None,
    form: str | None = None,
) -> MuFiling:
    """A copy of base with a new accession and optionally new timing/values."""

    metadata = base.metadata
    period_end = str(pd.Timestamp(metadata["period_end"]).date())
    fiscal_period = base.filing.xbrl().reporting_periods[0]["fiscal_period"]
    fiscal_year = base.filing.xbrl().reporting_periods[0]["fiscal_year"]

    return _mu_filing(
        accession=accession,
        form=form or str(metadata["form"]),
        period_end=period_end,
        accepted_at=accepted_at or str(base.available_at),
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        revenue=revenue if revenue is not None else {},
    )


def test_history_reason_when_only_prior_arrives_later_is_missing_prior():
    # Rewritten: the version at Q3 time is built only from filings known
    # by then, so it cannot know a prior "arrives later". The amendment's
    # existence must not leak into the earlier row's reason code.
    result = _run([q3_fy25(), q2a_fy25_late()])

    lineage = _lineage(result, "Q3")
    assert lineage["method"] == "unavailable"
    assert lineage["reason"] == "missing_prior_fiscal_observation"
    assert pd.isna(lineage["prior_accession_number"])
    assert lineage["skipped_prior_accessions"] == ()

    # Once the amendment is public, Q3 gets a new, derived version.
    reversion = _lineage(result, "Q3", at=_t(q2a_fy25_late()))
    assert reversion["reason"] == "ok"
    assert reversion["value"] == pytest.approx(25000.0 - 15800.0)


def test_only_later_prior_does_not_leak_into_earlier_lineage_row():
    full = _run([q3_fy25(), q2a_fy25_late()])
    alone = _run([q3_fy25()], clock=_t(q3_fy25()) + pd.Timedelta(days=1))

    at_q3 = full.reconciliation[full.reconciliation["available_at"].eq(_t(q3_fy25()))]

    pd.testing.assert_frame_equal(
        at_q3.reset_index(drop=True),
        alone.reconciliation.reset_index(drop=True),
    )


def test_history_reason_missing_current_ytd_wins_over_prior_problems():
    # Q3 reports something for revenue, but not its YTD; the only Q2
    # candidate is also late. The current filing's own gap is the reason.
    q3_without_ytd = _filing_variant(q3_fy25(), accession="Q3", revenue={"FY": 99999.0})

    result = _run([q3_without_ytd, q2a_fy25_late()])

    assert _lineage(result, "Q3")["reason"] == "missing_current_ytd"


def test_history_reason_missing_current_fy():
    k_with_ytd_label = _filing_variant(k_fy25(), accession="K", revenue={"YTD": 37000.0})

    result = _run([q3_fy25(), k_with_ytd_label])

    lineage = _lineage(result, "K")
    assert lineage["reason"] == "missing_current_fy"
    assert pd.isna(lineage["prior_accession_number"])


def test_history_reason_missing_prior_ytd_with_skipped_amendment_visible():
    exhibit_only = _filing_variant(
        q2_fy25(),
        accession="Q2-EXHIBIT",
        form="10-Q/A",
        accepted_at="2025-05-15 20:00",
        revenue={},
    )

    result = _run([exhibit_only, q3_fy25()])

    lineage = _lineage(result, "Q3")
    assert lineage["reason"] == "missing_prior_ytd"
    assert lineage["skipped_prior_accessions"] == ("Q2-EXHIBIT",)


def test_item7_newer_amendment_without_input_is_skipped_but_audited():
    # Approved behavior: a partial amendment that omits revenue does not
    # prove the original revenue was superseded. The original is used and
    # the skipped amendment stays visible in lineage.
    exhibit_only = _filing_variant(
        q2_fy25(),
        accession="Q2-EXHIBIT",
        form="10-Q/A",
        accepted_at="2025-05-15 20:00",
        revenue={},
    )

    result = _run([q2_fy25(), exhibit_only, q3_fy25()])

    lineage = _lineage(result, "Q3")
    assert lineage["method"] == "ytd_minus_prior_ytd"
    assert lineage["prior_accession_number"] == "Q2"
    assert lineage["skipped_prior_accessions"] == ("Q2-EXHIBIT",)
    assert _row(result, "Q3")["revenue"] == pytest.approx(9000.0)


def test_history_reason_ambiguous_prior_at_same_instant():
    conflicting = _filing_variant(q2_fy25(), accession="Q2-B", revenue={"YTD": 15900.0})

    result = _run([q2_fy25(), conflicting, q3_fy25()])

    lineage = _lineage(result, "Q3")
    assert lineage["reason"] == "ambiguous_prior_observation"
    assert lineage["conflicting_prior_accessions"] == ("Q2", "Q2-B")
    assert pd.isna(_row(result, "Q3")["revenue"])


# ---------------------------------------------------------------------------
# Review fixes: lineage representation (item 2) and timing (item 5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("column", ["prior_accession_number", "prior_value_source"])
def test_lineage_missing_text_values_are_real_missing_values(column):
    lineage = _run(_full_timeline()).reconciliation

    # The missing-value-aware "string" dtype marks missing as pd.NA on
    # every supported pandas. (pandas 3's default "str" dtype uses NaN, and
    # pandas 2's astype(str) produces the literal text "None".)
    assert isinstance(lineage[column].dtype, pd.StringDtype)
    assert lineage[column].dtype.na_value is pd.NA

    direct_or_q1 = lineage[lineage["method"].isin(["direct_quarter", "q1_ytd"])]
    assert not direct_or_q1.empty
    assert all(value is pd.NA for value in direct_or_q1[column])

    # Never the literal text of a missing value.
    assert not lineage[column].fillna("").isin(["None", "nan", "<NA>"]).any()


def test_lineage_and_canonical_available_at_match_fixture_knowledge_time():
    timeline = [*_full_timeline()[:-1], k_fy25_without_acceptance()]
    expected = {filing.accession: filing.available_at for filing in timeline}

    result = _run(timeline)

    # Each filing's own version is available exactly at the fixture time.
    own = {
        accession: pd.Timestamp(_row(result, accession)["available_at"]) for accession in expected
    }
    assert own == expected

    # Every version (including re-versions) is available at some filing's
    # fixture time, and every lineage row belongs to exactly one version.
    assert set(pd.to_datetime(result.frame["available_at"], utc=True)) <= set(expected.values())

    versions = set(zip(result.frame["sec_accession_number"], result.frame["available_at"]))
    for _, row in result.reconciliation.iterrows():
        assert (row["sec_accession_number"], row["available_at"]) in versions


# ---------------------------------------------------------------------------
# Review fixes: accession normalization and non-strict behavior (item 3)
# ---------------------------------------------------------------------------


def test_accession_whitespace_is_normalized():
    q2 = q2_fy25()
    listing = [
        q1_fy25().metadata,
        {**q2.metadata, "accession_number": "  Q2 "},
        q3_fy25().metadata,
    ]

    result = _run([q1_fy25(), q2, q3_fy25()], listing_rows=listing)

    assert "Q2" in set(result.frame["sec_accession_number"])
    assert _lineage(result, "Q3")["prior_accession_number"] == "Q2"
    assert "Q2" in set(result.diagnostics["accession_number"])


def _duplicate_listing() -> list[dict[str, object]]:
    q2 = q2_fy25()
    return [
        q1_fy25().metadata,
        q2.metadata,
        {**q2.metadata, "accession_number": "Q2 "},
        q3_fy25().metadata,
    ]


def test_duplicate_accessions_are_skipped_when_not_strict():
    result = _run(
        [q1_fy25(), q2_fy25(), q3_fy25()],
        listing_rows=_duplicate_listing(),
        strict=False,
    )

    duplicates = result.diagnostics[result.diagnostics["reason"].eq("duplicate_accession")]
    assert duplicates["accession_number"].tolist() == ["Q2", "Q2"]
    assert duplicates["status"].eq("skipped").all()

    assert "Q2" not in set(result.frame["sec_accession_number"])
    assert _lineage(result, "Q3")["reason"] == "missing_prior_fiscal_observation"


def test_duplicate_accessions_raise_when_strict():
    with pytest.raises(history_module.EdgarHistoryBuildError, match="Duplicate"):
        _run(
            [q1_fy25(), q2_fy25(), q3_fy25()],
            listing_rows=_duplicate_listing(),
            strict=True,
        )


@pytest.mark.parametrize("bad_accession", ["", "   ", None])
def test_malformed_accessions_are_skipped_when_not_strict(bad_accession):
    listing = [
        {**q2_fy25().metadata, "accession_number": bad_accession},
        q3_fy25().metadata,
    ]

    result = _run([q3_fy25()], listing_rows=listing, strict=False)

    invalid = result.diagnostics[result.diagnostics["reason"].eq("invalid_accession")]
    assert len(invalid) == 1
    assert invalid.iloc[0]["status"] == "skipped"
    assert result.frame["sec_accession_number"].tolist() == ["Q3"]


def test_malformed_accession_raises_when_strict():
    listing = [{**q2_fy25().metadata, "accession_number": ""}]

    with pytest.raises(history_module.EdgarHistoryBuildError, match="accession"):
        _run([], listing_rows=listing, strict=True)


def test_non_finite_value_is_skipped_when_not_strict():
    bad_q2 = _filing_variant(q2_fy25(), accession="Q2", revenue={"YTD": float("inf")})

    result = _run([q1_fy25(), bad_q2, q3_fy25()], strict=False)

    bad = result.diagnostics[result.diagnostics["accession_number"].eq("Q2")].iloc[0]
    assert bad["status"] == "skipped"
    assert bad["reason"] == "quarter_build_error"
    assert _lineage(result, "Q3")["reason"] == "missing_prior_fiscal_observation"


def test_non_strict_build_failure_keeps_other_filings_reconciled():
    broken_q2 = _mu_filing(
        accession="Q2",
        form="10-Q",
        period_end="2025-02-27",
        accepted_at="2025-03-26 20:00",
        fiscal_year=2025,
        fiscal_period="Q2",
        revenue={},
        without_statements=True,
    )

    result = _run([q1_fy25(), broken_q2, q3_fy25(), q3a_fy25(), k_fy25()], strict=False)

    statuses = result.diagnostics.set_index("accession_number")["status"].to_dict()
    assert statuses["Q2"] == "skipped"

    # Without Q2, Q3 cannot be derived - and is not guessed.
    assert _lineage(result, "Q3")["reason"] == "missing_prior_fiscal_observation"
    # Unrelated derivations still work: Q4 = FY - Q3A YTD.
    assert _row(result, "K")["revenue"] == pytest.approx(37000.0 - 25500.0)


# ---------------------------------------------------------------------------
# Review fixes: boundary, fallback, multiple metrics (item 6)
# ---------------------------------------------------------------------------


def test_prior_accepted_at_the_same_instant_is_eligible():
    same_instant = _filing_variant(
        q2_fy25(),
        accession="Q2A-SAME",
        form="10-Q/A",
        accepted_at="2025-06-27 20:00",
        revenue={"YTD": 15800.0},
    )

    result = _run([q2_fy25(), same_instant, q3_fy25()])

    assert _lineage(result, "Q3")["prior_accession_number"] == "Q2A-SAME"
    assert _row(result, "Q3")["revenue"] == pytest.approx(9200.0)


def test_prior_accepted_one_second_later_is_not_eligible():
    one_second_later = _filing_variant(
        q2_fy25(),
        accession="Q2A-LATER",
        form="10-Q/A",
        accepted_at="2025-06-27 20:00:01",
        revenue={"YTD": 15800.0},
    )

    result = _run([q2_fy25(), one_second_later, q3_fy25()])

    assert _lineage(result, "Q3")["prior_accession_number"] == "Q2"
    assert _row(result, "Q3")["revenue"] == pytest.approx(9000.0)


def test_missing_acceptance_time_uses_conservative_fallback():
    # Filed on the Q3 acceptance date but without an acceptance time:
    # available_at = filing date + 1 day, which is AFTER Q3's acceptance.
    q2_fallback = _mu_filing(
        accession="Q2",
        form="10-Q",
        period_end="2025-02-27",
        accepted_at=None,
        filing_date="2025-06-27",
        fiscal_year=2025,
        fiscal_period="Q2",
        revenue={"YTD": 16000.0},
    )

    result = _run([q2_fallback, q3_fy25()])

    q2_row = _row(result, "Q2")
    assert pd.Timestamp(q2_row["available_at"]) == pd.Timestamp("2025-06-28", tz="UTC")
    assert q2_row["availability_source"] == "sec_filing_date_plus_1d"

    # Rewritten: at Q3's own time the fallback-dated Q2 is not yet known,
    # so the Q3 version has no prior (no leak of the later Q2).
    assert _lineage(result, "Q3")["reason"] == "missing_prior_fiscal_observation"

    # Q2 becomes usable only at its conservative time (filing date + 1
    # day); the derived Q3 version appears then, never earlier.
    reversion = _lineage(result, "Q3", at=pd.Timestamp("2025-06-28", tz="UTC"))
    assert reversion["prior_accession_number"] == "Q2"
    assert reversion["value"] == pytest.approx(9000.0)
    assert (
        _row(result, "Q3", at=pd.Timestamp("2025-06-28", tz="UTC"))["availability_source"]
        == "sec_filing_date_plus_1d"
    )


def test_second_flow_metric_is_reconciled_independently():
    result = _run([q1_fy25(), q2_fy25(), q3_fy25(), k_fy25()])

    ocf = result.frame.set_index("sec_accession_number")["operating_cash_flow"].to_dict()
    assert ocf == {
        "Q1": pytest.approx(3000.0),
        "Q2": pytest.approx(3000.0),
        "Q3": pytest.approx(3500.0),
        "K": pytest.approx(3500.0),
    }

    methods = {
        accession: _lineage(result, accession, "operating_cash_flow")["method"]
        for accession in ("Q1", "Q2", "Q3", "K")
    }
    assert methods == {
        "Q1": "q1_ytd",
        "Q2": "ytd_minus_prior_ytd",
        "Q3": "ytd_minus_prior_ytd",
        "K": "fy_minus_q3_ytd",
    }


# ---------------------------------------------------------------------------
# Real-shaped MU regression fixture (F1-F4)
#
# Shapes, accession numbers, acceptance times, filing dates and the
# own-period values are copied from the controlled MU smoke test against
# real EDGAR/edgartools output (FY2025). Prior-year comparative columns
# exist in the real statements; their values here are placeholders (111,
# 222, ...) because the pipeline must ignore them.
#
# Real edgartools shapes this fixture pins:
#   - Q1 10-Q exposes only "(Q1)" columns (no YTD) -> Q1 direct fallback
#   - Q2/Q3 income statements carry direct "(Qn)" and "(YTD)" columns
#   - cash flow statements are YTD-only in Q2/Q3, FY-only in the 10-K
#   - several XBRL reporting periods share one period end
#     (Q2 + YTD6 + unlabeled; FY + Q4 + unlabeled) plus comparatives
#   - 10-Qs accepted after 5:30 p.m. ET get the next business day as
#     their SEC filing date; the 10-K was accepted during the day
# ---------------------------------------------------------------------------


def _real_shaped_filing(
    *,
    accession: str,
    form: str,
    period_end: str,
    accepted_at: str,
    filing_date: str,
    expected_available_at: str,
    reporting_periods: list[tuple[str, int, str | None]],
    income: dict[str, float],
    cash_flow: dict[str, float],
) -> MuFiling:
    def statement(concept: str, columns: dict[str, float]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "concept": concept,
                    "label": "Statement line",
                    "dimension": False,
                    "abstract": False,
                    **columns,
                }
            ]
        )

    xbrl = _FakeXbrl(
        _FakeStatements(
            income=statement(REVENUE_CONCEPT, income),
            cash_flow=statement(OCF_CONCEPT, cash_flow),
        ),
        reporting_periods=[
            {
                "type": "duration",
                "end_date": end_date,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
            }
            for end_date, fiscal_year, fiscal_period in reporting_periods
        ],
    )

    accepted = pd.Timestamp(accepted_at)

    return MuFiling(
        accession=accession,
        available_at=pd.Timestamp(expected_available_at),
        metadata={
            "form": form,
            "period_end": pd.Timestamp(period_end, tz="UTC"),
            "filing_date": pd.Timestamp(filing_date, tz="UTC"),
            "accepted_at": accepted,
            "accession_number": accession,
            "is_xbrl": True,
        },
        filing=_FakeFiling(xbrl),
    )


MU_Q1 = "0000723125-24-000047"
MU_Q2 = "0000723125-25-000009"
MU_Q3 = "0000723125-25-000021"
MU_K = "0000723125-25-000028"


def _real_mu_fy2025() -> list[MuFiling]:
    return [
        _real_shaped_filing(
            accession=MU_Q1,
            form="10-Q",
            period_end="2024-11-28",
            accepted_at="2024-12-18T23:52:13Z",  # 18:52:13 EST
            filing_date="2024-12-19",
            expected_available_at="2024-12-19T11:00:00Z",  # 06:00 EST
            reporting_periods=[
                ("2024-11-28", 2025, "Q1"),
                ("2024-11-28", 2025, None),
                ("2023-11-30", 2024, "Q1"),
            ],
            income={"2024-11-28 (Q1)": 8_709e6, "2023-11-30 (Q1)": 111.0},
            cash_flow={"2024-11-28 (Q1)": 3_244e6, "2023-11-30 (Q1)": 222.0},
        ),
        _real_shaped_filing(
            accession=MU_Q2,
            form="10-Q",
            period_end="2025-02-27",
            accepted_at="2025-03-20T23:20:23Z",  # 19:20:23 EDT
            filing_date="2025-03-21",
            expected_available_at="2025-03-21T10:00:00Z",  # 06:00 EDT
            reporting_periods=[
                ("2025-02-27", 2025, "Q2"),
                ("2025-02-27", 2025, "YTD6"),
                ("2025-02-27", 2025, None),
                ("2024-02-29", 2024, "Q2"),
            ],
            income={
                "2025-02-27 (Q2)": 8_053e6,
                "2024-02-29 (Q2)": 333.0,
                "2025-02-27 (YTD)": 16_762e6,
                "2024-02-29 (YTD)": 444.0,
            },
            cash_flow={"2025-02-27 (YTD)": 7_186e6, "2024-02-29 (YTD)": 555.0},
        ),
        _real_shaped_filing(
            accession=MU_Q3,
            form="10-Q",
            period_end="2025-05-29",
            accepted_at="2025-06-25T22:50:42Z",  # 18:50:42 EDT
            filing_date="2025-06-26",
            expected_available_at="2025-06-26T10:00:00Z",  # 06:00 EDT
            reporting_periods=[
                ("2025-05-29", 2025, "Q3"),
                ("2025-05-29", 2025, "YTD9"),
                ("2025-05-29", 2025, None),
                ("2024-05-30", 2024, "Q3"),
            ],
            income={
                "2025-05-29 (Q3)": 9_301e6,
                "2024-05-30 (Q3)": 666.0,
                "2025-05-29 (YTD)": 26_063e6,
                "2024-05-30 (YTD)": 777.0,
            },
            cash_flow={"2025-05-29 (YTD)": 11_795e6, "2024-05-30 (YTD)": 888.0},
        ),
        _real_shaped_filing(
            accession=MU_K,
            form="10-K",
            period_end="2025-08-28",
            accepted_at="2025-10-03T18:42:25Z",  # 14:42:25 EDT, same day
            filing_date="2025-10-03",
            expected_available_at="2025-10-03T18:42:25Z",
            reporting_periods=[
                ("2025-08-28", 2025, "FY"),
                ("2025-08-28", 2025, "Q4"),
                ("2025-08-28", 2025, None),
                ("2024-08-29", 2024, "FY"),
            ],
            income={
                "2025-08-28 (FY)": 37_378e6,
                "2024-08-29 (FY)": 999.0,
                "2023-08-31 (FY)": 1_111.0,
            },
            cash_flow={
                "2025-08-28 (FY)": 17_525e6,
                "2024-08-29 (FY)": 1_222.0,
                "2023-08-31 (FY)": 1_333.0,
            },
        ),
    ]


def test_f1_real_shaped_mu_chain_reproduces_smoke_values():
    result = _run(_real_mu_fy2025())

    by_accession = result.frame.set_index("sec_accession_number")

    expected_ocf = {MU_Q1: 3_244e6, MU_Q2: 3_942e6, MU_Q3: 4_609e6, MU_K: 5_730e6}
    expected_revenue = {MU_Q1: 8_709e6, MU_Q2: 8_053e6, MU_Q3: 9_301e6, MU_K: 11_315e6}

    assert by_accession["operating_cash_flow"].to_dict() == pytest.approx(expected_ocf)
    assert by_accession["revenue"].to_dict() == pytest.approx(expected_revenue)

    # Standalone quarters add back up to the reported fiscal-year totals.
    assert sum(expected_ocf.values()) == pytest.approx(17_525e6)
    assert sum(expected_revenue.values()) == pytest.approx(37_378e6)

    ocf = {a: _lineage(result, a, "operating_cash_flow") for a in expected_ocf}
    assert ocf[MU_Q1]["method"] == "direct_quarter"
    assert ocf[MU_Q2]["method"] == "ytd_minus_prior_ytd"
    assert ocf[MU_Q2]["prior_accession_number"] == MU_Q1
    assert ocf[MU_Q2]["prior_value_source"] == "q1_direct"
    assert ocf[MU_Q3]["prior_accession_number"] == MU_Q2
    assert ocf[MU_Q3]["prior_value_source"] == "ytd"
    assert ocf[MU_K]["method"] == "fy_minus_q3_ytd"
    assert ocf[MU_K]["prior_accession_number"] == MU_Q3

    # Revenue Q1-Q3 are reported directly; only Q4 needs subtraction.
    assert _lineage(result, MU_Q3)["method"] == "direct_quarter"
    assert _lineage(result, MU_K)["method"] == "fy_minus_q3_ytd"


def test_f2_multiple_reporting_period_entries_resolve_fiscal_identity():
    lineage = _run(_real_mu_fy2025()).reconciliation
    revenue = lineage[lineage["metric_name"].eq("revenue")]

    identity = revenue.set_index("sec_accession_number")[["fiscal_year", "fiscal_quarter"]]

    assert identity.to_dict("index") == {
        MU_Q1: {"fiscal_year": 2025, "fiscal_quarter": 1},
        MU_Q2: {"fiscal_year": 2025, "fiscal_quarter": 2},
        MU_Q3: {"fiscal_year": 2025, "fiscal_quarter": 3},
        MU_K: {"fiscal_year": 2025, "fiscal_quarter": 4},
    }


def test_f3_after_hours_filings_use_deferred_filing_date_availability():
    filings = _real_mu_fy2025()
    result = _run(filings)

    canonical = result.frame.set_index("sec_accession_number")

    for filing in filings:
        row = canonical.loc[filing.accession]
        assert pd.Timestamp(row["available_at"]) == filing.available_at, filing.accession

    assert canonical.loc[MU_Q1, "availability_source"] == "sec_deferred_filing_date_6am"
    assert canonical.loc[MU_Q2, "availability_source"] == "sec_deferred_filing_date_6am"
    assert canonical.loc[MU_Q3, "availability_source"] == "sec_deferred_filing_date_6am"
    assert canonical.loc[MU_K, "availability_source"] == "sec_acceptance_datetime"

    # Lineage timing follows the deferred availability, for both the
    # current filing and the prior it used.
    expected = {filing.accession: filing.available_at for filing in filings}

    for accession, prior in ((MU_Q2, MU_Q1), (MU_Q3, MU_Q2), (MU_K, MU_Q3)):
        row = _lineage(result, accession, "operating_cash_flow")
        assert row["available_at"] == expected[accession]
        assert row["prior_available_at"] == expected[prior]


def test_f4_real_shaped_mu_history_is_prefix_stable():
    timeline = _real_mu_fy2025()
    knowable_at = {filing.accession: filing.available_at for filing in timeline}
    full = _run(timeline)

    for cutoff in sorted(set(knowable_at.values())):
        known = [filing for filing in timeline if filing.available_at <= cutoff]
        prefix = _run(known, clock=cutoff + pd.Timedelta(days=1))

        pd.testing.assert_frame_equal(
            _known_by(full.frame, cutoff),
            _known_by(prefix.frame, cutoff),
        )
        pd.testing.assert_frame_equal(
            _lineage_known_by(full.reconciliation, cutoff),
            _lineage_known_by(prefix.reconciliation, cutoff),
        )
        pd.testing.assert_frame_equal(
            _diagnostics_known_by(full.diagnostics, cutoff, knowable_at),
            _diagnostics_known_by(prefix.diagnostics, cutoff, knowable_at),
        )

    # Between acceptance (18:52 EST Dec 18) and the deferred filing time
    # (06:00 EST Dec 19), the Q1 filing is not yet usable.
    overnight = pd.Timestamp("2024-12-19T03:00:00Z")
    assert _known_by(full.frame, overnight).empty


# ---------------------------------------------------------------------------
# Runs between an after-hours acceptance and its deferred availability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strict", [True, False])
def test_filing_not_yet_available_at_run_time_is_skipped(strict):
    # Clock: 22:00 EST Dec 18 - after MU Q1 acceptance (18:52 EST), before
    # its deferred availability (06:00 EST Dec 19).
    clock = pd.Timestamp("2024-12-19T03:00:00Z")

    result = _run(_real_mu_fy2025()[:1], clock=clock, strict=strict)

    assert result.frame.empty

    diagnostic = result.diagnostics.iloc[0]
    assert diagnostic["accession_number"] == MU_Q1
    assert diagnostic["status"] == "skipped"
    assert diagnostic["reason"] == "not_yet_available"
    assert "2024-12-19T11:00:00+00:00" in diagnostic["detail"]


def test_filing_is_used_from_its_available_at_onward():
    result = _run(_real_mu_fy2025()[:1], clock=pd.Timestamp("2024-12-19T11:00:00Z"))

    assert result.frame["sec_accession_number"].tolist() == [MU_Q1]
    assert result.diagnostics["status"].tolist() == ["success"]


def test_not_yet_available_filings_do_not_block_earlier_ones():
    # Clock: 23:00 EDT Jun 25 2025, after MU Q3 acceptance, before 06:00.
    result = _run(_real_mu_fy2025(), clock=pd.Timestamp("2025-06-26T03:00:00Z"), strict=True)

    assert result.frame["sec_accession_number"].tolist() == [MU_Q1, MU_Q2]
    assert _lineage(result, MU_Q2, "operating_cash_flow")["value"] == pytest.approx(3_942e6)

    skipped = result.diagnostics[result.diagnostics["reason"].eq("not_yet_available")]
    assert sorted(skipped["accession_number"]) == sorted([MU_Q3, MU_K])


# ---------------------------------------------------------------------------
# Versioned derived observations (T1-T12)
#
# A fiscal period p gets a new version whenever one of its own filings is
# accepted, or a prior-quarter filing changes any reconciled metric. Each
# version is available at the instant of the event that produced it.
# ---------------------------------------------------------------------------


def q1a_fy25(accepted_at: str = "2025-07-20 20:00") -> MuFiling:
    # Q1 amendment restating revenue only (no cash-flow statement).
    return _mu_filing(
        accession="Q1A",
        form="10-Q/A",
        period_end="2024-11-28",
        accepted_at=accepted_at,
        fiscal_year=2025,
        fiscal_period="Q1",
        revenue={"YTD": 8600.0},
    )


def q2a_fy25_noop() -> MuFiling:
    # Repeats the already-current Q2 YTD (15800) after Q2A-LATE.
    return _mu_filing(
        accession="Q2A-NOOP",
        form="10-Q/A",
        period_end="2025-02-27",
        accepted_at="2025-07-25 20:00",
        fiscal_year=2025,
        fiscal_period="Q2",
        revenue={"YTD": 15800.0},
    )


def q3a_fy25_partial(accepted_at: str = "2025-09-05 20:00") -> MuFiling:
    # Partial own-period amendment: restates OCF only, omits revenue.
    return _mu_filing(
        accession="Q3A-PARTIAL",
        form="10-Q/A",
        period_end="2025-05-29",
        accepted_at=accepted_at,
        fiscal_year=2025,
        fiscal_period="Q3",
        revenue={},
        operating_cash_flow={"YTD": 9600.0},
    )


def _versioned_timeline() -> list[MuFiling]:
    return [
        q1_fy25(),
        q2_fy25(),
        q3_fy25(),
        q2a_fy25_late(),
        q1a_fy25(),
        q2a_fy25_noop(),
        q3a_fy25(),
        broken_filing(),
        q3a_fy25_partial(),
        k_fy25(),
    ]


def _t(filing: MuFiling) -> pd.Timestamp:
    return filing.available_at


Q1_END, Q2_END, Q3_END, K_END = "2024-11-28", "2025-02-27", "2025-05-29", "2025-08-28"

# (period_end, anchor, available_at, revenue, operating_cash_flow)
_VERSIONED_EXPECTED = [
    (Q1_END, "Q1", _t(q1_fy25()), 8700.0, 3000.0),
    (Q2_END, "Q2", _t(q2_fy25()), 7300.0, 3000.0),
    (Q3_END, "Q3", _t(q3_fy25()), 9000.0, 3500.0),
    (Q2_END, "Q2A-LATE", _t(q2a_fy25_late()), 7100.0, 3000.0),
    (Q3_END, "Q3", _t(q2a_fy25_late()), 9200.0, 3500.0),
    (Q1_END, "Q1A", _t(q1a_fy25()), 8600.0, 3000.0),
    (Q2_END, "Q2A-LATE", _t(q1a_fy25()), 7200.0, 3000.0),
    (Q2_END, "Q2A-NOOP", _t(q2a_fy25_noop()), 7200.0, 3000.0),
    (Q3_END, "Q3A", _t(q3a_fy25()), 9700.0, 3500.0),
    (Q3_END, "Q3A-PARTIAL", _t(q3a_fy25_partial()), 9700.0, 3600.0),
    (K_END, "K", _t(k_fy25()), 11500.0, 3400.0),
]


def _version_tuples(result: EdgarHistoryResult) -> list[tuple[object, ...]]:
    return [
        (
            str(pd.Timestamp(row["period_end"]).date()),
            row["sec_accession_number"],
            pd.Timestamp(row["available_at"]),
            row["revenue"],
            row["operating_cash_flow"],
        )
        for _, row in result.frame.iterrows()
    ]


def _assert_prefix_stable(timeline: list[MuFiling]) -> EdgarHistoryResult:
    knowable_at = {filing.accession: filing.available_at for filing in timeline}

    full = _run(timeline, strict=False)

    for cutoff in sorted(set(knowable_at.values())):
        known_filings = [filing for filing in timeline if filing.available_at <= cutoff]
        prefix = _run(known_filings, clock=cutoff + pd.Timedelta(days=1), strict=False)

        pd.testing.assert_frame_equal(
            _known_by(full.frame, cutoff),
            _known_by(prefix.frame, cutoff),
            obj=f"canonical history known by {cutoff}",
        )
        pd.testing.assert_frame_equal(
            _lineage_known_by(full.reconciliation, cutoff),
            _lineage_known_by(prefix.reconciliation, cutoff),
            obj=f"reconciliation lineage known by {cutoff}",
        )
        pd.testing.assert_frame_equal(
            _diagnostics_known_by(full.diagnostics, cutoff, knowable_at),
            _diagnostics_known_by(prefix.diagnostics, cutoff, knowable_at),
            obj=f"diagnostics known by {cutoff}",
        )

        # The prefix run has nothing beyond the cutoff.
        assert (pd.to_datetime(prefix.frame["available_at"], utc=True) <= cutoff).all()

    return full


def test_t1_versioned_history_is_prefix_stable_at_every_knowledge_time():
    """
    history(all filings) restricted to knowledge time <= t
    == history(only filings knowable by t), for frame, lineage and
    diagnostics, at every filing time - including the times at which
    earlier periods are re-versioned (Q2A-LATE, Q1A, Q2A-NOOP).
    """

    full = _assert_prefix_stable(_versioned_timeline())

    assert _version_tuples(full) == _VERSIONED_EXPECTED


def test_t1_prefix_stable_when_only_prior_arrives_later():
    # Q3 and the 10-K precede every Q2 filing: the Q3 version at Q3 time
    # must not know that a prior will arrive later.
    full = _assert_prefix_stable([q3_fy25(), k_fy25(), q2a_fy25_late()])

    versions = _versions(full, Q3_END)
    assert [(anchor, at) for anchor, at, _ in versions] == [
        ("Q3", _t(q3_fy25())),
        ("Q3", _t(q2a_fy25_late())),
    ]
    assert pd.isna(versions[0][2])
    assert versions[1][2] == pytest.approx(9200.0)
    assert _lineage(full, "Q3")["reason"] == "missing_prior_fiscal_observation"


# Independent oracle input, written from the fixture definitions above:
# accession -> (fiscal quarter, own-period input per metric). Inputs are
# YTD for fiscal Q1-Q3 and FY for Q4; a missing metric is omitted.
_ORACLE_FILINGS: dict[str, tuple[int, dict[str, float]]] = {
    "Q1": (1, {"revenue": 8700.0, "operating_cash_flow": 3000.0}),
    "Q2": (2, {"revenue": 16000.0, "operating_cash_flow": 6000.0}),
    "Q3": (3, {"revenue": 25000.0, "operating_cash_flow": 9500.0}),
    "Q2A-LATE": (2, {"revenue": 15800.0}),
    "Q1A": (1, {"revenue": 8600.0}),
    "Q2A-NOOP": (2, {"revenue": 15800.0}),
    "Q3A": (3, {"revenue": 25500.0}),
    "Q3A-PARTIAL": (3, {"operating_cash_flow": 9600.0}),
    "K": (4, {"revenue": 37000.0, "operating_cash_flow": 13000.0}),
}

_ORACLE_PERIOD_ENDS = {1: Q1_END, 2: Q2_END, 3: Q3_END, 4: K_END}


def _oracle_value(
    available_at: dict[str, pd.Timestamp],
    t: pd.Timestamp,
    quarter: int,
    metric: str,
) -> float | None:
    """Brute force: latest knowable own input minus latest knowable prior input."""

    def latest_input(q: int) -> float | None:
        known = [
            (available_at[accession], values[metric])
            for accession, (fiscal_quarter, values) in _ORACLE_FILINGS.items()
            if fiscal_quarter == q and metric in values and available_at[accession] <= t
        ]
        if not known:
            return None

        latest_time = max(time for time, _ in known)
        latest_values = {value for time, value in known if time == latest_time}

        # Equal-time filings with different values are ambiguous: the
        # design makes the value unavailable rather than picking one.
        return latest_values.pop() if len(latest_values) == 1 else None

    current = latest_input(quarter)

    if current is None or quarter == 1:
        return current

    prior = latest_input(quarter - 1)
    return None if prior is None else current - prior


def test_t2_history_is_as_of_complete_against_brute_force_oracle():
    timeline = _versioned_timeline()
    available_at = {filing.accession: filing.available_at for filing in timeline}

    result = _run(timeline, strict=False)
    frame = result.frame.assign(
        _available_at=pd.to_datetime(result.frame["available_at"], utc=True),
        _period_end=pd.to_datetime(result.frame["period_end"], utc=True),
    )

    filing_times = sorted(set(available_at.values()))
    query_times = sorted(
        {
            *filing_times,
            *(time + pd.Timedelta(seconds=1) for time in filing_times),
            *(time - pd.Timedelta(seconds=1) for time in filing_times),
            pd.Timestamp("2026-12-31", tz="UTC"),
        }
    )

    checked = 0

    for t in query_times:
        for quarter, period_end in _ORACLE_PERIOD_ENDS.items():
            own_known = any(
                fiscal_quarter == quarter and available_at[accession] <= t
                for accession, (fiscal_quarter, _) in _ORACLE_FILINGS.items()
            )

            versions = frame[
                frame["_period_end"].eq(pd.Timestamp(period_end, tz="UTC"))
                & frame["_available_at"].le(t)
            ]

            if not own_known:
                assert versions.empty, (t, quarter)
                continue

            latest = versions.sort_values("_available_at").iloc[-1]

            for metric in ("revenue", "operating_cash_flow"):
                expected = _oracle_value(available_at, t, quarter, metric)
                if expected is None:
                    assert pd.isna(latest[metric]), (t, quarter, metric)
                else:
                    assert latest[metric] == pytest.approx(expected), (t, quarter, metric)
                checked += 1

    assert checked > 100

    # Literal pins, independent of the oracle.
    end = pd.Timestamp("2026-12-31", tz="UTC")
    assert _oracle_value(available_at, end, 3, "revenue") == 9700.0
    assert _oracle_value(available_at, end, 4, "revenue") == 11500.0
    assert _oracle_value(available_at, end, 4, "operating_cash_flow") == 3400.0
    assert _oracle_value(available_at, _t(q2a_fy25_late()), 3, "revenue") == 9200.0
    assert _row(result, "Q3", at=_t(q2a_fy25_late()))["revenue"] == 9200.0
    assert _row(result, "K")["revenue"] == 11500.0


def test_t3_versions_are_never_backdated():
    timeline = _versioned_timeline()
    filing_times = {filing.available_at for filing in timeline}

    result = _run(timeline, strict=False)
    lineage = result.reconciliation

    assert set(pd.to_datetime(result.frame["available_at"], utc=True)) <= filing_times
    assert set(lineage["available_at"]) <= filing_times

    with_source = lineage[lineage["source_available_at"].notna()]
    # Every reconciled value has a source filing; only metrics with no
    # usable current-side input (e.g. never reported) have none.
    assert lineage[lineage["value"].notna()].index.isin(with_source.index).all()
    assert lineage.loc[~lineage.index.isin(with_source.index), "method"].eq("unavailable").all()
    assert (with_source["source_available_at"] <= with_source["available_at"]).all()

    with_prior = lineage[lineage["prior_available_at"].notna()]
    assert not with_prior.empty
    assert (with_prior["prior_available_at"] <= with_prior["available_at"]).all()


def _asof_rows(result: EdgarHistoryResult, dates: list[pd.Timestamp]) -> pd.DataFrame:
    market = pd.DataFrame({"date": dates, "symbol": "MU"})
    return align_quarterly_fundamentals_asof(market_frame=market, fundamentals=result.frame)


def _daily_and_event_dates(filings: list[MuFiling]) -> list[pd.Timestamp]:
    daily = pd.date_range("2024-12-01", "2025-12-31", freq="D", tz="UTC")
    events = {filing.available_at for filing in filings}
    return sorted({*daily, *events, *(time - pd.Timedelta(seconds=1) for time in events)})


def test_t4_late_prior_amendment_is_never_the_current_asof_row():
    timeline = _full_timeline()
    result = _run(timeline)
    dates = _daily_and_event_dates(timeline)
    aligned = _asof_rows(result, dates)

    late, q3a = _t(q2a_fy25_late()), _t(q3a_fy25())

    # Q2A-LATE (accepted after Q3) is never the current row.
    assert not aligned["sec_accession_number"].eq("Q2A-LATE").any()

    after_q3 = aligned[aligned["date"] >= _t(q3_fy25())]
    assert (
        not pd.to_datetime(after_q3["period_end"], utc=True)
        .eq(pd.Timestamp(Q2_END, tz="UTC"))
        .any()
    )

    # The Q3 re-version (9200) is current exactly on [Q2A-LATE, Q3A).
    window = aligned[(aligned["date"] >= late) & (aligned["date"] < q3a)]
    assert not window.empty
    assert window["sec_accession_number"].eq("Q3").all()
    assert window["revenue"].eq(9200.0).all()
    assert (window["available_at"] == late).all()

    reversion_current = aligned[aligned["available_at"] == late]
    assert reversion_current["date"].min() == late
    assert reversion_current["date"].max() < q3a


def test_t4_early_prior_amendment_is_current_only_until_q3():
    timeline = [*_full_timeline(), q2a_fy25_early()]
    result = _run(timeline)
    aligned = _asof_rows(result, _daily_and_event_dates(timeline))

    early, q3 = _t(q2a_fy25_early()), _t(q3_fy25())

    current_early = aligned[aligned["sec_accession_number"].eq("Q2A-EARLY")]
    assert not current_early.empty
    assert current_early["date"].min() == early
    assert current_early["date"].max() < q3
    assert (
        aligned[(aligned["date"] >= early) & (aligned["date"] < q3)]["sec_accession_number"]
        .eq("Q2A-EARLY")
        .all()
    )


def test_t5_versioned_history_is_independent_of_filing_order():
    timeline = _versioned_timeline()
    baseline = _run(timeline, strict=False)

    rng = random.Random(20261005)

    for _ in range(10):
        shuffled = timeline[:]
        rng.shuffle(shuffled)

        result = _run(shuffled, strict=False)

        pd.testing.assert_frame_equal(result.frame, baseline.frame)
        pd.testing.assert_frame_equal(result.reconciliation, baseline.reconciliation)
        pd.testing.assert_frame_equal(
            _sorted_diagnostics(result.diagnostics),
            _sorted_diagnostics(baseline.diagnostics),
        )


def test_t6a_partial_own_period_amendment_carries_omitted_metrics_forward():
    partial = q3a_fy25_partial(accepted_at="2025-07-01 20:00")
    result = _run([q1_fy25(), q2_fy25(), q3_fy25(), partial])

    assert _versions(
        result, Q3_END, ("sec_accession_number", "revenue", "operating_cash_flow")
    ) == [
        ("Q3", 9000.0, 3500.0),
        ("Q3A-PARTIAL", 9000.0, 3600.0),
    ]

    revenue = _lineage(result, "Q3A-PARTIAL", "revenue")
    assert revenue["source_accession_number"] == "Q3"
    assert revenue["source_available_at"] == _t(q3_fy25())
    assert revenue["method"] == "ytd_minus_prior_ytd"

    ocf = _lineage(result, "Q3A-PARTIAL", "operating_cash_flow")
    assert ocf["source_accession_number"] == "Q3A-PARTIAL"
    assert ocf["version_trigger_accessions"] == ("Q3A-PARTIAL",)


@pytest.mark.parametrize(
    "amendment",
    [
        _filing_variant(
            q2_fy25(),
            accession="Q2A-SAME",
            form="10-Q/A",
            accepted_at="2025-07-15 20:00",
            revenue={"YTD": 16000.0},
        ),
        _filing_variant(
            q2_fy25(),
            accession="Q2A-EXHIBIT",
            form="10-Q/A",
            accepted_at="2025-07-15 20:00",
            revenue={},
        ),
    ],
    ids=["same_value", "metric_missing"],
)
def test_t6b_prior_amendment_without_changed_input_emits_no_version(amendment):
    result = _run([q1_fy25(), q2_fy25(), q3_fy25(), amendment])

    # The amendment's own period is re-versioned (filings stay visible)...
    assert [v[0] for v in _versions(result, Q2_END)] == ["Q2", amendment.accession]

    # ...but Q3's (value, method, reason) is unchanged, so no Q3 version.
    assert _versions(result, Q3_END) == [("Q3", _t(q3_fy25()), 9000.0)]


def test_t7_q3_amendment_then_late_q2_amendment():
    late_after_q3a = _filing_variant(
        q2a_fy25_late(),
        accession="Q2A-AFTER-Q3A",
        accepted_at="2025-08-15 20:00",
        revenue={"YTD": 15800.0},
    )

    result = _run([q1_fy25(), q2_fy25(), q3_fy25(), q3a_fy25(), late_after_q3a])

    assert _versions(result, Q3_END) == [
        ("Q3", _t(q3_fy25()), 9000.0),
        ("Q3A", _t(q3a_fy25()), 9500.0),
        ("Q3A", _t(late_after_q3a), 9700.0),
    ]

    # Once Q3A exists, the original Q3 is never re-derived.
    lineage = result.reconciliation
    after_q3a = lineage[
        lineage["available_at"].ge(_t(q3a_fy25()))
        & lineage["period_end"].eq(pd.Timestamp(Q3_END, tz="UTC"))
        & lineage["metric_name"].eq("revenue")
    ]
    assert after_q3a["source_accession_number"].eq("Q3A").all()
    assert after_q3a["sec_accession_number"].eq("Q3A").all()


def test_t8_q1_amendment_reversions_q2_amendment_not_q3():
    result = _run([q1_fy25(), q2_fy25(), q3_fy25(), q2a_fy25_late(), q1a_fy25()])

    assert _versions(result, Q2_END) == [
        ("Q2", _t(q2_fy25()), 7300.0),
        ("Q2A-LATE", _t(q2a_fy25_late()), 7100.0),
        ("Q2A-LATE", _t(q1a_fy25()), 7200.0),
    ]
    assert _versions(result, Q3_END) == [
        ("Q3", _t(q3_fy25()), 9000.0),
        ("Q3", _t(q2a_fy25_late()), 9200.0),
    ]

    reversion = _lineage(result, "Q2A-LATE", at=_t(q1a_fy25()))
    assert reversion["prior_accession_number"] == "Q1A"
    assert reversion["version_trigger_accessions"] == ("Q1A",)


def test_t9_q3_amendment_after_10k_reversions_q4_within_fiscal_year():
    q3a_after_k = _filing_variant(
        q3a_fy25(),
        accession="Q3A-AFTER-K",
        accepted_at="2025-11-01 20:00",
        revenue={"YTD": 25500.0},
    )

    result = _run([q2_fy25(), q3_fy25(), k_fy25(), q3a_after_k, q1_fy26(), q2_fy26()])

    assert _versions(result, K_END) == [
        ("K", _t(k_fy25()), 12000.0),
        ("K", _t(q3a_after_k), 37000.0 - 25500.0),
    ]

    lineage = _lineage(result, "K", at=_t(q3a_after_k))
    assert lineage["prior_accession_number"] == "Q3A-AFTER-K"
    assert lineage["method"] == "fy_minus_q3_ytd"

    # FY2026 is never touched by FY2025 filings.
    assert _versions(result, "2025-11-27") == [("Q1-FY26", _t(q1_fy26()), 8900.0)]
    q2_fy26_lineage = _lineage(result, "Q2-FY26")
    assert q2_fy26_lineage["prior_accession_number"] == "Q1-FY26"
    assert len(_versions(result, "2026-02-26")) == 1


def test_t11_reconciliation_inputs_are_raw_filings_only():
    timeline = [q1_fy25(), q2_fy25(), q3_fy25(), q1a_fy25("2025-07-01 20:00"), k_fy25()]
    raw_times = {filing.accession: filing.available_at for filing in timeline}

    result = _run(timeline)
    lineage = result.reconciliation

    for _, row in lineage.iterrows():
        for accession_column, time_column in (
            ("source_accession_number", "source_available_at"),
            ("prior_accession_number", "prior_available_at"),
        ):
            if pd.isna(row[accession_column]):
                assert pd.isna(row[time_column])
                continue
            assert row[accession_column] in raw_times
            assert row[time_column] == raw_times[row[accession_column]]

    # Q1A re-versions Q2 from the RAW Q1A YTD...
    assert _versions(result, Q2_END) == [
        ("Q2", _t(q2_fy25()), 7300.0),
        ("Q2", raw_times["Q1A"], 16000.0 - 8600.0),
    ]
    # ...and Q3 (= raw Q3 YTD - raw Q2 YTD) is unchanged, with no version.
    assert _versions(result, Q3_END) == [("Q3", _t(q3_fy25()), 9000.0)]

    # Q4 = FY - raw Q3 YTD (not FY minus any derived standalone value).
    k = _lineage(result, "K")
    assert k["prior_accession_number"] == "Q3"
    assert _row(result, "K")["revenue"] == 37000.0 - 25000.0


def test_t12_same_timestamp_events_are_coalesced():
    q1a_with_q2 = q1a_fy25(accepted_at="2025-03-26 20:00")
    result = _run([q1_fy25(), q2_fy25(), q1a_with_q2])

    q2_versions = _versions(result, Q2_END)
    assert q2_versions == [("Q2", _t(q2_fy25()), 16000.0 - 8600.0)]

    lineage = _lineage(result, "Q2")
    assert lineage["version_trigger_accessions"] == ("Q1A", "Q2")
    assert lineage["prior_accession_number"] == "Q1A"


def test_t12_same_instant_prior_amendment_and_reversion_asof_is_order_independent():
    result = _run(_full_timeline())
    late = _t(q2a_fy25_late())

    # Q2A-LATE and the Q3 re-version share one instant; Q3 is current.
    at_late = result.frame[pd.to_datetime(result.frame["available_at"], utc=True).eq(late)]
    assert sorted(at_late["sec_accession_number"]) == ["Q2A-LATE", "Q3"]

    dates = _daily_and_event_dates(_full_timeline())
    market = pd.DataFrame({"date": dates, "symbol": "MU"})

    baseline = align_quarterly_fundamentals_asof(market_frame=market, fundamentals=result.frame)
    at = baseline[baseline["date"].eq(late)].iloc[0]
    assert at["sec_accession_number"] == "Q3"
    assert at["revenue"] == 9200.0

    canonical = baseline.sort_values("date", kind="mergesort").reset_index(drop=True)

    orders = [
        (market.iloc[::-1], result.frame.iloc[::-1]),
        *(
            (
                market.sample(frac=1.0, random_state=seed),
                result.frame.sample(frac=1.0, random_state=seed + 100),
            )
            for seed in (1, 2, 3)
        ),
    ]

    for shuffled_market, shuffled_fundamentals in orders:
        shuffled_market = shuffled_market.reset_index(drop=True)
        aligned = align_quarterly_fundamentals_asof(
            market_frame=shuffled_market,
            fundamentals=shuffled_fundamentals.reset_index(drop=True),
        )

        assert aligned["date"].equals(shuffled_market["date"])
        pd.testing.assert_frame_equal(
            aligned.sort_values("date", kind="mergesort").reset_index(drop=True),
            canonical,
        )


# ---------------------------------------------------------------------------
# Conflicting period ends within one fiscal period
# ---------------------------------------------------------------------------


def _q3_with_other_period_end(
    accession: str = "Q3-ODD",
    accepted_at: str = "2025-07-01 20:00",
) -> MuFiling:
    # Same fiscal identity (FY2025 Q3) but a different period end, with a
    # YTD that would visibly change Q3 and Q4 if it were ever used.
    return _mu_filing(
        accession=accession,
        form="10-Q/A",
        period_end="2025-05-30",
        accepted_at=accepted_at,
        fiscal_year=2025,
        fiscal_period="Q3",
        revenue={"YTD": 26000.0},
    )


def _conflict_timeline() -> list[MuFiling]:
    return [q2_fy25(), q3_fy25(), _q3_with_other_period_end(), q3a_fy25(), k_fy25()]


def test_conflicting_period_end_raises_when_strict():
    with pytest.raises(history_module.EdgarHistoryBuildError, match="period_end"):
        _run([q2_fy25(), q3_fy25(), _q3_with_other_period_end()], strict=True)


def test_conflicting_period_end_rejects_only_the_disagreeing_filing():
    result = _run(_conflict_timeline(), strict=False)

    conflicts = result.diagnostics[result.diagnostics["reason"].eq("conflicting_period_end")]
    assert conflicts["accession_number"].tolist() == ["Q3-ODD"]
    assert conflicts["status"].tolist() == ["skipped"]

    # The rejected filing has no success diagnostic.
    odd = result.diagnostics[result.diagnostics["accession_number"].eq("Q3-ODD")]
    assert odd["reason"].tolist() == ["conflicting_period_end"]

    # Earlier versions are unaffected; Q3-ODD is never anchor, trigger,
    # source, or prior, and never feeds the 10-K.
    assert _versions(result, Q3_END) == [
        ("Q3", _t(q3_fy25()), 9000.0),
        ("Q3A", _t(q3a_fy25()), 25500.0 - 16000.0),
    ]
    assert _row(result, "K")["revenue"] == pytest.approx(37000.0 - 25500.0)

    lineage = result.reconciliation
    assert "Q3-ODD" not in set(result.frame["sec_accession_number"])
    for column in ("sec_accession_number", "source_accession_number", "prior_accession_number"):
        assert not lineage[column].eq("Q3-ODD").any()
    for column in (
        "version_trigger_accessions",
        "skipped_prior_accessions",
        "equivalent_prior_accessions",
        "conflicting_prior_accessions",
    ):
        assert not any("Q3-ODD" in value for value in lineage[column])


def test_rejected_filing_never_feeds_next_quarter_prior():
    # Without Q3A, the latest Q3-identity filing before the 10-K is the
    # rejected Q3-ODD (YTD 26000). Q4 must still use the original Q3.
    result = _run([q2_fy25(), q3_fy25(), _q3_with_other_period_end(), k_fy25()], strict=False)

    k = _lineage(result, "K")
    assert k["prior_accession_number"] == "Q3"
    assert _row(result, "K")["revenue"] == pytest.approx(37000.0 - 25000.0)


def test_conflicting_period_end_is_prefix_stable():
    _assert_prefix_stable(_conflict_timeline())


def test_conflicting_period_end_is_order_independent():
    timeline = _conflict_timeline()
    baseline = _run(timeline, strict=False)

    rng = random.Random(7)

    for _ in range(10):
        shuffled = timeline[:]
        rng.shuffle(shuffled)
        result = _run(shuffled, strict=False)

        pd.testing.assert_frame_equal(result.frame, baseline.frame)
        pd.testing.assert_frame_equal(result.reconciliation, baseline.reconciliation)
        pd.testing.assert_frame_equal(
            _sorted_diagnostics(result.diagnostics),
            _sorted_diagnostics(baseline.diagnostics),
        )


def test_earliest_same_instant_period_end_tie_rejects_all_and_moves_reference():
    # Q3 and Q3-TIE are the earliest Q3 filings, at the same instant, with
    # different period ends: both are rejected. Q3A becomes the reference.
    tie = _q3_with_other_period_end(accession="Q3-TIE", accepted_at="2025-06-27 20:00")

    result = _run([q2_fy25(), q3_fy25(), tie, q3a_fy25(), k_fy25()], strict=False)

    conflicts = result.diagnostics[result.diagnostics["reason"].eq("conflicting_period_end")]
    assert sorted(conflicts["accession_number"]) == ["Q3", "Q3-TIE"]

    assert _versions(result, Q3_END) == [("Q3A", _t(q3a_fy25()), 25500.0 - 16000.0)]
    assert _row(result, "K")["revenue"] == pytest.approx(37000.0 - 25500.0)

    _assert_prefix_stable([q2_fy25(), q3_fy25(), tie, q3a_fy25(), k_fy25()])


def test_lineage_new_columns_have_explicit_dtypes():
    lineage = _run(_full_timeline()).reconciliation

    assert lineage.columns.tolist() == RECONCILIATION_COLUMNS
    assert RECONCILIATION_COLUMNS[-3:] == [
        "source_accession_number",
        "source_available_at",
        "version_trigger_accessions",
    ]
    assert isinstance(lineage["source_accession_number"].dtype, pd.StringDtype)
    assert str(lineage["source_available_at"].dtype) == "datetime64[ns, UTC]"
    assert all(isinstance(value, tuple) for value in lineage["version_trigger_accessions"])


# ---------------------------------------------------------------------------
# Same-instant own-period filings
# ---------------------------------------------------------------------------


def _same_instant_q3_timeline(second_q3_ytd: float) -> list[MuFiling]:
    second = _filing_variant(q3_fy25(), accession="Q3-B", revenue={"YTD": second_q3_ytd})
    return [q1_fy25(), q2_fy25(), q3_fy25(), second, k_fy25()]


def _assert_order_independent(timeline: list[MuFiling], seed: int) -> None:
    baseline = _run(timeline, strict=False)
    rng = random.Random(seed)

    for _ in range(10):
        shuffled = timeline[:]
        rng.shuffle(shuffled)
        result = _run(shuffled, strict=False)

        pd.testing.assert_frame_equal(result.frame, baseline.frame)
        pd.testing.assert_frame_equal(result.reconciliation, baseline.reconciliation)
        pd.testing.assert_frame_equal(
            _sorted_diagnostics(result.diagnostics),
            _sorted_diagnostics(baseline.diagnostics),
        )


def test_same_instant_own_filings_with_different_values_are_ambiguous():
    timeline = _same_instant_q3_timeline(25100.0)
    result = _run(timeline, strict=False)
    q3_time = _t(q3_fy25())

    q3_versions = _versions(result, Q3_END)
    assert [(anchor, at) for anchor, at, _ in q3_versions] == [("Q3", q3_time)]
    assert pd.isna(q3_versions[0][2])

    lineage = _lineage(result, "Q3", "revenue", at=q3_time)
    assert lineage["method"] == "unavailable"
    assert lineage["reason"] == "ambiguous_current_observation"
    assert pd.isna(lineage["value"])
    assert pd.isna(lineage["source_accession_number"])
    assert lineage["version_trigger_accessions"] == ("Q3", "Q3-B")

    # Q3-B omits OCF, so OCF is not ambiguous: Q3 supplies it.
    ocf = _lineage(result, "Q3", "operating_cash_flow", at=q3_time)
    assert ocf["value"] == pytest.approx(3500.0)
    assert ocf["source_accession_number"] == "Q3"

    _assert_prefix_stable(timeline)
    _assert_order_independent(timeline, seed=31)


def test_same_instant_own_filings_with_equal_values_use_smallest_accession():
    timeline = _same_instant_q3_timeline(25000.0)
    result = _run(timeline, strict=False)
    q3_time = _t(q3_fy25())

    assert _versions(result, Q3_END) == [("Q3", q3_time, 9000.0)]

    lineage = _lineage(result, "Q3", "revenue", at=q3_time)
    assert lineage["method"] == "ytd_minus_prior_ytd"
    assert lineage["reason"] == "ok"
    assert lineage["source_accession_number"] == "Q3"
    assert lineage["version_trigger_accessions"] == ("Q3", "Q3-B")

    _assert_prefix_stable(timeline)
    _assert_order_independent(timeline, seed=37)
