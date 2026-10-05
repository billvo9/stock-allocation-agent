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


def _row(result: EdgarHistoryResult, accession: str) -> pd.Series:
    rows = result.frame[result.frame["sec_accession_number"].eq(accession)]
    assert len(rows) == 1
    return rows.iloc[0]


def _lineage(
    result: EdgarHistoryResult,
    accession: str,
    metric_name: str = "revenue",
) -> pd.Series:
    rows = result.reconciliation[
        result.reconciliation["sec_accession_number"].eq(accession)
        & result.reconciliation["metric_name"].eq(metric_name)
    ]
    assert len(rows) == 1
    return rows.iloc[0]


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

    assert _row(result, "Q3")["revenue"] == pytest.approx(9000.0)
    assert _lineage(result, "Q3")["prior_accession_number"] == "Q2"

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

    # Q3 keeps the value that was knowable at the Q3 acceptance time:
    # 25000 - 16000 (original Q2), not 25000 - 15800 (later amendment).
    assert _row(full, "Q3")["revenue"] == pytest.approx(9000.0)
    assert _lineage(full, "Q3")["prior_accession_number"] == "Q2"

    # The late Q2 amendment did not rewrite earlier state: exactly one Q3
    # row exists for the original Q3 filing, still available at Q3 time.
    q3_rows = full.frame[full.frame["sec_accession_number"].eq("Q3")]
    assert len(q3_rows) == 1
    assert pd.Timestamp(q3_rows.iloc[0]["available_at"]) == q3_time

    # The amendment appears as additional, later-known information with
    # its own acceptance timestamp, never backdated - and so does the
    # diagnostic it causes.
    amendment = _row(full, "Q2A-LATE")
    assert pd.Timestamp(amendment["available_at"]) == q2a_late_time
    assert pd.Timestamp(amendment["available_at"]) > q3_time
    assert "Q2A-LATE" not in set(_known_by(full.frame, q3_time)["sec_accession_number"])
    assert (
        _diagnostics_known_by(full.diagnostics, q3_time, knowable_at)["reason"]
        .ne("derived_quarter_input_superseded")
        .all()
    )

    # Later knowledge is genuinely added, not merged into earlier rows;
    # only the broken filing is absent.
    assert len(full.frame) == len(timeline) - 1
    assert "BROKEN" not in set(full.frame["sec_accession_number"])


def test_p2_current_amendment_creates_new_later_version():
    without = _run([q2_fy25(), q3_fy25(), q2a_fy25_late()])
    with_amendment = _run([q2_fy25(), q3_fy25(), q2a_fy25_late(), q3a_fy25()])

    original_before = _row(without, "Q3")
    original_after = _row(with_amendment, "Q3")

    pd.testing.assert_series_equal(original_before, original_after)

    amended = _row(with_amendment, "Q3A")
    assert pd.Timestamp(amended["available_at"]) > pd.Timestamp(original_after["available_at"])

    # Q3A runs its own selection at its own available_at: by then the
    # late Q2 amendment is public, so Q3A = 25500 - 15800.
    assert amended["revenue"] == pytest.approx(9700.0)
    assert _lineage(with_amendment, "Q3A")["prior_accession_number"] == "Q2A-LATE"


def test_p3_late_prior_amendment_does_not_rederive_current_quarter():
    result = _run([q2_fy25(), q3_fy25(), q2a_fy25_late()])

    # Documents the accepted limitation: the history is prefix-stable but
    # not as-of complete. After Q2A-LATE is public, Q3 is still 9000
    # (not 9200) and no re-derived Q3 version is emitted.
    q3_period = result.frame[
        pd.to_datetime(result.frame["period_end"], utc=True).eq(
            pd.Timestamp("2025-05-29", tz="UTC")
        )
    ]
    assert q3_period["sec_accession_number"].tolist() == ["Q3"]
    assert q3_period.iloc[0]["revenue"] == pytest.approx(9000.0)


def test_p4_no_duplicate_symbol_available_at():
    result = _run([*_full_timeline(), q2a_fy25_early(), q1_fy26(), q2_fy26()])

    assert not result.frame.duplicated(subset=["symbol", "available_at"]).any()


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


def test_l1_lineage_key_is_symbol_accession_metric():
    lineage = _lineage_fixture().reconciliation

    assert not lineage.duplicated(subset=["symbol", "sec_accession_number", "metric_name"]).any()


def test_l2_every_canonical_flow_value_has_matching_lineage():
    result = _lineage_fixture()

    lineage = result.reconciliation.set_index(["symbol", "sec_accession_number", "metric_name"])

    assert len(lineage) == len(result.frame) * len(FLOW_METRICS)

    for _, row in result.frame.iterrows():
        for metric_name in FLOW_METRICS:
            entry = lineage.loc[(row["symbol"], row["sec_accession_number"], metric_name)]
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


def test_history_reason_prior_available_after_current():
    result = _run([q3_fy25(), q2a_fy25_late()])

    lineage = _lineage(result, "Q3")
    assert lineage["method"] == "unavailable"
    assert lineage["reason"] == "prior_available_after_current"
    assert pd.isna(lineage["prior_accession_number"])


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

    canonical = result.frame.set_index("sec_accession_number")["available_at"]
    assert {accession: pd.Timestamp(value) for accession, value in canonical.items()} == expected

    for _, row in result.reconciliation.iterrows():
        assert row["available_at"] == canonical[row["sec_accession_number"]]


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

    assert _lineage(result, "Q3")["reason"] == "prior_available_after_current"


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
