from __future__ import annotations

import itertools

import pandas as pd
import pytest

from stock_agent.data.fundamentals.edgar_reconciliation import (
    CandidateReconciliation,
    EdgarReconciliationError,
    FiscalFlowObservation,
    PriorSelection,
    find_superseding_prior_observations,
    is_valid_prior_fiscal_observation,
    reconcile_fiscal_flow,
    reconcile_fiscal_flow_from_candidates,
    select_prior_fiscal_observation,
)


def _observation(
    *,
    symbol: str = "MU",
    metric_name: str = "operating_cash_flow",
    fiscal_year: int = 2026,
    fiscal_quarter: int = 3,
    period_end: str = "2026-05-28",
    available_at: str = "2026-06-24 22:59:46",
    accession_number: str = "Q3",
    direct_value: float | None = None,
    ytd_value: float | None = None,
    fy_value: float | None = None,
) -> FiscalFlowObservation:
    return FiscalFlowObservation(
        symbol=symbol,
        metric_name=metric_name,
        fiscal_year=fiscal_year,
        fiscal_quarter=fiscal_quarter,
        period_end=pd.Timestamp(
            period_end,
            tz="UTC",
        ),
        available_at=pd.Timestamp(
            available_at,
            tz="UTC",
        ),
        accession_number=accession_number,
        direct_value=direct_value,
        ytd_value=ytd_value,
        fy_value=fy_value,
    )


def test_direct_quarter_value_wins():
    current = _observation(
        direct_value=25.0,
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
    )

    assert result.value == pytest.approx(25.0)
    assert result.method == "direct_quarter"
    assert result.reason == "ok"
    assert result.current_accession_number == "Q3"
    assert result.prior_accession_number is None


def test_q1_uses_current_ytd():
    current = _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        ytd_value=20.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
    )

    assert result.value == pytest.approx(20.0)
    assert result.method == "q1_ytd"
    assert result.reason == "ok"


def test_q2_derives_from_prior_q1_ytd():
    prior = _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        ytd_value=20.0,
    )

    current = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value == pytest.approx(25.0)
    assert result.method == "ytd_minus_prior_ytd"
    assert result.reason == "ok"
    assert result.current_accession_number == "Q2"
    assert result.prior_accession_number == "Q1"


def test_q3_derives_from_prior_q2_ytd():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value == pytest.approx(25.0)
    assert result.method == "ytd_minus_prior_ytd"
    assert result.reason == "ok"
    assert result.current_accession_number == "Q3"
    assert result.prior_accession_number == "Q2"


def test_q4_derives_from_fy_minus_q3_ytd():
    prior = _observation(
        fiscal_year=2025,
        fiscal_quarter=3,
        period_end="2025-05-29",
        available_at="2025-06-25 22:50:42",
        accession_number="FY2025-Q3",
        ytd_value=72.0,
    )

    current = _observation(
        fiscal_year=2025,
        fiscal_quarter=4,
        period_end="2025-08-28",
        available_at="2025-10-03 18:42:25",
        accession_number="FY2025-10K",
        fy_value=100.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value == pytest.approx(28.0)
    assert result.method == "fy_minus_q3_ytd"
    assert result.reason == "ok"
    assert result.current_accession_number == "FY2025-10K"
    assert result.prior_accession_number == "FY2025-Q3"


def test_prior_from_different_fiscal_year_is_rejected():
    prior = _observation(
        fiscal_year=2025,
        fiscal_quarter=2,
        period_end="2025-02-27",
        available_at="2025-03-20 23:20:23",
        accession_number="FY2025-Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_year=2026,
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="FY2026-Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "different_fiscal_year"


def test_wrong_prior_fiscal_quarter_is_rejected():
    prior = _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        ytd_value=20.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "wrong_prior_fiscal_quarter"


def test_future_prior_observation_is_rejected():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-07-01 12:00:00",
        accession_number="Q2-AMENDMENT",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "prior_available_after_current"


def test_prior_period_end_must_be_before_current_period_end():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-06-30",
        available_at="2026-06-20 12:00:00",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "prior_period_not_before_current"


def test_missing_prior_ytd_returns_missing():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=None,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "missing_prior_ytd"


def test_missing_current_ytd_returns_missing():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=None,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "missing_current_ytd"


def test_missing_fy_for_q4_returns_missing():
    prior = _observation(
        fiscal_year=2025,
        fiscal_quarter=3,
        period_end="2025-05-29",
        available_at="2025-06-25 22:50:42",
        accession_number="Q3",
        ytd_value=72.0,
    )

    current = _observation(
        fiscal_year=2025,
        fiscal_quarter=4,
        period_end="2025-08-28",
        available_at="2025-10-03 18:42:25",
        accession_number="10K",
        fy_value=None,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "missing_current_fy"


def test_different_symbol_is_rejected():
    prior = _observation(
        symbol="NVDA",
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="NVDA-Q2",
        ytd_value=45.0,
    )

    current = _observation(
        symbol="MU",
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="MU-Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "different_symbol"


def test_different_metric_is_rejected():
    prior = _observation(
        metric_name="revenue",
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        metric_name="operating_cash_flow",
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "different_metric"


def test_same_accession_cannot_be_used_as_prior_filing():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="SAME",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="SAME",
        ytd_value=70.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=prior,
    )

    assert result.value is None
    assert result.reason == "same_accession_number"


def test_valid_prior_helper_returns_true_for_same_fiscal_sequence():
    prior = _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )

    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at="2026-06-24 22:59:46",
        accession_number="Q3",
        ytd_value=70.0,
    )

    assert is_valid_prior_fiscal_observation(
        current=current,
        prior=prior,
    )


def test_invalid_fiscal_quarter_raises():
    current = _observation(
        fiscal_quarter=5,
    )

    with pytest.raises(
        EdgarReconciliationError,
        match="fiscal_quarter must be between 1 and 4",
    ):
        reconcile_fiscal_flow(
            current=current,
        )


# ---------------------------------------------------------------------------
# Prior selection (S1-S10)
#
# Timeline used below (MU FY2026, illustrative values):
#   Q2 10-Q    accepted 2026-03-18, Q2 YTD 45
#   Q2 10-Q/A  accepted 2026-04-15 (early) or 2026-07-10 (late), Q2 YTD 44
#   Q3 10-Q    accepted 2026-06-24, Q3 YTD 70   <- current
# ---------------------------------------------------------------------------

Q2_ORIGINAL_AVAILABLE_AT = "2026-03-18 23:00:06"
Q2_EARLY_AMENDMENT_AVAILABLE_AT = "2026-04-15 21:00:00"
Q3_AVAILABLE_AT = "2026-06-24 22:59:46"
Q2_LATE_AMENDMENT_AVAILABLE_AT = "2026-07-10 21:00:00"


def _q3_current() -> FiscalFlowObservation:
    return _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at=Q3_AVAILABLE_AT,
        accession_number="Q3",
        ytd_value=70.0,
    )


def _q2_prior(
    *,
    accession_number: str = "Q2",
    available_at: str = Q2_ORIGINAL_AVAILABLE_AT,
    ytd_value: float | None = 45.0,
    **overrides: object,
) -> FiscalFlowObservation:
    values: dict[str, object] = {
        "fiscal_quarter": 2,
        "period_end": "2026-02-26",
        "available_at": available_at,
        "accession_number": accession_number,
        "ytd_value": ytd_value,
    }
    values.update(overrides)

    return _observation(**values)


def test_s1_amendment_available_before_current_is_selected():
    original = _q2_prior()
    amendment = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
        ytd_value=44.0,
    )

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[original, amendment],
    )

    assert isinstance(selection, PriorSelection)
    assert selection.reason == "ok"
    assert selection.prior == amendment
    assert selection.skipped_prior_accessions == ()
    assert selection.equivalent_prior_accessions == ("Q2A",)


def test_s2_amendment_available_after_current_is_not_eligible():
    original = _q2_prior()
    late_amendment = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
        ytd_value=44.0,
    )

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[original, late_amendment],
    )

    assert selection.reason == "ok"
    assert selection.prior == original
    assert "Q2A" not in selection.skipped_prior_accessions
    assert "Q2A" not in selection.equivalent_prior_accessions


def test_s3_only_candidate_available_after_current_is_rejected():
    late_amendment = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
    )

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[late_amendment],
    )

    assert selection.prior is None
    assert selection.reason == "prior_available_after_current"


def test_s4_no_candidates_is_missing_prior():
    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[],
    )

    assert selection.prior is None
    assert selection.reason == "missing_prior_fiscal_observation"


def test_s5_later_amendment_without_value_does_not_supersede_original():
    original = _q2_prior()
    valueless_amendment = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
        ytd_value=None,
    )

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[original, valueless_amendment],
    )

    assert selection.reason == "ok"
    assert selection.prior == original
    assert selection.skipped_prior_accessions == ("Q2A",)


def test_s6_equal_values_at_same_timestamp_use_smallest_accession():
    first = _q2_prior(accession_number="0000723125-26-000020")
    second = _q2_prior(accession_number="0000723125-26-000010")

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[first, second],
    )

    assert selection.reason == "ok"
    assert selection.prior == second
    assert selection.equivalent_prior_accessions == (
        "0000723125-26-000010",
        "0000723125-26-000020",
    )


def test_s7_different_values_at_same_timestamp_are_ambiguous():
    first = _q2_prior(accession_number="Q2-B", ytd_value=45.0)
    second = _q2_prior(accession_number="Q2-A", ytd_value=44.0)

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[first, second],
    )

    assert selection.prior is None
    assert selection.reason == "ambiguous_prior_observation"
    assert selection.conflicting_prior_accessions == ("Q2-A", "Q2-B")


def test_s8_ineligible_candidates_are_never_selected():
    valid = _q2_prior()

    # Every ineligible candidate is more recent than the valid one, so a
    # rank-then-validate implementation would pick one of them.
    recent = Q2_EARLY_AMENDMENT_AVAILABLE_AT
    ineligible = [
        _q2_prior(accession_number="WRONG-FY", available_at=recent, fiscal_year=2025),
        _q2_prior(accession_number="WRONG-Q", available_at=recent, fiscal_quarter=1),
        _q2_prior(accession_number="Q3", available_at=recent),
        _q2_prior(accession_number="LATER-END", available_at=recent, period_end="2026-05-28"),
        _q2_prior(accession_number="OTHER-SYM", available_at=recent, symbol="AAPL"),
        _q2_prior(accession_number="OTHER-METRIC", available_at=recent, metric_name="revenue"),
    ]

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[*ineligible, valid],
    )

    assert selection.reason == "ok"
    assert selection.prior == valid


@pytest.mark.parametrize(
    "candidates",
    [
        pytest.param(
            [
                _q2_prior(),
                _q2_prior(
                    accession_number="Q2A",
                    available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
                    ytd_value=44.0,
                ),
                _q2_prior(
                    accession_number="Q2A-LATE",
                    available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
                    ytd_value=43.0,
                ),
            ],
            id="s1-amendments",
        ),
        pytest.param(
            [
                _q2_prior(),
                _q2_prior(
                    accession_number="Q2A",
                    available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
                    ytd_value=None,
                ),
                _q2_prior(
                    accession_number="Q2B",
                    available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
                    ytd_value=None,
                ),
            ],
            id="s5-valueless",
        ),
        pytest.param(
            [
                _q2_prior(accession_number="C"),
                _q2_prior(accession_number="A"),
                _q2_prior(accession_number="B"),
            ],
            id="s6-ties",
        ),
        pytest.param(
            [
                _q2_prior(accession_number="C", ytd_value=45.0),
                _q2_prior(accession_number="A", ytd_value=44.0),
                _q2_prior(accession_number="B", ytd_value=45.0),
            ],
            id="s7-conflict",
        ),
    ],
)
def test_s9_selection_is_independent_of_candidate_order(candidates):
    current = _q3_current()

    selections = {
        select_prior_fiscal_observation(
            current=current,
            candidates=list(permutation),
        )
        for permutation in itertools.permutations(candidates)
    }

    assert len(selections) == 1


def test_s10_eligible_candidates_without_values_are_missing_prior_ytd():
    first = _q2_prior(accession_number="Q2B", ytd_value=None)
    second = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
        ytd_value=None,
    )

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[first, second],
    )

    assert selection.prior is None
    assert selection.reason == "missing_prior_ytd"
    assert selection.skipped_prior_accessions == ("Q2A", "Q2B")


def test_selection_for_fiscal_q1_expects_no_prior():
    current = _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        ytd_value=20.0,
    )

    selection = select_prior_fiscal_observation(
        current=current,
        candidates=[_q2_prior()],
    )

    assert selection.prior is None
    assert selection.reason == "no_prior_expected"


def test_selected_prior_feeds_reconciliation():
    current = _q3_current()
    selection = select_prior_fiscal_observation(
        current=current,
        candidates=[
            _q2_prior(),
            _q2_prior(
                accession_number="Q2A",
                available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
                ytd_value=44.0,
            ),
        ],
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=selection.prior,
    )

    assert result.value == pytest.approx(26.0)
    assert result.prior_accession_number == "Q2A"


def test_duplicate_candidate_accessions_raise():
    with pytest.raises(EdgarReconciliationError):
        select_prior_fiscal_observation(
            current=_q3_current(),
            candidates=[_q2_prior(), _q2_prior(ytd_value=44.0)],
        )


# ---------------------------------------------------------------------------
# Fiscal-Q1 prior fallback (Q1a-Q1e)
#
# Standalone fiscal Q1 and Q1 YTD cover the same duration (fiscal-year
# start to Q1 end), so a Q1 prior's direct value may stand in for its
# missing YTD value. No other quarter has that equivalence.
# ---------------------------------------------------------------------------


def _q1_prior(
    *,
    direct_value: float | None = None,
    ytd_value: float | None = None,
) -> FiscalFlowObservation:
    return _observation(
        fiscal_quarter=1,
        period_end="2025-11-27",
        available_at="2025-12-17 23:47:44",
        accession_number="Q1",
        direct_value=direct_value,
        ytd_value=ytd_value,
    )


def _q2_current() -> FiscalFlowObservation:
    return _observation(
        fiscal_quarter=2,
        period_end="2026-02-26",
        available_at="2026-03-18 23:00:06",
        accession_number="Q2",
        ytd_value=45.0,
    )


def test_q1a_q2_uses_q1_prior_direct_value_when_ytd_is_absent():
    result = reconcile_fiscal_flow(
        current=_q2_current(),
        prior=_q1_prior(direct_value=20.0),
    )

    assert result.value == pytest.approx(25.0)
    assert result.method == "ytd_minus_prior_ytd"
    assert result.reason == "ok"
    assert result.prior_accession_number == "Q1"
    assert result.prior_value_source == "q1_direct"


def test_q1a_selector_accepts_q1_prior_with_direct_value_only():
    prior = _q1_prior(direct_value=20.0)

    selection = select_prior_fiscal_observation(
        current=_q2_current(),
        candidates=[prior],
    )

    assert selection.reason == "ok"
    assert selection.prior == prior


def test_q1b_q1_prior_with_equal_direct_and_ytd_prefers_ytd():
    result = reconcile_fiscal_flow(
        current=_q2_current(),
        prior=_q1_prior(direct_value=20.0, ytd_value=20.0),
    )

    assert result.value == pytest.approx(25.0)
    assert result.reason == "ok"
    assert result.prior_value_source == "ytd"


def test_q1c_q1_prior_with_conflicting_direct_and_ytd_is_unavailable():
    result = reconcile_fiscal_flow(
        current=_q2_current(),
        prior=_q1_prior(direct_value=21.0, ytd_value=20.0),
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "prior_q1_direct_ytd_conflict"
    assert result.prior_accession_number == "Q1"
    assert result.prior_value_source is None


def test_q1c_selector_reports_conflict_when_it_is_the_only_cause():
    selection = select_prior_fiscal_observation(
        current=_q2_current(),
        candidates=[_q1_prior(direct_value=21.0, ytd_value=20.0)],
    )

    assert selection.prior is None
    assert selection.reason == "prior_q1_direct_ytd_conflict"
    assert selection.skipped_prior_accessions == ("Q1",)


def test_q1d_q2_prior_direct_value_never_substitutes_for_q2_ytd():
    q2_prior = _q2_prior(direct_value=25.0, ytd_value=None)

    result = reconcile_fiscal_flow(
        current=_q3_current(),
        prior=q2_prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "missing_prior_ytd"

    selection = select_prior_fiscal_observation(
        current=_q3_current(),
        candidates=[q2_prior],
    )

    assert selection.prior is None
    assert selection.reason == "missing_prior_ytd"


def test_q1e_q3_prior_direct_value_never_substitutes_for_q3_ytd():
    current = _observation(
        fiscal_quarter=4,
        period_end="2026-08-27",
        available_at="2026-10-02 21:00:00",
        accession_number="FY",
        fy_value=100.0,
    )
    q3_prior = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at=Q3_AVAILABLE_AT,
        accession_number="Q3",
        direct_value=25.0,
    )

    result = reconcile_fiscal_flow(
        current=current,
        prior=q3_prior,
    )

    assert result.value is None
    assert result.method == "unavailable"
    assert result.reason == "missing_prior_ytd"

    selection = select_prior_fiscal_observation(
        current=current,
        candidates=[q3_prior],
    )

    assert selection.prior is None
    assert selection.reason == "missing_prior_ytd"


# ---------------------------------------------------------------------------
# Superseding priors (D2 diagnostics)
# ---------------------------------------------------------------------------


def test_late_amendment_with_different_value_supersedes_used_prior():
    used = _q2_prior()
    late = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
        ytd_value=44.0,
    )

    superseding = find_superseding_prior_observations(
        current=_q3_current(),
        prior=used,
        candidates=[used, late],
    )

    assert superseding == (late,)


def test_late_amendment_with_same_or_missing_value_does_not_supersede():
    used = _q2_prior()
    same_value = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
        ytd_value=45.0,
    )
    valueless = _q2_prior(
        accession_number="Q2B",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
        ytd_value=None,
    )

    superseding = find_superseding_prior_observations(
        current=_q3_current(),
        prior=used,
        candidates=[used, same_value, valueless],
    )

    assert superseding == ()


def test_prior_available_before_current_is_never_superseding():
    used = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
        ytd_value=44.0,
    )
    original = _q2_prior()

    superseding = find_superseding_prior_observations(
        current=_q3_current(),
        prior=used,
        candidates=[original, used],
    )

    assert superseding == ()


def test_superseding_priors_are_sorted_by_availability_then_accession():
    used = _q2_prior()
    later = _q2_prior(
        accession_number="A-LATER",
        available_at="2026-08-01 21:00:00",
        ytd_value=43.0,
    )
    late_b = _q2_prior(
        accession_number="B-LATE",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
        ytd_value=44.0,
    )
    late_a = _q2_prior(
        accession_number="A-LATE",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
        ytd_value=44.5,
    )

    superseding = find_superseding_prior_observations(
        current=_q3_current(),
        prior=used,
        candidates=[later, late_b, used, late_a],
    )

    assert [observation.accession_number for observation in superseding] == [
        "A-LATE",
        "B-LATE",
        "A-LATER",
    ]


# ---------------------------------------------------------------------------
# Reason ordering: the current filing's own inputs are checked first, and a
# prior is selected only when subtraction is actually required.
# ---------------------------------------------------------------------------


def test_missing_current_ytd_is_reported_before_missing_prior():
    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at=Q3_AVAILABLE_AT,
        accession_number="Q3",
        ytd_value=None,
        fy_value=99.0,
    )

    result = reconcile_fiscal_flow(current=current)

    assert result.method == "unavailable"
    assert result.reason == "missing_current_ytd"


def test_missing_current_fy_is_reported_before_missing_prior():
    current = _observation(
        fiscal_quarter=4,
        period_end="2026-08-27",
        available_at="2026-10-02 21:00:00",
        accession_number="FY",
        ytd_value=100.0,
    )

    result = reconcile_fiscal_flow(current=current)

    assert result.reason == "missing_current_fy"


def _from_candidates(current, candidates):
    outcome = reconcile_fiscal_flow_from_candidates(
        current=current,
        candidates=candidates,
    )
    assert isinstance(outcome, CandidateReconciliation)
    return outcome


def test_candidates_not_consulted_for_direct_value():
    current = _observation(direct_value=25.0, ytd_value=70.0, accession_number="Q3")

    # Duplicate accessions would make selection raise; it must not run.
    outcome = _from_candidates(current, [_q2_prior(), _q2_prior(ytd_value=44.0)])

    assert outcome.result.method == "direct_quarter"
    assert outcome.result.value == pytest.approx(25.0)
    assert outcome.selection is None


def test_candidates_not_consulted_for_fiscal_q1():
    current = _q1_prior(ytd_value=20.0)

    outcome = _from_candidates(current, [_q2_prior(), _q2_prior(ytd_value=44.0)])

    assert outcome.result.method == "q1_ytd"
    assert outcome.selection is None


def test_candidates_not_consulted_when_current_input_missing():
    current = _observation(
        fiscal_quarter=3,
        period_end="2026-05-28",
        available_at=Q3_AVAILABLE_AT,
        accession_number="Q3",
        fy_value=99.0,
    )
    late = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
    )

    outcome = _from_candidates(current, [late])

    assert outcome.result.reason == "missing_current_ytd"
    assert outcome.selection is None


def test_no_reported_value_is_its_own_reason():
    current = _observation(fiscal_quarter=3, accession_number="Q3")

    outcome = _from_candidates(current, [_q2_prior()])

    assert outcome.result.method == "unavailable"
    assert outcome.result.reason == "no_reported_value"
    assert outcome.selection is None


def test_selection_reason_is_the_unavailable_reason():
    late = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_LATE_AMENDMENT_AVAILABLE_AT,
    )

    outcome = _from_candidates(_q3_current(), [late])

    assert outcome.result.method == "unavailable"
    assert outcome.result.reason == "prior_available_after_current"
    assert outcome.result.prior_accession_number is None
    assert outcome.selection is not None
    assert outcome.selection.reason == "prior_available_after_current"


def test_selected_prior_is_used_for_subtraction():
    amendment = _q2_prior(
        accession_number="Q2A",
        available_at=Q2_EARLY_AMENDMENT_AVAILABLE_AT,
        ytd_value=44.0,
    )

    outcome = _from_candidates(_q3_current(), [_q2_prior(), amendment])

    assert outcome.result.value == pytest.approx(26.0)
    assert outcome.result.prior_accession_number == "Q2A"
    assert outcome.selection.prior == amendment
