from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.edgar_concepts import (
    resolve_statement_concept,
)


def test_resolve_statement_concept_prefers_standard_concept():
    frame = pd.DataFrame(
        [
            {
                "concept": "company_CustomRevenue",
                "standard_concept": "Revenue",
                "dimension": False,
                "value": 100.0,
            }
        ]
    )

    result = resolve_statement_concept(
        frame,
        "revenue",
    )

    assert result is not None
    assert result["value"] == pytest.approx(100.0)


def test_resolve_statement_concept_falls_back_to_provider_concept():
    frame = pd.DataFrame(
        [
            {
                "concept": ("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"),
                "standard_concept": None,
                "dimension": False,
                "value": 100.0,
            }
        ]
    )

    result = resolve_statement_concept(
        frame,
        "revenue",
    )

    assert result is not None
    assert result["value"] == pytest.approx(100.0)


def test_resolve_statement_concept_ignores_dimension_breakdown():
    frame = pd.DataFrame(
        [
            {
                "concept": ("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"),
                "standard_concept": "Revenue",
                "dimension": True,
                "value": 40.0,
            },
            {
                "concept": ("us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax"),
                "standard_concept": "Revenue",
                "dimension": False,
                "value": 100.0,
            },
        ]
    )

    result = resolve_statement_concept(
        frame,
        "revenue",
    )

    assert result is not None
    assert result["value"] == pytest.approx(100.0)


def test_resolve_statement_concept_returns_none_when_unavailable():
    frame = pd.DataFrame(
        [
            {
                "concept": "company_OtherMetric",
                "standard_concept": None,
                "dimension": False,
            }
        ]
    )

    result = resolve_statement_concept(
        frame,
        "revenue",
    )

    assert result is None


def test_resolve_statement_concept_rejects_unknown_canonical_metric():
    frame = pd.DataFrame()

    with pytest.raises(
        KeyError,
        match="Unsupported EDGAR concept",
    ):
        resolve_statement_concept(
            frame,
            "mystery_metric",
        )


def test_resolve_statement_concept_rejects_ambiguous_top_level_rows():
    frame = pd.DataFrame(
        [
            {
                "concept": "company_RevenueOne",
                "standard_concept": "Revenue",
                "dimension": False,
            },
            {
                "concept": "company_RevenueTwo",
                "standard_concept": "Revenue",
                "dimension": False,
            },
        ]
    )

    with pytest.raises(
        ValueError,
        match="Multiple top-level EDGAR rows",
    ):
        resolve_statement_concept(
            frame,
            "revenue",
        )
