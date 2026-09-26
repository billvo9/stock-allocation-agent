from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EdgarConceptSpec:
    canonical_name: str
    standard_concepts: tuple[str, ...]
    provider_concepts: tuple[str, ...]


EDGAR_CONCEPT_SPECS = {
    "revenue": EdgarConceptSpec(
        canonical_name="revenue",
        standard_concepts=("Revenue",),
        provider_concepts=(
            "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap_SalesRevenueNet",
        ),
    ),
    "gross_profit": EdgarConceptSpec(
        canonical_name="gross_profit",
        standard_concepts=("GrossProfit",),
        provider_concepts=("us-gaap_GrossProfit",),
    ),
    "operating_income": EdgarConceptSpec(
        canonical_name="operating_income",
        standard_concepts=("OperatingIncomeLoss",),
        provider_concepts=("us-gaap_OperatingIncomeLoss",),
    ),
    "net_income": EdgarConceptSpec(
        canonical_name="net_income",
        standard_concepts=("NetIncome",),
        provider_concepts=(
            "us-gaap_NetIncomeLoss",
            "us-gaap_ProfitLoss",
        ),
    ),
    "operating_cash_flow": EdgarConceptSpec(
        canonical_name="operating_cash_flow",
        standard_concepts=("NetCashFromOperatingActivities",),
        provider_concepts=("us-gaap_NetCashProvidedByUsedInOperatingActivities",),
    ),
}


def _top_level_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Exclude dimensional breakdown rows.

    Company-wide fundamentals should come from the
    consolidated/top-level statement rather than a
    product, geography, segment, or other dimension.
    """

    result = frame.copy()

    if "dimension" in result.columns:
        result = result[~result["dimension"].fillna(False).astype(bool)]

    return result


def resolve_statement_concept(
    frame: pd.DataFrame,
    canonical_name: str,
) -> pd.Series | None:
    """
    Resolve one canonical project metric to one
    top-level EDGAR statement row.

    Resolution priority:
        1. EdgarTools standard_concept
        2. Explicit SEC/XBRL concept aliases

    Returns None when no supported concept exists.
    """

    if canonical_name not in EDGAR_CONCEPT_SPECS:
        raise KeyError(f"Unsupported EDGAR concept: {canonical_name}")

    spec = EDGAR_CONCEPT_SPECS[canonical_name]

    candidates = _top_level_rows(frame)

    if "standard_concept" in candidates.columns:
        standard_matches = candidates[candidates["standard_concept"].isin(spec.standard_concepts)]

        if len(standard_matches) == 1:
            return standard_matches.iloc[0]

        if len(standard_matches) > 1:
            raise ValueError(
                f"Multiple top-level EDGAR rows matched standard concept for {canonical_name}."
            )

    if "concept" in candidates.columns:
        provider_matches = candidates[candidates["concept"].isin(spec.provider_concepts)]

        if len(provider_matches) == 1:
            return provider_matches.iloc[0]

        if len(provider_matches) > 1:
            raise ValueError(
                f"Multiple top-level EDGAR rows matched provider concept for {canonical_name}."
            )

    return None
