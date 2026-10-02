from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class EdgarConceptSpec:
    canonical_name: str
    standard_concepts: tuple[str, ...]
    provider_concepts: tuple[str, ...]
    preferred_raw_concepts: tuple[str, ...] = ()


EDGAR_CONCEPT_SPECS = {
    "revenue": EdgarConceptSpec(
        canonical_name="revenue",
        standard_concepts=("Revenue",),
        provider_concepts=(
            "us-gaap_RevenueFromContractWithCustomerExcludingAssessedTax",
            "us-gaap_SalesRevenueNet",
        ),
        preferred_raw_concepts=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
        ),
    ),
    "gross_profit": EdgarConceptSpec(
        canonical_name="gross_profit",
        standard_concepts=("GrossProfit",),
        provider_concepts=("us-gaap_GrossProfit",),
        preferred_raw_concepts=("GrossProfit",),
    ),
    "operating_income": EdgarConceptSpec(
        canonical_name="operating_income",
        standard_concepts=("OperatingIncomeLoss",),
        provider_concepts=("us-gaap_OperatingIncomeLoss",),
        preferred_raw_concepts=("OperatingIncomeLoss",),
    ),
    "net_income": EdgarConceptSpec(
        canonical_name="net_income",
        standard_concepts=("NetIncome",),
        provider_concepts=(
            "us-gaap_NetIncomeLoss",
            "us-gaap_ProfitLoss",
        ),
        preferred_raw_concepts=(
            "NetIncomeLoss",
            "ProfitLoss",
        ),
    ),
    "operating_cash_flow": EdgarConceptSpec(
        canonical_name="operating_cash_flow",
        standard_concepts=("NetCashFromOperatingActivities",),
        provider_concepts=("us-gaap_NetCashProvidedByUsedInOperatingActivities",),
        preferred_raw_concepts=("NetCashProvidedByUsedInOperatingActivities",),
    ),
}


def _matches_raw_concept(
    value: object,
    preferred_concepts: tuple[str, ...],
) -> bool:
    """
    Return True when an EDGAR/XBRL concept exactly matches
    one of our preferred SEC concepts.

    EdgarTools may expose concepts as, for example:

        us-gaap_NetCashProvidedByUsedInOperatingActivities

    while another representation may use:

        us-gaap:NetCashProvidedByUsedInOperatingActivities

    We compare the exact local concept name rather than
    doing a substring search.
    """

    if value is None:
        return False

    text = str(value).strip()

    if not text:
        return False

    normalized = text.replace(":", "_").casefold()

    for preferred in preferred_concepts:
        candidate = preferred.strip().casefold()

        if not candidate:
            continue

        if normalized == candidate:
            return True

        if normalized.endswith(f"_{candidate}"):
            return True

    return False


def _is_true(value: object) -> bool:
    if value is True:
        return True

    if value is None:
        return False

    return str(value).strip().casefold() in {
        "true",
        "1",
        "yes",
    }


def _top_level_rows(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prefer consolidated, non-abstract financial-statement rows.

    Dimensional or breakdown rows can contain valid accounting
    facts, but they represent segments/components rather than the
    consolidated metric used by the model.
    """

    result = frame.copy()

    for column in (
        "abstract",
        "dimension",
        "is_breakdown",
    ):
        if column in result.columns:
            result = result.loc[~result[column].map(_is_true)]

    return result


def resolve_statement_concept(
    frame: pd.DataFrame,
    canonical_name: str,
    *,
    symbol: str | None = None,
    accession_number: str | None = None,
) -> pd.Series | None:
    """
    Resolve one canonical project metric to one
    top-level EDGAR statement row.

    Resolution priority:
        1. Preferred exact SEC/XBRL raw concept
        2. EdgarTools standard_concept
        3. Explicit provider concept aliases

    Returns None when no supported concept exists.
    """

    if canonical_name not in EDGAR_CONCEPT_SPECS:
        raise KeyError(f"Unsupported EDGAR concept: {canonical_name}")

    spec = EDGAR_CONCEPT_SPECS[canonical_name]

    candidates = _top_level_rows(frame)

    preferred = candidates.iloc[0:0]

    if "concept" in candidates.columns:
        preferred = candidates.loc[
            candidates["concept"].map(
                lambda value: _matches_raw_concept(
                    value,
                    spec.preferred_raw_concepts,
                )
            )
        ]

    if len(preferred) == 1:
        return preferred.iloc[0]

    if len(preferred) > 1:
        symbol_context = symbol or "<unknown>"
        accession_context = accession_number or "<unknown>"

        raise ValueError(
            "Ambiguous EDGAR preferred concept resolution "
            f"for {canonical_name}; "
            f"symbol={symbol_context}, "
            f"accession={accession_context}. "
            "Multiple consolidated rows match the preferred "
            "SEC XBRL concept."
        )

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
