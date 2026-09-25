from __future__ import annotations

import pandas as pd
import pytest

from stock_agent.data.fundamentals.edgar_source import (
    EDGAR_FILING_COLUMNS,
    EDGAR_FINANCIAL_FORMS,
    list_financial_filings,
)


class FakeFilings:
    def __init__(
        self,
        frame: pd.DataFrame,
    ) -> None:
        self._frame = frame

    def to_pandas(self) -> pd.DataFrame:
        return self._frame.copy()


class FakeCompany:
    def __init__(
        self,
        frame: pd.DataFrame,
        calls: list[dict[str, object]],
    ) -> None:
        self._frame = frame
        self._calls = calls

    def get_filings(
        self,
        *,
        form: list[str],
    ) -> FakeFilings:
        self._calls.append(
            {
                "form": form,
            }
        )

        return FakeFilings(self._frame)


def _raw_filing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "form": "10-Q",
                "filing_date": "2026-06-25",
                "reportDate": "2026-05-28",
                "acceptanceDateTime": ("2026-06-24T22:59:46+00:00"),
                "accession_number": ("0000723125-26-000015"),
                "isXBRL": 1,
            },
            {
                "form": "10-Q",
                "filing_date": "2026-03-19",
                "reportDate": "2026-02-26",
                "acceptanceDateTime": ("2026-03-18T23:00:06+00:00"),
                "accession_number": ("0000723125-26-000006"),
                "isXBRL": 1,
            },
        ]
    )


def test_list_financial_filings_requests_expected_forms():
    raw = _raw_filing_frame()

    calls: list[dict[str, object]] = []

    def company_factory(
        symbol: str,
    ) -> FakeCompany:
        assert symbol == "MU"

        return FakeCompany(
            raw,
            calls,
        )

    list_financial_filings(
        provider_symbol="MU",
        company_factory=company_factory,
    )

    assert calls == [{"form": list(EDGAR_FINANCIAL_FORMS)}]


def test_list_financial_filings_normalizes_metadata():
    raw = _raw_filing_frame()

    def company_factory(
        symbol: str,
    ) -> FakeCompany:
        return FakeCompany(
            raw,
            [],
        )

    result = list_financial_filings(
        provider_symbol="MU",
        company_factory=company_factory,
    )

    assert result.columns.tolist() == (EDGAR_FILING_COLUMNS)

    assert len(result) == 2

    assert result["period_end"].tolist() == [
        pd.Timestamp(
            "2026-02-26",
            tz="UTC",
        ),
        pd.Timestamp(
            "2026-05-28",
            tz="UTC",
        ),
    ]

    assert result["filing_date"].tolist() == [
        pd.Timestamp(
            "2026-03-19",
            tz="UTC",
        ),
        pd.Timestamp(
            "2026-06-25",
            tz="UTC",
        ),
    ]


def test_list_financial_filings_preserves_acceptance_time():
    raw = _raw_filing_frame()

    def company_factory(
        symbol: str,
    ) -> FakeCompany:
        return FakeCompany(
            raw,
            [],
        )

    result = list_financial_filings(
        provider_symbol="MU",
        company_factory=company_factory,
    )

    assert result.loc[
        0,
        "accepted_at",
    ] == pd.Timestamp("2026-03-18T23:00:06+00:00")


def test_list_financial_filings_preserves_accession_number():
    raw = _raw_filing_frame()

    def company_factory(
        symbol: str,
    ) -> FakeCompany:
        return FakeCompany(
            raw,
            [],
        )

    result = list_financial_filings(
        provider_symbol="MU",
        company_factory=company_factory,
    )

    assert result["accession_number"].tolist() == [
        "0000723125-26-000006",
        "0000723125-26-000015",
    ]


def test_list_financial_filings_rejects_missing_columns():
    raw = _raw_filing_frame().drop(columns=["reportDate"])

    def company_factory(
        symbol: str,
    ) -> FakeCompany:
        return FakeCompany(
            raw,
            [],
        )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        list_financial_filings(
            provider_symbol="MU",
            company_factory=company_factory,
        )


def test_list_financial_filings_rejects_empty_symbol():
    with pytest.raises(
        ValueError,
        match="Provider symbol cannot be empty",
    ):
        list_financial_filings(
            provider_symbol="   ",
        )
