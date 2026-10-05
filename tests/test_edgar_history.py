from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import stock_agent.data.fundamentals.edgar_history as history_module
from stock_agent.data.fundamentals.edgar_history import (
    DIAGNOSTIC_COLUMNS,
    build_edgar_history,
)
from stock_agent.data.fundamentals.edgar_quarterly import (
    EdgarQuarterlyBuildError,
)
from stock_agent.data.fundamentals.quarterly_schema import (
    QUARTERLY_FUNDAMENTAL_COLUMNS,
)


class FakeStatement:
    def __init__(
        self,
        frame: pd.DataFrame,
    ) -> None:
        self._frame = frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()


class FakeStatements:
    def __init__(
        self,
        *,
        income: pd.DataFrame | None = None,
        balance: pd.DataFrame | None = None,
        cash_flow: pd.DataFrame | None = None,
    ) -> None:
        self._income = income
        self._balance = balance
        self._cash_flow = cash_flow

    def income_statement(self):
        if self._income is None:
            return None
        return FakeStatement(self._income)

    def balance_sheet(self):
        if self._balance is None:
            return None
        return FakeStatement(self._balance)

    def cash_flow_statement(self):
        if self._cash_flow is None:
            return None
        return FakeStatement(self._cash_flow)


class FakeXbrl:
    def __init__(
        self,
        statements: FakeStatements,
        *,
        reporting_periods: list[dict[str, Any]],
    ) -> None:
        self.statements = statements
        self.reporting_periods = reporting_periods


class FakeFiling:
    def __init__(
        self,
        xbrl: FakeXbrl | None,
    ) -> None:
        self._xbrl = xbrl

    def xbrl(self):
        return self._xbrl


def _q3_statement() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept": ["example"],
            "2026-05-28 (Q3)": [100.0],
        }
    )


def _filings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "form": "10-Q",
                "period_end": pd.Timestamp(
                    "2026-05-28",
                    tz="UTC",
                ),
                "filing_date": pd.Timestamp(
                    "2026-06-25",
                    tz="UTC",
                ),
                "accepted_at": pd.Timestamp(
                    "2026-06-25 22:00",
                    tz="UTC",
                ),
                "accession_number": "A1",
                "is_xbrl": True,
            }
        ]
    )


def _fake_xbrl(
    statements: FakeStatements,
    *,
    period_end: str,
    fiscal_year: int,
    fiscal_period: str,
) -> FakeXbrl:
    return FakeXbrl(
        statements,
        reporting_periods=_reporting_period(
            period_end=period_end,
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
        ),
    )


def _reporting_period(
    *,
    period_end: str,
    fiscal_year: int,
    fiscal_period: str,
) -> list[dict[str, Any]]:
    return [
        {
            "type": "duration",
            "end_date": period_end,
            "fiscal_year": fiscal_year,
            "fiscal_period": fiscal_period,
        }
    ]


@pytest.fixture
def fake_quarter_builder(monkeypatch):
    monkeypatch.setattr(
        history_module,
        "build_edgar_quarter",
        _fake_quarter_builder,
    )

    monkeypatch.setattr(
        history_module,
        "validate_quarterly_fundamental_frame",
        lambda frame: None,
    )


def test_build_history_from_multiple_filings(
    monkeypatch,
    fake_quarter_builder,
):
    filings = pd.DataFrame(
        [
            {
                "form": "10-Q",
                "period_end": pd.Timestamp(
                    "2026-02-26",
                    tz="UTC",
                ),
                "filing_date": pd.Timestamp(
                    "2026-03-19",
                    tz="UTC",
                ),
                "accepted_at": pd.Timestamp(
                    "2026-03-19 22:00",
                    tz="UTC",
                ),
                "accession_number": "A1",
                "is_xbrl": True,
            },
            {
                "form": "10-Q",
                "period_end": pd.Timestamp(
                    "2026-05-28",
                    tz="UTC",
                ),
                "filing_date": pd.Timestamp(
                    "2026-06-25",
                    tz="UTC",
                ),
                "accepted_at": pd.Timestamp(
                    "2026-06-25 22:00",
                    tz="UTC",
                ),
                "accession_number": "A2",
                "is_xbrl": True,
            },
        ]
    )

    filing_map = {
        "A1": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=pd.DataFrame(
                        {
                            "2026-02-26 (Q2)": [1.0],
                        }
                    )
                ),
                period_end="2026-02-26",
                fiscal_year=2026,
                fiscal_period="Q2",
            )
        ),
        "A2": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=pd.DataFrame(
                        {
                            "2026-05-28 (Q3)": [1.0],
                        }
                    )
                ),
                period_end="2026-05-28",
                fiscal_year=2026,
                fiscal_period="Q3",
            )
        ),
    }

    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda accession: filing_map[accession],
        clock=lambda: pd.Timestamp(
            "2026-09-29",
            tz="UTC",
        ),
    )

    assert len(result.frame) == 2

    assert result.diagnostics["status"].eq("success").all()


def _filing_row(
    *,
    accession_number: str,
    period_end: str,
    filing_date: str,
    accepted_at: str,
    form: str = "10-Q",
    is_xbrl: bool = True,
) -> dict[str, object]:
    return {
        "form": form,
        "period_end": pd.Timestamp(
            period_end,
            tz="UTC",
        ),
        "filing_date": pd.Timestamp(
            filing_date,
            tz="UTC",
        ),
        "accepted_at": pd.Timestamp(
            accepted_at,
            tz="UTC",
        ),
        "accession_number": accession_number,
        "is_xbrl": is_xbrl,
    }


def _quarter_statement(
    *,
    period_end: str,
    fiscal_quarter: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept": ["example"],
            (f"{period_end} (Q{fiscal_quarter})"): [1.0],
        }
    )


def _fake_quarter_builder(
    **kwargs: Any,
) -> pd.DataFrame:
    row: dict[str, Any] = {column: None for column in QUARTERLY_FUNDAMENTAL_COLUMNS}

    filing_date = pd.Timestamp(kwargs["filing_date"])

    accepted_at = kwargs["accepted_at"]

    if accepted_at is None:
        available_at = filing_date + pd.Timedelta(days=1)

        availability_source = "sec_filing_date_plus_1d"
    else:
        available_at = pd.Timestamp(accepted_at)

        availability_source = "sec_acceptance_datetime"

    row.update(
        {
            "symbol": kwargs["symbol"],
            "provider_symbol": kwargs["provider_symbol"],
            "period_end": pd.Timestamp(kwargs["period_end"]),
            "available_at": available_at,
            "retrieved_at": kwargs["retrieved_at"],
            "filing_date": filing_date,
            "sec_form_type": kwargs["sec_form_type"],
            "sec_accession_number": kwargs["sec_accession_number"],
            "availability_source": (availability_source),
            "currency": kwargs["currency"],
            "source": kwargs["source"],
        }
    )

    return pd.DataFrame(
        [row],
        columns=QUARTERLY_FUNDAMENTAL_COLUMNS,
    )


@pytest.fixture
def quarter_calls(
    monkeypatch,
) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake_builder(
        **kwargs: Any,
    ) -> pd.DataFrame:
        calls.append(dict(kwargs))

        return _fake_quarter_builder(**kwargs)

    monkeypatch.setattr(
        history_module,
        "build_edgar_quarter",
        fake_builder,
    )

    return calls


def test_history_is_sorted_by_available_at(
    quarter_calls,
):
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="A2",
                period_end="2026-05-28",
                filing_date="2026-06-25",
                accepted_at="2026-06-25 22:00",
            ),
            _filing_row(
                accession_number="A1",
                period_end="2026-02-26",
                filing_date="2026-03-19",
                accepted_at="2026-03-19 22:00",
            ),
        ]
    )

    filing_map = {
        "A1": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=_quarter_statement(
                        period_end="2026-02-26",
                        fiscal_quarter=2,
                    )
                ),
                period_end="2026-02-26",
                fiscal_year=2026,
                fiscal_period="Q2",
            )
        ),
        "A2": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=_quarter_statement(
                        period_end="2026-05-28",
                        fiscal_quarter=3,
                    )
                ),
                period_end="2026-05-28",
                fiscal_year=2026,
                fiscal_period="Q3",
            )
        ),
    }

    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda accession: filing_map[accession],
        clock=lambda: pd.Timestamp(
            "2026-09-29",
            tz="UTC",
        ),
    )

    assert result.frame["available_at"].is_monotonic_increasing

    assert result.frame["sec_accession_number"].tolist() == [
        "A1",
        "A2",
    ]


def test_history_preserves_accession_numbers(
    quarter_calls,
):
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="0001",
                period_end="2026-05-28",
                filing_date="2026-06-25",
                accepted_at="2026-06-25 22:00",
            )
        ]
    )

    filing = FakeFiling(
        _fake_xbrl(
            FakeStatements(
                income=_quarter_statement(
                    period_end="2026-05-28",
                    fiscal_quarter=3,
                )
            ),
            period_end="2026-05-28",
            fiscal_year=2026,
            fiscal_period="Q3",
        )
    )

    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda _: filing,
        clock=lambda: pd.Timestamp(
            "2026-09-29",
            tz="UTC",
        ),
    )

    assert result.frame["sec_accession_number"].tolist() == ["0001"]

    assert result.diagnostics["accession_number"].tolist() == ["0001"]


def test_history_preserves_original_and_amended_filing(
    quarter_calls,
):
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="ORIGINAL",
                period_end="2026-05-28",
                filing_date="2026-06-25",
                accepted_at="2026-06-25 22:00",
            ),
            _filing_row(
                accession_number="AMENDMENT",
                period_end="2026-05-28",
                filing_date="2026-07-10",
                accepted_at="2026-07-10 18:00",
                form="10-Q/A",
            ),
        ]
    )

    statement = _quarter_statement(
        period_end="2026-05-28",
        fiscal_quarter=3,
    )

    filing_map = {
        "ORIGINAL": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=statement,
                ),
                period_end="2026-05-28",
                fiscal_year=2026,
                fiscal_period="Q3",
            )
        ),
        "AMENDMENT": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=statement,
                ),
                period_end="2026-05-28",
                fiscal_year=2026,
                fiscal_period="Q3",
            )
        ),
    }

    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda accession: filing_map[accession],
        clock=lambda: pd.Timestamp(
            "2026-09-29",
            tz="UTC",
        ),
    )

    assert len(result.frame) == 2

    assert set(result.frame["sec_accession_number"]) == {
        "ORIGINAL",
        "AMENDMENT",
    }

    assert result.frame["period_end"].nunique() == 1


def test_history_skips_non_xbrl_filing_with_diagnostic():
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="OLD",
                period_end="2008-05-29",
                filing_date="2008-06-20",
                accepted_at="2008-06-20 18:00",
                is_xbrl=False,
            )
        ]
    )

    def loader_should_not_run(
        accession: str,
    ):
        raise AssertionError(f"Loader should not run: {accession}")

    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=loader_should_not_run,
    )

    assert result.frame.empty

    assert len(result.diagnostics) == 1

    diagnostic = result.diagnostics.iloc[0]

    assert diagnostic["status"] == "skipped"
    assert diagnostic["reason"] == "no_xbrl"


def test_history_skips_bad_quarter_without_losing_good_quarters(
    monkeypatch,
):
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="GOOD",
                period_end="2026-02-26",
                filing_date="2026-03-19",
                accepted_at="2026-03-19 22:00",
            ),
            _filing_row(
                accession_number="BAD",
                period_end="2026-05-28",
                filing_date="2026-06-25",
                accepted_at="2026-06-25 22:00",
            ),
        ]
    )

    good_statement = _quarter_statement(
        period_end="2026-02-26",
        fiscal_quarter=2,
    )

    bad_statement = pd.DataFrame(
        {
            "concept": ["example"],
            "2026-05-28 (YTD)": [100.0],
        }
    )

    filing_map = {
        "GOOD": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=good_statement,
                ),
                period_end="2026-02-26",
                fiscal_year=2026,
                fiscal_period="Q2",
            )
        ),
        "BAD": FakeFiling(
            _fake_xbrl(
                FakeStatements(
                    income=bad_statement,
                ),
                period_end="2026-05-28",
                fiscal_year=2026,
                fiscal_period="Q3",
            )
        ),
    }

    def fake_builder(**kwargs: Any) -> pd.DataFrame:
        if kwargs["sec_accession_number"] == "BAD":
            raise EdgarQuarterlyBuildError("Synthetic bad-quarter failure.")

        return _fake_quarter_builder(**kwargs)

    monkeypatch.setattr(
        history_module,
        "build_edgar_quarter",
        fake_builder,
    )

    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda accession: filing_map[accession],
        clock=lambda: pd.Timestamp(
            "2026-09-29",
            tz="UTC",
        ),
    )

    assert result.frame["sec_accession_number"].tolist() == ["GOOD"]

    bad_diagnostic = result.diagnostics[result.diagnostics["accession_number"].eq("BAD")].iloc[0]

    assert bad_diagnostic["status"] == "skipped"

    assert bad_diagnostic["reason"] == "quarter_build_error"


def test_history_records_missing_statement_failure():
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="EMPTY",
                period_end="2026-05-28",
                filing_date="2026-06-25",
                accepted_at="2026-06-25 22:00",
            )
        ]
    )

    filing = FakeFiling(
        _fake_xbrl(
            FakeStatements(),
            period_end="2026-05-28",
            fiscal_year=2026,
            fiscal_period="Q3",
        )
    )

    result = build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda _: filing,
    )

    assert result.frame.empty

    diagnostic = result.diagnostics.iloc[0]

    assert diagnostic["status"] == "skipped"

    assert diagnostic["reason"] == "quarter_build_error"

    assert "No usable EDGAR financial statements" in diagnostic["detail"]


def test_history_infers_10k_as_q4(
    quarter_calls,
):
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="K1",
                period_end="2026-08-31",
                filing_date="2026-10-15",
                accepted_at="2026-10-15 22:00",
                form="10-K",
            )
        ]
    )

    statement = pd.DataFrame(
        {
            "concept": ["example"],
            "2026-08-31 (FY)": [100.0],
        }
    )

    filing = FakeFiling(
        _fake_xbrl(
            FakeStatements(
                income=statement,
            ),
            period_end="2026-08-31",
            fiscal_year=2026,
            fiscal_period="FY",
        )
    )

    build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda _: filing,
        clock=lambda: pd.Timestamp(
            "2026-11-01",
            tz="UTC",
        ),
    )

    assert len(quarter_calls) == 1

    assert quarter_calls[0]["fiscal_quarter"] == 4


def test_history_uses_statement_label_for_10q_quarter(
    quarter_calls,
):
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="Q3",
                period_end="2026-05-28",
                filing_date="2026-06-25",
                accepted_at="2026-06-25 22:00",
            )
        ]
    )

    statement = _quarter_statement(
        period_end="2026-05-28",
        fiscal_quarter=3,
    )

    filing = FakeFiling(
        _fake_xbrl(
            FakeStatements(
                income=statement,
            ),
            period_end="2026-05-28",
            fiscal_year=2026,
            fiscal_period="Q3",
        )
    )

    build_edgar_history(
        symbol="MU",
        provider_symbol="MU",
        filing_lister=lambda _: filings,
        filing_loader=lambda _: filing,
        clock=lambda: pd.Timestamp(
            "2026-09-29",
            tz="UTC",
        ),
    )

    assert quarter_calls[0]["fiscal_quarter"] == 3


def test_history_empty_filings_returns_empty_frames():
    result = build_edgar_history(
        symbol="TEST",
        provider_symbol="TEST",
        filing_lister=lambda _: pd.DataFrame(),
    )

    assert result.frame.empty
    assert result.diagnostics.empty

    assert result.frame.columns.tolist() == (QUARTERLY_FUNDAMENTAL_COLUMNS)

    assert result.diagnostics.columns.tolist() == (DIAGNOSTIC_COLUMNS)


def test_history_strict_mode_reraises_unexpected_error():
    filings = pd.DataFrame(
        [
            _filing_row(
                accession_number="BROKEN",
                period_end="2026-05-28",
                filing_date="2026-06-25",
                accepted_at="2026-06-25 22:00",
            )
        ]
    )

    def broken_loader(
        accession: str,
    ):
        raise RuntimeError(f"loader exploded: {accession}")

    with pytest.raises(
        RuntimeError,
        match="loader exploded",
    ):
        build_edgar_history(
            symbol="MU",
            provider_symbol="MU",
            filing_lister=lambda _: filings,
            filing_loader=broken_loader,
            strict=True,
        )
