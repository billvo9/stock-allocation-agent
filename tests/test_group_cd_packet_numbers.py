"""
The group C and group D decision packets quote numbers computed by
`scripts/group_cd_packet_numbers.py`. These tests keep each packet in step with
the script, and recompute the key numbers independently: with the existing
risk-free accrual (`prepare_risk_free_returns`) and with closed forms.
"""

from __future__ import annotations

import importlib.util
import math
import re
from pathlib import Path
from statistics import NormalDist

import pandas as pd
import pytest

from stock_agent.data.rates.schema import RATE_COLUMNS
from stock_agent.data.rates.transform import prepare_risk_free_returns

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "research"
SCRIPT = ROOT / "scripts" / "group_cd_packet_numbers.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("group_cd_packet_numbers", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


NUMBERS = _load_script()
GROUP_C = NUMBERS.group_c_numbers()
GROUP_D = NUMBERS.group_d_numbers()


def _packet(name: str) -> str:
    # The packets use the typographic minus sign; the script prints ASCII.
    return (DOCS / name).read_text(encoding="utf-8").replace("−", "-")


@pytest.mark.parametrize(
    ("numbers", "packet"),
    [(GROUP_C, "group_c_decisions.md"), (GROUP_D, "group_d_decisions.md")],
)
def test_every_computed_number_appears_in_its_packet(numbers, packet):
    text = _packet(packet)

    def quoted(value: str) -> bool:
        # A short value ("20", "0.0") must appear in bold, so a stray digit cannot match it;
        # any value must stand alone, not inside a longer number.
        token = f"**{value}**" if len(value) < 4 else value
        return re.search(rf"(?<![\d.]){re.escape(token)}(?!\d)", text) is not None

    missing = {key: value for key, value in numbers.items() if not quoted(value)}
    assert not missing, f"{packet} does not quote: {missing}"


def _rates(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp(day),
                "rate_id": "USD_TREASURY_3M",
                "provider_series_id": "DGS3MO",
                "currency": "USD",
                "tenor": "3M",
                "annual_yield": value,
                "quote_convention": "investment_basis",
                "source": "FRED",
            }
            for day, value in rows
        ],
        columns=RATE_COLUMNS,
    )


def _accruals(rows: list[tuple[str, float]], dates: list[str]) -> list[float]:
    returns = prepare_risk_free_returns(
        _rates(rows), pd.DatetimeIndex(pd.to_datetime(dates)), "USD_TREASURY_3M"
    )
    return returns.tolist()


def test_the_existing_accrual_reproduces_the_c1_and_c2_examples():
    """Approving C1 adds no arithmetic: the existing function already gives these."""

    rows = [("2023-12-01", 0.05)]
    (fri_mon,) = _accruals(rows, ["2023-12-15", "2023-12-18"])
    (crossing,) = _accruals(rows, ["2023-12-29", "2024-01-02"])
    assert fri_mon == pytest.approx(0.05 * 3 / 365, abs=1e-15)
    assert crossing == pytest.approx(0.05 * (3 / 365 + 1 / 366), abs=1e-15)
    assert float(GROUP_C["C1.fri_mon"]) == pytest.approx(fri_mon, abs=5e-10)
    assert float(GROUP_C["C1.year_crossing"]) == pytest.approx(crossing, abs=5e-10)

    # Columbus Day 2024: no value for Monday, so Friday's rate is carried.
    columbus = [("2024-10-10", 0.050), ("2024-10-11", 0.051)]
    monday, tuesday = _accruals(columbus, ["2024-10-11", "2024-10-14", "2024-10-15"])
    assert monday == pytest.approx(0.051 * 3 / 366, abs=1e-15)
    assert tuesday == pytest.approx(0.051 * 1 / 366, abs=1e-15)
    assert float(GROUP_C["C2.columbus_fri_mon"]) == pytest.approx(monday, abs=5e-10)
    assert float(GROUP_C["C2.columbus_mon_tue_carried"]) == pytest.approx(tuesday, abs=5e-10)


def test_the_bond_equivalent_conversion_recovers_the_investment_yield():
    price, days = 98.75, 91
    investment = (100 - price) / price * 365 / days
    discount = (100 - price) / 100 * 360 / days
    assert 365 * discount / (360 - discount * days) == pytest.approx(investment, abs=1e-15)
    assert investment * days / 365 == pytest.approx(100 / price - 1, abs=1e-15)


def test_calendar_cagr_and_lo_standard_error_closed_forms():
    assert GROUP_D["M1.cagr_calendar"] == f"{1.10 ** (365.25 / 365) - 1:.7f}"
    assert GROUP_D["M1.cagr_sessions"] == f"{1.10 ** (252 / 248) - 1:.7f}"
    lo_se = math.sqrt((1 + 0.5 * 0.05**2) / 2520) * math.sqrt(252)
    assert GROUP_D["M2.lo_se"] == f"{lo_se:.3f}"


def test_deflated_sharpe_with_one_trial_is_the_probabilistic_sharpe_ratio():
    n, sr, skew, kurtosis = 1260, 0.80 / math.sqrt(252), -0.5, 6.0
    psr = NormalDist().cdf(sr * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + 1.25 * sr**2))
    sr0, dsr = NUMBERS.deflated_sharpe(sr, n, skew, kurtosis, 1, 1.0 / n)
    assert sr0 == 0.0
    assert dsr == pytest.approx(psr, abs=1e-9)

    # More trials raise the bar: SR_0 grows and the DSR falls.
    bars = [NUMBERS.deflated_sharpe(sr, n, skew, kurtosis, k, 1.0 / n) for k in (2, 10, 100)]
    assert [b[0] for b in bars] == sorted(b[0] for b in bars)
    assert [b[1] for b in bars] == sorted((b[1] for b in bars), reverse=True)


def test_expected_maximum_of_five_normals():
    # The known value of E[max of 5 standard normals] is 1.162964.
    assert NUMBERS.expected_max_of_normals(5) == pytest.approx(1.162964, abs=1e-5)


def test_the_power_table_uses_the_stated_formula():
    z = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    assert GROUP_D["M9.z_sum"] == f"{z:.4f}"
    assert GROUP_D["M9.mde_sr_normal_rho0.9_t10"] == f"{z * math.sqrt(2 * 0.1 / 10):.2f}"


def test_no_source_module_imports_the_packet_script():
    """The script is documentation; production code never depends on it."""

    offenders = [
        path
        for path in (ROOT / "src").rglob("*.py")
        if "group_cd_packet_numbers" in path.read_text(encoding="utf-8")
    ]
    assert not offenders
