import pandas as pd
import pytest

from stock_agent.data.rates.transform import (
    percent_yield_to_decimal,
)


def test_percent_yield_to_decimal():
    yields = pd.Series(
        [3.88, 4.00],
        dtype=float,
    )

    result = percent_yield_to_decimal(yields)

    assert result.iloc[0] == pytest.approx(0.0388)

    assert result.iloc[1] == pytest.approx(0.04)
