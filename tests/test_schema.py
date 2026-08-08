import pandas as pd

from stock_agent.data.yfinance_source import standardize_yfinance_columns


def test_standardize_yfinance_columns():
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [98.0],
            "Close": [103.0],
            "Adj Close": [102.5],
            "Volume": [1_000_000],
        }
    )

    result = standardize_yfinance_columns(raw)

    expected_columns = {
        "open",
        "high",
        "low",
        "close",
        "adjusted_close",
        "volume",
    }

    assert expected_columns.issubset(result.columns)


def test_standardization_does_not_modify_original_frame():
    raw = pd.DataFrame(
        {
            "Open": [100.0],
            "Close": [103.0],
        }
    )

    original_columns = raw.columns.tolist()

    standardize_yfinance_columns(raw)

    assert raw.columns.tolist() == original_columns