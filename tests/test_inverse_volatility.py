import pytest

from stock_agent.agents.inverse_volatility import inverse_volatility


def test_inverse_volatility_expected_weights():
    result = inverse_volatility({"MU": 0.02, "SNDK": 0.04, "MRVL": 0.05})

    assert result["MU"] == pytest.approx(50 / 95)
    assert result["SNDK"] == pytest.approx(25 / 95)
    assert result["MRVL"] == pytest.approx(20 / 95)
    assert sum(result.values()) == pytest.approx(1.0)


def test_inverse_volatility_equal_volatility():
    result = inverse_volatility({"MU": 0.03, "SNDK": 0.03, "MRVL": 0.03})

    assert result["MU"] == pytest.approx(1 / 3)
    assert result["SNDK"] == pytest.approx(1 / 3)
    assert result["MRVL"] == pytest.approx(1 / 3)


def test_inverse_volatility_rejects_zero_volatility():

    with pytest.raises(ValueError, match="be positive"):
        inverse_volatility({"MU": 0.03, "SNDK": 0.0, "MRVL": 0.03})


def test_inverse_volatility_rejects_empty_input():

    with pytest.raises(ValueError, match="is empty"):
        inverse_volatility({})


def test_inverse_volatility_rejects_negative_volatility():

    with pytest.raises(ValueError, match="be positive"):
        inverse_volatility({"MU": 0.03, "SNDK": 0.5, "MRVL": -0.03})
