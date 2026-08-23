import pytest

from stock_agent.agents.momentum import momentum_weights


def test_momentum_weights_normalizes_positive_scores():
    scores = {
        "MU": 0.12,
        "SNDK": 0.06,
        "MRVL": -0.03,
    }

    result = momentum_weights(scores)

    assert result["MU"] == pytest.approx(2 / 3)
    assert result["SNDK"] == pytest.approx(1 / 3)
    assert result["MRVL"] == pytest.approx(0.0)
    assert result["CASH"] == pytest.approx(0.0)
    assert sum(result.values()) == pytest.approx(1.0)


def test_momentum_weights_uses_cash_when_all_scores_negative():
    scores = {
        "MU": -0.02,
        "SNDK": -0.04,
        "MRVL": -0.01,
    }

    result = momentum_weights(scores)

    assert result["CASH"] == pytest.approx(1.0)
    assert result["MU"] == pytest.approx(0.0)
    assert result["SNDK"] == pytest.approx(0.0)
    assert result["MRVL"] == pytest.approx(0.0)


def test_momentum_weights_empty_input():

    with pytest.raises(
        ValueError,
        match="Momentum scores cannot be empty.",
    ):
        momentum_weights({})


def test_momentum_weights_rejects_cash_in_scores():

    scores = {"MU": -0.02, "SNDK": -0.04, "MRVL": -0.01, "CASH": 0.01}

    with pytest.raises(ValueError, match="CASH must not be included in momentum scores."):
        momentum_weights(scores)
