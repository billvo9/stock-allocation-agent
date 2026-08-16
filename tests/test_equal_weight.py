import pytest

from stock_agent.agents.equal_weight import equal_weight


def test_equal_weight_three_symbols():
    result = equal_weight(["MU", "SNDK", "MRVL"])

    assert result["MU"] == pytest.approx(1 / 3)
    assert result["SNDK"] == pytest.approx(1 / 3)
    assert result["MRVL"] == pytest.approx(1 / 3)

    assert sum(result.values()) == pytest.approx(1.0)


def test_equal_weight_empty_symbols():
    with pytest.raises(ValueError, match="empty"):
        equal_weight([])


def test_equal_weight_duplicate_symbols():
    with pytest.raises(ValueError, match="duplicates"):
        equal_weight(["MU", "SNDK", "MU"])
