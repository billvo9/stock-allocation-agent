from __future__ import annotations


def equal_weight(symbols: list[str]) -> dict[str, float]:
    """Return a fully invested equal-weight allocation.

    Raises:
        ValueError: if symbols is empty or contains duplicates.
    """
    if not symbols:
        raise ValueError("symbols is empty")

    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols contains duplicates")

    percentage_per_symbol = 1 / len(symbols)
    equal_stock_weight = {}
    for symbol in symbols:
        equal_stock_weight[symbol] = percentage_per_symbol
    return equal_stock_weight
