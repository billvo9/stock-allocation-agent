from __future__ import annotations


def inverse_volatility(
    volatilities: dict[str, float],
) -> dict[str, float]:
    if not volatilities:
        raise ValueError("volatilities dictionary is empty")

    inverse_scores: dict[str, float] = {}
    total_inverse_score = 0.0

    for symbol, volatility in volatilities.items():
        if volatility <= 0:
            raise ValueError("Volatility must be positive")

        inverse_score = 1.0 / volatility

        inverse_scores[symbol] = inverse_score
        total_inverse_score += inverse_score

    return {symbol: score / total_inverse_score for symbol, score in inverse_scores.items()}
