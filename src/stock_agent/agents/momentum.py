from __future__ import annotations


def momentum_weights(
    momentum_scores: dict[str, float],
    cash_symbol: str = "CASH",
) -> dict[str, float]:
    """Convert momentum scores into long-only portfolio weights."""

    if not momentum_scores:
        raise ValueError("Momentum scores cannot be empty.")

    if cash_symbol in momentum_scores:
        raise ValueError(f"{cash_symbol} must not be included in momentum scores.")

    positive_scores = {symbol: max(score, 0.0) for symbol, score in momentum_scores.items()}

    total_positive_score = sum(positive_scores.values())

    if total_positive_score == 0:
        return {
            **{symbol: 0.0 for symbol in momentum_scores},
            cash_symbol: 1.0,
        }

    return {
        **{symbol: score / total_positive_score for symbol, score in positive_scores.items()},
        cash_symbol: 0.0,
    }
