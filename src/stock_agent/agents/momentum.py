from __future__ import annotations

from stock_agent.agents.equal_weight import equal_weight

def momentum_weights(
    momentum_scores: dict[str, float],
) -> dict[str, float]:
    """Convert momentum scores into long-only portfolio weights."""

    if not momentum_scores:
        raise ValueError("Momentum scores cannot be empty.")

    positive_scores = {
        symbol: max(score, 0.0)
        for symbol, score in momentum_scores.items()
    }

    total_positive_score = sum(positive_scores.values())

    # 1. If total_positive_score == 0:
    #       return equal weights across all symbols
    if total_positive_score == 0:
        return equal_weight(list(momentum_scores.keys()))
    #
    # 2. Otherwise:
    #       normalize positive_scores so the weights sum to 1
    return {
        symbol: positive_scores[symbol] / total_positive_score  
        for symbol in positive_scores
        }

