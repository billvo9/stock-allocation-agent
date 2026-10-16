"""Stock Probability: the saved watchlist and each stock's daily history (experimental)."""

from stock_agent.app import stock_lab
from stock_agent.stock_lab.store import StockLabStore
from stock_agent.stock_lab.synthetic import demo_histories

stock_lab.render(StockLabStore, demo_histories)
