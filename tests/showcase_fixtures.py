"""Shared fixtures for the showcase tests: a small registered configuration and fake bars."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from stock_agent.showcase.config import config_from_dict
from stock_agent.showcase.returns import window_sessions

REGISTERED = (
    Path(__file__).resolve().parents[1] / "configs" / "showcase" / "real_rebalancing_v1.yaml"
)

STRATEGIES = [
    {"key": "buy_and_hold", "rule": "buy_and_hold", "every": 21, "label": "Buy and hold",
     "description": "Equal weights bought once."},
    {"key": "equal_weight_weekly", "rule": "equal_weight", "every": 5,
     "label": "Equal weight, weekly", "description": "Every 5 sessions."},
    {"key": "equal_weight_monthly", "rule": "equal_weight", "every": 21,
     "label": "Equal weight, monthly", "description": "Every 21 sessions."},
    {"key": "equal_weight_quarterly", "rule": "equal_weight", "every": 63,
     "label": "Equal weight, quarterly", "description": "Every 63 sessions."},
    {"key": "inverse_volatility_monthly", "rule": "inverse_volatility", "every": 21,
     "label": "Inverse volatility, monthly", "description": "1/sigma every 21 sessions."},
]  # fmt: skip


def record(**changes) -> dict:
    """A small valid configuration record (2024 only, two universes), with `changes` applied."""

    base = {
        "schema_version": 1,
        "protocol": "exploratory_real_data_showcase",
        "name": "tiny_showcase",
        "window": {"start": "2024-01-02", "end": "2024-12-31"},
        "initial_wealth": 10000,
        "first_decision": 20,
        "execution_lag": 1,
        "offset": 0,
        "inverse_volatility_window": 20,
        "cost_classes": {"etf": "broad_etf", "stock": "single_stock"},
        "cost_multipliers": [0.0, 1.0, 2.0],
        "constraints": {"max_weight": 1.0, "max_turnover": None},
        "strategies": copy.deepcopy(STRATEGIES),
        "universes": [
            {
                "key": "funds",
                "label": "Two funds",
                "selection": "Two test funds.",
                "members": [
                    {"ticker": "XAA", "kind": "etf", "sector": "Energy"},
                    {"ticker": "XBB", "kind": "etf", "sector": "Utilities"},
                ],
            },
            {
                "key": "stocks",
                "label": "Two stocks",
                "selection": "Two test stocks.",
                "members": [
                    {"ticker": "SAA", "kind": "stock", "sector": "Energy"},
                    {"ticker": "SBB", "kind": "stock", "sector": "Utilities"},
                ],
            },
        ],
    }
    base.update(changes)
    return base


def config(**changes):
    return config_from_dict(record(**changes))


def write_config(path: Path, **changes) -> Path:
    path.write_text(yaml.safe_dump(record(**changes)), encoding="utf-8")
    return path


def sessions_2024() -> pd.DatetimeIndex:
    return window_sessions(pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-12-31").date())


def bars(sessions, closes, *, div=None, split=None) -> pd.DataFrame:
    """Bars in the store's columns; NaN closes are sessions the vendor did not serve."""

    closes = np.asarray(closes, dtype=float)
    frame = pd.DataFrame(
        {
            "session": sessions,
            "open": closes,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": np.where(np.isfinite(closes), 1e6, np.nan),
            "div_cash": 0.0,
            "split_factor": 1.0,
        }
    )
    for position, amount in (div or {}).items():
        frame.loc[position, "div_cash"] = amount
    for position, ratio in (split or {}).items():
        frame.loc[position, "split_factor"] = ratio
    return frame


def growing(sessions, rate: float, start: float = 100.0) -> np.ndarray:
    return start * (1.0 + rate) ** np.arange(len(sessions))


def noisy(sessions, seed: int, vol: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 50.0 * np.cumprod(1.0 + rng.normal(0.0003, vol, len(sessions)))


def stored(frame: pd.DataFrame, batch_id: str = "TEST-1") -> SimpleNamespace:
    """A stand-in for `StoredBars` (bars with batch_id, batches, revised sessions)."""

    batch = SimpleNamespace(
        batch_id=batch_id,
        source="test",
        retrieved_at="2026-10-26T18:00:00.000000Z",
        first_session="2024-01-02",
        last_session="2024-12-31",
        content_sha256="0" * 64,
    )
    return SimpleNamespace(
        bars=frame.assign(batch_id=batch_id), batches=(batch,), revised_sessions=()
    )


def loader(frames: dict[str, pd.DataFrame]):
    cache = {ticker: stored(frame) for ticker, frame in frames.items()}
    return cache.__getitem__


def default_frames(sessions) -> dict[str, pd.DataFrame]:
    split_closes = noisy(sessions, 4, 0.03)
    split_closes[100:] /= 2.0  # a real 2-for-1 split: the raw closes halve on its ex-date
    return {
        "XAA": bars(sessions, noisy(sessions, 1, 0.01), div={60: 0.4, 123: 0.4}),
        "XBB": bars(sessions, noisy(sessions, 2, 0.02)),
        "SAA": bars(sessions, noisy(sessions, 3, 0.015)),
        "SBB": bars(sessions, split_closes, split={100: 2.0}),
    }
