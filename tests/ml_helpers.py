"""Shared helpers for the experiment-runner tests: a small, fast synthetic config."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

from stock_agent.forecasting.config import parse_config
from stock_agent.forecasting.experiment import run_experiment

PROVENANCE = {
    "code": {"git_sha": "0" * 40, "git_dirty": False},
    "environment": {"python": "3.12"},
}
SMALL = {
    "schema_version": "1.0",
    "experiment": "test_planted",
    "track": "equity_cross_sectional",
    "description": "small planted-signal run for tests",
    "data": {
        "source": "synthetic_equity_panel",
        "params": {
            "n_assets": 16,
            "n_sessions": 780,
            "start": "2012-01-03",
            "signal": "linear",
            "signal_strength": 0.001,
            "late_listings": 2,
            "delistings": 0,
            "seed": 3,
        },
    },
    "universe": {"id": "test_universe", "min_history_sessions": 60},
    "features": {"families": ["relative_technical", "risk_liquidity", "fundamental_quality"]},
    "label": {"horizon": 10, "entry_lag": 1, "benchmark": "equal_weight"},
    "folds": {
        "start": "2013-09-02",
        "block_sessions": 42,
        "inner_block_sessions": 42,
        "inner_windows": 2,
        "inner_min_train_sessions": 126,
    },
    "models": {"candidates": ["ridge"], "include_ols": True, "increment_reference": "ridge"},
    "evaluation": {
        "seed": 0,
        "permutation_draws": 1,
        "permutation_models": ["ridge"],
        "bootstrap_reps": 19,
        "calibration_reps": 20,
        "calibration_bootstrap_reps": 9,
    },
    "registry": {"model_prefix": "test_xs"},
}


def config_with(**changes):
    raw = copy.deepcopy(SMALL)
    for path, value in changes.items():
        target = raw
        *parents, last = path.split("__")
        for key in parents:
            target = target[key]
        target[last] = value
    return parse_config(raw)


def run(config, root, minute=0):
    return run_experiment(
        config,
        output_root=root,
        provenance=PROVENANCE,
        started_at=datetime(2026, 10, 14, 12, minute, tzinfo=UTC),
        clock=lambda: datetime(2026, 10, 14, 13, minute, tzinfo=UTC),
    )
