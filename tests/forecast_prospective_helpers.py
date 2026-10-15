"""
Shared fixtures for the prospective-track tests: a small test variant of the
registered config, a synthetic F1 research store, one frozen bundle and its
registry, and a synthetic stand-in for the prospective store.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.forecasting.prospective import forecast, freeze, synthetic
from stock_agent.forecasting.prospective.config import ProspectiveConfig, parse_config
from stock_agent.registry import open_registry

REPO = Path(__file__).resolve().parents[1]
REGISTERED = REPO / "configs" / "prospective" / "etf_prospective_v1.yaml"
ASSETS = ("VTI", "AGG", "SHV")
PROVENANCE = {
    "code": {"git_sha": "0" * 40, "git_dirty": False},
    "environment": {"python": "3.12"},
}
STARTED = datetime(2026, 10, 14, 12, tzinfo=UTC)
REGISTERED_AT = datetime(2026, 10, 14, 13, tzinfo=UTC)
AS_OF = "2026-10-14"  # H_end, a Wednesday
CREATED = pd.Timestamp("2026-10-14T23:00:00Z")  # 19:00 New York, after the 18:00 close rule
STORE_FIRST, STORE_LAST = "2026-05-01", "2027-02-26"


def registered_raw() -> dict:
    with REGISTERED.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def small_config(**changes) -> ProspectiveConfig:
    """The registered config with a small fold geometry (a different variant, for tests)."""

    raw = copy.deepcopy(registered_raw())
    raw["experiment"] = "etf_prospective_test"
    raw["registry_prefix"] = "etf_prospective_test"
    raw["folds"] = {
        "start": "2021-01-04",
        "block_sessions": 126,
        "inner_block_sessions": 63,
        "inner_windows": 2,
        "inner_min_train_sessions": 252,
    }
    raw["band"]["min_calibration"] = 50
    raw["evaluation"]["bootstrap_reps"] = 19
    for path, value in changes.items():
        target = raw
        *parents, last = path.split("__")
        for key in parents:
            target = target[key]
        target[last] = value
    return parse_config(raw)


def research_batch():
    return synthetic.fixture_batch(
        ASSETS,
        start=pd.Timestamp("2019-01-02", tz="UTC"),
        end_exclusive=pd.Timestamp("2025-01-01", tz="UTC"),
        retrieved_at=pd.Timestamp("2026-10-14T12:00:00Z"),
    )


@dataclass(frozen=True, eq=False)
class Frozen:
    root: Path
    store: Path
    output: Path
    config: ProspectiveConfig
    result: freeze.FreezeResult

    @property
    def registry(self):
        return open_registry(self.output / freeze.REGISTRY_DIRECTORY, self.output)

    def models(self) -> forecast.Models:
        return forecast.load_models(self.result.bundle.path, self.registry)


def freeze_fixture(root: Path) -> Frozen:
    store = root / "f1_research_store"
    store.mkdir()
    ingest_batch(store, research_batch())
    config = small_config()
    output = root / "outputs"
    result = freeze.freeze(
        config,
        research_store_root=store,
        output_root=output,
        provenance=PROVENANCE,
        started_at=STARTED,
        clock=lambda: REGISTERED_AT,
    )
    return Frozen(root, store, output, config, result)


def prospective_store(last: str = STORE_LAST) -> synthetic.FixtureStore:
    store = synthetic.fixture_store(ASSETS, first=STORE_FIRST, last=STORE_LAST)
    return synthetic.truncated(store, last)


def record(
    frozen: Frozen,
    ledger_path: Path,
    *,
    store=None,
    as_of=AS_OF,
    created=CREATED,
    loader=synthetic.load_fixture_window,
    models=None,
):
    return forecast.record_forecast(
        frozen.models() if models is None else models,
        loader=loader,
        store_root=prospective_store() if store is None else store,
        forecast_close=as_of,
        ledger_path=ledger_path,
        created_at=created,
        code=PROVENANCE["code"],
        calendar=synthetic.weekday_sessions,
    )


def score(ledger_path: Path, score_close: str, scored_at: str, store=None):
    return forecast.score_matured(
        ledger_path=ledger_path,
        loader=synthetic.load_fixture_window,
        store_root=prospective_store() if store is None else store,
        score_close=score_close,
        scored_at=pd.Timestamp(scored_at),
        code=PROVENANCE["code"],
        calendar=synthetic.weekday_sessions,
    )
