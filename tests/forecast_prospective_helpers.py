"""
Shared fixtures for the prospective-track tests: a small test variant of the
registered config, a synthetic F1 research store, one frozen bundle and its
registry, and a synthetic prospective store written by the real prospective
writer and read back through the real `load_inference_window` (PR #73).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from prospective_helpers import new_store

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.schema import Instrument
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.data.prospective.store import ProspectiveBatch, ingest_prospective
from stock_agent.data.prospective.window import WINDOW_START
from stock_agent.forecasting.prospective import forecast, freeze, synthetic, window
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


# --- a synthetic prospective store, written and read by PR #73's real modules ---

STORE_SEED = 7
STORE_LAST = "2026-12-31"
# One batch per retrieval: each covers WINDOW_START to that session and is retrieved at
# 18:30 New York that day, after the session counts as complete.
RETRIEVALS = ("2026-10-14", "2026-10-15", "2026-10-16", "2026-10-22", "2026-10-23", STORE_LAST)


def exchange_calendar(start, end) -> pd.DatetimeIndex:
    return nyse_rules.sessions(start, end)


exchange_calendar.calendar_id = nyse_rules.CALENDAR_ID


def retrieved_after(session: str) -> pd.Timestamp:
    return window.local_instant(session, "18:30", "America/New_York").as_unit("us")


def prospective_tables(seed: int = STORE_SEED):
    return synthetic.synthetic_tables(ASSETS, nyse_rules.sessions(WINDOW_START, STORE_LAST), seed)


def _until(tables, last: pd.Timestamp):
    prices, distributions, splits = tables
    return (
        prices.loc[prices["session_date"] <= last].reset_index(drop=True),
        distributions.loc[distributions["ex_date"] <= last].reset_index(drop=True),
        splits.loc[splits["effective_date"] <= last].reset_index(drop=True),
    )


def prospective_batch(tables, retrieval: str) -> ProspectiveBatch:
    last = pd.Timestamp(retrieval, tz="UTC").as_unit("us")
    prices, distributions, splits = _until(tables, last)
    return ProspectiveBatch(
        adapter_id="fixture_prospective",
        adapter_version="1",
        fund_ids=ASSETS,
        start=WINDOW_START,
        end=last,
        instruments=tuple(Instrument(fund, fund, None) for fund in ASSETS),
        retrieved_at=retrieved_after(retrieval),
        raw_payload=f"synthetic prospective fixture {retrieval}".encode(),
        anchors_sha256="0" * 64,
        terms_note="synthetic fixture; no vendor terms",
        vintage_note="synthetic GARCH fixture",
        prices=prices,
        distributions=distributions,
        splits=splits,
    )


def prospective_store(root: Path, *, retrievals=RETRIEVALS, edit=None, seed=STORE_SEED) -> Path:
    """
    A prospective store under `root`, one batch per retrieval. `edit(retrieval,
    tables)` may return changed (prices, distributions, splits) for one batch, to
    plant a missing close, a revision or a flat stretch in that vintage only.
    """

    full = prospective_tables(seed)
    new_store(root)
    for retrieval in retrievals:
        tables = full if edit is None else edit(retrieval, tuple(t.copy() for t in full))
        ingest_prospective(root, prospective_batch(tables, retrieval))
    return root


def without_close(fund: str, sessions: list[str], from_retrieval: str = RETRIEVALS[0]):
    """An edit dropping `fund`'s closes on `sessions` from every batch retrieved from then on."""

    days = [pd.Timestamp(day, tz="UTC").as_unit("us") for day in sessions]

    def edit(retrieval, tables):
        prices, distributions, splits = tables
        if retrieval >= from_retrieval:
            drop = (prices["fund_id"] == fund) & prices["session_date"].isin(days)
            prices = prices.loc[~drop].reset_index(drop=True)
        return prices, distributions, splits

    return edit


def record(
    frozen: Frozen,
    ledger_path: Path,
    store: Path,
    *,
    as_of=AS_OF,
    created=CREATED,
    loader=load_inference_window,
    models=None,
):
    return forecast.record_forecast(
        frozen.models() if models is None else models,
        loader=loader,
        store_root=store,
        forecast_close=as_of,
        ledger_path=ledger_path,
        created_at=created,
        code=PROVENANCE["code"],
        calendar=exchange_calendar,
    )


def score(ledger_path: Path, score_close: str, scored_at: str, store: Path):
    return forecast.score_matured(
        ledger_path=ledger_path,
        loader=load_inference_window,
        store_root=store,
        score_close=score_close,
        scored_at=pd.Timestamp(scored_at),
        code=PROVENANCE["code"],
        calendar=exchange_calendar,
    )
