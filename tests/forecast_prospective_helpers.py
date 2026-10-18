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
from stock_agent.forecasting.prospective import (
    forecast,
    freeze,
    registration_record,
    synthetic,
    window,
)
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
# Invented Git object ids: the code the fixture bundle was frozen from, and the running code.
CODE_TREES = {
    path: f"{index + 1:040x}" for index, path in enumerate(registration_record.CODE_PATHS)
}
RUN_CODE = {**PROVENANCE["code"], "trees": CODE_TREES, "trees_dirty": False}
DOCUMENT_COMMIT = "d" * 40  # the commit of docs/research/prospective_registration.md
REGISTRATION_COMMIT = "e" * 40  # the registration pull request's merge commit
# GitHub's merge time of the registration pull request: after the fixture's freeze
# (REGISTERED_AT) and before 18:00 New York on AS_OF, so AS_OF is the first eligible close.
REGISTRATION_TIME = pd.Timestamp("2026-10-14T14:00:00Z")
REGISTERED_RECORDED = pd.Timestamp("2026-10-14T15:00:00Z")  # when the entry was appended
RECORD_PATH = "docs/research/registrations/fixture.json"


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


def registration(models: forecast.Models) -> dict:
    """The fixture's registration record for `models`."""

    return registration_record.build_record(
        models, code_trees=CODE_TREES, registration_document_commit=DOCUMENT_COMMIT
    )


PULL_REQUEST = 120  # the fixture registration pull request's number


def github_pull_request(merged_at=REGISTRATION_TIME, **changes) -> dict:
    """
    A GitHub API pull-request response for the fixture registration (invented
    values; unrelated fields included, as the API returns them).
    """

    if merged_at is None:
        text = None
    else:
        stamp = pd.Timestamp(merged_at)
        stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
        text = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    response = {
        "number": PULL_REQUEST,
        "state": "closed",
        "title": "Register the ETF forecasters",
        "merged": True,
        "merged_at": text,
        "merge_commit_sha": REGISTRATION_COMMIT,
        "base": {"ref": "main", "repo": {"full_name": "billvo9/stock-allocation-agent"}},
        "head": {"ref": "claude/register", "sha": "c" * 40},
        "user": {"login": "fixture"},
    }
    response.update(changes)
    return response


def evidence(pull_request=None, *, source="github_api", fetched_at=None) -> dict:
    """The registration evidence as the job collects it."""

    return {
        "source": source,
        "fetched_at": pd.Timestamp("2026-10-14T14:30:00Z") if fetched_at is None else fetched_at,
        "pull_request": github_pull_request() if pull_request is None else pull_request,
    }


def register(
    models: forecast.Models,
    ledger_path: Path,
    *,
    registered_at=REGISTRATION_TIME,
    recorded_at=REGISTERED_RECORDED,
):
    """Append the registration entry (the ledger's first) for `models`, with its evidence."""

    return registration_record.register(
        ledger_path,
        registration(models),
        models,
        record_path=RECORD_PATH,
        registration_commit=REGISTRATION_COMMIT,
        registered_at=registered_at,
        recorded_at=recorded_at,
        code=RUN_CODE,
        # Fetched just before the entry is appended, as the job's script does.
        evidence=evidence(github_pull_request(merged_at=registered_at), fetched_at=recorded_at),
    )


def record(
    frozen: Frozen,
    ledger_path: Path,
    store: Path,
    *,
    as_of=AS_OF,
    created=CREATED,
    loader=load_inference_window,
    models=None,
    code=None,
):
    """One forecast; a ledger with no entry is first registered for these models."""

    models = frozen.models() if models is None else models
    # Only a new or empty ledger is registered first: the job itself reads (and refuses)
    # anything else.
    if not Path(ledger_path).exists() or Path(ledger_path).stat().st_size == 0:
        register(models, ledger_path)
    return forecast.record_forecast(
        models,
        loader=loader,
        store_root=store,
        forecast_close=as_of,
        ledger_path=ledger_path,
        created_at=created,
        code=RUN_CODE if code is None else code,
        calendar=exchange_calendar,
    )


def score(ledger_path: Path, score_close: str, scored_at: str, store: Path):
    return forecast.score_matured(
        ledger_path=ledger_path,
        loader=load_inference_window,
        store_root=store,
        score_close=score_close,
        scored_at=pd.Timestamp(scored_at),
        code=RUN_CODE,
        calendar=exchange_calendar,
    )
