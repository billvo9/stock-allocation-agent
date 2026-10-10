from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.training import LabelSpec


def build_labeled_panel(
    *,
    symbols: tuple[str, ...] = ("A", "B", "C"),
    sessions: int = 120,
    horizon: int = 5,
    entry_lag: int = 1,
    seed: int = 0,
    start: str = "2021-01-04",
    drop_fraction: float = 0.0,
    listing: dict[str, tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """
    Seeded synthetic labeled panel: independent log random walks per symbol,
    two noise features, optional listing windows and randomly dropped
    sessions (gaps). Dates are session dates at 00:00 UTC; the result has a
    default RangeIndex sorted by (date, symbol).
    """

    rng = np.random.RandomState(seed)  # legacy stream: stable across numpy versions
    calendar = pd.bdate_range(start, periods=sessions, tz="UTC")
    rows = []
    for symbol in symbols:
        prices = 100.0 * np.exp(np.cumsum(0.01 * rng.randn(sessions)))
        first, last = (listing or {}).get(symbol, (0, sessions))
        for index in range(first, last):
            if drop_fraction and rng.rand() < drop_fraction:
                continue
            rows.append(
                {
                    "date": calendar[index],
                    "symbol": symbol,
                    "adjusted_close": prices[index],
                    "feature_a": rng.randn(),
                    "feature_b": rng.randn(),
                }
            )
    labeled = LabelSpec(horizon=horizon, entry_lag=entry_lag).apply(pd.DataFrame(rows))
    return labeled.sort_values(["date", "symbol"]).reset_index(drop=True)


@pytest.fixture
def labeled_panel() -> Callable[..., pd.DataFrame]:
    return build_labeled_panel


@pytest.fixture(scope="session")
def panel_factory() -> Callable[..., pd.DataFrame]:
    """build_labeled_panel for module-scoped fixtures (expensive shared runs)."""

    return build_labeled_panel


@pytest.fixture(scope="session")
def dashboard_root(tmp_path_factory, panel_factory):
    """
    A compact, internally consistent T2 output root for dashboard tests:
    one development run written by the real writer (hashes and contract
    valid), built once per session from a tiny synthetic panel.
    """

    from datetime import UTC, datetime

    from stock_agent.model_diagnostics import artifacts, runner
    from stock_agent.model_validation.folds import make_expanding_folds, make_test_windows

    symbols = ["A", "B", "C", "D"]
    spec = LabelSpec(horizon=5, entry_lag=1)
    raw = panel_factory(symbols=tuple(symbols), sessions=220, horizon=5, seed=8)
    raw = raw.drop(columns=["target_return", "target_start_date", "target_end_date"])
    lockbox = raw["date"].drop_duplicates().sort_values().iloc[200]
    labeled, facts = runner.prepare_development_frame(
        raw, symbols=symbols, label_spec=spec, holdout_start=lockbox
    )
    sessions = labeled["date"].drop_duplicates().sort_values()
    windows = make_test_windows(sessions, start=sessions.iloc[80], end=lockbox, block_sessions=30)
    folds = make_expanding_folds(labeled, windows, holdout_start=lockbox, lockbox_start=lockbox)
    config = runner.DiagnosticsConfig(
        hac_lag=3,
        block_length=10,
        bootstrap_reps=19,
        acf_max_lag=4,
        permutation_draws=3,
        permutation_block_length=10,
        stale_lags=(15,),
        stale_models=("feature_a",),
        permutation_models=("per_symbol_mean", "feature_a"),
        calibration_reps=30,
        calibration_bootstrap_reps=9,
    )
    run = runner.run_diagnostics(
        labeled,
        folds,
        universe=symbols,
        label_spec=spec,
        diagnostic_features=["feature_a", "feature_b"],
        provenance={
            "code": {"git_sha": "abc123", "git_dirty": False},
            "environment": {"python": "3.12"},
            "truncation": {**facts, "holdout_rows_loaded": True},
            "universe_note": "Synthetic test universe.",
        },
        started_at=datetime(2026, 10, 9, 12, 0, tzinfo=UTC),
        models=runner.default_models(score_feature="feature_a"),
        config=config,
        lockbox_start=lockbox,
    )
    root = tmp_path_factory.mktemp("dashboard_outputs")
    artifacts.write_run(run, root, extra_record={"status": "completed"})
    artifacts.append_ledger(
        root,
        {"run_id": run.run_id, "spec_sha256": run.record["spec_sha256"], "status": "completed"},
    )
    return root, run


SCHEMA_1_0_RUN_ID = "20261009T120000Z-100000000000"


def write_schema_1_0_run(run, root, run_id: str = SCHEMA_1_0_RUN_ID):
    """
    Write `run` as the schema 1.0 writer (PR #43) laid it out: no 1.1 record
    fields (decision_method, reference_predictors, eligible_as_candidate,
    checks_summary.families), curves without ci_method / ci_level / hac_lag,
    and "1.0" in the manifest and record. Statistics are unchanged between
    1.0 and 1.1, so every other stored value is what 1.0 wrote.
    """

    import copy
    import json

    from stock_agent.model_diagnostics import contract
    from stock_agent.model_diagnostics.record import canonical_json, sha256_json
    from stock_agent.model_validation.audit import file_sha256, table_sha256

    def dump(path, value):
        text = json.dumps(json.loads(canonical_json(value)), indent=2, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")

    record = copy.deepcopy(run.record)
    spec = record["spec"]
    for key in ("decision_method", "reference_predictors"):
        spec.pop(key)
    for model in spec["models"]:
        model.pop("eligible_as_candidate")
    spec["output_schema_version"] = spec["record_schema_version"] = "1.0"
    record["checks_summary"].pop("families")
    record["spec_sha256"] = sha256_json(spec)
    record["run_id"] = run_id
    directory = root / "runs" / run_id
    directory.mkdir(parents=True)
    files = {}
    for name, table in run.tables.items():
        if name in contract.SCHEMAS:
            table = contract.conform(
                name, table.drop(columns=list(contract.added_after(name, "1.0"))), "1.0"
            )
        path = directory / f"{name}.parquet"
        table.to_parquet(path, index=False)
        files[path.name] = {
            "table": name,
            "rows": len(table),
            "content_sha256": table_sha256(table),
        }
    dump(directory / "record.json", {**record, "status": "completed"})
    files["record.json"] = {}
    for filename, entry in files.items():
        entry["file_sha256"] = file_sha256(directory / filename)
    dump(
        directory / "manifest.json",
        {"output_schema_version": "1.0", "run_id": run_id, "files": files},
    )
    return directory


@pytest.fixture(scope="session")
def schema_1_0_writer():
    return write_schema_1_0_run


@pytest.fixture(scope="session")
def schema_1_0_root(tmp_path_factory, dashboard_root):
    """The dashboard fixture run, written in the schema 1.0 layout under its own root."""

    _, run = dashboard_root
    root = tmp_path_factory.mktemp("schema_1_0_outputs")
    write_schema_1_0_run(run, root)
    return root, run
