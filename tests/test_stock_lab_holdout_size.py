"""
The holdout's size and power study (scripts/research/stock_lab_holdout_size.py,
SIMULATION ONLY): its design is the registered protocol's, each piece computes
what the frozen model and the holdout's test compute, it is deterministic, and
the committed record matches the code. Nothing here reads data or a store.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.calendars import nyse_rules
from stock_agent.model_diagnostics.inference import fold_block_t, student_t_ppf
from stock_agent.stock_lab import estimators as est
from stock_agent.stock_lab import evaluation, registry, scores
from stock_agent.stock_lab import planned_simulation as ps
from stock_agent.stock_lab.contracts import DEFAULT_BAND
from stock_agent.stock_lab.size_simulation import one_sided_p

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "research" / "stock_lab_holdout_size.py"
RECORD = PROJECT / "docs" / "research" / "stock_lab_holdout_size.json"
ALLOWED_IMPORTS = {
    "__future__",
    "argparse",
    "itertools",
    "json",
    "math",
    "sys",
    "collections.abc",
    "dataclasses",
    "pathlib",
    "numpy",
    "stock_agent.model_diagnostics.inference",
    "stock_agent.stock_lab",
    "stock_agent.stock_lab.size_simulation",
}


@pytest.fixture(scope="module")
def study():
    name = "research_stock_lab_holdout_size"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses resolve their annotations through it
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def tiny(study, **changes):
    """A small design with the primary's structure: 3 blocks of 20 decisions."""

    settings = {
        "block_sizes": (20, 20, 20),
        "window": 60,
        "half_life": 60.0,
        "development_years": 3,
        "sessions_per_year": 60,
        "reps": 6,
        "chunk": 4,
    }
    return study.HoldoutDesign(**{**settings, **changes})


# --- the study reads nothing ---


def test_the_study_is_labelled_and_imports_no_reader(study):
    text = SCRIPT.read_text(encoding="utf-8")
    assert text.startswith('"""\nSIMULATION ONLY')
    imports = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ImportFrom):
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    assert imports <= ALLOWED_IMPORTS, imports - ALLOWED_IMPORTS
    for forbidden in ("read_parquet", "duckdb", "data/", "reports/", "stock-agent-private"):
        assert forbidden not in text, forbidden


# --- the design is the protocol's ---


def _holdout_block_sizes() -> list[int]:
    """Decisions per calendar quarter, by the protocol's rules (the docs' section 2)."""

    days = nyse_rules.sessions("2024-01-02", "2026-10-14")
    first = pd.Timestamp("2024-12-31", tz="UTC")
    window = days[days <= first]
    assert len(window) == 252 and window[-1] == first  # FETCH_START opens one window
    decisions = days[days >= first][:-5]
    starts = pd.DatetimeIndex(
        [pd.Timestamp(f"{year}-{month:02d}-01", tz="UTC") for year in (2025, 2026)
         for month in (1, 4, 7, 10)][:7]
    )  # fmt: skip
    index = np.clip(starts.searchsorted(decisions, side="right") - 1, 0, len(starts) - 1)
    return np.bincount(index, minlength=len(starts)).tolist()


def test_the_design_is_the_registered_protocols(study):
    design = study.HoldoutDesign()
    assert list(study.BLOCK_SIZES) == _holdout_block_sizes() == [61, 62, 64, 64, 61, 62, 69]
    assert design.decisions == 443 and len(design.block_sizes) == 7
    assert (design.window, design.half_life, design.nominal) == (252, 252.0, 0.025)
    config = evaluation.EvaluationConfig()
    assert (design.kappa, design.alpha_prior) == (config.kappa, config.alpha_prior)
    assert design.tau == DEFAULT_BAND.tau and evaluation.ALPHA == design.nominal
    # The first decision is the last development session; outcomes run five sessions past
    # the last decision: 252 window sessions and 447 holdout sessions, as fetched.
    assert design.first == design.development - 1
    assert design.sessions - design.fetch_start == 252 + 447
    # Decision A7's planned panels: the persistent volatility regimes of its simulation.
    planned = ps.PlannedDesign(names=4)
    for name in ("correlation", "daily_vol", "t_df", "arch", "garch", "sessions_per_year"):
        assert getattr(design, name) == getattr(planned, name), name
    assert design.development_years == 2024 - 2003 + 1


def test_the_blocks_are_the_protocols_when_it_is_present(study):
    protocol = pytest.importorskip("stock_agent.stock_lab.holdout.protocol")
    registered = protocol.PROTOCOL
    decisions = registered.decisions(nyse_rules.sessions)
    labels = registered.block_labels(decisions)
    sizes = [int((labels == name).sum()) for name, _ in registered.blocks]
    assert sizes == list(study.BLOCK_SIZES)
    assert (registered.window_sessions, registered.alpha) == (252, 0.025)
    assert (registered.comparator, registered.deciding_score) == ("B0", "rps")


# --- each piece is the holdout's own computation ---


def test_window_b1_is_the_frozen_models_windowed_b1(study):
    rng = np.random.RandomState(1)
    window, half_life, kappa, sessions, names = 30, 21.0, 40.0, 80, 2
    codes = rng.randint(0, 3, size=(1, sessions, names))
    prior = rng.dirichlet(np.ones(3), size=(1, names))
    first = window - 1
    count = sessions - first
    found = study.window_b1(
        np.eye(3)[codes],
        prior,
        half_life=half_life,
        kappa=kappa,
        window=window,
        first=first,
        count=count,
    )
    for j in (0, 1, 17, count - 1):
        decision = first + j
        for i in range(names):
            # The window's first session has no return inside it: no category.
            window_codes = codes[0, decision - window + 1 : decision + 1, i].astype("float64")
            window_codes[0] = np.nan
            expected = registry._recency_weighted(window_codes, half_life, kappa, prior[0, i])
            np.testing.assert_allclose(found[0, j, i], expected, rtol=0, atol=1e-12)


def test_frozen_b0_is_the_estimators_b0_at_the_last_development_row(study):
    rng = np.random.RandomState(2)
    sessions, names = 40, 3
    codes = rng.choice(3, size=(sessions, names), p=[0.5, 0.2, 0.3])
    panel = pd.DataFrame(
        {
            "symbol": np.repeat([f"S{i}" for i in range(names)], sessions),
            "session": np.tile(pd.date_range("2020-01-01", periods=sessions), names),
            "category_code": codes.T.reshape(-1),
        }
    )
    b0 = est.b0(panel, est.FrequencyParams(alpha=50.0))
    last = [(i + 1) * sessions - 1 for i in range(names)]
    counts = np.stack([(codes == k).sum(axis=0) for k in range(3)], axis=-1)[None].astype(float)
    np.testing.assert_allclose(study.frozen_b0(counts, 50.0)[0], b0[last], atol=1e-12)


def test_differentials_are_the_pooled_daily_rps_difference(study):
    rng = np.random.RandomState(3)
    reps, sessions, names, first, count = 2, 20, 3, 4, 10
    codes = rng.randint(0, 3, size=(reps, sessions, names))
    b1 = rng.dirichlet(np.ones(3), size=(reps, count, names))
    b0 = rng.dirichlet(np.ones(3), size=(reps, names))
    found = study.differentials(codes, b1, b0, first=first, count=count)
    for r in range(reps):
        for j in range(count):
            outcomes = codes[r, first + j + 1 : first + j + 6]  # (horizons, names)
            per_target = [
                (
                    scores.rps(b1[r, j], outcomes[h]) - scores.rps(b0[r], outcomes[h])
                ).mean()  # over stocks, as the holdout averages each target first
                for h in range(5)
            ]
            assert found[r, j] == pytest.approx(np.mean(per_target), abs=1e-12)


def test_block_t_is_the_holdouts_one_sided_fold_block_t(study):
    rng = np.random.RandomState(4)
    sizes = study.BLOCK_SIZES
    labels = np.repeat(np.arange(len(sizes)), sizes)
    series = rng.normal(0.0, 0.01, size=(400, sum(sizes)))
    series += rng.uniform(-0.004, 0.002, size=(400, 1))  # some clear rejections
    estimate, se = study.block_t(study.block_means(series, sizes))
    critical = student_t_ppf(0.025, len(sizes) - 1)
    decisions = []
    for row, (mean, error) in enumerate(zip(estimate, se, strict=True)):
        result = fold_block_t(series[row], labels)
        assert (result.estimate, result.se) == pytest.approx((mean, error), rel=1e-12)
        decisions.append(result.estimate < 0 and one_sided_p(result) < 0.025)
    assert decisions == list(estimate / se < critical)
    assert 20 < sum(decisions) < 380
    assert critical == pytest.approx(-2.4469, abs=1e-4)


def test_a_shift_changes_the_returns_only_from_its_onset(study):
    design = tiny(study)
    returns = np.random.RandomState(5).normal(size=(2, design.sessions, design.names))
    tail = returns[:, design.fetch_start :]
    whole = study.shifted(returns, study.Shift("w", drift=0.01, vol_scale=2.0), design)
    np.testing.assert_allclose(whole, 2.0 * tail + 0.01)
    late = study.shifted(returns, study.Shift("h", drift=0.01, onset="holdout"), design)
    onset = design.first + 1 - design.fetch_start
    np.testing.assert_array_equal(late[:, :onset], tail[:, :onset])
    np.testing.assert_allclose(late[:, onset:], tail[:, onset:] + 0.01)
    with pytest.raises(ValueError, match="onset"):
        study.shifted(returns, study.Shift("x", onset="later"), design)


# --- the summary ---


def test_independent_normal_blocks_give_the_nominal_size(study):
    rng = np.random.RandomState(6)
    means = rng.normal(0.001, 0.002, size=(40_000, 7))
    mix = np.zeros((40_000, 3))
    found = study.summarize(means, mix, nominal=0.025, deltas=(0.001, 0.002))
    size = found["boundary_size"]
    assert abs(size["rate"] - 0.025) < 4 * size["mc_se"]
    assert found["variance_ratio"] == pytest.approx(1.0, abs=0.03)
    assert abs(found["block_correlation"]["all_pairs_mean"]) < 0.01
    assert found["exact_critical"] == pytest.approx(student_t_ppf(0.025, 6), abs=0.1)
    assert found["true_mean"] == pytest.approx(0.001, abs=4 * found["true_mean_mc_se"])
    assert found["reject"]["means"] == "false_positive_rate"
    power = [row["rate"] for row in found["rps_gain_power"]]
    assert 0.025 < power[0] < power[1]
    rates = [row["rate"] for row in found["calibration"]]
    assert rates == sorted(rates)


def test_the_simulation_is_deterministic_and_a_drift_moves_the_mix(study):
    shifts = (study.Shift("none"), study.Shift("up", drift=0.004), study.Shift("wide", 0, 2.0))
    design = tiny(study, shifts=shifts)
    first, again = study.run_design(design), study.run_design(design)
    assert first == again
    other = study.run_design(tiny(study, shifts=shifts, seed=7))
    assert other["scenarios"][0]["true_mean"] != first["scenarios"][0]["true_mean"]
    none, up, wide = (row["mix_shift_pp"] for row in first["scenarios"])
    assert up["increase"] - none["increase"] > 2.0 and up["decrease"] - none["decrease"] < -2.0
    assert wide["stable"] - none["stable"] < -2.0
    assert [row["shift"]["name"] for row in first["scenarios"]] == ["none", "up", "wide"]
    assert len(first["scenarios"][0]["rps_gain_power"]) == len(study.DELTAS)


def test_main_writes_a_record_and_prints_the_tables(study, tmp_path, capsys):
    out = tmp_path / "record.json"
    assert study.main(["--reps", "3", "--design", "constant_volatility", "--out", str(out)]) == 0
    found = json.loads(out.read_text())
    assert found["simulation_only"] is True
    assert [d["design"]["label"] for d in found["designs"]] == ["constant_volatility"]
    assert "| Design | Scenario |" in capsys.readouterr().out


# --- the committed record ---


def test_the_committed_record_is_the_codes_design(study):
    found = json.loads(RECORD.read_text(encoding="utf-8"))
    assert (found["schema"], found["schema_version"]) == (study.SCHEMA, study.SCHEMA_VERSION)
    assert found["simulation_only"] is True
    assert found["test"]["critical"] == pytest.approx(-2.44691, abs=1e-5)
    expected = [study._rounded(study.design_record(d)) for d in study.DESIGNS]
    assert [d["design"] for d in found["designs"]] == json.loads(json.dumps(expected))
    for design, record in zip(study.DESIGNS, found["designs"], strict=True):
        assert [row["shift"]["name"] for row in record["scenarios"]] == [
            shift.name for shift in design.shifts
        ]
        for row in record["scenarios"]:
            assert math.isfinite(row["true_mean"]) and 0 <= row["reject"]["rate"] <= 1
