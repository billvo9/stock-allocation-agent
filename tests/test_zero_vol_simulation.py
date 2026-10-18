"""
The zero-volatility simulation study (scripts/research/, SIMULATION ONLY) is
deterministic, so the evidence in docs/research/zero_vol_floor.md section 5
can be reproduced, and its floor agrees with the production formula.
"""

from __future__ import annotations

import importlib.util
import json
import math

import pandas as pd
import pytest
from forecast_prospective_helpers import REPO

from stock_agent.forecasting.prospective import zero_vol

RESEARCH = REPO / "scripts" / "research"
SCENARIOS = ("P110.30_mu0.0pct_s0.25bp", "P50.00_mu0.5pct_s0.25bp")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"research_{name}", RESEARCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def simulation():
    return _load("zero_vol_simulation")


def test_the_simulation_is_labelled_and_reads_nothing(simulation):
    for name in (
        "zero_vol_simulation",
        "zero_vol_simulation_tables",
        "zero_vol_simulation_examples",
    ):
        text = (RESEARCH / f"{name}.py").read_text(encoding="utf-8")
        assert "SIMULATION ONLY" in text
        for forbidden in ("stock_agent", "requests", "urllib", "read_parquet", "data/"):
            assert forbidden not in text, (name, forbidden)
    assert set(SCENARIOS) <= set(simulation.scenarios())
    assert "B2_asof" in simulation.RULES


def test_the_simulation_is_deterministic(simulation, tmp_path):
    for run in ("first", "second"):
        simulation.main(tmp_path / run, seeds=(0, 1), names=SCENARIOS)
    for name in ("sim_results_by_seed.csv", "sim_summary.csv"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    run = json.loads((tmp_path / "first" / "sim_run.json").read_text())
    assert run["seeds"] == [0, 1] and run["scenarios"] == sorted(SCENARIOS)
    simulation.main(tmp_path / "other", seeds=(2,), names=SCENARIOS)
    first = pd.read_csv(tmp_path / "first" / "sim_results_by_seed.csv")
    other = pd.read_csv(tmp_path / "other" / "sim_results_by_seed.csv")
    assert not first["coverage"].head(7).equals(other["coverage"].head(7))  # seeds matter


def test_the_simulated_rules_behave_as_described(simulation, tmp_path):
    simulation.main(tmp_path, seeds=(0,), names=SCENARIOS[:1])
    frame = pd.read_csv(tmp_path / "sim_results_by_seed.csv")
    ok = frame.loc[frame["status"] == "ok"].set_index(["h", "rule"])
    # Rule A scores measured zeros as misses: its coverage on zero rows is 0.
    zero_rows = ok.xs("A", level="rule")["coverage_zero_rows"].dropna()
    assert len(zero_rows) and (zero_rows == 0.0).all()
    # Rule C drops zero rows from scoring, so its coverage is its non-zero coverage.
    c = ok.xs("C", level="rule")
    assert (c["coverage"] == c["coverage_nonzero_rows"]).all()
    _load("zero_vol_simulation_tables").main(tmp_path)
    assert (tmp_path / "sim_tables.txt").read_text().startswith("\n## Zero-window frequency")


def test_the_simulations_floor_is_the_production_floor(tmp_path):
    examples = _load("zero_vol_simulation_examples")
    path = tmp_path / "examples.json"
    examples.main(path)
    for row in json.loads(path.read_text()):
        production = float(zero_vol.sigma_floor(row["P_tilde"], row["h"]))
        assert math.isclose(row["B2_sqrt_h"]["floor_bp"], production * 1e4, rel_tol=1e-12)
