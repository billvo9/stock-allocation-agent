"""
Owner decision A7's simulation under the planned conditions
(stock_agent.stock_lab.planned_simulation): the pre-registered design, the
no-skill panels, the family's selection rule, the criterion, determinism and
the owner-run script.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_diagnostics.inference import normal_cdf, normal_ppf
from stock_agent.stock_lab import decision_tests as dt
from stock_agent.stock_lab import estimators as est
from stock_agent.stock_lab import evaluation, scores
from stock_agent.stock_lab import planned_simulation as ps
from stock_agent.stock_lab.contracts import DEFAULT_BAND

PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "stock_lab_a7_simulation.py"
RECORD = PROJECT / "docs" / "research" / "stock_lab_a7_simulation.json"
SMALL = {
    "years": 3,
    "warmup_years": 1,
    "sessions_per_year": 60,
    "half_lives": (21.0, 63.0),
    "fixed_b_steps": 100,
    "fixed_b_draws": 500,
    "chunk": 4,
}


def small(names: int = 6, reps: int = 8, **changes) -> ps.PlannedDesign:
    return ps.planned_design(names, reps=reps, **{**SMALL, **changes})


# --- the design ---


def test_the_planned_design_is_the_evaluations_own_settings():
    design = ps.planned_design(30)
    config = evaluation.EvaluationConfig()
    assert ps.UNIVERSE_SIZES == (4, 6, 10, 30)
    assert design.half_lives == evaluation.H_GRID
    assert (design.alpha_prior, design.kappa) == (config.alpha_prior, config.kappa)
    assert design.tau == DEFAULT_BAND.tau
    # The packet's pre-registered level is 5% and stays so; the owner confirmed 2.5% for the
    # evaluation (decision A7), which the packet simulated in review (--set nominal=0.025).
    assert design.nominal == ps.NOMINAL == 0.05 and evaluation.ALPHA == 0.025
    assert design.first == evaluation.MIN_TRAIN_SESSIONS
    assert design.years == 2024 - 2006 + 1 and design.dates == 4788
    assert design.challengers == 4  # B0, B3, B2 and M1 against B1
    assert (design.reps, design.seed) == (4000, ps.SEED + 30)
    assert ps.SIMPLER_FIRST == tuple(
        ["fold_block_t", "nw_lag10", "nw_fixed_b_sqrt", "nw_fixed_b_tenth"]
        + ["cbb_block20", "sb_politis_white"]
    )
    assert set(ps.SIMPLER_FIRST) == set(dt.PROCEDURES)


def test_late_listings_follow_the_pre_registered_schedule():
    assert ps.listing_dates(4, 4788).tolist() == [0, 0, 0, 0]
    assert ps.listing_dates(6, 4788).tolist() == [0] * 5 + [2394]
    assert ps.listing_dates(10, 4788).tolist() == [0] * 7 + [1197, 2394, 3591]
    late = ps.listing_dates(30, 4788)
    assert (late[:21] == 0).all() and late[21:].tolist() == [k * 4788 // 10 for k in range(1, 10)]
    assert ps.listing_dates(3, 100).tolist() == [0, 0, 0]


def test_returns_have_the_planned_volatility_correlation_and_clustering():
    rng = np.random.RandomState(0)
    blocks = ps.garch_blocks(
        rng, 2, 20_000, 4, correlation=0.5, daily_vol=0.025, t_df=5.0, arch=0.06, garch=0.92
    )
    returns = np.concatenate(list(blocks), axis=1)
    assert returns.shape == (2, 20_000, 4)
    assert returns.std() == pytest.approx(0.025, rel=0.1)
    pairs = [np.corrcoef(returns[0, :, 0], returns[0, :, j])[0, 1] for j in (1, 2, 3)]
    assert np.mean(pairs) == pytest.approx(0.5, abs=0.05)
    squared = returns[0, :, 0] ** 2
    assert np.corrcoef(squared[1:], squared[:-1])[0, 1] > 0.05  # volatility clusters
    signs = np.sign(returns[0, :, 0])
    assert abs(np.corrcoef(signs[1:], signs[:-1])[0, 1]) < 0.03  # direction does not


def test_the_shadow_differentials_are_the_estimators_own_scores():
    """RPS on the five daily targets of B1 read from each shadow's history
    minus the stock's own B1, averaged over the stocks present on the date
    (the second stock enters at date 40); computed with est.b1 and scores.rps
    row by row."""

    rng = np.random.RandomState(3)
    names, challengers, sessions, first, half_life = 2, 2, 140, 30, 21.0
    total = names * (1 + challengers)
    codes = rng.choice(3, size=(1, sessions, total), p=[0.4, 0.2, 0.4])
    panel = pd.DataFrame(
        {
            "symbol": np.tile([f"S{i}" for i in range(total)], sessions),
            "session": np.repeat(pd.bdate_range("2020-01-01", periods=sessions, tz="UTC"), total),
            "category_code": codes[0].reshape(-1).astype(float),
        }
    )
    b1, _ = est.b1(panel, est.FrequencyParams(alpha=50.0, half_life=half_life, kappa=40.0))
    b1 = b1.reshape(sessions, total, 3)
    entry = np.array([0, 40])
    expected = []
    for t in range(first, sessions - 5):
        row = []
        for j in range(challengers):
            per_stock = []
            for i in range(names):
                if t - first < entry[i]:
                    continue
                shadow = names + i * challengers + j
                values = []
                for h in range(1, 6):
                    y = np.array([codes[0, t + h, i]])
                    values.append(
                        scores.rps(b1[t, shadow][None], y)[0] - scores.rps(b1[t, i][None], y)[0]
                    )
                per_stock.append(np.mean(values))
            row.append(np.mean(per_stock))
        expected.append(row)
    got = ps.shadow_differentials(
        codes,
        (half_life,),
        names=names,
        challengers=challengers,
        entry=entry,
        alpha_prior=50.0,
        kappa=40.0,
        first=first,
    )
    assert got.shape == (1, 1, sessions - 5 - first, challengers)
    np.testing.assert_allclose(got[0, 0], expected, rtol=1e-10, atol=1e-14)


# --- the family's selection ---


def _reference_selection(t_own, t_pair, pairs):
    """Holm (scores.holm) on one-sided normal p-values, then evaluation._select."""

    p = {j: normal_cdf(value) for j, value in enumerate(t_own)}
    adjusted = scores.holm(p)
    winners = sorted(j for j, value in enumerate(t_own) if value < 0 and adjusted[j] < 0.05)
    if not winners:
        return -1
    current = winners[0]
    for challenger in winners[1:]:
        if normal_cdf(t_pair[pairs.index((challenger, current))]) < 0.05:
            current = challenger
    return current


def test_selection_is_holm_then_the_simplest_first_rule():
    rng = np.random.RandomState(9)
    reps, challengers = 3000, 4
    pairs = ps._pairs(challengers)
    levels = dt.holm_levels(0.05, challengers)
    normal = np.array([normal_ppf(q) for q in levels])
    own = dt.Statistics(
        rng.normal(-1.8, 1.2, (reps, challengers)),
        np.ones((reps, challengers)),
        np.broadcast_to(normal, (reps, challengers, len(levels))),
    )
    pair = dt.Statistics(
        rng.normal(-1.0, 1.5, (reps, len(pairs))),
        np.ones((reps, len(pairs))),
        np.broadcast_to(normal, (reps, len(pairs), len(levels))),
    )
    for shifts in (np.zeros(challengers), np.array([0.0, 0.0, 0.0, 0.7])):
        got = ps.select(own, pair, shifts)
        pair_shift = np.array([shifts[k] - shifts[j] for k, j in pairs])
        for rep in range(reps):
            want = _reference_selection(own.mean[rep] - shifts, pair.mean[rep] - pair_shift, pairs)
            assert got[rep] == want, rep
    assert {-1, 0, 1, 2, 3} <= set(got.tolist())  # every outcome is exercised


# --- the simulation ---


def test_the_simulation_is_seeded_and_repeats_exactly():
    first = ps.simulate_planned(small())
    again = ps.simulate_planned(small())
    assert json.dumps(first, sort_keys=True) == json.dumps(again, sort_keys=True)
    other = ps.simulate_planned(ps.PlannedDesign(names=6, reps=8, seed=1, **SMALL))
    assert json.dumps(first["cells"]) != json.dumps(other["cells"])
    json.dumps(first, allow_nan=False)  # strict JSON
    assert first["design"]["listing_dates"] == [0, 0, 0, 0, 0, 90]
    assert [(c["half_life"], c["procedure"]) for c in first["cells"]] == [
        (h, name) for h in (21.0, 63.0) for name in dt.PROCEDURES
    ]
    for cell in first["cells"]:
        assert [p["delta"] for p in cell["power_own"]] == [0.0005, 0.001]
        assert [(f["delta"], f["true"]) for f in cell["family"]] == [
            (0.0, None),
            (0.0005, 0),
            (0.0005, 3),
            (0.001, 0),
            (0.001, 3),
        ]
        assert cell["size_own"]["rate"] <= cell["power_own"][0]["rate"]


def test_the_nulls_have_mean_zero():
    record = ps.simulate_planned(small(names=4, reps=24))
    for row in record["null_mean"]:
        assert abs(row["mean"]) < 4 * row["se"], row


# --- the pre-registered criterion ---


def _cells(spec):
    return [
        {
            "procedure": name,
            "names": 4,
            "half_life": 63.0,
            "size_own": {"rate": own},
            "size_symmetric": {"rate": symmetric},
            "power_own": [{"delta": 0.0005, "rate": power}, {"delta": 0.001, "rate": 1.0}],
        }
        for name, (own, symmetric, power) in spec.items()
    ]


BASE = {name: (0.08, 0.05, 0.9) for name in dt.PROCEDURES}


def test_the_criterion_takes_the_most_powerful_admissible_procedure():
    spec = {**BASE, "fold_block_t": (0.05, 0.05, 0.30), "nw_fixed_b_tenth": (0.055, 0.05, 0.40)}
    choice = ps.choose_procedure(_cells(spec))
    assert (
        choice["chosen"] == "nw_fixed_b_tenth" and choice["reason"] == "admissible; highest power"
    )
    assert choice["admissible"] == ["fold_block_t", "nw_fixed_b_tenth"]
    assert choice["distortion"]["nw_lag10"] == pytest.approx(0.03)


def test_within_one_point_of_power_the_smaller_distortion_wins_then_the_simpler():
    spec = {**BASE, "fold_block_t": (0.05, 0.05, 0.30), "nw_fixed_b_tenth": (0.055, 0.05, 0.309)}
    assert ps.choose_procedure(_cells(spec))["chosen"] == "fold_block_t"
    spec = {**BASE, "fold_block_t": (0.054, 0.05, 0.30), "nw_fixed_b_tenth": (0.046, 0.05, 0.305)}
    assert ps.choose_procedure(_cells(spec))["chosen"] == "fold_block_t"  # equal distortion


def test_the_symmetric_null_and_the_boundary_count():
    spec = {**BASE, "fold_block_t": (0.05, 0.065, 0.9), "cbb_block20": (0.06, 0.04, 0.2)}
    choice = ps.choose_procedure(_cells(spec))
    assert choice["admissible"] == ["cbb_block20"]  # exactly 1 pp away is admissible
    assert choice["chosen"] == "cbb_block20"


def test_with_none_admissible_the_smallest_distortion_is_taken():
    spec = {**BASE, "sb_politis_white": (0.065, 0.05, 0.1), "nw_fixed_b_sqrt": (0.035, 0.05, 0.1)}
    choice = ps.choose_procedure(_cells(spec))
    assert choice["admissible"] == []
    assert choice["chosen"] == "nw_fixed_b_sqrt"  # tied at 1.5 pp: the simpler
    assert choice["reason"] == "none admissible; smallest distortion"


def test_the_recorded_run_reproduces_its_choice_and_critical_values():
    """The committed record (docs/research/stock_lab_a7_simulation.json): the
    pre-registered design at every planned universe size, the recorded fixed-b
    critical values, and the criterion re-applied to its cells."""

    record = json.loads(RECORD.read_text(encoding="utf-8"))
    assert record["role"] == "pre-registered" and record["changed"] == {}
    assert [size["design"]["names"] for size in record["sizes"]] == list(ps.UNIVERSE_SIZES)
    for size in record["sizes"]:
        planned = ps.planned_design(size["design"]["names"])
        assert size["design"]["seed"] == planned.seed and size["design"]["reps"] == ps.REPS
        assert size["design"]["half_lives"] == list(ps.HALF_LIVES)
        for name, recorded in dt.PLANNED_FIXED_B.items():
            np.testing.assert_allclose(size["fixed_b"][name], recorded, atol=5e-5)
        assert len(size["cells"]) == len(ps.HALF_LIVES) * len(dt.PROCEDURES)
    cells = [cell for size in record["sizes"] for cell in size["cells"]]
    assert ps.choose_procedure(cells) == record["choice"]
    assert record["choice"]["chosen"] == "fold_block_t"
    # The checks run after the choice are labelled as reported only.
    sensitivity = json.loads((RECORD.parent / "stock_lab_a7_sensitivity.json").read_text("utf-8"))
    assert sensitivity["role"] == "sensitivity (reported only)"
    assert sensitivity["changed"] == {"arch": 0.0, "garch": 0.0}
    check = json.loads((RECORD.parent / "stock_lab_a7_evaluation_check.json").read_text("utf-8"))
    assert "reported only" in check["role"] and check["changed"] == {}
    assert [size["design"]["names"] for size in check["sizes"]] == list(ps.UNIVERSE_SIZES)
    # The review's stricter-level option: the same seeds at N = 4 and 6.
    shifts = [0.0001666667, 0.00025, 0.0003333333, 0.0005, 0.001]
    for name, changes in (
        ("stock_lab_a7_noise_power.json", {"deltas": shifts}),
        ("stock_lab_a7_stricter_level.json", {"deltas": shifts, "nominal": 0.025}),
    ):
        run = json.loads((RECORD.parent / name).read_text("utf-8"))
        assert run["role"] == "sensitivity (reported only)" and run["changed"] == changes
        for size, planned in zip(run["sizes"], record["sizes"][:2], strict=True):
            assert size["design"]["seed"] == planned["design"]["seed"]
            if "nominal" not in changes:  # extra shifts leave every size untouched
                for cell, same in zip(size["cells"], planned["cells"], strict=True):
                    assert cell["size_own"] == same["size_own"]
    strict = json.loads((RECORD.parent / "stock_lab_a7_evaluation_check_025.json").read_text())
    assert strict["changed"] == {"nominal": 0.025}
    assert all(size["design"]["nominal"] == 0.025 for size in strict["sizes"])


# --- the owner-run script ---


def test_the_script_writes_the_record_and_prints_the_tables(tmp_path):
    spec = importlib.util.spec_from_file_location("stock_lab_a7_simulation", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    out = io.StringIO()
    target = tmp_path / "a7.json"
    overrides = {key: value for key, value in SMALL.items()}
    code = script.main(
        ["--out", str(target), "--sizes", "4", "6", "--reps", "6"], out=out, overrides=overrides
    )
    assert code == 0
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["schema"] == "stock_lab_a7_simulation" and record["data_class"] == "synthetic"
    assert [size["design"]["names"] for size in record["sizes"]] == [4, 6]
    assert record["choice"]["chosen"] in dt.PROCEDURES
    text = out.getvalue()
    assert "Size, own-history null" in text and f"Chosen: `{record['choice']['chosen']}`" in text
    assert math.isfinite(record["sizes"][0]["cells"][0]["variance_ratio"])
    assert record["role"] == "sensitivity (reported only)"  # not the pre-registered sizes and reps
    sensitivity = tmp_path / "sensitivity.json"
    argv = ["--out", str(sensitivity), "--sizes", "4", "--reps", "6", "--set", "arch=0.0"]
    sets = ["garch=0.0", "deltas=0.00025,0.0005"]
    assert script.main([*argv, *sets], out=io.StringIO(), overrides=overrides) == 0
    changed = json.loads(sensitivity.read_text(encoding="utf-8"))
    assert changed["role"] == "sensitivity (reported only)"
    assert changed["changed"] == {"arch": 0.0, "garch": 0.0, "deltas": [0.00025, 0.0005]}
    assert changed["sizes"][0]["design"]["garch"] == 0.0
    cell = changed["sizes"][0]["cells"][0]
    assert [p["delta"] for p in cell["power_own"]] == [0.00025, 0.0005]
    with pytest.raises(SystemExit):
        script.main(["--out", str(sensitivity), "--set", "names=3"], out=io.StringIO())


def test_the_script_runs_the_evaluations_size_check_at_a_planned_shape(tmp_path, monkeypatch):
    """--evaluation-check: size_simulation with the evaluation's own design at
    each universe size and 19 test years (here shrunk, through size_design)."""

    spec = importlib.util.spec_from_file_location("stock_lab_a7_simulation", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    seen = []
    original = evaluation.size_design

    def shrunk(config, *, names, years):
        seen.append((names, years))
        design = original(config, names=names, years=years)
        return replace(design, years=2, reps=4, half_lives=(21.0,), bootstrap_reps=9)

    monkeypatch.setattr(evaluation, "size_design", shrunk)
    out, target = io.StringIO(), tmp_path / "check.json"
    argv = ["--evaluation-check", "--sizes", "4", "--out", str(target), "--set", "nominal=0.025"]
    assert script.main(argv, out=out) == 0
    assert seen == [(4, 19)]
    record = json.loads(target.read_text(encoding="utf-8"))
    assert record["sizes"][0]["status"] == "ok" and "reported only" in record["role"]
    assert record["changed"] == {"nominal": 0.025}
    assert record["sizes"][0]["design"]["nominal"] == 0.025
    with pytest.raises(SystemExit):  # only the level can change there
        script.main(["--evaluation-check", "--out", str(target), "--set", "reps=3"], out=out)
    assert "other_vs_own" in out.getvalue() and "`fold_block_t`" in out.getvalue()
