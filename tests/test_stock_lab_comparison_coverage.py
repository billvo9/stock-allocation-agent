"""
The coverage of the briefing's matched-comparison interval: a smaller version
of scripts/stock_lab_comparison_coverage.py (synthetic series only).
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from stock_agent.stock_lab import comparison_coverage as cc
from stock_agent.stock_lab.briefing.reliability import block_t_interval

SMALL = cc.CoverageDesign(
    reps=4000,
    score_reps=200,
    truth_reps=200,
    sizes=(100, 101, 119),
    persistence=(0.0, 0.5, 0.9),
    panel_persistence=(0.5,),
    warmup=252,
    chunk=200,
)


@pytest.fixture(scope="module")
def found():
    return cc.simulate(SMALL)


def _cells(found, series):
    return [cell for cell in found["cells"] if cell["series"] == series]


def _row(cell, dates):
    return next(row for row in cell["sizes"] if row["dates"] == dates)


def test_the_interval_is_unchanged_by_the_mean_and_scale_of_the_series():
    values = np.random.default_rng(1).normal(size=130)
    plain, moved = block_t_interval(values), block_t_interval(3.0 + 0.01 * values)
    assert float(moved.low) == pytest.approx(3.0 + 0.01 * float(plain.low))
    assert float(moved.high) == pytest.approx(3.0 + 0.01 * float(plain.high))


def test_the_ar1_series_has_unit_variance_and_its_persistence():
    series = cc.ar1(np.random.RandomState(2), 2000, 300, 0.5, t_df=5.0)
    assert series[:, 0].var() == pytest.approx(1.0, abs=0.1)
    assert series[:, -1].var() == pytest.approx(1.0, abs=0.1)
    assert cc.autocorrelations(series, (1,))["1"] == pytest.approx(0.5, abs=0.03)


def test_the_previous_rule_matches_the_current_one_on_whole_blocks():
    values = np.random.default_rng(3).normal(size=(4, 140))
    current, previous = block_t_interval(values), cc.previous_interval(values)
    assert current.blocks == previous.blocks == 7
    assert current.se == pytest.approx(previous.se)


def test_the_interval_covers_about_ninety_percent_when_dependence_is_short(found):
    for cell in _cells(found, "ar1") + _cells(found, "panel"):
        if cell["phi"] == 0.9:
            continue
        for row in cell["sizes"]:
            assert 0.85 <= row["current"]["coverage"] <= 0.95, (cell["series"], cell["phi"], row)


def test_realistic_score_differences_keep_the_coverage(found):
    scores = [cell for cell in found["cells"] if cell["series"].startswith("scores:")]
    assert len(scores) == 6  # three series for each of the two targets
    for cell in scores:
        for row in cell["sizes"]:
            assert 0.84 <= row["current"]["coverage"] <= 0.98, (cell["series"], row)


def test_strong_persistence_makes_the_interval_too_narrow(found):
    # The documented limit: blocks of 20 dates miss dependence that lasts longer.
    (cell,) = [c for c in _cells(found, "ar1") if c["phi"] == 0.9]
    for row in cell["sizes"]:
        assert row["current"]["coverage"] < 0.85
        assert row["current"]["se_ratio"] < 0.8


def test_one_date_past_whole_blocks_no_longer_widens_the_interval(found):
    (cell,) = [c for c in _cells(found, "ar1") if c["phi"] == 0.0]
    at_100, at_101 = _row(cell, 100), _row(cell, 101)
    current, previous = at_100["current"], at_100["previous"]
    assert (current["blocks"], current["coverage"]) == (previous["blocks"], previous["coverage"])
    assert current["median_width"] == pytest.approx(previous["median_width"])
    assert (at_101["current"]["blocks"], at_101["previous"]["blocks"]) == (5, 6)
    assert at_101["previous"]["coverage"] >= at_101["current"]["coverage"] + 0.025
    assert at_101["previous"]["median_width"] > 1.1 * at_101["current"]["median_width"]
    assert at_101["current"]["median_width"] < 1.05 * at_100["current"]["median_width"]


def test_the_placebos_have_a_true_mean_of_zero(found):
    for cell in found["cells"]:
        if cell["series"] in ("scores:placebo", "scores:own_history"):
            assert abs(cell["truth_run_mean"]) <= 4 * cell["truth_run_mc_se"], cell["series"]


def test_the_five_session_total_overlaps_and_the_next_session_does_not(found):
    by = {(c["series"], c["target"]): c["autocorrelation"] for c in found["cells"] if "target" in c}
    assert by["scores:placebo", "cum"]["1"] > 0.4  # neighbouring totals share four sessions
    assert abs(by["scores:placebo", "1"]["1"]) < 0.1
    assert abs(by["scores:placebo", "cum"]["20"]) < 0.1


def test_sizes_below_one_hundred_dates_are_refused():
    with pytest.raises(ValueError):
        cc.simulate(cc.CoverageDesign(sizes=(99,)))


def test_the_script_writes_a_synthetic_record_and_a_summary(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "stock_lab_comparison_coverage.py"
    spec = importlib.util.spec_from_file_location("stock_lab_comparison_coverage", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    tiny = cc.CoverageDesign(
        reps=50, score_reps=20, truth_reps=20, sizes=(100, 101, 119, 120, 200, 300, 500),
        persistence=(0.0,), panel_persistence=(), warmup=30, chunk=20,
    )  # fmt: skip
    monkeypatch.setattr(script.cc, "CoverageDesign", lambda: tiny)
    out, stdout = tmp_path / "record.json", io.StringIO()
    assert script.main(["--out", str(out)], stdout=stdout) == 0
    written = json.loads(out.read_text())
    assert (written["schema"], written["data_class"]) == (script.SCHEMA, "synthetic")
    assert "never a decision test" in written["use"]
    assert "| ar1 φ = 0.0 |" in stdout.getvalue()
