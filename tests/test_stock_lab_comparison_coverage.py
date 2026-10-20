"""
The coverage of the briefing's matched-comparison interval, per score, and
how often its persistence flag (SE40 / SE20 above the threshold) fires: a
smaller version of scripts/stock_lab_comparison_coverage.py (synthetic series
only).
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from stock_agent.stock_lab import comparison_coverage as cc
from stock_agent.stock_lab.briefing.reliability import PERSISTENCE_THRESHOLD, block_t_interval

SMALL = cc.CoverageDesign(
    reps=4000,
    score_reps=200,
    truth_reps=200,
    sizes=(100, 101, 119, 120, 200),
    persistence=(0.0, 0.5, 0.9),
    panel_persistence=(0.5,),
    breaks=(1.0,),
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


def _score_cell(found, series, target, score):
    (cell,) = [c for c in found["cells"] if (c["series"], c.get("target"), c.get("score"))
               == (f"scores:{series}", target, score)]  # fmt: skip
    return cell


def _flag(cell, dates):
    return _row(cell, dates)["persistence"]["flag_rate"][str(PERSISTENCE_THRESHOLD)]


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
    assert len(scores) == 18  # three series and three scores for each of the two targets
    assert {cell["score"] for cell in scores} == {"rps", "brier", "log_loss"}
    for cell in scores:
        for row in cell["sizes"]:
            coverage = row["current"]["coverage"]
            if row["dates"] < 120:
                assert 0.84 <= coverage <= 0.98, (cell["series"], cell["score"], row["dates"])
            else:
                # From 120 dates, B1's learning from the sessions it is scored on makes the
                # own-history and volatility cells conservative (99% here; the record shows
                # it), so only the symmetric placebo keeps an upper bound.
                assert coverage >= 0.84, (cell["series"], cell["score"], row["dates"])
                if cell["series"] == "scores:placebo":
                    assert coverage <= 0.95, (cell["score"], row["dates"])


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
    by = {(c["series"], c["target"]): c["autocorrelation"] for c in found["cells"]
          if c.get("score") == "rps"}  # fmt: skip
    assert by["scores:placebo", "cum"]["1"] > 0.4  # neighbouring totals share four sessions
    assert abs(by["scores:placebo", "1"]["1"]) < 0.1
    assert abs(by["scores:placebo", "cum"]["20"]) < 0.1


def test_log_loss_differences_are_skewed_and_the_symmetric_placebo_is_not(found):
    assert _score_cell(found, "volatility", "1", "log_loss")["skewness"] > 0.2
    for score in ("rps", "brier", "log_loss"):
        assert abs(_score_cell(found, "placebo", "1", score)["skewness"]) < 0.1, score


def test_the_flag_is_reported_only_from_three_blocks_of_forty(found):
    (cell,) = [c for c in _cells(found, "ar1") if c["phi"] == 0.0]
    for dates in (100, 101, 119):
        assert _row(cell, dates)["persistence"] is None
    found_120 = _row(cell, 120)["persistence"]
    assert found_120["long_blocks"] == 3 and set(found_120["flag_rate"]) == {"1.2", "1.3"}
    assert 0.9 < found_120["median_ratio"] < 1.05


def test_the_flag_seldom_fires_on_independent_dates_or_the_symmetric_placebo(found):
    (iid,) = [c for c in _cells(found, "ar1") if c["phi"] == 0.0]
    assert _flag(iid, 120) <= 0.21 and _flag(iid, 200) <= 0.11  # 18% and 8% in the record
    placebo = _score_cell(found, "placebo", "1", "rps")
    assert _flag(placebo, 120) <= 0.27 and _flag(placebo, 200) <= 0.16  # 200 records only
    # The threshold is a real trade-off: the 1.2 starting point fires far more often.
    assert _row(iid, 200)["persistence"]["flag_rate"]["1.2"] >= 0.15


def test_the_flag_fires_more_often_under_persistence_and_after_a_shift(found):
    (iid,) = [c for c in _cells(found, "ar1") if c["phi"] == 0.0]
    (persistent,) = [c for c in _cells(found, "ar1") if c["phi"] == 0.9]
    (shift,) = _cells(found, "break")
    for dates in (120, 200):
        assert _flag(persistent, dates) >= _flag(iid, dates) + 0.07
        assert _flag(shift, dates) >= _flag(iid, dates) + 0.2
    assert _flag(persistent, 200) <= 0.35  # a weak signal: most persistent records pass


def test_a_mid_record_shift_leaves_the_interval_conservative_for_the_period_mean(found):
    (shift,) = _cells(found, "break")
    assert shift["delta"] == 1.0
    for row in shift["sizes"]:
        assert row["current"]["coverage"] >= 0.95, row["dates"]


def test_the_shift_moves_the_mean_at_half_the_dates_and_keeps_its_mean():
    for dates in (100, 101, 119):
        step = cc.mid_record_step(dates)
        assert step.mean() == pytest.approx(0.0, abs=1e-12)
        assert step[dates // 2] - step[dates // 2 - 1] == pytest.approx(1.0)
        assert len(np.unique(step)) == 2


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
        persistence=(0.0,), panel_persistence=(), breaks=(0.5,), warmup=30, chunk=20,
    )  # fmt: skip
    monkeypatch.setattr(script.cc, "CoverageDesign", lambda: tiny)
    out, stdout = tmp_path / "record.json", io.StringIO()
    assert script.main(["--out", str(out)], stdout=stdout) == 0
    written = json.loads(out.read_text())
    assert (written["schema"], written["data_class"]) == (script.SCHEMA, "synthetic")
    assert "never a decision test" in written["use"]
    assert "| ar1 φ = 0.0 |" in stdout.getvalue()
    assert "| scores:placebo log_loss (target 1) |" in stdout.getvalue()
    assert f"SE40 / SE20 above {PERSISTENCE_THRESHOLD}" in stdout.getvalue()
    assert "| break δ = 0.5 |" in stdout.getvalue()
    assert written["schema_version"] == "1.1"
