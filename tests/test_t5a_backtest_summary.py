"""
The `backtest_summary` health input (agreed v1.0) produced from the approved
engine, and the synthetic-baseline script that writes it and attributes the
engine change (`scripts/run_t5a_baselines.py`).
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_agent.health.backtest import backtest_checks
from stock_agent.health.model import HealthState
from stock_agent.portfolio import backtest_summary, engine
from stock_agent.portfolio.descriptive import drawdown_summary
from stock_agent.portfolio.transition import Constraints

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
NAN = float("nan")
GENERATED_AT = datetime(2026, 10, 15, 12, 0, 0, 123456, tzinfo=UTC)
SHA = "0123456789abcdef0123456789abcdef01234567"


def _load(name):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _capped_run():
    # Build at a 50% cap from cash (10 bp), A doubles (drift above the cap), then a rebalance.
    returns = np.array([[NAN] * 3, [0.0] * 3, [0.0] * 3, [1.0, 0.0, 0.0]] + [[0.0] * 3] * 3)
    constraints = Constraints(2, w_max=0.5, tau_max=0.6)
    result = engine.simulate(
        returns,
        {0: np.array([0.7, 0.3, 0.0]), 4: np.array([0.7, 0.3, 0.0])},
        lag=1,
        cost_rate=np.array([0.001, 0.001, 0.0]),
        constraints=constraints,
        initial_wealth=100.0,
    )
    return result, constraints


def _record(result, constraints, **overrides):
    arguments = {
        "constraints": constraints,
        "run_id": "20261015T120000123456Z-capped",
        "policy": "capped_test",
        "evidence_status": "synthetic_engineering_only",
        "synthetic": True,
        "generated_at": GENERATED_AT,
        "code_sha": SHA,
        "assumptions": {"cost_rate_bps": {"A": 10.0, "B": 10.0}},
        **overrides,
    }
    return backtest_summary.summary_record(result, **arguments)


def test_the_measures_come_from_the_engine_steps():
    result, constraints = _capped_run()
    measures = backtest_summary.measures(result, constraints.cash_index)
    build, rebalance = (step for step in result.steps if step.turnover_one_way > 0)
    assert measures["turnover"] == max(build.turnover_one_way, rebalance.turnover_one_way)
    assert measures["turnover"] <= constraints.tau_max  # the build is turnover-limited too
    assert measures["costs_bps"] == pytest.approx(1e4 * (build.cost + rebalance.cost))
    assert measures["max_weight"] > constraints.w_max  # drift above the cap is reported
    dated = pd.Series(
        result.wealth, index=pd.bdate_range("2026-01-05", periods=len(result.wealth)), dtype=float
    )
    expected = drawdown_summary(dated).max_drawdown  # the approved F6.5 implementation
    assert measures["max_drawdown"] == pytest.approx(expected, abs=1e-15)


def test_max_weight_reports_drift_up_to_the_last_close():
    # A doubles into the final close: the last drifted weights carry the largest weight.
    returns = np.array([[NAN] * 3, [0.0] * 3, [0.0] * 3, [1.0, 0.0, 0.0]])
    result = engine.simulate(
        returns,
        {0: np.array([0.5, 0.3, 0.2])},
        lag=1,
        cost_rate=np.zeros(3),
        constraints=Constraints(2, w_max=0.5),
        initial_wealth=1.0,
    )
    measured = backtest_summary.measures(result, cash_index=2)["max_weight"]
    assert measured == pytest.approx(1.0 / 1.5, abs=1e-15)


def test_drawdown_is_unavailable_over_an_incomplete_path():
    assert backtest_summary.max_drawdown([100.0, None, 103.0]) is None
    assert backtest_summary.max_drawdown([100.0]) is None
    assert backtest_summary.max_drawdown([100.0, 110.0, 99.0, 105.0]) == pytest.approx(0.1)


def test_the_record_follows_the_agreed_v1_format():
    result, constraints = _capped_run()
    record = _record(result, constraints)
    assert {
        "artifact",
        "schema_version",
        "generated_at",
        "code_sha",
        "synthetic",
        "run_id",
        "policy",
        "evidence_status",
        "transition_status",
        "constraints",
        "measures",
    } <= set(record)
    assert record["artifact"] == "backtest_summary" and record["schema_version"] == "1.0"
    assert record["generated_at"] == "2026-10-15T12:00:00.123456Z"
    assert record["transition_status"] == "approved"
    assert record["constraints"] == {
        "max_weight": 0.5,
        "max_turnover": 0.6,
        "max_cost_bps": None,
        "max_drawdown": None,
    }
    assert set(record["measures"]) == {"turnover", "costs_bps", "max_weight", "max_drawdown"}
    assert set(record["units"]) == set(record["measures"])
    assert record["assumptions"]["cash_return"] == engine.CASH_RETURN_LABEL
    json.dumps(record, allow_nan=False)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"run_id": "../escape"}, "run_id"),
        ({"code_sha": "abc"}, "code_sha"),
        ({"policy": " "}, "policy"),
        ({"generated_at": GENERATED_AT.replace(tzinfo=None)}, "timezone"),
        ({"max_cost_bps": -1.0}, "max_cost_bps"),
    ],
)
def test_a_malformed_record_is_refused(override, match):
    result, constraints = _capped_run()
    with pytest.raises(ValueError, match=match):
        _record(result, constraints, **override)


def test_the_health_reader_accepts_the_file_and_judges_only_declared_limits(tmp_path):
    result, constraints = _capped_run()
    path = backtest_summary.write_summary(_record(result, constraints), tmp_path)
    assert path == tmp_path / "backtest_summary" / "20261015T120000123456Z-capped.json"
    with pytest.raises(FileExistsError):
        backtest_summary.write_summary(_record(result, constraints), tmp_path)
    checks = {
        check.check_id: check
        for check in backtest_checks(tmp_path / "backtest_summary", allow_synthetic=True)
    }
    assert checks["portfolio.concentration"].state is HealthState.FAILED  # drift above the cap
    assert checks["portfolio.concentration"].reason == "constraint_breached"
    assert checks["portfolio.turnover"].state is HealthState.HEALTHY
    assert checks["portfolio.costs"].reason == "no_limit_declared"
    assert checks["portfolio.drawdown"].reason == "no_limit_declared"
    assert all(check.reason != "unapproved_transition" for check in checks.values())


# --- the synthetic-baseline script ---

# The pre-T5a `run_rebalanced_backtest` on the script's synthetic panel, computed with the
# code at 8b5e564 before the migration: (final wealth, two-way turnover, max drawdown).
# Every baseline is checked here; the script reports the attribution of one (equal weight).
LEGACY = {
    "equal_weight": (125362.25125292479, 1.3389310694121508, 0.11979373311982477),
    "inverse_volatility": (129512.53655693728, 4.133904549548478, 0.08041197275873457),
    "momentum": (129971.58307458991, 35.52441840956754, 0.1442881272158434),
}


@pytest.fixture(scope="module")
def script():
    return _load("run_t5a_baselines")


@pytest.fixture(scope="module")
def frame(script):
    return script.synthetic_feature_frame()


def _rows(script, frame, name):
    return pd.DataFrame(script.attribution(frame, name)).set_index("step")


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_the_attribution_starts_from_the_exact_legacy_numbers(script, frame, name):
    final, two_way, drawdown = LEGACY[name]
    legacy = _rows(script, frame, name).loc["legacy"]
    assert legacy["terminal_wealth_ratio"] * script.INITIAL_WEALTH == pytest.approx(
        final, rel=1e-12
    )
    # The legacy engine reported two-way turnover and gave the initial portfolio for free.
    assert 2 * (legacy["total_turnover_one_way"] - 1.0) == pytest.approx(two_way, rel=1e-12)
    returns, targets = script.baseline_frames(frame)[name]
    result = script.variant(
        returns,
        targets,
        lag=0,
        transition=script.free_first_trade(script.retired_legacy_transition),
        offset=script.LEGACY_OFFSETS[name],
    )
    assert result.max_drawdown == pytest.approx(drawdown, rel=1e-12)


def test_the_reported_attribution_is_one_baseline_ending_at_its_production_run(script, frame):
    rows = script.attribution(frame)
    assert {row["baseline"] for row in rows} == {script.ATTRIBUTED} == {"equal_weight"}
    assert [row["step"] for row in rows[:4]] == ["legacy", "lag", "cost_base", "build"]
    production = script.run_baselines(frame)[script.ATTRIBUTED]
    build = _rows(script, frame, script.ATTRIBUTED).loc["build"]
    assert build["terminal_wealth_ratio"] == pytest.approx(
        production.final_wealth / production.initial_wealth, rel=1e-15
    )


@pytest.mark.parametrize("name", sorted(LEGACY))
def test_each_step_changes_one_thing(script, frame, name):
    rows = _rows(script, frame, name)
    legacy, lag = rows.loc["legacy"], rows.loc["lag"]
    # D1: the first valued close moves one session later (the build executes a close later).
    assert pd.Timestamp(lag["first_valued_date"]) > pd.Timestamp(legacy["first_valued_date"])
    # D3: the build adds exactly its own charge, 10 bp per unit of risky notional bought.
    added = rows.loc["build", "costs_bp"] - rows.loc["cost_base", "costs_bp"]
    assert 0.0 < added <= 10.0 + 1e-9
    # D2 sensitivity: costs scale with the multiplier, everything else is fixed.
    zero, one, two = (rows.loc[f"cost_{m}x"] for m in ("0", "1", "2"))
    assert zero["costs_bp"] == 0.0 and two["costs_bp"] == pytest.approx(2 * one["costs_bp"])
    assert zero["total_turnover_one_way"] == one["total_turnover_one_way"]
    assert (
        zero["terminal_wealth_ratio"] > one["terminal_wealth_ratio"] > two["terminal_wealth_ratio"]
    )


def test_the_production_baselines_share_one_build(script, frame):
    results = script.run_baselines(frame)
    build = script.common_build_date(frame)
    assert {run.build_date for run in results.values()} == {build}
    assert len({run.history.wealth.index[0] for run in results.values()}) == 1


def test_the_script_writes_one_readable_summary_per_baseline(script, tmp_path):
    ticks = iter(range(10))

    def clock():
        return GENERATED_AT.replace(microsecond=next(ticks))

    assert (
        script.main(
            ["--health-inputs-root", str(tmp_path)],
            clock=clock,
            code_version=lambda: {"git_sha": SHA, "git_dirty": False},
        )
        == 0
    )
    files = sorted((tmp_path / "backtest_summary").glob("*.json"))
    assert [path.name.split("-", 1)[1] for path in files] == [
        "equal_weight_every_5.json",
        "inverse_volatility_every_5.json",
        "momentum_20_every_5.json",
    ]
    records = [json.loads(path.read_text()) for path in files]
    assert len({record["run_id"] for record in records}) == 3
    assert len({record["generated_at"] for record in records}) == 3  # each stamped when written
    assert len({record["assumptions"]["build_date"] for record in records}) == 1
    for record in records:
        assert record["synthetic"] is True and record["code_sha"] == SHA
        assert record["evidence_status"] == "synthetic_engineering_only"
        assert record["transition_status"] == "approved"
        assert record["constraints"]["max_weight"] == 1.0
        assert record["constraints"]["max_turnover"] is None
        assert all(v is None or math.isfinite(v) for v in record["measures"].values())
        assert record["measures"]["turnover"] == pytest.approx(1.0)  # the build is a full switch
    checks = backtest_checks(tmp_path / "backtest_summary", allow_synthetic=True)
    assert {check.state for check in checks} <= {
        HealthState.HEALTHY,
        HealthState.INSUFFICIENT_EVIDENCE,
    }


def test_no_source_module_imports_or_reimplements_the_retired_legacy_step():
    import ast

    root = Path(__file__).resolve().parents[1]
    retired = root / "src" / "stock_agent" / "environment" / "portfolio_math.py"  # unimported
    for path in (root / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom):
                assert "run_t5a_baselines" not in (node.module or ""), path
            elif isinstance(node, ast.Import):
                assert all("run_t5a_baselines" not in a.name for a in node.names), path
            elif isinstance(node, ast.FunctionDef):
                assert "legacy_transition" not in node.name, path
        if path != retired:
            # The attribution step's own formulas: the cash leg on two-way turnover, additive.
            assert "np.max(cost_rate)) * two_way" not in text, path
            assert "net_return=gross - cost" not in text, path
