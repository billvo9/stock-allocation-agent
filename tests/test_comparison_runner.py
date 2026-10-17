"""
The registered comparison end to end with a light learned kind behind the
same seam as PPO and CEM: the frozen selection, the artifact and its labels,
every strategy, scenario and seed reported, cost sensitivity, build
reporting, determinism, the write-once evaluation log, the test fold never
built while anything is chosen, and no selection on test results.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
from comparison_helpers import (
    CODE,
    FAKE_KINDS,
    FakeKind,
    FixedClock,
    small_config,
    tiny_config,
    tiny_record,
)

from stock_agent.comparison import report, runner
from stock_agent.comparison.fairness import SelectionError, check_no_test_selection
from stock_agent.comparison.ledger import EvaluationLog, LogUnreadable, RepeatLookError
from stock_agent.comparison.protocol import ProtocolError, comparison_config_from_dict
from stock_agent.health.backtest import backtest_checks
from stock_agent.health.model import HealthState
from stock_agent.health.policy import policy_checks
from stock_agent.portfolio.engine import CASH_RETURN_LABEL
from stock_agent.registry.stages import STAGES
from stock_agent.rl.scenarios import generate_scenario
from stock_agent.rl.splits import LeakageError

RULES = ("buy_and_hold", "equal_weight", "fixed_balanced", "inverse_volatility")
RULES += ("mean_variance", "all_shv")
LEARNED = ("linear_softmax_cem", "ppo", "ppo_guarded")
EPISODES = 4  # two scenarios, two scenario seeds


def _train(root, *, config=None, workers=1, kinds=FAKE_KINDS, loader=generate_scenario):
    return runner.train(
        config or tiny_config(),
        run_dir=root / "run",
        code=CODE,
        clock=FixedClock(),
        kinds=kinds,
        loader=loader,
        workers=workers,
    )


def _evaluate(
    root, *, config=None, fold="test", loader=generate_scenario, repeat=False, clock=None
):
    return runner.evaluate(
        config or _registered(root),
        run_dir=root / "run",
        fold_role=fold,
        reports_root=root / "reports",
        health_inputs_root=root / "health",
        code=CODE,
        clock=clock or FixedClock(),
        kinds=FAKE_KINDS,
        loader=loader,
        repeat_look=repeat,
    )


def _registered(root):
    """The configuration the run under `root` was trained for."""

    selection = json.loads((root / "run" / runner.SELECTION_FILE).read_text())
    return comparison_config_from_dict(selection["config"])


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison")
    selection = _train(root)
    artifact = _evaluate(root)
    return SimpleNamespace(
        root=root,
        selection=json.loads(selection.read_text()),
        document=json.loads(artifact.read_text()),
        path=artifact,
    )


RULES_ONLY = small_config("equal_weight", "all_shv")
WITH_PPO = small_config("equal_weight", "ppo")


@pytest.fixture
def trained(tmp_path):
    """A trained rules-only registration: the evaluation log's behaviour, quickly."""

    _train(tmp_path, config=RULES_ONLY)
    return tmp_path


@pytest.fixture
def trained_ppo(tmp_path):
    """A trained registration with PPO's five seeds and one rule."""

    _train(tmp_path, config=WITH_PPO)
    return tmp_path


# --- the artifact ---


def test_a_run_writes_the_selection_the_artifact_the_log_and_health_inputs(completed):
    members = [(m["constraint_set"], m["kind"], m["seed"]) for m in completed.selection["members"]]
    assert members == [
        (cs, kind, seed)
        for cs in ("plain", "strict")
        for kind in ("cem", "ppo")
        for seed in range(5)
    ]
    assert completed.path.parent.parent == completed.root / "reports"
    (look,) = EvaluationLog(completed.root / "reports" / runner.LOG_FILE).entries()
    assert (look["fold"], look["look_number"], look["repeat"]) == ("test", 1, False)
    assert look["artifact_id"] == completed.document["artifact_id"]
    health = sorted((completed.root / "health" / "backtest_summary").glob("*.json"))
    assert len(health) == 2 * (len(RULES) + len(LEARNED))


def test_the_artifact_carries_the_labels_stages_and_integrity_hash(completed):
    document = completed.document
    assert document["labels"]["scope"] == (
        "Synthetic scenarios only — not evidence of real investment performance"
    )
    assert document["labels"]["evaluation"] == "engineering evaluation"
    assert document["labels"]["cash"] == CASH_RETURN_LABEL
    assert (
        document["synthetic"] is True
        and document["evidence_status"] == "synthetic_engineering_only"
    )
    assert document["transition"]["status"] == "approved" and document["execution_lag"] == 1
    for strategy in document["strategies"]:
        assert strategy["registry_stage"] == "engineering_only"
        assert STAGES.index(strategy["registry_stage"]) <= STAGES.index("development_evaluated")
    assert document["content_sha256"] == report.content_sha256(document)
    assert document["look"]["number"] == 1 and document["fold"]["role"] == "test"


def test_no_unapproved_formula_is_reported(completed):
    assert set(completed.document["not_reported"]["measures"]) == {
        "CAGR",
        "Sharpe ratio",
        "confidence intervals",
    }
    text = json.dumps(
        {k: v for k, v in completed.document.items() if k not in ("not_reported", "config")}
    ).lower()
    for word in ("cagr", "sharpe", "interval", "annualized_return", 'volatility":'):
        assert word not in text, word


def test_every_strategy_scenario_and_seed_is_reported(completed):
    rows = completed.document["rows"]
    assert len(rows) == 2 * (len(RULES) * EPISODES + len(LEARNED) * 5 * EPISODES)
    for row in rows:
        assert set(row["cost_sensitivity"]) == {"0x", "1x", "2x"}
        assert {"terminal_wealth", "total_return", "drawdown", "turnover_one_way"} <= set(row)
        assert {"rebalance_turnover_per_year", "traded_notional", "interventions"} <= set(row)
        assert {"fallback_events", "build", "invalid_targets"} <= set(row)
        assert (row["member_seed"] is None) == (row["strategy"] in RULES)
    summary = completed.document["summary"]
    assert len(summary) == 2 * (len(RULES) + len(LEARNED)) * 2
    learned = [s for s in summary if s["strategy"] in LEARNED]
    assert {s["episodes"] for s in learned} == {5 * 2}  # every seed on every path


def test_cost_sensitivity_replays_the_same_decisions_at_zero_one_and_two_times(completed):
    for row in completed.document["rows"]:
        wealth = {k: v["terminal_wealth"] for k, v in row["cost_sensitivity"].items()}
        assert wealth["1x"] == row["terminal_wealth"]  # the 1x replay is the run itself
        assert wealth["0x"] > wealth["1x"] > wealth["2x"]  # every strategy builds at a cost


def test_learned_seeds_are_a_full_distribution_never_the_best_seed(completed):
    entries = completed.document["seed_variability"]
    assert len(entries) == 2 * len(LEARNED) * 2
    for entry in entries:
        assert entry["seeds"] == [0, 1, 2, 3, 4]
        spread = entry["distribution"]
        assert spread["n"] == 5 and len(set(entry["per_seed_mean_terminal_wealth"])) == 5
        assert spread["min"] <= spread["q1"] <= spread["median"] <= spread["q3"] <= spread["max"]
        assert spread["min"] == min(entry["per_seed_mean_terminal_wealth"])
        assert spread["max"] == max(entry["per_seed_mean_terminal_wealth"])


def test_the_build_is_reported_for_every_strategy_and_set(completed):
    builds = {(b["constraint_set"], b["strategy"]): b for b in completed.document["build"]}
    assert len(builds) == 2 * (len(RULES) + len(LEARNED))
    for (cs, _), build in builds.items():
        assert build["builds_executed"] == build["episodes"]
        assert (build["incomplete_construction"] > 0) == (cs == "strict")
    held = builds[("strict", "buy_and_hold")]
    assert held["incomplete_construction"] == held["never_fully_built"] == EPISODES
    shv = builds[("strict", "all_shv")]
    # The cap holds the fund at 50% (the excess is uninvested cash); the turnover limit
    # builds a quarter of the way first: 25% fund and 75% cash at the build close.
    assert shv["executed_weights"]["median"] == pytest.approx([0.0, 0.0, 0.25, 0.75])
    assert shv["uninvested_cash"]["median"] == pytest.approx(0.75)
    assert shv["terminal_cash_weight"]["median"] == pytest.approx(0.5, abs=1e-3)
    assert shv["never_fully_built"] == 0 and shv["sessions_until_fully_built"]["max"] > 0
    plain = builds[("plain", "all_shv")]
    assert plain["incomplete_construction"] == 0 and plain["uninvested_cash"]["max"] == 0.0


def test_the_contract_checked_every_strategy_episode_and_replay(completed):
    checked = completed.document["fairness"]["checked"]
    entries = len(RULES) + 3 * 5 * 2 // 2  # rules, then CEM, PPO and guarded PPO seeds
    assert checked["strategies"] == 2 * (len(RULES) + len(LEARNED) * 5) == 2 * entries
    assert checked["replays"] == checked["strategies"] * EPISODES * 3
    assert completed.document["fairness"]["contract"] == "passed"


def test_the_health_inputs_are_read_by_model_health(completed):
    directory = completed.root / "health" / "backtest_summary"
    checks = {c.check_id: c for c in backtest_checks(directory, allow_synthetic=True)}
    assert checks["portfolio.turnover"].state is HealthState.INSUFFICIENT_EVIDENCE  # plain: none
    assert "strict/ppo" in checks["portfolio.turnover"].detail
    record = json.loads(next(directory.glob("*-strict-all_shv.json")).read_text())
    assert record["measures"]["turnover"] <= record["constraints"]["max_turnover"] == 0.25
    # D5: drift above the cap between rebalances is allowed and reported, not sold.
    assert record["measures"]["max_weight"] > record["constraints"]["max_weight"] == 0.5
    assert record["constraint_violations"] == 0 and record["build"]["incomplete_construction"]
    states = {c.check_id: c.state for c in policy_checks(directory, None, allow_synthetic=True)}
    assert states["portfolio.violations"] is HealthState.HEALTHY


# --- determinism ---


def test_the_same_configuration_and_seeds_give_the_same_artifact(tmp_path):
    documents = []
    for name in ("first", "second"):
        root = tmp_path / name
        _train(root, config=WITH_PPO)
        documents.append(json.loads(_evaluate(root).read_text()))
    assert documents[0] == documents[1]
    assert documents[0]["content_sha256"] == documents[1]["content_sha256"]


def test_parallel_training_freezes_the_same_selection(completed, tmp_path):
    path = _train(tmp_path, workers=2)
    assert json.loads(path.read_text()) == completed.selection


# --- the write-once evaluation log ---


def test_the_test_fold_is_evaluated_once_unless_a_repeat_is_recorded(trained):
    _evaluate(trained)
    log = EvaluationLog(trained / "reports" / runner.LOG_FILE)
    with pytest.raises(RepeatLookError, match="once per registered configuration"):
        _evaluate(trained)
    assert len(log.entries()) == 1
    second = json.loads(_evaluate(trained, repeat=True).read_text())
    assert second["look"] == {
        "number": 2,
        "repeat": True,
        "log_entry_sha256": log.entries()[1]["entry_sha256"],
    }
    assert [e["look_number"] for e in log.entries()] == [1, 2]


def test_dev_looks_are_unlimited_and_counted(trained):
    for _ in range(3):
        _evaluate(trained, fold="dev")
    log = EvaluationLog(trained / "reports" / runner.LOG_FILE)
    assert [(e["fold"], e["look_number"]) for e in log.entries()] == [
        ("dev", 1),
        ("dev", 2),
        ("dev", 3),
    ]
    _evaluate(trained)  # dev looks do not use up the test look
    assert log.looks(RULES_ONLY.sha256(), "test") == 1


def test_a_failed_evaluation_still_counts_its_look(trained):
    def broken(scenario, seed):
        raise RuntimeError("the evaluation data could not be built")

    with pytest.raises(RuntimeError):
        _evaluate(trained, loader=broken)
    with pytest.raises(RepeatLookError):
        _evaluate(trained)


def test_an_edited_log_is_refused(trained):
    _evaluate(trained, fold="dev")
    path = trained / "reports" / runner.LOG_FILE
    path.write_text(path.read_text().replace('"dev"', '"test"', 1), encoding="utf-8")
    with pytest.raises(LogUnreadable, match="edited"):
        _evaluate(trained)
    path.write_text("", encoding="utf-8")
    assert EvaluationLog(path).entries() == []


# --- the test fold is never built while anything is chosen ---


def test_training_never_builds_the_test_fold(tmp_path):
    requested = []

    def spy(scenario, seed):
        requested.append(seed)
        return generate_scenario(scenario, seed)

    _train(tmp_path, loader=spy)
    plan = tiny_config().rl.plan
    assert not set(requested) & set(plan.test.seeds)
    assert set(plan.dev.seeds) <= set(requested)  # selection read the dev fold


class PeekingKind(FakeKind):
    def train(self, run_config, seed, *, loader, **kwargs):
        loader("calm", run_config.plan.test.seeds[0])
        return super().train(run_config, seed, loader=loader, **kwargs)


def test_a_trainer_that_reads_the_test_fold_is_refused(tmp_path):
    with pytest.raises(LeakageError, match="never built"):
        _train(tmp_path, kinds={"ppo": PeekingKind("ppo"), "cem": FakeKind("cem")})


def test_the_look_is_counted_before_any_evaluation_data_exists(trained):
    log = EvaluationLog(trained / "reports" / runner.LOG_FILE)
    seen = []

    def spy(scenario, seed):
        seen.append(log.looks(RULES_ONLY.sha256(), "test"))
        return generate_scenario(scenario, seed)

    _evaluate(trained, loader=spy)
    assert seen and set(seen) == {1}


# --- no selection on test results ---


def test_the_reported_members_are_exactly_the_frozen_dev_selected_ones(completed):
    assert check_no_test_selection(completed.document, completed.selection) == 20


def _best_seed(document, strategy="ppo"):
    rows = [r for r in document["rows"] if r["strategy"] == strategy]
    by_seed = {}
    for row in rows:
        by_seed.setdefault(row["member_seed"], []).append(row["terminal_wealth"])
    return max(by_seed, key=lambda seed: sum(by_seed[seed]))


def _only_best(document):
    best = _best_seed(document)
    edited = copy.deepcopy(document)
    edited["rows"] = [
        r for r in edited["rows"] if r["strategy"] != "ppo" or r["member_seed"] == best
    ]
    return edited


def _best_distribution(document):
    edited = copy.deepcopy(document)
    for entry in edited["seed_variability"]:
        best = max(entry["per_seed_mean_terminal_wealth"])
        entry["distribution"] = report.describe([best])
    return edited


def _best_values(document):
    edited = copy.deepcopy(document)
    for entry in edited["seed_variability"]:
        best = max(entry["per_seed_mean_terminal_wealth"])
        entry["per_seed_mean_terminal_wealth"] = [best] * len(entry["seeds"])
    return edited


def _swapped_checkpoint(document):
    edited = copy.deepcopy(document)
    row = next(r for r in edited["rows"] if r["strategy"] == "linear_softmax_cem")
    row["checkpoint_sha256"] = "f" * 64
    return edited


def _best_summary(document):
    edited = copy.deepcopy(document)
    best = _best_seed(document)
    for summary in edited["summary"]:
        if summary["strategy"] == "ppo":
            values = [
                r["terminal_wealth"]
                for r in document["rows"]
                if r["strategy"] == "ppo"
                and r["member_seed"] == best
                and (r["constraint_set"], r["scenario"])
                == (summary["constraint_set"], summary["scenario"])
            ]
            summary["terminal_wealth"] = report.describe(values)
    return edited


@pytest.mark.parametrize(
    "mutant", [_only_best, _best_distribution, _best_values, _swapped_checkpoint, _best_summary]
)
def test_a_report_that_selects_on_test_results_is_refused(completed, mutant):
    with pytest.raises(SelectionError):
        check_no_test_selection(mutant(completed.document), completed.selection)


def test_a_runner_that_picks_the_best_test_seed_fails(trained_ppo, monkeypatch):
    real = report.seed_values

    def best_seed_only(rows, constraint_set, strategy, scenario):
        values = real(rows, constraint_set, strategy, scenario)
        best = max(values, key=values.get)
        return {best: values[best]}

    monkeypatch.setattr(report, "seed_values", best_seed_only)
    with pytest.raises(SelectionError):
        _evaluate(trained_ppo)
    assert not list((trained_ppo / "reports").glob("*/comparison.json"))  # nothing published


def test_a_runner_that_drops_all_but_the_best_seed_fails(trained_ppo, monkeypatch):
    real = runner.assemble

    def keep_best(**kwargs):
        document = {"rows": kwargs["rows"]}
        best = _best_seed(document)
        kwargs["rows"] = [
            r for r in kwargs["rows"] if r["strategy"] != "ppo" or r["member_seed"] == best
        ]
        return real(**kwargs)

    monkeypatch.setattr(runner, "assemble", keep_best)
    with pytest.raises(SelectionError):
        _evaluate(trained_ppo)


# --- a frozen selection belongs to one registered configuration ---


def test_a_selection_for_another_configuration_is_refused(trained):
    record = tiny_record()
    record["constraint_sets"][1]["max_turnover"] = 0.5
    with pytest.raises(ProtocolError, match="another registered configuration"):
        _evaluate(trained, config=comparison_config_from_dict(record))


def test_a_selection_missing_a_member_is_refused(trained_ppo):
    path = trained_ppo / "run" / runner.SELECTION_FILE
    selection = json.loads(path.read_text())
    selection["members"].pop()
    path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(SelectionError, match="every registered member"):
        _evaluate(trained_ppo)


def test_a_checkpoint_changed_after_selection_is_refused(trained_ppo):
    path = next((trained_ppo / "run" / "training" / "strict").glob("ppo_seed_3.json"))
    path.write_text(path.read_text().replace("0.3", "0.31"), encoding="utf-8")
    with pytest.raises(SelectionError, match="frozen"):
        _evaluate(trained_ppo)
