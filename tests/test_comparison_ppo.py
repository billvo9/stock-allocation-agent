"""
The registered comparison with the real learned strategies (stable-baselines3
PPO and the CEM baseline) on a tiny plan, and the script end to end. A
completed run is an engineering result on synthetic scenarios, never
evidence of investment value.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from comparison_helpers import CODE, FAKE_KINDS, FixedClock, tiny_config, tiny_record

from stock_agent.comparison import runner
from stock_agent.comparison.fairness import check_no_test_selection
from stock_agent.comparison.ledger import LOOKS_ENV, RepeatLookError
from stock_agent.rl.observation import ObservationBuilder, fit_normalizer
from stock_agent.rl.ppo import METADATA_FILE, MODEL_FILE
from stock_agent.rl.scenarios import ASSET_NAMES, CASH_INDEX, generate_scenario

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    root = tmp_path_factory.mktemp("comparison_ppo")
    config = tiny_config()
    clock = FixedClock()
    selection = runner.train(config, run_dir=root / "run", code=CODE, clock=clock)
    common = {
        "run_dir": root / "run",
        "fold_role": "test",
        "reports_root": root / "reports",
        "health_inputs_root": root / "health",
        "code": CODE,
        "clock": clock,
        "looks_root": root / "looks",
    }
    first = runner.evaluate(config, **common)
    second = runner.evaluate(config, **common, repeat_look=True)
    return SimpleNamespace(
        root=root,
        selection=json.loads(selection.read_text()),
        first=json.loads(first.read_text()),
        second=json.loads(second.read_text()),
    )


def test_every_real_member_is_dev_selected_and_frozen_with_its_hash(real):
    members = real.selection["members"]
    assert len(members) == 2 * (5 + 5)
    for member in members:
        assert member["selection"]["checkpoints_compared"] >= 1
        if member["kind"] == "ppo":
            metadata = json.loads(
                (real.root / "run" / member["checkpoint"]["path"] / METADATA_FILE).read_text()
            )
            assert metadata["model"]["sha256"] == member["checkpoint"]["sha256"]
            assert metadata["config"]["env"]["max_turnover"] == (
                0.25 if member["constraint_set"] == "strict" else None
            )
            assert metadata["selection"]["metric"].startswith("mean dev-fold")
        else:
            assert member["selection"]["metric"].startswith("mean dev-fold")
    assert check_no_test_selection(real.first, real.selection) == 20


def test_the_real_comparison_reports_every_seed_and_passes_the_contract(real):
    document = real.first
    assert document["fairness"]["contract"] == "passed"
    seeds = {
        (r["strategy"], r["member_seed"]) for r in document["rows"] if r["member_seed"] is not None
    }
    for strategy in ("linear_softmax_cem", "ppo", "ppo_guarded"):
        assert {seed for name, seed in seeds if name == strategy} == {0, 1, 2, 3, 4}
    for entry in document["seed_variability"]:
        assert entry["distribution"]["n"] == 5


def test_a_repeat_evaluation_is_labelled_and_reproduces_the_rows_exactly(real):
    assert (real.first["look"]["number"], real.first["look"]["repeat"]) == (1, False)
    assert (real.second["look"]["number"], real.second["look"]["repeat"]) == (2, True)
    assert real.second["rows"] == real.first["rows"]  # same frozen members, same results


def test_every_member_carries_a_deterministic_parameter_hash(real):
    frozen = {
        (m["constraint_set"], m["kind"], m["seed"]): m["parameters_sha256"]
        for m in real.selection["members"]
    }
    learned = {"linear_softmax_cem": "cem", "ppo": "ppo", "ppo_guarded": "ppo"}
    for row in real.first["rows"]:
        if row["member_seed"] is not None:
            key = (row["constraint_set"], learned[row["strategy"]], row["member_seed"])
            assert row["parameters_sha256"] == frozen[key]


def test_identical_trainings_give_identical_parameters_though_the_archives_differ(tmp_path):
    config = tiny_config()
    run_config = config.run_config(config.constraint_set("plain"))
    normalizer = fit_normalizer(run_config.plan.train, run_config.features)
    builder = ObservationBuilder(run_config.features, normalizer, ASSET_NAMES, CASH_INDEX)
    context = {"run_id": "twice", "code": dict(CODE), "created_at": "2026-10-17T00:00:00Z"}
    records = [
        runner.PPOKind().train(
            run_config,
            0,
            builder=builder,
            run_dir=tmp_path / name,
            subpath="ppo_seed_0",
            loader=generate_scenario,
            context=context,
        )
        for name in ("first", "second")
    ]
    assert records[0]["parameters_sha256"] == records[1]["parameters_sha256"]
    # The archives differ only in stable-baselines3's recorded training start time.
    assert records[0]["checkpoint"]["sha256"] != records[1]["checkpoint"]["sha256"]
    archives = [
        zipfile.ZipFile(tmp_path / n / "ppo_seed_0" / MODEL_FILE) for n in ("first", "second")
    ]
    assert archives[0].read("policy.pth") == archives[1].read("policy.pth")
    data = [json.loads(archive.read("data")) for archive in archives]
    assert data[0]["start_time"] != data[1]["start_time"]
    data[0].pop("start_time"), data[1].pop("start_time")
    assert data[0] == data[1]


def _script():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "run_ppo_comparison", SCRIPTS / "run_ppo_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_trains_then_evaluates_the_test_fold_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(LOOKS_ENV, str(tmp_path / "private" / "looks"))
    for name, kind in FAKE_KINDS.items():
        monkeypatch.setitem(runner.KINDS, name, kind)
    record = tiny_record()
    record["strategies"] = ["equal_weight", "all_shv", "ppo"]
    record["cem_seeds"] = []
    path = tmp_path / "registered.yaml"
    path.write_text(yaml.safe_dump(record), encoding="utf-8")
    script = _script()
    kwargs = {
        "clock": lambda: datetime(2026, 10, 17, 12, tzinfo=UTC),
        "code_version": lambda: dict(CODE),
    }
    assert (
        script.main(
            ["train", "--config", str(path), "--output-root", str(tmp_path / "models")], **kwargs
        )
        == 0
    )
    (run_dir,) = (tmp_path / "models").iterdir()
    evaluate = [
        "evaluate",
        "--config",
        str(path),
        "--run-dir",
        str(run_dir),
        "--reports-root",
        str(tmp_path / "reports"),
        "--health-inputs-root",
        str(tmp_path / "health"),
    ]
    assert script.main(evaluate, **kwargs) == 0
    output = capsys.readouterr().out
    assert "Synthetic scenarios only" in output and "1 test look(s)" in output
    with pytest.raises(RepeatLookError):
        script.main(evaluate, **kwargs)
    elsewhere = [*evaluate[:-4], "--reports-root", str(tmp_path / "elsewhere"), *evaluate[-2:]]
    with pytest.raises(RepeatLookError):  # another reports root is the same configuration
        script.main(elsewhere, **kwargs)
    logs = list((tmp_path / "private" / "looks").glob("*.jsonl"))
    assert len(logs) == 1 and not list((tmp_path / "reports").glob("*.jsonl"))
    assert script.main([*evaluate, "--record-repeat-look"], **kwargs) == 0
    artifacts = sorted((tmp_path / "reports").glob("*/comparison.json"))
    assert [json.loads(p.read_text())["look"]["number"] for p in artifacts] == [1, 2]
