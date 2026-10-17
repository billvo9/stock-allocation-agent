"""
The registered comparison, in two stages.

1. `train`: fit the observation normalizer on the train fold, then train
   every learned member (PPO and CEM, each seed, under each constraint set)
   with checkpoint selection on the dev fold only, and freeze the selection
   in `selection.json` (written once; its SHA-256 identifies it). The loader
   the trainers get refuses every scenario seed of the test fold
   (`TrainingLoader`), so the test fold cannot be built while anything is
   chosen.
2. `evaluate`: read the frozen selection, load every member (checking its
   recorded checkpoint hash and that it observes through the same frozen
   normalizer), record the look in the evaluation log BEFORE any evaluation
   data exists, then run every strategy on every episode of the fold through
   one API (`harness.run_comparison`, so `replay.rollout` on `AllocationEnv`
   with the approved `portfolio_step`), at 1x the D2 rates, and replay each
   strategy's decisions at 0x, 1x and 2x. `fairness.check_contract` checks
   every record (the 1x replay must reproduce the run exactly), measures come
   from `measures`, and `fairness.check_no_test_selection` checks the
   assembled artifact before it and the health inputs are written.

Learned members come from `LearnedKind`s (`PPOKind`, `CEMKind`); tests
inject light ones through the same seam. Training can run members in
parallel processes: each member is seeded and independent, so the result
does not depend on the number of workers.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from stock_agent.comparison import measures
from stock_agent.comparison.fairness import (
    FairnessError,
    SelectionError,
    check_contract,
    check_no_test_selection,
)
from stock_agent.comparison.ledger import EvaluationLog
from stock_agent.comparison.protocol import (
    HEADLINE_COST_MULTIPLIER,
    RULE_FACTORIES,
    STRATEGIES,
    ComparisonConfig,
    ProtocolError,
)
from stock_agent.comparison.report import (
    assemble,
    health_records,
    multiplier_key,
    write_artifact,
    write_health_inputs,
)
from stock_agent.portfolio.transition import portfolio_step
from stock_agent.rl.config import RunConfig
from stock_agent.rl.env import AllocationEnv
from stock_agent.rl.harness import PolicyEntry, run_comparison
from stock_agent.rl.learned import LinearSoftmaxParams, LinearSoftmaxPolicy, train_linear_softmax
from stock_agent.rl.observation import (
    FeatureNormalizer,
    FeatureSpec,
    ObservationBuilder,
    ScenarioLoader,
    fit_normalizer,
    require_fitted_on,
)
from stock_agent.rl.replay import ScriptedPolicy, rollout
from stock_agent.rl.reward import REWARD_INFO
from stock_agent.rl.safety import GuardedPolicy
from stock_agent.rl.scenarios import (
    ASSET_NAMES,
    CASH_INDEX,
    GENERATOR_VERSION,
    ScenarioData,
    generate_scenario,
)
from stock_agent.rl.splits import LeakageError
from stock_agent.rl.transition import APPROVED_TRANSITION

SELECTION_SCHEMA = "ppo_comparison_selection"
SELECTION_SCHEMA_VERSION = "1.0.0"
SELECTION_FILE = "selection.json"
LOG_FILE = "evaluation_log.jsonl"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _timestamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


@dataclass(frozen=True)
class TrainingLoader:
    """Scenario loader for training and selection: the evaluation fold's seeds are refused."""

    forbidden_seeds: frozenset[int]
    loader: ScenarioLoader = generate_scenario

    def __call__(self, scenario_id: str, seed: int) -> ScenarioData:
        if int(seed) in self.forbidden_seeds:
            raise LeakageError(
                f"Scenario seed {seed} belongs to the test fold, which is never built while "
                "anything is trained or selected."
            )
        return self.loader(scenario_id, seed)


def scaled_costs(data: ScenarioData, multiplier: float) -> ScenarioData:
    """The same scenario with its base cost rates times a D2 sensitivity multiplier."""

    rates = tuple(rate * multiplier for rate in data.base_cost_rates)
    return dataclasses.replace(data, base_cost_rates=rates)


class LearnedKind(Protocol):
    """A learned strategy's trainer and loader (`PPOKind`, `CEMKind`, or a test double)."""

    name: str

    def train(
        self,
        run_config: RunConfig,
        seed: int,
        *,
        builder: ObservationBuilder,
        run_dir: Path,
        subpath: str,
        loader: ScenarioLoader,
        context: Mapping[str, object],
    ) -> dict: ...

    def load(
        self,
        member: Mapping[str, object],
        *,
        run_dir: Path,
        builder: ObservationBuilder,
        run_config: RunConfig,
    ) -> Callable[[], object]: ...


def _same_observation(normalizer: FeatureNormalizer, builder: ObservationBuilder) -> None:
    if normalizer != builder.normalizer:
        raise FairnessError("A learned member observes through another normalizer.")


class PPOKind:
    """PPO through stable-baselines3 (`rl.ppo`, imported only when used)."""

    name = "ppo"

    def train(self, run_config, seed, *, builder, run_dir, subpath, loader, context) -> dict:
        from stock_agent.rl.ppo import MODEL_FILE, train_ppo_seed

        run = train_ppo_seed(
            run_config,
            seed,
            builder=builder,
            transition=portfolio_step,
            transition_info=APPROVED_TRANSITION,
            out_dir=Path(run_dir) / subpath,
            run_id=str(context["run_id"]),
            code=dict(context["code"]),
            created_at=str(context["created_at"]),
            loader=loader,
        )
        return {
            "checkpoint": {
                "path": subpath,
                "file": f"{subpath}/{MODEL_FILE}",
                "sha256": run.metadata["model"]["sha256"],
            },
            "selection": run.metadata["selection"],
            "healthy": run.health["status"] in ("healthy", "warning"),
            "health_status": run.health["status"],
            "health_findings": list(run.health["reasons"]) + list(run.health["warnings"]),
            "timesteps": run.metadata["timesteps"],
        }

    def load(self, member, *, run_dir, builder, run_config) -> Callable[[], object]:
        from stock_agent.rl.ppo import load_ppo_checkpoint

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # the code SHA is recorded instead
            checkpoint = load_ppo_checkpoint(
                Path(run_dir) / member["checkpoint"]["path"],
                expected_config_sha256=run_config.sha256(),
            )
        if checkpoint.metadata["model"]["sha256"] != member["checkpoint"]["sha256"]:
            raise SelectionError("A PPO checkpoint is not the one frozen at selection.")
        _same_observation(checkpoint.builder.normalizer, builder)
        policy = checkpoint.policy
        return lambda: policy


class CEMKind:
    """The linear-softmax baseline trained by the cross-entropy method (`rl.learned`)."""

    name = "cem"

    def train(self, run_config, seed, *, builder, run_dir, subpath, loader, context) -> dict:
        result = train_linear_softmax(
            run_config.plan,
            builder=builder,
            transition=portfolio_step,
            transition_info=APPROVED_TRANSITION,
            env_config=run_config.env,
            config=dataclasses.replace(run_config.cem, seed=seed),
            loader=loader,
        )
        path = Path(run_dir) / f"{subpath}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
        return {
            "checkpoint": {"path": f"{subpath}.json", "sha256": file_sha256(path)},
            "selection": {
                "metric": "mean dev-fold log growth (deterministic policy)",
                "selected_iteration": result.selected_iteration,
                "dev_scores": list(result.dev_scores),
                "checkpoints_compared": result.checkpoints_compared,
            },
            "variants_evaluated": result.variants_evaluated,
            "healthy": True,
        }

    def load(self, member, *, run_dir, builder, run_config) -> Callable[[], object]:
        path = Path(run_dir) / member["checkpoint"]["path"]
        if file_sha256(path) != member["checkpoint"]["sha256"]:
            raise SelectionError("A CEM checkpoint is not the one frozen at selection.")
        record = json.loads(path.read_text(encoding="utf-8"))
        _same_observation(FeatureNormalizer.from_dict(record["normalizer"]), builder)
        if record["env_config"] != run_config.env.to_dict():
            raise FairnessError("A CEM member was trained under another constraint set.")
        params = LinearSoftmaxParams.from_dict(record["params"])
        return lambda: LinearSoftmaxPolicy(params, builder)


KINDS: dict[str, LearnedKind] = {"ppo": PPOKind(), "cem": CEMKind()}


def _train_member(job: tuple) -> dict:
    kind, run_config, seed, builder, run_dir, subpath, loader, context, set_name = job
    record = kind.train(
        run_config,
        seed,
        builder=builder,
        run_dir=run_dir,
        subpath=subpath,
        loader=loader,
        context=context,
    )
    return {"constraint_set": set_name, "kind": kind.name, "seed": int(seed), **record}


def train(
    config: ComparisonConfig,
    *,
    run_dir: Path,
    code: Mapping[str, object],
    clock: Callable[[], datetime],
    kinds: Mapping[str, LearnedKind] = KINDS,
    loader: ScenarioLoader = generate_scenario,
    workers: int = 1,
) -> Path:
    """
    Train and dev-select every learned member, then write the frozen
    selection (module docstring). Returns the selection file's path.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    plan = config.rl.plan
    guarded = TrainingLoader(frozenset(plan.test.seeds), loader)
    normalizer = fit_normalizer(plan.train, config.rl.features, loader=guarded)
    builder = ObservationBuilder(config.rl.features, normalizer, ASSET_NAMES, CASH_INDEX)
    created_at = _timestamp(clock())
    context = {"run_id": run_dir.name, "code": dict(code), "created_at": created_at}
    jobs = [
        (
            kinds[kind],
            config.run_config(constraint_set),
            seed,
            builder,
            run_dir,
            f"training/{constraint_set.name}/{kind}_seed_{seed}",
            guarded,
            context,
            constraint_set.name,
        )
        for constraint_set in config.constraint_sets
        for kind in config.learned_kinds()
        for seed in config.seeds_for(kind)
    ]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            members = list(pool.map(_train_member, jobs))
    else:
        members = [_train_member(job) for job in jobs]
    selection = {
        "schema": SELECTION_SCHEMA,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "run_id": run_dir.name,
        "created_at": created_at,
        "config": config.to_dict(),
        "config_sha256": config.sha256(),
        "code": dict(code),
        "scenario_generator_version": GENERATOR_VERSION,
        "transition": APPROVED_TRANSITION.to_dict(),
        "reward": REWARD_INFO.to_dict(),
        "feature_spec": config.rl.features.to_dict(),
        "normalizer": normalizer.to_dict(),
        "test_fold_guard": "every scenario seed of the test fold was refused while members "
        "were trained and selected",
        "members": members,
    }
    path = run_dir / SELECTION_FILE
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(selection, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


@dataclass(frozen=True)
class FrozenSelection:
    record: dict
    sha256: str
    builder: ObservationBuilder


def read_selection(config: ComparisonConfig, run_dir: Path) -> FrozenSelection:
    """The frozen selection, after checking it belongs to this registered configuration."""

    path = Path(run_dir) / SELECTION_FILE
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("schema"), record.get("schema_version")) != (
        SELECTION_SCHEMA,
        SELECTION_SCHEMA_VERSION,
    ):
        raise ProtocolError("Unsupported selection schema.")
    if record["config_sha256"] != config.sha256():
        raise ProtocolError("The run was trained for another registered configuration.")
    if record["scenario_generator_version"] != GENERATOR_VERSION:
        raise ProtocolError("The run was trained on another scenario generator version.")
    normalizer = FeatureNormalizer.from_dict(record["normalizer"])
    spec = FeatureSpec.from_dict(record["feature_spec"])
    require_fitted_on(normalizer, config.rl.plan.train, spec)
    builder = ObservationBuilder(spec, normalizer, ASSET_NAMES, CASH_INDEX)
    expected = [
        (cs.name, kind, seed)
        for cs in config.constraint_sets
        for kind in config.learned_kinds()
        for seed in config.seeds_for(kind)
    ]
    members = [(m["constraint_set"], m["kind"], int(m["seed"])) for m in record["members"]]
    if members != expected:
        raise SelectionError("The frozen selection does not hold every registered member.")
    return FrozenSelection(record=record, sha256=file_sha256(path), builder=builder)


def _entries(
    config: ComparisonConfig,
    selection: FrozenSelection,
    run_dir: Path,
    kinds: Mapping[str, LearnedKind],
) -> dict[str, list[tuple[PolicyEntry, dict | None]]]:
    """Every strategy's entries per constraint set, frozen before any evaluation data exists."""

    by_set: dict[str, list[tuple[PolicyEntry, dict | None]]] = {}
    for constraint_set in config.constraint_sets:
        run_config = config.run_config(constraint_set)
        loaded = {}
        for member in selection.record["members"]:
            if member["constraint_set"] == constraint_set.name:
                loaded[(member["kind"], int(member["seed"]))] = (
                    kinds[member["kind"]].load(
                        member, run_dir=run_dir, builder=selection.builder, run_config=run_config
                    ),
                    member,
                )
        entries: list[tuple[PolicyEntry, dict | None]] = []
        for key in config.strategies:
            spec = STRATEGIES[key]
            if not spec.is_learned:
                entries.append((PolicyEntry(key, RULE_FACTORIES[key], kind=spec.kind), None))
                continue
            for seed in config.seeds_for(spec.learned):
                factory, member = loaded[(spec.learned, seed)]
                if key == "ppo_guarded":
                    factory = lambda raw=factory, healthy=member["healthy"], gate=run_config.gate: (
                        GuardedPolicy(raw(), gate=gate, model_healthy=healthy)
                    )
                entries.append(
                    (PolicyEntry(f"{key}/seed_{seed}", factory, kind="learned", group=key), member)
                )
        by_set[constraint_set.name] = entries
    return by_set


def _row(entry: PolicyEntry, member: dict | None, constraint_set: str, episode, measured) -> dict:
    key = entry.group or entry.name
    return {
        "constraint_set": constraint_set,
        "strategy": key,
        "member_seed": None if member is None else int(member["seed"]),
        "checkpoint_sha256": None if member is None else member["checkpoint"]["sha256"],
        "scenario": episode[0],
        "scenario_seed": int(episode[1]),
        **measured,
    }


def evaluate(
    config: ComparisonConfig,
    *,
    run_dir: Path,
    fold_role: str,
    reports_root: Path,
    health_inputs_root: Path,
    code: Mapping[str, object],
    clock: Callable[[], datetime],
    kinds: Mapping[str, LearnedKind] = KINDS,
    loader: ScenarioLoader = generate_scenario,
    repeat_look: bool = False,
) -> Path:
    """
    Evaluate the frozen selection on one fold (module docstring) and write
    the artifact and the health inputs. Returns the artifact's path.
    """

    if fold_role not in ("test", "dev"):
        raise ValueError("The comparison is evaluated on the test or the dev fold.")
    run_dir = Path(run_dir)
    selection = read_selection(config, run_dir)
    entries = _entries(config, selection, run_dir, kinds)
    builder = selection.builder
    fold = config.rl.plan.test if fold_role == "test" else config.rl.plan.dev
    generated_at = clock()
    log = EvaluationLog(Path(reports_root) / LOG_FILE)
    number = log.looks(config.sha256(), fold_role) + 1
    stamp = f"{generated_at.astimezone(UTC):%Y%m%dT%H%M%S%fZ}"
    artifact_id = f"{config.name}-{fold_role}-look{number:03d}-{stamp}"
    look = log.record_look(
        config_name=config.name,
        config_sha256=config.sha256(),
        fold=fold_role,
        artifact_id=artifact_id,
        selection_sha256=selection.sha256,
        code=code,
        recorded_at=generated_at,
        repeat_allowed=repeat_look,
    )
    if look["look_number"] != number:
        raise RuntimeError("The evaluation log changed while this look was recorded.")

    raw = {episode: loader(*episode) for episode in fold.episodes()}
    data = {
        multiplier: {episode: scaled_costs(d, multiplier) for episode, d in raw.items()}
        for multiplier in config.cost_multipliers
    }
    headline_data = data[HEADLINE_COST_MULTIPLIER]
    rows: list[dict] = []
    checked = {"strategies": 0, "episodes": len(fold.episodes()), "steps": 0, "replays": 0}
    for constraint_set in config.constraint_sets:
        env_config = config.run_config(constraint_set).env
        for entry, member in entries[constraint_set.name]:
            headline = run_comparison(
                [entry],
                fold,
                builder=builder,
                transition=portfolio_step,
                transition_info=APPROVED_TRANSITION,
                env_config=env_config,
                loader=lambda scenario, seed: headline_data[(scenario, seed)],
            ).records[entry.name]
            contract = check_contract(
                {entry.name: headline},
                data=headline_data,
                fold=fold,
                env_config=env_config,
                builder=builder,
                cost_multiplier=HEADLINE_COST_MULTIPLIER,
            )
            checked["strategies"] += 1
            checked["steps"] += contract["steps"]
            sensitivity: dict[tuple, dict] = {episode: {} for episode in headline}
            for multiplier in config.cost_multipliers:
                replayed = {}
                for episode, record in headline.items():
                    env = AllocationEnv(
                        data[multiplier][episode],
                        start=fold.start,
                        stop=fold.stop,
                        transition=portfolio_step,
                        transition_info=APPROVED_TRANSITION,
                        builder=builder,
                        config=env_config,
                    )
                    replayed[episode] = rollout(env, ScriptedPolicy(record.targets()))
                    if multiplier == HEADLINE_COST_MULTIPLIER and not np.array_equal(
                        replayed[episode].wealth_path, record.wealth_path
                    ):
                        raise FairnessError(
                            f"{entry.name} on {episode}: the result is not determined by its "
                            "decisions (the 1x replay differs)."
                        )
                    sensitivity[episode][multiplier_key(multiplier)] = measures.terminal_measures(
                        replayed[episode]
                    )
                check_contract(
                    {entry.name: replayed},
                    data=data[multiplier],
                    fold=fold,
                    env_config=env_config,
                    builder=builder,
                    cost_multiplier=multiplier,
                )
                checked["replays"] += len(replayed)
            for episode in fold.episodes():
                measured = measures.episode_measures(headline[episode], CASH_INDEX)
                measured["cost_sensitivity"] = sensitivity[episode]
                rows.append(_row(entry, member, constraint_set.name, episode, measured))

    document = assemble(
        config=config,
        fold=fold,
        look=look,
        selection_sha256=selection.sha256,
        selection=selection.record,
        code=code,
        generated_at=generated_at,
        artifact_id=artifact_id,
        rows=rows,
        contract={
            "contract": "passed",
            "checked": checked,
            "rules": [
                "one API: replay.rollout on AllocationEnv for every strategy",
                "the approved portfolio_step reproduces every booked step bit for bit",
                "decide at a complete session, execute at the next (D1)",
                "the registered schedule and offset for every strategy (D7)",
                "D2 provisional rates times the scenario regime and the registered multiplier",
                "100% cash until the common build close (D3)",
                "the registered D5 set through feasible_target for every target",
                "the 1x replay of each strategy's decisions reproduces its run exactly",
            ],
        },
    )
    members_checked = check_no_test_selection(document, selection.record)
    if members_checked != len(selection.record["members"]):
        raise SelectionError("Not every frozen member was checked.")
    path = write_artifact(document, reports_root)
    write_health_inputs(health_records(document, clock=clock), health_inputs_root)
    return path
