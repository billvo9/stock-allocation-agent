"""Train / dev / test folds defined by time and by seed."""

from __future__ import annotations

import dataclasses

import pytest

from stock_agent.rl import splits


def _plan(**changes):
    plan = splits.default_split_plan()
    return dataclasses.replace(plan, **changes)


def test_the_default_plan_separates_time_and_seeds():
    plan = splits.default_split_plan()
    assert plan.train.stop + plan.embargo <= plan.dev.start
    assert plan.dev.stop + plan.embargo <= plan.test.start
    seeds = [set(plan.train.seeds), set(plan.dev.seeds), set(plan.test.seeds)]
    assert not (seeds[0] & seeds[1] or seeds[0] & seeds[2] or seeds[1] & seeds[2])
    assert [plan.train.role, plan.dev.role, plan.test.role] == ["train", "dev", "test"]


def test_overlapping_seeds_are_rejected():
    plan = splits.default_split_plan()
    leaky_dev = dataclasses.replace(plan.dev, seeds=(plan.train.seeds[0],))
    with pytest.raises(splits.LeakageError, match="seed"):
        _plan(dev=leaky_dev)


def test_a_dev_window_inside_the_embargo_is_rejected():
    plan = splits.default_split_plan()
    early_dev = dataclasses.replace(plan.dev, start=plan.train.stop + plan.embargo - 1)
    with pytest.raises(splits.LeakageError, match="embargo"):
        _plan(dev=early_dev)


def test_a_test_window_before_dev_is_rejected():
    plan = splits.default_split_plan()
    early_test = dataclasses.replace(plan.test, start=plan.dev.start, stop=plan.dev.stop)
    with pytest.raises(splits.LeakageError):
        _plan(test=early_test)


def test_roles_must_match_their_slot():
    plan = splits.default_split_plan()
    with pytest.raises(ValueError, match="role"):
        _plan(dev=dataclasses.replace(plan.dev, role="train"))


def test_a_fold_needs_a_known_role_seeds_and_a_window():
    with pytest.raises(ValueError, match="role"):
        splits.FoldSpec("holdout", ("calm",), (1,), 60, 100)
    with pytest.raises(ValueError, match="seed"):
        splits.FoldSpec("train", ("calm",), (), 60, 100)
    with pytest.raises(ValueError, match="window"):
        splits.FoldSpec("train", ("calm",), (1,), 100, 100)
    with pytest.raises(KeyError):
        splits.FoldSpec("train", ("unknown",), (1,), 60, 100)


def test_fold_episodes_enumerate_scenarios_by_seeds_in_a_fixed_order():
    fold = splits.FoldSpec("dev", ("calm", "crash"), (5, 6), 960, 1200)
    assert fold.episodes() == [("calm", 5), ("calm", 6), ("crash", 5), ("crash", 6)]


def test_folds_serialize_for_artifacts():
    plan = splits.default_split_plan()
    record = plan.to_dict()
    assert record["train"]["role"] == "train"
    assert record["embargo"] == plan.embargo
    assert splits.SplitPlan.from_dict(record) == plan
