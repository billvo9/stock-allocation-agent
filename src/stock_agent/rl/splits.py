"""
Train / dev / test folds for synthetic RL research, separated by time AND seed.

A fold is a set of (scenario, seed) paths and a session window [start, stop).
Episodes in a fold make decisions, execute and earn only inside its window;
observations may look back before `start` (that is the past, not leakage).

    train  fit policies and observation normalization
    dev    tune hyperparameters and select checkpoints
    test   report once, after every choice is fixed

Separation rules (enforced, `LeakageError` otherwise):
- windows are ordered train < dev < test, with at least `embargo` sessions
  between them, so no reward in one fold uses returns of the next and no
  later fold's lookback reaches into an earlier fold's rewards;
- seed sets are pairwise disjoint, so dev and test paths are realizations the
  training never saw.
"""

from __future__ import annotations

from dataclasses import dataclass

from stock_agent.rl.scenarios import SCENARIOS

ROLES = ("train", "dev", "test")


class LeakageError(ValueError):
    """An operation would let information cross a fold or time boundary."""


@dataclass(frozen=True)
class FoldSpec:
    role: str
    scenario_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"Unknown fold role {self.role!r}; expected one of {ROLES}.")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("A fold needs distinct seeds.")
        if not self.scenario_ids:
            raise ValueError("A fold needs at least one scenario.")
        for scenario_id in self.scenario_ids:
            SCENARIOS[scenario_id]  # KeyError for an unknown scenario
        if not 0 <= self.start < self.stop:
            raise ValueError("A fold window needs 0 <= start < stop.")

    def episodes(self) -> list[tuple[str, int]]:
        return [(scenario, seed) for scenario in self.scenario_ids for seed in self.seeds]

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "scenario_ids": list(self.scenario_ids),
            "seeds": list(self.seeds),
            "start": self.start,
            "stop": self.stop,
        }

    @classmethod
    def from_dict(cls, record: dict) -> FoldSpec:
        return cls(
            role=record["role"],
            scenario_ids=tuple(record["scenario_ids"]),
            seeds=tuple(int(seed) for seed in record["seeds"]),
            start=int(record["start"]),
            stop=int(record["stop"]),
        )


@dataclass(frozen=True)
class SplitPlan:
    train: FoldSpec
    dev: FoldSpec
    test: FoldSpec
    embargo: int

    def __post_init__(self) -> None:
        for role, fold in zip(ROLES, (self.train, self.dev, self.test), strict=True):
            if fold.role != role:
                raise ValueError(f"The {role} slot holds a fold with role {fold.role!r}.")
        if self.embargo < 0:
            raise ValueError("embargo must be non-negative.")
        for earlier, later in ((self.train, self.dev), (self.dev, self.test)):
            if earlier.stop + self.embargo > later.start:
                raise LeakageError(
                    f"The {later.role} window starts at {later.start}, inside the "
                    f"{self.embargo}-session embargo after the {earlier.role} window "
                    f"(ends {earlier.stop})."
                )
        folds = (self.train, self.dev, self.test)
        for i, first in enumerate(folds):
            for second in folds[i + 1 :]:
                shared = set(first.seeds) & set(second.seeds)
                if shared:
                    raise LeakageError(
                        f"The {first.role} and {second.role} folds share seeds {sorted(shared)}."
                    )

    def to_dict(self) -> dict[str, object]:
        return {
            "train": self.train.to_dict(),
            "dev": self.dev.to_dict(),
            "test": self.test.to_dict(),
            "embargo": self.embargo,
        }

    @classmethod
    def from_dict(cls, record: dict) -> SplitPlan:
        return cls(
            train=FoldSpec.from_dict(record["train"]),
            dev=FoldSpec.from_dict(record["dev"]),
            test=FoldSpec.from_dict(record["test"]),
            embargo=int(record["embargo"]),
        )


EVALUATION_SCENARIOS = (
    "calm",
    "crash",
    "high_correlation",
    "extreme_volatility",
    "stale_observations",
    "missing_closes",
    "high_costs",
    "mixed",
)


def default_split_plan() -> SplitPlan:
    """
    The registered synthetic plan: train on the mixed-regime scenario, then
    dev and test on every scenario, with later windows and unseen seeds.
    """

    return SplitPlan(
        train=FoldSpec("train", ("mixed",), tuple(range(16)), 60, 900),
        dev=FoldSpec("dev", EVALUATION_SCENARIOS, tuple(range(100, 104)), 960, 1200),
        test=FoldSpec("test", EVALUATION_SCENARIOS, tuple(range(200, 204)), 1260, 1500),
        embargo=60,
    )
