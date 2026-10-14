"""
A simple, inspectable learned baseline: a linear-softmax policy trained by
the cross-entropy method (CEM).

Policy. For each fund i (cash is the reference, logit 0):

    logit_i = b_i + beta_mom * z(ret_20)_i + beta_vol * z(vol_20)_i
    w       = softmax(logits)

Five numbers, each readable: b_i is a static tilt toward fund i over cash,
beta_mom > 0 chases 20-session momentum, beta_vol < 0 avoids recent
volatility. The features are the same frozen, training-fold z-scores the PPO
observation uses.

Why CEM rather than REINFORCE. The objective (mean log growth over training
episodes) is a noisy, non-differentiable function of five parameters.
- CEM evaluates deterministic policies on the same training episodes
  (common random numbers), needs no likelihood or step size, and has three
  hyperparameters (population, elite fraction, initial spread).
- REINFORCE would need a stochastic policy, a baseline to tame the variance
  of financial rewards, and a learning-rate schedule; with five parameters
  its gradient estimates are far noisier than CEM's ranking.
- Every candidate CEM tries is recorded, so the number of variants searched
  is known (multiple-testing accounting), and a deterministic policy has no
  logits-to-weights likelihood question at all.

Checkpoint selection. Each iteration's mean parameters are a checkpoint;
the one with the best mean dev-fold log growth is selected. The test fold is
never built here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from stock_agent.model_diagnostics.seeds import derive_seed
from stock_agent.rl.actions import logits_to_weights
from stock_agent.rl.env import EnvConfig, fold_envs
from stock_agent.rl.observation import Observation, ObservationBuilder
from stock_agent.rl.policies import PolicyInputError
from stock_agent.rl.replay import rollout
from stock_agent.rl.splits import FoldSpec, LeakageError
from stock_agent.rl.transition import PortfolioStep, TransitionInfo

__all__ = [
    "CEMConfig",
    "CEMIteration",
    "LearnedResult",
    "LinearSoftmaxParams",
    "LinearSoftmaxPolicy",
    "PolicyInputError",
    "cross_entropy_method",
    "train_linear_softmax",
]

MOMENTUM_FEATURE = "ret_20"
VOLATILITY_FEATURE = "vol_20"


@dataclass(frozen=True)
class LinearSoftmaxParams:
    bias: tuple[float, ...]
    beta_momentum: float
    beta_volatility: float

    @classmethod
    def zeros(cls, n_funds: int) -> LinearSoftmaxParams:
        return cls(bias=(0.0,) * n_funds, beta_momentum=0.0, beta_volatility=0.0)

    def vector(self) -> np.ndarray:
        return np.array([*self.bias, self.beta_momentum, self.beta_volatility])

    @classmethod
    def from_vector(cls, vector, n_funds: int) -> LinearSoftmaxParams:
        values = [float(v) for v in vector]
        if len(values) != n_funds + 2:
            raise ValueError(f"Expected {n_funds + 2} parameters, got {len(values)}.")
        return cls(
            bias=tuple(values[:n_funds]), beta_momentum=values[-2], beta_volatility=values[-1]
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "bias": list(self.bias),
            "beta_momentum": self.beta_momentum,
            "beta_volatility": self.beta_volatility,
        }

    @classmethod
    def from_dict(cls, record: dict) -> LinearSoftmaxParams:
        return cls(
            bias=tuple(float(v) for v in record["bias"]),
            beta_momentum=float(record["beta_momentum"]),
            beta_volatility=float(record["beta_volatility"]),
        )


class LinearSoftmaxPolicy:
    name = "linear_softmax_cem"

    def __init__(self, params: LinearSoftmaxParams, builder: ObservationBuilder) -> None:
        self.params = params
        self.cash_index = builder.cash_index
        self.funds = [i for i in range(len(builder.asset_names)) if i != builder.cash_index]
        if len(params.bias) != len(self.funds):
            raise ValueError("One bias per fund is required.")
        self.momentum_index = [
            builder.names.index(f"{builder.asset_names[i]}:{MOMENTUM_FEATURE}") for i in self.funds
        ]
        self.volatility_index = [
            builder.names.index(f"{builder.asset_names[i]}:{VOLATILITY_FEATURE}")
            for i in self.funds
        ]
        self.n_assets = len(builder.asset_names)

    def reset(self) -> None:
        pass

    def decide(self, observation: Observation) -> np.ndarray:
        momentum = observation.vector[self.momentum_index]
        volatility = observation.vector[self.volatility_index]
        if not (np.all(np.isfinite(momentum)) and np.all(np.isfinite(volatility))):
            raise PolicyInputError("Momentum or volatility features are missing.")
        logits = np.zeros(self.n_assets)
        logits[self.funds] = (
            np.asarray(self.params.bias)
            + self.params.beta_momentum * momentum
            + self.params.beta_volatility * volatility
        )
        return logits_to_weights(logits)


@dataclass(frozen=True)
class CEMConfig:
    iterations: int = 10
    population: int = 24
    elite_fraction: float = 0.25
    init_std: float = 1.0
    min_std: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.population < 2:
            raise ValueError("CEM needs at least one iteration and a population of two.")
        if not 0 < self.elite_fraction <= 1:
            raise ValueError("elite_fraction must be in (0, 1].")
        if self.init_std <= 0 or self.min_std <= 0:
            raise ValueError("Standard deviations must be positive.")

    def to_dict(self) -> dict[str, object]:
        return {
            "iterations": self.iterations,
            "population": self.population,
            "elite_fraction": self.elite_fraction,
            "init_std": self.init_std,
            "min_std": self.min_std,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class CEMIteration:
    index: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    best_score: float
    elite_mean_score: float

    def mean_params(self, n_funds: int) -> LinearSoftmaxParams:
        return LinearSoftmaxParams.from_vector(self.mean, n_funds)


def cross_entropy_method(
    objective: Callable[[np.ndarray], float],
    dimension: int,
    config: CEMConfig,
    init_mean: np.ndarray | None = None,
) -> list[CEMIteration]:
    """
    Maximize `objective`. Each iteration samples a Gaussian population, keeps
    the top elite_fraction (ties broken by sample order), and refits the mean
    and standard deviation (floored at min_std) to the elite.
    """

    rng = np.random.RandomState(derive_seed(config.seed, "rl_cem"))
    mean = np.zeros(dimension) if init_mean is None else np.asarray(init_mean, dtype=np.float64)
    std = np.full(dimension, config.init_std)
    n_elite = max(1, round(config.elite_fraction * config.population))
    history: list[CEMIteration] = []
    for index in range(config.iterations):
        samples = mean + std * rng.standard_normal((config.population, dimension))
        scores = np.array([objective(sample) for sample in samples])
        order = np.argsort(-scores, kind="stable")
        elite = samples[order[:n_elite]]
        mean = elite.mean(axis=0)
        std = np.maximum(elite.std(axis=0), config.min_std)
        history.append(
            CEMIteration(
                index=index,
                mean=tuple(float(v) for v in mean),
                std=tuple(float(v) for v in std),
                best_score=float(scores[order[0]]),
                elite_mean_score=float(scores[order[:n_elite]].mean()),
            )
        )
    return history


@dataclass(frozen=True)
class LearnedResult:
    params: LinearSoftmaxParams
    selected_iteration: int
    dev_scores: tuple[float, ...]
    history: tuple[CEMIteration, ...]
    variants_evaluated: int
    checkpoints_compared: int
    config: CEMConfig


def train_linear_softmax(
    train: FoldSpec,
    dev: FoldSpec,
    *,
    builder: ObservationBuilder,
    transition: PortfolioStep,
    transition_info: TransitionInfo,
    env_config: EnvConfig | None = None,
    config: CEMConfig | None = None,
) -> LearnedResult:
    if train.role != "train" or dev.role != "dev":
        raise LeakageError("Training uses the train fold and selects on the dev fold only.")
    config = config or CEMConfig()
    kwargs = {
        "transition": transition,
        "transition_info": transition_info,
        "builder": builder,
        "config": env_config,
    }
    train_envs = fold_envs(train, **kwargs)
    dev_envs = fold_envs(dev, **kwargs)
    n_funds = len(builder.asset_names) - 1

    def mean_log_growth(params: LinearSoftmaxParams, envs) -> float:
        policy = LinearSoftmaxPolicy(params, builder)
        return float(np.mean([rollout(env, policy).total_reward for env in envs]))

    history = cross_entropy_method(
        lambda vector: mean_log_growth(
            LinearSoftmaxParams.from_vector(vector, n_funds), train_envs
        ),
        dimension=n_funds + 2,
        config=config,
    )
    dev_scores = tuple(mean_log_growth(step.mean_params(n_funds), dev_envs) for step in history)
    selected = int(np.argmax(dev_scores))
    return LearnedResult(
        params=history[selected].mean_params(n_funds),
        selected_iteration=selected,
        dev_scores=dev_scores,
        history=tuple(history),
        variants_evaluated=config.iterations * config.population,
        checkpoints_compared=len(history),
        config=config,
    )
