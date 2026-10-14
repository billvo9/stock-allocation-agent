"""
A `gymnasium.Env` around the allocation environment, for learning libraries.

The action is a vector of fund logits (one per fund; cash is the reference
with its logit pinned at 0), in the box [-ACTION_BOUND, ACTION_BOUND]. The
wrapper maps it to weights with the deterministic clip-pad-softmax map
(`rl.actions.fund_logits_to_weights`) and passes the weights to the same
`AllocationEnv.step` every other policy uses; the learner scores its
likelihood on the raw logits it sampled.

Each `reset` samples one training episode, a (scenario, seed) pair of the
fold and a window of `episode_sessions` inside the fold's window, from the
environment's seeded generator, so `reset(seed=s)` replays exactly.

Network input. Market features that cannot be computed (too little visible
history, for example under a stale feed) are set to 0, the training mean of
a z-score, and counted in `info["imputed_features"]`. A deployed policy is
wrapped by `safety.GuardedPolicy`, which falls back before the network ever
sees a non-finite feature.
"""

from __future__ import annotations

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stock_agent.rl.actions import ACTION_BOUND, fund_logits_to_weights
from stock_agent.rl.env import AllocationEnv, EnvConfig
from stock_agent.rl.observation import Observation, ObservationBuilder, ScenarioLoader
from stock_agent.rl.scenarios import ScenarioData, generate_scenario
from stock_agent.rl.splits import FoldSpec
from stock_agent.rl.transition import PortfolioStep, TransitionInfo


def network_input(observation: Observation) -> tuple[np.ndarray, int]:
    """The observation vector as float32 with non-finite entries set to 0."""

    vector = np.array(observation.vector, dtype=np.float64)
    missing = ~np.isfinite(vector)
    vector[missing] = 0.0
    return vector.astype(np.float32), int(missing.sum())


class EpisodeSampler:
    """Uniform (scenario, seed) and window start inside one fold."""

    def __init__(
        self,
        fold: FoldSpec,
        episode_sessions: int,
        loader: ScenarioLoader = generate_scenario,
    ) -> None:
        if not 3 <= episode_sessions <= fold.stop - fold.start:
            raise ValueError("episode_sessions must fit inside the fold window.")
        self.fold = fold
        self.episode_sessions = episode_sessions
        self.loader = loader
        self.episodes = fold.episodes()
        self._cache: dict[tuple[str, int], ScenarioData] = {}

    def data(self, scenario_id: str, seed: int) -> ScenarioData:
        key = (scenario_id, int(seed))
        if key not in self._cache:
            self._cache[key] = self.loader(scenario_id, int(seed))
        return self._cache[key]

    def sample(self, rng: np.random.Generator) -> tuple[str, int, int, int]:
        scenario_id, seed = self.episodes[int(rng.integers(len(self.episodes)))]
        latest = self.fold.stop - self.episode_sessions
        start = int(rng.integers(self.fold.start, latest + 1))
        return scenario_id, seed, start, start + self.episode_sessions


class AllocationGymEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        sampler: EpisodeSampler,
        *,
        builder: ObservationBuilder,
        transition: PortfolioStep,
        transition_info: TransitionInfo,
        env_config: EnvConfig | None = None,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.builder = builder
        self.transition = transition
        self.transition_info = transition_info
        self.env_config = env_config or EnvConfig()
        n_market = len(builder.normalizer.names)
        n_assets = len(builder.asset_names)
        clip = builder.normalizer.clip
        low = np.concatenate([np.full(n_market, -clip), np.zeros(n_assets + 1)])
        high = np.concatenate([np.full(n_market, clip), np.ones(n_assets + 1)])
        self.observation_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-ACTION_BOUND, high=ACTION_BOUND, shape=(n_assets - 1,), dtype=np.float32
        )
        self.core: AllocationEnv | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "episode" in options:
            scenario_id, scenario_seed, start, stop = options["episode"]
        else:
            scenario_id, scenario_seed, start, stop = self.sampler.sample(self.np_random)
        self.core = AllocationEnv(
            self.sampler.data(scenario_id, scenario_seed),
            start=int(start),
            stop=int(stop),
            transition=self.transition,
            transition_info=self.transition_info,
            builder=self.builder,
            config=self.env_config,
        )
        observation, info = self.core.reset()
        vector, imputed = network_input(observation)
        info = {
            **info,
            "episode": (scenario_id, int(scenario_seed), int(start), int(stop)),
            "imputed_features": imputed,
        }
        return vector, info

    def step(self, action):
        if self.core is None:
            raise RuntimeError("Call reset() before step().")
        weights = fund_logits_to_weights(np.asarray(action), self.builder.cash_index)
        observation, reward, terminated, truncated, info = self.core.step(weights)
        vector, imputed = network_input(observation)
        info = {**info, "imputed_features": imputed}
        return vector, float(reward), terminated, truncated, info
