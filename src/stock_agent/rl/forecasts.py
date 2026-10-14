"""
Forecast inputs for learned policies: interface only.

PPO with forecasts in its state is a registered comparison ("what do the
forecasts add?"), but it needs versioned, out-of-fold, point-in-time forecast
artifacts from the ML runner and registry (Track B), which do not exist
yet. Until they do, the harness reports the method as unavailable rather
than running it on anything weaker.

A provider is accepted only when its artifact is certified out-of-fold (no
forecast was fitted on the period it predicts) and point-in-time (each
forecast is available at the decision session that reads it).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from stock_agent.rl.splits import LeakageError


class ForecastUnavailableError(RuntimeError):
    """No certified forecast artifact is available for this method."""


@dataclass(frozen=True)
class ForecastArtifact:
    artifact_id: str
    version: str
    out_of_fold: bool
    point_in_time: bool


class ForecastProvider(Protocol):
    artifact: ForecastArtifact

    def forecasts_at(self, scenario_id: str, seed: int, decision_session: int) -> np.ndarray: ...


def require_certified(provider: ForecastProvider) -> ForecastProvider:
    artifact = provider.artifact
    if not artifact.out_of_fold:
        raise LeakageError(f"Forecast artifact {artifact.artifact_id} is not out-of-fold.")
    if not artifact.point_in_time:
        raise LeakageError(f"Forecast artifact {artifact.artifact_id} is not point-in-time.")
    return provider


def unavailable(source: str) -> Callable[[], object]:
    """A policy factory that reports why the forecast method cannot run yet."""

    def factory():
        raise ForecastUnavailableError(
            f"No versioned out-of-fold forecast artifact is available yet ({source})."
        )

    return factory
