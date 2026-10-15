"""
Frozen model bundles: the prospective forecasters as plain numbers.

A bundle directory (write-once, like every run output) holds

    models.json     every fit: per (asset, horizon, target) the ridge
                    statistics (training medians, means, scales, the
                    coefficients on the standardized features, the
                    intercept and penalty), the historical-mean null's value,
                    the band quantiles and the development coverage; plus the
                    config and its hash, the data fingerprints, the code
                    state and the registry versions the fits are registered as
    config.json     the registered config
    manifest.json   every file's SHA-256 (the registry's reference)

No pickle: a fit is reloaded from its numbers and predicts with exactly the
arithmetic of model_diagnostics.linear.LinearForecast.predict (tested equal),
so loading a bundle runs no stored code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stock_agent.model_validation.audit import file_sha256
from stock_agent.registry.store import MANIFEST_FILE, verify_manifest_directory

BUNDLE_SCHEMA_VERSION = "1.0"
MODELS_FILE = "models.json"
CONFIG_FILE = "config.json"
BUNDLE_KIND = "frozen_model_bundle"
STATUS = "prospective record pending"


class BundleError(ValueError):
    """A bundle is missing, edited or does not match what refers to it."""


@dataclass(frozen=True, eq=False)
class FrozenRidge:
    """A fitted LinearForecast's statistics, predicting exactly as it did."""

    features: tuple[str, ...]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray
    coef: np.ndarray
    intercept: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        matrix = np.asarray(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.features):
            raise ValueError(f"X must have one column per feature {list(self.features)}.")
        filled = np.where(np.isnan(matrix), self.medians, matrix)
        standardized = (filled - self.means) / self.scales
        return self.intercept + np.sum(standardized * self.coef, axis=1)


def ridge_json(estimator, features: Sequence[str]) -> dict[str, object]:
    """The numbers of a fitted LinearForecast (floats round-trip exactly in JSON)."""

    return {
        "features": list(features),
        "method": estimator.method,
        "penalty": estimator.penalty,
        "alpha": float(estimator.alpha_),
        "medians": [float(v) for v in estimator.medians_],
        "means": [float(v) for v in estimator.means_],
        "scales": [float(v) for v in estimator.scales_],
        "coef": [float(v) for v in estimator.coef_],
        "intercept": float(estimator.intercept_),
        "converged": bool(estimator.converged_),
    }


def ridge_from_json(payload: Mapping[str, object]) -> FrozenRidge:
    features = tuple(str(f) for f in payload["features"])
    arrays = [
        np.asarray(payload[k], dtype=np.float64) for k in ("medians", "means", "scales", "coef")
    ]
    if any(a.shape != (len(features),) for a in arrays):
        raise BundleError(
            "A frozen ridge needs one median, mean, scale and coefficient per feature."
        )
    if not all(np.isfinite(a).all() for a in arrays) or not (arrays[2] > 0).all():
        raise BundleError("A frozen ridge's statistics must be finite with positive scales.")
    return FrozenRidge(features, *arrays, intercept=float(payload["intercept"]))


@dataclass(frozen=True, eq=False)
class Bundle:
    path: Path
    manifest_sha256: str
    document: Mapping[str, object]

    @property
    def bundle_id(self) -> str:
        return str(self.document["bundle_id"])

    def fit(self, asset: str, horizon: int, target: str) -> Mapping[str, object]:
        for item in self.document["fits"]:
            if (item["asset"], item["horizon"], item["target"]) == (asset, horizon, target):
                return item
        raise BundleError(f"The bundle has no fit for {asset} h{horizon} {target}.")


def read_bundle(path: str | Path, expected_sha256: str | None = None) -> Bundle:
    """Read a bundle after verifying every file against its manifest."""

    directory = Path(path)
    manifest = directory / MANIFEST_FILE
    if directory.is_symlink() or not manifest.is_file():
        raise BundleError(f"{directory} is not a frozen bundle.")
    digest = file_sha256(manifest)
    if expected_sha256 is not None and digest != expected_sha256:
        raise BundleError(f"{directory}: the manifest does not match the registered SHA-256.")
    if not verify_manifest_directory(directory, digest):
        raise BundleError(f"{directory}: a file does not match its manifest (edited bundle).")
    listed = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    if set(listed) != {MODELS_FILE, CONFIG_FILE}:
        raise BundleError(f"{directory}: a bundle holds exactly {MODELS_FILE} and {CONFIG_FILE}.")
    document = json.loads((directory / MODELS_FILE).read_text(encoding="utf-8"))
    if document.get("bundle_schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"{directory}: bundle schema {document.get('bundle_schema_version')!r}.")
    if document.get("bundle_id") != directory.name:
        raise BundleError(f"{directory}: the bundle id differs from the directory name.")
    return Bundle(directory, digest, document)
