"""
The Stock Lab's daily-job record and its Model Health input (standard
library, numpy and pandas only).

`monitor_document` turns one job run into the `stock_lab_monitor` health
input (agreed envelope v1.0: artifact, schema_version, generated_at,
code_sha, synthetic, run_id) with the fields `stock_agent.health.stock_lab`
reads. `write_monitor` writes it once to
`<health_inputs>/stock_lab_monitor/<run_id>.json` (exclusive create, strict
JSON, mode 0600, in folders created with mode 0700). Health recomputes
nothing: every number here is measured by the job.

Where. The real job writes under `shared_inputs()`,
~/stock-agent-private/health_inputs (since 2026-10-21): outside every
checkout, so the evening run from a runner clone outside ~/Desktop and Model
Health run from the owner's own checkout read the same files. Before, it
wrote under the checkout it ran from (`reports/health_inputs/`), which the
synthetic demo replay still uses. STOCK_AGENT_STOCK_LAB_HEALTH_INPUTS moves
this writer's default only: the tests point it at a temporary folder, so no
test can write into the owner's. Model Health does not follow it (read such
a folder with `scripts/health_check.py local --stock-lab-inputs`), and the
evening's run.sh passes the folder explicitly, with no variable reaching it.

`symbols` (each watched ticker's listing status) is an optional, additive
field, added 2026-10-17. `outcomes` (the real job's scoring counts: scored,
unscorable, pending, missing and refused reads) is another, added 2026-10-19,
and so is the job's per-stock `results` ({ticker, status, reason}). `evenings`
(the real job's prospective record: {since, due, recorded, missed, recent,
recent_missed}, stock_lab.job.evening_record) is another, added 2026-10-20.
SCHEMA_VERSION stays "1.0" on purpose: the agreed
envelope ignores extra fields, so earlier files and readers stay valid,
whereas a new version would make the current reader (which accepts "1.0"
only) refuse every file. A reader that finds no `symbols` reports the
tickers' status as not recorded.

Input drift is the population stability index of a few model inputs over
the most recent RECENT sessions against the development distribution (bins
from development deciles; empty bins floored), a standard, distribution-free
shift measure; values above 0.25 are conventionally a material shift.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT = "stock_lab_monitor"
PRIVATE_FOLDER = "stock-agent-private"  # in the home folder: every private store and output
SHARED_INPUTS = "health_inputs"
HOME_ENV = "STOCK_AGENT_STOCK_LAB_HEALTH_INPUTS"  # moves shared_inputs() (module docstring)
PRIVATE_MODE = 0o700  # every folder the writer creates
SCHEMA_VERSION = "1.0"
RECENT = 63
DRIFT_FEATURES = ("ewma_vol", "ret_20", "rel_dollar_volume", "band_over_vol")
PSI_FLOOR = 1e-4


def drift_reference(development: np.ndarray, bins: int = 10) -> dict | None:
    """
    The development distribution PSI compares against: decile edges and the
    share of development values in each bin (None with too few values). A
    frozen model keeps one per input (registry), so a later drift check needs
    no development data.
    """

    development = development[np.isfinite(development)]
    if len(development) < 100:
        return None
    edges = np.unique(np.quantile(development, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    expected = np.bincount(
        np.searchsorted(edges[1:-1], development, side="right"), minlength=len(edges) - 1
    )
    return {"edges": edges.tolist(), "expected": (expected / expected.sum()).tolist()}


def psi_against(reference: Mapping | None, recent: np.ndarray) -> float | None:
    """Population stability index of `recent` against a `drift_reference`."""

    recent = recent[np.isfinite(recent)]
    if reference is None or len(recent) < 20:
        return None
    edges = np.asarray(reference["edges"], dtype="float64")
    actual = np.bincount(
        np.searchsorted(edges[1:-1], recent, side="right"), minlength=len(edges) - 1
    )
    e = np.maximum(np.asarray(reference["expected"], dtype="float64"), PSI_FLOOR)
    a = np.maximum(actual / actual.sum(), PSI_FLOOR)
    return float(((a - e) * np.log(a / e)).sum())


def psi(development: np.ndarray, recent: np.ndarray, bins: int = 10) -> float | None:
    """Population stability index of `recent` against `development` (decile bins)."""

    return psi_against(drift_reference(development, bins), recent)


def input_drift(frames: Mapping[str, pd.DataFrame], development_end: pd.Timestamp) -> list[dict]:
    rows = []
    for feature in DRIFT_FEATURES:
        development, recent = [], []
        for frame in frames.values():
            values = frame[feature].to_numpy(dtype="float64")
            sessions = pd.DatetimeIndex(frame["session"])
            development.append(values[np.asarray(sessions <= development_end)])
            recent.append(values[-RECENT:])
        value = psi(
            np.concatenate(development or [np.array([])]), np.concatenate(recent or [np.array([])])
        )
        rows.append({"feature": feature, "psi": value})
    return rows


def _clean(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    return value


def monitor_document(
    *,
    run_id: str,
    generated_at: pd.Timestamp,
    code_sha: str | None,
    synthetic: bool,
    job: Mapping,
    freshness: Sequence[Mapping],
    scores: Mapping,
    inputs_drift: Sequence[Mapping],
    traps: Mapping,
    ledger: Mapping,
    lineage: Mapping,
    schedule: Mapping,
    symbols: Mapping | None = None,
    outcomes: Mapping | None = None,
    evenings: Mapping | None = None,
) -> dict:
    """The health input; `symbols`, `outcomes` and `evenings` are optional (module docstring)."""

    extra = {} if symbols is None else {"symbols": dict(symbols)}
    if outcomes is not None:
        extra["outcomes"] = dict(outcomes)
    if evenings is not None:
        extra["evenings"] = dict(evenings)
    return _clean(
        {
            "artifact": ARTIFACT,
            "schema_version": SCHEMA_VERSION,
            "generated_at": generated_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "code_sha": code_sha,
            "synthetic": bool(synthetic),
            "run_id": run_id,
            "job": dict(job),
            "freshness": [dict(row) for row in freshness],
            "scores": dict(scores),
            "inputs_drift": [dict(row) for row in inputs_drift],
            "traps": dict(traps),
            "ledger": dict(ledger),
            "lineage": dict(lineage),
            "schedule": dict(schedule),
            **extra,
        }
    )


def shared_inputs(home: Path | None = None) -> Path:
    """
    The real job's health-inputs root: <home>/stock-agent-private/health_inputs
    for a given `home`; otherwise HOME_ENV when set, else under the user's home
    folder (module docstring). `stock_agent.health.stock_lab` and the
    evening's run.sh name the same folder (tested equal).
    """

    if home is None:
        configured = os.environ.get(HOME_ENV, "").strip()
        if configured:
            return Path(configured).expanduser()
    return (home or Path.home()) / PRIVATE_FOLDER / SHARED_INPUTS


def write_monitor(directory: Path, document: Mapping) -> Path:
    """
    Write the document once to <directory>/stock_lab_monitor/<run_id>.json
    (mode 0600). Every missing folder on the way, `directory` included, is
    created with mode 0700 whatever the umask; an existing one is never
    changed.
    """

    target = directory / ARTIFACT
    missing = []
    folder = target
    while not folder.exists():
        missing.append(folder)
        folder = folder.parent
    for folder in reversed(missing):
        folder.mkdir(mode=PRIVATE_MODE)
        os.chmod(folder, PRIVATE_MODE)  # mkdir's mode is masked by the umask
    path = target / f"{document['run_id']}.json"
    text = json.dumps(document, sort_keys=True, indent=1, allow_nan=False)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    return path
