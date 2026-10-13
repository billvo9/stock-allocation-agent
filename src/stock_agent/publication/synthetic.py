"""
Restricted synthetic-fixture adapter: version 1's only producer of
publications.

A producer cannot hand this adapter data, a frame or a path. It names a
generator from a closed, read-only registry and gives integer and date
parameters; the adapter builds every row itself. Generators are pure and
bit-identical across platforms: they use the legacy RandomState stream,
integer arithmetic and one division, and never exp, log or normal draws,
whose last bit can differ between math libraries.

Provenance for the synthetic kinds is `verify_provenance`: it rebuilds the
rows from the recorded specification and requires identical content. This
is an adapter-level protection; real-data kinds need their own provenance
check and must not depend on it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from stock_agent.model_validation.audit import table_sha256
from stock_agent.publication.envelope import (
    Draft,
    Envelope,
    EvidenceStatus,
    Kind,
    KnowledgeTimeRule,
    Limitation,
    RefusalReason,
    RowReason,
    RowStatus,
    refuse,
)

UNIVERSE_ID = "synthetic"
ASSET_ID_PATTERN = re.compile(r"SYN_[A-Z0-9]{1,6}")  # applied with fullmatch
ASSET_IDS = tuple(f"SYN_{letter}" for letter in "ABCDEFGH")
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_LIMITATIONS = (Limitation.SYNTHETIC_CALENDAR, Limitation.SYNTHETIC_DATA)


@dataclass(frozen=True)
class SyntheticSpec:
    generator_id: str
    version: str
    params: Mapping[str, Any] = field(hash=False)

    def __post_init__(self) -> None:
        if not isinstance(self.params, Mapping):
            raise refuse(RefusalReason.INVALID_SPEC, "params must be a mapping")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, eq=False)
class _Generator:
    kind: Kind
    evidence_status: EvidenceStatus
    params: Mapping[str, Callable[[str, object], object]]
    build: Callable[[Mapping[str, Any]], pd.DataFrame]
    check: Callable[[Mapping[str, Any]], None] = lambda params: None


def _int_in(low: int, high: int) -> Callable[[str, object], object]:
    def check(name: str, value: object) -> object:
        if type(value) is not int or not low <= value <= high:
            raise refuse(RefusalReason.INVALID_SPEC, f"{name} must be an int in [{low}, {high}]")
        return value

    return check


def _weekday(name: str, value: object) -> object:
    if not isinstance(value, str) or not _DATE_PATTERN.fullmatch(value):
        raise refuse(RefusalReason.INVALID_SPEC, f"{name} must be a YYYY-MM-DD date")
    try:
        day = pd.Timestamp(value)
    except ValueError as error:
        raise refuse(RefusalReason.INVALID_SPEC, f"{name} {value!r} is not a date") from error
    if day.dayofweek >= 5:
        raise refuse(RefusalReason.INVALID_SPEC, f"{name} {value} is not a weekday")
    return value


def _session(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC").as_unit("us")


def _frame(
    dates: pd.DatetimeIndex,
    assets: np.ndarray,
    values: np.ndarray,
    column: str,
    status: np.ndarray,
    reason: list[str | None],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.Series(dates, dtype="datetime64[us, UTC]"),
            "asset_id": pd.Series(assets, dtype="str"),
            column: pd.Series(values, dtype="float64"),
            "status": pd.Series(status, dtype="str"),
            "reason": pd.Series(reason, dtype="str"),
            "detail": pd.Series([None] * len(dates), dtype="str"),
        }
    )


def _price_path_v1(params: Mapping[str, Any]) -> pd.DataFrame:
    """Integer cent random walks starting at 100.00, one per asset, on weekdays."""

    n_assets, n_sessions = params["n_assets"], params["n_sessions"]
    sessions = pd.bdate_range(_session(params["start"]), periods=n_sessions).as_unit("us")
    ticks = np.random.RandomState(params["seed"]).randint(
        -50, 51, size=(n_sessions, n_assets), dtype=np.int64
    )
    values = ((10_000 + np.cumsum(ticks, axis=0)) / 100.0).reshape(-1)
    assets = np.tile(np.array(ASSET_IDS[:n_assets], dtype=object), n_sessions)
    missing = (np.arange(len(values)) // n_assets < params["n_missing"]) & (assets == ASSET_IDS[0])
    values[missing] = np.nan
    status = np.where(missing, RowStatus.UNAVAILABLE.value, RowStatus.OK.value)
    reason = [RowReason.MISSING_INPUT.value if gap else None for gap in missing]
    return _frame(sessions.repeat(n_assets), assets, values, "value", status, reason)


def _check_price_path(params: Mapping[str, Any]) -> None:
    if params["n_missing"] > params["n_sessions"]:
        raise refuse(RefusalReason.INVALID_SPEC, "n_missing exceeds n_sessions")


def _equal_weight_v1(params: Mapping[str, Any]) -> pd.DataFrame:
    """Weights 1/n on one decision date: a declared rule with no inputs."""

    n_assets = params["n_assets"]
    dates = pd.DatetimeIndex([_session(params["decision_date"])] * n_assets)
    status = np.full(n_assets, RowStatus.OK.value, dtype=object)
    assets = np.array(ASSET_IDS[:n_assets], dtype=object)
    return _frame(
        dates, assets, np.full(n_assets, 1.0 / n_assets), "weight", status, [None] * n_assets
    )


_SEED = _int_in(0, 2**31 - 1)

# Closed and read-only: (generator_id, version) -> generator.
REGISTRY: Mapping[tuple[str, str], _Generator] = MappingProxyType(
    {
        ("syn_price_path", "1"): _Generator(
            kind=Kind.SYNTHETIC_DESCRIPTOR,
            evidence_status=EvidenceStatus.DESCRIPTIVE,
            params=MappingProxyType(
                {
                    "n_assets": _int_in(1, len(ASSET_IDS)),
                    "start": _weekday,
                    "n_sessions": _int_in(1, 520),
                    "seed": _SEED,
                    "n_missing": _int_in(0, 520),
                }
            ),
            build=_price_path_v1,
            check=_check_price_path,
        ),
        ("syn_equal_weight", "1"): _Generator(
            kind=Kind.SYNTHETIC_RULE_OUTPUT,
            evidence_status=EvidenceStatus.RULES_ILLUSTRATION,
            params=MappingProxyType(
                {"n_assets": _int_in(1, len(ASSET_IDS)), "decision_date": _weekday}
            ),
            build=_equal_weight_v1,
        ),
    }
)


def _generator_for(spec: object) -> _Generator:
    if not isinstance(spec, SyntheticSpec):
        raise refuse(RefusalReason.INVALID_SPEC, f"expected a SyntheticSpec, got {type(spec)}")
    generator = REGISTRY.get((spec.generator_id, spec.version))
    if generator is None:
        raise refuse(
            RefusalReason.INVALID_SPEC,
            f"unknown generator {spec.generator_id!r} version {spec.version!r}",
        )
    if set(spec.params) != set(generator.params):
        raise refuse(
            RefusalReason.INVALID_SPEC,
            f"params {sorted(spec.params)}, expected {sorted(generator.params)}",
        )
    for name, check in generator.params.items():
        check(name, spec.params[name])
    generator.check(spec.params)
    return generator


def draft(spec: SyntheticSpec) -> Draft:
    """Build a draft publication from a registered generator. No I/O."""

    generator = _generator_for(spec)
    return Draft(
        kind=generator.kind,
        evidence_status=generator.evidence_status,
        knowledge_time_rule=KnowledgeTimeRule.INTERIM_SYNTHETIC_SESSION_END,
        universe_id=UNIVERSE_ID,
        method_id=spec.generator_id,
        method_version=spec.version,
        parameters=dict(spec.params),
        limitations=_LIMITATIONS,
        input_hashes=(),
        rows=generator.build(spec.params),
    )


def verify_provenance(envelope: Envelope, rows: pd.DataFrame) -> None:
    """
    Rebuild the rows from the recorded specification; any difference in
    canonical content (`table_sha256`) refuses. This proves the rows derive
    from the recorded specification; it does not prove who wrote them.
    """

    if envelope.universe_id != UNIVERSE_ID or envelope.input_hashes != ():
        raise refuse(
            RefusalReason.PROVENANCE_MISMATCH,
            "synthetic publications have universe 'synthetic' and no external inputs",
        )
    spec = SyntheticSpec(envelope.method_id, envelope.method_version, envelope.parameters)
    rebuilt = draft(spec)
    if (rebuilt.kind, rebuilt.evidence_status, rebuilt.limitations) != (
        envelope.kind,
        envelope.evidence_status,
        envelope.limitations,
    ):
        raise refuse(RefusalReason.PROVENANCE_MISMATCH, "kind, status or limitations differ")
    if table_sha256(rebuilt.rows) != table_sha256(rows):
        raise refuse(RefusalReason.PROVENANCE_MISMATCH, "rows differ from the regenerated rows")
