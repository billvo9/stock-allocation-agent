"""
The closed per-kind policy table and the one validation shared by the
writer and the reader.

`validate` runs every semantic check in a fixed order and runs the kind's
provenance verifier last, so each earlier check is exercised on its own.
The writer calls it before staging anything; the reader calls it after the
integrity checks. The reader is therefore at least as strict as the writer.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from stock_agent.publication import synthetic
from stock_agent.publication.envelope import (
    SCHEMA_VERSION,
    ClaimType,
    Envelope,
    EvidenceStatus,
    Kind,
    KnowledgeTimeRule,
    Limitation,
    RefusalReason,
    RowReason,
    RowStatus,
    check_code_sha,
    check_lockbox,
    refuse,
    synthetic_session_end,
)

KEY_COLUMNS = ("date", "asset_id")
TRAILING_COLUMNS = ("status", "reason", "detail")
WEIGHT_TOLERANCE = 1e-12


@dataclass(frozen=True, eq=False)
class Publication:
    """A publication that passed every check. Rows are a private copy."""

    envelope: Envelope
    rows: pd.DataFrame
    path: Path

    @property
    def claim_type(self) -> ClaimType:
        return POLICY[self.envelope.kind].claim_type


class DescriptorPublication(Publication):
    """Rows: date, asset_id, value, status, reason, detail."""


class RuleOutputPublication(Publication):
    """Rows: date, asset_id, weight, status, reason, detail."""


@dataclass(frozen=True, eq=False)
class KindPolicy:
    claim_type: ClaimType
    allowed_status: EvidenceStatus
    knowledge_time_rule: KnowledgeTimeRule
    knowledge_time: Callable[[pd.Timestamp], pd.Timestamp]
    required_limitations: frozenset[Limitation]
    value_column: str
    asset_id_pattern: re.Pattern[str]
    verify_provenance: Callable[[Envelope, pd.DataFrame], None]
    publication_type: type[Publication]

    @property
    def columns(self) -> tuple[str, ...]:
        return (*KEY_COLUMNS, self.value_column, *TRAILING_COLUMNS)


# Version 1 admits synthetic kinds only. A real-data kind needs its own row,
# its own provenance check and the production availability rule (OD9).
POLICY: Mapping[Kind, KindPolicy] = MappingProxyType(
    {
        Kind.SYNTHETIC_DESCRIPTOR: KindPolicy(
            claim_type=ClaimType.DESCRIPTIVE_FACT,
            allowed_status=EvidenceStatus.DESCRIPTIVE,
            knowledge_time_rule=KnowledgeTimeRule.INTERIM_SYNTHETIC_SESSION_END,
            knowledge_time=synthetic_session_end,
            required_limitations=frozenset({Limitation.SYNTHETIC_DATA}),
            value_column="value",
            asset_id_pattern=synthetic.ASSET_ID_PATTERN,
            verify_provenance=synthetic.verify_provenance,
            publication_type=DescriptorPublication,
        ),
        Kind.SYNTHETIC_RULE_OUTPUT: KindPolicy(
            claim_type=ClaimType.RULE_ILLUSTRATION,
            allowed_status=EvidenceStatus.RULES_ILLUSTRATION,
            knowledge_time_rule=KnowledgeTimeRule.INTERIM_SYNTHETIC_SESSION_END,
            knowledge_time=synthetic_session_end,
            required_limitations=frozenset({Limitation.SYNTHETIC_DATA}),
            value_column="weight",
            asset_id_pattern=synthetic.ASSET_ID_PATTERN,
            verify_provenance=synthetic.verify_provenance,
            publication_type=RuleOutputPublication,
        ),
    }
)


def policy_for(kind: object) -> KindPolicy:
    policy = POLICY.get(kind) if isinstance(kind, Kind) else None
    if policy is None:
        raise refuse(RefusalReason.UNSUPPORTED_KIND, f"kind {kind!r}")
    return policy


def claim_type(kind: Kind) -> ClaimType:
    return policy_for(kind).claim_type


def knowledge_time_for(kind: Kind, rows: pd.DataFrame) -> pd.Timestamp:
    """The latest information date in the rows, under the kind's rule."""

    if len(rows) == 0:
        raise refuse(RefusalReason.INVALID_ROW, "a publication needs at least one row")
    return policy_for(kind).knowledge_time(rows["date"].max())


def check_schema(policy: KindPolicy, rows: object) -> pd.DataFrame:
    if not isinstance(rows, pd.DataFrame):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "rows must be a DataFrame")
    if rows.attrs:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "rows carry pandas attrs, which no hash covers")
    if not rows.index.equals(pd.RangeIndex(len(rows))):
        raise refuse(RefusalReason.SCHEMA_MISMATCH, "rows need a default RangeIndex")
    if tuple(rows.columns) != policy.columns:
        raise refuse(
            RefusalReason.SCHEMA_MISMATCH,
            f"columns {list(rows.columns)}, expected {list(policy.columns)}",
        )
    dates = rows["date"].dtype
    if not (
        isinstance(dates, pd.DatetimeTZDtype) and dates.unit == "us" and str(dates.tz) == "UTC"
    ):
        raise refuse(
            RefusalReason.SCHEMA_MISMATCH, f"date dtype {dates}, expected UTC microseconds"
        )
    if rows[policy.value_column].dtype != np.float64:
        raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{policy.value_column} must be float64")
    for column in ("asset_id", *TRAILING_COLUMNS):
        if not pd.api.types.is_string_dtype(rows[column].dtype):
            raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{column} must be a string column")
        present = rows[column].dropna()
        if not all(isinstance(value, str) for value in present):
            raise refuse(RefusalReason.SCHEMA_MISMATCH, f"{column} holds non-string values")
    return rows


def check_rows(policy: KindPolicy, rows: pd.DataFrame) -> None:
    """Row invariants. Rows are never dropped or repaired: any failure refuses."""

    if len(rows) == 0:
        raise refuse(RefusalReason.INVALID_ROW, "a publication needs at least one row")
    if rows["asset_id"].isna().any():
        raise refuse(RefusalReason.INVALID_ASSET_ID, "missing asset_id")
    matched = np.array([bool(policy.asset_id_pattern.fullmatch(a)) for a in rows["asset_id"]])
    if not matched.all():
        bad = sorted(set(rows["asset_id"][~matched]))
        raise refuse(RefusalReason.INVALID_ASSET_ID, f"asset ids {bad[:5]}")
    dates = rows["date"]
    if dates.isna().any() or not (dates == dates.dt.normalize()).all():
        raise refuse(RefusalReason.INVALID_ROW, "dates must be session dates at 00:00 UTC")
    keys = list(zip(dates, rows["asset_id"], strict=True))
    if keys != sorted(set(keys)):
        raise refuse(RefusalReason.INVALID_ROW, "rows must be unique and sorted by date, asset")
    status = rows["status"]
    if not status.isin([member.value for member in RowStatus]).all():
        raise refuse(RefusalReason.INVALID_ROW, "unknown row status")
    reason = rows["reason"]
    if not (reason.isna() | reason.isin([member.value for member in RowReason])).all():
        raise refuse(RefusalReason.INVALID_ROW, "unknown row reason")
    values = rows[policy.value_column].to_numpy(dtype=np.float64)
    ok = (status == RowStatus.OK.value).to_numpy()
    if not (np.isfinite(values[ok]).all() and reason[ok].isna().all()):
        raise refuse(RefusalReason.INVALID_ROW, "an ok row needs a finite value and no reason")
    if not (np.isnan(values[~ok]).all() and reason[~ok].notna().all()):
        raise refuse(RefusalReason.INVALID_ROW, "an unavailable row needs a reason and no value")
    if policy.value_column == "weight":
        _check_weights(rows, values, ok)


def _check_weights(rows: pd.DataFrame, weights: np.ndarray, ok: np.ndarray) -> None:
    if ((weights[ok] < 0.0) | (weights[ok] > 1.0)).any():
        raise refuse(RefusalReason.INVALID_ROW, "weights must lie in [0, 1]")
    totals = pd.Series(np.where(ok, weights, 0.0)).groupby(rows["date"].to_numpy()).sum()
    if (np.abs(totals.to_numpy() - 1.0) > WEIGHT_TOLERANCE).any():
        raise refuse(RefusalReason.INVALID_ROW, "available weights must sum to 1 on each date")


def validate(envelope: Envelope, rows: pd.DataFrame) -> None:
    """Every semantic check, in a fixed order, with provenance last."""

    if envelope.schema_version != SCHEMA_VERSION:
        raise refuse(RefusalReason.UNSUPPORTED_VERSION, envelope.schema_version)
    policy = policy_for(envelope.kind)
    if envelope.evidence_status != policy.allowed_status:
        raise refuse(
            RefusalReason.STATUS_NOT_ALLOWED,
            f"{envelope.evidence_status} is not allowed for {envelope.kind}",
        )
    if envelope.knowledge_time_rule != policy.knowledge_time_rule:
        raise refuse(RefusalReason.KNOWLEDGE_RULE_NOT_PERMITTED, envelope.knowledge_time_rule)
    missing = policy.required_limitations - set(envelope.limitations)
    if missing:
        raise refuse(RefusalReason.MISSING_LIMITATION, f"missing {sorted(missing)}")
    if envelope.code_dirty and envelope.evidence_status != EvidenceStatus.DESCRIPTIVE:
        raise refuse(RefusalReason.DIRTY_TREE, "a dirty tree is refused above descriptive")
    check_code_sha(envelope.code_sha)
    check_rows(policy, check_schema(policy, rows))
    latest = rows["date"].max()
    check_lockbox("latest row date", latest)
    check_lockbox("as_of", envelope.as_of)
    check_lockbox("knowledge_time", envelope.knowledge_time)
    if envelope.knowledge_time != policy.knowledge_time(latest):
        raise refuse(
            RefusalReason.KNOWLEDGE_TIME_MISMATCH,
            f"knowledge_time {envelope.knowledge_time} does not follow from {latest}",
        )
    if envelope.knowledge_time > envelope.as_of:
        raise refuse(RefusalReason.KNOWLEDGE_AFTER_AS_OF, "knowledge_time is after as_of")
    if envelope.as_of > envelope.created_at:
        raise refuse(RefusalReason.AS_OF_AFTER_CREATED_AT, "as_of is after created_at")
    policy.verify_provenance(envelope, rows)
