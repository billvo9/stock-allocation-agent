"""
Registry stages, computed from recorded evidence (never stored, never set).

    engineering_only          a loadable, hash-verified entry (the floor)
    development_evaluated     evaluated on a registered research panel under
                              the development protocol (criteria below)
    historically_checked      proposed: one pre-registered holdout check (S1)
    prospectively_validated   proposed: a registered confirmatory test on
                              post-registration dates (S2/S3 model claim)
    approved_for_scope        proposed: an owner approval for one scope

A stage is reached only if every criterion of that stage and of every lower
stage is met. `assess` is pure: the store tells it whether the entry's
artifacts verified and what evidence the verified run record yields.

Fail closed. The evidence rules of historically_checked and above are new
evidence rules. They are written as proposals in docs/research/ml_registry.md
and their RULES entry is `approved=False`, so those stages are unattainable
whatever an entry records. Approving one is an owner decision; it changes
this table and adds that stage's criteria in the same reviewed change.

development_evaluated implements evidence rules already in the product
direction (docs/product/vision_and_prd.md, section 6, "Evidence required for
every rung above L0") and the T2/T4 protocol: a point-in-time,
survivorship-free research panel; development mode with the lockbox closed;
every leakage and instrument check passed; fold-block t as the decision
method; registered nulls with at least 199 permutation draws; a recorded
multiple-testing count. Synthetic and sandbox runs never reach it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from stock_agent.registry.entries import EntryRef, RegistryEntry

STAGES = (
    "engineering_only",
    "development_evaluated",
    "historically_checked",
    "prospectively_validated",
    "approved_for_scope",
)
FLOOR = STAGES[0]
DECISION_METHOD = "fold_block_t"
MIN_PERMUTATION_DRAWS = 199
RESEARCH_DATA_CLASS = "research_panel"


@dataclass(frozen=True)
class Rule:
    """Whether a stage's evidence rule is approved, and where it is written."""

    approved: bool
    basis: str


RULES: Mapping[str, Rule] = {
    "engineering_only": Rule(True, "floor: the entry loads and its hashes verify"),
    "development_evaluated": Rule(
        True,
        "docs/product/vision_and_prd.md section 6 (evidence above L0) and the T2/T4 "
        "protocol; criteria in docs/research/ml_registry.md",
    ),
    "historically_checked": Rule(
        False, "proposal only: docs/research/ml_registry.md, 'Proposed: historically_checked'"
    ),
    "prospectively_validated": Rule(
        False, "proposal only: docs/research/ml_registry.md, 'Proposed: prospectively_validated'"
    ),
    "approved_for_scope": Rule(
        False, "proposal only: docs/research/ml_registry.md, 'Proposed: approved_for_scope'"
    ),
}


@dataclass(frozen=True)
class Criterion:
    """One stage requirement and whether the entry meets it."""

    stage: str
    criterion: str
    met: bool
    reason: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class StageAssessment:
    """The computed stage of one entry and every criterion behind it."""

    ref: EntryRef
    stage: str
    criteria: tuple[Criterion, ...]

    @property
    def next_stage(self) -> str | None:
        index = STAGES.index(self.stage)
        return STAGES[index + 1] if index + 1 < len(STAGES) else None

    @property
    def blockers(self) -> tuple[Criterion, ...]:
        """Unmet criteria of the next stage (why the entry is not higher)."""

        return tuple(c for c in self.criteria if c.stage == self.next_stage and not c.met)

    def frame(self) -> pd.DataFrame:
        """One row per criterion (stage, criterion, met, reason, detail)."""

        return pd.DataFrame(
            [c.__dict__ for c in self.criteria],
            columns=["stage", "criterion", "met", "reason", "detail"],
        )


def _criterion(stage: str, name: str, met: bool, reason: str, detail: str | None = None):
    return Criterion(stage, name, bool(met), None if met else reason, detail)


def _before_holdout(lockbox: Mapping[str, object]) -> bool:
    """Every recorded data and label date is strictly before the holdout start."""

    try:
        holdout = pd.Timestamp(str(lockbox["holdout_start"]))
        latest = [pd.Timestamp(str(lockbox[key])) for key in ("max_date", "max_target_end_date")]
    except (KeyError, ValueError, TypeError):
        return False
    if holdout.tzinfo is None or any(stamp.tzinfo is None for stamp in latest):
        return False
    return all(stamp < holdout for stamp in latest)


def _development_criteria(
    entry: RegistryEntry,
    *,
    artifacts_verified: bool,
    rederived_evidence: Mapping[str, object] | None,
) -> list[Criterion]:
    stage = "development_evaluated"
    evidence = entry.evidence
    reproducible = rederived_evidence is not None and dict(rederived_evidence) == dict(evidence)
    lockbox = evidence.get("lockbox") or {}
    checks = evidence.get("checks") or {}
    nulls = evidence.get("nulls") or {}
    universe = evidence.get("universe") or {}
    testing = evidence.get("multiple_testing") or {}

    def family_clean(name: str) -> bool:
        counts = checks.get(name)
        return (
            isinstance(counts, Mapping)
            and counts.get("total", 0) > 0
            and counts.get("failed") == 0
            and counts.get("undetermined") == 0
        )

    closed = (
        evidence.get("mode") == "development"
        and lockbox.get("holdout_values_used") is False
        and _before_holdout(lockbox)
    )
    draws = nulls.get("permutation_draws")
    return [
        _criterion(stage, "artifacts_verified", artifacts_verified, "artifacts_unverified"),
        _criterion(
            stage,
            "evidence_reproducible",
            reproducible,
            "evidence_not_reproducible",
            "stored evidence must equal the evidence re-derived from the verified run record",
        ),
        _criterion(stage, "candidate_role", evidence.get("role") == "candidate", "not_candidate"),
        _criterion(
            stage,
            "research_data_class",
            evidence.get("data_class") == RESEARCH_DATA_CLASS,
            "data_class_engineering_only",
            f"data_class={evidence.get('data_class')!r}; synthetic and sandbox data verify the "
            "pipeline only",
        ),
        _criterion(
            stage,
            "point_in_time_survivorship_free_universe",
            universe.get("point_in_time") is True and universe.get("survivorship_free") is True,
            "universe_not_point_in_time",
        ),
        _criterion(stage, "development_lockbox_closed", closed, "lockbox_not_proven_closed"),
        _criterion(stage, "leakage_checks_passed", family_clean("leakage"), "leakage_checks"),
        _criterion(
            stage, "instrument_checks_passed", family_clean("instrument"), "instrument_checks"
        ),
        _criterion(
            stage,
            "decision_method_fold_block_t",
            evidence.get("decision_method") == DECISION_METHOD,
            "decision_method",
        ),
        _criterion(
            stage,
            "registered_nulls_compared",
            bool(nulls.get("models")) and isinstance(draws, int) and draws >= MIN_PERMUTATION_DRAWS,
            "nulls_insufficient",
            f"at least one registered null and >= {MIN_PERMUTATION_DRAWS} permutation draws "
            f"(recorded: {draws})",
        ),
        _criterion(
            stage,
            "multiple_testing_recorded",
            isinstance(testing.get("candidates_in_run"), int)
            and testing.get("candidates_in_run", 0) >= 1,
            "multiple_testing_not_recorded",
        ),
    ]


# Criteria per approved stage above the floor. A stage approved in RULES but
# missing here stays unattainable (criteria_not_implemented).
_STAGE_CRITERIA = {"development_evaluated": _development_criteria}


def assess(
    entry: RegistryEntry,
    *,
    artifacts_verified: bool,
    rederived_evidence: Mapping[str, object] | None,
) -> StageAssessment:
    """
    The highest stage whose criteria, and every lower stage's, are met.

    `artifacts_verified`: every artifact exists and matches its SHA-256.
    `rederived_evidence`: derive_evidence() of the verified run record, or
    None when it could not be re-derived.
    """

    criteria: list[Criterion] = [Criterion(FLOOR, "entry_hash_verified", True)]
    for stage in STAGES[1:]:
        rule = RULES[stage]
        if not rule.approved:
            criteria.append(
                _criterion(
                    stage, "owner_approved_evidence_rule", False, "rule_not_approved", rule.basis
                )
            )
            continue
        implementation = _STAGE_CRITERIA.get(stage)
        if implementation is None:  # approved but not implemented: still unattainable
            criteria.append(
                _criterion(stage, "criteria_implemented", False, "criteria_not_implemented")
            )
            continue
        criteria += implementation(
            entry, artifacts_verified=artifacts_verified, rederived_evidence=rederived_evidence
        )
    reached = FLOOR
    for stage in STAGES[1:]:
        own = [c for c in criteria if c.stage == stage]
        if not RULES[stage].approved or not own or not all(c.met for c in own):
            break
        reached = stage
    return StageAssessment(entry.ref, reached, tuple(criteria))
