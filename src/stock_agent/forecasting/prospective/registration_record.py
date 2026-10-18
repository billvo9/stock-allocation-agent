"""
The registration record and the ledger's registration entry
(docs/research/prospective_registration.md, sections 1 and 4). Built for
activation; nothing here registers, freezes or records by itself.

The record (one JSON file, added to the repository by a registration pull
request; default directory docs/research/registrations/):

    record_type, record_schema_version, experiment, assets
    bundle                   bundle_id, manifest_sha256, run_id, data_class
    registry                 "<target>:h<h>" -> model_id, model_version,
                             entry_sha256 (the six registry entries)
    config_sha256            the registered config's hash
    training_data_sha256     SHA-256 of the bundle's training-data
                             fingerprints (prices, distributions, splits,
                             features, universe, batch ids)
    band_method              the registered band method
    zero_volatility_rule     the config's rule, equal to every fit's rule
    code                     git_sha (the commit the bundle was frozen from)
                             and trees: the Git object id of each path in
                             CODE_PATHS at that commit
    registration_document    path and commit of the registration rules

It holds hashes and identifiers only: never vendor data, a forecast value or
the ledger head hash (section 6), so it may be committed publicly.

The registration entry (kind "registration", always the ledger's first entry,
one per ledger: a new registration starts a new ledger file, so two
registrations are never pooled):

    registered_at            GitHub's merge time of the registration pull
                             request: an explicit, timezone-aware input,
                             never a local commit date, and equal to the
                             evidence's `merged_at` to the second
    registration_commit      that pull request's merge commit
    registration_evidence    source (`github_api`, fetched by the job's
                             script, or `owner_evidence_file`, the same
                             fields supplied by the owner), fetched_at,
                             pull_request (number, merged, merged_at,
                             merge_commit_sha, base.ref, base.repo.full_name:
                             the fields checked, nothing else) and its sha256
    record_path, record_sha256, record

Registration time (section 1). A time without evidence is refused. The
evidence must show the pull request merged into `main` of REPOSITORY, with
`merge_commit_sha` equal to the registration commit and `merged_at` equal to
the registration time (`registration_unverified` otherwise), and it is
checked again whenever the entry is read. The evidence is what GitHub or the
owner reported, recorded as such; this module never calls a network. The
caller fetches it (the job's script through `gh api`, with the owner's
existing login; no token is recorded) and checks that the merge commit is in
the local origin/main history.

Code identity. A code change is a new registration (section 1), but an
unrelated commit is not: the job compares the Git object ids of CODE_PATHS
(the package, the job's script, the registered configs and the dependency
pins) in the running checkout with the record's, which are those of the
bundle's freeze commit, and refuses any difference or any uncommitted change
under them (`registration_mismatch`). The job's `register` step refuses a
merge commit whose paths differ from the freeze commit's, so a checkout of
the registration commit passes; so does a later commit that leaves those
paths alone. Any other change under them, however unrelated, needs a new
freeze and registration, so the job runs from a pinned checkout.

Checks against a registration (each a `ForecastRefused` with its reason):

    not_registered            the ledger's first entry is not a registration
    registration_not_first    a registration entry after other entries
    registration_not_unique   more than one registration entry
    registration_tampered     the record does not match its recorded hash, or
                              the entry is malformed
    registration_invalid      a malformed record, commit id or time
    registration_unverified   no evidence for the registration time, or
                              evidence that does not support it
    registration_mismatch     the bundle, registry entries, config, training
                              data, band method, rule or code differ from the
                              record: a new registration is needed
    code_not_committed        a bundle frozen from uncommitted code
    zero_vol_rule_mismatch    the bundle's rule differs from its config's
    zero_vol_rule_not_registrable   a research-only rule
    registered_before_freeze  a registration time before a registry entry
    registration_in_future    an entry appended before the registration time
    ledger_not_empty          a registration appended to a used ledger
    record_exists             a record file is written once

Limits. The chain cannot show a consistent rewrite of the whole ledger, the
registration entry included; the kept head (section 6) does, and the job's
script compares the entry's record with the committed record file. The
evidence is as trustworthy as its source: GitHub's API, or a file the owner
supplied.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd

from stock_agent.forecasting.prospective import ledger, zero_vol
from stock_agent.forecasting.prospective.config import parse_config
from stock_agent.forecasting.prospective.refusal import ForecastRefused
from stock_agent.model_diagnostics.record import canonical_json, sha256_json

RECORD_TYPE = "prospective_registration_record"
RECORD_SCHEMA_VERSION = "1.0"
ENTRY_TYPE = "prospective_registration"
ENTRY_SCHEMA_VERSION = "1.1"  # 1.1: the registration evidence
KIND = "registration"
REGISTRATION_DOCUMENT = "docs/research/prospective_registration.md"
RECORD_DIRECTORY = PurePosixPath("docs/research/registrations")
# The code that produces or scores forecasts, by repository path: a change under any of
# them is a new registration.
CODE_PATHS = (
    "src",
    "scripts/forecast_prospective.py",
    "configs/prospective",
    "pyproject.toml",
    "requirements.txt",
)
REGISTERED_AT_RULE = (
    "GitHub's merge time of the registration pull request, equal to the merged_at of the "
    "evidence recorded with it"
)
REPOSITORY = "billvo9/stock-allocation-agent"
BASE_BRANCH = "main"
EVIDENCE_SOURCES = ("github_api", "owner_evidence_file")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_KEYS = {
    "record_type",
    "record_schema_version",
    "experiment",
    "assets",
    "bundle",
    "registry",
    "config_sha256",
    "training_data_sha256",
    "band_method",
    "zero_volatility_rule",
    "code",
    "registration_document",
}
_ENTRY_KEYS = {
    "record_type",
    "registration_schema_version",
    "registered_at",
    "registered_at_rule",
    "registration_commit",
    "record_path",
    "record_sha256",
    "record",
    "registration_evidence",
}
_EVIDENCE_KEYS = {"source", "fetched_at", "pull_request", "sha256"}


@dataclass(frozen=True)
class Registration:
    """The ledger's registration entry, checked."""

    entry: ledger.LedgerEntry
    record: Mapping[str, object]
    registered_at: pd.Timestamp
    registration_commit: str

    @property
    def rule(self) -> str:
        return str(self.record["zero_volatility_rule"])

    @property
    def evidence_source(self) -> str:
        return str(self.entry.payload["registration_evidence"]["source"])

    def reference(self) -> dict[str, str]:
        """What each forecast and score entry keeps of its registration."""

        return {
            "entry_sha256": self.entry.entry_sha256,
            "record_sha256": record_sha256(self.record),
            "registration_commit": self.registration_commit,
            "registered_at": self.registered_at.isoformat(),
        }


def record_sha256(record: Mapping[str, object]) -> str:
    """The SHA-256 of the record's canonical JSON."""

    return sha256_json(record)


def _invalid(detail: str) -> ForecastRefused:
    return ForecastRefused("registration_invalid", detail)


def commit_id(value: object, name: str) -> str:
    if not isinstance(value, str) or not _COMMIT.fullmatch(value):
        raise _invalid(f"{name} must be a full 40-character lower-case Git commit id")
    return value


def _trees(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(CODE_PATHS):
        raise _invalid(f"code trees must name exactly {list(CODE_PATHS)}")
    for path, object_id in value.items():
        commit_id(object_id, f"the tree id of {path}")
    return {path: value[path] for path in CODE_PATHS}


def bundle_rule(bundle) -> str:
    """The zero-volatility rule every fit of the bundle was trained under."""

    rules = {
        fit.get("training", {}).get("zero_volatility_rule", zero_vol.CURRENT)
        for fit in bundle.document["fits"]
    }
    if len(rules) != 1:
        raise ForecastRefused("models_unverified", f"the fits disagree on the rule: {rules}")
    rule = rules.pop()
    if rule not in zero_vol.RULES:
        raise ForecastRefused("models_unverified", f"unknown zero-volatility rule {rule!r}")
    return rule


def checked_rule(bundle) -> str:
    """The bundle's rule, refused unless its config names the same rule."""

    rule = bundle_rule(bundle)
    named = parse_config(bundle.document["config"]).zero_volatility_rule
    if rule != named:
        raise ForecastRefused(
            "zero_vol_rule_mismatch",
            f"the bundle's fits were trained under {rule!r}, its config names {named!r}",
        )
    return rule


def build_record(
    models, *, code_trees: Mapping[str, str], registration_document_commit: str
) -> dict[str, object]:
    """
    The record for verified `models` (forecast.Models). `code_trees` are the
    Git object ids of CODE_PATHS at the bundle's commit; the registration
    document's commit is the one the owner approved.
    """

    bundle = models.bundle
    document = bundle.document
    config = parse_config(document["config"])
    rule = checked_rule(bundle)
    if not zero_vol.registrable(rule):
        raise ForecastRefused(
            "zero_vol_rule_not_registrable", f"{rule!r} is research only and is never registered"
        )
    code = document.get("code", {})
    if code.get("git_dirty") is not False:
        raise ForecastRefused(
            "code_not_committed",
            "the bundle was frozen from uncommitted code; freeze again from a clean commit",
        )
    pairs = [f"{t}:h{h}" for h in config.horizons for t in config.targets]
    if set(models.entries) != set(pairs):
        raise ForecastRefused("models_unverified", f"registry entries must be {pairs}")
    return {
        "record_type": RECORD_TYPE,
        "record_schema_version": RECORD_SCHEMA_VERSION,
        "experiment": config.experiment,
        "assets": list(config.assets),
        "bundle": {
            "bundle_id": bundle.bundle_id,
            "manifest_sha256": bundle.manifest_sha256,
            "run_id": document["run_id"],
            "data_class": document["data_class"],
        },
        "registry": {
            key: {
                "model_id": models.entries[key]["model_id"],
                "model_version": int(models.entries[key]["model_version"]),
                "entry_sha256": models.entries[key]["entry_sha256"],
            }
            for key in pairs
        },
        "config_sha256": document["config_sha256"],
        "training_data_sha256": sha256_json(document["training_data"]),
        "band_method": config.band.method,
        "zero_volatility_rule": rule,
        "code": {
            "git_sha": commit_id(code.get("git_sha"), "the bundle's code commit"),
            "trees": _trees(code_trees),
        },
        "registration_document": {
            "path": REGISTRATION_DOCUMENT,
            "commit": commit_id(registration_document_commit, "the registration document's commit"),
        },
    }


def validate_record(record: object) -> dict[str, object]:
    """A record's structure, refused (`registration_invalid`) unless exact."""

    if not isinstance(record, Mapping) or set(record) != _RECORD_KEYS:
        raise _invalid(f"a record holds exactly {sorted(_RECORD_KEYS)}")
    if (record["record_type"], record["record_schema_version"]) != (
        RECORD_TYPE,
        RECORD_SCHEMA_VERSION,
    ):
        raise _invalid("unknown record type or schema version")
    bundle = record["bundle"]
    if not isinstance(bundle, Mapping) or set(bundle) != {
        "bundle_id",
        "manifest_sha256",
        "run_id",
        "data_class",
    }:
        raise _invalid("bundle must hold bundle_id, manifest_sha256, run_id and data_class")
    for name in ("config_sha256", "training_data_sha256"):
        if not isinstance(record[name], str) or not _SHA256.fullmatch(record[name]):
            raise _invalid(f"{name} must be a SHA-256")
    if not isinstance(bundle["manifest_sha256"], str) or not _SHA256.fullmatch(
        bundle["manifest_sha256"]
    ):
        raise _invalid("bundle.manifest_sha256 must be a SHA-256")
    registry = record["registry"]
    if not isinstance(registry, Mapping) or len(registry) != 6:
        raise _invalid("registry must hold the six registry entries")
    for key, identity in registry.items():
        if not isinstance(identity, Mapping) or set(identity) != {
            "model_id",
            "model_version",
            "entry_sha256",
        }:
            raise _invalid(f"registry {key} must hold model_id, model_version and entry_sha256")
    rule = record["zero_volatility_rule"]
    if rule not in zero_vol.RULES or not zero_vol.registrable(rule):
        raise _invalid(f"zero_volatility_rule {rule!r} is not registrable")
    code = record["code"]
    if not isinstance(code, Mapping) or set(code) != {"git_sha", "trees"}:
        raise _invalid("code must hold git_sha and trees")
    commit_id(code["git_sha"], "code.git_sha")
    _trees(code["trees"])
    document = record["registration_document"]
    if not isinstance(document, Mapping) or set(document) != {"path", "commit"}:
        raise _invalid("registration_document must hold path and commit")
    if document["path"] != REGISTRATION_DOCUMENT:
        raise _invalid(f"registration_document.path must be {REGISTRATION_DOCUMENT}")
    commit_id(document["commit"], "registration_document.commit")
    if not isinstance(record["assets"], list) or not record["assets"]:
        raise _invalid("assets must be a non-empty list")
    return json.loads(canonical_json(record))


def check_record(record: Mapping[str, object], models) -> None:
    """Refuse `models` unless they are exactly what `record` registered."""

    expected = build_record(
        models,
        code_trees=record["code"]["trees"],
        registration_document_commit=record["registration_document"]["commit"],
    )
    differ = sorted(key for key in expected if expected[key] != record.get(key))
    if differ:
        raise ForecastRefused(
            "registration_mismatch",
            f"{', '.join(differ)} differ from the registration record; any change is a new "
            "registration (section 1)",
        )


def check_code(record: Mapping[str, object], code: Mapping[str, object]) -> None:
    """Refuse running code whose CODE_PATHS differ from the registered commit's."""

    trees = code.get("trees")
    if not isinstance(trees, Mapping):
        raise ForecastRefused(
            "registration_mismatch",
            f"the running code's tree ids for {list(CODE_PATHS)} were not given",
        )
    differ = sorted(path for path in CODE_PATHS if trees.get(path) != record["code"]["trees"][path])
    if differ:
        raise ForecastRefused(
            "registration_mismatch",
            f"the running code differs from the registered code under {differ}; a code change "
            "is a new registration (section 1)",
        )
    if code.get("trees_dirty") is not False:
        raise ForecastRefused(
            "registration_mismatch",
            f"uncommitted changes under {list(CODE_PATHS)}: run the job from a clean checkout",
        )


def write_record(path: str | Path, record: Mapping[str, object]) -> Path:
    """Write a record file once (indented canonical JSON); refuses an existing file."""

    path = Path(path)
    body = validate_record(record)
    text = json.dumps(body, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError:
        raise ForecastRefused(
            "record_exists", f"{path} exists; a registration record is written once"
        ) from None
    return path


def read_record(path: str | Path) -> dict[str, object]:
    """A record file, validated."""

    def refuse_constant(token: str) -> None:
        raise _invalid(f"the record holds a non-finite number ({token})")

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=refuse_constant)
    except json.JSONDecodeError as error:
        raise _invalid(f"{path} is not JSON ({error.msg})") from None
    return validate_record(raw)


def _stamp(value: object, name: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise _invalid(f"{name} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _record_path(value: str | PurePosixPath) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise _invalid("record_path must be a path relative to the repository root")
    return path.as_posix()


def _unverified(detail: str) -> ForecastRefused:
    return ForecastRefused("registration_unverified", detail)


def pull_request_fields(pull_request: object) -> dict[str, object]:
    """The fields of a GitHub pull request that the registration checks, and nothing else."""

    if not isinstance(pull_request, Mapping):
        raise _unverified("the evidence holds no pull request")
    base = pull_request.get("base")
    repo = base.get("repo") if isinstance(base, Mapping) else None
    try:
        fields = {
            "number": pull_request["number"],
            "merged": pull_request["merged"],
            "merged_at": pull_request["merged_at"],
            "merge_commit_sha": pull_request["merge_commit_sha"],
            "base": {"ref": base["ref"], "repo": {"full_name": repo["full_name"]}},
        }
    except (KeyError, TypeError):
        raise _unverified(
            "the pull request lacks number, merged, merged_at, merge_commit_sha, base.ref or "
            "base.repo.full_name"
        ) from None
    if isinstance(fields["number"], bool) or not isinstance(fields["number"], int):
        raise _unverified("the pull request number is not an integer")
    return json.loads(canonical_json(fields))


def verify_pull_request(
    fields: Mapping[str, object], *, registration_commit: str, registered_at: pd.Timestamp
) -> None:
    """Refuse a registration time or commit that the pull request does not support."""

    base = fields["base"]
    if fields["merged"] is not True:
        raise _unverified(f"pull request #{fields['number']} is not merged")
    if base["repo"]["full_name"] != REPOSITORY:
        raise _unverified(f"pull request #{fields['number']} is not in {REPOSITORY}")
    if base["ref"] != BASE_BRANCH:
        raise _unverified(f"pull request #{fields['number']} was merged into {base['ref']!r}")
    if fields["merge_commit_sha"] != registration_commit:
        raise _unverified(
            f"pull request #{fields['number']} merged as {fields['merge_commit_sha']}, not "
            f"{registration_commit}"
        )
    merged = fields["merged_at"]
    try:
        merged_at = pd.Timestamp(merged)
    except (ValueError, TypeError):
        raise _unverified(f"merged_at {merged!r} is not a time") from None
    if not isinstance(merged, str) or merged_at.tzinfo is None:
        raise _unverified(f"merged_at {merged!r} is not a time with an explicit offset")
    if merged_at.tz_convert("UTC") != pd.Timestamp(registered_at).tz_convert("UTC"):
        raise _unverified(
            f"pull request #{fields['number']} merged at {merged}, not at the registration "
            f"time {pd.Timestamp(registered_at).isoformat()}"
        )


def registration_evidence(
    evidence: Mapping[str, object] | None,
    *,
    registration_commit: str,
    registered_at: pd.Timestamp,
    recorded_at: pd.Timestamp,
) -> dict[str, object]:
    """
    The entry's evidence: source, fetch time, the checked fields and their
    hash. It must have been fetched after the merge and no later than the
    entry was appended (registered_at <= fetched_at <= recorded_at).
    """

    if not isinstance(evidence, Mapping):
        raise _unverified(
            "no evidence: a registration time is accepted only with the pull request's "
            "merged_at from GitHub or from the owner's evidence file"
        )
    if evidence.get("source") not in EVIDENCE_SOURCES:
        raise _unverified(f"the evidence source must be one of {EVIDENCE_SOURCES}")
    try:
        fetched = pd.Timestamp(evidence.get("fetched_at"))
    except (ValueError, TypeError):
        fetched = pd.NaT
    if pd.isna(fetched) or fetched.tzinfo is None:
        raise _unverified("the evidence's fetched_at must be a timezone-aware time")
    if not pd.Timestamp(registered_at) <= fetched <= pd.Timestamp(recorded_at):
        raise _unverified(
            f"the evidence was fetched at {fetched.isoformat()}: not between the merge "
            f"({pd.Timestamp(registered_at).isoformat()}) and the entry "
            f"({pd.Timestamp(recorded_at).isoformat()})"
        )
    fields = pull_request_fields(evidence.get("pull_request"))
    verify_pull_request(
        fields, registration_commit=registration_commit, registered_at=registered_at
    )
    return {
        "source": evidence["source"],
        "fetched_at": fetched.tz_convert("UTC").isoformat(),
        "pull_request": fields,
        "sha256": sha256_json(fields),
    }


def register(
    ledger_path: str | Path,
    record: Mapping[str, object],
    models,
    *,
    record_path: str | PurePosixPath,
    registration_commit: str,
    registered_at: pd.Timestamp,
    recorded_at: pd.Timestamp,
    code: Mapping[str, object],
    evidence: Mapping[str, object] | None,
) -> ledger.LedgerEntry:
    """
    Append the registration entry: the first entry of a new ledger. The
    record must match `models` and the running `code`; `registered_at` is
    GitHub's merge time of the registration pull request, which `evidence`
    (source, fetched_at and the pull request as GitHub or the owner reported
    it) must support exactly.
    """

    body = validate_record(record)
    check_record(body, models)
    check_code(body, code)
    commit = commit_id(registration_commit, "registration_commit")
    when = _stamp(registered_at, "registered_at")
    now = _stamp(recorded_at, "recorded_at")
    if now < when:
        raise ForecastRefused(
            "registration_in_future",
            f"the registration time {when.isoformat()} is after now ({now.isoformat()})",
        )
    for key, identity in models.entries.items():
        if pd.Timestamp(identity["registered_at"]) > when:
            raise ForecastRefused(
                "registered_before_freeze",
                f"{key} entered the registry at {identity['registered_at']}, after the "
                f"registration time {when.isoformat()}: freeze first, then register",
            )
    payload = {
        "record_type": ENTRY_TYPE,
        "registration_schema_version": ENTRY_SCHEMA_VERSION,
        "registered_at": when.isoformat(),
        "registered_at_rule": REGISTERED_AT_RULE,
        "registration_commit": commit,
        "record_path": _record_path(record_path),
        "record_sha256": record_sha256(body),
        "record": body,
        "registration_evidence": registration_evidence(
            evidence, registration_commit=commit, registered_at=when, recorded_at=now
        ),
    }

    def guard(entries: tuple[ledger.LedgerEntry, ...]) -> None:
        if entries:
            raise ForecastRefused(
                "ledger_not_empty",
                f"the ledger holds {len(entries)} entries; a registration is the first entry "
                "of its ledger, and a new registration starts a new ledger",
            )

    return ledger.append_entry(ledger_path, KIND, payload, recorded_at=now, guard=guard)


def current(entries: Sequence[ledger.LedgerEntry]) -> Registration | None:
    """
    The ledger's registration, checked; None for an empty ledger. A ledger
    whose first entry is not a registration is refused.
    """

    if not entries:
        return None
    found = [entry for entry in entries if entry.kind == KIND]
    if entries[0].kind != KIND:
        if found:
            raise ForecastRefused(
                "registration_not_first",
                f"entry {found[0].sequence} is a registration; it must be the ledger's first",
            )
        raise ForecastRefused(
            "not_registered",
            "the ledger's first entry is not a registration: nothing in it was recorded under "
            "a registration",
        )
    if len(found) > 1:
        raise ForecastRefused(
            "registration_not_unique",
            f"the ledger holds {len(found)} registration entries; one registration per ledger",
        )
    entry = entries[0]
    payload = entry.payload
    if (
        payload.get("registration_schema_version") == "1.0"
        and "registration_evidence" not in payload
    ):
        raise _unverified(
            "the registration entry predates the evidence requirement (schema 1.0, a typed time "
            "only): register again in a new ledger, with the pull request's evidence"
        )
    if set(payload) != _ENTRY_KEYS or payload.get("record_type") != ENTRY_TYPE:
        raise ForecastRefused("registration_tampered", "the registration entry is malformed")
    try:
        body = validate_record(payload["record"])
    except ForecastRefused as error:
        raise ForecastRefused("registration_tampered", error.detail) from None
    if record_sha256(body) != payload["record_sha256"]:
        raise ForecastRefused(
            "registration_tampered", "the registration record does not match its recorded hash"
        )
    try:
        when = _stamp(payload["registered_at"], "registered_at")
        commit = commit_id(payload["registration_commit"], "registration_commit")
        kept = payload["registration_evidence"]
        if not isinstance(kept, Mapping) or set(kept) != _EVIDENCE_KEYS:
            raise _unverified("the registration evidence is malformed")
        fields = pull_request_fields(kept["pull_request"])
        if fields != kept["pull_request"] or sha256_json(fields) != kept["sha256"]:
            raise _unverified("the registration evidence does not match its recorded hash")
        rebuilt = registration_evidence(
            {**kept, "pull_request": fields},
            registration_commit=commit,
            registered_at=when,
            recorded_at=pd.Timestamp(entry.recorded_at),
        )
        if rebuilt != dict(kept):
            raise _unverified("the registration evidence was rewritten")
    except (ForecastRefused, ValueError, TypeError) as error:
        raise ForecastRefused("registration_tampered", str(error)) from None
    return Registration(entry, body, when, commit)


__all__ = [
    "CODE_PATHS",
    "KIND",
    "RECORD_DIRECTORY",
    "REGISTRATION_DOCUMENT",
    "Registration",
    "build_record",
    "bundle_rule",
    "check_code",
    "check_record",
    "checked_rule",
    "commit_id",
    "current",
    "read_record",
    "record_sha256",
    "register",
    "validate_record",
    "write_record",
]
