"""
The registration record and the ledger's registration entry
(docs/research/prospective_registration.md, sections 1 and 4).

The record holds the frozen bundle's manifest SHA-256, its six registry
entries, the config hash, the code commit and tree ids, the training-data
fingerprint, the band method, the zero-volatility rule and the commit of the
registration document: never vendor data or the ledger head hash. The
registration entry is the ledger's first entry; its time is GitHub's merge
time of the registration pull request, an explicit input. The forecast job
refuses a forecast before a registration, before the first eligible close,
from anything that differs from the record, and from a tampered entry.
"""

from __future__ import annotations

import copy
import dataclasses
import json

import pandas as pd
import pytest
from forecast_prospective_helpers import (
    AS_OF,
    ASSETS,
    CODE_TREES,
    CREATED,
    DOCUMENT_COMMIT,
    RECORD_PATH,
    REGISTERED_RECORDED,
    REGISTRATION_COMMIT,
    REGISTRATION_TIME,
    RUN_CODE,
    RUN_ENVIRONMENT,
    evidence,
    exchange_calendar,
    github_pull_request,
    prospective_store,
    record,
    register,
    registration,
    score,
    without_close,
)

from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.forecasting.prospective import forecast, ledger, registration_record, zero_vol
from stock_agent.forecasting.prospective.config import parse_config
from stock_agent.model_diagnostics.record import canonical_json, sha256_json

HEX64 = set("0123456789abcdef")


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "private" / "ledger.jsonl"


@pytest.fixture
def store(prospective_store_root):
    return prospective_store_root


@pytest.fixture
def models(prospective_frozen):
    return prospective_frozen.models()


def _refused(call, *args, **kwargs) -> str:
    with pytest.raises(forecast.ForecastRefused) as refused:
        call(*args, **kwargs)
    return refused.value.reason


def _forecast(
    models, ledger_path, store, *, as_of=AS_OF, created=CREATED, code=RUN_CODE, loader=None
):
    return forecast.record_forecast(
        models,
        loader=load_inference_window if loader is None else loader,
        store_root=store,
        forecast_close=as_of,
        ledger_path=ledger_path,
        created_at=pd.Timestamp(created),
        code=code,
        calendar=exchange_calendar,
        environment=RUN_ENVIRONMENT,
    )


def _kinds(path):
    return [entry.kind for entry in ledger.read_ledger(path)]


# --- the record ---------------------------------------------------------------------------


def test_the_record_holds_every_registered_hash(prospective_frozen, models):
    record_ = registration(models)
    bundle = prospective_frozen.result.bundle
    assert record_["record_type"] == "prospective_registration_record"
    assert record_["bundle"]["bundle_id"] == bundle.bundle_id
    assert record_["bundle"]["manifest_sha256"] == bundle.manifest_sha256
    entries = {str(e.ref): e.entry_sha256 for e in prospective_frozen.result.entries}
    assert len(record_["registry"]) == 6 == len(entries)
    for identity in record_["registry"].values():
        assert (
            entries[f"{identity['model_id']}@v{identity['model_version']}"]
            == (identity["entry_sha256"])
        )
    assert record_["config_sha256"] == prospective_frozen.config.sha256
    assert record_["code"] == {"git_sha": "0" * 40, "trees": CODE_TREES}
    assert record_["zero_volatility_rule"] == zero_vol.CURRENT
    assert record_["band_method"] == "split_conformal_signed_v1"
    assert record_["training_data_sha256"] == sha256_json(bundle.document["training_data"])
    assert record_["registration_document"] == {
        "path": "docs/research/prospective_registration.md",
        "commit": DOCUMENT_COMMIT,
    }
    assert record_["assets"] == list(ASSETS)
    assert set(registration_record.record_sha256(record_)) <= HEX64


def test_the_record_holds_no_vendor_data_or_head_hash(models):
    record_ = registration(models)
    text = canonical_json(record_)
    for word in ("close", "price", "amount", "head", "ledger", "batch_id", "retrieved"):
        assert word not in text, word


def test_the_record_file_is_written_once_and_read_back(models, tmp_path):
    record_ = registration(models)
    path = registration_record.write_record(tmp_path / "registrations" / "r.json", record_)
    assert registration_record.read_record(path) == record_
    assert json.loads(path.read_text(encoding="utf-8")) == record_
    assert _refused(registration_record.write_record, path, record_) == "record_exists"


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r.pop("band_method"),
        lambda r: r.update(extra="value"),
        lambda r: r["code"].update(trees={}),
        lambda r: r["registry"].popitem(),
        lambda r: r.update(zero_volatility_rule=zero_vol.B2),
        lambda r: r["registration_document"].update(commit="not a commit"),
    ],
)
def test_a_malformed_record_file_is_refused(models, tmp_path, edit):
    record_ = registration(models)
    edit(record_)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(record_), encoding="utf-8")
    assert _refused(registration_record.read_record, path) == "registration_invalid"


def test_a_bundle_frozen_from_uncommitted_code_cannot_be_registered(models):
    document = copy.deepcopy(dict(models.bundle.document))
    document["code"] = {**document["code"], "git_dirty": True}
    dirty = forecast.Models(dataclasses.replace(models.bundle, document=document), models.entries)
    reason = _refused(
        registration_record.build_record,
        dirty,
        code_trees=CODE_TREES,
        registration_document_commit=DOCUMENT_COMMIT,
    )
    assert reason == "code_not_committed"


def test_a_bundle_whose_rule_differs_from_its_config_cannot_be_registered(models):
    document = copy.deepcopy(dict(models.bundle.document))
    for fit in document["fits"]:
        fit["training"]["zero_volatility_rule"] = zero_vol.B2
    other = forecast.Models(dataclasses.replace(models.bundle, document=document), models.entries)
    reason = _refused(
        registration_record.build_record,
        other,
        code_trees=CODE_TREES,
        registration_document_commit=DOCUMENT_COMMIT,
    )
    assert reason == "zero_vol_rule_mismatch"


def test_the_record_needs_every_registered_code_path(models):
    partial = dict(CODE_TREES)
    partial.pop("src")
    reason = _refused(
        registration_record.build_record,
        models,
        code_trees=partial,
        registration_document_commit=DOCUMENT_COMMIT,
    )
    assert reason == "registration_invalid"


# --- the registration entry ---------------------------------------------------------------


def test_the_registration_entry_points_to_the_commit_and_holds_the_record(models, ledger_path):
    entry = register(models, ledger_path)
    assert (entry.kind, entry.sequence) == ("registration", 0)
    payload = entry.payload
    assert payload["record_type"] == "prospective_registration"
    assert payload["registered_at"] == REGISTRATION_TIME.isoformat()
    assert payload["registration_commit"] == REGISTRATION_COMMIT
    assert payload["record_path"] == RECORD_PATH
    assert payload["record"] == registration(models)
    assert payload["record_sha256"] == registration_record.record_sha256(registration(models))
    assert entry.recorded_at == REGISTERED_RECORDED.isoformat()
    assert "head" not in canonical_json(payload)
    assert ledger.read_ledger(ledger_path) == (entry,)


def test_a_registration_is_only_the_first_entry_of_its_ledger(models, ledger_path):
    register(models, ledger_path)
    assert _refused(register, models, ledger_path) == "ledger_not_empty"
    assert _kinds(ledger_path) == ["registration"]


@pytest.mark.parametrize(
    ("registered_at", "recorded_at", "reason"),
    [
        (pd.Timestamp("2026-10-14T14:00:00"), REGISTERED_RECORDED, "registration_invalid"),
        (pd.Timestamp("2026-10-14T12:00:00Z"), REGISTERED_RECORDED, "registered_before_freeze"),
        (REGISTRATION_TIME, pd.Timestamp("2026-10-14T13:59:59Z"), "registration_in_future"),
    ],
)
def test_the_registration_time_is_explicit_aware_and_after_the_freeze(
    models, ledger_path, registered_at, recorded_at, reason
):
    # The fixture's models were registered (frozen) at 13:00 UTC.
    got = _refused(
        register, models, ledger_path, registered_at=registered_at, recorded_at=recorded_at
    )
    assert got == reason
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r.update(config_sha256="f" * 64),
        lambda r: r["bundle"].update(manifest_sha256="f" * 64),
        lambda r: next(iter(r["registry"].values())).update(entry_sha256="f" * 64),
        lambda r: r.update(training_data_sha256="f" * 64),
        lambda r: r["code"].update(git_sha="f" * 40),
        lambda r: r.update(band_method="another_band_v2"),
        lambda r: r["bundle"].update(bundle_id="etf_prospective_test-another"),
        lambda r: r["bundle"].update(run_id="another-run"),
        lambda r: r["bundle"].update(data_class="engineering_sandbox"),
        lambda r: r.update(assets=["VTI", "AGG"]),
        lambda r: r.update(experiment="another_experiment"),
        lambda r: next(iter(r["registry"].values())).update(model_version=99),
        lambda r: next(iter(r["registry"].values())).update(model_id="another.model"),
    ],
)
def test_a_tampered_record_file_is_never_registered(models, ledger_path, edit):
    record_ = registration(models)
    edit(record_)
    reason = _refused(
        registration_record.register,
        ledger_path,
        record_,
        models,
        record_path=RECORD_PATH,
        registration_commit=REGISTRATION_COMMIT,
        registered_at=REGISTRATION_TIME,
        recorded_at=REGISTERED_RECORDED,
        code=RUN_CODE,
        evidence=evidence(),
    )
    assert reason == "registration_mismatch"
    assert not ledger_path.exists()


def test_a_merge_at_the_freeze_time_itself_is_accepted(models, ledger_path):
    frozen_at = max(pd.Timestamp(i["registered_at"]) for i in models.entries.values())
    entry = register(models, ledger_path, registered_at=frozen_at, recorded_at=frozen_at)
    assert entry.payload["registered_at"] == frozen_at.tz_convert("UTC").isoformat()


def test_registering_needs_the_registered_code_and_a_commit_id(models, ledger_path):
    other_code = {**RUN_CODE, "trees": {**CODE_TREES, "src": "f" * 40}}
    common = {
        "record_path": RECORD_PATH,
        "registered_at": REGISTRATION_TIME,
        "recorded_at": REGISTERED_RECORDED,
        "evidence": evidence(),
    }
    record_ = registration(models)
    reason = _refused(
        registration_record.register,
        ledger_path,
        record_,
        models,
        registration_commit=REGISTRATION_COMMIT,
        code=other_code,
        **common,
    )
    assert reason == "registration_mismatch"
    reason = _refused(
        registration_record.register,
        ledger_path,
        record_,
        models,
        registration_commit="main",
        code=RUN_CODE,
        **common,
    )
    assert reason == "registration_invalid"
    assert not ledger_path.exists()


# --- the registration time, verified against GitHub ---------------------------------------


def _register_with(models, ledger_path, proof, *, registered_at=REGISTRATION_TIME):
    return registration_record.register(
        ledger_path,
        registration(models),
        models,
        record_path=RECORD_PATH,
        registration_commit=REGISTRATION_COMMIT,
        registered_at=registered_at,
        recorded_at=REGISTERED_RECORDED,
        code=RUN_CODE,
        evidence=proof,
    )


@pytest.mark.parametrize("source", ["github_api", "owner_evidence_file"])
def test_the_entry_keeps_the_pull_requests_evidence(models, ledger_path, source):
    entry = _register_with(models, ledger_path, evidence(source=source))
    kept = entry.payload["registration_evidence"]
    assert kept["source"] == source
    assert kept["fetched_at"] == "2026-10-14T14:30:00+00:00"
    assert kept["pull_request"] == {
        "number": 120,
        "merged": True,
        "merged_at": "2026-10-14T14:00:00Z",
        "merge_commit_sha": REGISTRATION_COMMIT,
        "base": {"ref": "main", "repo": {"full_name": "billvo9/stock-allocation-agent"}},
    }  # the fields checked, and nothing else the API returned
    assert kept["sha256"] == sha256_json(kept["pull_request"])
    assert "independently attested" not in canonical_json(entry.payload)
    found = registration_record.current(ledger.read_ledger(ledger_path))
    assert found.evidence_source == source


@pytest.mark.parametrize(
    "pull_request",
    [
        github_pull_request(merged_at=REGISTRATION_TIME + pd.Timedelta(seconds=1)),
        github_pull_request(merged_at=REGISTRATION_TIME - pd.Timedelta(hours=1)),
        github_pull_request(merge_commit_sha="f" * 40),
        github_pull_request(merged=False, merged_at=None),
        github_pull_request(merged=False),  # closed unmerged, with a plausible time anyway
        github_pull_request(merged="true"),  # not the boolean true
        github_pull_request(number=True),
        github_pull_request(number="120"),
        {**github_pull_request(), "merged_at": "2026-10-14T14:00:00"},  # no offset
        github_pull_request(
            base={"ref": "develop", "repo": {"full_name": "billvo9/stock-allocation-agent"}}
        ),
        github_pull_request(
            base={"ref": "main", "repo": {"full_name": "someone/stock-allocation-agent"}}
        ),
        {"number": 120, "merged": True},  # fields missing
    ],
    ids=[
        "merged_1s_later",
        "merged_1h_earlier",
        "other_merge_commit",
        "not_merged",
        "not_merged_with_a_time",
        "merged_as_text",
        "boolean_number",
        "text_number",
        "naive_merge_time",
        "base_not_main",
        "another_repository",
        "incomplete",
    ],
)
def test_a_registration_time_the_evidence_does_not_support_is_refused(
    models, ledger_path, pull_request
):
    reason = _refused(_register_with, models, ledger_path, evidence(pull_request))
    assert reason == "registration_unverified"
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    "proof",
    [
        None,  # a typed time alone
        {**evidence(), "source": "typed_by_hand"},
        {**evidence(), "fetched_at": pd.Timestamp("2026-10-14T14:30:00")},  # naive
        {"source": "github_api", "fetched_at": pd.Timestamp("2026-10-14T14:30:00Z")},
        {**evidence(), "fetched_at": "yesterday"},
        evidence(fetched_at=pd.Timestamp("2026-10-14T13:59:59Z")),  # before the merge
        evidence(fetched_at=pd.Timestamp("2026-10-14T15:00:01Z")),  # after the entry
    ],
    ids=[
        "missing",
        "unknown_source",
        "naive_fetch_time",
        "no_pull_request",
        "garbage_fetch_time",
        "fetched_before_the_merge",
        "fetched_after_the_entry",
    ],
)
def test_a_registration_without_usable_evidence_is_refused(models, ledger_path, proof):
    assert _refused(_register_with, models, ledger_path, proof) == "registration_unverified"
    assert not ledger_path.exists()


@pytest.mark.parametrize(
    "change",
    [
        lambda e: e["pull_request"].update(merged_at="2026-10-14T13:00:00Z"),
        lambda e: e.update(sha256="0" * 64),
        lambda e: e["pull_request"]["base"].update(ref="develop"),
        lambda e: e.update(source="typed_by_hand"),
        lambda e: e.update(fetched_at="2026-10-14T15:00:01+00:00"),  # after the entry
        lambda e: e.update(fetched_at="2026-10-14T13:59:59+00:00"),  # before the merge
    ],
    ids=["earlier_merge_time", "hash", "base", "source", "fetched_after", "fetched_before"],
)
def test_evidence_edited_after_the_entry_was_written_is_refused(models, ledger_path, store, change):
    register(models, ledger_path)

    def edit(payload):
        change(payload["registration_evidence"])
        kept = payload["registration_evidence"]
        if "sha256" in kept and kept["sha256"] != "0" * 64:  # a consistent rewrite of the hash
            kept["sha256"] = sha256_json(kept["pull_request"])

    _rewrite_first(ledger_path, edit)
    assert _refused(_forecast, models, ledger_path, store) == "registration_tampered"


def test_a_rewritten_registration_time_is_refused(models, ledger_path, store):
    # Backdating: only the entry's time is rewritten (the chain re-made); the evidence's
    # merged_at no longer equals it.
    register(models, ledger_path)
    _rewrite_first(ledger_path, lambda p: p.update(registered_at="2026-10-14T13:30:00+00:00"))
    assert _refused(_forecast, models, ledger_path, store) == "registration_tampered"


def test_an_entry_from_before_evidence_was_required_is_refused(models, ledger_path, store):
    register(models, ledger_path)

    def legacy(payload):  # the schema 1.0 entry of #97: a typed time, no evidence
        payload.pop("registration_evidence")
        payload["registration_schema_version"] = "1.0"

    _rewrite_first(ledger_path, legacy)
    assert _refused(_forecast, models, ledger_path, store) == "registration_unverified"


# --- the forecast job and the registration ------------------------------------------------


def test_a_forecast_before_any_registration_is_refused(models, ledger_path, store):
    assert _refused(_forecast, models, ledger_path, store) == "not_registered"
    assert not ledger_path.exists()


def test_a_forecast_whose_close_is_not_after_the_registration_is_refused(
    models, ledger_path, store
):
    # Merged at 18:30 New York on AS_OF: the AS_OF close (18:00) is not after it.
    register(
        models,
        ledger_path,
        registered_at=pd.Timestamp("2026-10-14T22:30:00Z"),
        recorded_at=pd.Timestamp("2026-10-14T22:45:00Z"),
    )
    assert _refused(_forecast, models, ledger_path, store) == "before_registration"
    assert _kinds(ledger_path) == ["registration"]
    first = _forecast(
        models, ledger_path, store, as_of="2026-10-15", created="2026-10-15T23:00:00Z"
    )
    assert first.payload["session"] == {
        "number": 1,
        "phase": "trial",
        "first_session": "2026-10-15",
    }


@pytest.mark.parametrize(
    ("registered_at", "refused"),
    [("2026-10-14T22:00:00Z", True), ("2026-10-14T21:59:00Z", False)],  # 18:00 New York
)
def test_the_close_must_be_strictly_after_the_registration(
    models, ledger_path, store, registered_at, refused
):
    when = pd.Timestamp(registered_at)
    register(models, ledger_path, registered_at=when, recorded_at=when + pd.Timedelta(minutes=5))
    if refused:
        assert _refused(_forecast, models, ledger_path, store) == "before_registration"
    else:
        assert _forecast(models, ledger_path, store).payload["session"]["number"] == 1


def test_a_forecast_created_before_the_registration_entry_is_refused(models, ledger_path, store):
    # The entry was appended at 23:30 UTC, after a forecast "created" at 23:00 would be.
    register(models, ledger_path, recorded_at=pd.Timestamp("2026-10-14T23:30:00Z"))
    assert _refused(_forecast, models, ledger_path, store) == "before_registration"
    assert _kinds(ledger_path) == ["registration"]


def test_the_first_eligible_close_needs_every_fund_complete(models, ledger_path, tmp_path, store):
    gap = prospective_store(tmp_path / "gap", edit=without_close("VTI", ["2026-09-15"]))
    register(models, ledger_path)
    assert _refused(_forecast, models, ledger_path, gap) == "not_first_eligible_close"
    assert _kinds(ledger_path) == ["registration"]  # nothing recorded for that session
    # After session 1, a fund with an incomplete window is refused alone and recorded.
    _forecast(models, ledger_path, store)
    later = _forecast(models, ledger_path, gap, as_of="2026-10-15", created="2026-10-15T23:00:00Z")
    assert later.kind == "forecast"
    assert [r["asset"] for r in later.payload["refusals"]] == ["VTI"]


def test_the_first_eligible_close_needs_spot_checked_batches(models, ledger_path, store):
    def unchecked(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        win = load_inference_window(
            store_root, assets, forecast_close, lookback, retrieved_by=retrieved_by
        )
        batches = win.batches.assign(anchors_sha256=None)
        return dataclasses.replace(win, batches=batches)

    register(models, ledger_path)
    reason = _refused(_forecast, models, ledger_path, store, loader=unchecked)
    assert reason == "not_first_eligible_close"
    assert _kinds(ledger_path) == ["registration"]


@pytest.mark.parametrize(
    "code",
    [
        {**RUN_CODE, "trees": {**CODE_TREES, "src": "f" * 40}},  # a source change
        {**RUN_CODE, "trees": {**CODE_TREES, "configs/prospective": "f" * 40}},  # a config file
        {**RUN_CODE, "trees_dirty": True},  # uncommitted changes under a registered path
        {"git_sha": "0" * 40, "git_dirty": False},  # no tree ids given
    ],
)
def test_a_forecast_from_other_code_is_refused(models, ledger_path, store, code):
    register(models, ledger_path)
    assert _refused(_forecast, models, ledger_path, store, code=code) == "registration_mismatch"
    assert _kinds(ledger_path) == ["registration"]


def _with_document(models, edit):
    document = copy.deepcopy(dict(models.bundle.document))
    edit(document)
    return forecast.Models(dataclasses.replace(models.bundle, document=document), models.entries)


def _other_config(document):
    document["config"]["labels"] = [*document["config"]["labels"], "another variant"]
    document["config_sha256"] = parse_config(document["config"]).sha256


def _other_data(document):
    document["training_data"] = {**document["training_data"], "prices_sha256": "f" * 64}


@pytest.mark.parametrize("edit", [_other_config, _other_data])
def test_a_forecast_from_another_config_or_data_is_refused(models, ledger_path, store, edit):
    register(models, ledger_path)
    changed = _with_document(models, edit)
    assert _refused(_forecast, changed, ledger_path, store) == "registration_mismatch"


def test_a_forecast_from_another_frozen_bundle_is_refused(models, ledger_path, store):
    register(models, ledger_path)
    refrozen = forecast.Models(
        dataclasses.replace(models.bundle, manifest_sha256="f" * 64), models.entries
    )
    assert _refused(_forecast, refrozen, ledger_path, store) == "registration_mismatch"


def test_a_forecast_from_other_registry_entries_is_refused(models, ledger_path, store):
    register(models, ledger_path)
    key = next(iter(models.entries))
    entries = {**models.entries, key: {**models.entries[key], "entry_sha256": "f" * 64}}
    other = forecast.Models(models.bundle, entries)
    assert _refused(_forecast, other, ledger_path, store) == "registration_mismatch"


def _rewrite_first(path, change):
    """Edit the first line and re-chain every line, as a consistent rewrite would."""

    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines]
    change(rows[0]["payload"])
    previous = ledger.GENESIS
    out = []
    for row in rows:
        row.pop("entry_sha256")
        row["prev_entry_sha256"] = previous
        row["entry_sha256"] = previous = sha256_json(row)
        out.append(canonical_json(row) + "\n")
    path.write_text("".join(out), encoding="utf-8")


def test_an_edited_registration_line_breaks_the_chain(models, ledger_path, store):
    register(models, ledger_path)
    text = ledger_path.read_text(encoding="utf-8")
    ledger_path.write_text(text.replace(REGISTRATION_COMMIT, "a" * 40), encoding="utf-8")
    with pytest.raises(ledger.LedgerError):
        _forecast(models, ledger_path, store)


def test_a_rewritten_registration_record_is_refused(models, ledger_path, store):
    register(models, ledger_path)
    _rewrite_first(ledger_path, lambda p: p["record"].update(zero_volatility_rule="x"))
    assert _refused(_forecast, models, ledger_path, store) == "registration_tampered"


def test_a_record_that_no_longer_matches_its_recorded_hash_is_refused(models, ledger_path, store):
    register(models, ledger_path)  # a valid record; only its recorded hash is rewritten
    _rewrite_first(ledger_path, lambda p: p.update(record_sha256="0" * 64))
    assert _refused(_forecast, models, ledger_path, store) == "registration_tampered"


def test_a_consistently_rewritten_record_is_still_refused_against_the_models(
    models, ledger_path, store
):
    def rewrite(payload):
        payload["record"]["config_sha256"] = "f" * 64
        payload["record_sha256"] = registration_record.record_sha256(payload["record"])

    register(models, ledger_path)
    _rewrite_first(ledger_path, rewrite)  # chain, record and hash all consistent
    assert registration_record.current(ledger.read_ledger(ledger_path)) is not None
    assert _refused(_forecast, models, ledger_path, store) == "registration_mismatch"


def test_a_second_or_late_registration_entry_is_refused(models, ledger_path, store):
    first = register(models, ledger_path)
    ledger.append_entry(
        ledger_path, "registration", dict(first.payload), recorded_at=REGISTERED_RECORDED
    )
    assert _refused(_forecast, models, ledger_path, store) == "registration_not_unique"
    other = ledger_path.parent / "late.jsonl"
    ledger.append_entry(other, "refusal", {"as_of_session": AS_OF}, recorded_at=CREATED)
    ledger.append_entry(other, "registration", dict(first.payload), recorded_at=CREATED)
    assert _refused(_forecast, models, other, store) == "registration_not_first"


def _appending(ledger_path, payload):
    """The real loader, after another writer appended `payload` to the ledger."""

    def load(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        ledger.append_entry(ledger_path, "forecast", payload, recorded_at=CREATED)
        return load_inference_window(
            store_root, assets, forecast_close, lookback, retrieved_by=retrieved_by
        )

    return load


def test_a_record_appended_while_a_forecast_runs_is_refused(models, ledger_path, store):
    register(models, ledger_path)
    other = {"as_of_session": "2026-10-13", "rows": [], "refusals": []}
    loader = _appending(ledger_path, other)
    assert _refused(_forecast, models, ledger_path, store, loader=loader) == "ledger_changed"
    assert _kinds(ledger_path) == ["registration", "forecast"]  # only the other writer's line


def test_a_refusal_is_also_refused_when_the_ledger_changed(models, ledger_path, tmp_path, store):
    def edit(retrieval, tables):
        for fund in ("VTI", "AGG", "SHV"):
            tables = without_close(fund, ["2026-09-15"])(retrieval, tables)
        return tables

    gaps = prospective_store(tmp_path / "gaps", edit=edit)
    register(models, ledger_path)
    _forecast(models, ledger_path, store)  # session 1
    loader = _appending(ledger_path, {"as_of_session": "2026-10-13", "rows": []})
    reason = _refused(
        _forecast,
        models,
        ledger_path,
        gaps,
        as_of="2026-10-15",
        created="2026-10-15T23:00:00Z",
        loader=loader,
    )
    assert reason == "ledger_changed"
    assert _kinds(ledger_path) == ["registration", "forecast", "forecast"]


def test_a_session_already_refused_is_never_recorded_again(models, ledger_path, tmp_path, store):
    def edit(retrieval, tables):
        for fund in ("VTI", "AGG", "SHV"):
            tables = without_close(fund, ["2026-09-15"])(retrieval, tables)
        return tables

    gaps = prospective_store(tmp_path / "gaps", edit=edit)
    register(models, ledger_path)
    _forecast(models, ledger_path, store)  # session 1
    later = {"as_of": "2026-10-15", "created": "2026-10-15T23:00:00Z"}
    assert _forecast(models, ledger_path, gaps, **later).kind == "refusal"
    retry = {"as_of": "2026-10-15", "created": "2026-10-15T23:30:00Z"}
    assert _refused(_forecast, models, ledger_path, store, **retry) == "duplicate_forecast"
    assert _refused(_forecast, models, ledger_path, gaps, **retry) == "duplicate_forecast"
    assert _kinds(ledger_path) == ["registration", "forecast", "refusal"]


def test_each_record_carries_its_registration_and_session(models, ledger_path, tmp_path):
    days = [f"{d:%Y-%m-%d}" for d in exchange_calendar(AS_OF, "2026-11-30")]
    session_20, session_21 = days[19], days[20]
    assert (session_20, session_21) == ("2026-11-10", "2026-11-11")
    retrievals = (AS_OF, "2026-10-16", session_20, session_21)
    store = prospective_store(tmp_path / "long", retrievals=retrievals)
    registration_entry = register(models, ledger_path)
    first = _forecast(models, ledger_path, store)
    assert first.payload["forecast_schema_version"] == "1.3"
    assert first.payload["registration"] == {
        "entry_sha256": registration_entry.entry_sha256,
        "record_sha256": registration_record.record_sha256(registration(models)),
        "registration_commit": REGISTRATION_COMMIT,
        "registered_at": REGISTRATION_TIME.isoformat(),
    }
    assert first.payload["session"] == {"number": 1, "phase": "trial", "first_session": AS_OF}
    # 2026-10-15 is missed; numbers count calendar sessions from session 1, not records.
    third = _forecast(
        models, ledger_path, store, as_of="2026-10-16", created="2026-10-16T23:00:00Z"
    )
    assert third.payload["session"]["number"] == 3
    last_trial = _forecast(
        models, ledger_path, store, as_of=session_20, created=f"{session_20}T23:30:00Z"
    )
    assert last_trial.payload["session"] == {
        "number": 20,
        "phase": "trial",
        "first_session": AS_OF,
    }
    confirmatory = _forecast(
        models, ledger_path, store, as_of=session_21, created=f"{session_21}T23:30:00Z"
    )
    assert confirmatory.payload["session"]["number"] == 21
    assert confirmatory.payload["session"]["phase"] == "confirmatory"


# --- scoring and the registration ---------------------------------------------------------


def test_scoring_records_the_registration_and_refuses_other_code(
    prospective_frozen, ledger_path, store
):
    record(prospective_frozen, ledger_path, store)
    with pytest.raises(forecast.ForecastRefused) as refused:
        forecast.score_matured(
            ledger_path=ledger_path,
            loader=load_inference_window,
            store_root=store,
            score_close="2026-10-23",
            scored_at=pd.Timestamp("2026-10-23T23:00:00Z"),
            code={**RUN_CODE, "trees": {**CODE_TREES, "src": "f" * 40}},
            calendar=exchange_calendar,
            environment=RUN_ENVIRONMENT,
        )
    assert refused.value.reason == "registration_mismatch"
    written, _ = score(ledger_path, "2026-10-23", "2026-10-23T23:00:00Z", store)
    first = ledger.read_ledger(ledger_path)[0]
    assert written.payload["registration"]["entry_sha256"] == first.entry_sha256
    assert {
        s["zero_vol_rule"] for s in written.payload["scores"] if s["target"] != "total_log_return"
    } == {zero_vol.CURRENT}


def test_scoring_refuses_a_ledger_without_a_registration(prospective_frozen, ledger_path, store):
    written = record(prospective_frozen, ledger_path.parent / "registered.jsonl", store)
    ledger.append_entry(ledger_path, "forecast", dict(written.payload), recorded_at=CREATED)
    reason = _refused(score, ledger_path, "2026-10-23", "2026-10-23T23:00:00Z", store)
    assert reason == "not_registered"


def test_scoring_refuses_a_rule_other_than_the_registered_one(
    prospective_frozen, models, ledger_path, store
):
    genuine = record(prospective_frozen, ledger_path.parent / "genuine.jsonl", store)
    register(models, ledger_path)  # registered under the rule in force
    planted = copy.deepcopy(dict(genuine.payload))
    planted["bundle"]["zero_volatility_rule"] = zero_vol.B2  # written outside the job
    ledger.append_entry(ledger_path, "forecast", planted, recorded_at=CREATED)
    common = {
        "ledger_path": ledger_path,
        "loader": load_inference_window,
        "store_root": store,
        "score_close": "2026-10-23",
        "scored_at": pd.Timestamp("2026-10-23T23:00:00Z"),
        "code": RUN_CODE,
        "calendar": exchange_calendar,
        "environment": RUN_ENVIRONMENT,
    }
    reason = _refused(forecast.score_matured, **common, zero_vol_rule=zero_vol.B2)
    assert reason == "zero_vol_rule_mismatch"  # B2 is not the registered rule
    assert _refused(forecast.score_matured, **common) == "zero_vol_rule_not_trained"
    assert _kinds(ledger_path) == ["registration", "forecast"]


@pytest.mark.parametrize("field", ["manifest_sha256", "config_sha256"])
def test_scoring_refuses_a_forecast_from_another_bundle_or_config(
    prospective_frozen, models, ledger_path, store, field
):
    genuine = record(prospective_frozen, ledger_path.parent / "genuine.jsonl", store)
    register(models, ledger_path)
    planted = copy.deepcopy(dict(genuine.payload))
    planted["bundle"][field] = "f" * 64  # written outside the job
    ledger.append_entry(ledger_path, "forecast", planted, recorded_at=CREATED)
    reason = _refused(score, ledger_path, "2026-10-23", "2026-10-23T23:00:00Z", store)
    assert reason == "registration_mismatch"
    assert _kinds(ledger_path) == ["registration", "forecast"]  # nothing scored


def test_a_first_window_naming_no_batch_is_not_eligible(models, ledger_path, store):
    def unattributed(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        win = load_inference_window(
            store_root, assets, forecast_close, lookback, retrieved_by=retrieved_by
        )
        return dataclasses.replace(win, batches=win.batches.iloc[0:0])

    register(models, ledger_path)
    reason = _refused(_forecast, models, ledger_path, store, loader=unattributed)
    assert reason == "not_first_eligible_close"
    assert _kinds(ledger_path) == ["registration"]
