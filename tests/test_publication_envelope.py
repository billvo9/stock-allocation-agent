from datetime import timedelta

import pandas as pd
import pytest

from stock_agent.model_validation.folds import MODEL_HOLDOUT_START
from stock_agent.publication import envelope as env
from stock_agent.publication.envelope import PublicationRefused, RefusalReason


def _refused(reason, call, *args, **kwargs):
    with pytest.raises(PublicationRefused) as caught:
        call(*args, **kwargs)
    assert caught.value.reason is reason, caught.value


# --- timestamps ------------------------------------------------------------


def test_timestamp_round_trips_in_the_one_format():
    value = pd.Timestamp("2024-12-31T23:59:59.999999Z")
    text = env.format_timestamp("t", value)
    assert text == "2024-12-31T23:59:59.999999Z"
    assert env.parse_timestamp("t", text) == value
    assert env.format_timestamp("t", pd.Timestamp("2026-10-13T12:00:00Z")).endswith(".000000Z")


@pytest.mark.parametrize(
    "value",
    [
        pd.Timestamp("2024-12-31T23:59:59.999999"),  # naive
        pd.Timestamp("2024-12-31T18:59:59", tz="America/New_York"),  # aware, not UTC
        pd.Timestamp("2024-12-31T23:59:59.999999001Z"),  # sub-microsecond
        "2024-12-31T23:59:59.999999Z",  # a string is not a Timestamp
    ],
)
def test_format_refuses_naive_non_utc_and_nanosecond_values(value):
    _refused(RefusalReason.INVALID_TIMESTAMP, env.format_timestamp, "t", value)


@pytest.mark.parametrize(
    "text",
    [
        "2024-12-31T23:59:59Z",  # no microseconds
        "2024-12-31T20:00:00.000000-05:00",  # an offset: 2025-01-01T01:00Z
        "2024-12-31 23:59:59.999999Z",
        "2024-12-31T23:59:59.999999",
        "2024-02-30T00:00:00.000000Z",  # not a date
        20241231,
    ],
)
def test_parse_refuses_anything_but_the_exact_utc_format(text):
    _refused(RefusalReason.INVALID_TIMESTAMP, env.parse_timestamp, "t", text)


# --- versions and strict JSON ---------------------------------------------


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        ("1.0", None),
        ("1.1", RefusalReason.UNSUPPORTED_VERSION),
        ("2.0", RefusalReason.UNSUPPORTED_VERSION),
        ("0.9", RefusalReason.UNSUPPORTED_VERSION),
        ("1", RefusalReason.MALFORMED_VERSION),
        ("1.0.0", RefusalReason.MALFORMED_VERSION),
        ("1.-1", RefusalReason.MALFORMED_VERSION),
        ("01.0", RefusalReason.MALFORMED_VERSION),
        (1.0, RefusalReason.MALFORMED_VERSION),
        (None, RefusalReason.MALFORMED_VERSION),
    ],
)
def test_only_version_1_0_is_supported(value, reason):
    if reason is None:
        assert env.check_version(value) == "1.0"
    else:
        _refused(reason, env.check_version, value)


@pytest.mark.parametrize(
    "text",
    ['{"a": NaN}', '{"a": Infinity}', '{"a": -Infinity}', '{"a": 1, "a": 2}', "{", "[1,]"],
)
def test_strict_json_refuses_non_finite_tokens_duplicate_keys_and_garbage(text):
    _refused(RefusalReason.INVALID_JSON, env.loads_strict, text)


def test_serialization_refuses_non_finite_numbers():
    _refused(RefusalReason.INVALID_JSON, env.dumps_canonical, {"a": float("nan")})
    _refused(RefusalReason.INVALID_JSON, env.dumps_file, {"a": float("inf")})


# --- lockbox and the synthetic knowledge-time rule ------------------------


def test_lockbox_constant_is_the_validation_one():
    assert env.MODEL_HOLDOUT_START is MODEL_HOLDOUT_START


def test_lockbox_boundary_is_on_or_after_the_holdout_start():
    env.check_lockbox("t", MODEL_HOLDOUT_START - timedelta(microseconds=1))
    _refused(RefusalReason.LOCKBOX_VIOLATION, env.check_lockbox, "t", MODEL_HOLDOUT_START)


def test_synthetic_session_end_is_the_last_microsecond_of_the_utc_day():
    last_session = pd.Timestamp("2024-12-31", tz="UTC").as_unit("us")
    known = env.synthetic_session_end(last_session)
    assert env.format_timestamp("k", known) == "2024-12-31T23:59:59.999999Z"
    assert known < MODEL_HOLDOUT_START
    assert env.synthetic_session_end(pd.Timestamp("2024-06-03", tz="UTC")) == pd.Timestamp(
        "2024-06-03T23:59:59.999999Z"
    )


# --- envelope parsing and ids ----------------------------------------------


def _fields():
    return {
        "kind": "synthetic_rule_output",
        "schema_version": "1.0",
        "as_of": "2024-12-31T23:59:59.999999Z",
        "knowledge_time": "2024-12-31T23:59:59.999999Z",
        "knowledge_time_rule": "interim_synthetic_session_end",
        "created_at": "2026-10-13T12:00:00.000000Z",
        "code_sha": "a" * 40,
        "code_dirty": False,
        "input_hashes": [],
        "universe_id": "synthetic",
        "method_id": "syn_equal_weight",
        "method_version": "1",
        "parameters": {"n_assets": 2, "decision_date": "2024-12-31"},
        "evidence_status": "rules_illustration",
        "limitations": ["synthetic_calendar", "synthetic_data"],
    }


def test_publication_id_binds_content_and_ignores_key_order():
    fields = _fields()
    first = env.derive_publication_id(fields, "0" * 64)
    assert env.PUBLICATION_ID_PATTERN.match(first)
    assert first.startswith("20261013T120000000000Z-")
    assert env.derive_publication_id(dict(reversed(list(fields.items()))), "0" * 64) == first
    assert env.derive_publication_id(fields, "1" * 64) != first
    changed = {**fields, "parameters": {"n_assets": 3, "decision_date": "2024-12-31"}}
    assert env.derive_publication_id(changed, "0" * 64) != first


def test_envelope_round_trips_through_json():
    data = {"publication_id": env.derive_publication_id(_fields(), "0" * 64), **_fields()}
    parsed = env.Envelope.from_json_dict(env.loads_strict(env.dumps_file(data)))
    assert parsed.to_json_dict() == data
    assert "claim_type" not in parsed.to_json_dict()


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda d: d.pop("as_of"), RefusalReason.FIELD_SET),
        (lambda d: d.update(claim_type="rule_illustration"), RefusalReason.FIELD_SET),
        (lambda d: d.update(kind="performance_comparison"), RefusalReason.UNSUPPORTED_KIND),
        (
            lambda d: d.update(evidence_status="prospectively_confirmed"),
            RefusalReason.UNKNOWN_STATUS,
        ),
        (
            lambda d: d.update(knowledge_time_rule="production_close_1800_et"),
            RefusalReason.KNOWLEDGE_RULE_NOT_PERMITTED,
        ),
        (lambda d: d.update(code_dirty="false"), RefusalReason.INVALID_FIELD),
        (lambda d: d.update(code_dirty=0), RefusalReason.INVALID_FIELD),
        (lambda d: d.update(code_sha="abc"), RefusalReason.INVALID_FIELD),
        (
            lambda d: d.update(limitations=["synthetic_data", "synthetic_calendar"]),
            RefusalReason.INVALID_FIELD,
        ),
        (lambda d: d.update(limitations=["survivorship_biased"]), RefusalReason.INVALID_FIELD),
        (lambda d: d.update(parameters=[1, 2]), RefusalReason.INVALID_FIELD),
        (lambda d: d.update(input_hashes={}), RefusalReason.INVALID_FIELD),
        (lambda d: d.update(universe_id=""), RefusalReason.INVALID_FIELD),
        (
            lambda d: d.update(as_of="2024-12-31T18:59:59.999999-05:00"),
            RefusalReason.INVALID_TIMESTAMP,
        ),
        (lambda d: d.update(schema_version="1.1"), RefusalReason.UNSUPPORTED_VERSION),
    ],
)
def test_envelope_parser_refuses_each_malformed_field(edit, reason):
    data = {"publication_id": "x", **_fields()}
    edit(data)
    _refused(reason, env.Envelope.from_json_dict, data)


def test_envelope_parser_refuses_a_non_object():
    _refused(RefusalReason.INVALID_FIELD, env.Envelope.from_json_dict, [1, 2])


@pytest.mark.parametrize(
    ("call", "value", "reason"),
    [
        (env.check_version, "1.0\n", RefusalReason.MALFORMED_VERSION),
        (env.check_code_sha, "a" * 40 + "\n", RefusalReason.INVALID_FIELD),
    ],
)
def test_a_trailing_newline_never_matches(call, value, reason):
    _refused(reason, call, value)
    _refused(
        RefusalReason.INVALID_TIMESTAMP, env.parse_timestamp, "t", "2024-12-31T23:59:59.999999Z\n"
    )
    assert env.PUBLICATION_ID_PATTERN.fullmatch("20261013T120000000000Z-0123456789ab\n") is None


def test_overflowing_numbers_and_non_json_values_are_refused():
    _refused(RefusalReason.INVALID_JSON, env.loads_strict, '{"a": 1e999}')
    _refused(RefusalReason.INVALID_JSON, env.dumps_canonical, {"a": object()})
    _refused(RefusalReason.INVALID_JSON, env.dumps_canonical, {1: "a", "b": 2})
    fields = {**_fields(), "universe_id": "\ud800"}
    _refused(RefusalReason.INVALID_JSON, env.derive_publication_id, fields, "0" * 64)
