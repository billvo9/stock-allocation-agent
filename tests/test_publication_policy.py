from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from stock_agent.publication import synthetic
from stock_agent.publication.envelope import (
    ClaimType,
    EvidenceStatus,
    Kind,
    KnowledgeTimeRule,
    PublicationRefused,
    RefusalReason,
)
from stock_agent.publication.policy import POLICY, check_rows, check_schema, claim_type, validate

DESCRIPTOR = POLICY[Kind.SYNTHETIC_DESCRIPTOR]
RULE_OUTPUT = POLICY[Kind.SYNTHETIC_RULE_OUTPUT]


def _descriptor_rows():
    spec = synthetic.SyntheticSpec(
        "syn_price_path",
        "1",
        {"n_assets": 2, "start": "2024-12-23", "n_sessions": 4, "seed": 1, "n_missing": 1},
    )
    return synthetic.draft(spec).rows


def _rule_rows():
    spec = synthetic.SyntheticSpec(
        "syn_equal_weight", "1", {"n_assets": 3, "decision_date": "2024-12-31"}
    )
    return synthetic.draft(spec).rows


def _set(column, index, value):
    def edit(rows):
        rows = rows.copy()
        if column == "date":
            rows[column] = rows[column].astype(object)
        rows.loc[index, column] = value
        if column == "date":
            rows[column] = pd.to_datetime(rows[column], utc=True).dt.as_unit("us")
        return rows

    return edit


def test_policy_table_is_closed_read_only_and_synthetic_only():
    assert isinstance(POLICY, MappingProxyType)
    assert set(POLICY) == {Kind.SYNTHETIC_DESCRIPTOR, Kind.SYNTHETIC_RULE_OUTPUT}
    with pytest.raises(TypeError):
        POLICY["performance_comparison"] = DESCRIPTOR  # type: ignore[index]
    assert DESCRIPTOR.allowed_status is EvidenceStatus.DESCRIPTIVE
    assert RULE_OUTPUT.allowed_status is EvidenceStatus.RULES_ILLUSTRATION
    assert claim_type(Kind.SYNTHETIC_DESCRIPTOR) is ClaimType.DESCRIPTIVE_FACT
    assert claim_type(Kind.SYNTHETIC_RULE_OUTPUT) is ClaimType.RULE_ILLUSTRATION


def test_interim_rule_appears_only_on_synthetic_kinds_with_regeneration():
    # OD9 guard: the synthetic-only knowledge rule must never reach a kind
    # whose provenance is not the synthetic regeneration check.
    assert list(KnowledgeTimeRule) == [KnowledgeTimeRule.INTERIM_SYNTHETIC_SESSION_END]
    for kind, policy in POLICY.items():
        assert kind.value.startswith("synthetic_")
        if policy.knowledge_time_rule is KnowledgeTimeRule.INTERIM_SYNTHETIC_SESSION_END:
            assert policy.verify_provenance is synthetic.verify_provenance


def test_generated_rows_pass_the_validator():
    check_rows(DESCRIPTOR, check_schema(DESCRIPTOR, _descriptor_rows()))
    check_rows(RULE_OUTPUT, check_schema(RULE_OUTPUT, _rule_rows()))


@pytest.mark.parametrize(
    ("policy", "rows", "edit", "reason"),
    [
        # status, reason and value consistency
        (DESCRIPTOR, _descriptor_rows, _set("value", 1, np.nan), RefusalReason.INVALID_ROW),
        (DESCRIPTOR, _descriptor_rows, _set("reason", 0, None), RefusalReason.INVALID_ROW),
        (DESCRIPTOR, _descriptor_rows, _set("value", 0, 1.0), RefusalReason.INVALID_ROW),
        (
            DESCRIPTOR,
            _descriptor_rows,
            _set("reason", 1, "missing_input"),
            RefusalReason.INVALID_ROW,
        ),
        (DESCRIPTOR, _descriptor_rows, _set("status", 1, "stale"), RefusalReason.INVALID_ROW),
        (DESCRIPTOR, _descriptor_rows, _set("reason", 0, "vendor_gap"), RefusalReason.INVALID_ROW),
        (DESCRIPTOR, _descriptor_rows, _set("value", 1, np.inf), RefusalReason.INVALID_ROW),
        # ids and keys
        (DESCRIPTOR, _descriptor_rows, _set("asset_id", 1, "AAPL"), RefusalReason.INVALID_ASSET_ID),
        (
            DESCRIPTOR,
            _descriptor_rows,
            _set("asset_id", 1, "SYN_a"),
            RefusalReason.INVALID_ASSET_ID,
        ),
        (DESCRIPTOR, _descriptor_rows, _set("asset_id", 1, "SYN_A"), RefusalReason.INVALID_ROW),
        (
            DESCRIPTOR,
            _descriptor_rows,
            lambda r: r.iloc[::-1].reset_index(drop=True),
            RefusalReason.INVALID_ROW,
        ),
        (
            DESCRIPTOR,
            _descriptor_rows,
            _set("date", 1, "2024-12-23T12:00:00Z"),
            RefusalReason.INVALID_ROW,
        ),
        (DESCRIPTOR, _descriptor_rows, lambda r: r.iloc[:0], RefusalReason.INVALID_ROW),
        # weights
        (RULE_OUTPUT, _rule_rows, _set("weight", 0, 0.5), RefusalReason.INVALID_ROW),
        (RULE_OUTPUT, _rule_rows, _set("weight", 0, -0.1), RefusalReason.INVALID_ROW),
        (RULE_OUTPUT, _rule_rows, _set("weight", 0, 1.5), RefusalReason.INVALID_ROW),
    ],
)
def test_row_validator_refuses_each_violation(policy, rows, edit, reason):
    edited = edit(rows())
    with pytest.raises(PublicationRefused) as caught:
        check_rows(policy, check_schema(policy, edited))
    assert caught.value.reason is reason, caught.value


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r[["asset_id", "date", "value", "status", "reason", "detail"]],
        lambda r: r.drop(columns="detail"),
        lambda r: r.assign(extra=1.0),
        lambda r: r.assign(date=r["date"].dt.tz_localize(None)),
        lambda r: r.assign(date=r["date"].dt.tz_convert("America/New_York")),
        lambda r: r.assign(date=r["date"].dt.as_unit("ns")),
        lambda r: r.assign(value=r["value"].astype("float32")),
        lambda r: r.assign(asset_id=pd.Series([1] * len(r), dtype=object)),
        lambda r: "not a frame",
    ],
)
def test_schema_check_refuses_columns_dtypes_and_non_frames(edit):
    with pytest.raises(PublicationRefused) as caught:
        check_schema(DESCRIPTOR, edit(_descriptor_rows()))
    assert caught.value.reason is RefusalReason.SCHEMA_MISMATCH


def _weights(dates, weights):
    rows = _rule_rows().iloc[:0]
    n = len(weights)
    return pd.DataFrame(
        {
            "date": pd.Series(pd.to_datetime(dates, utc=True), dtype="datetime64[us, UTC]"),
            "asset_id": pd.Series([f"SYN_{chr(65 + i % 4)}" for i in range(n)], dtype="str"),
            "weight": pd.Series(weights, dtype="float64"),
            "status": pd.Series(["ok"] * n, dtype="str"),
            "reason": pd.Series([None] * n, dtype="str"),
            "detail": pd.Series([None] * n, dtype="str"),
        },
        columns=rows.columns,
    )


@pytest.mark.parametrize(
    ("dates", "weights"),
    [
        (["2024-12-31"] * 3, [1.5, -0.25, -0.25]),  # sums to exactly 1, out of bounds
        (["2024-12-30", "2024-12-30", "2024-12-31", "2024-12-31"], [0.25] * 4),  # sums to 1 overall
    ],
)
def test_weight_bounds_and_per_date_sums_are_checked_separately(dates, weights):
    rows = _weights(dates, weights)
    with pytest.raises(PublicationRefused) as caught:
        check_rows(RULE_OUTPUT, check_schema(RULE_OUTPUT, rows))
    assert caught.value.reason is RefusalReason.INVALID_ROW


def test_an_asset_id_with_a_trailing_newline_is_refused():
    rows = _descriptor_rows()
    rows["asset_id"] = rows["asset_id"].where(rows.index != 1, "SYN_B\n")
    with pytest.raises(PublicationRefused) as caught:
        check_rows(DESCRIPTOR, check_schema(DESCRIPTOR, rows))
    assert caught.value.reason is RefusalReason.INVALID_ASSET_ID


def _with_attrs(rows):
    rows.attrs["evidence_status"] = "prospectively_confirmed"
    return rows


@pytest.mark.parametrize(
    "edit",
    [_with_attrs, lambda r: r.set_index(pd.Index([str(i) for i in range(len(r))], name="date"))],
)
def test_unhashed_attrs_and_custom_indexes_are_refused(edit):
    with pytest.raises(PublicationRefused) as caught:
        check_schema(DESCRIPTOR, edit(_descriptor_rows()))
    assert caught.value.reason is RefusalReason.SCHEMA_MISMATCH


def test_validate_rechecks_the_code_sha_of_a_directly_built_envelope(tmp_path):
    import dataclasses

    from stock_agent.publication.envelope import CodeState
    from stock_agent.publication.reader import read_publication
    from stock_agent.publication.writer import write_publication

    spec = synthetic.SyntheticSpec(
        "syn_equal_weight", "1", {"n_assets": 3, "decision_date": "2024-12-31"}
    )
    path = write_publication(
        tmp_path,
        synthetic.draft(spec),
        as_of=pd.Timestamp("2024-12-31T23:59:59.999999Z"),
        created_at=pd.Timestamp("2026-10-13T12:00:00Z"),
        code_state=CodeState("e" * 40, False),
    )
    publication = read_publication(path)
    forged = dataclasses.replace(publication.envelope, code_sha="not-a-sha")
    with pytest.raises(PublicationRefused) as caught:
        validate(forged, publication.rows)
    assert caught.value.reason is RefusalReason.INVALID_FIELD
