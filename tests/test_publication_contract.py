"""
Adversarial contract tests: every case runs through the writer (refused,
nothing left in the root) and through the reader. Reader cases are built by
`republish`, which plays an adversary with write access: it edits the
content consistently, then recomputes every hash and the id, so the check
named by the expected reason is the one that must fire.
"""

import dataclasses
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_validation.audit import file_sha256, table_sha256
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START
from stock_agent.publication import synthetic
from stock_agent.publication.envelope import (
    ENVELOPE_FILE,
    MANIFEST_FILE,
    ROWS_FILE,
    CodeState,
    EvidenceStatus,
    Limitation,
    PublicationRefused,
    RefusalReason,
    derive_publication_id,
    format_timestamp,
    synthetic_session_end,
)
from stock_agent.publication.policy import DescriptorPublication, RuleOutputPublication
from stock_agent.publication.reader import read_publication
from stock_agent.publication.writer import write_publication

NOW = pd.Timestamp("2026-10-13T12:00:00Z")
AS_OF = pd.Timestamp("2024-12-31T23:59:59.999999Z")
CLEAN = CodeState("a" * 40, False)
PRICE = {"n_assets": 3, "start": "2024-12-02", "n_sessions": 22, "seed": 7, "n_missing": 2}
WEIGHT = {"n_assets": 3, "decision_date": "2024-12-31"}


def make_draft(base="price", **params):
    if base == "price":
        return synthetic.draft(synthetic.SyntheticSpec("syn_price_path", "1", {**PRICE, **params}))
    return synthetic.draft(synthetic.SyntheticSpec("syn_equal_weight", "1", {**WEIGHT, **params}))


def publish(root, draft=None, *, as_of=AS_OF, created_at=NOW, code_state=CLEAN):
    root.mkdir(parents=True, exist_ok=True)
    return write_publication(
        root,
        make_draft() if draft is None else draft,
        as_of=as_of,
        created_at=created_at,
        code_state=code_state,
    )


def _content_id(fields, rows_sha):
    try:
        return derive_publication_id(fields, rows_sha)
    except PublicationRefused:  # a deliberately malformed created_at
        payload = json.dumps([fields, rows_sha], sort_keys=True, default=str).encode()
        return f"20261013T120000000000Z-{hashlib.sha256(payload).hexdigest()[:12]}"


def republish(path, root, *, envelope=None, rows=None, manifest=None):
    """Edit a publication, recompute every hash and the id, and store it anew."""

    data = json.loads((path / ENVELOPE_FILE).read_text())
    frame = pd.read_parquet(path / ROWS_FILE)
    if envelope is not None:
        envelope(data)
    if rows is not None:
        frame = rows(frame)
    fields = {key: value for key, value in data.items() if key != "publication_id"}
    rows_sha = table_sha256(frame)
    data["publication_id"] = _content_id(fields, rows_sha)
    out = root / data["publication_id"]
    out.mkdir(parents=True)
    frame.to_parquet(out / ROWS_FILE, index=False)
    (out / ENVELOPE_FILE).write_text(json.dumps(data, indent=2, sort_keys=True))
    content = {
        "schema_version": data.get("schema_version"),
        "publication_id": data["publication_id"],
        "files": {
            ENVELOPE_FILE: {"file_sha256": file_sha256(out / ENVELOPE_FILE)},
            ROWS_FILE: {
                "file_sha256": file_sha256(out / ROWS_FILE),
                "table_sha256": rows_sha,
                "rows": len(frame),
            },
        },
    }
    if manifest is not None:
        manifest(content)
    (out / MANIFEST_FILE).write_text(json.dumps(content, indent=2, sort_keys=True))
    return out


def respec(**params):
    """A consistent specification edit: new params, regenerated rows, recomputed knowledge time."""

    state = {}

    def envelope(data):
        data["parameters"] = {**data["parameters"], **params}
        spec = synthetic.SyntheticSpec(
            data["method_id"], data["method_version"], data["parameters"]
        )
        state["rows"] = synthetic.draft(spec).rows
        latest = state["rows"]["date"].max()
        data["knowledge_time"] = format_timestamp("k", synthetic_session_end(latest))

    return {"envelope": envelope, "rows": lambda frame: state["rows"]}


def setter(**values):
    return {"envelope": lambda data: data.update(values)}


def draft_with(base="price", **changes):
    return lambda: {"draft": dataclasses.replace(make_draft(base), **changes)}


def rows_edit(edit):
    return {"rows": edit}


def _cell(column, value, index=4):  # row 4: SYN_B, an ok row
    def edit(frame):
        frame = frame.copy()
        frame.loc[index, column] = value
        return frame

    return edit


def _rename_assets(frame):
    names = {"SYN_A": "AAPL", "SYN_B": "MSFT", "SYN_C": "NVDA"}
    return frame.assign(asset_id=frame["asset_id"].map(names).astype("str"))


def _with_attrs(frame):
    frame = frame.copy()
    frame.attrs["evidence_status"] = "prospectively_confirmed"
    return frame


def _into_2025(frame):
    return frame.assign(date=frame["date"] + pd.Timedelta(days=1))


def _nudge(frame):
    frame = frame.copy()
    frame.loc[4, "value"] = frame.loc[4, "value"] + 1e-9
    return frame


@dataclass(frozen=True)
class Case:
    name: str
    reason: RefusalReason
    base: str = "price"
    write: Callable[[], dict] | None = None
    read: dict | None = None
    allow_dirty: bool = False


DIRTY = CodeState("a" * 40, True)
EXTERNAL = {"input_id": "prices", "sha256": "0" * 64}

CASES = [
    # lockbox
    Case(
        "run_crossing_into_2025",
        RefusalReason.LOCKBOX_VIOLATION,
        write=lambda: {"draft": make_draft(start="2024-12-27", n_sessions=5)},
        read=respec(start="2024-12-27", n_sessions=5),
    ),
    Case(
        "rule_decision_on_holdout_start",
        RefusalReason.LOCKBOX_VIOLATION,
        base="weight",
        write=lambda: {"draft": make_draft("weight", decision_date="2025-01-01")},
        read=respec(decision_date="2025-01-01"),
    ),
    Case(
        "as_of_at_holdout_start",
        RefusalReason.LOCKBOX_VIOLATION,
        write=lambda: {"as_of": MODEL_HOLDOUT_START},
        read=setter(as_of="2025-01-01T00:00:00.000000Z"),
    ),
    # knowledge time
    Case(
        "as_of_one_microsecond_short",
        RefusalReason.KNOWLEDGE_AFTER_AS_OF,
        write=lambda: {"as_of": pd.Timestamp("2024-12-31T23:59:59.999998Z")},
        read=setter(as_of="2024-12-31T23:59:59.999998Z"),
    ),
    Case(
        "as_of_at_whole_second",
        RefusalReason.KNOWLEDGE_AFTER_AS_OF,
        write=lambda: {"as_of": pd.Timestamp("2024-12-31T23:59:59Z")},
        read=setter(as_of="2024-12-31T23:59:59.000000Z"),
    ),
    Case(
        "as_of_between_two_sessions",
        RefusalReason.KNOWLEDGE_AFTER_AS_OF,
        write=lambda: {"as_of": pd.Timestamp("2024-12-30T23:59:59.999999Z")},
        read=setter(as_of="2024-12-30T23:59:59.999999Z"),
    ),
    Case(
        "created_before_as_of",
        RefusalReason.AS_OF_AFTER_CREATED_AT,
        write=lambda: {"created_at": pd.Timestamp("2024-12-31T12:00:00Z")},
        read=setter(created_at="2024-12-31T12:00:00.000000Z"),
    ),
    Case(
        "naive_as_of",
        RefusalReason.INVALID_TIMESTAMP,
        write=lambda: {"as_of": pd.Timestamp("2024-12-31T23:59:59.999999")},
        read=setter(as_of="2024-12-31T23:59:59.999999"),
    ),
    Case(
        "offset_as_of",  # 20:00 at -05:00 is 2025-01-01T01:00Z
        RefusalReason.INVALID_TIMESTAMP,
        write=lambda: {"as_of": pd.Timestamp("2024-12-31T20:00:00", tz="America/New_York")},
        read=setter(as_of="2024-12-31T20:00:00.000000-05:00"),
    ),
    Case(
        "knowledge_time_edited_alone",
        RefusalReason.KNOWLEDGE_TIME_MISMATCH,
        read=setter(knowledge_time="2024-12-30T23:59:59.999999Z"),
    ),
    Case(
        "non_synthetic_knowledge_rule",
        RefusalReason.KNOWLEDGE_RULE_NOT_PERMITTED,
        write=draft_with(knowledge_time_rule="production_close_1800_et"),
        read=setter(knowledge_time_rule="production_close_1800_et"),
    ),
    # evidence status and the clean tree
    Case(
        "descriptor_claiming_rules_illustration",
        RefusalReason.STATUS_NOT_ALLOWED,
        write=draft_with(evidence_status=EvidenceStatus.RULES_ILLUSTRATION),
        read=setter(evidence_status="rules_illustration"),
    ),
    Case(
        "rule_output_claiming_descriptive",
        RefusalReason.STATUS_NOT_ALLOWED,
        base="weight",
        write=draft_with("weight", evidence_status=EvidenceStatus.DESCRIPTIVE),
        read=setter(evidence_status="descriptive"),
    ),
    Case(
        "unknown_status",
        RefusalReason.UNKNOWN_STATUS,
        write=draft_with(evidence_status="prospectively_confirmed"),
        read=setter(evidence_status="prospectively_confirmed"),
    ),
    Case(
        "dirty_rules_illustration_even_in_research_mode",
        RefusalReason.DIRTY_TREE,
        base="weight",
        write=lambda: {"draft": make_draft("weight"), "code_state": DIRTY},
        read=setter(code_dirty=True),
        allow_dirty=True,
    ),
    # synthetic only: real data is refused
    Case(
        "unknown_kind",
        RefusalReason.UNSUPPORTED_KIND,
        write=draft_with(kind="performance_comparison"),
        read=setter(kind="performance_comparison"),
    ),
    Case(
        "real_universe",
        RefusalReason.PROVENANCE_MISMATCH,
        write=draft_with(universe_id="us_large_cap_u1"),
        read=setter(universe_id="us_large_cap_u1"),
    ),
    Case(
        "external_inputs",
        RefusalReason.PROVENANCE_MISMATCH,
        write=draft_with(input_hashes=(EXTERNAL,)),
        read=setter(input_hashes=[EXTERNAL]),
    ),
    Case(
        "synthetic_limitation_removed",
        RefusalReason.MISSING_LIMITATION,
        write=draft_with(limitations=(Limitation.SYNTHETIC_CALENDAR,)),
        read=setter(limitations=["synthetic_calendar"]),
    ),
    Case(
        "real_tickers",
        RefusalReason.INVALID_ASSET_ID,
        write=lambda: {
            "draft": dataclasses.replace(make_draft(), rows=_rename_assets(make_draft().rows))
        },
        read=rows_edit(_rename_assets),
    ),
    Case(
        "synthetic_ids_with_foreign_values",
        RefusalReason.PROVENANCE_MISMATCH,
        write=lambda: {"draft": dataclasses.replace(make_draft(), rows=_nudge(make_draft().rows))},
        read=rows_edit(_nudge),
    ),
    Case(
        "synthetic_calendar_limitation_removed",
        RefusalReason.PROVENANCE_MISMATCH,
        write=draft_with(limitations=(Limitation.SYNTHETIC_DATA,)),
        read=setter(limitations=["synthetic_data"]),
    ),
    Case(
        "rows_moved_into_2025_with_old_knowledge_time",
        RefusalReason.LOCKBOX_VIOLATION,
        base="weight",
        read=rows_edit(_into_2025),
    ),
    Case(
        "rows_with_unhashed_attrs",
        RefusalReason.SCHEMA_MISMATCH,
        write=lambda: {
            "draft": dataclasses.replace(make_draft(), rows=_with_attrs(make_draft().rows))
        },
        read=rows_edit(_with_attrs),
    ),
    Case(
        "code_sha_with_trailing_newline",
        RefusalReason.INVALID_FIELD,
        write=lambda: {"code_state": CodeState("a" * 40 + "\n", False)},
        read=setter(code_sha="a" * 40 + "\n"),
    ),
    Case(
        "numpy_parameter",
        RefusalReason.INVALID_JSON,
        write=draft_with(parameters={**PRICE, "n_assets": np.int64(3)}),
    ),
    Case(
        "lone_surrogate_universe",
        RefusalReason.INVALID_JSON,
        write=draft_with(universe_id="\ud800"),
        read=setter(universe_id="\ud800"),
    ),
    # row validity
    Case(
        "ok_row_without_a_value",
        RefusalReason.INVALID_ROW,
        write=lambda: {
            "draft": dataclasses.replace(
                make_draft(), rows=_cell("value", float("nan"))(make_draft().rows)
            )
        },
        read=rows_edit(_cell("value", float("nan"))),
    ),
]


@pytest.mark.parametrize("case", [c for c in CASES if c.write], ids=lambda c: c.name)
def test_writer_refuses_and_leaves_nothing(case, tmp_path):
    root = tmp_path / "root"
    with pytest.raises(PublicationRefused) as caught:
        publish(root, **case.write())
    assert caught.value.reason is case.reason, caught.value
    assert list(root.iterdir()) == []


@pytest.mark.parametrize("case", [c for c in CASES if c.read], ids=lambda c: c.name)
def test_reader_refuses_consistently_rehashed_tampering(case, tmp_path):
    draft = make_draft(case.base)
    original = publish(tmp_path / "original", draft)
    tampered = republish(original, tmp_path / "tampered", **case.read)
    with pytest.raises(PublicationRefused) as caught:
        read_publication(tampered, allow_dirty=case.allow_dirty)
    assert caught.value.reason is case.reason, caught.value


@pytest.mark.parametrize("base", ["price", "weight"])
def test_control_a_no_op_republish_reads_back(base, tmp_path):
    original = publish(tmp_path / "original", make_draft(base))
    copy = republish(original, tmp_path / "copy")
    assert copy.name == original.name
    publication = read_publication(copy)
    expected = DescriptorPublication if base == "price" else RuleOutputPublication
    assert type(publication) is expected
    pd.testing.assert_frame_equal(publication.rows, make_draft(base).rows)


# --- integrity and structure (no rehash) ------------------------------------


def _flip_last_byte(path):
    data = bytearray(path.read_bytes())
    data[-2] ^= 0x01
    path.write_bytes(bytes(data))


def _refused_read(path, reason, **kwargs):
    with pytest.raises(PublicationRefused) as caught:
        read_publication(path, **kwargs)
    assert caught.value.reason is reason, caught.value


@pytest.mark.parametrize("name", [ROWS_FILE, ENVELOPE_FILE])
def test_a_flipped_byte_is_a_hash_mismatch(name, tmp_path):
    path = publish(tmp_path / "root")
    _flip_last_byte(path / name)
    _refused_read(path, RefusalReason.HASH_MISMATCH)


def test_an_edited_table_hash_is_a_hash_mismatch(tmp_path):
    path = publish(tmp_path / "root")
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    manifest["files"][ROWS_FILE]["table_sha256"] = "0" * 64
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    _refused_read(path, RefusalReason.HASH_MISMATCH)


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda m: m.update(schema_version="1.1"), RefusalReason.UNSUPPORTED_VERSION),
        (lambda m: m.update(schema_version="2.0"), RefusalReason.UNSUPPORTED_VERSION),
        (lambda m: m.update(schema_version="1"), RefusalReason.MALFORMED_VERSION),
        (lambda m: m.update(schema_version="1.0.0"), RefusalReason.MALFORMED_VERSION),
        (lambda m: m.update(schema_version="1.-1"), RefusalReason.MALFORMED_VERSION),
        (lambda m: m.update(schema_version=1.0), RefusalReason.MALFORMED_VERSION),
        (lambda m: m.update(extra=True), RefusalReason.FIELD_SET),
        (lambda m: m["files"][ROWS_FILE].update(rows=True), RefusalReason.HASH_MISMATCH),
        (lambda m: m["files"][ROWS_FILE].update(rows=1), RefusalReason.HASH_MISMATCH),
        (lambda m: m["files"].pop(ENVELOPE_FILE), RefusalReason.FIELD_SET),
    ],
)
def test_manifest_versions_and_fields_fail_closed(edit, reason, tmp_path):
    original = publish(tmp_path / "original")
    _refused_read(republish(original, tmp_path / "t", manifest=edit), reason)


def test_envelope_version_must_match_the_manifest(tmp_path):
    original = publish(tmp_path / "original")
    tampered = republish(
        original,
        tmp_path / "t",
        envelope=lambda d: d.update(schema_version="1.1"),
        manifest=lambda m: m.update(schema_version="1.0"),
    )
    _refused_read(tampered, RefusalReason.UNSUPPORTED_VERSION)


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (lambda d: d.pop("as_of"), RefusalReason.FIELD_SET),
        (lambda d: d.update(claim_type="descriptive_fact"), RefusalReason.FIELD_SET),
        (lambda d: d.update(code_dirty="false"), RefusalReason.INVALID_FIELD),
        (lambda d: d.update(code_sha="0" * 39), RefusalReason.INVALID_FIELD),
    ],
)
def test_envelope_fields_fail_closed(edit, reason, tmp_path):
    original = publish(tmp_path / "original")
    _refused_read(republish(original, tmp_path / "t", envelope=edit), reason)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"code_dirty": false', '"code_dirty": NaN'),
        ('"code_dirty": false', '"code_dirty": false, "code_dirty": false'),
    ],
)
def test_non_finite_tokens_and_duplicate_keys_are_refused(old, new, tmp_path):
    path = republish(publish(tmp_path / "original"), tmp_path / "t")
    text = (path / ENVELOPE_FILE).read_text()
    assert old in text
    (path / ENVELOPE_FILE).write_text(text.replace(old, new))
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    manifest["files"][ENVELOPE_FILE]["file_sha256"] = file_sha256(path / ENVELOPE_FILE)
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    _refused_read(path, RefusalReason.INVALID_JSON)


def test_the_id_must_match_the_content(tmp_path):
    original = publish(tmp_path / "original")
    other_id = "20261013T120000000000Z-0123456789ab"

    path = tmp_path / "t" / other_id
    path.mkdir(parents=True)
    for name in (ROWS_FILE, ENVELOPE_FILE):
        (path / name).write_bytes((original / name).read_bytes())
    data = json.loads((path / ENVELOPE_FILE).read_text())
    data["publication_id"] = other_id
    (path / ENVELOPE_FILE).write_text(json.dumps(data))
    manifest = json.loads((original / MANIFEST_FILE).read_text())
    manifest["publication_id"] = other_id
    manifest["files"][ENVELOPE_FILE]["file_sha256"] = file_sha256(path / ENVELOPE_FILE)
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    _refused_read(path, RefusalReason.ID_MISMATCH)


def test_a_renamed_directory_is_refused(tmp_path):
    path = publish(tmp_path / "root")
    renamed = path.with_name("20261013T120000000000Z-ffffffffffff")
    path.rename(renamed)
    _refused_read(renamed, RefusalReason.ID_MISMATCH)
    plain = renamed.with_name("latest")
    renamed.rename(plain)
    _refused_read(plain, RefusalReason.NOT_A_PUBLICATION)


def test_missing_and_extra_files_are_refused(tmp_path):
    path = publish(tmp_path / "root")
    (path / "notes.txt").write_text("x")
    _refused_read(path, RefusalReason.FILE_SET)
    (path / "notes.txt").unlink()
    (path / MANIFEST_FILE).unlink()
    _refused_read(path, RefusalReason.FILE_SET)


def test_symlinks_are_refused(tmp_path):
    path = publish(tmp_path / "root")
    link = tmp_path / "links" / path.name
    link.parent.mkdir()
    os.symlink(path, link)
    _refused_read(link, RefusalReason.SYMLINK)

    copy = republish(path, tmp_path / "copy")
    rows = copy / ROWS_FILE
    target = tmp_path / "elsewhere.parquet"
    rows.rename(target)
    os.symlink(target, rows)
    _refused_read(copy, RefusalReason.SYMLINK)


def test_a_missing_path_is_not_a_publication(tmp_path):
    _refused_read(tmp_path / "20261013T120000000000Z-0123456789ab", RefusalReason.NOT_A_PUBLICATION)


def test_dirty_descriptive_needs_research_mode(tmp_path):
    path = publish(tmp_path / "root", code_state=DIRTY)
    _refused_read(path, RefusalReason.DIRTY_TREE)
    assert read_publication(path, allow_dirty=True).envelope.code_dirty is True


def test_a_custom_index_in_the_rows_file_is_refused(tmp_path):
    path = republish(publish(tmp_path / "original"), tmp_path / "t")
    frame = pd.read_parquet(path / ROWS_FILE)
    frame.index = pd.Index([f"r{i}" for i in range(len(frame))], name="date")
    frame.to_parquet(path / ROWS_FILE, index=True)
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    manifest["files"][ROWS_FILE]["file_sha256"] = file_sha256(path / ROWS_FILE)
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    _refused_read(path, RefusalReason.SCHEMA_MISMATCH)


def test_an_unreadable_rows_file_is_refused_with_a_reason(tmp_path):
    path = republish(publish(tmp_path / "original"), tmp_path / "t")
    (path / ROWS_FILE).write_bytes(b"not a parquet file")
    manifest = json.loads((path / MANIFEST_FILE).read_text())
    manifest["files"][ROWS_FILE]["file_sha256"] = file_sha256(path / ROWS_FILE)
    (path / MANIFEST_FILE).write_text(json.dumps(manifest))
    _refused_read(path, RefusalReason.SCHEMA_MISMATCH)
