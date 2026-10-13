import errno
import inspect
import json
import os

import pandas as pd
import pytest

from stock_agent.publication import synthetic, writer
from stock_agent.publication.envelope import (
    ENVELOPE_FILE,
    MANIFEST_FILE,
    ROWS_FILE,
    ClaimType,
    CodeState,
    EvidenceStatus,
    Kind,
    PublicationRefused,
    RefusalReason,
)
from stock_agent.publication.reader import read_publication
from stock_agent.publication.writer import write_publication

NOW = pd.Timestamp("2026-10-13T12:00:00Z")
AS_OF = pd.Timestamp("2024-12-31T23:59:59.999999Z")
CLEAN = CodeState("b" * 40, False)
PRICE = {"n_assets": 3, "start": "2024-12-02", "n_sessions": 22, "seed": 7, "n_missing": 2}
WEIGHT = {"n_assets": 3, "decision_date": "2024-12-31"}


def _draft(generator="syn_price_path", params=None):
    params = params if params is not None else PRICE if generator == "syn_price_path" else WEIGHT
    return synthetic.draft(synthetic.SyntheticSpec(generator, "1", params))


def _write(root, draft=None, **overrides):
    root.mkdir(parents=True, exist_ok=True)
    kwargs = {"as_of": AS_OF, "created_at": NOW, "code_state": CLEAN, **overrides}
    return write_publication(root, draft if draft is not None else _draft(), **kwargs)


def _refused(reason, call, *args, **kwargs):
    with pytest.raises(PublicationRefused) as caught:
        call(*args, **kwargs)
    assert caught.value.reason is reason, caught.value


@pytest.mark.parametrize(
    ("generator", "kind", "status", "claim"),
    [
        (
            "syn_price_path",
            Kind.SYNTHETIC_DESCRIPTOR,
            EvidenceStatus.DESCRIPTIVE,
            ClaimType.DESCRIPTIVE_FACT,
        ),
        (
            "syn_equal_weight",
            Kind.SYNTHETIC_RULE_OUTPUT,
            EvidenceStatus.RULES_ILLUSTRATION,
            ClaimType.RULE_ILLUSTRATION,
        ),
    ],
)
def test_both_kinds_write_and_read_back(generator, kind, status, claim, tmp_path):
    draft = _draft(generator)
    path = _write(tmp_path / "root", draft)
    assert sorted(entry.name for entry in path.parent.iterdir()) == [path.name]
    assert sorted(entry.name for entry in path.iterdir()) == [
        ENVELOPE_FILE,
        MANIFEST_FILE,
        ROWS_FILE,
    ]
    publication = read_publication(path)
    envelope = publication.envelope
    assert (envelope.kind, envelope.evidence_status, publication.claim_type) == (
        kind,
        status,
        claim,
    )
    assert envelope.knowledge_time == pd.Timestamp("2024-12-31T23:59:59.999999Z")
    assert envelope.created_at == NOW and envelope.as_of == AS_OF
    assert envelope.code_sha == "b" * 40 and envelope.code_dirty is False
    pd.testing.assert_frame_equal(publication.rows, draft.rows)
    stored = json.loads((path / ENVELOPE_FILE).read_text())
    assert "claim_type" not in stored
    assert stored["knowledge_time"] == "2024-12-31T23:59:59.999999Z"


def test_as_of_equal_to_knowledge_time_is_accepted(tmp_path):
    path = _write(tmp_path / "root", as_of=pd.Timestamp("2024-12-31T23:59:59.999999Z"))
    assert read_publication(path).envelope.as_of == read_publication(path).envelope.knowledge_time


def test_a_2026_write_time_is_not_a_lockbox_date(tmp_path):
    path = _write(tmp_path / "root", created_at=pd.Timestamp("2026-12-31T23:00:00Z"))
    assert read_publication(path).envelope.created_at.year == 2026


def test_a_second_write_of_the_same_publication_is_refused(tmp_path):
    root = tmp_path / "root"
    path = _write(root)
    _refused(RefusalReason.ALREADY_EXISTS, _write, root)
    assert sorted(entry.name for entry in root.iterdir()) == [path.name]


def test_a_pre_created_empty_directory_blocks_the_id(tmp_path):
    root = tmp_path / "root"
    name = _write(tmp_path / "probe").name
    (root / name).mkdir(parents=True)
    _refused(RefusalReason.ALREADY_EXISTS, _write, root)
    assert list((root / name).iterdir()) == []


def _fail_on_call(number, original):
    calls = {"n": 0}

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == number:
            raise OSError("injected failure")
        return original(*args, **kwargs)

    return wrapped


@pytest.mark.parametrize("stage", ["rows", "envelope", "manifest", "rename"])
def test_a_failure_at_any_stage_leaves_nothing(stage, tmp_path, monkeypatch):
    if stage == "rows":
        monkeypatch.setattr(pd.DataFrame, "to_parquet", _fail_on_call(1, pd.DataFrame.to_parquet))
    elif stage == "envelope":
        monkeypatch.setattr(writer, "_write_json", _fail_on_call(1, writer._write_json))
    elif stage == "manifest":
        monkeypatch.setattr(writer, "_write_json", _fail_on_call(2, writer._write_json))
    else:
        monkeypatch.setattr(writer.os, "rename", _fail_on_call(1, writer.os.rename))
    root = tmp_path / "root"
    with pytest.raises(OSError, match="injected failure"):
        _write(root)
    assert list(root.iterdir()) == []


def test_the_same_content_gets_the_same_id_and_identical_files(tmp_path):
    first = _write(tmp_path / "a")
    reordered = dict(reversed(list(PRICE.items())))
    second = _write(tmp_path / "b", _draft(params=reordered))
    assert first.name == second.name
    for name in (MANIFEST_FILE, ENVELOPE_FILE):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_different_content_gets_a_different_id(tmp_path):
    first = _write(tmp_path / "a")
    assert _write(tmp_path / "b", _draft(params={**PRICE, "seed": 8})).name != first.name
    assert _write(tmp_path / "d", code_state=CodeState("c" * 40, False)).name != first.name


def test_dirty_descriptive_is_written_but_needs_research_mode_to_read(tmp_path):
    path = _write(tmp_path / "root", code_state=CodeState("b" * 40, True))
    _refused(RefusalReason.DIRTY_TREE, read_publication, path)
    assert read_publication(path, allow_dirty=True).envelope.code_dirty is True


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"code_state": CodeState("abc", False)}, RefusalReason.INVALID_FIELD),
        ({"code_state": CodeState("b" * 40, "false")}, RefusalReason.INVALID_FIELD),
        ({"code_state": ("b" * 40, False)}, RefusalReason.INVALID_FIELD),
        ({"created_at": pd.Timestamp("2026-10-13T12:00:00")}, RefusalReason.INVALID_TIMESTAMP),
    ],
)
def test_injected_inputs_are_validated(overrides, reason, tmp_path):
    root = tmp_path / "root"
    _refused(reason, _write, root, **overrides)
    assert list(root.iterdir()) == []


def test_only_drafts_are_accepted(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    frame = _draft().rows
    _refused(
        RefusalReason.INVALID_FIELD,
        write_publication,
        root,
        frame,
        as_of=AS_OF,
        created_at=NOW,
        code_state=CLEAN,
    )


def test_root_must_exist_and_has_no_default(tmp_path):
    _refused(
        RefusalReason.INVALID_FIELD,
        write_publication,
        tmp_path / "missing",
        _draft(),
        as_of=AS_OF,
        created_at=NOW,
        code_state=CLEAN,
    )
    parameters = inspect.signature(write_publication).parameters
    assert parameters["root"].default is inspect.Parameter.empty
    assert inspect.signature(read_publication).parameters["path"].default is inspect.Parameter.empty


def test_a_dangling_symlink_blocks_the_id(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    name = _write(tmp_path / "probe").name
    os.symlink(tmp_path / "nowhere", root / name)
    _refused(RefusalReason.ALREADY_EXISTS, _write, root)
    assert [entry.name for entry in root.iterdir()] == [name]


def test_losing_a_rename_race_is_already_exists_and_leaves_nothing(tmp_path, monkeypatch):
    def lost(*args):
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr(writer.os, "rename", lost)
    root = tmp_path / "root"
    _refused(RefusalReason.ALREADY_EXISTS, _write, root)
    assert list(root.iterdir()) == []
