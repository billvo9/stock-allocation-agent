import builtins
import socket
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

from stock_agent.model_validation.audit import table_sha256
from stock_agent.publication import synthetic
from stock_agent.publication.envelope import (
    EvidenceStatus,
    Kind,
    Limitation,
    PublicationRefused,
    RefusalReason,
)

PRICE_PARAMS = {"n_assets": 3, "start": "2024-12-02", "n_sessions": 22, "seed": 7, "n_missing": 2}
WEIGHT_PARAMS = {"n_assets": 3, "decision_date": "2024-12-31"}

# Golden content hashes (audit.table_sha256: a fixed byte layout, independent
# of parquet and pandas versions). A generator change without a version bump
# fails here. There is deliberately no golden file hash: pyarrow is unpinned.
GOLDEN = {
    ("syn_price_path", "1"): "09acf53a14eb211e71ecfa89719d326dddbf836cbd8d6b774c0e460cb5014549",
    ("syn_equal_weight", "1"): "52411e158e99b32a5ea8ab6be326d3ad5adee0a9ccc2bd70c88207f7ea6dde30",
}


def _spec(generator_id="syn_price_path", version="1", **changes):
    base = PRICE_PARAMS if generator_id == "syn_price_path" else WEIGHT_PARAMS
    return synthetic.SyntheticSpec(generator_id, version, {**base, **changes})


def _refused(reason, call, *args):
    with pytest.raises(PublicationRefused) as caught:
        call(*args)
    assert caught.value.reason is reason, caught.value


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_generators_match_their_golden_content_hash(key):
    params = PRICE_PARAMS if key[0] == "syn_price_path" else WEIGHT_PARAMS
    rows = synthetic.draft(synthetic.SyntheticSpec(*key, params)).rows
    assert table_sha256(rows) == GOLDEN[key]


def test_price_path_is_integer_cents_with_the_requested_gaps():
    rows = synthetic.draft(_spec()).rows
    assert len(rows) == 3 * 22
    assert list(rows["asset_id"].iloc[:3]) == ["SYN_A", "SYN_B", "SYN_C"]
    assert rows["date"].iloc[0] == pd.Timestamp("2024-12-02", tz="UTC")
    assert rows["date"].iloc[-1] == pd.Timestamp("2024-12-31", tz="UTC")
    ok = rows[rows["status"] == "ok"]
    assert np.allclose(ok["value"] * 100, np.round(ok["value"] * 100), atol=1e-9)
    gaps = rows[rows["status"] == "unavailable"]
    assert list(gaps["asset_id"]) == ["SYN_A", "SYN_A"]
    assert list(gaps["date"].dt.strftime("%Y-%m-%d")) == ["2024-12-02", "2024-12-03"]
    assert gaps["value"].isna().all() and (gaps["reason"] == "missing_input").all()


def test_equal_weight_is_one_over_n_on_the_decision_date():
    rows = synthetic.draft(_spec("syn_equal_weight")).rows
    assert list(rows["weight"]) == [1 / 3] * 3
    assert (rows["date"] == pd.Timestamp("2024-12-31", tz="UTC")).all()


def test_draft_declares_kind_status_and_synthetic_limitations():
    price = synthetic.draft(_spec())
    weight = synthetic.draft(_spec("syn_equal_weight"))
    assert (price.kind, price.evidence_status) == (
        Kind.SYNTHETIC_DESCRIPTOR,
        EvidenceStatus.DESCRIPTIVE,
    )
    assert (weight.kind, weight.evidence_status) == (
        Kind.SYNTHETIC_RULE_OUTPUT,
        EvidenceStatus.RULES_ILLUSTRATION,
    )
    for draft in (price, weight):
        assert Limitation.SYNTHETIC_DATA in draft.limitations
        assert draft.universe_id == "synthetic" and draft.input_hashes == ()


def test_generation_is_deterministic_and_seed_sensitive():
    assert table_sha256(synthetic.draft(_spec()).rows) == table_sha256(
        synthetic.draft(_spec()).rows
    )
    assert table_sha256(synthetic.draft(_spec(seed=8)).rows) != GOLDEN[("syn_price_path", "1")]


@pytest.mark.parametrize(
    "spec",
    [
        lambda: _spec(colour="red"),  # unknown key
        lambda: synthetic.SyntheticSpec("syn_equal_weight", "1", {"n_assets": 3}),  # missing key
        lambda: _spec(n_assets=True),  # bool is not an int
        lambda: _spec(n_assets=3.0),  # float
        lambda: _spec(n_assets=0),
        lambda: _spec(n_assets=9),
        lambda: _spec(n_sessions=521),
        lambda: _spec(seed=-1),
        lambda: _spec(n_missing=23),  # more gaps than sessions
        lambda: _spec(start="2024-12-28"),  # a Saturday
        lambda: _spec(start="2024-13-01"),
        lambda: _spec(start="2024-12-02T00:00:00Z"),
        lambda: _spec(start="2024-12-02\n"),
        lambda: _spec("syn_equal_weight", decision_date=20241231),
        lambda: _spec(version="2"),
        lambda: synthetic.SyntheticSpec("syn_real_prices", "1", {}),
        lambda: pd.DataFrame({"date": [], "asset_id": [], "value": []}),
        lambda: Path("data/prices.parquet"),
        lambda: synthetic.SyntheticSpec("syn_price_path", "1", [("n_assets", 3)]),
    ],
)
def test_adapter_refuses_every_spec_outside_the_registry_contract(spec):
    with pytest.raises(PublicationRefused) as caught:
        synthetic.draft(spec())
    assert caught.value.reason is RefusalReason.INVALID_SPEC


def test_registry_is_closed_and_read_only():
    assert isinstance(synthetic.REGISTRY, MappingProxyType)
    assert set(synthetic.REGISTRY) == set(GOLDEN)
    with pytest.raises(TypeError):
        synthetic.REGISTRY[("syn_real_prices", "1")] = None  # type: ignore[index]


def test_generators_do_no_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("I/O attempted")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    for reader in ("read_parquet", "read_csv", "read_json", "read_sql", "read_excel"):
        monkeypatch.setattr(pd, reader, forbidden)
    for key in GOLDEN:
        params = PRICE_PARAMS if key[0] == "syn_price_path" else WEIGHT_PARAMS
        assert len(synthetic.draft(synthetic.SyntheticSpec(*key, params)).rows) > 0
