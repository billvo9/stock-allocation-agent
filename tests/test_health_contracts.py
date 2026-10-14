"""
Artifact-contract checks: healthy on the current code, and failed (never
crashed) when a contract is broken: a reader that stops refusing edits, a
generator whose output changes, a nondeterministic generator, or a probe
that raises.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from stock_agent.health import contracts
from stock_agent.health.model import HealthState
from stock_agent.publication import synthetic

NOW = datetime(2026, 10, 14, 18, tzinfo=UTC)


def run() -> dict:
    return {check.check_id: check for check in contracts.contract_checks(now=lambda: NOW)}


def test_every_contract_holds_on_the_current_code():
    checks = run()
    assert {key: (c.state, c.value) for key, c in checks.items()} == {
        "contract.publication": (HealthState.HEALTHY, 4),
        "contract.fund_data": (HealthState.HEALTHY, 5),
        "contract.generators": (HealthState.HEALTHY, 3),
    }
    assert "refused: lockbox_violation" in checks["contract.publication"].detail
    assert "refused: pending_od9" in checks["contract.fund_data"].detail


def test_the_recorded_hashes_are_the_generators_current_output():
    assert contracts.generator_hashes() == contracts.RECORDED_HASHES


def test_a_reader_that_accepts_edits_breaks_the_publication_contract(monkeypatch):
    rows = synthetic.draft(contracts.PRICE_PATH).rows
    monkeypatch.setattr(
        contracts, "read_publication", lambda path, **_: SimpleNamespace(rows=rows.copy())
    )
    check = run()["contract.publication"]
    assert (check.state, check.reason, check.value) == (HealthState.FAILED, "contract_broken", 2)
    assert "[BROKEN] an edited envelope is refused (hash_mismatch): accepted" in check.detail
    assert "[BROKEN] a dirty-tree publication is refused by default" in check.detail


def test_a_changed_generator_output_breaks_the_hash_contract(monkeypatch):
    changed = {**contracts.RECORDED_HASHES, "synthetic_funds v2 raw payload": "0" * 64}
    monkeypatch.setattr(contracts, "RECORDED_HASHES", changed)
    check = run()["contract.generators"]
    assert (check.state, check.value) == (HealthState.FAILED, 2)
    assert "recorded 0000000000000000" in check.detail


def test_a_nondeterministic_generator_is_named(monkeypatch):
    counter = itertools.count()
    real = contracts.generator_hashes()
    monkeypatch.setattr(
        contracts,
        "generator_hashes",
        lambda: {**real, "publication syn_equal_weight v1 rows": str(next(counter))},
    )
    check = run()["contract.generators"]
    assert check.state is HealthState.FAILED
    assert "two runs differ (nondeterministic)" in check.detail


@pytest.mark.parametrize("error", [OSError("disk full"), KeyError("batch_id")])
def test_a_probe_that_raises_fails_its_check_without_crashing(monkeypatch, error):
    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(contracts, "ingest_batch", broken)
    check = run()["contract.fund_data"]
    assert check.state is HealthState.FAILED
    assert f"raised {type(error).__name__}" in check.detail
