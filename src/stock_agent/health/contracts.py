"""
Artifact-contract checks on synthetic fixtures. Each check runs a few probes
through the real code in a temporary directory and expects an exact outcome:

contract.publication  (publication contract v1, `stock_agent.publication`)
    a written publication reads back with identical rows; an edited file is
    refused (`hash_mismatch`); rows reaching the lockbox are refused
    (`lockbox_violation`); a dirty-tree publication is refused by the default
    reader (`dirty_tree`).
contract.fund_data    (F1 fund-data contract, `stock_agent.data.funds`)
    an ingested synthetic batch reads back through the fail-closed store
    reader with schema 1.0, every requested session present and no missing
    session; the same batch twice is refused (`already_exists`); a request
    past the lockbox is refused (`request_after_cap`); an availability rule
    is refused before OD9 (`pending_od9`); an edited table is refused
    (`hash_mismatch`).
contract.generators   (synthetic generators reproducible by hash)
    each generator is run twice and must give the recorded SHA-256. A change
    means a generator, its inputs or the random stream changed; update a
    recorded hash only for an intended, reviewed generator version change.

The fixtures are synthetic and in the repository; nothing reads data/,
reports/ or the network. A probe that raises anything unexpected fails its
check with the exception named, rather than crashing the health run.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pandas as pd

from stock_agent.data.funds.adapter import SYNTHETIC_FUND_IDS, SyntheticFundAdapter
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.reader import read_store
from stock_agent.data.funds.schema import SCHEMA_VERSION as FUND_SCHEMA_VERSION
from stock_agent.data.funds.schema import FundDataRefused
from stock_agent.health.model import Area, Better, Check, HealthState
from stock_agent.model_validation.audit import table_sha256
from stock_agent.publication import synthetic
from stock_agent.publication.envelope import CodeState, PublicationRefused
from stock_agent.publication.reader import read_publication
from stock_agent.publication.writer import write_publication

FIXTURE_CODE = CodeState("0" * 40, False)
PRICE_PATH = synthetic.SyntheticSpec(
    "syn_price_path",
    "1",
    {"n_assets": 3, "start": "2020-01-06", "n_sessions": 30, "seed": 7, "n_missing": 2},
)
EQUAL_WEIGHT = synthetic.SyntheticSpec(
    "syn_equal_weight", "1", {"n_assets": 3, "decision_date": "2020-02-14"}
)
LOCKBOX_PATH = synthetic.SyntheticSpec(
    "syn_price_path",
    "1",
    {"n_assets": 1, "start": "2024-12-30", "n_sessions": 5, "seed": 1, "n_missing": 0},
)
AS_OF = pd.Timestamp("2020-03-01", tz="UTC")
CREATED_AT = pd.Timestamp("2020-03-01T01:00:00", tz="UTC")

FUND_RETRIEVED_AT = pd.Timestamp("2025-01-02T12:00:00", tz="UTC")
FUND_START = pd.Timestamp("2024-10-01", tz="UTC")
FUND_END = pd.Timestamp("2025-01-01", tz="UTC")
FUND_SESSIONS = 64  # NYSE sessions in [2024-10-01, 2025-01-01)

# Recorded SHA-256 of each generator's output for the fixed specifications above.
RECORDED_HASHES = {
    "publication syn_price_path v1 rows": (
        "9bacbe4ead25dfe342859beb163a7a480e90e05a36629c483b77c64f4e3497ec"
    ),
    "publication syn_equal_weight v1 rows": (
        "66d329e3fa71f253397058c979c239602ff76162c69d3c207059fd323ef8be2e"
    ),
    "synthetic_funds v2 raw payload": (
        "c163637b20b231fe0bd06610673247359d8b929ccff62497b04f6ffadbd0e78c"
    ),
}

Probe = tuple[str, bool, str]  # (what was expected, held, what happened)

# What a broken contract can raise; a probe reports these instead of crashing
# the health run.
PROBE_ERRORS = (
    PublicationRefused,
    FundDataRefused,
    OSError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    AttributeError,
    RuntimeError,
)


def _refusal(action: Callable[[], object], expected: str) -> tuple[bool, str]:
    try:
        action()
    except (PublicationRefused, FundDataRefused) as error:
        return error.reason.value == expected, f"refused: {error.reason.value}"
    return False, "accepted"


def _guarded(probe: Callable[[], tuple[bool, str]]) -> tuple[bool, str]:
    try:
        return probe()
    except PROBE_ERRORS as error:
        return False, f"raised {type(error).__name__}: {error}"


def publication_probes(root: Path) -> list[Probe]:
    draft = synthetic.draft(PRICE_PATH)

    def write(
        spec_draft, *, into: str, code=FIXTURE_CODE, as_of=AS_OF, created_at=CREATED_AT
    ) -> Path:
        directory = root / into
        directory.mkdir()
        return write_publication(
            directory, spec_draft, as_of=as_of, created_at=created_at, code_state=code
        )

    def round_trip() -> tuple[bool, str]:
        rows = read_publication(write(draft, into="round_trip")).rows
        same = table_sha256(rows) == table_sha256(draft.rows)
        return same, f"read {len(rows)} rows, {'identical' if same else 'different'}"

    def edited() -> tuple[bool, str]:
        directory = write(draft, into="edited")
        envelope = directory / "envelope.json"
        envelope.write_text(envelope.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return _refusal(lambda: read_publication(directory), "hash_mismatch")

    def lockbox() -> tuple[bool, str]:
        late = pd.Timestamp("2025-02-01", tz="UTC")
        return _refusal(
            lambda: write(
                synthetic.draft(LOCKBOX_PATH),
                into="lockbox",
                as_of=late,
                created_at=late + pd.Timedelta(hours=1),
            ),
            "lockbox_violation",
        )

    def dirty() -> tuple[bool, str]:
        path = write(draft, into="dirty", code=CodeState(FIXTURE_CODE.sha, True))
        return _refusal(lambda: read_publication(path), "dirty_tree")

    return [
        ("a written publication reads back identical", *_guarded(round_trip)),
        ("an edited envelope is refused (hash_mismatch)", *_guarded(edited)),
        ("rows in the lockbox are refused (lockbox_violation)", *_guarded(lockbox)),
        ("a dirty-tree publication is refused by default (dirty_tree)", *_guarded(dirty)),
    ]


def fund_data_probes(root: Path) -> list[Probe]:
    adapter = SyntheticFundAdapter(retrieved_at=FUND_RETRIEVED_AT)
    store_root = root / "store"
    store_root.mkdir()

    def fetch():
        return adapter.fetch(list(SYNTHETIC_FUND_IDS), FUND_START, FUND_END)

    def round_trip() -> tuple[bool, str]:
        result = ingest_batch(store_root, fetch())
        store = read_store(store_root)
        expected = FUND_SESSIONS * len(SYNTHETIC_FUND_IDS)
        version = store.batches["schema_version"].iloc[0]
        held = (
            len(store.prices) == expected
            and len(store.missing_sessions) == 0
            and version == FUND_SCHEMA_VERSION
            and result.record.batch_id == store.batches["batch_id"].iloc[0]
        )
        return held, (
            f"{len(store.prices)} prices (expected {expected}), "
            f"{len(store.missing_sessions)} missing sessions, schema {version}"
        )

    def again() -> tuple[bool, str]:
        return _refusal(lambda: ingest_batch(store_root, fetch()), "already_exists")

    def cap() -> tuple[bool, str]:
        late = pd.Timestamp("2025-02-01", tz="UTC")
        return _refusal(lambda: adapter.fetch(["SYN_EQ"], FUND_START, late), "request_after_cap")

    def od9() -> tuple[bool, str]:
        other = root / "od9"
        other.mkdir()
        return _refusal(
            lambda: ingest_batch(other, fetch(), assumed_availability_rule="assumed"),
            "pending_od9",
        )

    def edited() -> tuple[bool, str]:
        edited_root = root / "edited"
        edited_root.mkdir()
        table = ingest_batch(edited_root, fetch()).path / "prices.parquet"
        table.write_bytes(table.read_bytes() + b"\0")
        return _refusal(lambda: read_store(edited_root), "hash_mismatch")

    return [
        ("an ingested synthetic batch reads back complete", *_guarded(round_trip)),
        ("the same batch twice is refused (already_exists)", *_guarded(again)),
        ("a request past the lockbox is refused (request_after_cap)", *_guarded(cap)),
        ("an availability rule is refused before OD9 (pending_od9)", *_guarded(od9)),
        ("an edited table is refused (hash_mismatch)", *_guarded(edited)),
    ]


def generator_hashes() -> dict[str, str]:
    """The current SHA-256 of each recorded generator output."""

    raw = SyntheticFundAdapter(retrieved_at=FUND_RETRIEVED_AT).fetch(
        list(SYNTHETIC_FUND_IDS), FUND_START, FUND_END
    )
    return {
        "publication syn_price_path v1 rows": table_sha256(synthetic.draft(PRICE_PATH).rows),
        "publication syn_equal_weight v1 rows": table_sha256(synthetic.draft(EQUAL_WEIGHT).rows),
        "synthetic_funds v2 raw payload": hashlib.sha256(raw.raw_payload).hexdigest(),
    }


def generator_probes() -> list[Probe]:
    def run() -> tuple[dict[str, str] | None, str]:
        try:
            return generator_hashes(), ""
        except PROBE_ERRORS as error:
            return None, f"raised {type(error).__name__}: {error}"

    first, error = run()
    second, _ = run()
    if first is None or second is None:
        return [(name, False, error) for name in RECORDED_HASHES]
    probes = []
    for name, recorded in RECORDED_HASHES.items():
        held = first[name] == second[name] == recorded
        if first[name] != second[name]:
            happened = "two runs differ (nondeterministic)"
        else:
            happened = (
                "matches" if held else f"now {first[name][:16]}..., recorded {recorded[:16]}..."
            )
        probes.append((f"{name} reproduces its recorded hash", held, happened))
    return probes


_CONTRACTS = {
    "contract.publication": (
        "Publication contract",
        "the publication contract (write-once writer, fail-closed reader)",
        "fixture:stock_agent.publication.synthetic syn_price_path v1",
    ),
    "contract.fund_data": (
        "Fund-data contract (F1)",
        "the F1 fund-data contract (ingest and the fail-closed store reader)",
        "fixture:stock_agent.data.funds.adapter.SyntheticFundAdapter v2",
    ),
    "contract.generators": (
        "Synthetic generators",
        "the synthetic generators",
        "module:stock_agent.health.contracts.RECORDED_HASHES",
    ),
}


def _contract_check(check_id: str, probes: list[Probe], measured_at: datetime) -> Check:
    title, subject, evidence = _CONTRACTS[check_id]
    held = sum(ok for _, ok, _ in probes)
    broken = [expected for expected, ok, _ in probes if not ok]
    detail = "; ".join(f"[{'ok' if ok else 'BROKEN'}] {e}: {h}" for e, ok, h in probes)
    if broken:
        state, reason = HealthState.FAILED, "contract_broken"
        summary = f"{len(broken)} of {len(probes)} probes of {subject} did not behave as required."
    else:
        state, reason = HealthState.HEALTHY, None
        summary = f"All {len(probes)} probes of {subject} behaved as required."
    return Check(
        check_id=check_id,
        area=Area.CONTRACT,
        title=title,
        state=state,
        reason=reason,
        measured_at=measured_at,
        value=held,
        unit=f"of {len(probes)} probes held",
        threshold="every probe holds",
        better=Better.HIGHER,
        tolerance=0,
        summary=summary,
        detail=detail,
        evidence=evidence,
    )


def contract_checks(*, now: Callable[[], datetime]) -> list[Check]:
    """The three contract checks, each probed in its own temporary directory."""

    checks = []
    for check_id, run in (
        ("contract.publication", publication_probes),
        ("contract.fund_data", fund_data_probes),
    ):
        with tempfile.TemporaryDirectory(prefix="health-contract-") as directory:
            probes = run(Path(directory))
        checks.append(_contract_check(check_id, probes, now()))
    checks.append(_contract_check("contract.generators", generator_probes(), now()))
    return checks
