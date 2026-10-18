"""
Offline tests of Tiingo prospective mode and `scripts/collect_prospective.py`.
Responses are invented fixtures shaped like Tiingo's JSON; the key is a
sentinel and every amount is invented. No test contacts Tiingo.
"""

import dataclasses
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pandas as pd
import pytest
from prospective_helpers import AFTER_H_END, LATER, day

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.adapter import SyntheticFundAdapter
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.reader import read_store as read_research_store
from stock_agent.data.funds.schema import FundDataRefused, RefusalReason
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.data.prospective.store import ingest_prospective, prepare_store
from stock_agent.data.prospective.validation import MARKER_FILE
from stock_agent.data.prospective.window import REQUEST_CUTOFF_RULE, ProspectiveRefusal
from stock_agent.data.vendors import tiingo
from stock_agent.data.vendors.anchors import COLUMNS as ANCHOR_COLUMNS
from stock_agent.data.vendors.anchors import AnchorScope, DistributionAnchors, parse_anchors
from stock_agent.data.vendors.tiingo import TiingoAdapter, TiingoRefusal, parse_payload
from stock_agent.data.vendors.tiingo_prospective import (
    H_END,
    LOOKBACK_SESSIONS,
    WINDOW_START,
    collect_prospective,
    parse_prospective_payload,
    verified_prospective_batch,
    verify_prospective_basis,
)

TOKEN = "sentinel-tiingo-key-prospective-0123456789"  # invented; never a real key
FUNDS = ("VTI", "AGG", "SHV")
# Invented ex-dates: AGG and SHV monthly, VTI quarterly (one ex-date in the window).
EX_DATES = {
    "AGG": ["2026-08-03", "2026-09-01", "2026-10-01"],
    "SHV": ["2026-08-03", "2026-09-01", "2026-10-01"],
    "VTI": ["2026-09-28"],
}
AMOUNT = {"AGG": 0.1111, "SHV": 0.2222, "VTI": 0.3333}  # invented
SRC = Path(__file__).resolve().parents[1]


def element(session: pd.Timestamp, close: float, **fields) -> dict:
    row = {
        "date": f"{session:%Y-%m-%d}T00:00:00.000Z",
        "close": close,
        "divCash": 0.0,
        "splitFactor": 1.0,
    }
    row.update(fields)
    return row


def served(fund, start=WINDOW_START, end=H_END, *, skip=(), extra=()) -> list[dict]:
    rows = []
    for index, session in enumerate(nyse_rules.sessions(start, end)):
        text = f"{session:%Y-%m-%d}"
        if text in skip:
            continue
        paid = AMOUNT[fund] if text in EX_DATES.get(fund, ()) else 0.0
        rows.append(element(session, 70.0 + index % 7, divCash=paid))
    return rows + list(extra)


class FakeTransport:
    def __init__(self, prices=None, status=None):
        self.prices = prices or {}
        self.status = status or {}
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        ticker = url.removeprefix(tiingo.BASE_URL + "/").split("/")[0]
        if self.status.get(ticker, 200) != 200:
            return self.status[ticker], b""
        rows = self.prices[ticker] if ticker in self.prices else served(ticker)
        return 200, json.dumps(rows).encode("utf-8")


def adapter(transport, clock=AFTER_H_END, token_calls=None) -> TiingoAdapter:
    def token():
        if token_calls is not None:
            token_calls.append(1)
        return TOKEN

    return TiingoAdapter(token_provider=token, transport=transport, clock=lambda: clock)


def collect(transport=None, funds=FUNDS, **kwargs):
    transport = transport if transport is not None else FakeTransport()
    return collect_prospective(list(funds), adapter=adapter(transport), **kwargs)


def anchor_rows(funds=("AGG", "SHV"), per_fund=2) -> dict:
    return {(fund, date): AMOUNT[fund] for fund in funds for date in EX_DATES[fund][:per_fund]}


def anchor_csv(entries: dict) -> bytes:
    lines = [",".join(ANCHOR_COLUMNS)]
    for (fund, ex_date), amount in sorted(entries.items()):
        text = repr(amount)
        lines.append(
            f"{fund},{ex_date},{text},{len(text.partition('.')[2])},per_share_as_paid,anchor,"
            "Fixture publisher,https://example.invalid/fixture,2026-10-14,invented"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def prospective_anchors(entries: dict) -> DistributionAnchors:
    return parse_anchors(anchor_csv(entries), AnchorScope.PROSPECTIVE, today=AFTER_H_END)


def refused(reason, call, *args, **kwargs) -> FundDataRefused:
    with pytest.raises(FundDataRefused) as caught:
        call(*args, **kwargs)
    assert caught.value.reason == reason, caught.value
    return caught.value


# --- the request -----------------------------------------------------------------


def test_the_module_constants_follow_the_owner_decisions():
    assert H_END == pd.Timestamp("2026-10-14", tz="UTC")
    assert LOOKBACK_SESSIONS == 63
    assert WINDOW_START == nyse_rules.session_offset(H_END, -63) == day("2026-07-16")


def test_one_call_per_fund_for_the_lookback_and_later_sessions():
    transport = FakeTransport()
    retrieval = collect(transport)
    assert [url for url, _ in transport.calls] == [
        f"{tiingo.BASE_URL}/{fund}/prices?startDate=2026-07-16&endDate=2026-10-14"
        "&format=json&resampleFreq=daily&columns=date,close,divCash,splitFactor"
        for fund in FUNDS
    ]
    assert all(headers == {"Authorization": f"Token {TOKEN}"} for _, headers in transport.calls)
    assert (retrieval.start, retrieval.end) == (WINDOW_START, H_END)
    assert retrieval.retrieved_at == AFTER_H_END
    assert len(retrieval.prices) == 3 * 64
    assert set(retrieval.prices["adjustment_basis"]) == {"unadjusted"}
    assert retrieval.prices["volume"].isna().all()
    assert len(retrieval.distributions) == 7 and retrieval.splits.empty


def test_before_18_00_new_york_the_request_ends_at_the_previous_session():
    transport = FakeTransport(prices={"AGG": served("AGG", end=day("2026-10-13"))})
    early = pd.Timestamp("2026-10-14T21:59:00Z")  # 17:59 EDT
    retrieval = collect_prospective(
        ["AGG"],
        adapter=adapter(transport, clock=early),
    )
    assert retrieval.end == day("2026-10-13")
    assert "endDate=2026-10-13" in transport.calls[0][0]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"start": day("2026-07-15")}, ProspectiveRefusal.SEALED_HOLDOUT),
        ({"start": day("2025-03-03")}, ProspectiveRefusal.SEALED_HOLDOUT),
        ({"start": day("2024-12-02")}, ProspectiveRefusal.BEFORE_WINDOW),
        ({"end": day("2026-10-15")}, ProspectiveRefusal.INCOMPLETE_SESSION),
        ({"funds": [f"F{i}" for i in range(21)]}, RefusalReason.INVALID_REQUEST),
        ({"funds": ["vti"]}, RefusalReason.INVALID_REQUEST),
        ({"funds": ["VTI", "VTI"]}, RefusalReason.INVALID_REQUEST),
    ],
)
def test_out_of_window_or_malformed_requests_contact_nothing(kwargs, reason):
    transport, token_calls = FakeTransport(), []
    funds = kwargs.pop("funds", list(FUNDS))
    with pytest.raises(FundDataRefused) as caught:
        collect_prospective(funds, adapter=adapter(transport, token_calls=token_calls), **kwargs)
    assert caught.value.reason == reason
    assert transport.calls == [] and token_calls == []  # no key read, no request


@pytest.mark.parametrize(
    ("date", "reason"),
    [
        ("2026-07-15", ProspectiveRefusal.SEALED_HOLDOUT),
        ("2025-01-02", ProspectiveRefusal.SEALED_HOLDOUT),
        ("2024-12-31", ProspectiveRefusal.BEFORE_WINDOW),
        ("2026-10-15", RefusalReason.OUTSIDE_REQUEST),
    ],
)
def test_a_served_row_outside_the_window_refuses_the_batch(date, reason):
    rows = served("AGG", extra=[element(day(date), 70.0)])
    transport = FakeTransport(prices={"AGG": rows})
    refused(reason, collect, transport, funds=("AGG", "SHV"))
    assert len(transport.calls) == 1  # stopped before the next fund


def test_the_research_field_and_split_checks_apply_unchanged():
    rows = served("AGG")
    rows[3] = {**rows[3], "adjClose": 12345.678}
    refused(TiingoRefusal.NON_ALLOWLISTED_FIELD, collect, FakeTransport(prices={"AGG": rows}))
    rows = served("AGG")
    rows[5] = {**rows[5], "splitFactor": 2.0}
    refused(TiingoRefusal.UNVERIFIED_SPLIT_FACTOR, collect, FakeTransport(prices={"AGG": rows}))


# --- the payload -----------------------------------------------------------------


def test_the_envelope_names_the_store_and_neither_parser_accepts_the_other():
    retrieval = collect()
    envelope = json.loads(retrieval.raw_payload)
    assert set(envelope) == {
        "envelope_version",
        "source_id",
        "store",
        "h_end",
        "window_start",
        "start_date",
        "end_date",
        "fields",
        "prices",
    }
    assert envelope["envelope_version"] == "p1" and envelope["store"] == "prospective"
    assert (envelope["h_end"], envelope["window_start"]) == ("2026-10-14", "2026-07-16")
    assert TOKEN.encode() not in retrieval.raw_payload
    parsed = parse_prospective_payload(retrieval.raw_payload)
    assert parsed.prices.equals(retrieval.prices)
    refused(RefusalReason.SCHEMA_MISMATCH, parse_payload, retrieval.raw_payload)
    research = dict(envelope)
    for key in ("store", "h_end", "window_start"):
        del research[key]
    research["envelope_version"] = tiingo.ENVELOPE_VERSION
    refused(
        RefusalReason.SCHEMA_MISMATCH,
        parse_prospective_payload,
        json.dumps(research).encode(),
    )


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"start_date": "2026-07-15"}, ProspectiveRefusal.SEALED_HOLDOUT),
        ({"h_end": "2026-10-15"}, RefusalReason.INVALID_FIELD),
        ({"window_start": "2026-07-15"}, RefusalReason.INVALID_FIELD),
        ({"envelope_version": "4"}, RefusalReason.INVALID_FIELD),
    ],
)
def test_a_stored_payload_is_rechecked_on_re_parse(change, reason):
    envelope = {**json.loads(collect(funds=("AGG",)).raw_payload), **change}
    refused(reason, parse_prospective_payload, json.dumps(envelope).encode())


def test_tampered_in_memory_tables_are_refused():
    retrieval = collect(funds=("AGG", "SHV"))
    anchors = prospective_anchors(anchor_rows())
    changed = retrieval.prices.copy()
    changed.loc[changed.index[0], "close"] = 1.0
    tampered = dataclasses.replace(retrieval, prices=changed)
    for call in (verified_prospective_batch, verify_prospective_basis):
        refused(TiingoRefusal.PAYLOAD_MISMATCH, call, tampered, anchors=anchors)
    shifted = dataclasses.replace(retrieval, end=day("2026-10-13"))
    refused(TiingoRefusal.PAYLOAD_MISMATCH, verified_prospective_batch, shifted, anchors=anchors)


# --- the distribution-basis gate (unchanged rules, prospective anchors) ------------


def test_without_anchors_nothing_can_be_stored():
    retrieval = collect()
    empty = DistributionAnchors.empty(AnchorScope.PROSPECTIVE)
    checks = verify_prospective_basis(retrieval, anchors=empty)
    assert [check.passed for check in checks] == [False, False, False]
    refused(
        TiingoRefusal.UNVERIFIED_DISTRIBUTION_BASIS,
        verified_prospective_batch,
        retrieval,
        anchors=empty,
    )


def test_research_scope_anchors_are_refused():
    retrieval = collect()
    research = DistributionAnchors.empty(AnchorScope.RESEARCH)
    for call in (verified_prospective_batch, verify_prospective_basis):
        with pytest.raises(ValueError):
            call(retrieval, anchors=research)


def test_two_matching_anchors_per_fund_pass_and_one_quarterly_ex_date_cannot():
    """
    The gate is unchanged (at least 2 matching anchors in the window). VTI pays
    quarterly, so the 64-session window holds one VTI ex-date: VTI cannot pass
    until a collection window spans two of them, or the owner changes the gate.
    """

    anchors = prospective_anchors(anchor_rows(("AGG", "SHV", "VTI"), per_fund=2))
    checks = {
        check.fund_id: check for check in verify_prospective_basis(collect(), anchors=anchors)
    }
    assert checks["AGG"].passed and checks["SHV"].passed
    assert checks["AGG"].route == "primary_source_spot_check"
    assert not checks["VTI"].passed
    assert "verified distributions in the window: 1 (needs 2), 1 match, 0 do not" in (
        checks["VTI"].detail
    )


def test_a_mismatched_anchor_fails_its_fund():
    entries = anchor_rows() | {("AGG", "2026-10-01"): 0.1112}
    checks = verify_prospective_basis(collect(), anchors=prospective_anchors(entries))
    agg = next(check for check in checks if check.fund_id == "AGG")
    assert not agg.passed and "3 (needs 2), 2 match, 1 do not" in agg.detail


def test_a_verified_batch_reaches_the_inference_loader_unchanged(tmp_path):
    raw_csv = anchor_csv(anchor_rows())
    anchors = parse_anchors(raw_csv, AnchorScope.PROSPECTIVE, today=AFTER_H_END)
    retrieval = collect(funds=("AGG", "SHV"))
    item = verified_prospective_batch(retrieval, anchors=anchors)
    digest = hashlib.sha256(raw_csv).hexdigest()
    assert item.anchors_sha256 == digest
    assert f"distribution anchors file sha256 {digest} (prospective scope)" in item.vintage_note
    assert REQUEST_CUTOFF_RULE in item.vintage_note
    assert "0.1111" not in item.vintage_note
    root = tmp_path / "store"
    assert prepare_store(root) is None
    result = ingest_prospective(root, item)
    window = load_inference_window(root, ("AGG", "SHV"), H_END, retrieved_by=LATER)
    assert window.complete and list(window.batches["batch_id"]) == [result.record.batch_id]
    expected = retrieval.prices.sort_values(["fund_id", "session_date"]).reset_index(drop=True)
    assert window.prices["close"].tolist() == expected["close"].tolist()
    amounts = window.distributions.groupby("fund_id")["amount"].unique()
    assert amounts["AGG"].tolist() == [0.1111] and amounts["SHV"].tolist() == [0.2222]


# --- the collection script (offline; the adapter runs over the fake transport) ------

SCRIPT = SRC / "scripts" / "collect_prospective.py"
_spec = importlib.util.spec_from_file_location("stock_agent_collect_prospective", SCRIPT)
script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script
_spec.loader.exec_module(script)


def never():
    raise AssertionError("the adapter must not be built")


@pytest.fixture
def repo(tmp_path) -> Path:
    path = tmp_path / "repo"
    (path / "data" / "raw").mkdir(parents=True)
    return path


def no_git(repository, path):
    raise AssertionError("git must not run for paths outside every repository")


def run(repo, transport=None, *flags, anchors=None, clock=AFTER_H_END, factory=None, git=no_git):
    anchors = repo.parent / "no-anchors.csv" if anchors is None else anchors
    argv = [*flags, "--authorized", "--anchors", str(anchors)]
    factory = factory or (lambda: adapter(transport or FakeTransport()))
    return script.main(
        argv, adapter_factory=factory, project_root=repo, clock=lambda: clock, git_runner=git
    )


def write_anchors(tmp_path, entries) -> Path:
    path = tmp_path / "anchors.csv"
    path.write_bytes(anchor_csv(entries))
    return path


def snapshot(path: Path) -> dict[Path, int]:
    return {item: item.stat().st_mtime_ns for item in [path, *path.rglob("*")]}


def test_the_script_refuses_without_authorization(tmp_path, repo, capsys):
    code = script.main(["--funds", "AGG"], adapter_factory=never, project_root=repo)
    assert code == 2 and "--authorized" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--store"], ["--root", "x"]])
def test_store_and_root_go_together(tmp_path, repo, capsys, flags):
    assert run(repo, None, *flags, factory=never) == 2
    assert "--store and --root go together" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "reason"),
    [
        (["--start", "2026-07-15"], "sealed_holdout"),
        (["--start", "2024-06-03"], "before_window"),
        (["--end", "2026-10-15"], "incomplete_session"),
        (["--start", "2026-7-16"], "invalid_date"),
        (["--funds", "AGG", "AGG"], "invalid_request"),
        (["--funds", "agg"], "invalid_request"),
    ],
)
def test_window_refusals_happen_before_any_request(tmp_path, repo, capsys, flags, reason):
    before = snapshot(tmp_path)
    assert run(repo, None, *flags, factory=never) == 2
    assert f"Refused: {reason}" in capsys.readouterr().err
    assert snapshot(tmp_path) == before


def test_an_anchors_file_among_the_research_rows_is_refused(tmp_path, repo, capsys):
    gathered = tmp_path / "issuer_sources" / "2026-10-14" / "rows.csv"
    assert run(repo, None, anchors=gathered, factory=never) == 2
    assert "research_rows" in capsys.readouterr().err


def test_the_default_run_is_a_read_only_report(tmp_path, repo, capsys, monkeypatch):
    anchors = write_anchors(tmp_path, anchor_rows())
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    for name in ("prepare_store", "ingest_prospective", "read_batch", "verified_prospective_batch"):
        monkeypatch.setattr(script, name, lambda *a, **k: pytest.fail("a read-only run stored"))
    before = snapshot(tmp_path)
    transport = FakeTransport()
    code = run(repo, transport, anchors=anchors)
    out = capsys.readouterr().out
    assert code == 0 and len(transport.calls) == 3
    lines = out.splitlines()
    assert lines[1] == (
        "H_end 2026-10-14; lookback 63 sessions; window start 2026-07-16; calendar "
        "nyse_full_day_rules_2003_2030_v1"
    )
    assert "request: VTI AGG SHV [2026-07-16, 2026-10-14] (64 sessions)" in lines[2]
    assert f"  sha256 {hashlib.sha256(anchors.read_bytes()).hexdigest()}" in lines
    assert "  prospective-scope anchors per fund: AGG 2, SHV 2" in lines
    agg = lines[lines.index("AGG") + 1 : lines.index("SHV")]
    assert "  price rows (unadjusted): 64" in agg
    assert "  missing NYSE sessions: 0 (none)" in agg
    assert "  distributions served: 3 (ex-dates 2026-08-03, 2026-09-01, 2026-10-01)" in agg
    assert any(line.startswith(f"  distribution basis: {script.SPOT_CHECKED}") for line in agg)
    vti = lines[lines.index("VTI") + 1 : lines.index("AGG")]
    assert "  distributions served: 1 (ex-dates 2026-09-28)" in vti
    assert any(line.startswith("  distribution basis: NOT verified") for line in vti)
    assert out.rstrip().endswith(script.READ_ONLY)
    for secret in (TOKEN, "70.0", "0.1111", "0.2222", "0.3333"):
        assert secret not in out
    assert snapshot(tmp_path) == before


def test_a_verified_collection_is_stored_and_read_back(tmp_path, repo, capsys):
    anchors = write_anchors(tmp_path, anchor_rows())
    root = tmp_path / "private" / "prospective"
    code = run(repo, None, "--funds", "AGG", "SHV", "--store", "--root", str(root), anchors=anchors)
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "stored tables match a re-parse of the stored raw payload: True" in captured.out
    assert stat.S_IMODE(root.stat().st_mode) == 0o700 and (root / MARKER_FILE).is_file()
    window = load_inference_window(root, ("AGG", "SHV"), H_END, retrieved_by=LATER)
    assert window.complete and len(window.prices) == 128
    assert TOKEN.encode() not in b"".join(
        path.read_bytes() for path in root.rglob("*") if path.is_file()
    )
    assert set(repo.rglob("*")) == {repo / "data", repo / "data" / "raw"}


def test_one_unverified_fund_means_nothing_is_stored(tmp_path, repo, capsys):
    anchors = write_anchors(tmp_path, anchor_rows(("AGG", "SHV", "VTI")))
    root = tmp_path / "private"
    code = run(repo, None, "--store", "--root", str(root), anchors=anchors)
    captured = capsys.readouterr()
    assert code == 1 and "NOT STORED unverified_distribution_basis: VTI;" in captured.err
    assert not root.exists()  # the root is created only for a verified batch


def test_a_failed_read_back_exits_with_an_error(tmp_path, repo, capsys, monkeypatch):
    anchors = write_anchors(tmp_path, anchor_rows())
    monkeypatch.setattr(script, "reparse_matches", lambda batch: False)
    root = tmp_path / "private"
    code = run(repo, None, "--funds", "AGG", "--store", "--root", str(root), anchors=anchors)
    assert code == 1 and "FAILED the stored tables differ" in capsys.readouterr().err


def _research_store(path: Path) -> Path:
    path.mkdir(parents=True)
    os.chmod(path, 0o700)
    raw = SyntheticFundAdapter(retrieved_at=pd.Timestamp("2026-10-14T00:00:00Z")).fetch(
        ["SYN_EQ"], day("2024-12-02"), day("2025-01-01")
    )
    ingest_batch(path, raw)
    return path


def test_the_research_store_is_never_a_prospective_root(tmp_path, repo, capsys):
    research = _research_store(tmp_path / "private" / "research")
    os.chmod(tmp_path / "private", 0o700)
    before = snapshot(tmp_path / "private")
    for root in (research, research / "prospective", research / "a" / "b"):
        assert run(repo, None, "--store", "--root", str(root), factory=never) == 2
        assert "is, or lies inside, the research store" in capsys.readouterr().err
    # A root containing the research store is neither empty nor a prospective store.
    assert run(repo, None, "--store", "--root", str(tmp_path / "private"), factory=never) == 2
    assert "never the research store" in capsys.readouterr().err
    assert snapshot(tmp_path / "private") == before
    assert len(read_research_store(research).batches) == 1  # F1 still reads it


def test_roots_inside_a_repository_or_a_prospective_store_are_refused(tmp_path, repo, capsys):
    for root in (repo / "store", repo / "data"):
        assert run(repo, None, "--store", "--root", str(root), factory=never) == 2
        assert "inside the repository" in capsys.readouterr().err
    store = tmp_path / "store"
    assert prepare_store(store) is None
    assert run(repo, None, "--store", "--root", str(store / "inner"), factory=never) == 2
    assert "lies inside a prospective store" in capsys.readouterr().err
    allowed = repo / "data" / "raw" / "prospective"
    flags = ("--funds", "AGG", "--store", "--root", str(allowed))
    assert run(repo, None, *flags, factory=never, git=lambda r, p: 1) == 2  # not ignored
    assert "does not confirm it is ignored" in capsys.readouterr().err
    assert not allowed.exists()
    anchors = write_anchors(tmp_path, anchor_rows())
    code = run(repo, None, *flags, anchors=anchors, git=lambda r, p: 0)
    assert code == 0, capsys.readouterr().err


def test_an_existing_root_open_to_others_is_refused_and_never_changed(tmp_path, repo, capsys):
    root = tmp_path / "open"
    root.mkdir()
    os.chmod(root, 0o755)
    assert run(repo, None, "--store", "--root", str(root), factory=never) == 2
    assert "restrict it yourself" in capsys.readouterr().err
    assert stat.S_IMODE(root.stat().st_mode) == 0o755 and not any(root.iterdir())


def test_a_refused_retrieval_reports_its_reason_and_stores_nothing(tmp_path, repo, capsys):
    root = tmp_path / "private"
    transport = FakeTransport(status={"VTI": 429})
    code = run(repo, transport, "--store", "--root", str(root))
    captured = capsys.readouterr()
    assert code == 1 and "REFUSED rate_limited; ticker VTI; HTTP 429" in captured.err
    assert TOKEN not in captured.out + captured.err
    assert not root.exists()  # nothing created for a refused retrieval


def test_a_refused_write_removes_the_root_it_created_but_never_an_existing_store(
    tmp_path, repo, capsys, monkeypatch
):
    anchors = write_anchors(tmp_path, anchor_rows())

    def refuse_write(root, batch):
        raise FundDataRefused(RefusalReason.ALREADY_EXISTS, "simulated refusal")

    monkeypatch.setattr(script, "ingest_prospective", refuse_write)
    fresh = tmp_path / "fresh" / "store"
    code = run(repo, None, "--funds", "AGG", "--store", "--root", str(fresh), anchors=anchors)
    assert code == 1 and "simulated refusal" in capsys.readouterr().err
    assert not fresh.exists()
    existing = tmp_path / "existing"
    assert prepare_store(existing) is None
    code = run(repo, None, "--funds", "AGG", "--store", "--root", str(existing), anchors=anchors)
    assert code == 1 and [entry.name for entry in existing.iterdir()] == [MARKER_FILE]
    capsys.readouterr()


def test_new_parent_directories_are_private_and_existing_ones_unchanged(tmp_path, repo, capsys):
    anchors = write_anchors(tmp_path, anchor_rows())
    base = tmp_path / "home"
    base.mkdir()
    os.chmod(base, 0o755)
    root = base / "stock-agent-private" / "nested" / "prospective"
    code = run(repo, None, "--funds", "AGG", "--store", "--root", str(root), anchors=anchors)
    assert code == 0, capsys.readouterr().err
    for directory in (root, root.parent, root.parent.parent):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory
    assert stat.S_IMODE(base.stat().st_mode) == 0o755  # existing: never changed
