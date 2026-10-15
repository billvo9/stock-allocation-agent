"""
Collect real ETF data from Tiingo into the private prospective store.

Prepared, not run by any agent: it contacts Tiingo, so the owner runs it in
the owner's terminal, with `TIINGO_API_KEY` configured locally, passing
`--authorized` (owner decision of 2026-10-14, "Audit + prospective").

Read-only by default: without `--store` it writes nothing anywhere. Storing
needs the owner's OK at that step (`--store --root`).

0. Before any request: the window and the anchors.
   - The request is [--start, --end], both inclusive. --start defaults to
     WINDOW_START (2026-07-16, the session 63 sessions before H_end
     2026-10-14); an earlier start is refused (`sealed_holdout`, or
     `before_window` before 2025-01-01). --end defaults to the last complete
     session under the 18:00 America/New_York convention (OD9, descriptive,
     assumed; not proof of vendor publication time); a later end is refused
     (`incomplete_session`).
   - The owner's private anchors, prospective scope (ex-dates on or after
     WINDOW_START), from `--anchors`, `STOCK_AGENT_ANCHORS` or the default
     `data/raw/issuer_sources/anchors.csv`. A malformed file, or one inside a
     repository where Git does not ignore it, refuses the run; a missing file
     is an empty allowlist. Its sha256 and counts are printed, never amounts.
1. Retrieve and verify in memory (always): one prices call per fund,
   allowlisted columns only, each response checked before the next request,
   no retry. It prints aggregates only, per fund: row count, first and last
   session, missing NYSE sessions, the ex-dates of served distributions (so
   the owner knows which anchors to fill in), splits, and the
   distribution-basis verification (counts and ratios).
2. Store (only with `--store`): only when every fund passes. The root is
   created with mode 0o700 and the store marker, or must already be this
   prospective store (never the research store) with no group or other
   access; an existing root's mode is never changed. The batch is written
   once, read back through the store's only reader, and its tables compared
   with a re-parse of the stored payload (exit 1 if they differ).

Exit codes: 0 success; 1 a refused retrieval, an unverified fund (nothing
stored) or a failed read-back; 2 refused before any request.

It never prints prices, amounts or the key, requests no adjusted series and
computes no return.

Usage (the owner's commands and what to expect are in
docs/research/prospective_data_path.md):
    PYTHONPATH=src .venv/bin/python scripts/collect_prospective.py --authorized
    PYTHONPATH=src .venv/bin/python scripts/collect_prospective.py --authorized \\
        --store --root ~/stock-agent-private/prospective_store
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.schema import DATA_TABLES, FundDataRefused, conform, parse_date
from stock_agent.data.private_paths import GitRunner, private_root, run_git_check_ignore
from stock_agent.data.prospective.inference import StoredProspectiveBatch, read_batch
from stock_agent.data.prospective.store import ingest_prospective, prepare_store
from stock_agent.data.prospective.validation import MARKER_FILE
from stock_agent.data.prospective.window import (
    H_END,
    LOOKBACK_SESSIONS,
    REQUEST_CUTOFF_RULE,
    WINDOW_START,
    check_request,
)
from stock_agent.data.vendors.anchors import (
    AnchorScope,
    AnchorsRefused,
    DistributionAnchors,
    anchors_path,
    load_anchors,
)
from stock_agent.data.vendors.tiingo import (
    ALLOWED_FIELDS,
    BasisCheck,
    TiingoAdapter,
    TiingoRequestFailed,
    utc_now,
)
from stock_agent.data.vendors.tiingo_prospective import (
    ProspectiveRetrieval,
    check_tickers,
    collect_prospective,
    parse_prospective_payload,
    request_window,
    verified_prospective_batch,
    verify_prospective_basis,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUNDS = ("VTI", "AGG", "SHV")
SPOT_CHECKED = (
    "spot-checked against primary-source amounts (not a validation of every distribution)"
)
READ_ONLY = (
    "NOT STORED read-only collection report: nothing was written; storing needs the owner's "
    "OK at this step (--store --root)"
)
_SHOWN = 10


def _dates(values) -> str:
    texts = [f"{value:%Y-%m-%d}" for value in list(values)[:_SHOWN]]
    more = f" (+{len(values) - _SHOWN} more)" if len(values) > _SHOWN else ""
    return (", ".join(texts) + more) if texts else "none"


def collection_lines(
    retrieval: ProspectiveRetrieval,
    checks: tuple[BasisCheck, ...],
    anchors: DistributionAnchors,
    now: pd.Timestamp,
) -> list[str]:
    """Aggregates of the in-memory retrieval and its basis verification; no prices or amounts."""

    expected = nyse_rules.sessions(retrieval.start, retrieval.end)
    lines = [
        "Tiingo prospective collection (aggregate counts and dates; no prices or amounts)",
        (
            f"H_end {H_END:%Y-%m-%d}; lookback {LOOKBACK_SESSIONS} sessions; window start "
            f"{WINDOW_START:%Y-%m-%d}; calendar {nyse_rules.CALENDAR_ID}"
        ),
        (
            f"request: {' '.join(retrieval.fund_ids)} [{retrieval.start:%Y-%m-%d}, "
            f"{retrieval.end:%Y-%m-%d}] ({len(expected)} sessions), Tiingo endDate "
            f"{retrieval.end:%Y-%m-%d}"
        ),
        f"clock {now}; request rule: {REQUEST_CUTOFF_RULE}",
        f"retrieved_at {retrieval.retrieved_at}",
        f"field allowlist: every element had exactly {', '.join(ALLOWED_FIELDS)}",
        *anchors.summary_lines(),
    ]
    for fund, check in zip(retrieval.fund_ids, checks, strict=True):
        prices = retrieval.prices[retrieval.prices["fund_id"] == fund]
        sessions = prices["session_date"].sort_values()
        missing = expected[~expected.isin(sessions)]
        distributions = retrieval.distributions[retrieval.distributions["fund_id"] == fund]
        splits = retrieval.splits[retrieval.splits["fund_id"] == fund]
        lines.append("")
        lines.append(fund)
        lines.append(f"  price rows (unadjusted): {len(prices)}")
        if len(sessions):
            lines.append(
                f"  sessions: first {sessions.iloc[0]:%Y-%m-%d}, last {sessions.iloc[-1]:%Y-%m-%d}"
            )
        else:
            lines.append("  sessions: none")
        lines.append(f"  missing NYSE sessions: {len(missing)} ({_dates(missing)})")
        lines.append(
            f"  distributions served: {len(distributions)} (ex-dates "
            f"{_dates(distributions['ex_date'].sort_values())})"
        )
        lines.append(f"  splits: {_dates(splits['effective_date'])}")
        verdict = SPOT_CHECKED if check.passed else "NOT verified"
        lines.append(
            f"  distribution basis: {verdict}; split check {check.split_check} ({check.detail})"
        )
    return lines


def reparse_matches(batch: StoredProspectiveBatch) -> bool:
    """Whether the stored tables equal a fresh parse of the stored (hash-checked) payload."""

    parsed = parse_prospective_payload(batch.raw_payload)
    fresh = (parsed.prices, parsed.distributions, parsed.splits)
    stored = (batch.prices, batch.distributions, batch.splits)
    return all(
        conform(spec, frame.assign(batch_id=batch.record.batch_id)).equals(table)
        for spec, frame, table in zip(DATA_TABLES, fresh, stored, strict=True)
    )


def nested_in_prospective_store(root: Path) -> bool:
    """Whether an ancestor of the root (not the root itself) holds the store marker."""

    return any((path / MARKER_FILE).exists() for path in root.parents)


def main(
    argv: list[str] | None = None,
    *,
    adapter_factory: Callable[[], TiingoAdapter] = TiingoAdapter,
    project_root: Path = PROJECT_ROOT,
    clock: Callable[[], pd.Timestamp] = utc_now,
    git_runner: GitRunner = run_git_check_ignore,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", type=Path, help="private prospective store (with --store only)")
    parser.add_argument("--funds", nargs="+", default=list(DEFAULT_FUNDS))
    parser.add_argument("--start", help=f"first session (default {WINDOW_START:%Y-%m-%d})")
    parser.add_argument("--end", help="last session (default: the last complete session)")
    parser.add_argument(
        "--anchors",
        help="the private anchors CSV (default: $STOCK_AGENT_ANCHORS, else "
        "data/raw/issuer_sources/anchors.csv)",
    )
    parser.add_argument(
        "--authorized",
        action="store_true",
        help="confirm the owner configured TIINGO_API_KEY and authorized this network run",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="persist the verified batch under --root; needs the owner's OK at this step",
    )
    args = parser.parse_args(argv)

    if not args.authorized:
        print(
            "Refused: this collection contacts Tiingo. Only the owner runs it, after "
            "configuring TIINGO_API_KEY locally, with --authorized.",
            file=sys.stderr,
        )
        return 2
    if args.store != (args.root is not None):
        print(
            "Refused: --store and --root go together. A report-only run writes nothing; "
            "storing needs --root and the owner's OK at this step.",
            file=sys.stderr,
        )
        return 2
    now = clock()
    try:
        root = private_root(args.root, project_root, git_runner) if args.store else None
        start = None if args.start is None else parse_date("--start", args.start)
        end = None if args.end is None else parse_date("--end", args.end)
        first, last = request_window(now, start, end)
        ids, first, last = check_request(args.funds, first, last, now)
        check_tickers(ids)
    except (ValueError, FundDataRefused) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 2
    if root is not None and nested_in_prospective_store(root):
        print(f"Refused: --root {root} lies inside a prospective store", file=sys.stderr)
        return 2
    try:
        anchors = load_anchors(
            anchors_path(args.anchors, project_root=project_root),
            AnchorScope.PROSPECTIVE,
            project_root=project_root,
            today=now,
            git_runner=git_runner,
        )
    except AnchorsRefused as error:
        print(f"Refused: anchors {error}", file=sys.stderr)
        return 2
    refusal = prepare_store(root) if args.store else None
    if refusal is not None:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 2

    try:
        retrieval = collect_prospective(ids, adapter=adapter_factory(), start=first, end=last)
        checks = verify_prospective_basis(retrieval, anchors=anchors)
    except (FundDataRefused, TiingoRequestFailed) as error:
        print(f"REFUSED {error}", file=sys.stderr)
        return 1
    for line in collection_lines(retrieval, checks, anchors, now):
        print(line)
    print()
    print("total return: not computed by the collection (OD8 group A descriptive calculations)")
    if not args.store:
        print(READ_ONLY)
        return 0
    if not all(check.passed for check in checks):
        failing = " ".join(check.fund_id for check in checks if not check.passed)
        print(
            f"NOT STORED unverified_distribution_basis: {failing}; nothing may be stored "
            "under an unverified basis (collect the verified funds alone, or add anchors)",
            file=sys.stderr,
        )
        return 1
    try:
        result = ingest_prospective(root, verified_prospective_batch(retrieval, anchors=anchors))
        stored = read_batch(result.path)
    except FundDataRefused as error:
        print(f"REFUSED {error}", file=sys.stderr)
        return 1
    matches = reparse_matches(stored)
    print(f"stored batch: {stored.record.batch_id}")
    print(f"stored tables match a re-parse of the stored raw payload: {matches}")
    print(f"stored under: {stored.path}")
    if not matches:
        print(
            "FAILED the stored tables differ from a re-parse of the stored raw payload",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
