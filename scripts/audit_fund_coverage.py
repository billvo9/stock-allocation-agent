"""
Aggregate coverage audit of Tiingo fund data, for private local research.

Prepared, not yet run: it contacts Tiingo, so run it only after the owner has
configured `TIINGO_API_KEY` locally and authorized the run (`--authorized`).

Read-only by default: without `--store` it writes nothing anywhere. Storing
a real fund batch needs the owner's separate approval (`--store --root`).
0. Load the owner's private distribution anchors (research scope: ex-dates
   before 2025-01-01) from `--anchors`, `STOCK_AGENT_ANCHORS` or the default
   `data/raw/issuer_sources/anchors.csv`, before any request. A malformed
   file, or one inside a repository where Git does not ignore it, refuses the
   run (exit 2); a missing file is an empty allowlist. It prints the file's
   sha256, row counts and per-fund anchor counts, never an amount.
1. Verify in memory (always). The Tiingo adapter retrieves the funds (one prices call
   per fund, allowlisted columns only, each response checked before the next
   request, no retry). A success means every element held exactly the
   allowlisted fields and every reported split was verified. The script then
   verifies each fund's distribution basis (primary-source amounts are
   required; the split ratio can only fail a fund) and prints aggregates only:
   - row counts, first and last sessions, and missing NYSE sessions;
   - distributions per year;
   - verified splits, with the raw-close ratio the adapter enforced;
   - each fund's distribution-basis verification (counts and ratios).
2. Store (only with `--store`). Only when every fund passes does it store the batch with F1
   `ingest_batch` under the private root, read it back with the fail-closed
   reader, and check that the stored tables equal a re-parse of the stored
   payload (exit 1 if not). Otherwise nothing is stored and the reason is
   reported.

It never prints prices, amounts or the key, and writes vendor data only with
`--store`, under `--root`, which must lie outside every repository or under that repository's
ignored `data/raw/`, and must not be (or lie inside) a prospective store. A missing root is created with mode 0o700; an existing
root whose mode grants any group or other permission is refused, and the
owner restricts it. It requests no adjusted series and performs no
vendor-adjusted reconciliation (that needs a separate, owner-approved isolated
procedure), and it computes no return (returns belong to the OD8 group A
descriptive calculations, not the audit).

Usage (a read-only coverage report; the owner's command is in
docs/research/prospective_data_path.md):
    PYTHONPATH=src .venv/bin/python scripts/audit_fund_coverage.py --authorized \
        --funds VTI AGG SHV --start 2005-01-01 --end-exclusive 2025-01-01

Storing, only after the owner separately approves persisting a batch:
    PYTHONPATH=src .venv/bin/python scripts/audit_fund_coverage.py --authorized \
        --store --root ~/stock-agent-private/fund_store
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.reader import StoredBatch, read_batch
from stock_agent.data.funds.schema import (
    DATA_TABLES,
    PAYLOAD_FILE,
    FundDataRefused,
    conform,
    parse_date,
)
from stock_agent.data.private_paths import (
    GitRunner,
    prepare_root,
    private_root,
    run_git_check_ignore,
)
from stock_agent.data.prospective.validation import MARKER_FILE
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
    Retrieval,
    TiingoAdapter,
    TiingoRequestFailed,
    parse_payload,
    utc_now,
    verified_batch,
    verify_distribution_basis,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUNDS = ("VTI", "AGG", "SHV")
# A pass is a spot check of sampled amounts, never proof that the history is valid.
SPOT_CHECKED = (
    "spot-checked against primary-source amounts (not a validation of every distribution)"
)
READ_ONLY = (
    "NOT STORED read-only coverage report: nothing was written; storing a batch needs the "
    "owner's separate approval (--store --root)"
)
_SHOWN = 10  # dates listed per finding


def _dates(values: pd.Series) -> str:
    shown = [f"{value:%Y-%m-%d}" for value in values.iloc[:_SHOWN]]
    more = f" (+{len(values) - _SHOWN} more)" if len(values) > _SHOWN else ""
    return (", ".join(shown) + more) if shown else "none"


def _split_line(prices: pd.DataFrame, split) -> str:
    """The split with the raw-close ratio the adapter enforced: about 1 if unadjusted."""

    closes = prices[prices["fund_id"] == split.fund_id].set_index("session_date")["close"]
    previous = nyse.sessions(nyse.COVERAGE.first, split.effective_date - pd.Timedelta(days=1))[-1]
    ratio = closes[split.effective_date] * split.ratio_new_per_old / closes[previous]
    return (
        f"{split.effective_date:%Y-%m-%d} x{split.ratio_new_per_old:g} (verified; raw close x "
        f"factor / previous raw close {ratio:.2f})"
    )


def verification_lines(
    retrieval: Retrieval, checks: tuple[BasisCheck, ...], anchors: DistributionAnchors
) -> list[str]:
    """Phase 1: aggregates of the in-memory retrieval and its basis verification."""

    end_date = retrieval.end_exclusive - pd.Timedelta(days=1)
    window = f"[{retrieval.start:%Y-%m-%d}, {retrieval.end_exclusive:%Y-%m-%d})"
    expected = nyse.sessions(retrieval.start, end_date)
    lines = [
        "Tiingo coverage audit (aggregate counts, dates and ratios; no prices are printed)",
        f"request: {' '.join(retrieval.fund_ids)} {window}, Tiingo endDate {end_date:%Y-%m-%d}",
        f"retrieved_at {retrieval.retrieved_at}",
        f"field allowlist: every element had exactly {', '.join(ALLOWED_FIELDS)}",
        *anchors.summary_lines(),
    ]
    for fund, check in zip(retrieval.fund_ids, checks, strict=True):
        prices = retrieval.prices[retrieval.prices["fund_id"] == fund]
        sessions = prices["session_date"].sort_values()
        missing = pd.Series(expected[~expected.isin(sessions)])
        distributions = retrieval.distributions[retrieval.distributions["fund_id"] == fund]
        splits = retrieval.splits[retrieval.splits["fund_id"] == fund]
        lines.append("")
        lines.append(fund)
        lines.append(f"  price rows (unadjusted): {len(prices)}")
        if len(sessions):
            first, last = sessions.iloc[0], sessions.iloc[-1]
            lines.append(f"  sessions: first {first:%Y-%m-%d}, last {last:%Y-%m-%d}")
            before, after = missing[missing < first], missing[missing > last]
            inside = missing[(missing > first) & (missing < last)]
            lines.append(
                f"  missing NYSE sessions: {len(missing)} (before first {len(before)}, "
                f"inside {len(inside)}: {_dates(inside)}; after last {len(after)})"
            )
        else:
            lines.append(f"  sessions: none; missing NYSE sessions: {len(missing)}")
        per_year = distributions.groupby(distributions["ex_date"].dt.year).size()
        lines.append(
            "  distributions per year: "
            + (" ".join(f"{year}:{count}" for year, count in per_year.items()) or "none")
        )
        lines.append(
            "  splits: "
            + (
                ", ".join(_split_line(retrieval.prices, row) for row in splits.itertuples())
                or "none"
            )
        )
        verdict = SPOT_CHECKED if check.passed else "NOT verified"
        lines.append(
            f"  distribution basis: {verdict}; split check {check.split_check} ({check.detail})"
        )
    return lines


def reparse_matches(batch: StoredBatch) -> bool:
    """Whether the stored tables equal a fresh parse of the stored raw payload."""

    parsed = parse_payload((batch.path / PAYLOAD_FILE).read_bytes())
    fresh = (parsed.prices, parsed.distributions, parsed.splits)
    stored = (batch.prices, batch.distributions, batch.splits)
    return all(
        conform(spec, frame.assign(batch_id=batch.record.batch_id)).equals(table)
        for spec, frame, table in zip(DATA_TABLES, fresh, stored, strict=True)
    )


def inside_prospective_store(root: Path) -> bool:
    """Whether the root, or any of its ancestors, holds the prospective store's marker."""

    return any((path / MARKER_FILE).exists() for path in (root, *root.parents))


def main(
    argv: list[str] | None = None,
    *,
    adapter_factory: Callable[[], TiingoAdapter] = TiingoAdapter,
    project_root: Path = PROJECT_ROOT,
    clock: Callable[[], pd.Timestamp] = utc_now,
    git_runner: GitRunner = run_git_check_ignore,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", type=Path, help="private store directory (with --store only)")
    parser.add_argument("--funds", nargs="+", default=list(DEFAULT_FUNDS))
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end-exclusive", default=f"{MODEL_HOLDOUT_START:%Y-%m-%d}")
    parser.add_argument(
        "--authorized",
        action="store_true",
        help="confirm the owner configured TIINGO_API_KEY and authorized this network run",
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="persist the verified batch under --root; needs the owner's separate approval",
    )
    parser.add_argument(
        "--anchors",
        help="the private anchors CSV (default: $STOCK_AGENT_ANCHORS, else "
        "data/raw/issuer_sources/anchors.csv)",
    )
    args = parser.parse_args(argv)

    if not args.authorized:
        print(
            "Refused: this audit contacts Tiingo. Run it only after the owner configures "
            "TIINGO_API_KEY locally and authorizes the run, then pass --authorized.",
            file=sys.stderr,
        )
        return 2
    if args.store != (args.root is not None):
        print(
            "Refused: --store and --root go together. A report-only run writes nothing; "
            "storing needs --root and the owner's separate approval.",
            file=sys.stderr,
        )
        return 2
    try:
        root = private_root(args.root, project_root, git_runner) if args.store else None
        start = parse_date("--start", args.start)
        end_exclusive = parse_date("--end-exclusive", args.end_exclusive)
    except (ValueError, FundDataRefused) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 2
    if root is not None and inside_prospective_store(root):
        print(
            f"Refused: --root {root} is (or lies inside) a prospective store; the research "
            "store must be a separate directory",
            file=sys.stderr,
        )
        return 2
    try:
        anchors = load_anchors(
            anchors_path(args.anchors, project_root=project_root),
            AnchorScope.RESEARCH,
            project_root=project_root,
            today=clock(),
            git_runner=git_runner,
        )
    except AnchorsRefused as error:
        print(f"Refused: anchors {error}", file=sys.stderr)
        return 2

    refusal = prepare_root(root) if args.store else None
    if refusal is not None:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 2
    try:
        retrieval = adapter_factory().retrieve(args.funds, start, end_exclusive)
        checks = verify_distribution_basis(retrieval, anchors=anchors)
    except (FundDataRefused, TiingoRequestFailed) as error:
        print(f"REFUSED {error}", file=sys.stderr)
        return 1
    for line in verification_lines(retrieval, checks, anchors):
        print(line)
    print()
    print(
        "vendor-adjusted reconciliation: not performed; it needs an owner-approved isolated "
        "audit procedure (docs/research/f1_fund_data.md)"
    )
    print("total return: not computed by the audit (OD8 group A descriptive calculations)")
    if not args.store:
        print(READ_ONLY)
        return 0
    if not all(check.passed for check in checks):
        failing = " ".join(check.fund_id for check in checks if not check.passed)
        print(
            f"NOT STORED unverified_distribution_basis: {failing}; nothing may be stored "
            "under an unverified basis",
            file=sys.stderr,
        )
        return 1
    try:
        stored = read_batch(ingest_batch(root, verified_batch(retrieval, anchors=anchors)).path)
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
