"""
Aggregate coverage audit of Tiingo fund data, for private local research.

Prepared, not yet run: it contacts Tiingo, so run it only after the owner has
configured `TIINGO_API_KEY` locally and authorized the run (`--authorized`).

Read-only by default: without `--store` it writes nothing anywhere. Storing
a real fund batch needs the owner's separate approval (`--store --root`).
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
ignored `data/raw/`. A missing root is created with mode 0o700; an existing
root whose mode grants any group or other permission is refused, and the
owner restricts it. It requests no adjusted series and performs no
vendor-adjusted reconciliation (that needs a separate, owner-approved isolated
procedure), and it computes no return (returns belong to the OD8 group A
descriptive calculations, not the audit).

Usage (a read-only coverage report):
    PYTHONPATH=src .venv/bin/python scripts/audit_fund_coverage.py --authorized

Storing, only after the owner separately approves persisting a batch:
    PYTHONPATH=src .venv/bin/python scripts/audit_fund_coverage.py --authorized \
        --store --root ~/stock-agent-private/fund_store
"""

from __future__ import annotations

import argparse
import os
import stat
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
from stock_agent.data.vendors.tiingo import (
    ALLOWED_FIELDS,
    BasisCheck,
    Retrieval,
    TiingoAdapter,
    TiingoRequestFailed,
    parse_payload,
    verified_batch,
    verify_distribution_basis,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUNDS = ("VTI", "AGG", "SHV")
ROOT_MODE = 0o700
# A pass is a spot check of sampled amounts, never proof that the history is valid.
SPOT_CHECKED = (
    "spot-checked against primary-source amounts (not a validation of every distribution)"
)
READ_ONLY = (
    "NOT STORED read-only coverage report: nothing was written; storing a batch needs the "
    "owner's separate approval (--store --root)"
)
_SHOWN = 10  # dates listed per finding


def _is_repository(path: Path, project_root: Path) -> bool:
    return os.path.samefile(path, project_root) or (path / ".git").exists()


def private_root(root: Path, project_root: Path) -> Path:
    """
    The resolved root, or ValueError if it could place vendor data in a
    repository. Every existing ancestor is checked with `os.path.samefile`
    (which also matches a case-variant path on a case-insensitive file
    system). An ancestor that is this repository, or holds a `.git` (another
    clone or a worktree), is allowed only when the root lies strictly inside
    that repository's `data/raw/`. A root that contains this repository is
    refused too.
    """

    resolved = root.expanduser().resolve()
    existing = [path for path in (resolved, *resolved.parents) if path.exists()]
    for index, ancestor in enumerate(existing):
        if not _is_repository(ancestor, project_root):
            continue
        raw = ancestor / "data" / "raw"
        below = [path for path in existing[:index] if path != resolved]
        if not (raw.is_dir() and any(os.path.samefile(path, raw) for path in below)):
            raise ValueError(
                f"--root {resolved} is inside the repository {ancestor}; use a directory "
                "outside it or a subdirectory of its data/raw/"
            )
    project = project_root.resolve()
    if resolved.exists() and any(
        os.path.samefile(resolved, path) for path in (project, *project.parents)
    ):
        raise ValueError(f"--root {resolved} contains the repository")
    return resolved


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


def verification_lines(retrieval: Retrieval, checks: tuple[BasisCheck, ...]) -> list[str]:
    """Phase 1: aggregates of the in-memory retrieval and its basis verification."""

    end_date = retrieval.end_exclusive - pd.Timedelta(days=1)
    window = f"[{retrieval.start:%Y-%m-%d}, {retrieval.end_exclusive:%Y-%m-%d})"
    expected = nyse.sessions(retrieval.start, end_date)
    lines = [
        "Tiingo coverage audit (aggregate counts, dates and ratios; no prices are printed)",
        f"request: {' '.join(retrieval.fund_ids)} {window}, Tiingo endDate {end_date:%Y-%m-%d}",
        f"retrieved_at {retrieval.retrieved_at}",
        f"field allowlist: every element had exactly {', '.join(ALLOWED_FIELDS)}",
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


def prepare_root(root: Path) -> str | None:
    """
    Create a missing root with mode 0o700. An existing root is never changed:
    it must be a directory that grants no group or other permission. Returns
    the refusal message, or None.
    """

    if not root.exists():
        root.mkdir(mode=ROOT_MODE, parents=True)
        os.chmod(root, ROOT_MODE)  # only the directory created here
        return None
    if not root.is_dir():
        return f"--root {root} exists and is not a directory"
    mode = stat.S_IMODE(root.stat().st_mode)
    if mode & 0o077:
        return (
            f"--root {root} has mode {mode:o}, which grants group or other access; "
            f"restrict it yourself (for example `chmod 700 {root}`) and run again"
        )
    return None


def reparse_matches(batch: StoredBatch) -> bool:
    """Whether the stored tables equal a fresh parse of the stored raw payload."""

    parsed = parse_payload((batch.path / PAYLOAD_FILE).read_bytes())
    fresh = (parsed.prices, parsed.distributions, parsed.splits)
    stored = (batch.prices, batch.distributions, batch.splits)
    return all(
        conform(spec, frame.assign(batch_id=batch.record.batch_id)).equals(table)
        for spec, frame, table in zip(DATA_TABLES, fresh, stored, strict=True)
    )


def main(
    argv: list[str] | None = None,
    *,
    adapter_factory: Callable[[], TiingoAdapter] = TiingoAdapter,
    project_root: Path = PROJECT_ROOT,
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
        root = private_root(args.root, project_root) if args.store else None
        start = parse_date("--start", args.start)
        end_exclusive = parse_date("--end-exclusive", args.end_exclusive)
    except (ValueError, FundDataRefused) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 2

    refusal = prepare_root(root) if args.store else None
    if refusal is not None:
        print(f"Refused: {refusal}", file=sys.stderr)
        return 2
    try:
        retrieval = adapter_factory().retrieve(args.funds, start, end_exclusive)
        checks = verify_distribution_basis(retrieval)
    except (FundDataRefused, TiingoRequestFailed) as error:
        print(f"REFUSED {error}", file=sys.stderr)
        return 1
    for line in verification_lines(retrieval, checks):
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
        stored = read_batch(ingest_batch(root, verified_batch(retrieval)).path)
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
