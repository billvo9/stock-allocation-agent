"""
Aggregate coverage audit of Tiingo fund data, for private local research.

Prepared, not yet run: it contacts Tiingo, so run it only after the owner has
configured `TIINGO_API_KEY` locally and authorized the run (`--authorized`).

It fetches the funds through the Tiingo adapter (one prices call per fund,
allowlisted columns only, no retry). A success means every element held
exactly the allowlisted fields, so Tiingo honoured `columns`; any other field
refuses the batch (`non_allowlisted_field`). Before storing, it checks the
divCash share basis at each verified split. It then stores the batch with F1
`ingest_batch` under a private root, reads it back with the fail-closed
reader, and prints aggregate facts only:
- row counts, first and last sessions, and missing NYSE sessions;
- distributions per year;
- verified splits, with the raw-close consistency ratio the adapter enforced;
- the divCash share-basis check: the median of the last pre-split divCash
  amounts over the median of the first post-split amounts. About the split
  factor means per share as paid; about 1 means split-adjusted, and then the
  batch is not stored, because its `per_share_as_paid` label would be wrong;
- whether the stored tables match a re-parse of the stored raw payload.

It never prints prices or the key, and writes vendor data only under `--root`,
which must lie outside the repository or under its ignored `data/raw/`. It
requests no adjusted series and performs no vendor-adjusted reconciliation
(that needs a separate, owner-approved isolated procedure), and it computes no
return (pending OD8 group A, T5-A).

Usage:
    PYTHONPATH=src .venv/bin/python scripts/audit_fund_coverage.py \
        --root ~/stock-agent-private/fund_store --authorized
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_agent.calendars import nyse
from stock_agent.data.funds.adapter import FundDataAdapter
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
    TiingoAdapter,
    TiingoRequestFailed,
    parse_payload,
)
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FUNDS = ("VTI", "AGG", "SHV")
BASIS_WINDOW = 4  # distributions compared on each side of a split (a year of quarterlies)
PER_SHARE_AS_PAID = "per share as paid"
SPLIT_ADJUSTED = "split-adjusted"
INSUFFICIENT = "insufficient distributions"
_SHOWN = 10  # dates listed per finding


def private_root(root: Path, project_root: Path) -> Path:
    """The resolved root; ValueError if it would place vendor data in the repository."""

    resolved = root.expanduser().resolve()
    repo = project_root.resolve()
    raw = (repo / "data" / "raw").resolve()
    if resolved.is_relative_to(repo):
        if resolved == raw or not resolved.is_relative_to(raw):
            raise ValueError(
                f"--root {resolved} is inside the repository; use a directory outside it "
                "or a subdirectory of data/raw/"
            )
    elif repo.is_relative_to(resolved):
        raise ValueError(f"--root {resolved} contains the repository")
    return resolved


def _dates(values: pd.Series) -> str:
    shown = [f"{value:%Y-%m-%d}" for value in values.iloc[:_SHOWN]]
    more = f" (+{len(values) - _SHOWN} more)" if len(values) > _SHOWN else ""
    return (", ".join(shown) + more) if shown else "none"


@dataclass(frozen=True)
class ShareBasisCheck:
    fund_id: str
    split_date: pd.Timestamp
    factor: float
    before: int
    after: int
    ratio: float | None
    verdict: str


def share_basis_checks(distributions: pd.DataFrame, splits: pd.DataFrame) -> list[ShareBasisCheck]:
    """
    At each split: median divCash of the last `BASIS_WINDOW` ex-dates before it
    over the median of the first `BASIS_WINDOW` from it on. Amounts per share
    as paid fall by the split factor, so the ratio is near the factor;
    split-adjusted amounts give a ratio near 1. The verdict is whichever is
    nearer on a log scale. A heuristic: a dividend change near the split moves it.
    """

    checks = []
    for split in splits.itertuples():
        own = distributions[distributions["fund_id"] == split.fund_id].sort_values("ex_date")
        before = own.loc[own["ex_date"] < split.effective_date, "amount"].iloc[-BASIS_WINDOW:]
        after = own.loc[own["ex_date"] >= split.effective_date, "amount"].iloc[:BASIS_WINDOW]
        ratio, verdict = None, INSUFFICIENT
        if len(before) >= 2 and len(after) >= 2:
            ratio = float(before.median() / after.median())
            nearer_factor = abs(math.log(ratio / split.ratio_new_per_old)) < abs(math.log(ratio))
            verdict = PER_SHARE_AS_PAID if nearer_factor else SPLIT_ADJUSTED
        checks.append(
            ShareBasisCheck(
                split.fund_id,
                split.effective_date,
                split.ratio_new_per_old,
                len(before),
                len(after),
                ratio,
                verdict,
            )
        )
    return checks


def _split_line(batch: StoredBatch, split) -> str:
    """The split with the raw-close ratio the adapter enforced: about 1 if unadjusted."""

    closes = batch.prices[batch.prices["fund_id"] == split.fund_id].set_index("session_date")
    previous = nyse.sessions(nyse.COVERAGE.first, split.effective_date - pd.Timedelta(days=1))[-1]
    ratio = (
        closes.loc[split.effective_date, "close"]
        * split.ratio_new_per_old
        / closes.loc[previous, "close"]
    )
    return (
        f"{split.effective_date:%Y-%m-%d} x{split.ratio_new_per_old:g} (verified; raw close x "
        f"factor / previous raw close {ratio:.2f})"
    )


def audit_lines(batch: StoredBatch, checks: list[ShareBasisCheck]) -> list[str]:
    record = batch.record
    end_date = record.end_exclusive - pd.Timedelta(days=1)
    window = f"[{record.start:%Y-%m-%d}, {record.end_exclusive:%Y-%m-%d})"
    lines = [
        "Tiingo coverage audit (aggregate counts, dates and ratios; no prices are printed)",
        f"request: {' '.join(record.fund_ids)} {window}, Tiingo endDate {end_date:%Y-%m-%d}",
        f"batch: {record.batch_id} (retrieved_at {record.retrieved_at})",
        f"field allowlist: every element had exactly {', '.join(ALLOWED_FIELDS)}",
        "stored: unadjusted closes, divCash distributions, verified splits only",
    ]
    for fund in record.fund_ids:
        prices = batch.prices[batch.prices["fund_id"] == fund]
        sessions = prices["session_date"].sort_values()
        missing = batch.missing_sessions.loc[
            batch.missing_sessions["fund_id"] == fund, "session_date"
        ].sort_values()
        distributions = batch.distributions[batch.distributions["fund_id"] == fund]
        splits = batch.splits[batch.splits["fund_id"] == fund]
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
            + (", ".join(_split_line(batch, split) for split in splits.itertuples()) or "none")
        )
        for check in (check for check in checks if check.fund_id == fund):
            ratio = "n/a" if check.ratio is None else f"{check.ratio:.2f}"
            lines.append(
                f"  divCash share basis at {check.split_date:%Y-%m-%d} x{check.factor:g}: "
                f"median of {check.before} before / median of {check.after} after = {ratio} "
                f"-> {check.verdict}"
            )
    lines.append("")
    lines.append(
        "vendor-adjusted reconciliation: not performed; it needs an owner-approved isolated "
        "audit procedure (docs/research/f1_fund_data.md)"
    )
    lines.append("total return: pending OD8 group A (T5-A)")
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


def main(
    argv: list[str] | None = None,
    *,
    adapter_factory: Callable[[], FundDataAdapter] = TiingoAdapter,
    project_root: Path = PROJECT_ROOT,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", type=Path, required=True, help="private store directory")
    parser.add_argument("--funds", nargs="+", default=list(DEFAULT_FUNDS))
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end-exclusive", default=f"{MODEL_HOLDOUT_START:%Y-%m-%d}")
    parser.add_argument(
        "--authorized",
        action="store_true",
        help="confirm the owner configured TIINGO_API_KEY and authorized this network run",
    )
    args = parser.parse_args(argv)

    if not args.authorized:
        print(
            "Refused: this audit contacts Tiingo. Run it only after the owner configures "
            "TIINGO_API_KEY locally and authorizes the run, then pass --authorized.",
            file=sys.stderr,
        )
        return 2
    try:
        root = private_root(args.root, project_root)
        start = parse_date("--start", args.start)
        end_exclusive = parse_date("--end-exclusive", args.end_exclusive)
    except (ValueError, FundDataRefused) as error:
        print(f"Refused: {error}", file=sys.stderr)
        return 2

    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        raw = adapter_factory().fetch(args.funds, start, end_exclusive)
        checks = share_basis_checks(raw.distributions, raw.splits)
        wrong = [check for check in checks if check.verdict == SPLIT_ADJUSTED]
        if wrong:
            first = wrong[0]
            print(
                f"REFUSED divcash_basis_mismatch: {first.fund_id} divCash looks split-adjusted "
                f"at {first.split_date:%Y-%m-%d} (ratio {first.ratio:.2f}); not stored",
                file=sys.stderr,
            )
            return 1
        stored = read_batch(ingest_batch(root, raw).path)
    except (FundDataRefused, TiingoRequestFailed) as error:
        print(f"REFUSED {error}", file=sys.stderr)
        return 1
    for line in audit_lines(stored, checks):
        print(line)
    print(f"stored tables match a re-parse of the stored raw payload: {reparse_matches(stored)}")
    print(f"stored under: {stored.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
