"""
The owner's private distribution anchors: primary-source amounts per share as
paid, copied by hand from issuer pages into a git-ignored CSV, and loaded here
for the Tiingo distribution-basis spot check (`tiingo.py`, rules (i) and (ii)).

Why a private file
    Vanguard's and BlackRock's (iShares) website terms, read 2026-10-14, limit
    their content to personal, non-commercial use and forbid redistribution
    and automated access. This repository is public, so no issuer amount may
    live in code or in a committed file. Owner decision of 2026-10-14 ("Fill
    by hand"): the owner copies issuer amounts into this private file; rows the
    research agents gathered (under `issuer_sources/2026-10-14/`) are not
    anchors, and any path inside that directory is refused.

Where the file is
    Default `data/raw/issuer_sources/anchors.csv` under the project root
    (ignored by `.gitignore`'s `data/raw/*`). `STOCK_AGENT_ANCHORS` overrides
    it, and a script's `--anchors` overrides both. A path inside any
    repository (this one, another clone or a worktree) must be confirmed
    ignored by `git check-ignore`; otherwise it is refused, Git missing
    included. A missing file is an empty allowlist: every fund then fails the
    gate, so nothing is stored.

Columns (exactly these, in this order; a header-only template is
`docs/templates/distribution_anchors_template.csv`)
    fund              upper-case ticker, for example VTI
    ex_date           YYYY-MM-DD, an NYSE session (rules calendar, 2003-2027)
    amount            the issuer's string exactly as shown, with an optional
                      leading "$": "0.1234" or "$0.1234"; no sign, exponent,
                      thousands separator or leading "."
    decimals          the number of digits after the decimal point in `amount`
                      (catches a spreadsheet dropping trailing zeros)
    basis             `per_share_as_paid` (the only accepted value)
    status            `anchor` rows are used; any other lower-case word (for
                      example `draft`) is ignored and counted
    source_publisher  who published the amount, for example Vanguard
    source_url        the https:// page the amount was copied from
    retrieved_on      YYYY-MM-DD, the day it was copied (not after tomorrow, UTC)
    notes             free text, may be empty

Validation is strict: any failure refuses the whole file, naming the row and
column, never a value. Refused: a wrong header or field count, a repeated
(fund, ex_date) in any status, a malformed or non-session date, a
non-positive or malformed amount, a decimals count that does not match the
amount string, another basis, a malformed status, publisher or URL, control
characters, and invalid UTF-8 (a leading byte-order mark is accepted).

Scopes (the anchors lockbox)
    research      anchors with ex_date before `MODEL_HOLDOUT_START` (2025-01-01),
                  for the research store's audit
    prospective   anchors with ex_date on or after the prospective
                  `WINDOW_START`, for the prospective store
    An anchor dated in [MODEL_HOLDOUT_START, WINDOW_START), the sealed period
    before the prospective window, refuses the file in either scope. Anchor
    rows outside the requested scope are counted and never returned.

What it reports
    The file's sha256, row counts by status, per-fund anchor counts in scope
    and the count outside scope. Amounts never appear in any text; a batch's
    vintage note records the sha256.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import stat
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType

import pandas as pd

from stock_agent.calendars import nyse_rules
from stock_agent.calendars.nyse import CalendarRefused
from stock_agent.data.private_paths import GitRunner, git_ignored, repository_of
from stock_agent.data.private_paths import run_git_check_ignore as _run_git
from stock_agent.data.prospective.window import WINDOW_START
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START

ANCHORS_ENV = "STOCK_AGENT_ANCHORS"
DEFAULT_RELATIVE_PATH = Path("data/raw/issuer_sources/anchors.csv")
# The research agents' gathered rows; never read, never anchors.
EXCLUDED_DIRECTORY = ("issuer_sources", "2026-10-14")
COLUMNS = (
    "fund",
    "ex_date",
    "amount",
    "decimals",
    "basis",
    "status",
    "source_publisher",
    "source_url",
    "retrieved_on",
    "notes",
)
ANCHOR_STATUS = "anchor"
BASIS = "per_share_as_paid"
MAX_DECIMALS = 12
MAX_FILE_BYTES = 1 << 20

_TICKER = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,15}")
_DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")
_AMOUNT = re.compile(r"\$?(0|[1-9][0-9]{0,5})(\.([0-9]{1,12}))?")
_DECIMALS = re.compile(r"0|[1-9][0-9]?")
_STATUS = re.compile(r"[a-z][a-z_]{0,31}")
_URL = re.compile(r"https://[\x21-\x7e]{1,2040}")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class AnchorScope(StrEnum):
    RESEARCH = "research"
    PROSPECTIVE = "prospective"


class AnchorRefusal(StrEnum):
    UNSAFE_PATH = "unsafe_path"
    NOT_GIT_IGNORED = "not_git_ignored"
    RESEARCH_ROWS = "research_rows"
    INVALID_FILE = "invalid_file"
    INVALID_HEADER = "invalid_header"
    INVALID_ROW = "invalid_row"
    DUPLICATE_ANCHOR = "duplicate_anchor"
    INVALID_DATE = "invalid_date"
    INVALID_AMOUNT = "invalid_amount"
    DECIMALS_MISMATCH = "decimals_mismatch"
    INVALID_BASIS = "invalid_basis"
    INVALID_STATUS = "invalid_status"
    INVALID_SOURCE = "invalid_source"
    SEALED_PERIOD = "sealed_period"


class AnchorsRefused(ValueError):
    """The anchors file was refused; `reason` is machine-readable and `detail` names no value."""

    def __init__(self, reason: AnchorRefusal, detail: str) -> None:
        super().__init__(f"{reason.value}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, eq=False)
class DistributionAnchors:
    """
    The anchors usable in one scope. `entries` maps (fund, "YYYY-MM-DD") to
    the amount per share as paid; it never appears in the repr, and no method
    here formats an amount. `sha256` is None when no file exists.
    """

    scope: AnchorScope
    entries: Mapping[tuple[str, str], float] = field(repr=False)
    path: Path | None
    sha256: str | None
    rows: int
    status_counts: Mapping[str, int]
    outside_scope: int

    @classmethod
    def empty(cls, scope: AnchorScope, path: Path | None = None) -> DistributionAnchors:
        return cls(scope, MappingProxyType({}), path, None, 0, MappingProxyType({}), 0)

    def per_fund(self) -> dict[str, int]:
        return dict(sorted(Counter(fund for fund, _ in self.entries).items()))

    def provenance_note(self) -> str:
        """For a batch's vintage note: the file's sha256, never its contents."""

        if self.sha256 is None:
            return "distribution anchors file: none"
        return f"distribution anchors file sha256 {self.sha256} ({self.scope.value} scope)"

    def summary_lines(self) -> list[str]:
        """Counts and the sha256 only; amounts never appear."""

        if self.sha256 is None:
            return [
                (
                    f"anchors file: {self.path} is missing; empty allowlist, so no fund can "
                    "pass the distribution-basis gate and nothing can be stored"
                )
            ]
        statuses = ", ".join(f"{name} {count}" for name, count in self.status_counts.items())
        per_fund = ", ".join(f"{fund} {count}" for fund, count in self.per_fund().items())
        return [
            f"anchors file: {self.path}",
            f"  sha256 {self.sha256}",
            f"  rows {self.rows} ({statuses or 'none'})",
            f"  {self.scope.value}-scope anchors per fund: {per_fund or 'none'}",
            (
                f"  anchors outside the {self.scope.value} scope (counted, not used): "
                f"{self.outside_scope}"
            ),
        ]


def anchors_path(
    cli_value: str | os.PathLike | None,
    *,
    project_root: Path,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """`--anchors` if given, else `STOCK_AGENT_ANCHORS`, else the default under the project root."""

    environ = os.environ if environ is None else environ
    if cli_value is not None:
        return Path(cli_value).expanduser()
    if environ.get(ANCHORS_ENV, "").strip():
        return Path(environ[ANCHORS_ENV]).expanduser()
    return project_root / DEFAULT_RELATIVE_PATH


def _in_excluded_directory(path: Path) -> bool:
    # Case-folded: macOS's default file system ignores case, so ISSUER_SOURCES/ is the
    # same directory as issuer_sources/.
    parts = tuple(part.casefold() for part in path.parts)
    size = len(EXCLUDED_DIRECTORY)
    return any(parts[i : i + size] == EXCLUDED_DIRECTORY for i in range(len(parts) - size + 1))


def _read_regular_file(path: Path) -> bytes:
    """At most MAX_FILE_BYTES + 1 bytes of a regular file, refusing a swapped-in symlink."""

    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        descriptor = None
    if descriptor is None:
        raise AnchorsRefused(AnchorRefusal.UNSAFE_PATH, f"{path} cannot be opened as a file")
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise AnchorsRefused(AnchorRefusal.UNSAFE_PATH, f"{path} is not a regular file")
        return handle.read(MAX_FILE_BYTES + 1)


def check_path(path: Path, *, project_root: Path, git_runner: GitRunner = _run_git) -> Path:
    """
    The resolved path, or `AnchorsRefused`: inside `issuer_sources/2026-10-14/`
    (as given or resolved), or inside a repository where Git does not confirm
    it is ignored.
    """

    absolute = path.expanduser().absolute()
    resolved = absolute.resolve()
    if _in_excluded_directory(absolute) or _in_excluded_directory(resolved):
        raise AnchorsRefused(
            AnchorRefusal.RESEARCH_ROWS,
            f"{absolute} is inside issuer_sources/2026-10-14/, the research agents' gathered "
            "rows, which are not anchors",
        )
    if resolved.exists() and not resolved.is_file():
        raise AnchorsRefused(AnchorRefusal.UNSAFE_PATH, f"{resolved} is not a file")
    repository = repository_of(resolved, project_root)
    if repository is not None and not git_ignored(resolved, repository, git_runner):
        raise AnchorsRefused(
            AnchorRefusal.NOT_GIT_IGNORED,
            f"{resolved} is inside the repository {repository} but git check-ignore does not "
            "confirm it is ignored; keep the anchors outside every repository or in an ignored "
            "path such as data/raw/",
        )
    return resolved


def _session(row: int, column: str, text: str) -> pd.Timestamp:
    if not _DATE.fullmatch(text):
        raise AnchorsRefused(AnchorRefusal.INVALID_DATE, f"row {row} {column} is not YYYY-MM-DD")
    try:
        day = date.fromisoformat(text)
    except ValueError:
        day = None
    if day is None:
        raise AnchorsRefused(AnchorRefusal.INVALID_DATE, f"row {row} {column} is not a date")
    if column != "ex_date":
        return pd.Timestamp(day.isoformat(), tz="UTC").as_unit("us")
    try:
        session = nyse_rules.is_session(day)
    except CalendarRefused as error:
        raise AnchorsRefused(
            AnchorRefusal.INVALID_DATE, f"row {row} ex_date is {error.reason}"
        ) from None
    if not session:
        raise AnchorsRefused(
            AnchorRefusal.INVALID_DATE, f"row {row} ex_date is not an NYSE session"
        )
    return pd.Timestamp(day.isoformat(), tz="UTC").as_unit("us")


def _amount(row: int, text: str, decimals: str) -> float:
    match = _AMOUNT.fullmatch(text)
    if match is None:
        raise AnchorsRefused(
            AnchorRefusal.INVALID_AMOUNT,
            f"row {row} amount is not a plain decimal as shown by the issuer",
        )
    if not _DECIMALS.fullmatch(decimals) or int(decimals) > MAX_DECIMALS:
        raise AnchorsRefused(
            AnchorRefusal.DECIMALS_MISMATCH, f"row {row} decimals is not a whole number 0-12"
        )
    shown = len(match.group(3) or "")
    if shown != int(decimals):
        raise AnchorsRefused(
            AnchorRefusal.DECIMALS_MISMATCH,
            f"row {row} decimals says {decimals} but the amount string has {shown}",
        )
    value = Decimal(text.removeprefix("$"))
    if value <= 0:
        raise AnchorsRefused(AnchorRefusal.INVALID_AMOUNT, f"row {row} amount must be positive")
    return float(value)


def _check_row(row: int, record: dict[str, str], today: pd.Timestamp) -> tuple[str, str, float]:
    for column, text in record.items():
        if _CONTROL.search(text):
            raise AnchorsRefused(
                AnchorRefusal.INVALID_ROW, f"row {row} {column} has a control character"
            )
    fund = record["fund"]
    if not _TICKER.fullmatch(fund):
        raise AnchorsRefused(AnchorRefusal.INVALID_ROW, f"row {row} fund is not a ticker")
    ex_date = _session(row, "ex_date", record["ex_date"])
    amount = _amount(row, record["amount"], record["decimals"])
    if record["basis"] != BASIS:
        raise AnchorsRefused(AnchorRefusal.INVALID_BASIS, f"row {row} basis must be {BASIS}")
    if not _STATUS.fullmatch(record["status"]):
        raise AnchorsRefused(
            AnchorRefusal.INVALID_STATUS, f"row {row} status is not a lower-case word"
        )
    publisher = record["source_publisher"]
    if not publisher.strip() or len(publisher) > 200:
        raise AnchorsRefused(AnchorRefusal.INVALID_SOURCE, f"row {row} source_publisher is empty")
    if not _URL.fullmatch(record["source_url"]):
        raise AnchorsRefused(
            AnchorRefusal.INVALID_SOURCE, f"row {row} source_url is not an https:// URL"
        )
    retrieved = _session(row, "retrieved_on", record["retrieved_on"])
    if retrieved > today + pd.Timedelta(days=1):
        raise AnchorsRefused(AnchorRefusal.INVALID_DATE, f"row {row} retrieved_on is in the future")
    if MODEL_HOLDOUT_START <= ex_date < WINDOW_START:
        raise AnchorsRefused(
            AnchorRefusal.SEALED_PERIOD,
            f"row {row} ex_date is in the sealed holdout before the prospective window "
            f"({MODEL_HOLDOUT_START.date()} to {WINDOW_START.date()}); remove the row",
        )
    return fund, f"{ex_date:%Y-%m-%d}", amount


def _in_scope(scope: AnchorScope, ex_date: str) -> bool:
    day = pd.Timestamp(ex_date, tz="UTC")
    if scope is AnchorScope.RESEARCH:
        return day < MODEL_HOLDOUT_START
    return day >= WINDOW_START


def _text(raw: bytes) -> str:
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        pass
    raise AnchorsRefused(AnchorRefusal.INVALID_FILE, "the anchors file is not UTF-8")


def parse_anchors(
    raw: bytes, scope: AnchorScope, *, today: pd.Timestamp, path: Path | None = None
) -> DistributionAnchors:
    """Validate the file bytes (every row, whatever the scope) and keep the scope's anchors."""

    scope = AnchorScope(scope)
    if len(raw) > MAX_FILE_BYTES:
        raise AnchorsRefused(AnchorRefusal.INVALID_FILE, "the anchors file is over 1 MiB")
    try:
        rows = list(csv.reader(io.StringIO(_text(raw), newline=""), strict=True))
    except csv.Error:
        rows = None  # the refusal is raised below, so no chained error quotes a line
    if rows is None:
        raise AnchorsRefused(AnchorRefusal.INVALID_FILE, "the anchors file is not valid CSV")
    if not rows or tuple(rows[0]) != COLUMNS:
        raise AnchorsRefused(
            AnchorRefusal.INVALID_HEADER, f"the header must be exactly {','.join(COLUMNS)}"
        )
    seen: set[tuple[str, str]] = set()
    statuses: Counter[str] = Counter()
    entries: dict[tuple[str, str], float] = {}
    outside = 0
    for number, values in enumerate(rows[1:], start=2):
        if len(values) != len(COLUMNS):
            raise AnchorsRefused(
                AnchorRefusal.INVALID_ROW,
                f"row {number} has {len(values)} fields, expected {len(COLUMNS)}",
            )
        record = dict(zip(COLUMNS, values, strict=True))
        key = _check_row(number, record, today)
        if key[:2] in seen:
            raise AnchorsRefused(
                AnchorRefusal.DUPLICATE_ANCHOR, f"row {number} repeats {key[0]} {key[1]}"
            )
        seen.add(key[:2])
        statuses[record["status"]] += 1
        if record["status"] != ANCHOR_STATUS:
            continue
        if _in_scope(scope, key[1]):
            entries[key[:2]] = key[2]
        else:
            outside += 1
    return DistributionAnchors(
        scope=scope,
        entries=MappingProxyType(dict(sorted(entries.items()))),
        path=path,
        sha256=hashlib.sha256(raw).hexdigest(),
        rows=len(rows) - 1,
        status_counts=MappingProxyType(dict(sorted(statuses.items()))),
        outside_scope=outside,
    )


def load_anchors(
    path: Path,
    scope: AnchorScope,
    *,
    project_root: Path,
    today: pd.Timestamp,
    git_runner: GitRunner = _run_git,
) -> DistributionAnchors:
    """
    The scope's anchors from `path` after the path checks; an empty allowlist
    when the file does not exist.
    """

    resolved = check_path(path, project_root=project_root, git_runner=git_runner)
    if not resolved.exists():
        return DistributionAnchors.empty(AnchorScope(scope), resolved)
    return parse_anchors(_read_regular_file(resolved), scope, today=today, path=resolved)


__all__ = [
    "ANCHORS_ENV",
    "COLUMNS",
    "DEFAULT_RELATIVE_PATH",
    "AnchorRefusal",
    "AnchorScope",
    "AnchorsRefused",
    "DistributionAnchors",
    "anchors_path",
    "check_path",
    "load_anchors",
    "parse_anchors",
]
