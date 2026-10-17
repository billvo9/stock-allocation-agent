"""
The evaluation log: a write-once, hash-chained record of every look at a
fold, one file per registered configuration.

Location: ONE private place per machine, independent of where artifacts are
written, so a fresh reports directory cannot restart the count:
`<looks dir>/<config_sha256>.jsonl`, where the looks directory is
`~/stock-agent-private/comparison/looks` (the environment variable
STOCK_AGENT_COMPARISON_LOOKS replaces it; the test suite points it at a
temporary directory). A file holds only its configuration's looks.

Each line is one look: the configuration's name and SHA-256, the fold, the
look's number for that (configuration, fold), the artifact and frozen
selection it belongs to, the code version and the time. `entry_sha256` is
the SHA-256 of the canonical JSON (sorted keys, no whitespace) of the line's
other fields, and each line names the previous line's hash, so an edited,
removed or reordered line breaks the chain and the whole log is refused.
Lines are only ever appended; nothing is rewritten.

Rules:
- A look is recorded BEFORE the fold is evaluated, so a run that fails still
  counts.
- The test fold is evaluated once per registered configuration. A second
  look is refused (`RepeatLookError`) unless the caller explicitly records it
  as a repeat; it is then counted and labelled, never hidden.
- Development-fold looks are unlimited and counted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

LOOKS_ENV = "STOCK_AGENT_COMPARISON_LOOKS"
DEFAULT_LOOKS_DIR = Path.home() / "stock-agent-private" / "comparison" / "looks"
GENESIS = "0" * 64
_SHA256 = re.compile(r"[0-9a-f]{64}")
LOG_SCHEMA_VERSION = 1
FIELDS = {
    "log_schema_version",
    "sequence",
    "recorded_at",
    "config_name",
    "config_sha256",
    "fold",
    "look_number",
    "repeat",
    "artifact_id",
    "selection_sha256",
    "code_sha",
    "git_dirty",
    "prev_entry_sha256",
    "entry_sha256",
}


class LogUnreadable(ValueError):
    """The evaluation log failed its integrity check; no look may be counted from it."""


class RepeatLookError(RuntimeError):
    """The test fold of this configuration was already evaluated."""


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(entry: Mapping[str, object]) -> str:
    body = {key: value for key, value in entry.items() if key != "entry_sha256"}
    return hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()


def looks_dir() -> Path:
    configured = os.environ.get(LOOKS_ENV)
    return Path(configured) if configured else DEFAULT_LOOKS_DIR


class EvaluationLog:
    """The looks of one registered configuration (`for_config`)."""

    def __init__(self, path: Path, config_sha256: str) -> None:
        if not _SHA256.fullmatch(config_sha256):
            raise ValueError("config_sha256 must be a SHA-256 hex digest")
        self.path = Path(path)
        self.config_sha256 = config_sha256

    @classmethod
    def for_config(cls, config_sha256: str, directory: Path | None = None) -> EvaluationLog:
        root = looks_dir() if directory is None else Path(directory)
        return cls(root / f"{config_sha256}.jsonl", config_sha256)

    def head(self) -> str:
        """The hash of the last look (the genesis value for an empty log)."""

        entries = self.entries()
        return entries[-1]["entry_sha256"] if entries else GENESIS

    def entries(self) -> list[dict]:
        """Every look, after verifying the chain (`LogUnreadable` otherwise)."""

        if not self.path.exists():
            return []
        text = self.path.read_text(encoding="utf-8")
        if text and not text.endswith("\n"):
            raise LogUnreadable("its last line is incomplete")
        entries, previous = [], GENESIS
        for number, line in enumerate(text.splitlines()):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise LogUnreadable(f"line {number + 1} is not JSON") from error
            if not isinstance(entry, dict) or set(entry) != FIELDS:
                raise LogUnreadable(f"line {number + 1} has unexpected fields")
            if entry["sequence"] != number or entry["prev_entry_sha256"] != previous:
                raise LogUnreadable(f"line {number + 1} breaks the chain")
            if entry["entry_sha256"] != _digest(entry):
                raise LogUnreadable(f"line {number + 1} was edited")
            if entry["config_sha256"] != self.config_sha256:
                raise LogUnreadable(f"line {number + 1} belongs to another configuration")
            entries.append(entry)
            previous = entry["entry_sha256"]
        return entries

    def looks(self, fold: str) -> int:
        return sum(1 for entry in self.entries() if entry["fold"] == fold)

    def record_look(
        self,
        *,
        config_name: str,
        config_sha256: str,
        fold: str,
        artifact_id: str,
        selection_sha256: str,
        code: Mapping[str, object],
        recorded_at: datetime,
        repeat_allowed: bool = False,
    ) -> dict:
        """
        Append one look and return it. A second test-fold look at the same
        configuration is refused unless `repeat_allowed`.
        """

        if config_sha256 != self.config_sha256:
            raise ValueError("This log holds another configuration's looks.")
        entries = self.entries()
        earlier = sum(1 for e in entries if e["fold"] == fold)
        if fold == "test" and earlier and not repeat_allowed:
            raise RepeatLookError(
                f"The test fold of {config_name} ({config_sha256[:12]}) was already evaluated "
                f"{earlier} time(s). The test fold is evaluated once per registered "
                "configuration; a repeat must be recorded explicitly as one."
            )
        entry = {
            "log_schema_version": LOG_SCHEMA_VERSION,
            "sequence": len(entries),
            "recorded_at": recorded_at.strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "config_name": config_name,
            "config_sha256": config_sha256,
            "fold": fold,
            "look_number": earlier + 1,
            "repeat": earlier > 0,
            "artifact_id": artifact_id,
            "selection_sha256": selection_sha256,
            "code_sha": code.get("git_sha"),
            "git_dirty": code.get("git_dirty"),
            "prev_entry_sha256": entries[-1]["entry_sha256"] if entries else GENESIS,
        }
        entry["entry_sha256"] = _digest(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical(entry) + "\n")
        return entry
