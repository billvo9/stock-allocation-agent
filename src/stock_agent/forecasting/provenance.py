"""
Run provenance read from the outside world: the code state (Git) and the
environment (Python, platform, package versions). Only the job scripts call
these; the runner and the prospective job take their results as arguments,
so their tests never shell out. The helpers themselves are tested on a
temporary repository. Everything here reads the local repository; nothing is
fetched.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
from collections.abc import Sequence
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "pyyaml")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def code_provenance(root: str | Path) -> dict[str, object]:
    """HEAD SHA and dirty flag; a dirty tree also records a hash of its diff."""

    root = Path(root)
    status = _git(root, "status", "--porcelain")
    dirty = bool(status.strip())
    result: dict[str, object] = {
        "git_sha": _git(root, "rev-parse", "HEAD").strip(),
        "git_dirty": dirty,
    }
    if dirty:
        diff = _git(root, "diff", "HEAD", "--binary")
        untracked = _git(root, "ls-files", "--others", "--exclude-standard")
        result["git_diff_sha256"] = hashlib.sha256((diff + "\0" + untracked).encode()).hexdigest()
    return result


def code_trees(root: str | Path, commit: str, paths: Sequence[str]) -> dict[str, str]:
    """The Git object id of each path at `commit` (a tree for a directory, a blob for a file)."""

    root = Path(root)
    return {path: _git(root, "rev-parse", f"{commit}:{path}").strip() for path in paths}


def paths_dirty(root: str | Path, paths: Sequence[str]) -> bool:
    """Whether any of `paths` has an uncommitted or untracked change (ignored files aside)."""

    return bool(_git(Path(root), "status", "--porcelain", "--", *paths).strip())


def committed_file(root: str | Path, commit: str, path: str) -> bytes:
    """A file's bytes at `commit`, read from the local repository (never fetched)."""

    return subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=Path(root), check=True, capture_output=True
    ).stdout


def commit_time(root: str | Path, commit: str) -> datetime:
    """A commit's committer time (for a GitHub merge commit, the merge time)."""

    return datetime.fromisoformat(_git(Path(root), "show", "-s", "--format=%cI", commit).strip())


def is_ancestor(root: str | Path, older: str, newer: str) -> bool:
    """Whether `older` is in `newer`'s history (a commit is its own ancestor)."""

    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", older, newer],
        cwd=Path(root),
        capture_output=True,
        text=True,
        check=False,  # 1 means "not an ancestor"; any other failure is raised below
    )
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stderr)
    return result.returncode == 0


def commit_exists(root: str | Path, revision: str) -> bool:
    """Whether `revision` names a commit in the local repository."""

    result = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=Path(root),
        capture_output=True,
        check=False,  # a missing object is the answer, not an error
    )
    return result.returncode == 0


def environment() -> dict[str, object]:
    versions: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for package in PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions
