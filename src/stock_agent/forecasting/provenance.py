"""
Run provenance read from the outside world: the code state (Git) and the
environment (Python, platform, package versions). Only the experiment
script calls these; the runner takes their results as arguments, so tests
never shell out.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
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
