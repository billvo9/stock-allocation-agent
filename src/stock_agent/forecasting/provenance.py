"""
Run provenance read from the outside world: the code state (Git) and the
environment (Python, platform, package versions). Only the job scripts call
these; the runner and the prospective job take their results as arguments,
so their tests never shell out. The helpers themselves are tested on a
temporary repository. Everything here reads the local repository; nothing is
fetched.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import importlib.resources
import json
import os
import platform
import re
import subprocess
import sys
import zoneinfo
from collections.abc import Callable, Sequence
from datetime import datetime
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    packages_distributions,
    version,
)
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


# Rows of a RECORD that say where or how a distribution was installed, not what it contains.
_INSTALL_FILES = {"INSTALLER", "REQUESTED", "direct_url.json", "RECORD"}
# Run in a fresh interpreter: import the job's modules, then describe the installed
# distributions they loaded (and the required ones) with this module's own functions.
_CHILD = """
import json, sys
from stock_agent.forecasting.provenance import installed_distributions, timezone_file
modules, required = json.loads(sys.argv[1]), json.loads(sys.argv[2])
found = installed_distributions(modules, required)
print(json.dumps({"distributions": found, "timezone": timezone_file("America/New_York")}))
"""


def canonical_name(name: str) -> str:
    """A distribution name normalized as PEP 503 does (case, runs of `-`, `_` and `.`)."""

    return re.sub(r"[-_.]+", "-", name).lower()


def content_paths(record: str) -> list[str]:
    """
    The files of a RECORD that hold the distribution's content: rows with a
    hash, outside the install location's own directories (no `../` paths such
    as console scripts) and other than INSTALLER, REQUESTED, direct_url.json
    and RECORD.
    """

    paths = []
    for row in csv.reader(record.splitlines()):
        if len(row) < 2 or not row[1] or row[0].startswith("../"):
            continue
        name = row[0].rsplit("/", 1)[-1]
        if ".dist-info/" in row[0] and name in _INSTALL_FILES:
            continue
        paths.append(row[0])
    return sorted(paths)


def content_digest(paths: Sequence[str], locate: Callable[[str], Path]) -> str:
    """The SHA-256 over each path and the SHA-256 of its bytes on disk (or `missing`)."""

    digest = hashlib.sha256()
    for path in paths:
        file = Path(locate(path))
        content = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else "missing"
        digest.update(f"{path}\0{content}\n".encode())
    return digest.hexdigest()


def installed_distributions(
    modules: Sequence[str], required: Sequence[str] = ()
) -> dict[str, dict[str, object]]:
    """
    Import `modules`, then the version and content digest of every installed
    distribution owning a loaded top-level module, and of each `required` one.
    Meant for a fresh interpreter (environment_manifest runs it in one).
    """

    for name in modules:
        importlib.import_module(name)
    owners = packages_distributions()
    loaded = {m.split(".")[0] for m in list(sys.modules)}
    # One lookup per distribution, whatever spelling the metadata or the caller used.
    names = {canonical_name(d) for m in loaded for d in owners.get(m, ())}
    names |= {canonical_name(name) for name in required}
    found: dict[str, dict[str, object]] = {}
    for name in sorted(names):
        dist = distribution(name)
        key = canonical_name(dist.metadata["Name"])
        record = dist.read_text("RECORD")
        found[key] = {
            "version": dist.version,
            "content_sha256": None
            if record is None
            else content_digest(content_paths(record), lambda p, d=dist: Path(d.locate_file(p))),
        }
    return found


def timezone_file(key: str) -> dict[str, str]:
    """The zone file `zoneinfo` reads for `key`: the system database first, then tzdata."""

    for directory in zoneinfo.TZPATH:
        file = Path(directory) / key
        if file.is_file():
            return {"key": key, "source": directory, "sha256": _file_sha256(file)}
    data = importlib.resources.files("tzdata.zoneinfo").joinpath(*key.split("/"))
    return {"key": key, "source": "tzdata", "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment_manifest(
    root: str | Path, modules: Sequence[str], required: Sequence[str] = ()
) -> dict[str, object]:
    """
    The environment manifest (forecasting.prospective.environment) of the jobs
    whose entry modules are `modules`: the interpreter, the platform, the pins,
    the timezone file, and every installed distribution those modules import,
    plus `required` ones imported lazily, all read in a fresh interpreter (so
    the manifest does not depend on what this process loaded). Reads the local
    interpreter, installed files and metadata only.
    """

    root = Path(root)
    run = subprocess.run(
        [sys.executable, "-c", _CHILD, json.dumps(list(modules)), json.dumps(list(required))],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    child = json.loads(run.stdout)
    found = child["distributions"]
    found.pop("stock-allocation-agent", None)  # this project's code: the code identity covers it
    release = platform.mac_ver()[0] or platform.version()
    return {
        "manifest_schema_version": "1.0",
        "python": {
            "version": platform.python_version(),
            "minor": "{}.{}".format(*sys.version_info[:2]),
            "implementation": sys.implementation.name,
            "executable_sha256": _file_sha256(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": release,
            "machine": platform.machine(),
        },
        "pins": {
            name: _file_sha256(root / name) for name in ("requirements.txt", "pyproject.toml")
        },
        "timezone": child["timezone"],
        "distributions": dict(sorted(found.items())),
    }


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
