"""
Where private vendor and issuer data may live: shared by the coverage audit,
the prospective collection and the distribution-anchors loader.

Vendor data and issuer amounts are never committed (the repository is public;
vendor and issuer terms forbid redistribution). A private path must lie
outside every repository, or inside a repository only where Git ignores it.

    private_root     a store root: outside every repository, or strictly inside
                     that repository's `data/raw/` where `git check-ignore`
                     confirms it is ignored
    prepare_root     create a missing root with mode 0o700; never change an
                     existing one, and refuse it if it grants group or other access
    repository_of    the repository (this one, another clone or a worktree) that
                     contains a path, if any
    git_ignored      whether `git check-ignore` says a path is ignored in its repository

Nothing here touches the network.
"""

from __future__ import annotations

import os
import stat
import subprocess
from collections.abc import Callable
from pathlib import Path

ROOT_MODE = 0o700
GIT_TIMEOUT_SECONDS = 30

GitRunner = Callable[[Path, Path], int]


def _is_repository(path: Path, project_root: Path) -> bool:
    return os.path.samefile(path, project_root) or (path / ".git").exists()


def private_root(root: Path, project_root: Path, git_runner: GitRunner | None = None) -> Path:
    """
    The resolved root, or ValueError if it could place vendor data in a
    repository. Every existing ancestor is checked with `os.path.samefile`
    (which also matches a case-variant path on a case-insensitive file
    system). An ancestor that is this repository, or holds a `.git` (another
    clone or a worktree), is allowed only when the root lies strictly inside
    that repository's `data/raw/` and `git check-ignore` confirms the root is
    ignored there (a repository that tracks its `data/raw/` is refused). A
    root that contains this repository is refused too.
    """

    runner = run_git_check_ignore if git_runner is None else git_runner

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
        if not git_ignored(resolved, ancestor, runner):
            raise ValueError(
                f"--root {resolved} is inside the repository {ancestor} and git check-ignore "
                "does not confirm it is ignored; use a directory outside every repository"
            )
    project = project_root.resolve()
    if resolved.exists() and any(
        os.path.samefile(resolved, path) for path in (project, *project.parents)
    ):
        raise ValueError(f"--root {resolved} contains the repository")
    return resolved


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
    if root.is_symlink() or not root.is_dir():
        return f"--root {root} exists and is not a directory"
    mode = stat.S_IMODE(root.stat().st_mode)
    if mode & 0o077:
        return (
            f"--root {root} has mode {mode:o}, which grants group or other access; "
            f"restrict it yourself (for example `chmod 700 {root}`) and run again"
        )
    return None


def repository_of(path: Path, project_root: Path) -> Path | None:
    """
    The nearest existing ancestor of `path` (resolved) that is `project_root`
    or holds a `.git` file or directory, or None outside every repository.
    """

    resolved = path.expanduser().resolve()
    for ancestor in (resolved, *resolved.parents):
        if ancestor.exists() and ancestor.is_dir() and _is_repository(ancestor, project_root):
            return ancestor
    return None


def run_git_check_ignore(repository: Path, path: Path) -> int:
    """`git -C <repository> check-ignore -q -- <path>`: 0 ignored, 1 not, other on failure."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "check-ignore", "-q", "--", str(path)],
            capture_output=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return -1
    return completed.returncode


def git_ignored(path: Path, repository: Path, runner: GitRunner = run_git_check_ignore) -> bool:
    """
    True only when Git confirms the path is ignored in the repository. A
    tracked file is never ignored; Git missing or failing counts as not
    ignored (fail closed).
    """

    return runner(repository, path) == 0


__all__ = [
    "ROOT_MODE",
    "GitRunner",
    "git_ignored",
    "prepare_root",
    "private_root",
    "repository_of",
    "run_git_check_ignore",
]
