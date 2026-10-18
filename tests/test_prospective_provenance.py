"""
The local Git helpers behind the registration's code identity
(forecasting/provenance.py), on a temporary repository: offline and
deterministic. Tree ids follow content, a change under a registered path is
dirty, and a committed file is read back byte for byte.
"""

from __future__ import annotations

import shutil
import subprocess

import pandas as pd
import pytest

from stock_agent.forecasting import provenance

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
PATHS = ("src", "configs/prospective", "pyproject.toml")
# A fixed identity and dates: the repository and its commit ids are reproducible.
IDENTITY = {
    "GIT_AUTHOR_NAME": "fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    "GIT_AUTHOR_DATE": "2026-10-14T14:00:00+00:00",
    "GIT_COMMITTER_DATE": "2026-10-14T14:00:00+00:00",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(root, *args, date=None):
    env = {**IDENTITY, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
    if date is not None:
        env |= {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date}
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True, env=env
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "configs" / "prospective").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "src" / "pkg" / "code.py").write_text("VALUE = 1\n")
    (root / "configs" / "prospective" / "c.yaml").write_text("a: 1\n")
    (root / "pyproject.toml").write_text("[project]\n")
    (root / "docs" / "notes.md").write_text("notes\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "first")
    return root


def test_tree_ids_identify_the_content_of_each_path(repo):
    first = _git(repo, "rev-parse", "HEAD")
    trees = provenance.code_trees(repo, first, PATHS)
    assert set(trees) == set(PATHS) and all(len(v) == 40 for v in trees.values())
    (repo / "docs" / "notes.md").write_text("more notes\n")  # outside the registered paths
    _git(repo, "commit", "-qam", "docs only")
    assert provenance.code_trees(repo, "HEAD", PATHS) == trees
    (repo / "src" / "pkg" / "code.py").write_text("VALUE = 2\n")
    _git(repo, "commit", "-qam", "code")
    changed = provenance.code_trees(repo, "HEAD", PATHS)
    assert changed["src"] != trees["src"] and changed["pyproject.toml"] == trees["pyproject.toml"]


def test_uncommitted_or_untracked_changes_under_a_path_are_dirty(repo):
    assert provenance.paths_dirty(repo, PATHS) is False
    (repo / "docs" / "notes.md").write_text("edited\n")
    assert provenance.paths_dirty(repo, PATHS) is False  # outside the registered paths
    (repo / "src" / "pkg" / "new.py").write_text("NEW = 1\n")  # untracked
    assert provenance.paths_dirty(repo, PATHS) is True


def test_a_committed_file_is_read_back_and_a_missing_one_refused(repo):
    head = _git(repo, "rev-parse", "HEAD")
    assert provenance.committed_file(repo, head, "pyproject.toml") == b"[project]\n"
    with pytest.raises(subprocess.CalledProcessError):
        provenance.committed_file(repo, head, "docs/absent.json")


def test_a_commit_time_and_its_ancestry(repo):
    first = _git(repo, "rev-parse", "HEAD")
    (repo / "docs" / "notes.md").write_text("later\n")
    _git(repo, "commit", "-qam", "later", date="2026-10-15T09:00:00+00:00")
    second = _git(repo, "rev-parse", "HEAD")
    assert provenance.commit_time(repo, second) == pd.Timestamp("2026-10-15T09:00:00Z")
    assert provenance.is_ancestor(repo, first, second) is True
    assert provenance.is_ancestor(repo, second, first) is False
