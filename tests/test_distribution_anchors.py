"""
The private distribution-anchors loader (`data/vendors/anchors.py`). Every
amount here is invented; no issuer value appears in this repository.
"""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from stock_agent.data.vendors import anchors as anchors_module
from stock_agent.data.vendors.anchors import (
    ANCHORS_ENV,
    COLUMNS,
    DEFAULT_RELATIVE_PATH,
    AnchorRefusal,
    AnchorScope,
    AnchorsRefused,
    anchors_path,
    check_path,
    load_anchors,
    parse_anchors,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = PROJECT_ROOT / "docs" / "templates" / "distribution_anchors_template.csv"
TODAY = pd.Timestamp("2026-10-14T12:00:00Z")
HEADER = ",".join(COLUMNS)
URL = "https://example.invalid/fixture"


def row(
    fund="AGG",
    ex_date="2024-06-03",
    amount="0.1234",
    decimals="4",
    basis="per_share_as_paid",
    status="anchor",
    publisher="Fixture publisher",
    url=URL,
    retrieved="2026-10-14",
    notes="invented",
) -> str:
    return (
        f"{fund},{ex_date},{amount},{decimals},{basis},{status},{publisher},{url},"
        f"{retrieved},{notes}"
    )


def csv_bytes(*rows: str, header: str = HEADER) -> bytes:
    return ("\n".join((header, *rows)) + "\n").encode("utf-8")


def parse(*rows: str, scope=AnchorScope.RESEARCH, **kwargs):
    return parse_anchors(csv_bytes(*rows, **kwargs), scope, today=TODAY)


def refused(reason: AnchorRefusal, *rows: str, **kwargs) -> AnchorsRefused:
    with pytest.raises(AnchorsRefused) as caught:
        parse(*rows, **kwargs)
    assert caught.value.reason is reason, caught.value
    return caught.value


def never_git(repository, path):
    raise AssertionError("git must not run for a path outside every repository")


# --- the template and the default path ------------------------------------------


def test_the_template_is_header_only():
    text = TEMPLATE.read_text(encoding="utf-8")
    assert text == HEADER + "\n"  # no values: only the column names
    assert parse_anchors(text.encode(), AnchorScope.RESEARCH, today=TODAY).rows == 0


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_default_path_is_git_ignored_and_the_template_is_not():
    def ignored(relative: Path) -> bool:
        completed = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "check-ignore", "-q", "--", str(relative)],
            capture_output=True,
            check=False,
        )
        if completed.returncode == 128:
            pytest.skip("not inside a git checkout")
        return completed.returncode == 0

    assert DEFAULT_RELATIVE_PATH == Path("data/raw/issuer_sources/anchors.csv")
    assert ignored(DEFAULT_RELATIVE_PATH)
    assert not ignored(TEMPLATE.relative_to(PROJECT_ROOT))


def test_the_path_is_the_flag_then_the_environment_then_the_default(tmp_path):
    root = tmp_path / "project"
    assert anchors_path(None, project_root=root, environ={}) == root / DEFAULT_RELATIVE_PATH
    assert anchors_path(None, project_root=root, environ={ANCHORS_ENV: "  "}) == (
        root / DEFAULT_RELATIVE_PATH
    )
    env = {ANCHORS_ENV: str(tmp_path / "env.csv")}
    assert anchors_path(None, project_root=root, environ=env) == tmp_path / "env.csv"
    flag = tmp_path / "flag.csv"
    assert anchors_path(str(flag), project_root=root, environ=env) == flag


# --- path safety -------------------------------------------------------------------


def test_a_missing_file_is_an_empty_allowlist(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    loaded = load_anchors(
        tmp_path / "absent.csv",
        AnchorScope.RESEARCH,
        project_root=project,
        today=TODAY,
        git_runner=never_git,
    )
    assert dict(loaded.entries) == {} and loaded.sha256 is None and loaded.rows == 0
    assert "is missing; empty allowlist" in loaded.summary_lines()[0]
    assert loaded.provenance_note() == "distribution anchors file: none"


def test_an_in_repository_path_must_be_confirmed_ignored(tmp_path):
    project = tmp_path / "project"
    target = project / "data" / "raw" / "issuer_sources" / "anchors.csv"
    target.parent.mkdir(parents=True)
    target.write_bytes(csv_bytes(row()))
    calls = []

    def git(answer):
        def runner(repository, path):
            calls.append((repository, path))
            return answer

        return runner

    loaded = load_anchors(
        target, AnchorScope.RESEARCH, project_root=project, today=TODAY, git_runner=git(0)
    )
    assert loaded.rows == 1 and calls[-1] == (project.resolve(), target.resolve())
    for answer in (1, 128, -1):  # not ignored, not a repository, git missing or failing
        with pytest.raises(AnchorsRefused) as caught:
            load_anchors(
                target,
                AnchorScope.RESEARCH,
                project_root=project,
                today=TODAY,
                git_runner=git(answer),
            )
        assert caught.value.reason is AnchorRefusal.NOT_GIT_IGNORED


def test_another_clone_or_worktree_counts_as_a_repository(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    clone = tmp_path / "clone"
    clone.mkdir()
    (clone / ".git").write_text("gitdir: /elsewhere\n", encoding="utf-8")
    with pytest.raises(AnchorsRefused) as caught:
        check_path(clone / "anchors.csv", project_root=project, git_runner=lambda r, p: 1)
    assert caught.value.reason is AnchorRefusal.NOT_GIT_IGNORED
    outside = check_path(tmp_path / "outside.csv", project_root=project, git_runner=never_git)
    assert outside == (tmp_path / "outside.csv").resolve()


@pytest.mark.parametrize(
    "relative",
    [
        "data/raw/issuer_sources/2026-10-14/anchors.csv",
        "elsewhere/issuer_sources/2026-10-14/vanguard/rows.csv",
        "data/raw/ISSUER_SOURCES/2026-10-14/anchors.csv",  # one directory on macOS
        "data/raw/Issuer_Sources/2026-10-14/x.csv",
    ],
)
def test_the_research_agents_rows_are_never_anchors(tmp_path, relative):
    project = tmp_path / "project"
    project.mkdir()
    path = tmp_path / relative
    with pytest.raises(AnchorsRefused) as caught:
        check_path(path, project_root=project, git_runner=lambda r, p: 0)
    assert caught.value.reason is AnchorRefusal.RESEARCH_ROWS


def test_a_symlink_into_the_research_rows_is_refused_without_reading(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    gathered = tmp_path / "issuer_sources" / "2026-10-14"
    gathered.mkdir(parents=True)
    (gathered / "rows.csv").write_text("not read", encoding="utf-8")
    link = tmp_path / "anchors.csv"
    link.symlink_to(gathered / "rows.csv")
    with pytest.raises(AnchorsRefused) as caught:
        load_anchors(link, AnchorScope.RESEARCH, project_root=project, today=TODAY)
    assert caught.value.reason is AnchorRefusal.RESEARCH_ROWS


def test_a_case_variant_directory_on_disk_is_refused(tmp_path):
    project = tmp_path / "project"
    gathered = project / "data" / "raw" / "issuer_sources" / "2026-10-14"
    gathered.mkdir(parents=True)
    variant = project / "data" / "raw" / "ISSUER_SOURCES" / "2026-10-14" / "rows.csv"
    with pytest.raises(AnchorsRefused) as caught:
        check_path(variant, project_root=project, git_runner=lambda r, p: 0)
    assert caught.value.reason is AnchorRefusal.RESEARCH_ROWS


def test_an_oversized_file_is_refused_without_reading_it_all(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    big = tmp_path / "big.csv"
    big.write_bytes(csv_bytes(row()) + b"#" * (anchors_module.MAX_FILE_BYTES + 10))
    with pytest.raises(AnchorsRefused) as caught:
        load_anchors(big, AnchorScope.RESEARCH, project_root=project, today=TODAY)
    assert caught.value.reason is AnchorRefusal.INVALID_FILE


def test_a_fifo_is_not_an_anchors_file(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    fifo = tmp_path / "anchors.csv"
    os.mkfifo(fifo)
    with pytest.raises(AnchorsRefused) as caught:
        load_anchors(fifo, AnchorScope.RESEARCH, project_root=project, today=TODAY)
    assert caught.value.reason is AnchorRefusal.UNSAFE_PATH


def test_a_directory_is_not_an_anchors_file(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    with pytest.raises(AnchorsRefused) as caught:
        check_path(tmp_path, project_root=project, git_runner=never_git)
    assert caught.value.reason is AnchorRefusal.UNSAFE_PATH


# --- parsing and validation --------------------------------------------------------


def test_a_valid_file_gives_entries_counts_and_its_sha256():
    raw = csv_bytes(
        row(),
        row(ex_date="2024-07-01", amount="$0.25", decimals="2"),
        row(fund="VTI", ex_date="2024-03-22", amount="0.5678"),
        row(fund="VTI", ex_date="2024-06-28", status="draft"),
        row(fund="SHV", ex_date="2024-08-01", status="unverified"),
    )
    loaded = parse_anchors(raw, AnchorScope.RESEARCH, today=TODAY)
    assert dict(loaded.entries) == {
        ("AGG", "2024-06-03"): 0.1234,
        ("AGG", "2024-07-01"): 0.25,
        ("VTI", "2024-03-22"): 0.5678,
    }
    assert loaded.sha256 == hashlib.sha256(raw).hexdigest()
    assert loaded.rows == 5
    assert dict(loaded.status_counts) == {"anchor": 3, "draft": 1, "unverified": 1}
    assert loaded.per_fund() == {"AGG": 2, "VTI": 1}
    assert loaded.outside_scope == 0
    assert "entries" not in repr(loaded) and "0.1234" not in repr(loaded)


def test_the_summary_never_shows_an_amount(tmp_path):
    loaded = parse(row(amount="0.123456", decimals="6"), row(ex_date="2024-07-01"))
    text = "\n".join(loaded.summary_lines()) + loaded.provenance_note()
    assert "0.123456" not in text and "0.1234" not in text
    assert f"sha256 {loaded.sha256}" in text
    assert "rows 2 (anchor 2)" in text and "research-scope anchors per fund: AGG 2" in text


def test_a_byte_order_mark_is_accepted_and_hashed_as_is():
    raw = b"\xef\xbb\xbf" + csv_bytes(row())
    loaded = parse_anchors(raw, AnchorScope.RESEARCH, today=TODAY)
    assert loaded.rows == 1 and loaded.sha256 == hashlib.sha256(raw).hexdigest()


def test_scopes_partition_the_anchors_and_count_the_rest():
    rows = (row(ex_date="2024-06-03"), row(ex_date="2026-08-03"), row(ex_date="2026-09-01"))
    research = parse(*rows, scope=AnchorScope.RESEARCH)
    prospective = parse(*rows, scope=AnchorScope.PROSPECTIVE)
    assert list(research.entries) == [("AGG", "2024-06-03")] and research.outside_scope == 2
    assert list(prospective.entries) == [("AGG", "2026-08-03"), ("AGG", "2026-09-01")]
    assert prospective.outside_scope == 1
    assert research.sha256 == prospective.sha256


@pytest.mark.parametrize("ex_date", ["2025-01-02", "2025-06-02", "2026-07-15"])
def test_an_anchor_in_the_sealed_period_refuses_the_file_in_either_scope(ex_date):
    for scope in AnchorScope:
        error = refused(AnchorRefusal.SEALED_PERIOD, row(ex_date=ex_date), scope=scope)
        assert "0.1234" not in str(error)
    # Status does not matter: sealed-period values may not be written at all.
    refused(AnchorRefusal.SEALED_PERIOD, row(ex_date="2025-06-02", status="draft"))


@pytest.mark.parametrize(
    ("ex_date", "scope", "other"),
    [
        ("2026-07-16", AnchorScope.PROSPECTIVE, AnchorScope.RESEARCH),  # the window start
        ("2026-10-14", AnchorScope.PROSPECTIVE, AnchorScope.RESEARCH),  # H_end
        ("2024-12-31", AnchorScope.RESEARCH, AnchorScope.PROSPECTIVE),  # the last research day
    ],
)
def test_anchors_on_each_scope_edge_are_kept_in_scope_and_counted_outside(ex_date, scope, other):
    kept = parse(row(ex_date=ex_date), scope=scope)
    assert dict(kept.entries) == {("AGG", ex_date): 0.1234}
    assert kept.outside_scope == 0 and kept.per_fund() == {"AGG": 1}
    elsewhere = parse(row(ex_date=ex_date), scope=other)
    assert dict(elsewhere.entries) == {} and elsewhere.outside_scope == 1


@pytest.mark.parametrize(
    ("header", "reason"),
    [
        (",".join(COLUMNS[:-1]), AnchorRefusal.INVALID_HEADER),
        (",".join((*COLUMNS, "extra")), AnchorRefusal.INVALID_HEADER),
        (",".join(reversed(COLUMNS)), AnchorRefusal.INVALID_HEADER),
        (HEADER.upper(), AnchorRefusal.INVALID_HEADER),
    ],
)
def test_the_header_must_be_exact(header, reason):
    refused(reason, row(), header=header)


def test_an_empty_file_is_refused():
    with pytest.raises(AnchorsRefused) as caught:
        parse_anchors(b"", AnchorScope.RESEARCH, today=TODAY)
    assert caught.value.reason is AnchorRefusal.INVALID_HEADER


def test_duplicates_in_any_status_are_refused():
    refused(AnchorRefusal.DUPLICATE_ANCHOR, row(), row(amount="0.1235"))
    refused(AnchorRefusal.DUPLICATE_ANCHOR, row(), row(status="draft"))


@pytest.mark.parametrize(
    "ex_date",
    ["2024-6-3", "2024-06-31", "2024-02-30", "20240603", "2024-06-01", "2024-07-04", "2028-01-03"],
)
def test_bad_or_non_session_ex_dates_are_refused(ex_date):
    refused(AnchorRefusal.INVALID_DATE, row(ex_date=ex_date))


@pytest.mark.parametrize("retrieved", ["2026-10-16", "2026-1-14", "yesterday", ""])
def test_bad_or_future_retrieval_dates_are_refused(retrieved):
    refused(AnchorRefusal.INVALID_DATE, row(retrieved=retrieved))


def test_a_retrieval_date_one_day_ahead_of_utc_is_accepted():
    assert parse(row(retrieved="2026-10-15")).rows == 1  # a local date east of UTC


@pytest.mark.parametrize(
    "amount",
    ["0", "0.0000", "$0.00", "-0.25", "+0.25", ".25", '"0,25"', "1e-3", "", " 0.25", "00.25"],
)
def test_non_positive_or_malformed_amounts_are_refused(amount):
    decimals = str(len(amount.partition(".")[2])) if "." in amount else "0"
    error = refused(AnchorRefusal.INVALID_AMOUNT, row(amount=amount, decimals=decimals))
    assert amount.strip() == "" or amount not in error.detail


@pytest.mark.parametrize(
    ("amount", "decimals"),
    [("0.25", "4"), ("0.2500", "2"), ("1", "1"), ("0.25", ""), ("0.25", "2.0"), ("0.25", "-2")],
)
def test_the_decimals_must_match_the_amount_string(amount, decimals):
    refused(AnchorRefusal.DECIMALS_MISMATCH, row(amount=amount, decimals=decimals))


def test_trailing_zeros_are_kept_as_shown():
    loaded = parse(row(amount="0.2500", decimals="4"))
    assert dict(loaded.entries) == {("AGG", "2024-06-03"): 0.25}


@pytest.mark.parametrize("basis", ["split_adjusted_as_of_retrieval", "", "PER_SHARE_AS_PAID"])
def test_only_per_share_as_paid_is_accepted(basis):
    refused(AnchorRefusal.INVALID_BASIS, row(basis=basis))


@pytest.mark.parametrize("status", ["", "Anchor", "anchor!", "1anchor"])
def test_a_malformed_status_is_refused(status):
    refused(AnchorRefusal.INVALID_STATUS, row(status=status))


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"publisher": ""}, AnchorRefusal.INVALID_SOURCE),
        ({"publisher": " "}, AnchorRefusal.INVALID_SOURCE),
        ({"url": "http://example.invalid/x"}, AnchorRefusal.INVALID_SOURCE),
        ({"url": "https://example.invalid/a b"}, AnchorRefusal.INVALID_SOURCE),
        ({"url": ""}, AnchorRefusal.INVALID_SOURCE),
        ({"fund": "agg"}, AnchorRefusal.INVALID_ROW),
        ({"fund": ""}, AnchorRefusal.INVALID_ROW),
        ({"notes": "tab\there"}, AnchorRefusal.INVALID_ROW),
    ],
)
def test_other_fields_are_checked(change, reason):
    refused(reason, row(**change))


def test_a_wrong_field_count_or_bad_csv_is_refused():
    refused(AnchorRefusal.INVALID_ROW, row() + ",extra")
    refused(AnchorRefusal.INVALID_ROW, ",".join(row().split(",")[:-1]))
    refused(AnchorRefusal.INVALID_ROW, "")  # a blank line is a row with no fields
    with pytest.raises(AnchorsRefused) as caught:
        parse_anchors(csv_bytes('AGG,"2024-06-03'), AnchorScope.RESEARCH, today=TODAY)
    assert caught.value.reason is AnchorRefusal.INVALID_FILE
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_invalid_utf8_or_an_oversized_file_is_refused():
    for raw in (csv_bytes(row()) + b"\xff", b"x" * (anchors_module.MAX_FILE_BYTES + 1)):
        with pytest.raises(AnchorsRefused) as caught:
            parse_anchors(raw, AnchorScope.RESEARCH, today=TODAY)
        assert caught.value.reason is AnchorRefusal.INVALID_FILE


def test_a_quoted_field_with_a_comma_in_notes_is_accepted():
    loaded = parse(row(notes='"copied by hand, rounded as shown"'))
    assert loaded.rows == 1


def test_refusal_details_name_rows_and_columns_never_values():
    error = refused(AnchorRefusal.DECIMALS_MISMATCH, row(amount="0.98765", decimals="4"))
    assert "row 2" in error.detail and "0.98765" not in str(error)


# --- the offline check script -------------------------------------------------------

_SCRIPT = PROJECT_ROOT / "scripts" / "check_anchors.py"
_spec = importlib.util.spec_from_file_location("stock_agent_check_anchors", _SCRIPT)
check_script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check_script
_spec.loader.exec_module(check_script)


def run_check(tmp_path, *argv):
    project = tmp_path / "project"
    project.mkdir(exist_ok=True)
    return check_script.main(
        list(argv), project_root=project, clock=lambda: TODAY, git_runner=never_git
    )


def test_the_check_script_prints_counts_for_both_scopes_and_no_amount(tmp_path, capsys):
    path = tmp_path / "anchors.csv"
    path.write_bytes(csv_bytes(row(), row(ex_date="2026-08-03", amount="0.4321")))
    assert run_check(tmp_path, "--anchors", str(path)) == 0
    out = capsys.readouterr().out
    assert "research scope" in out and "prospective scope" in out
    assert "research-scope anchors per fund: AGG 1" in out
    assert "prospective-scope anchors per fund: AGG 1" in out
    assert out.count(hashlib.sha256(path.read_bytes()).hexdigest()) == 2
    assert "0.1234" not in out and "0.4321" not in out


def test_the_scripts_default_path_is_checked_with_git(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(ANCHORS_ENV, raising=False)
    project = tmp_path / "project"
    project.mkdir()
    asked = []

    def git(repository, path):
        asked.append((repository, path))
        return 0

    code = check_script.main([], project_root=project, clock=lambda: TODAY, git_runner=git)
    assert code == 0 and "is missing; empty allowlist" in capsys.readouterr().out
    default = (project / DEFAULT_RELATIVE_PATH).resolve()
    assert asked and set(asked) == {(project.resolve(), default)}
    code = check_script.main(
        [], project_root=project, clock=lambda: TODAY, git_runner=lambda r, p: 1
    )
    assert code == 2 and "not_git_ignored" in capsys.readouterr().err


def test_the_check_script_reports_a_missing_file_and_refusals(tmp_path, capsys):
    assert run_check(tmp_path, "--anchors", str(tmp_path / "absent.csv")) == 0
    assert "is missing; empty allowlist" in capsys.readouterr().out
    bad = tmp_path / "bad.csv"
    bad.write_bytes(csv_bytes(row(ex_date="2025-06-02")))
    assert run_check(tmp_path, "--anchors", str(bad)) == 2
    assert "Refused: anchors sealed_period: row 2" in capsys.readouterr().err
