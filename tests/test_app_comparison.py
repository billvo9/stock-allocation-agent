"""
The Experiments page's "Synthetic strategy comparison (engineering
evaluation)" panel, rendered through the running app and scanned for advice
wording: the honest empty state, a registered comparison with its labels,
registry stages and tables, and every file outside the registered protocol
withheld, including an artifact whose look the private evaluation log does
not verify.
"""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest
from comparison_helpers import CODE, FAKE_KINDS, FixedClock, small_config
from test_app_destinations import running, text_of

from stock_agent.app import comparison, wording
from stock_agent.comparison import ledger, report, runner

STRATEGIES = ("buy_and_hold", "equal_weight", "all_shv", "ppo", "ppo_guarded")


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    """One registered (tiny) comparison artifact, produced by the real runner."""

    root = tmp_path_factory.mktemp("lab_comparison")
    config = small_config(*STRATEGIES)
    runner.train(config, run_dir=root / "run", code=CODE, clock=FixedClock(), kinds=FAKE_KINDS)
    path = runner.evaluate(
        config,
        run_dir=root / "run",
        fold_role="test",
        reports_root=root / "reports",
        health_inputs_root=root / "health",
        code=CODE,
        clock=FixedClock(),
        kinds=FAKE_KINDS,
        looks_root=root / "looks",
    )
    (log,) = (root / "looks").glob("*.jsonl")
    return SimpleNamespace(path=path, log=log)


@pytest.fixture
def directory(tmp_path, monkeypatch, produced):
    target = tmp_path / "comparison"
    (target / produced.path.parent.name).mkdir(parents=True)
    (target / produced.path.parent.name / "comparison.json").write_bytes(produced.path.read_bytes())
    looks = tmp_path / "looks"
    looks.mkdir()
    shutil.copy(produced.log, looks / produced.log.name)
    monkeypatch.setenv(comparison.COMPARISON_ENV, str(target))
    monkeypatch.setenv(comparison.LOOKS_ENV, str(looks))
    return target


def _log(directory):
    (path,) = (directory.parent / "looks").glob("*.jsonl")
    return path


def _rewrite(directory, edit, *, rehash=True):
    (path,) = directory.glob("*/comparison.json")
    document = json.loads(path.read_text())
    edit(document)
    if rehash:
        document["content_sha256"] = report.content_sha256(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_the_readers_contract_is_the_producers():
    assert comparison.SCHEMA == report.ARTIFACT_SCHEMA
    assert comparison.SCHEMA_VERSION == report.ARTIFACT_SCHEMA_VERSION
    assert comparison.PROTOCOL == report.PROTOCOL
    assert comparison.SCOPE_LABEL == report.SCOPE_LABEL == wording.COMPARISON_SCOPE_BADGE
    assert comparison.EVALUATION_LABEL == report.EVALUATION_LABEL
    assert comparison.EVIDENCE_STATUS == report.EVIDENCE_STATUS
    assert comparison.SHOWN_STAGES[-1] == report.MAX_STAGE
    assert comparison.LOOKS_ENV == ledger.LOOKS_ENV
    assert comparison.DEFAULT_LOOKS_DIR == ledger.DEFAULT_LOOKS_DIR
    assert comparison.LOG_FIELDS == ledger.FIELDS and comparison.GENESIS == ledger.GENESIS


def test_without_a_comparison_the_panel_is_an_honest_empty_state():
    app = running("experiments")
    text = text_of(app)
    assert wording.COMPARISON_TITLE in text
    assert wording.COMPARISON_EMPTY_TITLE in text and wording.COMPARISON_EMPTY in text
    assert not app.dataframe and not app.selectbox


def test_a_registered_comparison_is_shown_with_its_labels_and_stages(directory, produced):
    app = running("experiments")
    text = text_of(app)  # scanned for advice wording
    assert wording.COMPARISON_SCOPE_BADGE in text and wording.COMPARISON_EVALUATION_BADGE in text
    assert wording.COMPARISON_EMPTY_TITLE not in text
    assert wording.EXPERIMENTS_EMPTY_TITLE not in text
    assert wording.COMPARISON_NOT_REPORTED in text and wording.COMPARISON_NO_CLAIM in text
    document = json.loads(produced.path.read_text())
    assert "Uninvested cash earns 0% (synthetic engineering only; group C pending)." in text
    assert "look 1 of 1 recorded for this configuration" in text
    assert "The private evaluation log verifies this look" in text
    results, seeds, builds, strategies = (frame.value for frame in app.dataframe)
    stages = set(strategies[wording.COMPARISON_STRATEGY_COLUMNS["stage"]])
    assert stages == {wording.PROSPECTIVE_STAGES["engineering_only"]}
    labels = [s["label"] for s in document["strategies"]]
    assert (
        list(results[wording.COMPARISON_RESULT_COLUMNS["strategy"]]) == labels
    )  # registered order
    assert list(builds[wording.COMPARISON_BUILD_COLUMNS["strategy"]]) == labels
    assert set(seeds[wording.COMPARISON_SEED_COLUMNS["seeds"]]) == {"5"}
    for word in ("CAGR", "Sharpe"):
        assert word not in text.replace(wording.COMPARISON_NOT_REPORTED, "")


def test_the_tables_show_the_artifacts_numbers_unchanged(directory, produced):
    document = json.loads(produced.path.read_text())
    summary = next(
        s
        for s in document["summary"]
        if (s["constraint_set"], s["strategy"], s["scenario"]) == ("plain", "all_shv", "calm")
    )
    app = running("experiments")
    results = app.dataframe[0].value
    row = results[
        results[wording.COMPARISON_RESULT_COLUMNS["strategy"]] == "100% short-Treasury fund"
    ]
    spread = summary["terminal_wealth"]
    expected = f"{spread['median']:.4f} ({spread['min']:.4f} to {spread['max']:.4f})"
    assert row.iloc[0][wording.COMPARISON_RESULT_COLUMNS["wealth"]] == expected


def test_choosing_the_strict_set_shows_its_incomplete_builds(directory):
    app = running("experiments")
    strict = next(o for o in app.selectbox[0].options if o.startswith("strict"))
    app.selectbox[0].set_value(strict).run()
    text_of(app)
    builds = app.dataframe[2].value
    held = builds[builds[wording.COMPARISON_BUILD_COLUMNS["strategy"]].str.startswith("60/30/10")]
    episodes = held.iloc[0][wording.COMPARISON_BUILD_COLUMNS["never"]]
    assert episodes == "4 of 4"  # held after a partial build: never fully built
    cash = held.iloc[0][wording.COMPARISON_BUILD_COLUMNS["cash"]]
    # Capped to 0.5 / 0.3 / 0.1 (0.1 cash), then tau = 0.9 from cash and alpha = 0.25 / 0.9:
    # 0.25 invested at the build, 75% cash left uninvested.
    assert cash == "75.0%"


@pytest.mark.parametrize(
    ("edit", "rehash", "reason"),
    [
        (lambda d: d["rows"][0].update(terminal_wealth=2.0), False, "edited"),
        (lambda d: d["strategies"][0].update(registry_stage="historically_checked"), True, "stage"),
        (lambda d: d.update(synthetic=False), True, "synthetic"),
        (lambda d: d["labels"].update(scope="Simulated results"), True, "labels"),
        (lambda d: d["fold"].update(role="dev"), True, "test-fold"),
        (lambda d: d["transition"].update(status="engineering_only"), True, "approved"),
        (lambda d: d["fairness"].update(contract="failed"), True, "fairness"),
        (lambda d: d.update(protocol="ad_hoc"), True, "registered comparison protocol"),
        (lambda d: d["look"].update(number=0), True, "on record"),
    ],
)
def test_anything_outside_the_registered_protocol_is_withheld(directory, edit, rehash, reason):
    _rewrite(directory, edit, rehash=rehash)
    app = running("experiments")
    text = text_of(app)
    assert "Withheld:" in text and reason in text
    assert not app.dataframe and not app.selectbox
    assert wording.COMPARISON_SCOPE_BADGE not in text.replace(wording.COMPARISON_EMPTY, "")


def test_the_default_location_is_ignored_by_git():
    relative = comparison.DEFAULT_DIR.relative_to(comparison.DEFAULT_DIR.parents[1])
    assert relative.parts == ("reports", "comparison")
    ignored = (comparison.DEFAULT_DIR.parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "reports/" in ignored.splitlines()


def test_a_look_the_private_log_does_not_verify_is_withheld(directory):
    log = _log(directory)
    text = log.read_text()
    for edited, reason in (
        (None, "not in the evaluation log"),
        (text.replace('"test"', '"dev"', 1), "integrity check"),
        ("", "does not match the evaluation log"),
    ):
        if edited is None:
            log.unlink()
        else:
            log.write_text(edited, encoding="utf-8")
        app = running("experiments")
        shown = text_of(app)
        assert "Withheld:" in shown and reason in shown, reason
        assert not app.dataframe
        log.write_text(text, encoding="utf-8")


def test_the_labs_log_check_agrees_with_the_producers(produced, tmp_path):
    text = produced.log.read_text()
    sha = produced.log.stem
    mine = comparison.log_entries(text, sha)
    theirs = ledger.EvaluationLog(produced.log, sha).entries()
    assert [e["entry_sha256"] for e in mine] == [e["entry_sha256"] for e in theirs]
    copy = tmp_path / produced.log.name
    for edited in (text.replace('"test"', '"dev"', 1), text.rstrip("\n"), text * 2):
        copy.write_text(edited, encoding="utf-8")
        with pytest.raises(ledger.LogUnreadable):
            ledger.EvaluationLog(copy, sha).entries()
        with pytest.raises(comparison.ComparisonWithheld):
            comparison.log_entries(edited, sha)


def test_the_turnover_columns_are_split_and_not_annualized(directory):
    app = running("experiments")
    text_of(app)
    columns = list(app.dataframe[0].value.columns)
    for key in ("construction", "rebalance", "per_rebalance", "sessions"):
        assert wording.COMPARISON_RESULT_COLUMNS[key] in columns
    assert not [c for c in columns if "year" in c.lower()]
