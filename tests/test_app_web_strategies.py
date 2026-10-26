"""The Lab app's read-only Strategies view: newest saved results, read strictly, worded, never recomputed."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from showcase_fixtures import config, default_frames, loader, sessions_2024

from stock_agent.app.web import strategies
from stock_agent.app.web.strategies import (
    DEMO_TITLE,
    NONE_TITLE,
    UNREADABLE_TITLE,
    build_strategies,
    newest_results,
    strategies_view,
)
from stock_agent.showcase.report import check_wording
from stock_agent.showcase.run import LABELS, run_showcase

NOW = datetime(2026, 10, 26, 21, 0, tzinfo=UTC)
SESSIONS = sessions_2024()


@pytest.fixture(scope="module")
def results():
    return run_showcase(
        config(offset_family=True),
        loader(default_frames(SESSIONS)),
        sessions=SESSIONS,
        code_sha="c" * 40,
        generated_at=NOW,
    )


def _save(root: Path, name: str, record) -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    path = folder / "results.json"
    path.write_text(record if isinstance(record, str) else json.dumps(record), encoding="utf-8")
    return path


def test_the_newest_run_is_chosen_by_its_stamp_not_its_name(tmp_path, results):
    _save(tmp_path, "real_rebalancing_v2-20261026T195110Z", results)
    newest = _save(tmp_path, "real_rebalancing_v1-20261026T201000Z", results)
    (tmp_path / "notes").mkdir()
    (tmp_path / "real_rebalancing_v3-20991231T000000Z").mkdir()  # no results.json: skipped
    assert newest_results(tmp_path) == newest
    assert newest_results(tmp_path / "missing") is None


def test_the_screen_shows_the_reports_numbers_worded_in_registered_order(tmp_path, results):
    _save(tmp_path, "tiny_showcase-20261026T220000Z", results)
    view = build_strategies(tmp_path, now=NOW)
    json.dumps(view, allow_nan=False)
    assert view["schema"] == "lab_app_strategies" and view["problem"] is None
    assert view["labels"] == list(LABELS)  # the report's caveats, unchanged
    assert [u["key"] for u in view["universes"]] == ["funds", "stocks"]
    funds = view["universes"][0]
    registered = [s["label"] for s in results["universes"][0]["strategies"]]
    assert [r["label"] for r in funds["rows"]] == registered
    hold = results["universes"][0]["strategies"][0]
    row = funds["rows"][0]
    assert row["cells"][0] == f"${hold['final_wealth']:,.0f}"
    sign = "+" if hold["total_return"] > 0 else "\u2212"
    assert row["cells"][1] == f"{sign}{abs(hold['total_return']) * 100:,.0f}%"
    assert row["cells"][2] == f"−{hold['drawdown']['value'] * 100:.0f}%"
    assert row["cells"][5] == "Never rebalances"
    weekly = funds["rows"][1]["cells"][5]
    assert weekly.endswith("across 5 days")
    assert "c" * 7 in view["source"] and "10 variants" in view["source"]


def test_the_chart_samples_the_saved_paths_without_recomputing(results):
    view = strategies_view(results, now=NOW)
    chart = view["universes"][0]["chart"]
    saved = results["universes"][0]["strategies"][2]["wealth"]
    drawn = chart["series"][2]["values"]
    assert drawn[0] == saved[0] and drawn[1] == saved[5] and drawn[-1] == saved[-1]
    assert len(chart["days"]) == len(drawn) == len(chart["series"][2]["words"])
    assert chart["days"][0].endswith("2024")


def test_the_demo_and_a_missing_report_show_a_plain_state(tmp_path):
    assert build_strategies(None, now=NOW)["problem"]["title"] == DEMO_TITLE
    assert build_strategies(tmp_path, now=NOW)["problem"]["title"] == NONE_TITLE


@pytest.mark.parametrize(
    "record",
    [
        '{"artifact": "showcase_results", "schema_version": 1, "x": NaN}',
        '{"artifact": "backtest_summary", "schema_version": 1}',
        '{"artifact": "showcase_results", "schema_version": 2}',
        '{"artifact": "showcase_results", "schema_version": 1, "universes": [{}]}',
        "not json",
    ],
)
def test_an_unreadable_report_shows_no_number(tmp_path, record):
    _save(tmp_path, "tiny_showcase-20261026T220000Z", record)
    view = build_strategies(tmp_path, now=NOW)
    assert view["problem"]["title"] == UNREADABLE_TITLE
    assert view["universes"] == [] and view["labels"] == []


def test_a_report_over_the_size_limit_is_not_read(tmp_path, results, monkeypatch):
    _save(tmp_path, "tiny_showcase-20261026T220000Z", results)
    monkeypatch.setattr(strategies, "MAX_BYTES", 100)
    assert build_strategies(tmp_path, now=NOW)["problem"]["title"] == UNREADABLE_TITLE


def test_the_screen_uses_no_ranking_words(results):
    view = strategies_view(results, now=NOW)
    assert check_wording(json.dumps(view, ensure_ascii=False)) == []


def test_the_app_and_the_showcase_agree_on_the_reports_folder(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / "showcase_run.py"
    spec = importlib.util.spec_from_file_location("showcase_run_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("STOCK_AGENT_SHOWCASE_REPORTS", raising=False)
    assert strategies.reports_root() == module.reports_root()
    monkeypatch.setenv("STOCK_AGENT_SHOWCASE_REPORTS", "/tmp/elsewhere")
    assert strategies.reports_root() == module.reports_root() == Path("/tmp/elsewhere")


def test_a_folder_that_cannot_be_read_shows_no_path(tmp_path, monkeypatch):
    def refuse(self):
        raise PermissionError(13, "Permission denied", str(tmp_path))

    monkeypatch.setattr(Path, "iterdir", refuse)
    tmp_path.joinpath("x").mkdir()
    view = build_strategies(tmp_path, now=NOW)
    assert view["problem"]["title"] == UNREADABLE_TITLE
    assert (
        str(tmp_path) not in json.dumps(view) and "PermissionError" in view["problem"]["technical"]
    )


@pytest.mark.parametrize(
    "path, value",
    [
        (("universes", 0, "strategies", 0, "drawdown", "value"), "deep"),
        (("universes", 0, "strategies", 0, "turnover", "rebalances"), "many"),
        (("universes", 0, "dates", 0), "not-a-date"),
        (("schema_version",), True),
    ],
)
def test_a_wrong_typed_field_shows_the_unreadable_screen(tmp_path, results, path, value):
    broken = json.loads(json.dumps(results))
    target = broken
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _save(tmp_path, "tiny_showcase-20261026T220000Z", broken)
    view = build_strategies(tmp_path, now=NOW)
    assert view["problem"]["title"] == UNREADABLE_TITLE and view["universes"] == []


def test_an_unreadable_newest_report_never_falls_back_to_an_older_one(tmp_path, results):
    _save(tmp_path, "tiny_showcase-20261026T210000Z", results)
    _save(tmp_path, "tiny_showcase-20261026T220000Z", "{broken")
    view = build_strategies(tmp_path, now=NOW)
    assert view["problem"]["title"] == UNREADABLE_TITLE and view["universes"] == []


def test_the_plain_states_use_no_ranking_words(tmp_path):
    for view in (build_strategies(None, now=NOW), build_strategies(tmp_path, now=NOW)):
        assert check_wording(json.dumps(view, ensure_ascii=False)) == []


def test_the_readout_words_and_days_match_the_saved_wealth(results):
    view = strategies_view(results, now=NOW)
    for universe, saved in zip(view["universes"], results["universes"], strict=True):
        chart = universe["chart"]
        n = len(saved["dates"])
        picks = [*range(0, n, strategies.STEP)]
        if picks[-1] != n - 1:
            picks.append(n - 1)
        assert len(chart["days"]) == len(picks)
        for series, strategy in zip(chart["series"], saved["strategies"], strict=True):
            for at, i in enumerate(picks):
                value = strategy["wealth"][i]
                assert series["values"][at] == value
                assert series["words"][at] == f"${value:,.0f}"
            assert series["label"] == strategy["label"]
