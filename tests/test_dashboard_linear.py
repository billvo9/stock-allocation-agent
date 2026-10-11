"""
The dashboard on a T4 run: the Linear models page, candidate identity in the
design system, candidates as defaults, the OLS diagnostic as a comparator,
and the stored-row invariants the loader re-verifies for 1.2 tables.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from stock_agent.dashboard import evaluation, loader, style, theme
from stock_agent.model_diagnostics import artifacts, contract

APP = Path(__file__).resolve().parents[1] / "src" / "stock_agent" / "dashboard" / "app.py"
PAGE = "page_scripts/linear_models.py"
VARIANT = "20991231T000000Z-eeeeeeeeeeee"


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def _app(root, monkeypatch, page=None) -> AppTest:
    monkeypatch.setenv(loader.ROOT_ENVIRONMENT_VARIABLE, str(root))
    app = AppTest.from_file(str(APP), default_timeout=180)
    app.run()
    if page:
        app.switch_page(page).run()
    return app


def _charts(app):
    return {chart.proto.id.rsplit("-", 1)[-1]: chart for chart in app.get("plotly_chart")}


def _view(root):
    return loader.load_run(loader.list_runs(root)[0].path)


def test_a_t4_run_loads_cleanly(linear_run_root):
    root, *_ = linear_run_root
    view = _view(root)
    assert view.schema_version == contract.OUTPUT_SCHEMA_VERSION
    assert view.record_issues == () and view.legacy_gaps == ()
    roles = dict(zip(view.models["name"], view.models["role"], strict=True))
    assert roles["ols"] == "diagnostic" and roles["ridge"] == "candidate"
    eligible = dict(zip(view.models["name"], view.models["eligible_as_candidate"], strict=True))
    assert eligible["ridge"] and not eligible["ols"]


def test_the_ols_diagnostic_is_a_labelled_comparator(linear_run_root):
    root, *_ = linear_run_root
    view = _view(root)
    kinds = {c.name: c.kind for c in evaluation.comparators(view.record, view.models)}
    assert kinds == {
        "zero": evaluation.NULL_BASELINE,
        "per_symbol_mean": evaluation.SELECTION_CONTROL,
        "ols": evaluation.DIAGNOSTIC_COMPARATOR,
    }


def test_the_linear_models_page_shows_selection_coefficients_and_clark_west(
    linear_run_root, monkeypatch
):
    root, *_ = linear_run_root
    app = _app(root, monkeypatch, PAGE)
    assert not app.exception, [e.value for e in app.exception]
    assert app.selectbox(key="linear_model").value == "ridge"  # candidates first, run order
    assert {"linear_penalty", "linear_scores", "linear_coefficients"} <= set(_charts(app))
    stability = next(
        f.value for f in app.dataframe if "sign consistency (majority share)" in f.value.columns
    )
    assert set(stability["feature"]) == {"feature_a", "feature_b"}
    assert any("not product evidence" in w.value for w in app.warning)
    app.selectbox(key="linear_model").set_value("ols").run()
    assert not app.exception
    assert "linear_penalty" not in _charts(app)  # OLS has no selection
    labels = [m.label for m in app.metric]
    assert any("Clark-West" in label and "nested: ols vs zero" in label for label in labels)
    for name in ("ridge", "lasso", "elastic_net", "ols"):
        app.selectbox(key="linear_model").set_value(name).run()
        assert not app.exception, name


def test_candidates_are_the_default_where_models_are_compared(linear_run_root, monkeypatch):
    root, *_ = linear_run_root
    app = _app(root, monkeypatch)
    assert {"ridge", "lasso", "elastic_net", "ols"} <= set(app.sidebar.multiselect[0].value)
    app.switch_page("page_scripts/forecast_error.py").run()
    assert app.selectbox(key="gw_model").value.startswith("ridge")
    comparators = app.multiselect(key="gw_comparators").options
    assert "ols: diagnostic comparator (unpenalized OLS, not a null baseline)" in comparators


def test_candidate_identity_is_distinct_and_legible():
    fixed = dict(style.KNOWN_MODEL_COLOURS)
    assert len(set(fixed.values())) == len(fixed)  # no two known models share a colour
    reserved = set(style.OKABE_ITO) | set(style.CATEGORY_PALETTE)
    for name, colour in theme.CANDIDATE_COLOURS.items():
        assert colour not in reserved, name
        assert theme.contrast_ratio(colour, theme.COLOURS["surface"]) >= 4.5, name
        assert style.model_colour(name) == colour
    assert style.role_marker("candidate")[0] != style.role_marker("null")[0]
    assert style.role_marker("diagnostic")[0] not in {
        style.role_marker(r)[0] for r in ("null", "control", "canary", "candidate")
    }


# --- stored-row invariants of 1.2 tables ---


def _variant(run, tmp_path, *, record=None, tables=None):
    record = copy.deepcopy(run.record if record is None else record)
    record["run_id"] = VARIANT
    variant = dataclasses.replace(
        run, run_id=VARIANT, record=record, tables={**run.tables, **(tables or {})}
    )
    artifacts.write_run(variant, tmp_path)
    return tmp_path


def test_inner_selection_reaching_the_outer_test_period_blocks_the_run(linear_run_root, tmp_path):
    _, run, _, folds, _ = linear_run_root
    selection = run.tables["selection"].copy()
    first = selection.index[selection["fold_id"] == folds[0].fold_id][0]
    selection.loc[first, "max_inner_label_end"] = folds[0].test_start
    root = _variant(run, tmp_path, tables={"selection": contract.conform("selection", selection)})
    view = _view(root)
    assert any("selection rows have inner labels" in issue for issue in view.record_issues)


def test_a_fold_scoring_after_the_evaluation_asof_blocks_the_run(linear_run_root, tmp_path):
    _, run, *_ = linear_run_root
    record = copy.deepcopy(run.record)
    record["lockbox_evidence"]["max_date"] = (
        pd.Timestamp(record["lockbox_evidence"]["max_date"]) - pd.Timedelta(days=5)
    ).isoformat()
    view = _view(_variant(run, tmp_path, record=record))
    assert any("score_available_at disagrees with evaluation_asof" in i for i in view.record_issues)


def test_selection_dates_are_part_of_the_lockbox_scan(linear_run_root, tmp_path):
    _, run, *_ = linear_run_root
    selection = run.tables["selection"].copy()
    holdout = pd.Timestamp(run.record["spec"]["lockbox"]["holdout_start"])
    selection.loc[selection.index[0], "inner_test_end_exclusive"] = holdout
    root = _variant(run, tmp_path, tables={"selection": contract.conform("selection", selection)})
    with pytest.raises(loader.RunRefused) as refused:
        _view(root)
    assert refused.value.reason == "lockbox_dates_present"
