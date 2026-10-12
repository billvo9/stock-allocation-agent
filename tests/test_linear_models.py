"""
T4 plumbing around the linear core: nested purged selection inside
each outer training fold, the single-call outer refit, the coefficient and
selection tables, Clark-West only for registered nested pairs, the label
rules for comparators, and the pre-training evaluation as-of check. Most
tests replace the estimator and selector with statistics-free doubles; the
real core is exercised in test_linear_estimator.py and test_linear_core.py.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from stock_agent.features.training import LabelSpec
from stock_agent.model_diagnostics import contract, linear, nested, runner, scoring
from stock_agent.model_validation.audit import LeakageError

FEATURES = ["feature_a", "feature_b"]


def _check(run, name, model=None):
    checks = run.tables["checks"]
    rows = checks[(checks["check"] == name) & (checks["model"].fillna("") == (model or ""))]
    assert len(rows) == 1, (name, model, len(rows))
    return rows.iloc[0]


# --- inner folds ---


def test_inner_folds_live_inside_each_outer_fold_s_training_rows(linear_run_root):
    _, _, labeled, folds, design = linear_run_root
    ends = pd.to_datetime(labeled["target_end_date"], utc=True)
    for outer in folds:
        inner = nested.build_inner_folds(labeled, outer, design)
        assert len(inner) == design.n_windows
        for fold in inner:
            rows = np.concatenate([fold.train_rows, fold.test_rows])
            assert np.isin(rows, outer.train_rows).all()
            assert (ends.iloc[rows] < outer.test_start).all()
            assert (ends.iloc[fold.train_rows] < fold.test_start).all()  # inner purge
            assert fold.test_end <= outer.test_start


def test_an_inner_fold_reaching_the_outer_test_period_is_refused(linear_run_root):
    _, _, labeled, folds, design = linear_run_root
    later = nested.build_inner_folds(labeled, folds[-1], design)
    with pytest.raises(LeakageError):
        nested.verify_inner_folds(labeled, folds[0], later)


def test_an_inner_label_ending_exactly_at_the_outer_test_start_is_refused(linear_run_root):
    _, _, labeled, folds, design = linear_run_root
    outer = folds[-1]
    inner = nested.build_inner_folds(labeled, outer, design)
    last = inner[-1].test_rows[-1]
    sessions = labeled["date"].drop_duplicates().sort_values()
    before = sessions[sessions < outer.test_start].iloc[-1]
    edited = labeled.copy()
    edited.loc[edited.index[last], "target_end_date"] = before
    nested.verify_inner_folds(edited, outer, inner)  # one session earlier is allowed
    edited.loc[edited.index[last], "target_end_date"] = outer.test_start
    with pytest.raises(LeakageError, match="at or after the outer test start"):
        nested.verify_inner_folds(edited, outer, inner)


def test_inner_folds_respect_the_outer_train_start(linear_run_root):
    _, _, labeled, folds, design = linear_run_root
    floor = labeled["date"].drop_duplicates().sort_values().iloc[10]
    last = folds[-1]
    kept = (labeled["date"].iloc[last.train_rows] >= floor).to_numpy()
    floored = dataclasses.replace(last, train_start=floor, train_rows=last.train_rows[kept])
    unfloored = nested.build_inner_folds(labeled, last, design)
    with pytest.raises(LeakageError):  # rows before the floor are not this fold's training rows
        nested.verify_inner_folds(labeled, floored, unfloored)
    for fold in nested.build_inner_folds(labeled, floored, design):
        assert (labeled["date"].iloc[fold.train_rows] >= floor).all()


def test_too_little_history_for_nested_selection_is_an_error(linear_run_root):
    _, _, labeled, folds, _ = linear_run_root
    with pytest.raises(ValueError, match="nested selection needs"):
        nested.build_inner_folds(labeled, folds[0], nested.InnerDesign(min_train_sessions=10_000))


# --- selection and refit ---


def test_each_outer_fold_is_refit_once_with_its_chosen_scalars(linear_run_root, linear_doubles):
    _, _, labeled, folds, design = linear_run_root
    double, _ = linear_doubles
    spec = next(
        s
        for s in nested.linear_models(
            LabelSpec(5, 1, benchmark="equal_weight"), FEATURES, design, estimator=double
        )
        if s.name == "lasso"
    )
    grid = spec.params["selection"]["grid"]
    picks = iter([3, 0, 5, 1])

    def selector(grid, inner_predictions, *, target_column="target_return"):
        return linear.Selection(next(picks), tuple(0.0 for _ in grid), tuple(1 for _ in grid))

    frame = labeled.assign(control_symbol_code=0, control_session_index=0)
    inner = {f.fold_id: nested.build_inner_folds(frame, f, design) for f in folds}
    result = nested.run_nested(
        frame, folds, spec, inner, selector=selector, lockbox_start=folds[0].holdout_start
    )
    assert [e.penalty for e in result.estimators] == [grid[i]["penalty"] for i in (3, 0, 5, 1)]
    assert all(set(vars(e)) >= {"method", "penalty", "tol", "max_iter"} for e in result.estimators)
    assert not any(
        isinstance(v, (list, tuple, dict))
        for e in result.estimators
        for v in vars(e).values()
        if not isinstance(v, np.ndarray)
    )
    assert len(result.estimators) == len(folds)
    chosen = pd.DataFrame(result.selection_rows).query("chosen")
    assert chosen.groupby("fold_id").size().eq(1).all()


def test_the_outer_refit_is_one_harness_call_over_every_fold(
    linear_run_root, linear_doubles, monkeypatch
):
    _, _, labeled, folds, design = linear_run_root
    double, selector = linear_doubles
    spec = next(
        s
        for s in nested.linear_models(
            LabelSpec(5, 1, benchmark="equal_weight"), FEATURES, design, estimator=double
        )
        if s.name == "ridge"
    )
    calls = []
    original = nested.run_walk_forward

    def recording(frame, folds_given, **kwargs):
        calls.append(tuple(folds_given))
        return original(frame, folds_given, **kwargs)

    monkeypatch.setattr(nested, "run_walk_forward", recording)
    frame = labeled.assign(control_symbol_code=0, control_session_index=0)
    inner = {f.fold_id: nested.build_inner_folds(frame, f, design) for f in folds}
    nested.run_nested(
        frame, folds, spec, inner, selector=selector, lockbox_start=folds[0].holdout_start
    )
    grid = spec.params["selection"]["grid"]
    assert len(calls) == len(grid) * len(folds) + 1
    for k, outer in enumerate(folds):  # one call per grid point on that fold's inner folds
        assert calls[k * len(grid) : (k + 1) * len(grid)] == [inner[outer.fold_id]] * len(grid)
    assert calls[-1] == tuple(folds)  # every outer fold, in one call


def test_a_selector_must_score_every_grid_point(linear_run_root, linear_doubles):
    _, _, labeled, folds, design = linear_run_root
    double, _ = linear_doubles
    spec = nested.linear_models(
        LabelSpec(5, 1, benchmark="equal_weight"), FEATURES, design, estimator=double
    )[0]

    def partial(grid, inner_predictions, *, target_column="target_return"):
        return linear.Selection(0, (0.0,), (1,))

    frame = labeled.assign(control_symbol_code=0, control_session_index=0)
    inner = {f.fold_id: nested.build_inner_folds(frame, f, design) for f in folds}
    with pytest.raises(ValueError, match="every grid point"):
        nested.run_nested(
            frame, folds, spec, inner, selector=partial, lockbox_start=folds[0].holdout_start
        )


def test_selection_and_coefficient_tables_are_complete_and_training_scope(linear_run_root):
    _, run, _, folds, _ = linear_run_root
    selection = run.tables["selection"]
    sizes = {"ridge": 6, "lasso": 7, "elastic_net": 21}
    for model, size in sizes.items():
        rows = selection[selection["model"] == model]
        assert len(rows) == size * len(folds)
        assert rows.groupby("fold_id")["chosen"].sum().eq(1).all()
    starts = {f.fold_id: f.test_start for f in folds}
    assert (selection["max_inner_label_end"] < selection["fold_id"].map(starts)).all()
    assert set(selection["scope"]) == {"training"}
    coefficients = run.tables["coefficients"]
    assert set(coefficients["model"]) == {"ridge", "lasso", "elastic_net", "ols"}
    per_fold = coefficients.groupby(["model", "fold_id"])["feature"].apply(sorted)
    assert all(features == ["__intercept__", *sorted(FEATURES)] for features in per_fold)
    assert set(coefficients["scope"]) == {"training"}
    assert (coefficients.loc[coefficients["model"] == "ols", "alpha"] == 0).all()


def test_candidates_run_nested_selection_inside_every_null_draw(linear_run_root):
    _, run, _, _, _ = linear_run_root
    draws = run.tables["null_draws"]
    assert (
        len(draws[(draws["model"] == "ridge") & (draws["null_kind"] == "block_permutation")]) == 2
    )
    metrics = run.tables["metrics"]
    assert (
        (metrics["model"] == "ridge") & metrics["metric"].str.startswith("rank_ic_minus_stale_")
    ).any()


def test_selection_is_rerun_on_the_inputs_of_every_null_draw(linear_run_root, linear_doubles):
    _, _, labeled, folds, design = linear_run_root
    double, _ = linear_doubles
    spec = LabelSpec(5, 1, benchmark="equal_weight")
    ridge = next(
        m
        for m in nested.linear_models(spec, FEATURES, design, estimator=double)
        if m.name == "ridge"
    )
    seen = []

    def recording(grid, inner_predictions, *, target_column="target_return"):
        first = inner_predictions[0]
        window = (first["date"].min(), first["date"].max())
        signature = tuple(np.round(first["prediction"].to_numpy(), 15))
        seen.append((window, signature))
        return linear.select_hyperparameters(grid, inner_predictions)  # a data-driven choice

    draws, lags = 2, (15,)
    runner.run_diagnostics(
        labeled,
        folds,
        universe=list("ABCD"),
        label_spec=spec,
        diagnostic_features=FEATURES,
        provenance={"code": {}, "environment": {}},
        started_at=datetime(2026, 10, 10, tzinfo=UTC),
        models=[*runner.default_models(score_feature="feature_a"), ridge],
        config=runner.DiagnosticsConfig(
            baselines=("zero", "per_symbol_mean"),
            permutation_draws=draws,
            permutation_block_length=10,
            permutation_models=("ridge",),
            stale_lags=lags,
            stale_models=("ridge",),
            calibration_reps=4,
            calibration_bootstrap_reps=3,
            bootstrap_reps=9,
        ),
        lockbox_start=folds[0].holdout_start,
        selector=recording,
    )
    # Once per outer fold in the main run, in each permutation draw and each stale lag.
    assert len(seen) == len(folds) * (1 + draws + len(lags))
    by_fold = {}
    for window, signature in seen:
        by_fold.setdefault(window, []).append(signature)
    assert len(by_fold) == len(folds)
    for signatures in by_fold.values():  # each draw selected on its own (changed) inputs
        assert len(set(signatures)) == 1 + draws + len(lags)


def test_the_grid_and_selection_design_are_part_of_the_variant(linear_doubles):
    double, _ = linear_doubles
    spec = LabelSpec(5, 1, benchmark="equal_weight")
    small = nested.linear_models(spec, FEATURES, nested.InnerDesign(n_windows=3), estimator=double)
    default = nested.linear_models(spec, FEATURES, estimator=double)
    raw = nested.linear_models(LabelSpec(5, 1), FEATURES, estimator=double)
    ids = [{s.name: s.describe()["variant_id"] for s in specs} for specs in (small, default, raw)]
    assert ids[0]["ridge"] != ids[1]["ridge"] != ids[2]["ridge"]
    assert default[0].describe()["preprocessing"] == nested.PREPROCESSING
    assert default[-1].role == contract.DIAGNOSTIC_ROLE and default[-1].name == "ols"


# --- comparisons and Clark-West ---


def test_clark_west_rows_exist_only_for_registered_nested_pairs(linear_run_root):
    _, run, _, _, _ = linear_run_root
    metrics = run.tables["metrics"]
    rows = metrics[metrics["metric"].str.startswith("clark_west_vs_")]
    assert set(zip(rows["model"], rows["metric"], strict=True)) == {("ols", "clark_west_vs_zero")}
    assert set(rows["inference_method"]) == {
        "newey_west",
        "fold_block_t",
        "circular_block_bootstrap",
    }
    info = _check(run, "clark_west_one_sided", "ols")
    assert info["severity"] == "info"


def test_clark_west_is_the_mean_adjusted_loss_difference_per_date():
    dates = pd.to_datetime(["2024-01-02"] * 2 + ["2024-01-03"] * 2, utc=True)
    base = pd.DataFrame(
        {
            "fold_id": 0,
            "date": dates,
            "symbol": ["A", "B"] * 2,
            "target_return": [0.1, -0.1, 0.2, 0.0],
        }
    )
    model = base.assign(prediction=[0.05, -0.02, 0.1, 0.03])
    baseline = base.assign(prediction=0.0)
    ctx = scoring.build_context(model, block_length=1, bootstrap_reps=9, seed=0, hac_lag=0)
    nw = next(r for r in scoring.clark_west_vs(model, baseline, ctx) if r.method == "newey_west")
    y, f, b = base["target_return"].to_numpy(), model["prediction"].to_numpy(), 0.0
    per_obs = (y - b) ** 2 - ((y - f) ** 2 - (b - f) ** 2)
    np.testing.assert_allclose(per_obs, 2 * (y - b) * (f - b))
    expected = np.mean([per_obs[:2].mean(), per_obs[2:].mean()])
    assert nw.estimate == pytest.approx(expected)


def test_clark_west_weighs_each_date_equally_whatever_its_row_count():
    dates = pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"], utc=True)
    base = pd.DataFrame(
        {
            "fold_id": 0,
            "date": dates,
            "symbol": ["A", "B", "C", "A"],
            "target_return": [0.1, -0.1, 0.05, 0.2],
        }
    )
    model = base.assign(prediction=[0.05, -0.02, 0.01, 0.1])
    baseline = base.assign(prediction=0.0)
    ctx = scoring.build_context(model, block_length=1, bootstrap_reps=9, seed=0, hac_lag=0)
    nw = next(r for r in scoring.clark_west_vs(model, baseline, ctx) if r.method == "newey_west")
    per_obs = 2 * base["target_return"].to_numpy() * model["prediction"].to_numpy()
    by_date = np.mean([per_obs[:3].mean(), per_obs[3:].mean()])
    assert nw.estimate == pytest.approx(by_date)
    assert nw.estimate != pytest.approx(per_obs.mean())  # not the pooled-row mean


@pytest.mark.parametrize(
    ("baselines", "nested_pairs", "match"),
    [
        (("zero", "pooled_mean"), (), "identically zero"),
        (("zero",), (("ridge", "zero"),), "registered forecast"),
        (("zero",), (("ols", "per_symbol_mean"),), "registered baseline"),
    ],
)
def test_comparisons_that_do_not_fit_the_label_or_nesting_are_refused(
    linear_run_root, linear_doubles, baselines, nested_pairs, match
):
    _, _, labeled, folds, design = linear_run_root
    double, selector = linear_doubles
    spec = LabelSpec(5, 1, benchmark="equal_weight")
    models = [
        *runner.default_models(score_feature="feature_a"),
        *nested.linear_models(spec, FEATURES, design, estimator=double),
    ]
    models = [
        m if m.name != "ridge" else dataclasses.replace(m, output_kind="score") for m in models
    ]
    config = runner.DiagnosticsConfig(baselines=baselines, nested_comparisons=nested_pairs)
    with pytest.raises(ValueError, match=match):
        runner.run_diagnostics(
            labeled,
            folds,
            universe=list("ABCD"),
            label_spec=spec,
            diagnostic_features=FEATURES,
            provenance={"code": {}, "environment": {}},
            started_at=datetime(2026, 10, 10, tzinfo=UTC),
            models=models,
            config=config,
            lockbox_start=folds[0].holdout_start,
            selector=selector,
        )


# --- checks ---


def test_the_t4_checks_are_recorded(linear_run_root):
    _, run, _, folds, _ = linear_run_root
    assert _check(run, "evaluation_asof_matches_folds")["passed"]
    for model in ("ridge", "lasso", "elastic_net"):
        row = _check(run, "nested_selection_inside_training", model)
        assert row["passed"] and row["observed"] == len(folds)
        below = _check(run, "candidate_below_selection_control", model)
        assert below["severity"] == "research_warning"
        metrics = run.tables["metrics"]
        stored = metrics[
            (metrics["model"] == model)
            & (metrics["metric"] == "mse_improvement_vs_per_symbol_mean")
            & (metrics["inference_method"] == "fold_block_t")
            & metrics["fold_id"].isna()
        ]["estimate"].iloc[0]
        assert below["observed"] == stored and below["passed"] == (stored <= 0)
    for model in ("ridge", "lasso", "elastic_net", "ols"):
        assert _check(run, "selection_fits_converged", model)["passed"]
    assert _check(run, "excess_label_centred", "pooled_mean")["passed"]
    assert run.record["checks_summary"]["leakage_checks_failed"] == 0


def _run_with(labeled, folds, model, selector, **config):
    """A small excess-label run: the T2 set plus one linear model."""

    return runner.run_diagnostics(
        labeled,
        folds,
        universe=list("ABCD"),
        label_spec=LabelSpec(5, 1, benchmark="equal_weight"),
        diagnostic_features=FEATURES,
        provenance={"code": {}, "environment": {}},
        started_at=datetime(2026, 10, 10, tzinfo=UTC),
        models=[*runner.default_models(score_feature="feature_a"), model],
        config=runner.DiagnosticsConfig(
            **{
                "baselines": ("zero", "per_symbol_mean"),
                "permutation_models": (),
                "stale_models": (),
                "calibration_reps": 4,
                "calibration_bootstrap_reps": 3,
                "bootstrap_reps": 9,
                **config,
            }
        ),
        lockbox_start=folds[0].holdout_start,
        selector=selector,
    )


def _lasso(design, estimator):
    spec = LabelSpec(5, 1, benchmark="equal_weight")
    return next(
        m
        for m in nested.linear_models(spec, FEATURES, design, estimator=estimator)
        if m.name == "lasso"
    )


def test_a_fit_that_did_not_converge_fails_the_instrument_check(linear_run_root, linear_doubles):
    _, _, labeled, folds, design = linear_run_root
    double, selector = linear_doubles

    class StallsOffTheChosenPoint(double):
        """Converges at the chosen grid point (r = 1) and nowhere else: inner fits only."""

        def fit(self, X, y):
            super().fit(X, y)
            self.converged_ = self.penalty == 1.0
            return self

    run = _run_with(labeled, folds, _lasso(design, StallsOffTheChosenPoint), selector)
    row = _check(run, "selection_fits_converged", "lasso")
    grid = len(nested.LASSO_GRID)
    assert pd.notna(row["passed"]) and not row["passed"]
    assert row["severity"] == "instrument"
    assert row["observed"] == len(folds) * (grid - 1) * design.n_windows
    assert row["detail"] == f"{len(folds) * (1 + grid * design.n_windows)} fits (outer and inner)"


def test_a_fit_without_a_convergence_flag_leaves_the_check_undetermined(
    linear_run_root, linear_doubles
):
    _, _, labeled, folds, design = linear_run_root
    double, selector = linear_doubles

    class NoFlag(double):
        def fit(self, X, y):
            super().fit(X, y)
            del self.converged_
            return self

    run = _run_with(labeled, folds, _lasso(design, NoFlag), selector)
    row = _check(run, "selection_fits_converged", "lasso")
    assert pd.isna(row["passed"]) and pd.isna(row["observed"])
    assert row["detail"] == "no converged_ flag"


def test_a_recorded_inner_label_reaching_the_test_start_fails_the_leakage_check(
    linear_run_root, linear_doubles, monkeypatch
):
    _, _, labeled, folds, design = linear_run_root
    double, selector = linear_doubles
    original = nested.inner_bounds
    late = folds[1]

    def misrecorded(frame, inner):
        bounds = original(frame, inner)
        if inner[-1].test_end == late.test_start:  # that fold's inner design ends at its test
            bounds["max_inner_label_end"] = late.test_start
        return bounds

    monkeypatch.setattr(nested, "inner_bounds", misrecorded)
    run = _run_with(labeled, folds, _lasso(design, double), selector)
    row = _check(run, "nested_selection_inside_training", "lasso")
    assert pd.notna(row["passed"]) and not row["passed"]
    assert row["severity"] == "leakage"
    assert row["observed"] == len(folds) - 1


def test_training_never_starts_when_folds_score_after_the_evaluation_asof(
    linear_run_root, linear_doubles, monkeypatch
):
    _, _, labeled, folds, design = linear_run_root
    double, selector = linear_doubles
    earlier = pd.to_datetime(labeled["target_end_date"], utc=True).max() - pd.Timedelta(days=3)
    assert runner.asof_violations(labeled, folds, {"max_date": earlier.isoformat()}) == [
        folds[-1].fold_id
    ]
    original = runner.lockbox_evidence
    monkeypatch.setattr(
        runner,
        "lockbox_evidence",
        lambda *a, **k: {**original(*a, **k), "max_date": earlier.isoformat()},
    )
    built = []

    class Recording(double):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    spec = LabelSpec(5, 1, benchmark="equal_weight")
    with pytest.raises(LeakageError, match="evaluation as-of"):
        runner.run_diagnostics(
            labeled,
            folds,
            universe=list("ABCD"),
            label_spec=spec,
            diagnostic_features=FEATURES,
            provenance={"code": {}, "environment": {}},
            started_at=datetime(2026, 10, 10, tzinfo=UTC),
            models=nested.linear_models(spec, FEATURES, design, estimator=Recording),
            config=runner.DiagnosticsConfig(baselines=("ols",)),
            lockbox_start=folds[0].holdout_start,
            selector=selector,
        )
    assert not built  # stopped before any estimator was created


def test_the_excess_label_run_records_its_label_and_stores_the_raw_return(linear_run_root):
    _, run, _, _, _ = linear_run_root
    label = run.record["spec"]["label"]
    assert label["label_id"].startswith("equal_weight_excess_forward_simple_return")
    assert label["benchmark"] == "equal_weight" and "availability_rule" in label
    inputs = run.tables["inputs"]
    assert "target_return_raw" in inputs.columns
    sums = inputs.dropna(subset=["target_return"]).groupby("date")["target_return"].sum()
    np.testing.assert_allclose(sums, 0.0, atol=1e-12)
    assert "target_return_raw" not in run.tables["predictions"].columns
    assert run.record["spec"]["output_schema_version"] == "1.2"


def test_comparators_and_nested_pairs_follow_the_label():
    excess = LabelSpec(20, 1, benchmark="equal_weight")
    raw = LabelSpec(20, 1)
    assert nested.comparison_settings(excess, with_linear=True) == {
        "baselines": ("zero", "per_symbol_mean", "ols"),
        "nested_comparisons": (("ols", "zero"),),
    }
    assert nested.comparison_settings(raw, with_linear=True) == {
        "baselines": ("zero", "pooled_mean", "per_symbol_mean", "ols"),
        "nested_comparisons": (("ols", "zero"), ("ols", "pooled_mean")),
    }
    assert nested.comparison_settings(excess, with_linear=False)["nested_comparisons"] == ()
    assert "pooled_mean" not in nested.comparison_settings(excess, with_linear=False)["baselines"]


def test_inner_folds_are_verified_as_they_are_built(linear_run_root, monkeypatch):
    # Fault injection: if the fold builder ever returned a later fold's inner folds,
    # build_inner_folds must refuse them rather than trust its own construction.
    _, _, labeled, folds, design = linear_run_root
    later = nested.build_inner_folds(labeled, folds[-1], design)
    monkeypatch.setattr(nested, "make_expanding_folds", lambda *a, **k: later)
    with pytest.raises(LeakageError):
        nested.build_inner_folds(labeled, folds[0], design)
