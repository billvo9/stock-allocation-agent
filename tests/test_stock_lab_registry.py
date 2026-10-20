"""
The frozen, registered Stock Lab model (stock_agent.stock_lab.registry and
frozen_model) and its write-once table in the research store. Synthetic and
fixture data only; nothing here reads a private store or the network.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from stock_agent.stock_lab import contracts, frozen_model, logistic, registry, synthetic
from stock_agent.stock_lab.contracts import DataClass, StableBand
from stock_agent.stock_lab.evaluation import EvaluationConfig, _frequency_state, build_panel
from stock_agent.stock_lab.evaluation import evaluate as run_evaluation
from stock_agent.stock_lab.inference import schema as inference_schema
from stock_agent.stock_lab.predict import _probabilities, chosen_half_life, fit_final
from stock_agent.stock_lab.reversal import after_streak
from stock_agent.stock_lab.store import DB_NAME, StockLabStore, StoreRefused

CREATED = pd.Timestamp("2026-10-16T22:00:00Z")
LATER = pd.Timestamp("2026-10-16T22:30:00Z")
CONFIG = EvaluationConfig(size_reps=8, sensitivity_bands=())
# Every setting a forecast reads, away from its default, so a load that ignored them shows.
UNUSUAL = EvaluationConfig(
    band=StableBand(0.004),
    alpha_prior=30.0,
    kappa=25.0,
    tilt_ridge=5.0,
    error_kappa=150.0,
    size_reps=8,
    sensitivity_bands=(),
)
CHALLENGERS = ("B0", "B2", "B3", "M1")
CONFIRMED = "fixture_confirmed_rule"
REAL_NAMES = dict(zip(synthetic.DEMO_IDS, ("AMD", "MU", "NVDA"), strict=True))


def evaluate(histories, config, *, data_class="synthetic", run_id="S-registry"):
    return run_evaluation(
        histories, config, data_class=data_class, run_id=run_id, created_at=CREATED
    )


@pytest.fixture(scope="module")
def histories():
    return dict(synthetic.demo_histories())


@pytest.fixture(scope="module")
def evaluation(histories):
    return evaluate(histories, CONFIG)


@pytest.fixture(scope="module")
def frozen(histories, evaluation):
    return registry.freeze(histories, evaluation, CONFIG, created_at=CREATED)


def _real(histories, end="2024-12-31"):
    out = {}
    for stock_id, bars in histories.items():
        moved = bars.copy()
        moved["session"] = pd.bdate_range(end=end, periods=len(bars), tz="UTC").as_unit("us")
        out[REAL_NAMES[stock_id]] = moved
    return out


REAL_CONFIG = EvaluationConfig(
    size_reps=8, sensitivity_bands=(), outcome_end=pd.Timestamp("2024-12-31", tz="UTC")
)


@pytest.fixture(scope="module")
def real_frozen(histories):
    real = _real(histories)
    record = evaluate(real, REAL_CONFIG, data_class="real", run_id="R-registry")
    return registry.freeze(real, record, REAL_CONFIG, created_at=CREATED)


def _confirmed(evaluation: dict, selected: str, rule: str = CONFIRMED) -> dict:
    """
    An evaluation record as a confirmed rule would write it, selecting `selected` on its own
    recorded evidence (the challenger beat B1; no step moved).
    """

    record = copy.deepcopy(evaluation)
    record["decision_rule"] = rule
    record["selection"].update(
        selected=selected,
        decision_rule=rule,
        decision_test="fold_block_t",
        beating_reference=[] if selected == "B1" else [selected],
        steps=[],
    )
    return record


def _reseal(record: dict, **changes) -> dict:
    """The record with `changes`, its parameters' hash and its id recomputed (a forged edit)."""

    edited = {**copy.deepcopy(record), **changes}
    edited["parameters_sha256"] = frozen_model.parameters_sha256(edited)
    edited["model_id"] = frozen_model.model_id(edited)
    return edited


def _last_rows(panel: pd.DataFrame) -> dict[str, int]:
    rows = {}
    for symbol in panel["symbol"].unique():
        mask = (panel["symbol"] == symbol).to_numpy() & panel["close"].notna().to_numpy()
        rows[symbol] = int(np.flatnonzero(mask)[-1])
    return rows


def _window_panel(histories, config, sessions=252):
    return build_panel({s: bars.iloc[-sessions:] for s, bars in histories.items()}, config)


# --- the artifact ---


def test_the_model_id_is_the_sha256_of_its_canonical_content(frozen):
    content = {key: value for key, value in frozen.items() if key != "model_id"}
    text = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert frozen["model_id"] == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert frozen["schema"] == registry.SCHEMA and registry.check(frozen) is frozen
    assert frozen_model.version(frozen) == frozen["model_id"][:16]
    assert json.loads(registry.canonical(frozen)) == frozen  # strict JSON: no NaN, no infinity


def test_freezing_is_deterministic_and_the_parameters_hash_spans_freezes(
    histories, evaluation, frozen
):
    again = registry.freeze(histories, evaluation, CONFIG, created_at=CREATED)
    assert again == frozen
    later = registry.freeze(histories, evaluation, CONFIG, created_at=LATER)
    assert later["model_id"] != frozen["model_id"]
    assert later["parameters_sha256"] == frozen["parameters_sha256"]
    other = registry.freeze(
        histories, evaluation, CONFIG, created_at=CREATED, state_rule="window_only"
    )
    assert other["parameters_sha256"] != frozen["parameters_sha256"]


def test_an_edited_record_is_refused_even_with_its_id_recomputed(frozen):
    edited = {**copy.deepcopy(frozen), "half_life": 21.0}
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.check(edited)
    assert caught.value.reason == "model_id_mismatch"
    edited["model_id"] = frozen_model.model_id(edited)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.check(edited)  # the parameters no longer match their hash
    assert caught.value.reason == "invalid_record"
    resealed = _reseal(frozen, half_life=21.0)
    assert registry.check(resealed)["model_id"] != frozen["model_id"]  # a new model, never the old


def _with_selection(record, **changes):
    return {**record["selection"], **changes}


def _cases(frozen):
    return {
        "extra field": ({**frozen, "extra": 1}, "invalid_record"),
        "schema version": ({**frozen, "schema_version": "0.9"}, "invalid_record"),
        "estimator": ({**frozen, "estimator": "B9"}, "invalid_record"),
        "benchmark": ({**frozen, "benchmark": "B0"}, "invalid_record"),
        "data class": ({**frozen, "data_class": "mixed"}, "invalid_record"),
        "state rule": ({**frozen, "state_rule": "latest"}, "unknown_state_rule"),
        "window": ({**frozen, "window_sessions": 253}, "invalid_record"),
        "created_at": ({**frozen, "created_at": "2026-10-16"}, "invalid_record"),
        "selection": (
            {**frozen, "selection": _with_selection(frozen, frozen="B3")},
            "invalid_record",
        ),
        "challenger without A7": (
            {
                **frozen,
                "estimator": "B3",
                "selection": _with_selection(
                    frozen, frozen="B3", evaluation_selected="B3", decision_rule=CONFIRMED
                ),
            },
            "a7_unresolved",
        ),
        "real names in a synthetic model": (
            {**frozen, "data_class": "real", "outcome_end": "2024-12-31"},
            "data_class_mismatch",
        ),
    }


@pytest.mark.parametrize(
    "case",
    [
        "extra field",
        "schema version",
        "estimator",
        "benchmark",
        "data class",
        "state rule",
        "window",
        "created_at",
        "selection",
        "challenger without A7",
        "real names in a synthetic model",
    ],
)
def test_the_record_check_refuses_what_freeze_never_writes(frozen, case):
    record, reason = _cases(frozen)[case]
    record = _reseal(record) if case != "extra field" else record
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.check(record)
    assert caught.value.reason == reason


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"outcome_end": "2025-06-30"}, "outcome_cutoff"),
        ({"outcome_end": None}, "outcome_cutoff"),
        ("late history", "protected_period"),
        ({"stocks": ["AMD", "DEMO_CALM", "NVDA"]}, "data_class_mismatch"),
        ({"data_class": "synthetic"}, "data_class_mismatch"),
        ({"state_rule": "window_only"}, "state_rule_not_approved"),
    ],
)
def test_a_real_record_never_reaches_past_development(real_frozen, change, reason):
    if change == "late history":
        data = copy.deepcopy(real_frozen["data"])
        data["histories"]["NVDA"]["last_session"] = "2025-01-02"
        change = {"data": data}
    if "stocks" in change:
        data = copy.deepcopy(real_frozen["data"])
        data["histories"]["DEMO_CALM"] = data["histories"].pop("MU")
        change = {**change, "data": data}
    record = _reseal(real_frozen, **change)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.check(record)
    assert caught.value.reason == reason


def test_it_records_everything_a_forecast_needs_to_be_reproduced(frozen, evaluation, histories):
    assert frozen["estimator"] == "B1" and frozen["benchmark"] == "B1"
    assert frozen["evaluation_run_id"] == evaluation["run_id"]
    assert frozen["evaluation_sha256"] == registry.evaluation_sha256(evaluation)
    assert frozen["config"] == CONFIG.as_dict()
    assert frozen["feature_version"] == CONFIG.as_dict()["feature_version"]
    assert frozen["half_life"] == chosen_half_life(evaluation)
    assert frozen["state_rule"] == registry.DEFAULT_STATE_RULE == "frozen_long_run"
    assert frozen["window_sessions"] == inference_schema.INFERENCE_SESSIONS == 252
    assert frozen["window_tail_weight"] == 2.0 ** (-252 / frozen["half_life"])
    assert frozen["stocks"] == sorted(histories)
    assert frozen["data"]["panel_sha256"] == evaluation["panel_sha256"]
    for symbol, bars in histories.items():
        info = frozen["data"]["histories"][symbol]
        assert info["first_session"] == str(bars["session"].iloc[0].date())
        assert info["last_session"] == "2024-12-31" and info["rows"] == len(bars)
        assert info["bars_sha256"] == evaluation["histories_sha256"][symbol]
    assert set(frozen["environment"]) == {"python", "numpy", "pandas", "scikit-learn"}
    assert frozen["environment"] == registry.environment()
    assert frozen["code"] is None and frozen["data"]["lineage"] is None


def test_the_contract_constants_match_the_rest_of_the_lab():
    assert frozen_model.WINDOW_SESSIONS == inference_schema.INFERENCE_SESSIONS
    assert frozen_model.DEVELOPMENT_END == str(contracts.DEVELOPMENT_END)
    # Decision A7 (2026-10-19): the fold-block t at 2.5% is the one confirmed rule.
    assert frozen_model.CONFIRMED_DECISION_RULES == {frozen_model.A7_DECISION_RULE}
    assert frozen_model.A7_DECISION_RULE == "fold_block_t_holm_one_sided_2.5pct_v1"


def test_the_record_contract_needs_only_the_standard_library():
    source = Path(frozen_model.__file__).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        for name in names:
            assert name.split(".")[0] in sys.stdlib_module_names, name


def test_a_b1_model_keeps_only_what_b1_needs(frozen):
    assert frozen["tilt"] is None and frozen["errors"] is None and frozen["m1"] is None
    assert set(frozen["long_run"]) == {"daily", "cumulative"}


def test_the_long_run_mix_is_b0_at_the_end_of_development(frozen, histories):
    panel = build_panel(histories, CONFIG)
    params_b0 = _frequency_state(
        panel, "category_code", registry.est.FrequencyParams(50.0, frozen["half_life"], 40.0)
    )[0]
    for symbol, row in _last_rows(panel).items():
        assert frozen["long_run"]["daily"][symbol] == params_b0[row].tolist()
        assert abs(sum(frozen["long_run"]["daily"][symbol]) - 1.0) < 1e-12


def test_the_development_reversal_tables_are_frozen_with_the_model(frozen, histories):
    panel = build_panel(histories, CONFIG)
    stock = panel[panel["symbol"] == "DEMO_SWING"].reset_index(drop=True)
    summary = after_streak(stock, 3, "decrease", CONFIG.band, outcome_end=None)
    stored = frozen["reversal"]["DEMO_SWING"]["decrease"]["3"]
    assert stored["definition"] == summary.definition and stored["events"] == summary.events
    assert set(frozen["reversal"]["DEMO_SWING"]) == {"increase", "decrease"}
    assert set(frozen["reversal"]["DEMO_SWING"]["increase"]) == {str(k) for k in range(1, 8)}


def test_the_drift_reference_comes_from_development_inputs(frozen):
    reference = frozen["drift_reference"]
    assert set(reference) == {"ewma_vol", "ret_20", "rel_dollar_volume", "band_over_vol"}
    for entry in reference.values():
        assert len(entry["expected"]) == len(entry["edges"]) - 1
        assert abs(sum(entry["expected"]) - 1.0) < 1e-9


def test_the_state_check_compares_windows_with_the_full_history(frozen):
    check = frozen["state_check"]
    assert check["role"].startswith("reported only")
    assert check["rows"] == 3 * 252 and check["window_sessions"] == 252
    assert check["first_session"] < check["last_session"] == "2024-12-31"
    assert 0 < check["scored_rows"] < check["rows"]  # the last session's outcome is censored
    assert set(check["rps"]) == {"full_history", "frozen_long_run", "window_only"}
    for rule in ("frozen_long_run", "window_only"):
        assert check["max_abs_difference"][rule] >= check["mean_abs_difference"][rule] > 0
        gap = check["rps"][rule] - check["rps"]["full_history"]
        assert check["rps_difference"][rule] == pytest.approx(gap, abs=1e-12)


# --- loading reproduces the evaluated model ---


@pytest.mark.parametrize("rule", registry.STATE_RULES)
def test_a_loaded_b1_model_reproduces_the_development_forecast(histories, evaluation, rule):
    record = registry.freeze(histories, evaluation, CONFIG, created_at=CREATED, state_rule=rule)
    model = registry.load(record)
    panel = build_panel(histories, CONFIG)
    final = fit_final(panel, CONFIG, evaluation)
    for symbol, row in _last_rows(panel).items():
        daily, cumulative, usual = _probabilities(panel, row, final, "B1")
        got = registry.probabilities(panel, symbol, model, "B1")
        assert np.array_equal(got.daily, daily) and np.array_equal(got.cumulative, cumulative)
        assert np.array_equal(got.usual, usual)


@pytest.mark.parametrize("selected", ("B1", "B2", "B3"))
def test_a_model_with_unusual_settings_round_trips_on_its_own_settings(
    histories, selected, monkeypatch
):
    monkeypatch.setattr(frozen_model, "CONFIRMED_DECISION_RULES", frozenset({CONFIRMED}))
    record = _confirmed(evaluate(histories, UNUSUAL, run_id="S-unusual"), selected)
    frozen = registry.freeze(histories, record, UNUSUAL, created_at=CREATED)
    model = registry.load(frozen)
    assert model.final.config.as_dict() == {**UNUSUAL.as_dict(), **_defaults_not_read()}
    panel = build_panel(histories, UNUSUAL)
    final = fit_final(panel, UNUSUAL, record)
    for symbol, row in _last_rows(panel).items():
        daily, cumulative, _ = _probabilities(panel, row, final, selected)
        got = registry.probabilities(panel, symbol, model, selected)
        assert np.allclose(got.daily, daily, rtol=0, atol=1e-12)
        assert np.allclose(got.cumulative, cumulative, rtol=0, atol=1e-12)


def _defaults_not_read() -> dict:
    """The settings a forecast never reads come back at their defaults (registry._config)."""

    defaults = EvaluationConfig().as_dict()
    read = {"band", "cumulative_band", "alpha_prior", "kappa", "tilt_ridge", "error_kappa",
            "outcome_end", "floor", "nw_lag", "block", "bootstrap_reps", "bootstrap_seed",
            "alpha", "m1_c_grid", "m1_inputs", "feature_version"}  # fmt: skip
    return {key: value for key, value in defaults.items() if key not in read}


@pytest.mark.parametrize("selected", CHALLENGERS)
def test_a_challenger_round_trips_once_a7_is_confirmed(
    histories, evaluation, selected, monkeypatch
):
    monkeypatch.setattr(frozen_model, "CONFIRMED_DECISION_RULES", frozenset({CONFIRMED}))
    record = _confirmed(evaluation, selected)
    frozen = registry.freeze(histories, record, CONFIG, created_at=CREATED)
    assert frozen["estimator"] == selected and frozen["selection"]["decision_rule"] == CONFIRMED
    model = registry.load(frozen)
    panel = build_panel(histories, CONFIG)
    final = fit_final(panel, CONFIG, record)
    for symbol, row in _last_rows(panel).items():
        for estimator in (selected, "B1"):
            daily, cumulative, _ = _probabilities(panel, row, final, estimator)
            got = registry.probabilities(panel, symbol, model, estimator)
            assert np.allclose(got.daily, daily, rtol=0, atol=1e-12)
            assert np.allclose(got.cumulative, cumulative, rtol=0, atol=1e-12)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.probabilities(panel, "DEMO_CALM", model, "B0" if selected != "B0" else "B2")
    assert caught.value.reason == "estimator_not_frozen"


def test_a_round_trip_that_differs_is_refused(histories, evaluation, monkeypatch):
    honest = registry.probabilities

    def drifting(*args, **kwargs):
        got = honest(*args, **kwargs)
        return registry.Probabilities(got.daily + 1e-9, got.cumulative, got.usual, got.row)

    monkeypatch.setattr(registry, "probabilities", drifting)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, evaluation, CONFIG, created_at=CREATED)
    assert caught.value.reason == "round_trip_failed"


@pytest.mark.parametrize("package", ["python", "numpy", "pandas", "scikit-learn"])
def test_loading_refuses_another_environment(frozen, package, monkeypatch):
    other = {**frozen["environment"], package: "0.0"}
    monkeypatch.setattr(registry, "environment", lambda: other)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.load(frozen)
    assert caught.value.reason == "environment_mismatch" and package in caught.value.detail


def test_loading_refuses_another_feature_version(frozen, monkeypatch):
    monkeypatch.setattr(registry, "FEATURE_VERSION", "stock_lab_features_v0")
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.load(frozen)
    assert caught.value.reason == "feature_version_mismatch"


def test_a_stock_outside_the_frozen_model_or_without_a_close_is_refused(frozen, histories):
    model = registry.load(frozen)
    panel = build_panel({"AAPL": histories["DEMO_CALM"]}, CONFIG)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.probabilities(panel, "AAPL", model, "B1")
    assert caught.value.reason == "stock_not_in_model"
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.probabilities(panel, "DEMO_CALM", model, "B1")  # not in this panel
    assert caught.value.reason == "no_close"


# --- the state rules on an inference-length window ---


def _by_hand_b1(codes: np.ndarray, half_life: float, kappa: float, prior: np.ndarray):
    """B1 on one window, from its definition: lambda^k weights, k = 0 at the last session."""

    lam = 2.0 ** (-1.0 / half_life)
    ages = np.arange(len(codes))[::-1]
    known = ~np.isnan(codes)
    weights = np.where(known, lam**ages, 0.0)
    sums = np.array([weights[known & (codes == c)].sum() for c in range(3)])
    return (sums + kappa * prior) / (weights.sum() + kappa)


def test_the_two_state_rules_differ_only_in_the_long_run_mix(histories, evaluation):
    long_run = registry.load(registry.freeze(histories, evaluation, CONFIG, created_at=CREATED))
    one_year = registry.load(
        registry.freeze(histories, evaluation, CONFIG, created_at=CREATED, state_rule="window_only")
    )
    full = build_panel(histories, CONFIG)
    window = _window_panel(histories, CONFIG)
    longer = _window_panel(histories, CONFIG, sessions=400)
    final = fit_final(full, CONFIG, evaluation)
    kappa = CONFIG.kappa
    params = registry.est.FrequencyParams(CONFIG.alpha_prior, long_run.final.half_life, kappa)
    window_b0 = _frequency_state(window, "category_code", params)[0]
    for symbol, row in _last_rows(window).items():
        frozen_mix = long_run.long_run["daily"][symbol]
        kept = registry.probabilities(window, symbol, long_run, "B1")
        assert np.array_equal(kept.usual, frozen_mix)
        codes = window.loc[window["symbol"] == symbol, "category_code"]
        expected = _by_hand_b1(codes.to_numpy(dtype="float64", na_value=np.nan),
                               long_run.final.half_life, kappa, frozen_mix)  # fmt: skip
        assert np.allclose(kept.daily[0], expected, rtol=0, atol=1e-12)
        own = registry.probabilities(window, symbol, one_year, "B1")
        assert np.array_equal(own.usual, window_b0[row])
        assert not np.allclose(own.usual, kept.usual)
        everything = _probabilities(full, _last_rows(full)[symbol], final, "B1")[0][0]
        assert not np.allclose(kept.daily[0], everything)
        assert not np.allclose(own.daily[0], everything)
        assert np.array_equal(registry.probabilities(longer, symbol, long_run, "B1").usual,
                              frozen_mix)  # fmt: skip


def test_window_only_needs_every_model_stocks_window(histories, evaluation):
    one_year = registry.load(
        registry.freeze(histories, evaluation, CONFIG, created_at=CREATED, state_rule="window_only")
    )
    partial = _window_panel({k: v for k, v in histories.items() if k != "DEMO_CALM"}, CONFIG)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.probabilities(partial, "DEMO_SWING", one_year, "B1")
    assert caught.value.reason == "panel_stocks_mismatch"
    long_run = registry.load(registry.freeze(histories, evaluation, CONFIG, created_at=CREATED))
    registry.probabilities(partial, "DEMO_SWING", long_run, "B1")  # each stock on its own


# --- M1 rebuilt from its coefficients ---


def test_a_two_class_m1_matches_its_fit_without_overflow():
    rng = np.random.default_rng(3)
    n = 400
    inputs = pd.DataFrame(rng.normal(size=(n, len(logistic.M1_INPUTS))), columns=logistic.M1_INPUTS)
    b3 = rng.dirichlet([5, 5, 5], size=n)
    symbols = list(rng.choice(["DEMO_A", "DEMO_B"], size=n))
    prep = logistic.prepare(inputs, b3, symbols)
    x = logistic._design(inputs, b3, symbols, prep)
    y = np.where(x[:, 0] + rng.normal(size=n) > 0, 2, 0)  # Decrease or Increase only
    fitted = LogisticRegression(C=1.0, max_iter=1000).fit(x, y)
    rebuilt = registry.FrozenM1(prep, fitted.classes_, fitted.coef_, fitted.intercept_, 1.0)
    expected = np.zeros((n, 3))
    expected[:, fitted.classes_] = fitted.predict_proba(x)
    assert np.allclose(rebuilt.predict(inputs, b3, symbols), expected, rtol=0, atol=1e-12)
    extreme = registry.FrozenM1(
        prep, fitted.classes_, fitted.coef_ * 1e4, fitted.intercept_ * 1e4, 1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = extreme.predict(inputs, b3, symbols)
    assert np.isfinite(values).all() and np.allclose(values.sum(axis=1), 1.0)


# --- what may be frozen ---


@pytest.mark.parametrize("selected", CHALLENGERS)
def test_an_evaluation_under_the_earlier_rule_freezes_only_b1(histories, evaluation, selected):
    record = copy.deepcopy(evaluation)
    del record["decision_rule"]  # an evaluation from before A7 (schema 1.2): Newey-West at 5%
    record["selection"]["selected"] = selected  # as that earlier rule might select
    frozen = registry.freeze(histories, record, CONFIG, created_at=CREATED)
    assert frozen["estimator"] == "B1"
    assert frozen["selection"]["evaluation_selected"] == selected
    assert frozen["selection"]["decision_rule"] is None
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, record, CONFIG, created_at=CREATED, estimator=selected)
    assert caught.value.reason == "a7_unresolved"


def test_an_evaluation_naming_an_unconfirmed_rule_still_freezes_only_b1(histories, evaluation):
    # No monkeypatch: the rule an evaluation names counts only once the owner confirms it.
    record = _confirmed(evaluation, "B3")
    frozen = registry.freeze(histories, record, CONFIG, created_at=CREATED)
    assert frozen["estimator"] == "B1" and frozen["selection"]["decision_rule"] is None
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, record, CONFIG, created_at=CREATED, estimator="B3")
    assert caught.value.reason == "a7_unresolved"


def test_a_confirmed_selection_cannot_be_overridden(histories, evaluation, monkeypatch):
    monkeypatch.setattr(frozen_model, "CONFIRMED_DECISION_RULES", frozenset({CONFIRMED}))
    record = _confirmed(evaluation, "B3")
    for requested in ("B1", "B2"):
        with pytest.raises(registry.RegistryRefused) as caught:
            registry.freeze(histories, record, CONFIG, created_at=CREATED, estimator=requested)
        assert caught.value.reason == "not_the_selection"


def _one_return_changed(histories, column="close"):
    changed = {k: v.copy() for k, v in histories.items()}
    bars = changed["DEMO_CALM"]
    bars.loc[100, column] = bars.loc[100, column] * 1.5  # a different return or volume
    return changed


def test_the_evaluation_must_describe_the_same_data_and_settings(histories, evaluation):
    cases = {
        "a stock removed": ({k: v for k, v in histories.items() if k != "DEMO_CALM"}, CONFIG),
        "a return changed": (_one_return_changed(histories), CONFIG),
        "a volume changed": (_one_return_changed(histories, "volume"), CONFIG),
    }
    for changed, config in cases.values():
        with pytest.raises(registry.RegistryRefused) as caught:
            registry.freeze(changed, evaluation, config, created_at=CREATED)
        assert caught.value.reason == "histories_mismatch"
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, evaluation, EvaluationConfig(kappa=20.0), created_at=CREATED)
    assert caught.value.reason == "config_mismatch"
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, evaluation, CONFIG, created_at=CREATED, state_rule="other")
    assert caught.value.reason == "unknown_state_rule"
    old = {k: v for k, v in evaluation.items() if k != "histories_sha256"}
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, old, CONFIG, created_at=CREATED)
    assert caught.value.reason == "evaluation_without_history_hashes"


def test_freezing_refuses_names_of_the_other_data_class(histories, evaluation):
    real = _real(histories)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(real, evaluation, CONFIG, created_at=CREATED)
    assert caught.value.reason == "data_class_mismatch"


def test_a_real_model_is_frozen_only_under_the_owners_state_rule(histories):
    real = _real(histories)
    record = evaluate(real, REAL_CONFIG, data_class="real", run_id="R-rule")
    frozen = registry.freeze(real, record, REAL_CONFIG, created_at=CREATED)
    assert frozen["state_rule"] == "frozen_long_run" == frozen_model.REAL_STATE_RULES[0]
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(real, record, REAL_CONFIG, created_at=CREATED, state_rule="window_only")
    assert caught.value.reason == "state_rule_not_approved"


def test_real_freezing_refuses_any_protected_period_session(histories):
    real = _real(histories, end="2025-01-03")
    record = evaluate(real, REAL_CONFIG, data_class="real", run_id="R-x")
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(real, record, REAL_CONFIG, created_at=CREATED)
    assert caught.value.reason == "protected_period"


@pytest.mark.parametrize("outcome_end", [None, "2025-06-30", "2024-06-28"])
def test_real_freezing_needs_the_development_outcome_cutoff(histories, outcome_end):
    real = _real(histories)
    end = None if outcome_end is None else pd.Timestamp(outcome_end, tz="UTC")
    config = EvaluationConfig(size_reps=8, sensitivity_bands=(), outcome_end=end)
    record = evaluate(real, config, data_class="real", run_id="R-y")
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(real, record, config, created_at=CREATED)
    assert caught.value.reason == "outcome_cutoff"


# --- the store's frozen_models table ---


def _store(tmp_path, evaluation=None, now=LATER) -> StockLabStore:
    store = StockLabStore(tmp_path / DB_NAME, clock=lambda: now)
    if evaluation is not None:
        store.record_evaluation(evaluation)
    return store


def test_frozen_models_are_registered_once_and_read_back(tmp_path, frozen, evaluation):
    store = _store(tmp_path)
    assert store.frozen_model(frozen["model_id"]) is None  # no store yet: nothing, nothing created
    assert not store.exists
    with pytest.raises(StoreRefused, match="evaluation must be stored first"):
        store.record_frozen_model(frozen)
    store.record_evaluation(evaluation)
    assert store.record_frozen_model(frozen) == frozen["model_id"]
    assert store.frozen_model(frozen["model_id"]) == frozen
    assert [m["model_id"] for m in store.frozen_models(DataClass.SYNTHETIC)] == [frozen["model_id"]]
    assert store.frozen_models(DataClass.REAL) == ()
    with pytest.raises(sqlite3.IntegrityError, match="write-once"):
        store.record_frozen_model(frozen)
    connection = sqlite3.connect(store.path)
    try:
        for statement in ("DELETE FROM frozen_models", "UPDATE frozen_models SET estimator = 'B3'"):
            with pytest.raises(sqlite3.IntegrityError, match="write-once"):
                connection.execute(statement)
    finally:
        connection.close()


def _tamper(store, statement: str, parameters=()) -> None:
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("DROP TRIGGER frozen_models_no_update")
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("statement", "message"),
    [
        (
            "UPDATE frozen_models SET record = replace(record, '\"half_life\":', '\"half_life\": ')",
            "does not match its hash",
        ),
        ("UPDATE frozen_models SET estimator = 'B3'", "columns do not match"),
    ],
)
def test_a_stored_model_changed_outside_the_store_is_refused(
    tmp_path, frozen, evaluation, statement, message
):
    store = _store(tmp_path, evaluation)
    store.record_frozen_model(frozen)
    _tamper(store, statement)
    with pytest.raises(StoreRefused, match=message):
        store.frozen_model(frozen["model_id"])


def test_a_stored_record_that_fails_its_check_is_refused_even_with_a_matching_hash(
    tmp_path, frozen, evaluation
):
    store = _store(tmp_path, evaluation)
    store.record_frozen_model(frozen)
    forged = registry.canonical({**frozen, "half_life": 21.0})  # its id no longer matches
    sha = hashlib.sha256(forged.encode("utf-8")).hexdigest()
    _tamper(store, "UPDATE frozen_models SET record = ?, content_sha256 = ?", (forged, sha))
    with pytest.raises(StoreRefused, match="failed its check"):
        store.frozen_model(frozen["model_id"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"evaluation_sha256": "0" * 64}, "does not match"),
        ({"config": {**CONFIG.as_dict(), "kappa": 20.0}}, "does not match"),
        ({"created_at": "2026-10-16T21:00:00.000000Z"}, "predate its evaluation"),
        ({"created_at": "2026-10-17T22:00:00.000000Z"}, "in the future"),
        ({"half_life": "x"}, "failed its check"),
    ],
)
def test_the_store_registers_only_a_model_that_describes_its_stored_evaluation(
    tmp_path, frozen, evaluation, change, message
):
    store = _store(tmp_path, evaluation)
    record = _reseal(frozen, **change) if change != {"half_life": "x"} else {**frozen, **change}
    with pytest.raises(StoreRefused, match=message):
        store.record_frozen_model(record)
    assert store.frozen_models(DataClass.SYNTHETIC) == ()


def test_the_store_refuses_a_hand_made_challenger(tmp_path, frozen, evaluation):
    store = _store(tmp_path, evaluation)
    selection = {
        **frozen["selection"],
        "frozen": "M1",
        "evaluation_selected": "M1",
        "decision_rule": CONFIRMED,
    }
    challenger = _reseal(frozen, estimator="M1", selection=selection)
    with pytest.raises(StoreRefused, match="a7_unresolved"):
        store.record_frozen_model(challenger)


def test_an_older_store_without_the_table_reads_as_empty(tmp_path, evaluation):
    store = _store(tmp_path, evaluation)
    connection = sqlite3.connect(store.path)
    try:
        connection.execute("DROP TABLE frozen_models")
        connection.commit()
    finally:
        connection.close()
    assert store.frozen_models(DataClass.SYNTHETIC) == ()
    assert store.frozen_model("0" * 64) is None


def test_an_evaluation_under_the_confirmed_rule_freezes_its_own_selection(histories, evaluation):
    assert evaluation["decision_rule"] == frozen_model.A7_DECISION_RULE
    assert evaluation["selection"]["decision_rule"] == frozen_model.A7_DECISION_RULE
    frozen = registry.freeze(histories, evaluation, CONFIG, created_at=CREATED)
    assert frozen["estimator"] == evaluation["selection"]["selected"]
    assert frozen["selection"]["decision_rule"] == frozen_model.A7_DECISION_RULE
    challenger = _confirmed(evaluation, "B3", frozen_model.A7_DECISION_RULE)  # no monkeypatch
    assert registry.freeze(histories, challenger, CONFIG, created_at=CREATED)["estimator"] == "B3"


@pytest.mark.parametrize(
    "forge",
    [
        lambda r: r["selection"].update(beating_reference=[]),  # B3 selected, nothing beat B1
        lambda r: r["selection"].update(beating_reference=["B0", "B3"]),  # B0 is simpler
        lambda r: r["selection"].update(
            beating_reference=["B3", "M1"],
            steps=[{"challenger": "M1", "against": "B3", "moved": True}],
        ),  # the step moved to M1
        lambda r: r["selection"].update(beating_reference=["B9"]),
        lambda r: r["selection"].pop("decision_rule"),
        lambda r: r["selection"].update(decision_test="newey_west"),
        lambda r: r.update(schema_version="1.2"),
    ],
)
def test_a_confirmed_selection_must_follow_from_its_own_record(histories, evaluation, forge):
    record = _confirmed(evaluation, "B3", frozen_model.A7_DECISION_RULE)
    forge(record)
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, record, CONFIG, created_at=CREATED)
    assert caught.value.reason == "inconsistent_selection"


def test_a_climbing_selection_that_follows_its_steps_is_frozen(histories, evaluation):
    record = _confirmed(evaluation, "M1", frozen_model.A7_DECISION_RULE)
    record["selection"].update(
        beating_reference=["M1", "B3"],
        steps=[{"challenger": "M1", "against": "B3", "moved": True}],
    )
    assert registry.freeze(histories, record, CONFIG, created_at=CREATED)["estimator"] == "M1"


def test_an_evaluation_stored_before_a7_is_refused_by_its_5_percent_settings(histories, evaluation):
    old = copy.deepcopy(evaluation)
    del old["decision_rule"]
    for key in ("decision_rule", "decision_test"):
        old["selection"].pop(key)
    old["schema_version"] = "1.2"
    old["config"] = {**old["config"], "alpha": 0.05}  # as every pre-A7 record stores it
    old["selection"]["selected"] = "M1"
    with pytest.raises(registry.RegistryRefused) as caught:
        registry.freeze(histories, old, CONFIG, created_at=CREATED)
    assert caught.value.reason == "config_mismatch"
