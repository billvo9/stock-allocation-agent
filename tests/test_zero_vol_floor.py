"""
Zero realized volatility: the rule in force stays the default, the research
option B2 (a half-tick floor at the as-of close, scaled by 1/sqrt(h)) is
explicit and non-default, measured zeros stay distinguishable from the
floor, and scripts/check_zero_vol_floor.py checks both rules on synthetic F1
fixtures, including a tick-discretized SHV-like series.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
from datetime import UTC, datetime
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import yaml
from forecast_prospective_helpers import (
    AS_OF,
    ASSETS,
    REGISTERED,
    REPO,
    exchange_calendar,
    prospective_store,
    record,
    small_config,
)
from prospective_helpers import batch, stored
from zero_vol_helpers import MILD_REGIMES, tick_batch

from stock_agent.calendars import nyse_rules
from stock_agent.data.funds.ingest import ingest_batch
from stock_agent.data.funds.reader import read_store
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.forecasting.prospective import (
    forecast,
    ledger,
    registration,
    series,
    training,
    zero_vol,
    zero_vol_check,
)
from stock_agent.forecasting.prospective.config import load_config
from stock_agent.model_validation.audit import table_sha256

SCRIPT = REPO / "scripts" / "check_zero_vol_floor.py"
ZERO_VOL_DOC = REPO / "docs" / "research" / "zero_vol_floor.md"
REGISTRATION_DOC = REPO / "docs" / "research" / "prospective_registration.md"
# The registered config's hash on main at b7293f9: this work must not change it.
REGISTERED_SHA256 = "ae5d5853c40f7e65df1b990f149f0071b403dd8409fd1e66d07dee452087c31a"
CLOCK = datetime(2026, 10, 17, 21, 0, tzinfo=UTC)


def _script():
    spec = importlib.util.spec_from_file_location("check_zero_vol_floor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tables(root):
    store = read_store(root)
    return store.prices, store.distributions, store.splits, store.batches


# --- the formula ------------------------------------------------------------------------


def test_the_floor_is_half_a_tick_at_the_as_of_close_over_sqrt_h():
    floor = float(zero_vol.sigma_floor(110.30, 5))
    assert floor == 0.5 * math.log1p(0.01 / 110.30) / math.sqrt(5)
    assert round(floor * 1e4, 4) == 0.2027  # basis points of daily volatility
    # About half a tick's relative size over sqrt(h): log(1 + x) ~ x for a small tick.
    approximate = 0.5 * 0.01 / 50 / math.sqrt(20)
    assert float(zero_vol.sigma_floor(50.00, 20)) == pytest.approx(approximate, rel=1e-3)
    assert zero_vol.sigma_floor(110.30, 20) < zero_vol.sigma_floor(110.30, 5)
    assert zero_vol.sigma_floor(110.30, 5) < zero_vol.sigma_floor(50.00, 5)


@pytest.mark.parametrize("close", [20.0, 50.0, 110.30, 500.0])
@pytest.mark.parametrize("horizon", [5, 20, 30])
@pytest.mark.parametrize("direction", [1, -1])
def test_one_full_tick_in_the_window_is_never_floored(close, horizon, direction):
    g = np.zeros(horizon + 2)
    g[0] = np.nan  # the as-of row's own return is not in its window
    g[2] = math.log1p(direction * 0.01 / close)
    labels = zero_vol.volatility_labels(
        pd.Series(pd.bdate_range("2024-01-02", periods=horizon + 2, tz="UTC")),
        pd.Series(g),
        np.full(horizon + 2, close),
        horizon=horizon,
        entry_lag=1,
        rule=zero_vol.B2,
    ).iloc[0]
    assert not labels["floored"] and not labels["measured_zero"]
    assert labels["sigma_raw"] / labels["sigma_floor"] == pytest.approx(2.0, rel=0.01)


def _labels(g, rule, close=110.30, horizon=5):
    n = len(g)
    return zero_vol.volatility_labels(
        pd.Series(pd.bdate_range("2024-01-02", periods=n, tz="UTC")),
        pd.Series(np.asarray(g, dtype=np.float64)),
        np.full(n, close) if np.isscalar(close) else np.asarray(close, dtype=np.float64),
        horizon=horizon,
        entry_lag=1,
        rule=rule,
    )


def test_the_current_rule_is_the_existing_label_bit_for_bit_with_flags():
    rng = np.random.RandomState(0)
    g = rng.randn(300) * 1e-4
    g[rng.rand(300) < 0.6] = 0.0  # long flat stretches: many all-zero windows
    g[[40, 41]] = np.nan
    current = _labels(g, zero_vol.CURRENT)
    existing = series.labels_from_log_returns(
        current.index.to_series(),
        pd.Series(g),
        horizon=5,
        entry_lag=1,
        target="log_realized_volatility",
    )
    np.testing.assert_array_equal(
        current["target_return"].to_numpy(), existing["target_return"].to_numpy()
    )
    zero = current["measured_zero"].to_numpy()
    assert zero.sum() > 10
    assert np.isnan(current.loc[zero, "target_return"]).all()  # dropped / a miss, as today
    assert (current.loc[zero, "sigma_raw"] == 0.0).all()
    assert not current["floored"].any() and current["sigma_floor"].isna().all()
    assert zero_vol.DEFAULT == zero_vol.CURRENT


def test_b2_floors_measured_zeros_and_sub_tick_windows_and_keeps_the_rest_bit_for_bit():
    tick = math.log1p(0.01 / 110.30)
    sub_tick = math.log1p(0.003 / 110.30)
    g = np.zeros(40)
    g[0] = np.nan
    g[9] = tick  # row 7's window (returns 9..13) holds one full tick
    g[23] = sub_tick  # row 21's window holds only a sub-tick distribution
    g[33] = np.nan  # rows whose window covers 33 are unlabeled
    current, b2 = _labels(g, zero_vol.CURRENT), _labels(g, zero_vol.B2)
    floor = float(zero_vol.sigma_floor(110.30, 5))
    assert b2.loc[0, "measured_zero"] and b2.loc[0, "floored"]
    assert b2.loc[0, "target_return"] == math.log(floor)
    assert np.isnan(current.loc[0, "target_return"])
    assert not b2.loc[7, "floored"]
    assert b2.loc[7, "target_return"] == current.loc[7, "target_return"]
    assert b2.loc[21, "floored"] and not b2.loc[21, "measured_zero"]
    assert b2.loc[21, "target_return"] == math.log(floor)
    assert current.loc[21, "target_return"] == pytest.approx(math.log(sub_tick / math.sqrt(5)))
    unlabeled = b2.loc[29:31]
    assert unlabeled["target_return"].isna().all()
    assert not unlabeled["floored"].any() and not unlabeled["measured_zero"].any()
    moved = ~b2["floored"].to_numpy()
    np.testing.assert_array_equal(
        b2["target_return"].to_numpy()[moved], current["target_return"].to_numpy()[moved]
    )


def test_the_floor_uses_only_the_rows_own_as_of_close():
    g = np.zeros(30)
    g[0] = np.nan
    closes = np.full(30, 110.30)
    before = _labels(g, zero_vol.B2, closes)
    later = closes.copy()
    later[1:] = 50.0  # every close after row 0, including its whole outcome window
    after = _labels(g, zero_vol.B2, later)
    assert after.loc[0, "sigma_floor"] == before.loc[0, "sigma_floor"]
    assert after.loc[1, "sigma_floor"] != before.loc[1, "sigma_floor"]


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="Unknown zero-volatility rule"):
        zero_vol.check_rule("floor_at_one_tick")
    with pytest.raises(ValueError, match="Unknown zero-volatility rule"):
        _labels(np.zeros(10), "b1")


# --- training ----------------------------------------------------------------------------


def test_the_registered_config_is_unchanged():
    config = load_config(REGISTERED)
    assert config.sha256 == REGISTERED_SHA256
    assert "zero" not in json.dumps(config.to_json())  # no zero-volatility choice in it


@pytest.fixture(scope="module")
def tick_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("tick") / "f1_research_store"
    root.mkdir()
    ingest_batch(root, tick_batch())
    return root


@pytest.fixture(scope="module")
def tick_data(tick_store):
    config = small_config()
    tables = (*_tables(tick_store), config)
    return {rule: training.training_data(*tables, zero_vol_rule=rule) for rule in zero_vol.RULES}


def test_the_default_is_the_rule_in_force(tick_store, tick_data):
    default = training.training_data(*_tables(tick_store), small_config())
    assert default.zero_vol_rule == zero_vol.CURRENT
    for key, frame in default.frames.items():
        pd.testing.assert_frame_equal(frame, tick_data[zero_vol.CURRENT].frames[key])


def test_b2_drops_nothing_and_changes_only_floored_labels(tick_data):
    current, b2 = tick_data[zero_vol.CURRENT], tick_data[zero_vol.B2]
    assert current.zero_volatility_rows[("SHV", 5)] > 0
    assert all(count == 0 for count in b2.zero_volatility_rows.values())
    for (asset, horizon), frame in current.frames.items():
        other = b2.frames[(asset, horizon)]
        assert list(other.columns) == list(frame.columns)
        flags = b2.volatility_diagnostics[(asset, horizon)]
        pd.testing.assert_frame_equal(flags[["date", "symbol"]], frame[["date", "symbol"]])
        floored = flags["floored"].to_numpy()
        np.testing.assert_array_equal(
            other.loc[~floored, "y_log_realized_volatility"].to_numpy(),
            frame.loc[~floored, "y_log_realized_volatility"].to_numpy(),
        )
        expected = np.log(flags.loc[floored, "sigma_floor"].to_numpy())
        np.testing.assert_array_equal(
            other.loc[floored, "y_log_realized_volatility"].to_numpy(), expected
        )
        zero = flags["measured_zero"].to_numpy()
        assert frame.loc[zero, "y_log_realized_volatility"].isna().all()  # dropped today
        assert frame.loc[zero, "y_total_log_return"].isna().all()
        assert np.isfinite(other.loc[zero, "y_total_log_return"]).all()  # kept under B2
        assert int(zero.sum()) == current.zero_volatility_rows[(asset, horizon)]
        if asset in ("VTI", "AGG"):
            assert not floored.any()
            pd.testing.assert_frame_equal(other, frame)
    assert b2.volatility_diagnostics[("SHV", 5)]["measured_zero"].sum() > 0


def test_measured_zeros_stay_distinguishable_from_the_floor(tick_data):
    for rule, data in tick_data.items():
        flags = data.volatility_diagnostics[("SHV", 5)]
        zero = flags["measured_zero"]
        assert (flags.loc[zero, "sigma_raw"] == 0.0).all()
        assert (flags.loc[~zero & flags["sigma_raw"].notna(), "sigma_raw"] > 0).all()
        assert flags["floored"].any() == (rule == zero_vol.B2)


def test_sub_tick_distributions_are_floored_but_not_measured_zero():
    days = nyse_rules.sessions("2021-02-01", "2022-05-31")
    cash = [(str(day.date()), 0.002) for day in days[::7]]
    raw = tick_batch(distributions=cash)
    prices, distributions, splits = raw.prices, raw.distributions, raw.splits
    batches = pd.DataFrame({"adapter_id": [raw.adapter_id], "batch_id": ["fixture"]})
    data = training.training_data(
        prices, distributions, splits, batches, small_config(), zero_vol_rule=zero_vol.B2
    )
    _, totals = zero_vol_check.window_counts(data)
    shv = totals.set_index(["asset", "horizon"]).loc[("SHV", 5)]
    assert shv["n_floored_not_zero"] > 0
    assert shv["n_floored"] == shv["n_zero"] + shv["n_floored_not_zero"]
    assert shv["min_sigma_raw_nonzero_bp"] < float(zero_vol.sigma_floor(110.30, 5)) * 1e4


def test_freezing_records_the_rule_in_force(prospective_frozen):
    fit = prospective_frozen.result.bundle.fit("SHV", 5, "log_realized_volatility")
    assert fit["training"]["zero_volatility_rule"] == zero_vol.CURRENT


def test_without_a_floored_window_b2_develops_bit_for_bit(prospective_frozen):
    # The GARCH fixture never has a flat window, so B2 must change nothing at all.
    data = training.load_training_data(
        prospective_frozen.store, prospective_frozen.config, zero_vol_rule=zero_vol.B2
    )
    assert not any(f["floored"].any() for f in data.volatility_diagnostics.values())
    developed = training.develop(data, prospective_frozen.config)
    frozen_dev = prospective_frozen.result.development
    for name in ("predictions", "bands", "metrics", "selection"):
        assert table_sha256(getattr(developed, name)) == table_sha256(getattr(frozen_dev, name))


# --- prospective scoring ------------------------------------------------------------------


@pytest.fixture
def flat_store(tmp_path):
    days = nyse_rules.sessions("2026-10-15", "2026-10-22")

    def flat(retrieval, tables):
        prices, distributions, splits = tables
        if retrieval >= "2026-10-22":
            shv = prices["fund_id"] == "SHV"
            entry_close = prices.loc[shv & (prices["session_date"] == days[0]), "close"].iloc[0]
            prices = prices.copy()
            prices.loc[shv & prices["session_date"].isin(days), "close"] = float(entry_close)
        return prices, distributions, splits

    return prospective_store(tmp_path / "flat", edit=flat)


def _score(ledger_path, store, rule=zero_vol.DEFAULT):
    return forecast.score_matured(
        ledger_path=ledger_path,
        loader=load_inference_window,
        store_root=store,
        score_close="2026-10-23",
        scored_at=pd.Timestamp("2026-10-23T23:00:00Z"),
        code={"git_sha": "0" * 40},
        calendar=exchange_calendar,
        zero_vol_rule=rule,
    )


def _vol_scores(written):
    scores = pd.DataFrame(written.payload["scores"])
    return scores.loc[scores["target"] == "log_realized_volatility"].set_index(["asset", "horizon"])


def test_scoring_by_default_keeps_the_miss_and_records_the_flags(
    prospective_frozen, tmp_path, flat_store
):
    ledger_path = tmp_path / "ledger.jsonl"
    entry = record(prospective_frozen, ledger_path, flat_store)
    assert entry.payload["forecast_schema_version"] == "1.1"
    assert entry.payload["bundle"]["zero_volatility_rule"] == zero_vol.CURRENT
    assert all(row["as_of_close"] > 0 for row in entry.payload["rows"])
    written, _ = _score(ledger_path, flat_store)
    assert written.payload["score_schema_version"] == "1.1"
    flat = _vol_scores(written).loc[("SHV", 5)]
    assert not flat["inside_band"]
    assert flat["outcome_note"] == "zero_realized_volatility" and pd.isna(flat["outcome"])
    assert flat["zero_vol_rule"] == zero_vol.CURRENT
    assert flat["measured_zero"] and not flat["floored"]
    assert flat["sigma_raw"] == 0.0 and pd.isna(flat["sigma_floor"])
    assert not flat["as_of_close_revised"]
    moved = _vol_scores(written).loc[("VTI", 5)]
    assert not moved["measured_zero"] and moved["sigma_raw"] > 0
    returns = pd.DataFrame(written.payload["scores"])
    assert returns.loc[returns["target"] == "total_log_return", "zero_vol_rule"].isna().all()


def _with_rule(models: forecast.Models, rule: str) -> forecast.Models:
    """The fixture's models as if their bundle had been trained under `rule` (freeze cannot)."""

    document = copy.deepcopy(dict(models.bundle.document))
    for fit in document["fits"]:
        fit["training"]["zero_volatility_rule"] = rule
    return forecast.Models(dataclasses.replace(models.bundle, document=document), models.entries)


def test_scoring_refuses_a_rule_the_bundle_was_not_trained_under(
    prospective_frozen, tmp_path, flat_store
):
    ledger_path = tmp_path / "ledger.jsonl"
    entry = record(prospective_frozen, ledger_path, flat_store)
    assert entry.payload["bundle"]["zero_volatility_rule"] == zero_vol.CURRENT
    with pytest.raises(forecast.ForecastRefused) as refused:
        _score(ledger_path, flat_store, zero_vol.B2)
    assert refused.value.reason == "zero_vol_rule_not_trained"
    assert len(ledger.read_ledger(ledger_path)) == 1  # nothing written


def test_a_bundle_whose_fits_disagree_on_the_rule_is_refused(prospective_frozen):
    models = prospective_frozen.models()
    document = copy.deepcopy(dict(models.bundle.document))
    document["fits"][0]["training"]["zero_volatility_rule"] = zero_vol.B2
    with pytest.raises(forecast.ForecastRefused) as refused:
        forecast.training_rule(dataclasses.replace(models.bundle, document=document))
    assert refused.value.reason == "models_unverified"


def test_a_b2_bundle_scores_a_flat_outcome_at_its_recorded_as_of_floor(
    prospective_frozen, tmp_path, flat_store
):
    ledger_path = tmp_path / "ledger.jsonl"
    models = _with_rule(prospective_frozen.models(), zero_vol.B2)
    entry = record(prospective_frozen, ledger_path, flat_store, models=models)
    assert entry.payload["bundle"]["zero_volatility_rule"] == zero_vol.B2
    with pytest.raises(forecast.ForecastRefused) as refused:
        _score(ledger_path, flat_store)  # the rule in force: not what this bundle used
    assert refused.value.reason == "zero_vol_rule_not_trained"
    written, _ = _score(ledger_path, flat_store, zero_vol.B2)
    flat = _vol_scores(written).loc[("SHV", 5)]
    recorded = next(
        row["as_of_close"]
        for row in entry.payload["rows"]
        if (row["asset"], row["horizon"]) == ("SHV", 5)
    )
    floor = float(zero_vol.sigma_floor(recorded, 5))
    assert flat["outcome"] == math.log(floor) and flat["sigma_floor"] == floor
    assert flat["measured_zero"] and flat["floored"]
    assert flat["outcome_note"] == "floored_realized_volatility"
    assert flat["as_of_close"] == recorded and not flat["as_of_close_revised"]
    assert flat["inside_band"] == (flat["band_low"] <= flat["outcome"] <= flat["band_high"])


def test_a_revised_as_of_close_never_moves_the_floor(prospective_frozen, tmp_path):
    days = nyse_rules.sessions("2026-10-15", "2026-10-22")
    as_of = pd.Timestamp(AS_OF, tz="UTC").as_unit("us")

    def flat_and_revised(retrieval, tables):
        prices, distributions, splits = tables
        if retrieval >= "2026-10-22":
            prices = prices.copy()
            shv = prices["fund_id"] == "SHV"
            entry_close = prices.loc[shv & (prices["session_date"] == days[0]), "close"].iloc[0]
            prices.loc[shv & prices["session_date"].isin(days), "close"] = float(entry_close)
            prices.loc[shv & (prices["session_date"] == as_of), "close"] += 0.05  # a revision
        return prices, distributions, splits

    store = prospective_store(tmp_path / "revised", edit=flat_and_revised)
    ledger_path = tmp_path / "ledger.jsonl"
    models = _with_rule(prospective_frozen.models(), zero_vol.B2)
    entry = record(prospective_frozen, ledger_path, store, models=models)
    written, _ = _score(ledger_path, store, zero_vol.B2)
    flat = _vol_scores(written).loc[("SHV", 5)]
    recorded = next(r["as_of_close"] for r in entry.payload["rows"] if r["asset"] == "SHV")
    assert flat["as_of_close_revised"] and flat["as_of_close"] == recorded
    assert flat["sigma_floor"] == float(zero_vol.sigma_floor(recorded, 5))
    assert flat["sigma_floor"] != float(zero_vol.sigma_floor(recorded + 0.05, 5))


def test_a_ledger_is_never_scored_under_two_rules(prospective_frozen, tmp_path, flat_store):
    ledger_path = tmp_path / "ledger.jsonl"
    record(prospective_frozen, ledger_path, flat_store)
    first, _ = forecast.score_matured(
        ledger_path=ledger_path,
        loader=load_inference_window,
        store_root=flat_store,
        score_close="2026-10-22",  # the h = 5 rows mature at the 2026-10-22 close
        scored_at=pd.Timestamp("2026-10-22T23:00:00Z"),
        code={"git_sha": "0" * 40},
        calendar=exchange_calendar,
    )
    assert first is not None
    later = _with_rule(prospective_frozen.models(), zero_vol.B2)
    record(
        prospective_frozen,
        ledger_path,
        flat_store,
        as_of="2026-10-15",
        created=pd.Timestamp("2026-10-15T23:00:00Z"),
        models=later,
    )
    with pytest.raises(forecast.ForecastRefused) as refused:
        _score(ledger_path, flat_store, zero_vol.B2)
    assert refused.value.reason == "zero_vol_rule_mismatch"  # earlier scores used CURRENT
    with pytest.raises(forecast.ForecastRefused) as refused:
        _score(ledger_path, flat_store)
    assert refused.value.reason == "zero_vol_rule_not_trained"  # a pending B2 forecast


# --- the development check -----------------------------------------------------------------


def _store(root, regimes):
    root.mkdir()
    ingest_batch(root, tick_batch(regimes=regimes))
    return root


def _check(store, root):
    """One run of the script's check (the test variant of the config) and its report."""

    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(small_config().to_json()), encoding="utf-8")
    output = root / "out"
    argv = ["check", "--research-store", str(store), "--config", str(config_path)]
    argv += ["--output-dir", str(output)]
    code = _script().main(argv, clock=lambda: CLOCK)
    directory = output / CLOCK.strftime("%Y%m%dT%H%M%SZ")
    document = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    return {"code": code, "directory": directory, "document": document, "argv": argv}


@pytest.fixture(scope="module")
def check_run(tmp_path_factory):
    """The passing case: a mild flat stretch (MILD_REGIMES)."""

    root = tmp_path_factory.mktemp("mild")
    return _check(_store(root / "f1_research_store", MILD_REGIMES), root)


@pytest.fixture(scope="module")
def heavy_run(tick_store, tmp_path_factory):
    """The failing case: the default REGIMES, about 10% zeros at h = 5, stops at Z2."""

    return _check(tick_store, tmp_path_factory.mktemp("heavy"))


def _frame(document, name):
    return pd.DataFrame(document["tables"][name])


def test_the_check_passes_on_the_mild_fixture(check_run):
    document = check_run["document"]
    assert check_run["code"] == 0 and document["status"] == "pass"
    assert [c["id"] for c in document["criteria"]] == ["Z1", "Z2", "Z3", "Z4"]
    assert all(c["status"] == "pass" for c in document["criteria"])
    assert document["data_class"] == "synthetic"
    assert document["config_sha256"] == small_config().sha256
    assert set(document["rules"]) == set(zero_vol.RULES)
    assert (check_run["directory"] / "report.md").read_text().startswith("# Zero-volatility")
    assert "research only" in " ".join(document["labels"])
    power = pd.DataFrame(document["registration_power"])  # only after every criterion passed
    vol = power.loc[power["target"] == "log_realized_volatility"]
    assert set(vol["loss"]) == {"squared_error", "qlike"}
    assert set(power.loc[power["target"] == "total_log_return", "loss"]) == {"squared_error"}
    (reported,) = document["reported_checks"]
    assert reported["id"] == "R1" and reported["status"] == "reported"


def test_the_heavy_fixture_stops_at_z2(heavy_run):
    document = heavy_run["document"]
    assert heavy_run["code"] == 1 and document["status"] == "fail"
    status = {c["id"]: c["status"] for c in document["criteria"]}
    assert status == {"Z1": "pass", "Z2": "fail", "Z3": "not_evaluated", "Z4": "not_evaluated"}
    assert document["registration_power"] is None
    assert document["reported_checks"][0]["id"] == "R1"  # reported even after a failure
    shares = _frame(document, "nonzero_coverage").set_index(["asset", "horizon"])
    assert shares.loc[("SHV", 5), "difference"] > zero_vol_check.NONZERO_COVERAGE_TOLERANCE
    assert (shares.drop(index=("SHV", 5))["difference"] == 0.0).all()
    z2 = next(c for c in document["criteria"] if c["id"] == "Z2")
    assert "measured zeros" in z2["measured"]


def test_the_window_counts_and_the_cent_grid(heavy_run):
    totals = _frame(heavy_run["document"], "window_totals").set_index(["asset", "horizon"])
    by_year = _frame(heavy_run["document"], "windows_by_year")
    summed = by_year.groupby(["asset", "horizon"])[["n_windows", "n_zero", "n_floored"]].sum()
    pd.testing.assert_frame_equal(
        summed, totals[["n_windows", "n_zero", "n_floored"]], check_dtype=False
    )
    assert totals.loc[("SHV", 5), "n_zero"] > 0
    assert totals.loc[("SHV", 5), "n_floored"] == totals.loc[("SHV", 5), "n_zero"]
    for asset in ("VTI", "AGG"):
        assert (totals.loc[asset, "n_floored"] == 0).all()
    grid = _frame(heavy_run["document"], "cent_grid").set_index("asset")
    assert (grid.loc["SHV", "share_on_grid"] == 1.0).all()  # tick-discretized closes
    assert (grid.loc["VTI", "share_on_grid"] < 0.1).all()  # the GARCH fixture's float closes


def test_the_current_bands_lose_coverage_when_zeros_count_as_misses(heavy_run):
    coverage = _frame(heavy_run["document"], "coverage")
    vol = coverage.loc[coverage["target"] == "log_realized_volatility"]
    shv5 = vol.loc[(vol["asset"] == "SHV") & (vol["horizon"] == 5)].set_index("rule")
    assert shv5.loc[zero_vol_check.CURRENT_AS_MISS, "n_zero_as_miss"] > 0
    assert (
        shv5.loc[zero_vol_check.CURRENT_AS_MISS, "estimate"]
        < shv5.loc[zero_vol.CURRENT, "estimate"]
    )
    assert shv5.loc[zero_vol.B2, "n"] == shv5.loc[zero_vol_check.CURRENT_AS_MISS, "n"]
    for _, cell in vol.loc[vol["rule"] == zero_vol_check.CURRENT_AS_MISS].iterrows():
        if cell["n_zero_as_miss"] == 0:  # no zero added: exactly the current row
            same = vol.loc[
                (vol["rule"] == zero_vol.CURRENT)
                & (vol["asset"] == cell["asset"])
                & (vol["horizon"] == cell["horizon"])
            ].iloc[0]
            for column in ("n", "estimate", "se", "boot_ci_low", "boot_ci_high"):
                assert cell[column] == same[column], column
    split = _frame(heavy_run["document"], "zero_split").set_index(["rule", "asset", "horizon"])
    assert split.loc[(zero_vol_check.CURRENT_AS_MISS, "SHV", 5), "coverage_zero_rows"] == 0.0
    assert split.loc[(zero_vol.B2, "SHV", 5), "coverage_zero_rows"] > 0.0


def test_bit_identity_and_the_residual_shift(heavy_run):
    identity = _frame(heavy_run["document"], "bit_identity").set_index(["asset", "horizon"])
    assert identity.loc[identity["required"], "identical"].all()
    assert not identity.loc[("SHV", 5), "required"]
    assert not identity.loc[("SHV", 5), "identical"]  # the floor changed SHV h5, as designed
    shift = _frame(heavy_run["document"], "residual_shift").set_index(["asset", "horizon"])
    assert (shift.drop(index=("SHV", 5))["mean_shift"] == 0.0).all()
    assert 0 < abs(shift.loc[("SHV", 5), "mean_shift"]) <= zero_vol_check.RESIDUAL_SHIFT_MAX
    losses = _frame(heavy_run["document"], "mse_improvement")
    assert set(losses["rule"]) == set(zero_vol.RULES)


def test_a_report_is_written_once(check_run, capsys):
    assert _script().main(check_run["argv"], clock=lambda: CLOCK) == 2
    assert "already exists" in capsys.readouterr().err


def _passing_tables(check_run):
    return {name: pd.DataFrame(rows) for name, rows in check_run["document"]["tables"].items()}


@pytest.mark.parametrize(
    ("failing", "name", "edit"),
    [
        ("Z1", "bit_identity", lambda t: t.assign(identical=False)),
        ("Z2", "nonzero_coverage", lambda t: t.assign(difference=0.03)),
        ("Z3", "window_totals", lambda t: t.assign(share_floored_not_zero=0.02)),
        ("Z4", "residual_shift", lambda t: t.assign(mean_shift=0.06)),
    ],
)
def test_the_first_failing_criterion_stops_the_evaluation(check_run, failing, name, edit):
    tables = _passing_tables(check_run)
    tables[name] = edit(tables[name])
    results = zero_vol_check.evaluate_criteria(tables, small_config())
    status = {r["id"]: r["status"] for r in results}
    order = ["Z1", "Z2", "Z3", "Z4"]
    stop = order.index(failing)
    assert [status[i] for i in order[:stop]] == ["pass"] * stop
    assert status[failing] == "fail"
    assert all(status[i] == "not_evaluated" for i in order[stop + 1 :])


def test_band_calibration_is_reported_and_never_a_criterion(check_run):
    tables = _passing_tables(check_run)
    coverage = tables["coverage"]
    shv = (coverage["rule"] == zero_vol.B2) & (coverage["asset"] == "SHV")
    tables["coverage"] = coverage.assign(estimate=np.where(shv, 0.60, coverage["estimate"]))
    results = zero_vol_check.evaluate_criteria(tables, small_config())
    assert all(r["status"] == "pass" for r in results)
    (reported,) = zero_vol_check.evaluate_reported(tables, small_config())
    assert reported["status"] == "reported" and reported["within"] is False


def test_a_failed_check_exits_1_and_names_the_rule_in_force(
    check_run, tmp_path, monkeypatch, capsys
):
    script = _script()
    tables = _passing_tables(check_run)
    planted = zero_vol_check.CheckReport(
        status="fail",
        data_class="synthetic",
        batch_ids=("fixture",),
        fingerprints={},
        tables=tables,
        criteria=(
            {"id": "Z1", "statement": "", "status": "fail", "measured": "planted failure"},
            *(
                {"id": i, "statement": "", "status": "not_evaluated", "measured": "stopped"}
                for i in ("Z2", "Z3", "Z4")
            ),
        ),
        registration_power=None,
    )
    monkeypatch.setattr(script.zero_vol_check, "run_check", lambda *args: planted)
    store = check_run["argv"][2]
    argv = ["check", "--research-store", store, "--output-dir", str(tmp_path / "out")]
    assert script.main(argv, clock=lambda: CLOCK) == 1
    assert zero_vol.CURRENT in capsys.readouterr().out
    document = json.loads(next((tmp_path / "out").rglob("report.json")).read_text())
    assert document["status"] == "fail" and document["registration_power"] is None


# --- refusals ---------------------------------------------------------------------------------


def test_the_check_refuses_a_prospective_store(tmp_path, capsys):
    root = stored(tmp_path / "prospective", batch())
    inner = next(entry for entry in root.iterdir() if entry.is_dir())
    for target in (root, inner, tmp_path / "missing"):
        assert _script().main(["check", "--research-store", str(target)]) == 2
        assert "is not an F1 research store" in capsys.readouterr().err
    assert not (REPO / "reports" / "ml" / "prospective" / "zero_vol_check" / "x").exists()


@pytest.mark.parametrize(
    ("table", "column", "value"),
    [
        ("prices", "session_date", "2025-01-02"),
        ("distributions", "pay_date", "2025-01-03"),
        ("distributions", "ex_date", "2025-01-02"),
        ("splits", "effective_date", "2025-01-02"),
    ],
)
def test_any_date_after_2024_is_refused_before_anything_is_computed(
    tick_store, table, column, value
):
    prices, distributions, splits, batches = _tables(tick_store)
    frames = {"prices": prices, "distributions": distributions, "splits": splits}
    late = pd.Timestamp(value, tz="UTC")
    if frames[table].empty:
        frames[table] = pd.DataFrame({column: [late]})
    else:
        frames[table] = frames[table].copy()
        frames[table].loc[frames[table].index[0], column] = late
    with pytest.raises(zero_vol_check.CheckRefused) as refused:
        zero_vol_check.run_check(
            frames["prices"], frames["distributions"], frames["splits"], batches, small_config()
        )
    assert refused.value.reason == "after_training_end"


# --- the documents ----------------------------------------------------------------------------


def test_the_doc_holds_the_scripts_worked_examples_and_criteria(capsys):
    assert _script().main(["examples"]) == 0
    printed = capsys.readouterr().out.strip()
    text = ZERO_VOL_DOC.read_text(encoding="utf-8")
    assert printed == zero_vol_check.render_examples()
    assert printed in text
    for identifier, statement in zero_vol_check.CRITERIA:
        assert f"**{identifier}.** {statement}" in text
    for identifier, statement in zero_vol_check.REPORTED_CHECKS:
        assert f"**{identifier}.** {statement}" in text
    assert zero_vol_check.ORIGINAL_Z2 in text  # the revised criterion stays visible


def test_the_registration_doc_holds_the_scripts_schedule(capsys):
    assert _script().main(["registration"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed in REGISTRATION_DOC.read_text(encoding="utf-8")


# --- the registration arithmetic ------------------------------------------------------------


def test_the_registered_outcome_counts_and_families():
    counts = {
        (h, name): registration.confirmatory_outcomes(h, c)
        for h in registration.HORIZONS
        for name, c in registration.CHECKPOINTS.items()
    }
    assert counts == {
        (5, "12_months"): 50,
        (5, "24_months"): 100,
        (20, "12_months"): 12,
        (20, "24_months"): 25,
        (30, "12_months"): 8,
        (30, "24_months"): 16,
    }
    assert registration.family(252) == [(f, 5) for f in ASSETS]
    assert registration.family(504) == [(f, h) for h in (5, 20) for f in ASSETS]
    assert registration.first_holm_level(252) == 0.025 / 3
    assert registration.first_holm_level(504, tests_per_cell=2) == 0.025 / 12
    assert registration.FORECASTS_PER_SESSION == 18


@pytest.mark.parametrize("horizon", [5, 20, 30])
def test_confirmatory_outcomes_never_overlap_and_mature_by_the_checkpoint(horizon):
    sessions = registration.confirmatory_sessions(horizon, 504)
    assert sessions[0] == registration.TRIAL_SESSIONS + 1
    windows = [(s + 1, s + 1 + horizon) for s in sessions]  # (entry, exit]
    assert all(later[0] >= earlier[1] for earlier, later in pairwise(windows))
    assert windows[-1][1] <= registration.TRIAL_SESSIONS + 504


def test_the_power_arithmetic():
    alpha = registration.first_holm_level(252)
    mde = registration.minimum_detectable_effect(50, alpha)
    assert registration.power(mde, 50, alpha) == pytest.approx(0.8, abs=1e-6)
    assert registration.outcomes_for_power(mde, alpha) in (50, 51)
    assert registration.outcomes_for_power(0.0, alpha) is None
    low, high = registration.binomial_acceptance(50, 0.8, alpha)
    assert low < 40 < high
    pmf = [math.comb(50, k) * 0.8**k * 0.2 ** (50 - k) for k in range(51)]
    assert sum(pmf[:low]) <= alpha / 2 and sum(pmf[high + 1 :]) <= alpha / 2
