"""
The prospective pipeline end to end on synthetic fixtures: an F1 research
store, training and freezing, a synthetic prospective store written by the
real prospective writer, the forecast record read through the real
`load_inference_window`, a matured outcome and its score through the same
loader. Leakage tests: the forecast is unchanged by any data after its close
or retrieved after its decision time, training refuses any session after
2024-12-31, an incomplete window is refused rather than median-filled, and
the ledger refuses edits.
"""

from __future__ import annotations

import ast
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from forecast_prospective_helpers import (
    AS_OF,
    ASSETS,
    CREATED,
    PROVENANCE,
    REPO,
    exchange_calendar,
    prospective_store,
    prospective_tables,
    record,
    research_batch,
    score,
    small_config,
    without_close,
)

from stock_agent.calendars import nyse_rules
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.data.prospective.window import WINDOW_START
from stock_agent.forecasting.prospective import (
    forecast,
    freeze,
    frozen,
    ledger,
    synthetic,
    training,
    window,
)
from stock_agent.model_diagnostics.linear import LinearForecast
from stock_agent.model_validation.audit import LeakageError
from stock_agent.model_validation.folds import MODEL_HOLDOUT_START, is_labeled
from stock_agent.registry.stages import STAGES

PACKAGE = REPO / "src" / "stock_agent" / "forecasting" / "prospective"


# --- freezing and registration ---


def test_freezing_registers_one_version_per_target_and_horizon(prospective_frozen):
    result = prospective_frozen.result
    config = prospective_frozen.config
    assert len(result.entries) == len(config.targets) * len(config.horizons)
    registry = prospective_frozen.registry
    for entry in result.entries:
        assert registry.artifacts_verified(entry)
        kinds = sorted(a.kind for a in entry.artifacts)
        assert kinds == ["experiment_run", frozen.BUNDLE_KIND]
        bundle_ref = next(a for a in entry.artifacts if a.kind == frozen.BUNDLE_KIND)
        assert bundle_ref.sha256 == result.bundle.manifest_sha256
        assert entry.config["experiment_config_sha256"] == config.sha256
        assert entry.config["prospective"]["status"] == "prospective record pending"
        assert entry.code == PROVENANCE["code"]
        assert entry.training_data["data_class"] == "synthetic"
        for key in ("prices_sha256", "distributions_sha256", "features_sha256", "universe_sha256"):
            assert len(entry.training_data[key]) == 64
        assert entry.training_data["batch_ids"]


def test_the_stage_is_computed_and_never_above_development_evaluated(prospective_frozen):
    registry = prospective_frozen.registry
    for entry in prospective_frozen.result.entries:
        assessment = registry.assessment(entry.ref)
        assert assessment.stage == "engineering_only"  # synthetic data verifies the pipeline
        assert assessment.stage in STAGES[:2]
        reasons = {c.reason for c in assessment.blockers}
        assert "data_class_engineering_only" in reasons


def test_every_leakage_and_instrument_check_passes(prospective_frozen):
    record_ = prospective_frozen.result.record
    families = record_["checks_summary"]["families"]
    assert families["leakage"]["failed"] == 0 and families["leakage"]["undetermined"] == 0
    assert families["instrument"]["failed"] == 0
    names = set(prospective_frozen.result.development.checks["check"])
    assert {"lockbox_closed", "nested_selection_inside_training", "predictions_complete"} <= names
    lockbox = record_["lockbox_evidence"]
    assert lockbox["holdout_values_used"] is False
    assert pd.Timestamp(lockbox["max_target_end_date"]) < MODEL_HOLDOUT_START
    assert record_["multiple_testing"]["candidates_in_run"] == 6


def test_development_coverage_is_reported_per_asset_horizon_and_target(prospective_frozen):
    metrics = prospective_frozen.result.development.metrics
    config = prospective_frozen.config
    coverage = metrics[
        (metrics["metric"] == "band_coverage_80") & (metrics["inference_method"] == "fold_block_t")
    ]
    keys = set(zip(coverage["asset"], coverage["horizon"], coverage["target"], strict=True))
    assert keys == {
        (a, h, t) for a in config.assets for h in config.horizons for t in config.targets
    }
    plugin = metrics[metrics["metric"] == "plugin_coverage_80"]
    assert set(plugin["target"]) == {"total_log_return"}  # the plug-in, for comparison
    fit = prospective_frozen.result.bundle.fit("VTI", 20, "total_log_return")
    assert fit["development"]["band_coverage_80"] is not None
    assert fit["development"]["band_rows_scored"] > 0


def test_band_quantiles_come_only_from_development_scores_before_the_lockbox(prospective_frozen):
    dev = prospective_frozen.result.development
    assert (pd.to_datetime(dev.bands["target_end_date"], utc=True) < MODEL_HOLDOUT_START).all()
    bundle = prospective_frozen.result.bundle
    for target in prospective_frozen.config.targets:
        rows = dev.bands[
            (dev.bands["symbol"] == "AGG")
            & (dev.bands["horizon"] == 5)
            & (dev.bands["target"] == target)
        ]
        band = bundle.fit("AGG", 5, target)["band"]
        expected = training.bands.conformal_quantiles(training.bands.scores(rows), 0.1, 0.9)
        assert (band["lower"], band["upper"]) == expected
        assert band["n_calibration"] == len(rows)


def test_the_frozen_null_is_the_mean_of_every_training_label(prospective_frozen):
    data = training.load_training_data(prospective_frozen.store, prospective_frozen.config)
    for asset, horizon, target in (
        ("VTI", 5, "total_log_return"),
        ("SHV", 30, "log_realized_volatility"),
    ):
        frame = training.target_frame(data.frames[(asset, horizon)], target)
        ends = pd.to_datetime(frame["target_end_date"], utc=True)
        labels = frame.loc[
            is_labeled(frame) & (ends < MODEL_HOLDOUT_START).to_numpy(), "target_return"
        ]
        null = prospective_frozen.result.bundle.fit(asset, horizon, target)["null"]
        assert null["n_train"] == len(labels) > 252
        assert null["value"] == pytest.approx(labels.mean(), rel=1e-12)
        assert null["value"] != pytest.approx(labels.iloc[-252:].mean(), rel=1e-6)  # not a window


def test_a_frozen_ridge_predicts_exactly_as_the_fitted_model():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 5)
    X[::7, 2] = np.nan
    y = X[:, 0] * 0.3 + rng.randn(300) * 0.1
    model = LinearForecast("ridge", 0.1).fit(np.where(np.isnan(X), 0.0, X), y)
    stored = json.loads(json.dumps(frozen.ridge_json(model, list("abcde"))))
    restored = frozen.ridge_from_json(stored)
    assert np.array_equal(restored.predict(X), model.predict(X))


def test_the_bundle_refuses_an_edited_file(prospective_frozen, tmp_path):
    import shutil

    copy = tmp_path / prospective_frozen.result.bundle.path.name
    shutil.copytree(prospective_frozen.result.bundle.path, copy)
    models = copy / frozen.MODELS_FILE
    models.write_text(models.read_text().replace('"intercept": ', '"intercept": 1'), "utf-8")
    with pytest.raises(frozen.BundleError, match="edited"):
        frozen.read_bundle(copy)


class _StageAs:
    """A registry whose computed stage is replaced (the rules themselves are untouched)."""

    def __init__(self, registry, stage):
        self._registry, self._stage = registry, stage

    def __getattr__(self, name):
        return getattr(self._registry, name)

    def assessment(self, ref):
        return dataclasses.replace(self._registry.assessment(ref), stage=self._stage)


def test_models_are_refused_above_development_evaluated(prospective_frozen):
    bundle = prospective_frozen.result.bundle.path
    for stage in STAGES[2:]:
        with pytest.raises(forecast.ForecastRefused) as refused:
            forecast.load_models(bundle, _StageAs(prospective_frozen.registry, stage))
        assert refused.value.reason == "stage"
    allowed = forecast.load_models(
        bundle, _StageAs(prospective_frozen.registry, "development_evaluated")
    )
    assert {e["stage"] for e in allowed.entries.values()} == {"development_evaluated"}


def test_training_and_freezing_never_read_the_prospective_store():
    for name in ("training.py", "freeze.py", "series.py", "bands.py", "frozen.py"):
        tree = ast.parse((PACKAGE / name).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)} | {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        assert "window" not in imported and "load_inference_window" not in imported, name
        assert "load_inference_window" not in (PACKAGE / name).read_text(encoding="utf-8"), name


# --- leakage: training refuses the sealed period ---


def _tables_with(extra_price_date: str):
    batch = research_batch()
    prices = batch.prices.assign(batch_id="b")
    late = prices.iloc[[-1]].assign(
        session_date=pd.Timestamp(extra_price_date, tz="UTC").as_unit("us")
    )
    batches = pd.DataFrame({"batch_id": ["b"], "adapter_id": [synthetic.FIXTURE_ADAPTER_ID]})
    return pd.concat([prices, late]), batch.distributions, batch.splits, batches


def test_training_refuses_any_session_after_2024_12_31():
    prices, distributions, splits, batches = _tables_with("2025-01-02")
    with pytest.raises(LeakageError, match="after 2024-12-31"):
        training.training_data(prices, distributions, splits, batches, small_config())


@pytest.mark.parametrize(
    ("table", "column", "match"),
    [
        ("distributions", "ex_date", "distribution ex_dates"),
        ("distributions", "pay_date", "distribution pay_dates"),
        ("splits", "effective_date", "split dates"),
    ],
)
def test_training_refuses_any_event_dated_after_2024_12_31(table, column, match):
    prices, distributions, splits, batches = _tables_with("2024-12-31")
    tables = {"distributions": distributions.copy(), "splits": splits.copy()}
    if table == "splits":  # the 2019-2024 fixture has no split: add one dated in 2025
        tables["splits"] = pd.DataFrame(
            {
                "fund_id": ["VTI"],
                "effective_date": [pd.Timestamp("2025-01-15", tz="UTC")],
                "ratio_new_per_old": [2.0],
                "source_id": ["fixture"],
            }
        )
    else:
        late = tables[table]
        late.loc[late.index[-1], column] = pd.Timestamp("2025-01-15", tz="UTC")
    with pytest.raises(LeakageError, match=match):
        training.training_data(
            prices.iloc[:-1], tables["distributions"], tables["splits"], batches, small_config()
        )


def test_a_training_frame_reaching_the_sealed_period_is_refused(prospective_frozen):
    data = training.load_training_data(prospective_frozen.store, prospective_frozen.config)
    piece = training.target_frame(data.frames[("VTI", 5)], "total_log_return").copy()
    piece.loc[piece.index[-1], "date"] = pd.Timestamp("2025-01-02", tz="UTC")
    with pytest.raises(LeakageError, match="dated after 2024-12-31"):
        training.final_fit(piece, prospective_frozen.config, "total_log_return", 5)


def test_unknown_or_mixed_data_sources_are_refused():
    with pytest.raises(ValueError, match="No data class"):
        training.data_class_of(pd.DataFrame({"adapter_id": ["someone_else"]}))
    mixed = pd.DataFrame({"adapter_id": ["tiingo_daily", synthetic.FIXTURE_ADAPTER_ID]})
    with pytest.raises(ValueError, match="mixes data classes"):
        training.data_class_of(mixed)
    assert training.data_class_of(pd.DataFrame({"adapter_id": ["tiingo_daily"]})) == (
        "engineering_sandbox"
    )


# --- the forecast record: the real loader on a synthetic prospective store ---

PARTS = ("sessions", "prices", "distributions", "splits", "missing", "batches", "revisions")


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "private" / "ledger.jsonl"


@pytest.fixture
def store(prospective_store_root):
    return prospective_store_root


def _parts(store_root, as_of=AS_OF, lookback=63, retrieved_by=CREATED, assets=ASSETS):
    result = load_inference_window(
        store_root, list(assets), as_of, lookback, retrieved_by=retrieved_by
    )
    return {name: getattr(result, name) for name in PARTS}


def _coerce(parts, as_of=AS_OF, lookback=63, retrieved_by=CREATED, assets=ASSETS):
    return window.coerce_window(
        parts,
        assets=list(assets),
        forecast_close=as_of,
        lookback=lookback,
        calendar=exchange_calendar,
        retrieved_by=retrieved_by,
    )


def test_a_forecast_records_every_field_once(prospective_frozen, ledger_path, store):
    entry = record(prospective_frozen, ledger_path, store)
    payload = entry.payload
    config = prospective_frozen.config
    assert payload["as_of_session"] == AS_OF
    assert payload["created_at"] == CREATED.isoformat()
    assert payload["close_available_at"] == "2026-10-14T22:00:00+00:00"  # 18:00 New York (EDT)
    assert payload["record_deadline"] == "2026-10-15T13:30:00+00:00"  # 09:30 the next weekday
    assert "unvalidated, no demonstrated skill" in payload["labels"]
    assert payload["bundle"]["manifest_sha256"] == prospective_frozen.result.bundle.manifest_sha256
    assert payload["bundle"]["config_sha256"] == config.sha256
    assert payload["bundle"]["status"] == "prospective record pending"
    data = payload["data"]
    assert len(data["window_fingerprint"]) == 64
    assert data["data_class"] == "synthetic"
    # 63 daily returns from 64 closes, read as known at the decision time
    assert (data["lookback_returns"], data["closes"]) == (63, 64)
    assert data["first_session"] == str(WINDOW_START.date())
    assert data["calendar"] == data["checked_against"] == nyse_rules.CALENDAR_ID
    assert data["retrieved_by"] == CREATED.isoformat()
    assert data["complete"] is True and data["missing"] == [] and data["revisions"] == []
    assert [b["retrieved_at"] for b in data["batches"]] == ["2026-10-14T22:30:00+00:00"]
    assert payload["refusals"] == []
    rows = pd.DataFrame(payload["rows"])
    assert len(rows) == len(config.assets) * len(config.horizons) * len(config.targets)
    assert (rows["status"] == "ok").all()
    assert (rows["band_low"] < rows["point"]).all() and (rows["point"] < rows["band_high"]).all()
    for column in ("null", "model_id", "model_version", "entry_sha256", "stage", "development"):
        assert rows[column].notna().all(), column
    assert set(rows["stage"]) == {"engineering_only"}
    entries = {str(e.ref): e.entry_sha256 for e in prospective_frozen.result.entries}
    for row in payload["rows"]:
        assert entries[f"{row['model_id']}@v{row['model_version']}"] == row["entry_sha256"]
    assert ledger.read_ledger(ledger_path) == (entry,)


def test_a_second_forecast_for_the_same_session_and_version_is_refused(
    prospective_frozen, ledger_path, store
):
    record(prospective_frozen, ledger_path, store)
    with pytest.raises(forecast.ForecastRefused) as refused:
        record(prospective_frozen, ledger_path, store, created=CREATED + pd.Timedelta(hours=1))
    assert refused.value.reason == "duplicate_forecast"
    assert len(ledger.read_ledger(ledger_path)) == 1


@pytest.mark.parametrize(
    ("as_of", "created", "reason"),
    [
        ("2026-10-13", "2026-10-13T23:00:00Z", "before_h_end"),
        (AS_OF, "2026-10-14T21:59:59Z", "close_not_available"),
        (AS_OF, "2026-10-15T13:30:00Z", "past_record_deadline"),
    ],
)
def test_timing_rules_refuse_backdated_early_or_late_records(
    prospective_frozen, ledger_path, store, as_of, created, reason
):
    with pytest.raises(forecast.ForecastRefused) as refused:
        record(prospective_frozen, ledger_path, store, as_of=as_of, created=pd.Timestamp(created))
    assert refused.value.reason == reason
    assert not ledger_path.exists()


def test_a_friday_forecast_may_be_recorded_until_monday_morning(
    prospective_frozen, ledger_path, store
):
    entry = record(
        prospective_frozen,
        ledger_path,
        store,
        as_of="2026-10-16",
        created=pd.Timestamp("2026-10-19T13:29:00Z"),
    )
    assert entry.payload["record_deadline"] == "2026-10-19T13:30:00+00:00"


# --- leakage: nothing after the forecast close, nothing retrieved after the decision ---


def _leaky(extra: str):
    """The real loader, plus a row dated after the close (prices, cash or splits)."""

    def load(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        parts = _parts(store_root, forecast_close, lookback, retrieved_by, assets)
        late = window.session_stamp(forecast_close) + pd.Timedelta(days=1)
        if extra == "prices":
            row = parts["prices"].iloc[[-1]].assign(session_date=late)
        elif extra == "distributions":
            row = pd.DataFrame(
                {
                    "fund_id": ["VTI"],
                    "ex_date": [late],
                    "amount": [1.0],
                    "amount_basis": ["per_share_as_paid"],
                }
            )
        else:
            row = pd.DataFrame(
                {"fund_id": ["VTI"], "effective_date": [late], "ratio_new_per_old": [2.0]}
            )
        return {**parts, extra: pd.concat([parts[extra], row], ignore_index=True)}

    return load


@pytest.mark.parametrize("extra", ["prices", "distributions", "splits"])
def test_a_loader_returning_any_row_after_the_close_is_refused(
    prospective_frozen, ledger_path, store, extra
):
    with pytest.raises(LeakageError, match="after the forecast close"):
        record(prospective_frozen, ledger_path, store, loader=_leaky(extra))
    assert not ledger_path.exists()


def test_a_loader_reading_batches_retrieved_after_the_decision_is_refused(
    prospective_frozen, ledger_path, store
):
    def late(store_root, assets, forecast_close, lookback=63, *, retrieved_by):
        return load_inference_window(
            store_root,
            assets,
            forecast_close,
            lookback,
            retrieved_by=pd.Timestamp("2027-01-02T00:00:00Z"),
        )

    with pytest.raises(LeakageError, match="retrieved after the decision time"):
        record(prospective_frozen, ledger_path, store, loader=late)
    assert not ledger_path.exists()


def _revising(factor: float, after: str):
    """An edit: every batch retrieved after `after` revises every close by `factor`."""

    def edit(retrieval, tables):
        prices, distributions, splits = tables
        if retrieval > after:
            prices = prices.assign(close=prices["close"] * factor)
        return prices, distributions, splits

    return edit


def test_the_forecast_is_unchanged_by_any_data_after_its_close(prospective_frozen, tmp_path):
    known_then = prospective_store(tmp_path / "then", retrievals=(AS_OF,))
    # later batches revise every close, the as-of session's included: retrieved_by excludes them
    revised = prospective_store(tmp_path / "revised", edit=_revising(1.5, AS_OF))
    first = record(prospective_frozen, tmp_path / "a.jsonl", known_then)
    second = record(prospective_frozen, tmp_path / "b.jsonl", revised)
    assert first.payload["rows"] == second.payload["rows"]
    assert first.payload["data"] == second.payload["data"]
    later = _coerce(
        _parts(revised, retrieved_by=pd.Timestamp("2027-01-01T00:00:00Z")),
        retrieved_by=pd.Timestamp("2027-01-01T00:00:00Z"),
    )
    assert not later.revisions.empty  # the revision exists; the forecast never saw it


def test_the_forecast_ignores_rows_before_its_window(prospective_frozen, tmp_path, store):
    early = nyse_rules.sessions(WINDOW_START, "2026-12-31")[:2]

    def edit(retrieval, tables):
        prices, distributions, splits = tables
        first_two = prices["session_date"].isin(early)
        return (
            prices.assign(close=prices["close"].where(~first_two, prices["close"] * 3)),
            distributions,
            splits,
        )

    changed = prospective_store(tmp_path / "early", edit=edit)
    as_of, created = "2026-10-16", pd.Timestamp("2026-10-16T23:00:00Z")
    first = record(prospective_frozen, tmp_path / "a.jsonl", store, as_of=as_of, created=created)
    second = record(prospective_frozen, tmp_path / "b.jsonl", changed, as_of=as_of, created=created)
    assert first.payload["data"]["first_session"] > str(early[-1].date())
    assert first.payload["rows"] == second.payload["rows"]


def test_a_window_missing_an_exchange_session_is_refused(store):
    parts = _parts(store)
    dropped = {**parts, "sessions": parts["sessions"].delete(30)}  # 63 sessions, one gap
    with pytest.raises(window.WindowRefused, match="differ from the exchange calendar"):
        _coerce(dropped, lookback=62)


def test_a_window_holding_data_after_the_close_or_too_short_is_refused(store):
    parts = _parts(store)
    late = (
        parts["prices"]
        .iloc[[-1]]
        .assign(session_date=pd.Timestamp("2026-10-15", tz="UTC").as_unit("us"))
    )
    with pytest.raises(LeakageError, match="after the forecast close"):
        _coerce({**parts, "prices": pd.concat([parts["prices"], late])})
    with pytest.raises(window.WindowRefused, match="63 sessions, not 64"):
        _coerce({**parts, "sessions": parts["sessions"][1:]})


def test_window_features_equal_the_features_of_the_full_history(store):
    as_of, created = "2026-10-16", pd.Timestamp("2026-10-16T23:00:00Z")
    win = _coerce(_parts(store, as_of, retrieved_by=created), as_of, retrieved_by=created)
    _, from_window = forecast._features_at_close(win, "VTI")
    prices, distributions, splits = prospective_tables()
    sessions = nyse_rules.sessions(WINDOW_START, as_of)  # two sessions longer than the window
    last = sessions[-1]
    returns = training.series.daily_log_returns(
        sessions,
        prices.loc[prices["session_date"] <= last],
        distributions.loc[distributions["ex_date"] <= last],
        splits.loc[splits["effective_date"] <= last],
        "VTI",
    )
    full = training.series.asset_frame(returns, "VTI").iloc[-1]
    assert len(sessions) == len(win.sessions) + 2
    assert np.array_equal(from_window, full[list(training.FEATURES)].to_numpy(np.float64))


# --- an incomplete window is refused, never median-filled ---


@pytest.mark.parametrize("missing", [["2026-09-15"], [AS_OF]])
def test_an_asset_with_a_missing_close_is_refused_not_filled(
    prospective_frozen, ledger_path, tmp_path, missing
):
    gap = prospective_store(tmp_path / "gap", edit=without_close("VTI", missing))
    entry = record(prospective_frozen, ledger_path, gap)
    assert entry.kind == "forecast"
    rows = pd.DataFrame(entry.payload["rows"])
    assert set(rows["asset"]) == {"AGG", "SHV"}  # no VTI row, filled or otherwise
    assert entry.payload["refusals"] == [
        {
            "asset": "VTI",
            "reason": "incomplete_inference_window",
            "sessions_without_close": missing,
        }
    ]
    data = entry.payload["data"]
    assert data["complete"] is False
    assert [(m["fund_id"], m["session_date"][:10]) for m in data["missing"]] == [
        ("VTI", day) for day in missing
    ]
    written, _ = score(ledger_path, "2026-12-31", "2027-01-01T00:00:00Z", gap)
    assert set(pd.DataFrame(written.payload["scores"])["asset"]) == {"AGG", "SHV"}


def test_when_every_asset_is_incomplete_a_refusal_is_recorded_never_a_forecast(
    prospective_frozen, ledger_path, tmp_path
):
    def edit(retrieval, tables):
        for fund in ASSETS:
            tables = without_close(fund, ["2026-09-15"])(retrieval, tables)
        return tables

    gap = prospective_store(tmp_path / "gaps", edit=edit)
    entry = record(prospective_frozen, ledger_path, gap)
    assert entry.kind == "refusal"
    assert "rows" not in entry.payload
    assert {r["asset"] for r in entry.payload["refusals"]} == set(ASSETS)
    assert [e.kind for e in ledger.read_ledger(ledger_path)] == ["refusal"]
    assert forecast.pending_rows(ledger.read_ledger(ledger_path)) == []


# --- the ledger refuses edits ---


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


@pytest.fixture
def two_entry_ledger(prospective_frozen, ledger_path, store):
    record(prospective_frozen, ledger_path, store)
    record(
        prospective_frozen,
        ledger_path,
        store,
        as_of="2026-10-15",
        created=pd.Timestamp("2026-10-15T23:00:00Z"),
    )
    return ledger_path


@pytest.mark.parametrize("edit", ["value", "reorder", "remove_first", "partial"])
def test_the_ledger_refuses_edits(prospective_frozen, two_entry_ledger, store, edit):
    lines = _lines(two_entry_ledger)
    if edit == "value":
        lines[0] = lines[0].replace('"as_of_session":"2026-10-14"', '"as_of_session":"2026-10-13"')
    elif edit == "reorder":
        lines = [lines[1], lines[0]]
    elif edit == "remove_first":
        lines = lines[1:]
    else:
        lines[-1] = lines[-1].rstrip("\n")
    two_entry_ledger.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ledger.LedgerError):
        ledger.read_ledger(two_entry_ledger)
    with pytest.raises(ledger.LedgerError):  # and nothing can be appended to it
        record(
            prospective_frozen,
            two_entry_ledger,
            store,
            as_of="2026-10-16",
            created=pd.Timestamp("2026-10-16T23:00:00Z"),
        )


def test_an_edit_that_rehashes_its_own_line_still_breaks_the_chain(two_entry_ledger):
    from stock_agent.model_diagnostics.record import canonical_json, sha256_json

    lines = _lines(two_entry_ledger)
    first = json.loads(lines[0])
    first.pop("entry_sha256")
    first["payload"]["rows"][0]["point"] = 0.5
    first["entry_sha256"] = sha256_json(first)
    lines[0] = canonical_json(first) + "\n"
    two_entry_ledger.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ledger.LedgerError, match="breaks the chain"):
        ledger.read_ledger(two_entry_ledger)


def test_the_ledger_head_is_the_last_hash(two_entry_ledger):
    entries = ledger.read_ledger(two_entry_ledger)
    assert ledger.head(entries) == entries[-1].entry_sha256
    assert entries[1].prev_entry_sha256 == entries[0].entry_sha256
    assert ledger.head(()) == ledger.GENESIS


def test_a_kept_head_detects_a_removed_last_line(two_entry_ledger):
    entries = ledger.read_ledger(two_entry_ledger)
    kept = ledger.head(entries)
    ledger.check_head(entries, kept)
    two_entry_ledger.write_text("".join(_lines(two_entry_ledger)[:-1]), encoding="utf-8")
    truncated = ledger.read_ledger(two_entry_ledger)  # the chain alone cannot see it
    assert len(truncated) == 1
    with pytest.raises(ledger.LedgerError, match="not in the ledger"):
        ledger.check_head(truncated, kept)


# --- scoring, through the same real loader ---


def test_a_matured_forecast_is_scored_once_and_never_edited(prospective_frozen, ledger_path, store):
    forecast_entry = record(prospective_frozen, ledger_path, store)
    # 2026-10-23: as_of + 1 entry session + 5 sessions = 2026-10-22 exits; 20 and 30 do not
    written, left = score(ledger_path, "2026-10-23", "2026-10-23T23:00:00Z", store)
    assert written is not None
    scored = pd.DataFrame(written.payload["scores"])
    assert set(scored["horizon"]) == {5}
    assert len(scored) == 6 and {row["reason"] for row in left} == {"not_matured"}
    assert (scored["exit_session"] == "2026-10-22").all()
    assert (scored["entry_session"] == "2026-10-15").all()
    assert written.payload["retrieved_by"] == "2026-10-23T23:00:00+00:00"
    assert written.payload["outcome_data"]["complete"] is True
    assert written.payload["outcome_data"]["batches"]
    variant = forecast_entry.payload["bundle"]
    assert (scored["bundle_id"] == variant["bundle_id"]).all()
    assert (scored["config_sha256"] == variant["config_sha256"]).all()
    assert {
        f"{m}@v{v}" for m, v in zip(scored["model_id"], scored["model_version"], strict=True)
    } <= {str(e.ref) for e in prospective_frozen.result.entries}
    row = scored.iloc[0]
    assert row["inside_band"] == (row["band_low"] <= row["outcome"] <= row["band_high"])
    assert row["error"] == pytest.approx(row["outcome"] - row["point"])
    assert row["null_error"] == pytest.approx(row["outcome"] - row["null"])
    # the outcome is the F5.1 log total return over (entry, exit] of the stored rows
    prices, distributions, splits = prospective_tables()
    sessions = nyse_rules.sessions(WINDOW_START, "2026-12-31")
    returns = training.series.daily_log_returns(
        sessions, prices, distributions, splits, row["asset"]
    )
    dates = returns["date"].dt.strftime("%Y-%m-%d")
    inside = (dates > "2026-10-15") & (dates <= "2026-10-22")
    expected = returns.loc[inside, "log_return"]
    if row["target"] == "total_log_return":
        assert row["outcome"] == pytest.approx(expected.sum())
    else:
        assert row["outcome"] == pytest.approx(0.5 * np.log(np.mean(expected**2)))
    # nothing is scored twice, and the forecast entry is untouched
    again, _ = score(ledger_path, "2026-10-23", "2026-10-24T00:00:00Z", store)
    assert again is None
    entries = ledger.read_ledger(ledger_path)
    assert entries[0] == forecast_entry
    final, left = score(ledger_path, "2026-12-31", "2027-01-01T00:00:00Z", store)
    assert len(final.payload["scores"]) == 12 and not left
    assert len(forecast.scored_keys(ledger.read_ledger(ledger_path))) == 18


def test_nothing_is_scored_before_its_exit_close_is_known(prospective_frozen, ledger_path, store):
    record(prospective_frozen, ledger_path, store)
    written, left = score(ledger_path, "2026-10-22", "2026-10-22T21:00:00Z", store)  # 17:00 NY
    assert written is None
    assert {row["reason"] for row in left} <= {
        "close_not_available",
        "outcome_inputs_unavailable",
        "not_matured",
    }


def test_the_cumulative_report_states_its_small_sample(prospective_frozen, ledger_path, store):
    record(prospective_frozen, ledger_path, store)
    written, _ = score(ledger_path, "2026-12-31", "2027-01-01T00:00:00Z", store)
    report = forecast.coverage_report(ledger.read_ledger(ledger_path))
    assert len(report) == 18 and (report["n_scored"] == 1).all()
    assert report["caveat"].str.contains("far too few").all()
    assert written.payload["cumulative"] == json.loads(json.dumps(report.to_dict("records")))
    low, high = forecast.wilson_interval(8, 10)
    assert 0.0 < low < 0.8 < high < 1.0


def test_an_outcome_with_a_missing_close_stays_pending(prospective_frozen, ledger_path, tmp_path):
    gap = prospective_store(
        tmp_path / "gap", edit=without_close("VTI", ["2026-10-20"], from_retrieval="2026-10-22")
    )
    record(prospective_frozen, ledger_path, gap)
    written, left = score(ledger_path, "2026-10-23", "2026-10-23T23:00:00Z", gap)
    reasons = {(row["asset"], row["horizon"]): row["reason"] for row in left}
    assert reasons[("VTI", 5)] == "outcome_inputs_unavailable"
    assert {s["asset"] for s in written.payload["scores"]} == {"AGG", "SHV"}
    assert written.payload["outcome_data"]["complete"] is False


def test_a_zero_volatility_outcome_is_scored_outside_the_band(
    prospective_frozen, ledger_path, tmp_path
):
    days = nyse_rules.sessions("2026-10-15", "2026-10-22")

    def flat(retrieval, tables):
        prices, distributions, splits = tables
        if retrieval >= "2026-10-22":
            shv = prices["fund_id"] == "SHV"
            entry_close = prices.loc[shv & (prices["session_date"] == days[0]), "close"].iloc[0]
            prices = prices.copy()
            prices.loc[shv & prices["session_date"].isin(days), "close"] = float(entry_close)
        return prices, distributions, splits

    flat_store = prospective_store(tmp_path / "flat", edit=flat)
    record(prospective_frozen, ledger_path, flat_store)
    written, _ = score(ledger_path, "2026-10-23", "2026-10-23T23:00:00Z", flat_store)
    scores = pd.DataFrame(written.payload["scores"]).set_index(["asset", "target"])
    flat_vol = scores.loc[("SHV", "log_realized_volatility")]
    assert flat_vol["inside_band"] == False
    assert flat_vol["outcome_note"] == "zero_realized_volatility"
    assert pd.isna(flat_vol["outcome"]) and flat_vol["outcome_display_percent"] == 0.0
    assert scores.loc[("SHV", "total_log_return"), "outcome"] == 0.0


def test_models_registered_after_the_forecast_are_refused(prospective_frozen, ledger_path, store):
    models = prospective_frozen.models()
    later = {
        key: {**value, "registered_at": "2026-10-15T00:00:00+00:00"}
        for key, value in models.entries.items()
    }
    with pytest.raises(forecast.ForecastRefused) as refused:
        record(prospective_frozen, ledger_path, store, models=forecast.Models(models.bundle, later))
    assert refused.value.reason == "not_frozen"


def test_freezing_counts_rows_dropped_for_zero_volatility(prospective_frozen):
    fit = prospective_frozen.result.bundle.fit("SHV", 5, "log_realized_volatility")
    assert fit["development"]["zero_volatility_rows_dropped"] == 0  # GARCH fixture: never flat


def test_a_re_freeze_is_never_pooled_with_earlier_forecasts():
    def one(bundle, version, inside):
        return {
            "bundle_id": bundle,
            "config_sha256": bundle * 4,
            "model_id": "m.ret.h5.ridge",
            "model_version": version,
            "asset": "VTI",
            "horizon": 5,
            "target": "total_log_return",
            "inside_band": inside,
            "error": 0.1,
            "null_error": 0.2,
        }

    table = forecast.coverage_table([one("a", 1, True), one("a", 1, True), one("b", 2, False)])
    assert len(table) == 2
    assert table.set_index("bundle_id")["n_scored"].to_dict() == {"a": 2, "b": 1}
    assert table.set_index("bundle_id")["coverage"].to_dict() == {"a": 1.0, "b": 0.0}


def test_freezing_requires_complete_provenance(prospective_frozen, tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        freeze.freeze(
            prospective_frozen.config,
            research_store_root=prospective_frozen.store,
            output_root=tmp_path,
            provenance={"code": {}},
            started_at=pd.Timestamp("2026-10-14T12:00Z"),
        )
