"""
Training, development evaluation and the frozen refit of the prospective
forecasters (docs/research/ml_prospective.md).

Data: the F1 research store only (data.funds.reader), sessions up to
data.training_end (2024-12-31). Any input dated after it is refused with a
LeakageError before anything is computed, and the frames, folds and final
fits check it again.

For every horizon h, asset and target (per-asset models, as in the ETF
track):

    frame     one row per session with a close: the five features and both
              targets; a row is labeled only when both targets are (so the
              return and volatility folds are identical)
    folds     purged expanding folds on that asset's rows, the same outer
              windows for every asset, development mode, lockbox closed
    models    historical_mean (the asset's expanding training-label mean,
              the null) and ridge (T4's grid, nested purged selection inside
              every outer fold)
    bands     split-conformal (bands.py); development coverage is measured
              sequentially, out of sample

The frozen refit fits ridge once per (asset, h, target) on every training
row, its penalty chosen by the same nested selection on the last inner
windows before the lockbox, and keeps the fitted statistics, the null's
value and the band quantiles as plain numbers (frozen.py).

Zero realized volatility (zero_vol.py): the labels follow the rule in force
(`zero_vol.CURRENT`) unless research code passes the unapproved option
`zero_vol.B2` to `training_data`. Freezing always uses the rule in force.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from stock_agent.calendars import nyse
from stock_agent.features.contracts import validate_feature_contracts
from stock_agent.features.frozen_universe import (
    FrozenUniverse,
    UniverseRule,
    eligible_keys,
    freeze_universe,
)
from stock_agent.features.training import (
    TARGET_END_COLUMN,
    TARGET_RETURN_COLUMN,
    TARGET_START_COLUMN,
    LabelSpec,
)
from stock_agent.forecasting import etf, etf_data
from stock_agent.forecasting import metrics as em
from stock_agent.forecasting.prospective import bands, series, zero_vol
from stock_agent.forecasting.prospective.config import ProspectiveConfig
from stock_agent.forecasting.prospective.synthetic import FIXTURE_ADAPTER_ID
from stock_agent.model_diagnostics import inference as inf
from stock_agent.model_diagnostics import linear, nested, scoring
from stock_agent.model_diagnostics import runner as t2
from stock_agent.model_diagnostics.controls import PooledMeanForecast
from stock_agent.model_diagnostics.seeds import derive_seed
from stock_agent.model_validation.audit import (
    LeakageError,
    frame_content_sha256,
    frame_rows_sha256,
    table_sha256,
)
from stock_agent.model_validation.folds import (
    MODEL_HOLDOUT_START,
    WalkForwardFold,
    is_labeled,
    make_expanding_folds,
    make_test_windows,
)
from stock_agent.model_validation.harness import run_walk_forward

FEATURES = series.FEATURES
NULL = "historical_mean"
CANDIDATE = "ridge"
FINAL_FOLD_ID = 1_000_000  # the refit "fold": train on everything before the lockbox
METRIC_COLUMNS = (*em.COLUMNS, "target", "asset", "horizon")
# A closed map from the F1 adapter that produced a batch to what its data can
# support. Real fund data is a hand-picked, hindsight-selected list of three
# funds, so it is an engineering sandbox, never a registered research panel.
DATA_CLASS_BY_ADAPTER = {
    "synthetic_funds": "synthetic",
    FIXTURE_ADAPTER_ID: "synthetic",
    "tiingo_daily": "engineering_sandbox",
}
ENGINEERING_NOTE = {
    "synthetic": "Synthetic fixture data: verifies the pipeline; never evidence about funds.",
    "engineering_sandbox": (
        "Real fund data for three hand-picked funds (hindsight-selected): development "
        "results are engineering evidence only."
    ),
}


@dataclass(frozen=True, eq=False)
class TrainingData:
    frames: Mapping[tuple[str, int], pd.DataFrame]  # (asset, horizon) -> labeled frame
    universe: FrozenUniverse
    data_class: str
    batch_ids: tuple[str, ...]
    fingerprints: Mapping[str, str]
    max_session: pd.Timestamp
    # (asset, horizon) -> rows whose return label exists but whose realized volatility is 0
    # (every return in the window was 0): log volatility is undefined, so both labels are
    # dropped. Counted, because dropping them biases development coverage upward. Always 0
    # under the research option B2, which drops nothing.
    zero_volatility_rows: Mapping[tuple[str, int], int]
    zero_vol_rule: str = zero_vol.DEFAULT
    # (asset, horizon) -> date, symbol, sigma_raw, measured_zero, floored and sigma_floor of
    # every row of that frame: measured zeros stay distinguishable from the modeling floor.
    volatility_diagnostics: Mapping[tuple[str, int], pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class Development:
    predictions: pd.DataFrame
    bands: pd.DataFrame
    metrics: pd.DataFrame
    checks: pd.DataFrame
    selection: pd.DataFrame
    lockbox: Mapping[str, object]
    n_folds: Mapping[str, int]


def label_id(target: str, horizon: int, entry_lag: int) -> str:
    return f"{target}:f5_1_total_return:horizon={horizon}:entry_lag={entry_lag}"


def target_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    """The frame with `target` as the label column the harness reads."""

    return frame.assign(**{TARGET_RETURN_COLUMN: frame[f"y_{target}"]})


def _refuse_after(name: str, values: pd.Series, last: pd.Timestamp) -> None:
    dates = pd.to_datetime(values, utc=True)
    if (dates > last).any():
        raise LeakageError(
            f"Training refuses {name} after {last.date()} (latest {dates.max().date()}): "
            "the sealed period is never trained, tuned or evaluated on."
        )


def data_class_of(batches: pd.DataFrame) -> str:
    adapters = sorted(set(batches["adapter_id"]))
    unknown = [a for a in adapters if a not in DATA_CLASS_BY_ADAPTER]
    if not adapters or unknown:
        raise ValueError(f"No data class is defined for F1 adapters {unknown or adapters}.")
    classes = {DATA_CLASS_BY_ADAPTER[a] for a in adapters}
    if len(classes) != 1:
        raise ValueError(f"The store mixes data classes {sorted(classes)}; train on one.")
    return classes.pop()


def training_data(
    prices: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
    batches: pd.DataFrame,
    config: ProspectiveConfig,
    *,
    zero_vol_rule: str | None = None,
) -> TrainingData:
    """
    Frames for every (asset, horizon) from F1 research-store tables, with the
    log-volatility label under `zero_vol_rule`: the config's registered rule
    by default. Research code may pass `zero_vol.B2`, a research-only option
    the owner has not approved; freeze refuses data built under any rule other
    than the config's.
    """

    zero_vol_rule = config.zero_volatility_rule if zero_vol_rule is None else zero_vol_rule
    zero_vol.check_rule(zero_vol_rule)
    last = config.training_end
    _refuse_after("price sessions", prices["session_date"], last)
    for column in ("ex_date", "declaration_date", "record_date", "pay_date"):
        if column in distributions:
            _refuse_after(f"distribution {column}s", distributions[column].dropna(), last)
    _refuse_after("split dates", splits["effective_date"], last)
    data_class = data_class_of(batches)
    assets = list(config.assets)
    selected = {
        "prices": prices.loc[prices["fund_id"].isin(assets)],
        "distributions": distributions.loc[distributions["fund_id"].isin(assets)],
        "splits": splits.loc[splits["fund_id"].isin(assets)],
    }
    base, observed = {}, []
    for asset in assets:
        own = selected["prices"].loc[
            (selected["prices"]["fund_id"] == asset)
            & (selected["prices"]["adjustment_basis"] == series.UNADJUSTED)
        ]
        if own.empty:
            raise ValueError(f"The research store has no unadjusted closes for {asset}.")
        dates = pd.to_datetime(own["session_date"], utc=True)
        sessions = nyse.sessions(dates.min(), dates.max())
        returns = series.daily_log_returns(
            sessions, selected["prices"], selected["distributions"], selected["splits"], asset
        )
        frame = series.asset_frame(returns, asset)
        base[asset] = frame
        closes = frame.loc[frame["close_observed"], ["date", "symbol"]]
        observed.append(closes.assign(close=1.0))
    universe = freeze_universe(
        pd.concat(observed, ignore_index=True),
        UniverseRule(config.data.min_history_sessions),
        universe_id=config.universe_id,
        survivorship_free=data_class == "synthetic",
    )
    keys = eligible_keys(universe)
    contracts = {name: etf_data.CONTRACTS_BY_NAME[name] for name in FEATURES}
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    zero_volatility: dict[tuple[str, int], int] = {}
    diagnostics: dict[tuple[str, int], pd.DataFrame] = {}
    diagnostic_columns = list(zero_vol.DIAGNOSTIC_COLUMNS)
    for asset, frame in base.items():
        closes = zero_vol.as_of_closes(frame["date"], selected["prices"], asset)
        for horizon in config.horizons:
            labeled = frame.copy()
            for target in config.targets:
                if target == "log_realized_volatility":
                    labels = zero_vol.volatility_labels(
                        frame["date"],
                        frame["log_return"],
                        closes,
                        horizon=horizon,
                        entry_lag=config.entry_lag,
                        rule=zero_vol_rule,
                    )
                    flags = labels.loc[:, diagnostic_columns]
                else:
                    labels = series.labels_from_log_returns(
                        frame["date"],
                        frame["log_return"],
                        horizon=horizon,
                        entry_lag=config.entry_lag,
                        target=target,
                    )
                labeled[f"y_{target}"] = labels[TARGET_RETURN_COLUMN]
                labeled[TARGET_START_COLUMN] = labels[TARGET_START_COLUMN]
                labeled[TARGET_END_COLUMN] = labels[TARGET_END_COLUMN]
            targets = [f"y_{target}" for target in config.targets]
            both = np.isfinite(labeled[targets].to_numpy(np.float64)).all(axis=1)
            if {"total_log_return", "log_realized_volatility"} <= set(config.targets):
                flat = (
                    np.isfinite(labeled["y_total_log_return"].to_numpy(np.float64))
                    & ~both
                    & labeled["close_observed"].to_numpy()
                )
                zero_volatility[(asset, horizon)] = int(flat.sum())
            for column in targets:  # labeled only when every target is
                labeled.loc[~both, column] = np.nan
            if "log_realized_volatility" in config.targets:
                labeled = labeled.join(flags)
            labeled = labeled.loc[labeled["close_observed"]].merge(
                keys, on=["date", "symbol"], how="inner", validate="one_to_one"
            )
            labeled = labeled.reset_index(drop=True)
            if "log_realized_volatility" in config.targets:
                diagnostics[(asset, horizon)] = labeled.loc[:, ["date", "symbol", *flags]]
                labeled = labeled.drop(columns=diagnostic_columns)
            labeled[TARGET_RETURN_COLUMN] = labeled[targets[0]]
            validate_feature_contracts(labeled, list(FEATURES), contracts)
            frames[(asset, horizon)] = labeled
    stacked = pd.concat([f for (_, h), f in frames.items() if h == config.horizons[0]])
    fingerprints = {
        "prices_sha256": table_sha256(
            selected["prices"].sort_values(["fund_id", "session_date"]).reset_index(drop=True)
        ),
        "distributions_sha256": table_sha256(
            selected["distributions"].sort_values(["fund_id", "ex_date"]).reset_index(drop=True)
        ),
        "splits_sha256": table_sha256(
            selected["splits"].sort_values(["fund_id", "effective_date"]).reset_index(drop=True)
        ),
        "features_sha256": frame_content_sha256(stacked, list(FEATURES)),
        "universe_sha256": universe.sha256,
    }
    batch_ids = tuple(sorted(set(batches["batch_id"])))
    max_session = max(frame["date"].max() for frame in base.values())
    return TrainingData(
        frames,
        universe,
        data_class,
        batch_ids,
        fingerprints,
        max_session,
        zero_volatility,
        zero_vol_rule,
        diagnostics,
    )


def load_training_data(
    store_root, config: ProspectiveConfig, *, zero_vol_rule: str | None = None
) -> TrainingData:
    """`training_data` from an F1 research store, read by its fail-closed reader."""

    from stock_agent.data.funds.reader import read_store

    store = read_store(store_root)
    return training_data(
        store.prices,
        store.distributions,
        store.splits,
        store.batches,
        config,
        zero_vol_rule=zero_vol_rule,
    )


def _design(config: ProspectiveConfig) -> nested.InnerDesign:
    return nested.InnerDesign(
        block_sessions=config.folds.inner_block_sessions,
        n_windows=config.folds.inner_windows,
        min_train_sessions=config.folds.inner_min_train_sessions,
    )


def ridge_spec(config: ProspectiveConfig, target: str, horizon: int):
    """T4's nested-selected ridge on the five features, labelled for this target."""

    label_spec = LabelSpec(horizon, config.entry_lag, price_column="close")
    specs = nested.linear_models(label_spec, FEATURES, _design(config), include_ols=False)
    spec = next(s for s in specs if s.name == CANDIDATE)
    return replace(
        spec, params={**spec.params, "label_id": label_id(target, horizon, config.entry_lag)}
    )


def _check_frame_before_lockbox(frame: pd.DataFrame, config: ProspectiveConfig, scope: str):
    if frame["date"].max() > config.training_end:
        raise LeakageError(f"{scope}: a training row is dated after {config.training_end.date()}.")
    ends = pd.to_datetime(frame[TARGET_END_COLUMN], utc=True)
    if (ends[is_labeled(frame)] >= MODEL_HOLDOUT_START).any():
        raise LeakageError(f"{scope}: a training label ends in the sealed period.")


def _metric_rows(rows: list[dict], target: str, asset: str, horizon: int) -> list[dict]:
    return [{**row, "target": target, "asset": asset, "horizon": horizon} for row in rows]


def _qlike(label: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Patton's QLIKE on variances, from log volatilities."""

    ratio = np.exp(2.0 * (label - forecast))
    return ratio - 2.0 * (label - forecast) - 1.0


def develop(data: TrainingData, config: ProspectiveConfig) -> Development:
    """Out-of-fold predictions, bands and development metrics for every piece."""

    checks: list[dict] = []
    selection: list[dict] = []
    frames: list[pd.DataFrame] = []
    max_dates, max_ends, n_folds = [], [], {}
    fold_starts: dict[tuple[str, int], dict[int, pd.Timestamp]] = {}
    for horizon in config.horizons:
        own = {a: data.frames[(a, horizon)] for a in config.assets}
        labeled = pd.concat([f.loc[is_labeled(f), "date"] for f in own.values()])
        sessions = pd.concat([f["date"] for f in own.values()]).drop_duplicates().sort_values()
        windows = make_test_windows(
            sessions,
            start=pd.Timestamp(config.folds.start, tz="UTC"),
            end=labeled.max() + pd.Timedelta(days=1),
            block_sessions=config.folds.block_sessions,
        )
        for asset, frame in own.items():
            for target in config.targets:
                scope = f"{target}:h{horizon}:{asset}"
                piece_frame = target_frame(frame, target)
                _check_frame_before_lockbox(piece_frame, config, scope)
                folds = make_expanding_folds(
                    piece_frame, windows, holdout_start=MODEL_HOLDOUT_START
                )
                lockbox = t2.lockbox_evidence(piece_frame, folds, MODEL_HOLDOUT_START)
                late = t2.asof_violations(piece_frame, folds, lockbox)
                if late:
                    raise LeakageError(f"{scope}: folds {late} score unmatured labels.")
                checks.append(
                    t2._check("lockbox_closed", scope, "leakage", "held_out_rows_total", 0, 0, "==")
                )
                checks.append(
                    t2._check(
                        "evaluation_asof_matches_folds", scope, "leakage", "late_folds", 0, 0, "=="
                    )
                )
                max_dates.append(pd.Timestamp(lockbox["max_date"]))
                max_ends.append(pd.Timestamp(lockbox["max_target_end_date"]))
                n_folds[scope] = len(folds)
                fold_starts[(asset, horizon)] = {f.fold_id: f.test_start for f in folds}
                piece = etf.AssetTarget(f"{target}:h{horizon}", asset, piece_frame, folds)
                entries = {
                    NULL: ("null", PooledMeanForecast, [FEATURES[0]]),
                    CANDIDATE: ("candidate", ridge_spec(config, target, horizon), list(FEATURES)),
                }
                for name, entry in entries.items():
                    predictions = etf._predict(piece, name, entry, checks, selection)
                    frames.append(predictions.assign(target=target, horizon=horizon))
    predictions = pd.concat(frames, ignore_index=True)
    band_frame = _bands(predictions, config, fold_starts)
    metrics = _metrics(predictions, band_frame, config)
    for row in selection:
        target, horizon = str(row["target"]).split(":h")
        row["target"], row["horizon"] = target, int(horizon)
    return Development(
        predictions=predictions,
        bands=band_frame,
        metrics=pd.DataFrame(metrics, columns=list(METRIC_COLUMNS)),
        checks=pd.DataFrame(checks),
        selection=pd.DataFrame(selection),
        lockbox={
            "max_date": max(max_dates).isoformat(),
            "max_target_end_date": max(max_ends).isoformat(),
        },
        n_folds=n_folds,
    )


KEYS = ["fold_id", "date", "symbol"]


def _ridge(predictions: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    own = predictions.loc[
        (predictions["model"] == CANDIDATE)
        & (predictions["target"] == target)
        & (predictions["horizon"] == horizon)
    ]
    return own.reset_index(drop=True)


def score_frame(returns: pd.DataFrame, volatility: pd.DataFrame, horizon: int, target: str):
    """Ridge rows with `point` and `scale` (bands.py) for one target."""

    joined = returns.merge(
        volatility[[*KEYS, "prediction"]],
        on=KEYS,
        how="inner",
        suffixes=("", "_vol"),
        validate="one_to_one",
    )
    if len(joined) != len(returns) or len(joined) != len(volatility):
        raise ValueError("Return and volatility forecasts must cover identical rows.")
    if target == "total_log_return":
        point, scale = joined["prediction"], np.exp(joined["prediction_vol"]) * np.sqrt(horizon)
        base = joined
    else:
        base = volatility.merge(joined[KEYS], on=KEYS, how="inner")
        point, scale = base["prediction"], pd.Series(1.0, index=base.index)
    return base.assign(point=point.to_numpy(), scale=np.asarray(scale, dtype=np.float64))


def _bands(predictions, config: ProspectiveConfig, fold_starts) -> pd.DataFrame:
    z = inf.normal_ppf(config.band.upper_quantile)
    pieces = []
    for horizon in config.horizons:
        returns = _ridge(predictions, "total_log_return", horizon)
        volatility = _ridge(predictions, "log_realized_volatility", horizon)
        for target in config.targets:
            scored = score_frame(returns, volatility, horizon, target)
            for asset, part in scored.groupby("symbol", sort=True):
                banded = bands.sequential_bands(
                    part.reset_index(drop=True),
                    fold_starts[(asset, horizon)],
                    lower_q=config.band.lower_quantile,
                    upper_q=config.band.upper_quantile,
                    min_calibration=config.band.min_calibration,
                )
                if target == "total_log_return":
                    banded["plugin_low"] = banded["point"] - z * banded["scale"]
                    banded["plugin_high"] = banded["point"] + z * banded["scale"]
                else:
                    banded["plugin_low"] = np.nan
                    banded["plugin_high"] = np.nan
                pieces.append(banded.assign(target=target, horizon=horizon))
    columns = [
        *KEYS,
        "target",
        "horizon",
        TARGET_START_COLUMN,
        TARGET_END_COLUMN,
        TARGET_RETURN_COLUMN,
        "point",
        "scale",
        "band_low",
        "band_high",
        "n_calibration",
        "plugin_low",
        "plugin_high",
    ]
    return pd.concat(pieces, ignore_index=True).loc[:, columns]


def _coverage(rows: pd.DataFrame, low: str, high: str, ctx, level: float) -> list[dict]:
    scored = rows.rename(columns={low: "interval_low", high: "interval_high"})
    return em.interval_coverage(CANDIDATE, "candidate", scored, ctx, level=level)


def _metrics(predictions, band_frame, config: ProspectiveConfig) -> list[dict]:
    rows: list[dict] = []
    level = config.band.level
    for horizon in config.horizons:
        for target in config.targets:
            own = predictions.loc[
                (predictions["target"] == target) & (predictions["horizon"] == horizon)
            ]
            for asset in config.assets:
                mine = own.loc[own["symbol"] == asset]
                model, base = (mine.loc[mine["model"] == name] for name in (CANDIDATE, NULL))
                ctx = scoring.build_context(
                    base,
                    block_length=inf.DEFAULT_BLOCK_LENGTH,
                    bootstrap_reps=config.evaluation.bootstrap_reps,
                    seed=derive_seed(config.evaluation.seed, "bootstrap", target, asset, horizon),
                )
                r2, improvement, _ = scoring.oos_r2_vs(model, base, ctx)
                rows += _metric_rows(
                    em.rows(CANDIDATE, "candidate", f"oos_r2_vs_{NULL}", [r2]),
                    target,
                    asset,
                    horizon,
                )
                rows += _metric_rows(
                    em.rows(CANDIDATE, "candidate", f"mse_improvement_vs_{NULL}", improvement),
                    target,
                    asset,
                    horizon,
                )
                if target == "log_realized_volatility":
                    joined = scoring.aligned_pair(model, base, TARGET_RETURN_COLUMN)
                    positions = scoring._positions(joined, ctx)
                    label = joined[TARGET_RETURN_COLUMN].to_numpy(np.float64)
                    gain = _qlike(label, joined["prediction_baseline"].to_numpy(np.float64))
                    gain -= _qlike(label, joined["prediction"].to_numpy(np.float64))
                    series_ = scoring.session_mean(gain, positions, ctx)
                    rows += _metric_rows(
                        em.rows(
                            CANDIDATE,
                            "candidate",
                            f"qlike_improvement_vs_{NULL}",
                            scoring.mean_inferences(series_, ctx),
                        ),
                        target,
                        asset,
                        horizon,
                    )
                banded = band_frame.loc[
                    (band_frame["target"] == target)
                    & (band_frame["horizon"] == horizon)
                    & (band_frame["symbol"] == asset)
                    & band_frame["band_low"].notna()
                ]
                if banded.empty:
                    continue
                band_ctx = scoring.build_context(
                    banded,
                    block_length=inf.DEFAULT_BLOCK_LENGTH,
                    bootstrap_reps=config.evaluation.bootstrap_reps,
                    seed=derive_seed(config.evaluation.seed, "bands", target, asset, horizon),
                )
                for low, high, name in (
                    ("band_low", "band_high", "band_coverage"),
                    ("plugin_low", "plugin_high", "plugin_coverage"),
                ):
                    if banded[low].isna().all():
                        continue
                    coverage = _coverage(banded, low, high, band_ctx, level)
                    for row in coverage:
                        row["metric"] = row["metric"].replace("interval_coverage", name)
                    rows += _metric_rows(coverage, target, asset, horizon)
    return rows


def _final_fold(frame: pd.DataFrame) -> WalkForwardFold:
    """Every labeled row before the lockbox as the training set of one refit."""

    labeled = is_labeled(frame)
    ends = pd.to_datetime(frame[TARGET_END_COLUMN], utc=True)
    train = np.flatnonzero(labeled & (ends < MODEL_HOLDOUT_START).to_numpy()).astype(np.int64)
    empty = np.array([], dtype=np.int64)
    return WalkForwardFold(
        fold_id=FINAL_FOLD_ID,
        mode="development",
        holdout_start=MODEL_HOLDOUT_START,
        train_start=None,
        test_start=MODEL_HOLDOUT_START,
        test_end=MODEL_HOLDOUT_START + pd.Timedelta(days=1),
        train_rows=train,
        test_rows=empty,
        purged_rows=empty,
        unlabeled_rows=np.flatnonzero(~labeled).astype(np.int64),
        held_out_rows=empty,
        frame_rows_sha256=frame_rows_sha256(frame),
    )


def final_fit(frame: pd.DataFrame, config: ProspectiveConfig, target: str, horizon: int):
    """
    One ridge fit on every training row, its penalty chosen by nested purged
    selection on the last inner windows before the lockbox. Returns the
    fitted LinearForecast, the chosen grid point, the selection scores, the
    inner-fold bounds and the training fold.
    """

    spec = ridge_spec(config, target, horizon)
    scope = f"{target}:h{horizon}:{frame['symbol'].iloc[0]}"
    _check_frame_before_lockbox(frame, config, scope)
    outer = _final_fold(frame)
    inner = nested.build_inner_folds(frame, outer, _design(config))  # verifies inside training
    base = nested.estimator_kwargs(spec.params)
    grid = spec.params["selection"]["grid"]
    outputs = []
    for point in grid:

        def make(point=point):
            return spec.estimator(**{**base, **point})

        with threadpool_limits(limits=1):
            result = run_walk_forward(
                frame,
                inner,
                feature_columns=list(FEATURES),
                make_estimator=make,
                lockbox_start=MODEL_HOLDOUT_START,
            )
        outputs.append(result.predictions)
    chosen = linear.select_hyperparameters(grid, outputs)
    estimator = spec.estimator(**{**base, **grid[chosen.chosen]})
    X = frame.loc[:, list(FEATURES)].to_numpy(dtype=np.float64)
    y = frame[TARGET_RETURN_COLUMN].to_numpy(dtype=np.float64)
    with threadpool_limits(limits=1):
        estimator.fit(X[outer.train_rows], y[outer.train_rows])
    return estimator, dict(grid[chosen.chosen]), chosen, nested.inner_bounds(frame, inner), outer
