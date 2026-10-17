"""
The development-data check of the zero-volatility rules
(docs/research/zero_vol_floor.md, section 6). Research only: it reads F1
research-store tables (sessions up to 2024-12-31), never the prospective
store or the sealed period, and freezes, registers and records nothing.

Under the rule in force (zero_vol.CURRENT) and the unapproved option
zero_vol.B2 it reports:

    windows        per asset x horizon x as-of year: eligible windows (all h
                   returns known), measured zeros, B2-floored windows, and
                   floored windows that are not measured zeros
    cent grid      per asset x year: the share of unadjusted closes on the
                   $0.01 grid, which the floor's tick premise needs
    coverage       development 80% band coverage (sequential, out of fold):
                   the fold-block estimate and standard error, and the
                   circular block-bootstrap 95% interval; also the CURRENT
                   bands re-scored with every measured-zero row in a scored
                   fold as a miss, which is what prospective scoring does
    zero split     coverage on measured-zero and non-zero rows separately
    band width     the mean width of the scored bands
    QLIKE          the improvement over the historical-mean null
    shift          the mean residual shift, B2 minus CURRENT, on non-zero rows

It then evaluates the pre-stated acceptance criteria Z1-Z4 in that order and
stops at the first failure: the remaining criteria are not evaluated, and the
failure is reported, never resolved by switching rules. Only when every
criterion passes does it add the registration power estimates (development
loss differentials under B2, on non-overlapping outcomes).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from stock_agent.features.training import TARGET_END_COLUMN, TARGET_START_COLUMN
from stock_agent.forecasting import metrics as em
from stock_agent.forecasting.prospective import bands, registration, training, zero_vol
from stock_agent.forecasting.prospective.config import ProspectiveConfig
from stock_agent.model_diagnostics import inference as inf
from stock_agent.model_diagnostics import scoring
from stock_agent.model_diagnostics.seeds import derive_seed
from stock_agent.model_validation.audit import table_sha256

VOL = "log_realized_volatility"
RET = "total_log_return"
CURRENT_AS_MISS = "current_zero_as_miss"  # the CURRENT bands, measured zeros scored as misses
NEVER_FLOORED = ("VTI", "AGG")  # pre-stated: the floor must never bind for these funds
FLOOR_ASSET = "SHV"  # the fund whose $0.01 tick can make a window flat
COVERAGE_TOLERANCE = 0.05
FLOORED_NOT_ZERO_MAX = 0.01
RESIDUAL_SHIFT_MAX = 0.05
GRID_TOLERANCE_CENTS = 1e-6
QLIKE_METRIC = f"qlike_improvement_vs_{training.NULL}"
PASS, FAIL, NOT_EVALUATED = "pass", "fail", "not_evaluated"
CRITERIA = (
    (
        "Z1",
        (
            "VTI, AGG and every asset-horizon without a floored window give bit-identical "
            "development predictions, bands, metrics and selections under both rules."
        ),
    ),
    (
        "Z2",
        (
            "SHV volatility-band coverage under B2 (fold-block estimate) is within 0.80 +- 0.05 "
            "at every horizon, and no lower than the current bands with measured zeros as misses."
        ),
    ),
    (
        "Z3",
        (
            "Floored windows that are not measured zeros are at most 1% of the eligible windows "
            "of every asset and horizon."
        ),
    ),
    (
        "Z4",
        (
            "The mean residual shift (B2 minus current, log volatility) on non-zero development "
            "rows is at most 0.05 in absolute value for every asset and horizon."
        ),
    ),
)
LABELS = (
    "research only: B2 is not approved and nothing here changes the rule in force",
    "development data to 2024-12-31 only; no sealed-period or prospective data",
    "no model is frozen or registered by this check",
)


class CheckRefused(ValueError):
    """Inputs the check refuses; `reason` is machine-readable."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True, eq=False)
class Evidence:
    """The training data and development of both rules, from the same store tables."""

    current: training.TrainingData
    b2: training.TrainingData
    dev_current: training.Development
    dev_b2: training.Development


@dataclass(frozen=True, eq=False)
class CheckReport:
    status: str  # PASS when every criterion passed, FAIL otherwise
    data_class: str
    batch_ids: tuple[str, ...]
    fingerprints: Mapping[str, str]
    tables: Mapping[str, pd.DataFrame]
    criteria: tuple[Mapping[str, object], ...]
    registration_power: pd.DataFrame | None
    labels: tuple[str, ...] = field(default=LABELS)


# --- inputs ---------------------------------------------------------------------------


def refuse_after(
    prices: pd.DataFrame, distributions: pd.DataFrame, splits: pd.DataFrame, last: pd.Timestamp
) -> None:
    """Refuse any price, distribution or split date after `last` (the training end)."""

    dated = [("price sessions", prices["session_date"])]
    for column in ("ex_date", "declaration_date", "record_date", "pay_date"):
        if column in distributions:
            dated.append((f"distribution {column}s", distributions[column].dropna()))
    dated.append(("split dates", splits["effective_date"]))
    for name, values in dated:
        stamps = pd.to_datetime(values, utc=True)
        if len(stamps) and (stamps > last).any():
            raise CheckRefused(
                "after_training_end",
                f"the store holds {name} after {last.date()} (latest {stamps.max().date()}); "
                "the zero-volatility check uses development data to 2024 only",
            )


def evidence(
    prices: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
    batches: pd.DataFrame,
    config: ProspectiveConfig,
) -> Evidence:
    """Training data and development under CURRENT and B2 (nothing frozen)."""

    refuse_after(prices, distributions, splits, config.training_end)
    tables = (prices, distributions, splits, batches, config)
    current = training.training_data(*tables, zero_vol_rule=zero_vol.CURRENT)
    b2 = training.training_data(*tables, zero_vol_rule=zero_vol.B2)
    return Evidence(current, b2, training.develop(current, config), training.develop(b2, config))


# --- descriptive tables ----------------------------------------------------------------


def diagnostics(data: training.TrainingData) -> pd.DataFrame:
    """Every frame row's volatility flags, with its horizon."""

    parts = [
        frame.assign(horizon=horizon)
        for (_, horizon), frame in sorted(data.volatility_diagnostics.items())
    ]
    return pd.concat(parts, ignore_index=True)


def window_counts(data_b2: training.TrainingData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per asset x horizon x as-of year, and per asset x horizon over every year."""

    flags = diagnostics(data_b2)
    flags = flags.loc[np.isfinite(flags["sigma_raw"].to_numpy(np.float64))].copy()
    flags["year"] = pd.DatetimeIndex(flags["date"]).year
    flags["floored_not_zero"] = flags["floored"] & ~flags["measured_zero"]
    # The smallest measured movement, in basis points: a value far below one tick (for
    # example a floating-point residual near 1e-12 bp) is kept as a label by the current rule.
    flags["sigma_raw_nonzero_bp"] = flags["sigma_raw"].where(~flags["measured_zero"]) * 1e4

    def counts(group_by: list[str]) -> pd.DataFrame:
        grouped = flags.groupby(group_by, sort=True)
        frame = pd.DataFrame(
            {
                "n_windows": grouped.size(),
                "n_zero": grouped["measured_zero"].sum(),
                "n_floored": grouped["floored"].sum(),
                "n_floored_not_zero": grouped["floored_not_zero"].sum(),
                "min_sigma_raw_nonzero_bp": grouped["sigma_raw_nonzero_bp"].min(),
            }
        ).reset_index()
        for name in ("zero", "floored_not_zero"):
            frame[f"share_{name}"] = frame[f"n_{name}"] / frame["n_windows"]
        return frame.rename(columns={"symbol": "asset"})

    return counts(["symbol", "horizon", "year"]), counts(["symbol", "horizon"])


def cent_grid(prices: pd.DataFrame, assets: Sequence[str]) -> pd.DataFrame:
    """Per asset x year: unadjusted closes, how many sit on the $0.01 grid, the worst offset."""

    own = prices.loc[
        prices["fund_id"].isin(assets) & (prices["adjustment_basis"] == "unadjusted")
    ].copy()
    cents = own["close"].to_numpy(np.float64) * 100.0
    own["offset_cents"] = np.abs(cents - np.rint(cents))
    own["on_grid"] = own["offset_cents"] <= GRID_TOLERANCE_CENTS
    own["year"] = pd.DatetimeIndex(pd.to_datetime(own["session_date"], utc=True)).year
    grouped = own.groupby(["fund_id", "year"], sort=True)
    frame = pd.DataFrame(
        {
            "n_closes": grouped.size(),
            "n_on_grid": grouped["on_grid"].sum(),
            "max_offset_cents": grouped["offset_cents"].max(),
        }
    ).reset_index()
    frame["share_on_grid"] = frame["n_on_grid"] / frame["n_closes"]
    return frame.rename(columns={"fund_id": "asset"})


def _inference(rows: Sequence[Mapping[str, object]], method: str) -> Mapping[str, object] | None:
    found = [row for row in rows if row["inference_method"] == method]
    return found[0] if found and found[0]["status"] == "ok" else None


def _summary(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """The fold-block estimate and standard error, and the bootstrap interval."""

    fold, boot = (_inference(rows, m) for m in (em.DECISION_METHOD, "circular_block_bootstrap"))
    return {
        "n": None if fold is None else int(fold["n"]),
        "estimate": None if fold is None else float(fold["estimate"]),
        "se": None if fold is None else float(fold["se"]),
        "boot_ci_low": None if boot is None else float(boot["ci_low"]),
        "boot_ci_high": None if boot is None else float(boot["ci_high"]),
    }


def _metric_summary(dev: training.Development, metric: str, rule: str) -> pd.DataFrame:
    frame = dev.metrics.loc[dev.metrics["metric"] == metric]
    rows = []
    for (target, asset, horizon), part in frame.groupby(["target", "asset", "horizon"]):
        rows.append(
            {
                "rule": rule,
                "asset": asset,
                "horizon": int(horizon),
                "target": target,
                **_summary(part.to_dict("records")),
            }
        )
    return pd.DataFrame(rows)


def _flag_lookup(flags: pd.DataFrame) -> pd.DataFrame:
    return flags.loc[:, ["date", "symbol", "horizon", "measured_zero", "floored"]]


def _check_windows(dev_current: training.Development, dev_b2: training.Development) -> None:
    """Both rules must share fold windows: each CURRENT fold lies inside the same B2 fold."""

    keys = ["symbol", "horizon", "target", "fold_id"]
    spans = {
        name: dev.bands.groupby(keys)["date"].agg(["min", "max"])
        for name, dev in (("current", dev_current), ("b2", dev_b2))
    }
    joined = spans["current"].join(spans["b2"], lsuffix="_c", rsuffix="_b", how="left")
    inside = (joined["min_c"] >= joined["min_b"]) & (joined["max_c"] <= joined["max_b"])
    if not inside.all():
        raise RuntimeError("The two rules' development folds do not share their test windows.")


def _coverage_rows(
    frame: pd.DataFrame, config: ProspectiveConfig, asset: str, horizon: int
) -> dict[str, object]:
    # The development metrics' own seed (training._metrics), so a cell without a measured
    # zero reproduces the CURRENT coverage row exactly.
    ctx = scoring.build_context(
        frame,
        block_length=inf.DEFAULT_BLOCK_LENGTH,
        bootstrap_reps=config.evaluation.bootstrap_reps,
        seed=derive_seed(config.evaluation.seed, "bands", VOL, asset, horizon),
    )
    scored = frame.rename(columns={"band_low": "interval_low", "band_high": "interval_high"})
    rows = em.interval_coverage(
        training.CANDIDATE, "candidate", scored, ctx, level=config.band.level
    )
    return _summary(rows)


def rescored_current(
    evidence_: Evidence, config: ProspectiveConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    The CURRENT volatility bands with every measured-zero row of a scored fold
    added as a miss (its log volatility is -inf): the coverage table and the
    rescored rows.
    """

    _check_windows(evidence_.dev_current, evidence_.dev_b2)
    flags = _flag_lookup(diagnostics(evidence_.b2))
    columns = ["fold_id", "date", "symbol", "horizon", "target_return", "band_low", "band_high"]
    current = evidence_.dev_current.bands
    current = current.loc[(current["target"] == VOL) & current["band_low"].notna(), columns]
    b2 = evidence_.dev_b2.bands
    b2 = b2.loc[b2["target"] == VOL, columns].merge(
        flags, on=["date", "symbol", "horizon"], how="left", validate="one_to_one"
    )
    rows, frames = [], []
    for (asset, horizon), mine in current.groupby(["symbol", "horizon"], sort=True):
        scored_folds = set(mine["fold_id"])
        zeros = b2.loc[
            (b2["symbol"] == asset)
            & (b2["horizon"] == horizon)
            & b2["measured_zero"].astype(bool)
            & b2["fold_id"].isin(scored_folds)
        ]
        zeros = zeros.assign(target_return=-np.inf, band_low=np.nan, band_high=np.nan)
        both = pd.concat(
            [mine.assign(measured_zero=False), zeros.loc[:, [*columns, "measured_zero"]]],
            ignore_index=True,
        )
        if both.duplicated(["date", "symbol"]).any():
            raise RuntimeError(f"{asset} h{horizon}: a measured-zero row was already scored.")
        both = both.sort_values(["date"]).reset_index(drop=True)
        frames.append(both)
        rows.append(
            {
                "rule": CURRENT_AS_MISS,
                "asset": asset,
                "horizon": int(horizon),
                "target": VOL,
                **_coverage_rows(both, config, asset, int(horizon)),
                "n_zero_as_miss": len(zeros),
            }
        )
    table = pd.DataFrame(rows)
    table["n_zero_as_miss"] = table["n_zero_as_miss"].astype("Int64")
    return table, pd.concat(frames, ignore_index=True)


def zero_split(evidence_: Evidence, rescored: pd.DataFrame) -> pd.DataFrame:
    """Volatility coverage on measured-zero and non-zero rows, per rule (point shares)."""

    flags = _flag_lookup(diagnostics(evidence_.b2))
    b2 = evidence_.dev_b2.bands
    b2 = b2.loc[(b2["target"] == VOL) & b2["band_low"].notna()].merge(
        flags, on=["date", "symbol", "horizon"], how="left", validate="one_to_one"
    )
    b2 = b2.assign(inside=bands.inside(b2))
    miss = rescored.assign(inside=bands.inside(rescored))
    rows = []
    for rule, frame in ((zero_vol.B2, b2), (CURRENT_AS_MISS, miss)):
        for (asset, horizon), part in frame.groupby(["symbol", "horizon"], sort=True):
            zero = part["measured_zero"].astype(bool)
            rows.append(
                {
                    "rule": rule,
                    "asset": asset,
                    "horizon": int(horizon),
                    "n_zero_rows": int(zero.sum()),
                    "coverage_zero_rows": float(part.loc[zero, "inside"].mean())
                    if zero.any()
                    else None,
                    "n_nonzero_rows": int((~zero).sum()),
                    "coverage_nonzero_rows": float(part.loc[~zero, "inside"].mean())
                    if (~zero).any()
                    else None,
                }
            )
    return pd.DataFrame(rows)


def band_widths(dev: training.Development, rule: str) -> pd.DataFrame:
    scored = dev.bands.loc[dev.bands["band_low"].notna()]
    width = (scored["band_high"] - scored["band_low"]).rename("width")
    grouped = scored.assign(width=width).groupby(["symbol", "horizon", "target"], sort=True)
    frame = grouped["width"].agg(["size", "mean"]).reset_index()
    frame = frame.rename(columns={"symbol": "asset", "size": "n", "mean": "mean_width"})
    return frame.assign(rule=rule).loc[:, ["rule", "asset", "horizon", "target", "n", "mean_width"]]


def residual_shift(evidence_: Evidence) -> pd.DataFrame:
    """Mean of (y_B2 - p_B2) - (y_C - p_C) on non-zero volatility rows scored by both."""

    keys = ["fold_id", "date", "symbol", "horizon"]

    def ridge(dev):
        own = dev.predictions
        own = own.loc[(own["model"] == training.CANDIDATE) & (own["target"] == VOL)]
        return own.assign(residual=own["target_return"] - own["prediction"]).loc[
            :, [*keys, "residual"]
        ]

    joined = ridge(evidence_.dev_current).merge(
        ridge(evidence_.dev_b2), on=keys, suffixes=("_c", "_b"), validate="one_to_one"
    )
    joined = joined.merge(
        _flag_lookup(diagnostics(evidence_.b2)),
        on=["date", "symbol", "horizon"],
        how="left",
        validate="many_to_one",
    )
    joined = joined.loc[~joined["measured_zero"].astype(bool)]
    joined["shift"] = joined["residual_b"] - joined["residual_c"]
    grouped = joined.groupby(["symbol", "horizon"], sort=True)["shift"]
    frame = grouped.agg(["size", "mean"]).reset_index()
    return frame.rename(columns={"symbol": "asset", "size": "n", "mean": "mean_shift"})


def bit_identity(
    evidence_: Evidence, config: ProspectiveConfig, totals: pd.DataFrame
) -> pd.DataFrame:
    """Per asset x horizon: whether every development table is identical under both rules."""

    def part(frame: pd.DataFrame, asset_column: str, asset: str, horizon: int) -> str:
        own = frame.loc[(frame[asset_column] == asset) & (frame["horizon"] == horizon)]
        return table_sha256(own.reset_index(drop=True))

    tables = (("predictions", "symbol"), ("bands", "symbol"), ("metrics", "asset"))
    floored = totals.set_index(["asset", "horizon"])["n_floored"]
    rows = []
    for asset in config.assets:
        for horizon in config.horizons:
            n_floored = int(floored.get((asset, horizon), 0))
            same = {
                name: part(getattr(evidence_.dev_current, name), column, asset, horizon)
                == part(getattr(evidence_.dev_b2, name), column, asset, horizon)
                for name, column in tables
            }
            same["selection"] = part(
                evidence_.dev_current.selection, "asset", asset, horizon
            ) == part(evidence_.dev_b2.selection, "asset", asset, horizon)
            rows.append(
                {
                    "asset": asset,
                    "horizon": horizon,
                    "n_floored": n_floored,
                    "required": asset in NEVER_FLOORED or n_floored == 0,
                    "identical": all(same.values()),
                    **{f"identical_{name}": value for name, value in same.items()},
                }
            )
    return pd.DataFrame(rows)


# --- the acceptance criteria -----------------------------------------------------------


def _z1(tables: Mapping[str, pd.DataFrame], config: ProspectiveConfig) -> tuple[bool, str]:
    identity = tables["bit_identity"]
    failed = identity.loc[identity["required"] & ~identity["identical"]]
    floored = identity.loc[identity["asset"].isin(NEVER_FLOORED) & (identity["n_floored"] > 0)]
    checked = int(identity["required"].sum())
    if failed.empty:
        return True, f"{checked} asset-horizons required to match, all bit-identical"
    cells = ", ".join(f"{a} h{h}" for a, h in zip(failed["asset"], failed["horizon"], strict=True))
    note = f"; the floor binds for {sorted(set(floored['asset']))}" if len(floored) else ""
    return False, f"not bit-identical: {cells}{note}"


def _z2(tables: Mapping[str, pd.DataFrame], config: ProspectiveConfig) -> tuple[bool, str]:
    coverage = tables["coverage"]
    level = config.band.level
    parts = []
    passed = True
    for horizon in config.horizons:
        own = coverage.loc[
            (coverage["asset"] == FLOOR_ASSET)
            & (coverage["horizon"] == horizon)
            & (coverage["target"] == VOL)
        ].set_index("rule")["estimate"]
        b2, miss = own.get(zero_vol.B2), own.get(CURRENT_AS_MISS)
        if b2 is None or miss is None or pd.isna(b2) or pd.isna(miss):
            passed = False
            parts.append(f"h{horizon}: coverage unavailable")
            continue
        ok = abs(b2 - level) <= COVERAGE_TOLERANCE + 1e-12 and b2 >= miss
        passed &= bool(ok)
        parts.append(f"h{horizon}: B2 {b2:.3f} vs zeros-as-misses {miss:.3f}")
    return passed, "; ".join(parts)


def _z3(tables: Mapping[str, pd.DataFrame], config: ProspectiveConfig) -> tuple[bool, str]:
    totals = tables["window_totals"]
    worst = totals.loc[totals["share_floored_not_zero"].idxmax()]
    over = totals.loc[totals["share_floored_not_zero"] > FLOORED_NOT_ZERO_MAX]
    text = (
        f"largest share {worst['share_floored_not_zero']:.4f} ({worst['asset']} "
        f"h{int(worst['horizon'])}, {int(worst['n_floored_not_zero'])} of "
        f"{int(worst['n_windows'])} windows)"
    )
    return over.empty, text


def _z4(tables: Mapping[str, pd.DataFrame], config: ProspectiveConfig) -> tuple[bool, str]:
    shift = tables["residual_shift"]
    expected = {(a, h) for a in config.assets for h in config.horizons}
    present = set(zip(shift["asset"], shift["horizon"], strict=True))
    if present != expected or shift["mean_shift"].isna().any():
        return False, f"residual shift unavailable for {sorted(expected - present)}"
    worst = shift.loc[shift["mean_shift"].abs().idxmax()]
    text = f"largest |shift| {abs(worst['mean_shift']):.4f} ({worst['asset']} h{worst['horizon']})"
    return bool((shift["mean_shift"].abs() <= RESIDUAL_SHIFT_MAX).all()), text


CHECKS: dict[str, Callable[[Mapping[str, pd.DataFrame], ProspectiveConfig], tuple[bool, str]]] = {
    "Z1": _z1,
    "Z2": _z2,
    "Z3": _z3,
    "Z4": _z4,
}


def evaluate_criteria(
    tables: Mapping[str, pd.DataFrame], config: ProspectiveConfig
) -> tuple[dict[str, object], ...]:
    """Z1-Z4 in order; the first failure stops the evaluation."""

    results, stopped = [], False
    for identifier, statement in CRITERIA:
        if stopped:
            results.append(
                {
                    "id": identifier,
                    "statement": statement,
                    "status": NOT_EVALUATED,
                    "measured": "stopped at an earlier failure",
                }
            )
            continue
        passed, measured = CHECKS[identifier](tables, config)
        results.append(
            {
                "id": identifier,
                "statement": statement,
                "status": PASS if passed else FAIL,
                "measured": measured,
            }
        )
        stopped = not passed
    return tuple(results)


# --- registration power (only after every criterion passed) ----------------------------


def _non_overlapping(frame: pd.DataFrame) -> pd.DataFrame:
    """Greedy from the first row: keep a row whose entry is at or after the last kept exit."""

    kept, last_end = [], None
    for position, (start, end) in enumerate(
        zip(frame[TARGET_START_COLUMN], frame[TARGET_END_COLUMN], strict=True)
    ):
        if last_end is None or start >= last_end:
            kept.append(position)
            last_end = end
    return frame.iloc[kept]


def registration_power(evidence_: Evidence, config: ProspectiveConfig) -> pd.DataFrame:
    """
    Per asset x horizon x target, from the B2 development: the loss
    differential d = loss(null) - loss(ridge) (QLIKE for volatility, squared
    error for return) on non-overlapping outcomes, its standardized mean, and
    the implied power at each registered checkpoint (first Holm step).
    """

    dev = evidence_.dev_b2
    keys = ["fold_id", "date", "symbol"]
    rows = []
    for (asset, horizon, target), part in dev.predictions.groupby(
        ["symbol", "horizon", "target"], sort=True
    ):
        model = part.loc[
            part["model"] == training.CANDIDATE, [*keys, "target_return", "prediction"]
        ]
        null = part.loc[part["model"] == training.NULL, [*keys, "prediction"]]
        joined = model.merge(null, on=keys, suffixes=("", "_null"), validate="one_to_one")
        spans = evidence_.b2.frames[(asset, int(horizon))].loc[
            :, ["date", TARGET_START_COLUMN, TARGET_END_COLUMN]
        ]
        joined = joined.merge(spans, on="date", how="left", validate="one_to_one")
        joined = _non_overlapping(joined.sort_values("date").reset_index(drop=True))
        y = joined["target_return"].to_numpy(np.float64)
        fitted, base = (joined[c].to_numpy(np.float64) for c in ("prediction", "prediction_null"))
        if target == VOL:
            d = training._qlike(y, base) - training._qlike(y, fitted)
        else:
            d = (y - base) ** 2 - (y - fitted) ** 2
        n = len(d)
        sd = float(np.std(d, ddof=1)) if n > 1 else math.nan
        effect = float(np.mean(d)) / sd if sd > 0 else math.nan
        row = {
            "asset": asset,
            "horizon": int(horizon),
            "target": target,
            "n_dev_outcomes": n,
            "mean_d": float(np.mean(d)) if n else math.nan,
            "sd_d": sd,
            "effect": effect,
        }
        for name, checkpoint in registration.CHECKPOINTS.items():
            outcomes = registration.confirmatory_outcomes(int(horizon), checkpoint)
            level = registration.first_holm_level(checkpoint)
            tested = outcomes >= registration.MIN_OUTCOMES and level is not None
            row[f"n_{name}"] = outcomes
            row[f"power_{name}"] = (
                registration.power(effect, outcomes, level)
                if tested and math.isfinite(effect)
                else None
            )
        level = registration.first_holm_level(registration.CHECKPOINTS["12_months"])
        row["outcomes_for_80pct_power"] = (
            registration.outcomes_for_power(effect, level) if math.isfinite(effect) else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --- the whole check -------------------------------------------------------------------


def run_check(
    prices: pd.DataFrame,
    distributions: pd.DataFrame,
    splits: pd.DataFrame,
    batches: pd.DataFrame,
    config: ProspectiveConfig,
) -> CheckReport:
    """Every table, the criteria in order, and the power estimates when all passed."""

    found = evidence(prices, distributions, splits, batches, config)
    by_year, totals = window_counts(found.b2)
    rescored, rescored_rows = rescored_current(found, config)
    metric = f"band_coverage_{round(config.band.level * 100)}"
    tables = {
        "windows_by_year": by_year,
        "window_totals": totals,
        "cent_grid": cent_grid(prices, config.assets),
        "coverage": pd.concat(
            [
                _metric_summary(found.dev_current, metric, zero_vol.CURRENT),
                _metric_summary(found.dev_b2, metric, zero_vol.B2),
                rescored,
            ],
            ignore_index=True,
        ),
        "zero_split": zero_split(found, rescored_rows),
        "band_width": pd.concat(
            [
                band_widths(found.dev_current, zero_vol.CURRENT),
                band_widths(found.dev_b2, zero_vol.B2),
            ],
            ignore_index=True,
        ),
        "qlike": pd.concat(
            [
                _metric_summary(found.dev_current, QLIKE_METRIC, zero_vol.CURRENT),
                _metric_summary(found.dev_b2, QLIKE_METRIC, zero_vol.B2),
            ],
            ignore_index=True,
        ),
        "residual_shift": residual_shift(found),
        "bit_identity": bit_identity(found, config, totals),
    }
    criteria = evaluate_criteria(tables, config)
    passed = all(c["status"] == PASS for c in criteria)
    return CheckReport(
        status=PASS if passed else FAIL,
        data_class=found.current.data_class,
        batch_ids=found.current.batch_ids,
        fingerprints=dict(found.current.fingerprints),
        tables=tables,
        criteria=criteria,
        registration_power=registration_power(found, config) if passed else None,
    )


# --- worked examples and rendering -----------------------------------------------------

EXAMPLE_CLOSES = (110.30, 50.00)
EXAMPLE_SUB_TICK_CASH = 0.003  # dollars: a distribution whose ex-date drop rounds to no tick
EXAMPLE_CASES = ("flat", "one_tick", "sub_tick_cash")


def _example_path(case: str, close: float, horizon: int):
    """Closes and cash for the as-of session, the entry session and the h window sessions."""

    closes = [close] * (horizon + 2)
    cash = [0.0] * (horizon + 2)
    if case == "one_tick":  # $0.01 up on the window's first session, flat after
        closes[2:] = [round(close + zero_vol.TICK, 2)] * horizon
    elif case == "sub_tick_cash":  # an ex-date on the window's first session, close unchanged
        cash[2] = EXAMPLE_SUB_TICK_CASH
    return closes, cash


def worked_examples(
    closes: Sequence[float] = EXAMPLE_CLOSES, horizons: Sequence[int] = (5, 20, 30)
) -> pd.DataFrame:
    """
    The floor and both rules' labels for a flat window, a one-tick window and
    a sub-tick-cash window, computed with the production arithmetic
    (series.daily_log_returns on F1-shaped tables, then volatility_labels).
    """

    from stock_agent.calendars import nyse
    from stock_agent.forecasting.prospective import series

    rows = []
    for close in closes:
        for horizon in horizons:
            sessions = nyse.sessions("2024-01-02", "2024-12-31")[: horizon + 2]
            for case in EXAMPLE_CASES:
                path, cash = _example_path(case, close, horizon)
                prices = pd.DataFrame(
                    {
                        "fund_id": "SHV",
                        "session_date": sessions,
                        "close": path,
                        "adjustment_basis": series.UNADJUSTED,
                    }
                )
                paid = pd.DataFrame(
                    {
                        "fund_id": "SHV",
                        "ex_date": [d for d, c in zip(sessions, cash, strict=True) if c],
                        "amount": [c for c in cash if c],
                        "amount_basis": series.PER_SHARE_AS_PAID,
                    }
                )
                splits = pd.DataFrame(
                    {"fund_id": [], "effective_date": [], "ratio_new_per_old": []}
                )
                returns = series.daily_log_returns(sessions, prices, paid, splits, "SHV")
                labels = {
                    rule: zero_vol.volatility_labels(
                        returns["date"],
                        returns["log_return"],
                        np.asarray(path, dtype=np.float64),
                        horizon=horizon,
                        entry_lag=1,
                        rule=rule,
                    ).iloc[0]
                    for rule in zero_vol.RULES
                }
                b2 = labels[zero_vol.B2]
                current = float(labels[zero_vol.CURRENT]["target_return"])
                rows.append(
                    {
                        "P_asof": close,
                        "horizon": horizon,
                        "case": case,
                        "sigma_floor_bp": float(b2["sigma_floor"]) * 1e4,
                        "log_sigma_floor": math.log(float(b2["sigma_floor"])),
                        "sigma_raw_bp": float(b2["sigma_raw"]) * 1e4,
                        "measured_zero": bool(b2["measured_zero"]),
                        "floored_b2": bool(b2["floored"]),
                        "y_current": current if math.isfinite(current) else None,
                        "y_b2": float(b2["target_return"]),
                    }
                )
    return pd.DataFrame(rows)


def _cell(value: object, digits: int) -> str:
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(frame: pd.DataFrame, digits: Mapping[str, int] | None = None) -> str:
    """A GitHub markdown table; floats with `digits` places (4 by default)."""

    digits = digits or {}
    header = "| " + " | ".join(frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join(_cell(row[c], digits.get(c, 4)) for c in frame.columns) + " |"
        for _, row in frame.iterrows()
    ]
    return "\n".join([header, rule, *body])


def render_examples() -> str:
    table = worked_examples()
    table["y_current"] = table["y_current"].astype(object)
    return markdown_table(table, {"P_asof": 2})


def render_registration() -> str:
    schedule = pd.DataFrame(registration.schedule_rows())
    coverage = pd.DataFrame(registration.coverage_rows())
    return "\n\n".join(
        [
            markdown_table(schedule, {"mde_12_months": 3, "mde_24_months": 3}),
            markdown_table(coverage, {"alpha": 5, "accepted_low": 2, "accepted_high": 2}),
        ]
    )


def jsonable(value: object) -> object:
    """Plain JSON values: NaN and infinities become None, numpy scalars Python numbers."""

    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.to_dict("records")]
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NaT or value is pd.NA or value is None:
        return None
    return value


def report_document(report: CheckReport, context: Mapping[str, object]) -> dict[str, object]:
    return jsonable(
        {
            "record_type": "zero_volatility_check",
            "status": report.status,
            "labels": list(report.labels),
            **context,
            "data_class": report.data_class,
            "data_note": training.ENGINEERING_NOTE[report.data_class],
            "batch_ids": list(report.batch_ids),
            "fingerprints": dict(report.fingerprints),
            "rules": {rule: zero_vol.DESCRIPTIONS[rule] for rule in zero_vol.RULES},
            "criteria": list(report.criteria),
            "tables": dict(report.tables),
            "registration_power": report.registration_power,
        }
    )


def render_report(report: CheckReport, context: Mapping[str, object]) -> str:
    lines = [
        "# Zero-volatility check (development data)",
        "",
        (
            f"Status: **{report.status.upper()}**. Data class: {report.data_class}. "
            f"{training.ENGINEERING_NOTE[report.data_class]}"
        ),
        "",
        *[f"- {label}" for label in report.labels],
        *[f"- {key}: {value}" for key, value in context.items()],
        "",
        "## Acceptance criteria (pre-stated, evaluated in order)",
        "",
        markdown_table(pd.DataFrame(report.criteria).loc[:, ["id", "status", "measured"]]),
    ]
    for name in (
        "window_totals",
        "cent_grid",
        "coverage",
        "zero_split",
        "band_width",
        "qlike",
        "residual_shift",
        "bit_identity",
        "windows_by_year",
    ):
        lines += ["", f"## {name}", "", markdown_table(report.tables[name])]
    if report.registration_power is not None:
        lines += [
            "",
            "## Registration power (development loss differentials under B2)",
            "",
            (
                "Planning estimates from out-of-fold development rows against an expanding-mean "
                "null; not evidence of prospective skill."
            ),
            "",
            markdown_table(report.registration_power),
        ]
    return "\n".join(lines) + "\n"


__all__ = [
    "CRITERIA",
    "CURRENT_AS_MISS",
    "EXAMPLE_CLOSES",
    "CheckRefused",
    "CheckReport",
    "Evidence",
    "evaluate_criteria",
    "evidence",
    "jsonable",
    "markdown_table",
    "refuse_after",
    "registration_power",
    "render_examples",
    "render_registration",
    "render_report",
    "report_document",
    "run_check",
    "worked_examples",
]
