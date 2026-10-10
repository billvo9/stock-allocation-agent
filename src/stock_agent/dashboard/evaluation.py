"""
Descriptive evaluation series derived from stored T2 rows.

Nothing here is inference. Dependence-aware intervals and tests come only
from T2's stored metrics; this module produces per-date descriptive series
and checks that, on the full sample, they reproduce T2's stored estimates.

Maturity. A stored prediction is mature when its label was realized by the
run's evaluation as-of time:

    mature  <=>  target_end_date <= evaluation_asof  and  a stored label

evaluation_asof is the latest session of the run's truncated development
frame (record.lockbox_evidence.max_date, written by T2 after truncating at
the lockbox and before labelling; not the raw data's last date): every label
T2 could compute ends on or before it, and a label is known after the close
of its target_end_date. Pending rows are reported, never dropped silently,
and never enter a series derived here. T2 scores every stored prediction, so
a pending stored prediction means T2's own results include it (the loader
blocks such a run).

Goyal-Welch, date-normalized. For a forecast model f and a comparator b
scored on identical observations (same fold, date, symbol and label):

    d_t = (1 / N_t) * sum_i [ (y_it - b_it)^2 - (y_it - f_it)^2 ]

the mean squared-error advantage over the N_t names scored on date t
(positive: the model's error is smaller). Its running sum is the primary
curve: every date weighs the same however many names it holds, so a
growing universe cannot steepen it. T2's stored cumulative
sum_t sum_i [...] weighs dates by their name count; it stays available as
the secondary, observation-weighted diagnostic. T2's Newey-West estimate of
mse_improvement is the mean of d_t over dates; its fold-block estimate is
the mean of per-fold means of d_t (equal only when folds hold equally many
dates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

KEYS = ["fold_id", "date", "symbol"]
EVALUATION_ASOF_SOURCE = "record.lockbox_evidence.max_date"
MATURE = "mature"
PENDING = "pending"
NULL_BASELINE = "null baseline"
SELECTION_CONTROL = "selection control (not a null baseline)"
# Tolerance for reproducing T2's stored values from stored rows: both sum the
# same float64 values in a different order, so the absolute error of a
# running sum grows with the magnitude of the losses summed (scaled below).
REPRODUCTION_RTOL = 1e-9
REPRODUCTION_ATOL = 1e-12


# --- maturity ---


def evaluation_asof(record: dict) -> pd.Timestamp | None:
    """The run's evaluation as-of session (UTC), or None when the run does not record it."""

    value = (record.get("lockbox_evidence") or {}).get("max_date")
    if value is None:
        return None
    return pd.Timestamp(pd.to_datetime(value, utc=True, format="ISO8601"))


def maturity(predictions: pd.DataFrame, asof: pd.Timestamp | None) -> pd.Series:
    """'mature' or 'pending' per stored prediction row (all pending without an as-of)."""

    if asof is None:
        return pd.Series(PENDING, index=predictions.index, name="maturity")
    end = pd.to_datetime(predictions["target_end_date"], utc=True)
    mature = end.le(asof).fillna(False) & predictions["target_return"].notna()
    return mature.map({True: MATURE, False: PENDING}).rename("maturity")


def pending_rows(predictions: pd.DataFrame, asof: pd.Timestamp | None) -> pd.DataFrame:
    return predictions[maturity(predictions, asof) == PENDING]


# --- comparators (Goyal-Welch baselines) ---


# How every registered comparator's forecasts are formed (a dashboard
# statement from T2's design, not a recorded field).
FORMED = (
    "per-fold forecast through the validation harness from purged training rows only "
    "(point-in-time); a constant needs no fitting"
)


@dataclass(frozen=True)
class Comparator:
    """One recorded comparator, with its identity carried alongside every series."""

    name: str
    role: str
    kind: str
    note: str  # as recorded in spec.models[].note

    @property
    def label(self) -> str:
        return f"{self.name}: {self.kind}"


def _classify(record: dict, models: pd.DataFrame) -> list[tuple[str, Comparator | None, str]]:
    registered = {model["name"]: model for model in record["spec"]["models"]}
    known = set(models["name"])
    out = []
    for name in record["spec"]["config"].get("baselines") or []:
        model = registered.get(name)
        if model is None or name not in known:
            out.append((name, None, "not a registered model of this run"))
            continue
        if model["output_kind"] != "forecast":
            out.append((name, None, f"output kind {model['output_kind']}: no forecast scale"))
            continue
        kind = {"null": NULL_BASELINE, "control": SELECTION_CONTROL}.get(model["role"])
        if kind is None:
            out.append((name, None, f"role {model['role']} is not a comparator role"))
            continue
        out.append((name, Comparator(name, model["role"], kind, model.get("note") or ""), ""))
    return out


def comparators(record: dict, models: pd.DataFrame) -> list[Comparator]:
    """
    The comparators T2 scored this run against (spec.config.baselines), in
    recorded order. As in T2, only registered forecast models qualify. No
    comparator is privileged: a raw-return label may call for the expanding
    training mean, an excess-return label for zero, and the reader chooses
    with the identity in view. The per-symbol mean is a selection control,
    never a null.
    """

    return [comparator for _, comparator, _ in _classify(record, models) if comparator]


def omitted_comparators(record: dict, models: pd.DataFrame) -> list[tuple[str, str]]:
    """Recorded baselines that cannot be comparators here, with the reason."""

    return [
        (name, reason) for name, comparator, reason in _classify(record, models) if not comparator
    ]


# --- squared-error advantage ---


def aligned_observations(
    predictions: pd.DataFrame, model: str, baseline: str, asof: pd.Timestamp | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    (mature, pending) rows of the model joined to the comparator on identical
    observations. Raises unless both were scored on exactly the same keys with
    the same labels: a comparison over different rows is not a comparison.
    """

    if model == baseline:
        raise ValueError("A model is not compared with itself.")
    columns = [*KEYS, "prediction", "target_return", "target_end_date"]
    left = predictions.loc[predictions["model"] == model, columns]
    right = predictions.loc[predictions["model"] == baseline, columns]
    if left.empty or right.empty:
        raise ValueError(f"No stored predictions for {model if left.empty else baseline}.")
    for name, rows in ((model, left), (baseline, right)):
        if rows.duplicated(KEYS).any():
            raise ValueError(
                f"{name} has duplicate stored predictions for one fold, date and symbol."
            )
    joined = left.merge(right, on=KEYS, how="outer", suffixes=("", "_baseline"), indicator=True)
    if (joined["_merge"] != "both").any():
        raise ValueError(f"{model} and {baseline} were not scored on identical observations.")
    same_label = (joined["target_return"] == joined["target_return_baseline"]) | (
        joined["target_return"].isna() & joined["target_return_baseline"].isna()
    )
    same_end = (joined["target_end_date"] == joined["target_end_date_baseline"]) | (
        joined["target_end_date"].isna() & joined["target_end_date_baseline"].isna()
    )
    if not (same_label.all() and same_end.all()):
        raise ValueError(f"{model} and {baseline} carry different labels for the same rows.")
    joined = joined.drop(
        columns=["_merge", "target_return_baseline", "target_end_date_baseline"]
    ).rename(columns={"prediction_baseline": "baseline_prediction"})
    mature = maturity(joined, asof) == MATURE
    return joined[mature].reset_index(drop=True), joined[~mature].reset_index(drop=True)


def squared_error_advantage(
    predictions: pd.DataFrame, model: str, baseline: str, asof: pd.Timestamp | None
) -> pd.DataFrame:
    """
    Per-date squared-error advantage of `model` over `baseline` on mature,
    identical observations: date, fold_id, n_obs (N_t), d (date-normalized),
    sse_difference (observation-weighted), and their running sums. Every row
    names its model and baseline.
    """

    mature, _ = aligned_observations(predictions, model, baseline, asof)
    label = mature["target_return"].to_numpy(dtype=np.float64)
    loss_baseline = (label - mature["baseline_prediction"].to_numpy(dtype=np.float64)) ** 2
    loss_model = (label - mature["prediction"].to_numpy(dtype=np.float64)) ** 2
    frame = mature[["date", "fold_id"]].assign(
        difference=loss_baseline - loss_model, loss_total=loss_baseline + loss_model
    )
    grouped = frame.groupby("date", sort=True)
    folds = grouped["fold_id"].agg(["min", "max"])
    if (folds["min"] != folds["max"]).any():
        raise ValueError("A scored date belongs to more than one fold.")
    series = pd.DataFrame(
        {
            "fold_id": folds["min"].astype("int64"),
            "n_obs": grouped.size().astype("int64"),
            "sse_difference": grouped["difference"].sum(),
            "loss_total": grouped["loss_total"].sum(),  # scale for reproduction tolerance
        }
    )
    series["d"] = series["sse_difference"] / series["n_obs"]
    series["cumulative_d"] = series["d"].cumsum()
    series["cumulative_sse"] = series["sse_difference"].cumsum()
    return series.reset_index().assign(model=model, baseline=baseline)


# --- reproducing T2's stored estimates (full sample only) ---


@dataclass(frozen=True)
class Reproduction:
    """Whether the derived full-sample series reproduces T2's stored values."""

    status: str  # "reproduced" | "differs" | "not_stored"
    detail: str

    @property
    def ok(self) -> bool:
        return self.status == "reproduced"


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=REPRODUCTION_RTOL, abs_tol=REPRODUCTION_ATOL)


def reproduce_t2(
    advantage: pd.DataFrame, metrics: pd.DataFrame, curves: pd.DataFrame, model: str, baseline: str
) -> Reproduction:
    """
    Compare the derived full-sample series with T2's stored values for the
    same model and comparator: the Newey-West estimate (mean of d over
    dates), the fold-block estimate (mean of fold means of d), and the stored
    observation-weighted cumulative curve. Any difference means the derived
    series is not what T2 measured, so it must not be shown as T2's.
    """

    stored = metrics[
        (metrics["model"] == model)
        & (metrics["metric"] == f"mse_improvement_vs_{baseline}")
        & metrics["fold_id"].isna()
    ]
    estimates = dict(zip(stored["inference_method"], stored["estimate"], strict=True))
    cumulative = curves[
        (curves["model"] == model)
        & (curves["curve"] == "cumulative_sse_improvement")
        & (curves["group"] == baseline)
    ].sort_values("date")
    if "newey_west" not in estimates or "fold_block_t" not in estimates or cumulative.empty:
        return Reproduction(
            "not_stored", f"T2 stored no squared-error comparison of {model} with {baseline}."
        )
    derived = {
        "newey_west": float(advantage["d"].mean()),
        "fold_block_t": float(advantage.groupby("fold_id")["d"].mean().mean()),
    }
    # A stored estimate T2 could not form (NaN) is not a value to reproduce.
    compared = [method for method in derived if math.isfinite(float(estimates[method]))]
    if not compared:
        return Reproduction(
            "not_stored", f"T2 stored no estimate comparing {model} with {baseline}."
        )
    problems = [
        f"{method}: derived {derived[method]:.6g} vs stored {estimates[method]:.6g}"
        for method in compared
        if not _close(derived[method], float(estimates[method]))
    ]
    stored_curve = cumulative.set_index("date")["value"]
    derived = advantage.set_index("date")
    # Summation-order error of a running sum grows with the losses summed so far.
    tolerance = REPRODUCTION_ATOL * (1.0 + derived["loss_total"].cumsum().to_numpy())
    gap = (
        np.abs(stored_curve.to_numpy() - derived["cumulative_sse"].to_numpy())
        if (stored_curve.index.equals(derived.index))
        else None
    )
    if gap is None or not np.all(
        gap <= tolerance + REPRODUCTION_RTOL * np.abs(stored_curve.to_numpy())
    ):
        problems.append("observation-weighted cumulative curve differs from the stored curve")
    if problems:
        return Reproduction("differs", "; ".join(problems))
    return Reproduction(
        "reproduced",
        "The derived series reproduces T2's stored mean, fold-mean and cumulative curve.",
    )


# --- interactive sample filters (descriptive only) ---


@dataclass(frozen=True)
class SampleFilter:
    """A reader-chosen subset of evaluation dates. Never used for inference."""

    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    folds: tuple[int, ...] | None = None
    regime: str | None = None
    regime_value: str | None = None

    @property
    def is_full_sample(self) -> bool:
        return (
            self.start is None
            and self.end is None
            and self.folds is None
            and (self.regime is None or self.regime_value is None)
        )

    def mask(self, frame: pd.DataFrame, regimes: pd.DataFrame | None = None) -> pd.Series:
        """Rows of a per-date frame (columns date, fold_id) inside the filter."""

        keep = pd.Series(True, index=frame.index)
        if self.start is not None:
            keep &= frame["date"] >= self.start
        if self.end is not None:
            keep &= frame["date"] <= self.end
        if self.folds is not None:
            keep &= frame["fold_id"].isin(self.folds)
        if self.regime is not None and self.regime_value is not None:
            if regimes is None or self.regime not in regimes.columns:
                return pd.Series(False, index=frame.index)
            labels = frame["date"].map(regimes.set_index("date")[self.regime])
            keep &= labels.eq(self.regime_value).fillna(False)
        return keep


def descriptive_summary(advantage: pd.DataFrame) -> dict[str, float | int]:
    """Counts and plain averages of a (possibly filtered) advantage series. No inference."""

    d = advantage["d"]
    return {
        "dates": len(advantage),
        "observations": int(advantage["n_obs"].sum()),
        "mean_d": float(d.mean()) if len(d) else math.nan,
        "share_dates_model_better": float((d > 0).mean()) if len(d) else math.nan,
        "sum_d": float(d.sum()),
    }


def rank_ic_dates(
    curves: pd.DataFrame, predictions: pd.DataFrame, model: str, asof: pd.Timestamp | None
) -> pd.DataFrame:
    """
    T2's stored per-date rank IC of `model`, with each date's maturity: a
    date is mature only when every stored row of the model on it is mature.
    """

    rows = curves[(curves["model"] == model) & (curves["curve"] == "rank_ic")]
    rows = rows[["date", "fold_id", "value", "n", "status", "reason"]].sort_values("date")
    own = predictions[predictions["model"] == model]
    state = (
        own.assign(maturity=maturity(own, asof))
        .groupby("date")["maturity"]
        .agg(lambda values: MATURE if (values == MATURE).all() else PENDING)
    )
    rows = rows.assign(maturity=rows["date"].map(state).fillna(PENDING))
    return rows.reset_index(drop=True)
