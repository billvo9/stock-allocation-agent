"""
Pure Plotly figure builders over stored T2 tables (no Streamlit, no I/O).

Contract for every builder:
- plotted numbers are the stored values, passed through unchanged (dates
  are drawn as the stored session dates, formatted YYYY-MM-DD so a time
  zone can never shift a day);
- no standard error, interval or p-value is computed here: intervals are
  the stored ci_low/ci_high of the named method, and reference lines use
  stored values (null_value, benchmarks), never assumed ones;
- unavailable rows are drawn as gaps or labelled text with their stored
  reason, never as zeros; a stored 'warning' status is spelled out in text;
- models appear grouped by role in run order, never sorted by a metric;
- colour encodes identity only (model, symbol, method), never a verdict.

The only derived display is the mean of the last N defined daily rank ICs
(descriptive smoothing, no interval), labelled as such.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_agent.dashboard import style

ROLE_ORDER = ("null", "control", "canary", "canary_unsafe_reference", "candidate")
METHOD_NAMES = {
    "fold_block_t": "fold-block t",
    "newey_west": "Newey-West",
    "circular_block_bootstrap": "circular block bootstrap",
}
METHOD_STYLE = {  # neutral greys/blues only: a method's colour is never a verdict
    "fold_block_t": ("#000000", "circle"),
    "newey_west": ("#4D4D4D", "square-open"),
    "circular_block_bootstrap": ("#4477AA", "diamond-open"),
}
WARNING_TEXT_COLOUR = style.STATUS_COLOURS["warning"]


def _day(values: pd.Series) -> list[str]:
    stamps = pd.to_datetime(values, utc=True)
    return [None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d") for stamp in stamps]


def _value(value: float):
    return None if pd.isna(value) else float(value)


def ordered_models(models: pd.DataFrame, names: Sequence[str] | None = None) -> pd.DataFrame:
    """Models grouped by role (null, control, canary, ...) in run order."""

    frame = models if names is None else models[models["name"].isin(list(names))]
    rank = {role: index for index, role in enumerate(ROLE_ORDER)}
    return frame.assign(_order=frame["role"].map(rank).fillna(len(rank))).sort_values(
        ["_order"], kind="stable"
    )


def pooled_rows(metrics: pd.DataFrame, model: str, metric: str) -> pd.DataFrame:
    return metrics[
        (metrics["model"] == model) & (metrics["metric"] == metric) & metrics["fold_id"].isna()
    ]


def pooled_row(metrics: pd.DataFrame, model: str, metric: str, method: str) -> pd.Series | None:
    rows = pooled_rows(metrics, model, metric)
    rows = rows[rows["inference_method"] == method]
    return None if rows.empty else rows.iloc[0]


def _unavailable_note(row: pd.Series | None) -> str:
    if row is None:
        return "not stored"
    return f"unavailable: {row['reason']}" if row["status"] == "unavailable" else ""


def size_note(rates: Mapping[str, Mapping[str, float]] | None, method: str) -> str:
    """Stored simulated false-positive rates of one method, e.g. '6.0% / 7.6% / 9.2%'."""

    if not rates:
        return ""
    values = [f"{rates[s][method]:.1%}" for s in rates if method in rates[s]]
    return f"simulated size {' / '.join(values)} ({' / '.join(rates)})" if values else ""


def method_label(
    method: str, decision_method: str | None, rates: Mapping | None = None, ci_method=None
) -> str:
    """'fold-block t (decision method)' or 'Newey-West (corroborating; simulated size ...)'."""

    name = METHOD_NAMES.get(method, method)
    if isinstance(ci_method, str) and ci_method == "percentile_uncalibrated":
        return f"{name} (percentile interval, uncalibrated)"
    if method == decision_method:
        return f"{name} (decision method)"
    note = size_note(rates, method)
    return f"{name} (corroborating{'; ' + note if note else ''})"


def _error(row: pd.Series, column: str = "estimate") -> dict:
    return {
        "type": "data",
        "symmetric": False,
        "array": [_value(row["ci_high"] - row[column])],
        "arrayminus": [_value(row[column] - row["ci_low"])],
    }


# --- run overview / folds ---

TIMELINE_SEGMENTS = (
    ("training rows (expanding)", "train_first_date", "train_last_date", "#332288"),
    ("purge gap (rows removed)", "train_last_date", "test_start", "#BBBBBB"),
    ("test window [start, end)", "test_start", "test_end_exclusive", "#DDCC77"),
    ("test labels mature by", "test_last_date", "score_available_at", "#AA4499"),
)


def fold_timeline(folds: pd.DataFrame, holdout_start: pd.Timestamp, selected: int | None = None):
    """Per fold: expanding training rows, purge gap, test window, label maturation."""

    figure = go.Figure()
    labels = [f"fold {int(fold)}" for fold in folds["fold_id"]]
    for name, start_column, end_column, colour in TIMELINE_SEGMENTS:
        x, y, text = [], [], []
        for label, (_, fold) in zip(labels, folds.iterrows(), strict=True):
            start, end = _day(pd.Series([fold[start_column], fold[end_column]]))
            x += [start, end, None]
            y += [label, label, None]
            hover = (
                f"{label}<br>{name}<br>{start} to {end}"
                f"<br>train rows {fold['n_train']}, test rows {fold['n_test']}, "
                f"purged rows {fold['n_purged']}"
            )
            text += [hover, hover, None]
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line={"color": colour, "width": 9},
                hovertext=text,
                hoverinfo="text",
            )
        )
    if selected is not None:
        row = folds[folds["fold_id"] == selected]
        if len(row):
            figure.add_trace(
                go.Scatter(
                    x=_day(row["test_start"]),
                    y=[f"fold {selected}"],
                    mode="markers",
                    name="selected fold",
                    marker={"symbol": "triangle-right", "size": 14, "color": "#000000"},
                )
            )
    lockbox = holdout_start.strftime("%Y-%m-%d")
    end = (holdout_start + pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    figure.add_shape(
        type="rect",
        x0=lockbox,
        x1=end,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="#444444",
        opacity=0.15,
        line={"width": 0},
    )
    figure.add_vline(x=lockbox, line={"color": "#444444", "dash": "dash"})
    figure.add_annotation(
        x=lockbox,
        y=0,
        yref="paper",
        text="lockbox (no data shown)",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
    )
    layout = style.base_layout(
        "Fold construction (stored fold table)",
        x_title="session date",
        height=max(340, 22 * len(folds) + 150),
    )
    layout["yaxis"]["autorange"] = "reversed"
    figure.update_layout(**layout)
    return figure


# --- feature diagnostics ---


def feature_quantiles(feature_stats: pd.DataFrame, feature: str):
    """TRAINING-SCOPE: stored training quantiles of one feature, per fold."""

    rows = feature_stats[
        (feature_stats["scope"] == "training") & (feature_stats["feature"] == feature)
    ]
    wide = rows.pivot(index="fold_id", columns="statistic", values="value").sort_index()
    folds = wide.index.tolist()
    colour = "#332288"
    figure = go.Figure()
    for low, high, label, opacity in (
        ("q05", "q95", "q05 to q95", 0.15),
        ("q25", "q75", "q25 to q75", 0.3),
    ):
        figure.add_trace(
            go.Scatter(
                x=folds,
                y=wide[high].tolist(),
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=folds,
                y=wide[low].tolist(),
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=f"rgba(51,34,136,{opacity})",
                name=label,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=folds,
            y=wide["median"].tolist(),
            mode="lines+markers",
            name="median",
            line={"color": colour},
        )
    )
    for tail in ("q01", "q99"):
        figure.add_trace(
            go.Scatter(
                x=folds,
                y=wide[tail].tolist(),
                mode="lines",
                name=tail,
                line={"color": colour, "dash": "dot", "width": 1},
            )
        )
    figure.update_layout(
        **style.base_layout(
            f"TRAINING-SCOPE · {feature}: stored training quantiles by fold",
            x_title="fold (training window expands)",
            y_title=feature,
        )
    )
    return figure


FEATURE_SYMBOLS = ("circle", "square", "diamond", "triangle-up", "cross", "x", "star")


def feature_statistic_by_fold(
    feature_stats: pd.DataFrame, statistic: str, scope: str, *, title: str, y_title: str
):
    """One stored statistic per feature across folds; unavailable values are gaps."""

    rows = feature_stats[
        (feature_stats["scope"] == scope) & (feature_stats["statistic"] == statistic)
    ]
    colours = style.category_colours(rows["feature"].unique())
    figure = go.Figure()
    for index, (feature, group) in enumerate(rows.groupby("feature", sort=True)):
        group = group.sort_values("fold_id")
        values = [
            _value(v) if s == "ok" else None
            for v, s in zip(group["value"], group["status"], strict=True)
        ]
        figure.add_trace(
            go.Scatter(
                x=group["fold_id"].tolist(),
                y=values,
                mode="lines+markers",
                name=feature,
                connectgaps=False,
                line={"color": colours[feature]},
                marker={"symbol": FEATURE_SYMBOLS[index % len(FEATURE_SYMBOLS)], "size": 8},
            )
        )
    unavailable = rows[rows["status"] != "ok"]
    if len(unavailable):
        reasons = unavailable["reason"].fillna("unknown").value_counts()
        figure.add_annotation(
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font={"color": style.STATUS_COLOURS["unavailable"]},
            text="gaps: " + ", ".join(f"{count} {reason}" for reason, count in reasons.items()),
        )
    prefix = "TRAINING-SCOPE" if scope == "training" else "EVALUATION-ONLY"
    figure.update_layout(
        **style.base_layout(f"{prefix} · {title}", x_title="fold", y_title=y_title)
    )
    return figure


SIGNED_STATISTICS = ("standardized_mean_difference",)


def drift_heatmap(feature_stats: pd.DataFrame, statistic: str):
    """EVALUATION-ONLY: stored train-vs-test distance per fold and feature (neutral colours)."""

    rows = feature_stats[
        (feature_stats["scope"] == "evaluation") & (feature_stats["statistic"] == statistic)
    ]
    wide = rows.pivot(index="feature", columns="fold_id", values="value").sort_index()
    text = [[("" if pd.isna(v) else f"{v:.2f}") for v in row] for row in wide.to_numpy()]
    signed = statistic in SIGNED_STATISTICS
    figure = go.Figure(
        go.Heatmap(
            z=[[_value(v) for v in row] for row in wide.to_numpy()],
            x=[str(c) for c in wide.columns],
            y=wide.index.tolist(),
            colorscale="PuOr" if signed else style.NEUTRAL_SCALE,
            zmid=0 if signed else None,
            text=text,
            texttemplate="%{text}",
            colorbar={"title": statistic.replace("_", " ")},
        )
    )
    figure.update_layout(
        **style.base_layout(
            f"EVALUATION-ONLY · {statistic.replace('_', ' ')}: distances only, never a removal rule",
            x_title="fold",
            height=280 + 30 * len(wide),
        )
    )
    return figure


# --- nulls, controls, canary ---


def rank_ic_forest(
    metrics: pd.DataFrame,
    models: pd.DataFrame,
    *,
    methods: Sequence[str],
    decision_method: str | None,
    rates: Mapping | None = None,
    title: str = "Mean rank IC by model (stored estimates and nominal intervals)",
):
    """Mean rank IC per model and method, methods side by side within each model row."""

    figure = go.Figure()
    ordered = ordered_models(models)
    labels = [
        style.legend_name(name, role)
        for name, role in zip(ordered["name"], ordered["role"], strict=True)
    ]
    notes = []
    for method in methods:
        primary = method == decision_method
        colour, symbol = METHOD_STYLE.get(method, ("#4D4D4D", "circle-open"))
        xs, ys, plus, minus = [], [], [], []
        for label, (_, model) in zip(labels, ordered.iterrows(), strict=True):
            row = pooled_row(metrics, model["name"], "mean_rank_ic", method)
            if row is None or row["status"] != "ok":
                if primary or len(methods) == 1:
                    notes.append((label, _unavailable_note(row)))
                continue
            xs.append(float(row["estimate"]))
            ys.append(label)
            plus.append(float(row["ci_high"] - row["estimate"]))
            minus.append(float(row["estimate"] - row["ci_low"]))
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=method_label(method, decision_method, rates),
                marker={"symbol": symbol, "size": 12 if primary else 9, "color": colour},
                opacity=style.PRIMARY_OPACITY if primary else style.CORROBORATING_OPACITY,
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": plus,
                    "arrayminus": minus,
                    "thickness": 2 if primary else 1,
                },
                offsetgroup=method,
                orientation="h",
            )
        )
    for label, note in notes:
        figure.add_annotation(
            x=0,
            y=label,
            text=note,
            showarrow=False,
            xanchor="left",
            font={"color": style.STATUS_COLOURS["unavailable"]},
        )
    figure.add_vline(x=0, line={"color": "#888888", "dash": "dot"})
    layout = style.base_layout(title, x_title="mean rank IC", height=160 + 60 * len(labels))
    layout["yaxis"]["categoryorder"] = "array"
    layout["yaxis"]["categoryarray"] = labels[::-1]
    figure.update_layout(**layout, scattermode="group", scattergap=0.4)
    return figure


def null_draws_histogram(
    null_draws: pd.DataFrame, model: str, observed: float | None, observed_label: str
):
    """Stored block-permutation draws of mean rank IC, with the observed value marked."""

    draws = null_draws[
        (null_draws["model"] == model) & (null_draws["null_kind"] == "block_permutation")
    ]
    figure = go.Figure(
        go.Histogram(
            x=draws["value"].tolist(),
            nbinsx=20,
            name="permutation draws",
            marker={"color": "#BBBBBB"},
        )
    )
    if observed is not None:
        figure.add_vline(x=observed, line={"color": style.model_colour(model), "width": 3})
        figure.add_annotation(
            x=observed,
            y=1,
            yref="paper",
            text=f"{observed_label}: {observed:.3f}",
            showarrow=False,
            yanchor="bottom",
        )
    figure.update_layout(
        **style.base_layout(
            f"{model}: block-permutation null ({len(draws)} stored draws)",
            x_title="mean rank IC",
            y_title="draws",
        )
    )
    return figure


def static_ordering_strip(
    null_draws: pd.DataFrame, observed: Mapping[str, tuple[float, str]], observed_label: str
):
    """Mean rank IC of every fixed ranking (labels only), with models' observed ICs marked."""

    orderings = null_draws[null_draws["null_kind"] == "static_ordering"].sort_values("draw")
    figure = go.Figure(
        go.Scatter(
            x=orderings["value"].tolist(),
            y=["fixed rankings"] * len(orderings),
            mode="markers",
            marker={"symbol": "line-ns-open", "size": 18, "color": "#777777"},
            hovertext=orderings["detail"].tolist(),
            name="each fixed ranking",
        )
    )
    for name, (value, role) in observed.items():
        symbol, _ = style.role_marker(role)
        figure.add_trace(
            go.Scatter(
                x=[value],
                y=[name],
                mode="markers+text",
                text=[f"{value:.3f}"],
                textposition="middle right",
                name=style.legend_name(name, role),
                marker={"symbol": symbol, "size": 12, "color": style.model_colour(name)},
            )
        )
    figure.update_layout(
        **style.base_layout(
            f"Every fixed ranking of the names (exact enumeration) vs each model's {observed_label}",
            x_title="mean rank IC",
            height=240 + 30 * len(observed),
        )
    )
    return figure


def stale_comparison(metrics: pd.DataFrame, model: str, method: str, decision_method):
    """Stored IC(model) - IC(stale k) per stale lag, with the method's stored interval."""

    rows = metrics[
        (metrics["model"] == model)
        & metrics["metric"].str.startswith("rank_ic_minus_stale_")
        & metrics["fold_id"].isna()
        & (metrics["inference_method"] == method)
    ]
    rows = rows.assign(lag=rows["metric"].str.rsplit("_", n=1).str[-1].astype(int)).sort_values(
        "lag"
    )
    ok = rows["status"] == "ok"
    figure = go.Figure(
        go.Bar(
            x=[f"{lag} sessions" for lag in rows.loc[ok, "lag"]],
            y=rows.loc[ok, "estimate"].tolist(),
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (rows.loc[ok, "ci_high"] - rows.loc[ok, "estimate"]).tolist(),
                "arrayminus": (rows.loc[ok, "estimate"] - rows.loc[ok, "ci_low"]).tolist(),
            },
            marker={"color": style.model_colour(model)},
            name=f"IC({model}) - IC(stale)",
        )
    )
    for row in rows[~ok].itertuples(index=False):
        figure.add_annotation(
            x=f"{row.lag} sessions", y=0, text=f"unavailable: {row.reason}", showarrow=False
        )
    figure.add_hline(y=0, line={"color": "#888888", "dash": "dot"})
    figure.update_layout(
        **style.base_layout(
            f"{model}: timeliness control, {method_label(method, decision_method)}",
            x_title="feature staleness",
            y_title="IC difference",
        )
    )
    return figure


def canary_panel(
    metrics: pd.DataFrame, safe_model: str, unsafe_model: str, method: str, decision_method
):
    """Purged canary vs the deliberately unpurged reference (a positive control)."""

    figure = go.Figure()
    for name, label, pattern in (
        (safe_model, f"{safe_model}: purged, through the harness", ""),
        (unsafe_model, "UNSAFE reference: unpurged on purpose (not a strategy)", "/"),
    ):
        row = pooled_row(metrics, name, "mean_rank_ic", method)
        if row is None or row["status"] != "ok":
            figure.add_annotation(x=label, y=0, text=_unavailable_note(row), showarrow=False)
            continue
        figure.add_trace(
            go.Bar(
                x=[label],
                y=[float(row["estimate"])],
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": [float(row["ci_high"] - row["estimate"])],
                    "arrayminus": [float(row["estimate"] - row["ci_low"])],
                },
                marker={"color": style.model_colour(name), "pattern": {"shape": pattern}},
                name=label,
            )
        )
    figure.add_hline(y=0, line={"color": "#888888", "dash": "dot"})
    figure.update_layout(
        **style.base_layout(
            f"Leakage canary: mean rank IC ({method_label(method, decision_method)})",
            y_title="mean rank IC",
        )
    )
    return figure


# --- validation results ---


def rank_ic_series(curves: pd.DataFrame, model: str, folds: pd.DataFrame, *, window: int = 63):
    """Stored per-date rank IC (gaps where unavailable) and a mean of the last N defined dates."""

    rows = curves[(curves["model"] == model) & (curves["curve"] == "rank_ic")].sort_values("date")
    values = rows["value"].where(rows["status"] == "ok")
    dates = _day(rows["date"])
    defined = values.dropna()
    trailing = defined.rolling(window, min_periods=window).mean().reindex(values.index)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=[_value(v) for v in values],
            mode="markers",
            name="daily rank IC (stored)",
            marker={"size": 3, "color": "#999999"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=[_value(v) for v in trailing],
            mode="lines",
            connectgaps=True,
            name=f"mean of the last {window} defined dates (descriptive, no interval)",
            line={"color": style.model_colour(model), "width": 2},
        )
    )
    if values.notna().sum() == 0 and len(rows):
        reasons = rows["reason"].dropna()
        reason = reasons.mode().iloc[0] if len(reasons) else "not stored"
        figure.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            text=f"rank IC unavailable on all {len(rows)} dates (stored reason: {reason})",
        )
    for start in _day(folds["test_start"]):
        figure.add_vline(x=start, line={"color": "#DDDDDD", "width": 1})
    figure.add_hline(y=0, line={"color": "#888888", "dash": "dot"})
    figure.update_layout(
        **style.base_layout(
            f"{model}: rank IC by date (thin lines: fold starts)",
            x_title="session date",
            y_title="rank IC",
        )
    )
    return figure


def ic_by_fold(metrics: pd.DataFrame, model: str, decision_method: str | None):
    """Stored per-fold mean IC (no per-fold error bars) and the stored pooled interval."""

    rows = metrics[
        (metrics["model"] == model)
        & (metrics["metric"] == "mean_rank_ic")
        & metrics["fold_id"].notna()
    ].sort_values("fold_id")
    figure = go.Figure(
        go.Bar(
            x=rows["fold_id"].astype(int).tolist(),
            y=[
                _value(v) if s == "ok" else None
                for v, s in zip(rows["estimate"], rows["status"], strict=True)
            ],
            hovertext=[
                f"n dates {n}" + ("" if s == "ok" else f"; unavailable: {r}")
                for n, s, r in zip(rows["n"], rows["status"], rows["reason"], strict=True)
            ],
            marker={"color": style.model_colour(model)},
            name="fold mean (stored, descriptive)",
        )
    )
    pooled = (
        pooled_row(metrics, model, "mean_rank_ic", decision_method) if decision_method else None
    )
    if pooled is not None and pooled["status"] == "ok":
        figure.add_hrect(
            y0=float(pooled["ci_low"]),
            y1=float(pooled["ci_high"]),
            fillcolor="#000000",
            opacity=0.08,
            line_width=0,
        )
        figure.add_hline(
            y=float(pooled["estimate"]),
            line={"color": "#000000"},
            annotation_text=(
                f"{method_label(decision_method, decision_method)}: {pooled['estimate']:.3f}; "
                "band = interval for the mean of fold means, not a range for single folds"
            ),
            annotation_position="top left",
        )
    figure.update_layout(
        **style.base_layout(
            f"{model}: mean rank IC by fold", x_title="fold", y_title="mean rank IC"
        )
    )
    return figure


def hit_rate_panel(metrics: pd.DataFrame, models: pd.DataFrame, decision_method: str | None):
    """Stored hit rate with its independence benchmark and base rate per model."""

    figure = go.Figure()
    ordered = ordered_models(models)
    labels = [style.legend_name(m.name, m.role) for m in ordered.itertuples(index=False)]
    for model, label in zip(ordered.itertuples(index=False), labels, strict=True):
        row = (
            pooled_row(metrics, model.name, "hit_rate", decision_method)
            if decision_method
            else None
        )
        if row is None or row["status"] == "unavailable":
            figure.add_annotation(
                x=0.5,
                xref="paper",
                y=label,
                text=_unavailable_note(row),
                showarrow=False,
                font={"color": style.STATUS_COLOURS["unavailable"]},
            )
            continue
        warning = row["status"] == "warning"
        figure.add_trace(
            go.Scatter(
                x=[float(row["estimate"])],
                y=[label],
                mode="markers+text" if warning else "markers",
                text=[f"warning: {row['reason']}"] if warning else None,
                textposition="top center",
                textfont={"color": WARNING_TEXT_COLOUR},
                name=label,
                marker={
                    "symbol": style.role_marker(model.role)[0],
                    "size": 11,
                    "color": style.model_colour(model.name),
                },
                error_x=_error(row),
            )
        )
        for metric, symbol, text in (
            ("hit_independence_expected", "line-ns-open", "independence benchmark"),
            ("hit_base_rate", "triangle-up-open", "base rate"),
        ):
            reference = pooled_row(metrics, model.name, metric, "point")
            if reference is not None and reference["status"] == "ok":
                figure.add_trace(
                    go.Scatter(
                        x=[float(reference["estimate"])],
                        y=[label],
                        mode="markers",
                        marker={"symbol": symbol, "size": 16, "color": "#444444"},
                        name=text,
                        showlegend=False,
                        hovertext=text,
                    )
                )
    figure.update_layout(
        **style.base_layout(
            "Hit rate vs independence benchmark (|) and base rate (△)",
            x_title="hit rate",
            height=160 + 50 * len(ordered),
        )
    )
    figure.update_yaxes(categoryorder="array", categoryarray=labels[::-1])
    return figure


def oos_r2_panel(
    metrics: pd.DataFrame, models: pd.DataFrame, baseline: str, decision_method: str | None
):
    """Stored OOS R² vs one baseline (descriptive interval) beside the stored decision test."""

    decision = method_label(decision_method, decision_method) if decision_method else "unknown"
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=(
            f"OOS R² vs {baseline} (percentile interval, uncalibrated)",
            f"MSE improvement vs {baseline}, {decision}",
        ),
    )
    ordered = ordered_models(models)
    labels = [style.legend_name(m.name, m.role) for m in ordered.itertuples(index=False)]
    for model, label in zip(ordered.itertuples(index=False), labels, strict=True):
        for column, metric, method in (
            (1, f"oos_r2_vs_{baseline}", "circular_block_bootstrap"),
            (2, f"mse_improvement_vs_{baseline}", decision_method),
        ):
            row = pooled_row(metrics, model.name, metric, method) if method else None
            if row is None or row["status"] != "ok":
                note = _unavailable_note(row)
                if row is None and model.name == baseline:
                    note = "baseline itself"
                figure.add_annotation(
                    x=0,
                    y=label,
                    text=note or "not stored",
                    showarrow=False,
                    row=1,
                    col=column,
                    xanchor="left",
                    font={"color": style.STATUS_COLOURS["unavailable"]},
                )
                continue
            figure.add_trace(
                go.Scatter(
                    x=[float(row["estimate"])],
                    y=[label],
                    mode="markers",
                    showlegend=False,
                    marker={
                        "size": 11,
                        "color": style.model_colour(model.name),
                        "symbol": style.role_marker(model.role)[0],
                    },
                    error_x=_error(row),
                ),
                row=1,
                col=column,
            )
    figure.add_vline(x=0, line={"color": "#888888", "dash": "dot"})
    layout = style.base_layout(
        "Out-of-sample error against an explicit baseline", height=180 + 50 * len(models)
    )
    del layout["xaxis"]
    del layout["yaxis"]
    figure.update_layout(**layout)
    figure.update_yaxes(categoryorder="array", categoryarray=labels[::-1], automargin=True)
    return figure


def rank_position_bars(
    curves: pd.DataFrame, model: str, curve: str, *, note: str = "", ci_level: float = 0.95
):
    """Stored realized outcome by same-date rank position 1..N (one panel per N)."""

    rows = curves[(curves["model"] == model) & (curves["curve"] == curve)].sort_values(
        ["group", "x"]
    )
    figure = go.Figure()
    for group, block in rows.groupby("group", sort=True):
        figure.add_trace(
            go.Bar(
                x=[f"#{int(x)}" for x in block["x"]],
                y=block["value"].tolist(),
                name=f"{group}; bars: nominal {ci_level:.0%} Newey-West",
                error_y={
                    "type": "data",
                    "symmetric": False,
                    "array": (block["ci_high"] - block["value"]).tolist(),
                    "arrayminus": (block["value"] - block["ci_low"]).tolist(),
                },
                marker={"color": style.model_colour(model)},
            )
        )
    if note:
        figure.add_annotation(x=0.5, y=0.95, xref="paper", yref="paper", text=note, showarrow=False)
    figure.add_hline(y=0, line={"color": "#888888", "dash": "dot"})
    label = (
        "realized minus same-date mean" if curve.endswith("minus_date_mean") else "realized label"
    )
    figure.update_layout(
        **style.base_layout(
            f"{model}: {label} by same-date rank position",
            x_title="rank position (1 = highest prediction)",
            y_title=label,
        )
    )
    positions = sorted({int(x) for x in rows["x"]})
    figure.update_xaxes(
        type="category", categoryorder="array", categoryarray=[f"#{p}" for p in positions]
    )
    return figure


def rank_position_occupancy(curves: pd.DataFrame, model: str):
    """Stored share of dates each symbol occupies each rank position (ties split)."""

    rows = curves[(curves["model"] == model) & (curves["curve"] == "rank_position_occupancy")]
    colours = style.category_colours(rows["x_label"].dropna().unique())
    order = rows[["group", "x"]].drop_duplicates().sort_values(["group", "x"])
    categories = [f"#{int(x)} ({group})" for group, x in order.itertuples(index=False)]
    figure = go.Figure()
    for symbol, block in rows.groupby("x_label", sort=True):
        block = block.sort_values("x")
        figure.add_trace(
            go.Bar(
                x=[f"#{int(x)} ({g})" for x, g in zip(block["x"], block["group"], strict=True)],
                y=block["value"].tolist(),
                name=str(symbol),
                marker={"color": colours[symbol]},
                text=[f"{symbol} {v:.0%}" if v >= 0.08 else "" for v in block["value"]],
                textposition="inside",
            )
        )
    figure.update_layout(
        barmode="stack",
        **style.base_layout(
            f"{model}: which symbol sits at each rank position (share of dates)",
            x_title="rank position (1 = highest prediction)",
            y_title="share of dates",
        ),
    )
    figure.update_xaxes(type="category", categoryorder="array", categoryarray=categories)
    return figure


def pooled_deciles(curves: pd.DataFrame, model: str, output_kind: str, *, ci_level: float = 0.95):
    """POOLED prediction buckets across all dates (descriptive; not same-date portfolios)."""

    rows = curves[(curves["model"] == model) & (curves["curve"] == "pooled_decile_realized")]
    rows = rows.sort_values("x")
    warned = (rows["status"] == "warning").any()
    figure = go.Figure(
        go.Scatter(
            x=rows["x"].tolist(),
            y=rows["value"].tolist(),
            mode="markers+lines",
            marker={"size": 10, "color": style.model_colour(model)},
            error_y={
                "type": "data",
                "symmetric": False,
                "array": (rows["ci_high"] - rows["value"]).tolist(),
                "arrayminus": (rows["value"] - rows["ci_low"]).tolist(),
            },
            text=rows["x_label"].tolist(),
            name=(
                f"bucket mean over rows; bars: nominal {ci_level:.0%} Newey-West "
                "around per-date bucket means"
            ),
        )
    )
    if output_kind == "forecast" and not warned and len(rows):
        low = float(min(rows["x"].min(), rows["value"].min()))
        high = float(max(rows["x"].max(), rows["value"].max()))
        figure.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                line={"color": "#888888", "dash": "dash"},
                name="45° (calibrated forecast)",
            )
        )
    reason = rows.loc[rows["status"] == "warning", "reason"].dropna().unique()
    if len(reason):
        figure.add_annotation(
            x=0,
            y=1,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            text=f"warning: {reason[0]}",
            font={"color": WARNING_TEXT_COLOUR},
        )
    x_title = (
        "mean prediction in bucket"
        if output_kind == "forecast"
        else "mean score in bucket (score units)"
    )
    figure.update_layout(
        **style.base_layout(
            f"{model}: POOLED buckets across all dates (descriptive)"
            + (f" · warning: {reason[0]}" if len(reason) else ""),
            x_title=x_title,
            y_title="mean realized label",
        )
    )
    return figure


def residual_acf(curves: pd.DataFrame, model: str, series: str, horizon: int):
    """Stored residual autocorrelation; lags below the label horizon overlap by construction."""

    rows = curves[
        (curves["model"] == model)
        & (curves["curve"] == "residual_acf")
        & (curves["x_label"] == series)
    ].sort_values("x")
    colours = [
        "#BBBBBB" if group == "overlap_expected" else style.model_colour(model)
        for group in rows["group"]
    ]
    figure = go.Figure(
        go.Bar(
            x=rows["x"].tolist(),
            y=[_value(v) for v in rows["value"]],
            marker={"color": colours},
            name="ACF",
        )
    )
    figure.add_vrect(
        x0=0.5,
        x1=horizon - 0.5,
        fillcolor="#999999",
        opacity=0.1,
        line_width=0,
        annotation_text="overlapping labels (expected)",
        annotation_position="top left",
    )
    figure.update_layout(
        **style.base_layout(
            f"{model}: residual autocorrelation ({series})",
            x_title="lag (sessions)",
            y_title="autocorrelation",
        )
    )
    return figure


def goyal_welch(curves: pd.DataFrame, model: str, folds: pd.DataFrame):
    """Stored cumulative SSE(baseline) - SSE(model) by date, one line per baseline."""

    rows = curves[(curves["model"] == model) & (curves["curve"] == "cumulative_sse_improvement")]
    figure = go.Figure()
    for baseline, block in rows.groupby("group", sort=True):
        block = block.sort_values("date")
        figure.add_trace(
            go.Scatter(
                x=_day(block["date"]),
                y=block["value"].tolist(),
                mode="lines",
                name=f"vs {baseline}",
                line={"color": style.model_colour(str(baseline))},
            )
        )
    for start in _day(folds["test_start"]):
        figure.add_vline(x=start, line={"color": "#EEEEEE", "width": 1})
    figure.add_hline(y=0, line={"color": "#888888", "dash": "dot"})
    figure.update_layout(
        **style.base_layout(
            f"{model}: cumulative squared-error improvement (rising = model better)",
            x_title="session date",
            y_title="cumulative SSE difference",
        )
    )
    return figure


# --- calibration and uncertainty ---


def prediction_vs_realized(
    predictions: pd.DataFrame, model: str, mz_intercept: float | None, mz_slope: float | None
):
    """Stored predictions vs realized labels, a 45° line, and the stored MZ line (no band)."""

    rows = predictions[predictions["model"] == model]
    figure = go.Figure(
        go.Scattergl(
            x=rows["prediction"].tolist(),
            y=rows["target_return"].tolist(),
            mode="markers",
            marker={"size": 4, "opacity": 0.35, "color": style.model_colour(model)},
            name="scored rows",
        )
    )
    if len(rows):
        low = float(min(rows["prediction"].min(), rows["target_return"].min()))
        high = float(max(rows["prediction"].max(), rows["target_return"].max()))
        figure.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                line={"color": "#888888", "dash": "dash"},
                name="45° (calibrated)",
            )
        )
        if mz_intercept is not None and mz_slope is not None:
            x_low = float(rows["prediction"].min())
            x_high = float(rows["prediction"].max())
            figure.add_trace(
                go.Scatter(
                    x=[x_low, x_high],
                    y=[mz_intercept + mz_slope * x_low, mz_intercept + mz_slope * x_high],
                    mode="lines",
                    line={"color": "#000000"},
                    name=f"stored MZ line a={mz_intercept:.3f}, b={mz_slope:.2f}",
                )
            )
    figure.update_layout(
        **style.base_layout(
            f"{model}: prediction vs realized label (all scored rows)",
            x_title="prediction",
            y_title="realized label",
        )
    )
    return figure


def uncertainty_forest(
    metrics: pd.DataFrame,
    model: str,
    metric: str,
    decision_method: str | None,
    rates: Mapping | None = None,
):
    """One stored estimate and interval per method, side by side (no recomputation)."""

    rows = pooled_rows(metrics, model, metric)
    rows = rows[rows["inference_method"].isin(METHOD_NAMES)]
    present = list(rows["inference_method"])
    order = list(dict.fromkeys([m for m in (decision_method, *METHOD_NAMES) if m in present]))
    figure = go.Figure()
    labels = []
    null_value = None
    for method in order:
        row = rows[rows["inference_method"] == method].iloc[0]
        legend = method_label(method, decision_method, rates, row["ci_method"])
        label = METHOD_NAMES.get(method, method) + (
            " (decision method)" if method == decision_method else ""
        )
        labels.append(label)
        if null_value is None and pd.notna(row["null_value"]):
            null_value = float(row["null_value"])
        if row["status"] != "ok":
            figure.add_annotation(
                x=0.02,
                xref="paper",
                y=label,
                text=_unavailable_note(row) or row["status"],
                showarrow=False,
                xanchor="left",
                font={"color": style.STATUS_COLOURS["unavailable"]},
            )
            continue
        colour, symbol = METHOD_STYLE.get(method, ("#4D4D4D", "circle-open"))
        primary = method == decision_method
        figure.add_trace(
            go.Scatter(
                x=[float(row["estimate"])],
                y=[label],
                mode="markers",
                name=legend,
                marker={"symbol": symbol, "size": 13 if primary else 10, "color": colour},
                opacity=style.PRIMARY_OPACITY if primary else style.CORROBORATING_OPACITY,
                error_x=_error(row),
            )
        )
    if null_value is not None:
        figure.add_vline(
            x=null_value,
            line={"color": "#888888", "dash": "dot"},
            annotation_text=f"stored null value {null_value:g}",
        )
    layout = style.base_layout(
        f"{model}: {metric}, stored estimate and nominal interval per method",
        x_title=metric,
        height=320,
    )
    layout["yaxis"]["categoryorder"] = "array"
    layout["yaxis"]["categoryarray"] = labels[::-1]
    figure.update_layout(**layout)
    return figure


def false_positive_rates(rates: Mapping, threshold: float | None, decision_method: str | None):
    """Stored simulated false-positive rates by method and no-skill signal (iid SE excluded)."""

    figure = go.Figure()
    for method in METHOD_NAMES:
        signals = [signal for signal in rates if method in rates[signal]]
        colour, symbol = METHOD_STYLE[method]
        figure.add_trace(
            go.Scatter(
                x=[float(rates[signal][method]) for signal in signals],
                y=signals,
                mode="markers",
                name=method_label(method, decision_method),
                marker={
                    "symbol": symbol,
                    "size": 13 if method == decision_method else 10,
                    "color": colour,
                },
            )
        )
    figure.add_vline(x=0.05, line={"color": "#000000", "dash": "dot"})
    figure.add_annotation(
        x=0.05, y=1, yref="paper", text="nominal 5%", showarrow=False, yanchor="bottom"
    )
    if threshold is not None:
        figure.add_vline(x=threshold, line={"color": "#4D4D4D", "dash": "dash"})
        figure.add_annotation(
            x=threshold,
            y=0,
            yref="paper",
            text=f"stored threshold {threshold:.3f}",
            showarrow=False,
            yanchor="top",
        )
    figure.update_layout(
        **style.base_layout(
            "Simulated false-positive rate at nominal 5% (stored no-skill simulations)",
            x_title="share of no-skill series declared significant",
            height=340,
        )
    )
    figure.update_xaxes(rangemode="tozero")
    return figure
