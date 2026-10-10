"""
Streamlit page bodies. Everything shown comes from a loaded RunView; this
module lays out stored values, stored statuses and stored reasons, and
explains them. It computes no statistics.

The unsafe canary reference (a deliberately leaky positive control) is
drawn only in the canary panel of the Nulls & controls page; everywhere
else it is filtered out by role.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from stock_agent.dashboard import figures, glossary, style
from stock_agent.dashboard.loader import (
    RunRefused,
    RunView,
    default_root,
    ledger_summary,
    list_runs,
    load_run,
)
from stock_agent.dashboard.status import (
    STATUS_RULES,
    check_outcomes,
    research_notices,
    run_status,
)
from stock_agent.model_diagnostics import contract

RUN_KEY = "selected_run"
MODELS_KEY = "selected_models"
UNSAFE_ROLE = contract.UNSAFE_REFERENCE_ROLE
METRIC_COLUMNS = [
    "model",
    "role",
    "metric",
    "inference_method",
    "estimate",
    "se",
    "ci_low",
    "ci_high",
    "ci_level",
    "ci_method",
    "p_value",
    "null_value",
    "n",
    "n_eff",
    "df",
    "hac_lag",
    "block_length",
    "bootstrap_reps",
    "seed",
    "status",
    "reason",
]


# --- loading and selection ---


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_run(path: str, signature: str) -> RunView:
    del signature  # part of the cache key only: any changed file forces re-verification
    return load_run(path)


def _signature(path: Path) -> str:
    """Manifest bytes plus every file's size and modification time in the run directory."""

    digest = hashlib.sha256()
    manifest = path / "manifest.json"
    if manifest.is_file():
        digest.update(manifest.read_bytes())
    for child in sorted(path.iterdir()):
        status = child.lstat()
        digest.update(f"{child.name}:{status.st_size}:{status.st_mtime_ns};".encode())
    return digest.hexdigest()


def sidebar() -> RunView | None:
    """Run picker. The root comes from the launcher's environment, never from a widget."""

    root = default_root()
    st.sidebar.markdown("### Run")
    entries = list_runs(root)
    if not entries:
        st.sidebar.info(
            f"No runs under {root / 'runs'}. Create one with scripts/run_null_diagnostics.py."
        )
        return None
    labels = {entry.run_id: entry for entry in entries}
    run_id = st.sidebar.selectbox(
        "Saved run (newest first)",
        list(labels),
        key=RUN_KEY,
        help="Only verified development runs are displayed; refused runs show their reason.",
    )
    entry = labels[run_id]
    try:
        view = _cached_run(str(entry.path), _signature(entry.path))
    except RunRefused as refused:
        st.sidebar.error(f"Refused: {refused.reason}")
        st.error(f"This run is not displayed ({refused.reason}). {refused.detail}")
        return None
    status = _status(view)
    st.sidebar.markdown(_status_badge(status.level), unsafe_allow_html=True)
    st.sidebar.caption(
        f"Mode: {view.record['spec']['lockbox']['mode']} · lockbox from {view.holdout_start.date()}"
    )
    selectable = _models(view)
    default = selectable.loc[selectable["role"].isin(["null", "control"]), "name"].tolist()
    st.sidebar.multiselect(
        "Models on comparison charts",
        selectable["name"].tolist(),
        default=default,
        key=MODELS_KEY,
        help="Grouped by role, in run order. The unsafe canary reference appears only in "
        "the canary panel of the Nulls & controls page.",
    )
    return view


def _models(view: RunView) -> pd.DataFrame:
    """Every registered model except the unsafe canary reference."""

    return view.models[view.models["role"] != UNSAFE_ROLE]


def _selected_models(view: RunView) -> pd.DataFrame:
    frame = _models(view)
    names = st.session_state.get(MODELS_KEY)
    if names is None:
        return frame
    return frame[frame["name"].isin(names)]


def _status(view: RunView):
    return run_status(
        view.tables["checks"],
        view.record.get("checks_summary"),
        record_issues=view.record_issues,
    )


def _method(view: RunView) -> str | None:
    """The run's decision method; None (no headline rows) when the run cannot say."""

    return view.decision_method


def _method_notice(view: RunView) -> None:
    if view.decision_method is None:
        st.error(
            f"Decision method {view.decision_method_source}: headline rows are not selected "
            "for this run."
        )


def _rates(view: RunView) -> dict:
    return (view.record.get("inference_calibration") or {}).get("rates") or {}


def _status_badge(level: str) -> str:
    colour = style.STATUS_COLOURS[level]
    return (
        f"<div style='padding:6px 10px;border:2px solid {colour};border-radius:6px;"
        f"color:{colour};background:#FFFFFF;font-weight:700'>{level}</div>"
    )


def _chart(figure, key: str) -> None:
    st.plotly_chart(figure, theme=None, key=key, config=style.PLOTLY_CONFIG)


def _download(frame: pd.DataFrame, name: str, view: RunView) -> None:
    st.download_button(
        f"Download {name} (CSV)",
        frame.to_csv(index=False).encode("utf-8"),
        file_name=f"{view.run_id}_{name}.csv",
        mime="text/csv",
        key=f"download_{name}",
    )


def _metric_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, METRIC_COLUMNS]


def _metric_card(
    column, label: str, row: pd.Series | None, *, fmt: str = "{:.3f}", term: str | None = None
):
    help_text = glossary.define(term) if term else None
    if row is None:
        column.metric(label, "—", help=help_text)
        column.caption("not stored for this model")
        return
    if row["status"] == "unavailable":
        column.metric(label, "—", help=help_text)
        column.caption(f"unavailable: {row['reason']}")
        return
    column.metric(label, fmt.format(row["estimate"]), help=help_text)
    details = []
    if not pd.isna(row["ci_low"]):
        details.append(
            f"{row['ci_level']:.0%} interval [{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
        )
    if not pd.isna(row["p_value"]):
        details.append(f"p = {row['p_value']:.3f} vs {row['null_value']:g}")
    if row["status"] == "warning":
        details.append(f"warning: {row['reason']}")
    if details:
        column.caption("; ".join(details))


def _day(value) -> str:
    return "—" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


# --- pages ---


def overview(view: RunView) -> None:
    st.title("Run overview")
    spec = view.record["spec"]
    checks = view.tables["checks"]
    status = _status(view)
    message = f"**{status.level}**" + (": " + "; ".join(status.reasons) if status.reasons else "")
    {"VALID": st.success, "WARNING": st.warning}.get(status.level, st.error)(message)
    if status.level == "VALID":
        st.caption("No T2 check failures. This certifies the pipeline, not the results.")
    with st.expander("How the run status is derived (from stored T2 checks only)"):
        st.table(pd.DataFrame(STATUS_RULES, columns=["status", "rule"]))
        st.caption(
            "Info rows, metric rows with status 'warning', and unavailable metrics never change "
            "the status. A research warning's stored passed=True means the warning is active."
        )
    links = st.columns(3)
    links[0].page_link("page_scripts/research_warnings.py", label="Research warnings")
    links[1].page_link("page_scripts/folds.py", label="Fold construction")
    links[2].page_link("page_scripts/glossary.py", label="Glossary")
    predictions = view.tables["predictions"]
    scored = predictions[predictions["role"] != UNSAFE_ROLE]
    st.markdown(
        f"Mode **{spec['lockbox']['mode']}** · decision method "
        f"**{view.decision_method or 'unavailable'}** ({view.decision_method_source}) · "
        f"label `{spec['label']['label_id']}` · output schema **{view.schema_version}**"
    )
    if view.legacy_gaps:
        st.info(
            f"Output schema {view.schema_version}: fields this run did not record are shown as "
            "not recorded, or as derived from stored rows where labelled. "
            + "; ".join(view.legacy_gaps)
            + "."
        )
    cards = st.columns(5)
    cards[0].metric("Folds", len(view.folds))
    cards[1].metric("Scored dates", scored["date"].nunique())
    cards[2].metric("Scored rows per model", int(scored.groupby("model").size().max()))
    cards[3].metric("Symbols", len(spec["universe"]["symbols"]))
    cards[4].metric("Active research warnings", status.counts["research_warnings"])

    left, right = st.columns(2)
    with left:
        st.subheader("Identity")
        code = spec["code"]
        ledger = ledger_summary(default_root(), view.record.get("spec_sha256"))
        ledger_text = (
            f"{ledger['attempts']} attempts ({ledger['failed']} failed), {ledger['same_spec']} "
            f"with this spec, {ledger['distinct_variants']} distinct model variants"
            if ledger["status"] == "ok"
            else f"ledger unreadable: {ledger['reason']}"
        )
        st.markdown(
            f"- **Run id** `{view.run_id}`\n"
            f"- **Spec hash** `{str(view.record.get('spec_sha256'))[:16]}…`\n"
            f"- **Git** `{code.get('git_sha')}` · dirty tree: **{code.get('git_dirty')}**"
            + (
                f" (diff `{str(code.get('git_diff_sha256'))[:12]}…`)"
                if code.get("git_dirty")
                else ""
            )
            + f"\n- **Started** {view.record.get('started_at')} · **finished** "
            f"{view.record.get('finished_at', 'not recorded')}\n"
            f"- **Ledger** {ledger_text}"
        )
        st.subheader("Label and timing")
        label = spec["label"]
        st.markdown(
            f"- **Label** `{label['label_id']}` (column `{label['target_column']}`)\n"
            f"- **Horizon / entry lag** {label['label_spec']['horizon']} / "
            f"{label['label_spec']['entry_lag']} sessions"
        )
        with st.expander("Decision and signal timing (stored convention)"):
            st.write(label["signal_timing_convention"])
    with right:
        st.subheader("Universe and lockbox")
        universe = spec["universe"]
        evidence = view.record["lockbox_evidence"]
        st.markdown(
            f"- **Symbols** {', '.join(universe['symbols'])}\n"
            f"- **Excluded** {universe.get('excluded_symbols') or 'none'}\n"
            f"- **Lockbox** from {view.holdout_start.date()} · holdout values used: "
            f"**{evidence.get('holdout_values_used')}** · holdout rows loaded by the feature "
            f"build: {evidence.get('holdout_rows_loaded')}"
        )
        if universe.get("note"):
            st.info(f"Recorded universe note: {universe['note']}")
        st.subheader("Inference settings")
        config = spec["config"]
        st.markdown(
            f"- Newey-West lag **{config['hac_lag']}** (Bartlett)\n"
            f"- Block bootstrap: block **{config['block_length']}** sessions, "
            f"**{config['bootstrap_reps']}** replicates, seed {config['seed']}\n"
            f"- Permutation draws {config['permutation_draws']} · stale lags "
            f"{config['stale_lags']} · calibration replicates {config['calibration_reps']}"
        )
    st.subheader("Models registered in the run")
    st.dataframe(_models(view), hide_index=True)
    st.caption(
        "The canary's unpurged positive-control reference is not a registered model; it is "
        "shown only in the canary panel of the Nulls & controls page."
    )
    st.subheader("Stored T2 checks")
    families = pd.DataFrame.from_dict(status.families, orient="index").rename_axis("family")
    st.dataframe(families.reset_index(), hide_index=True)
    st.caption(
        f"Counts per check family from the stored rows (summary: {status.summary_source}; "
        f"matches the rows: {status.summary_matches}). passed + failed + undetermined = total; "
        "undetermined means no stored result. For research warnings, passed means the "
        "warning is active."
    )
    st.dataframe(checks.assign(outcome=check_outcomes(checks)), hide_index=True)
    with st.expander("Artifact hashes (manifest and outputs)"):
        st.json(view.record["outputs"], expanded=False)
        st.json(spec["environment"], expanded=False)
        st.json(spec["inputs"], expanded=False)


FOLD_FIELDS = {
    "train_first_date": "first training row",
    "train_last_date": "last training row",
    "train_max_target_end_date": "latest training label end",
    "knowledge_cutoff": "knowledge cutoff",
    "test_start": "test window start",
    "test_end_exclusive": "test window end (exclusive)",
    "test_last_date": "last test row",
    "score_available_at": "test labels known by",
    "holdout_start": "lockbox start",
}
FOLD_COUNTS = {
    "n_train": "training rows",
    "n_test": "test rows",
    "n_purged": "purged rows",
    "n_unlabeled": "unlabeled rows",
}


def folds(view: RunView) -> None:
    st.title("Fold construction")
    st.caption(
        "Read from the stored fold table; folds are never rebuilt here. The purge gap is drawn "
        "between the last training row and the test start; its exact dates are not stored, only "
        "the number of purged rows."
    )
    fold_ids = view.folds["fold_id"].astype(int).tolist()
    selected = st.selectbox("Fold", fold_ids, index=len(fold_ids) - 1, key="fold")
    _chart(figures.fold_timeline(view.folds, view.holdout_start, selected), "fold_timeline")
    row = view.folds[view.folds["fold_id"] == selected].iloc[0]
    left, right = st.columns(2)
    with left:
        st.subheader(f"Fold {selected}")
        st.dataframe(
            pd.DataFrame(
                {
                    "field": list(FOLD_FIELDS.values()),
                    "date": [_day(row[column]) for column in FOLD_FIELDS],
                }
            ),
            hide_index=True,
        )
        counts = st.columns(4)
        for column, (name, label) in zip(counts, FOLD_COUNTS.items(), strict=True):
            column.metric(label, int(row[name]))
        st.caption(f"Purge rule: {row['purge_rule']} · embargo sessions: {row['embargo_sessions']}")
    with right:
        st.subheader("Rows by symbol")
        symbols = sorted(set(row["train_rows_by_symbol"]) | set(row["test_rows_by_symbol"]))
        st.dataframe(
            pd.DataFrame(
                {
                    "symbol": symbols,
                    "training rows": [row["train_rows_by_symbol"].get(s, 0) for s in symbols],
                    "test rows": [row["test_rows_by_symbol"].get(s, 0) for s in symbols],
                }
            ),
            hide_index=True,
        )
        with st.expander("Audit fingerprints"):
            for name in (
                "frame_rows_sha256",
                "train_keys_sha256",
                "test_keys_sha256",
                "purged_keys_sha256",
                "train_content_sha256",
                "test_content_sha256",
            ):
                st.code(f"{name}: {row[name]}")
    _download(view.tables["folds"], "folds", view)


def features(view: RunView) -> None:
    st.title("Data and feature diagnostics")
    stats = view.tables["feature_stats"]
    training = stats[stats["scope"] == "training"]
    feature_names = sorted(f for f in training["feature"].unique() if f != "__design__")
    st.header("TRAINING-SCOPE DIAGNOSTIC")
    st.caption("Computed by T2 on each fold's training rows only: what a fold's fit sees.")
    feature = st.selectbox("Feature", feature_names, key="feature")
    _chart(figures.feature_quantiles(stats, feature), "feature_quantiles")
    with st.expander(f"Stored training statistics for {feature} (table)"):
        st.dataframe(
            training[training["feature"] == feature]
            .pivot(index="fold_id", columns="statistic", values="value")
            .reset_index(),
            hide_index=True,
        )
    columns = st.columns(2)
    with columns[0]:
        _chart(
            figures.feature_statistic_by_fold(
                stats, "missing_rate", "training", title="missing rate", y_title="share missing"
            ),
            "missing_rate",
        )
        _chart(
            figures.feature_statistic_by_fold(
                stats,
                "date_variance_share",
                "training",
                title="date-variance share",
                y_title="share of variance",
            ),
            "date_variance",
        )
    with columns[1]:
        _chart(
            figures.feature_statistic_by_fold(
                stats,
                "n_extreme_robust_z",
                "training",
                title="extreme values (|robust z| > 5)",
                y_title="rows",
            ),
            "extremes",
        )
        _chart(
            figures.feature_statistic_by_fold(stats, "vif", "training", title="VIF", y_title="VIF"),
            "vif",
        )
    st.caption(glossary.define("VIF") + " " + glossary.define("Date-variance share"))
    design = training[training["feature"] == "__design__"]
    st.subheader("Design matrix (per fold)")
    st.dataframe(
        design.pivot(index="fold_id", columns="statistic", values="value").reset_index(),
        hide_index=True,
    )
    st.caption(glossary.define("Condition number"))
    near_constant = training[(training["statistic"] == "near_constant") & (training["value"] > 0)]
    st.markdown(f"**Near-constant flags:** {len(near_constant)} feature-fold pairs.")
    fold_ids = sorted(stats["fold_id"].unique())
    fold_choice = st.selectbox(
        "Fold for correlations", fold_ids, index=len(fold_ids) - 1, key="corr_fold"
    )
    pairs = view.tables["feature_pairs"]
    st.dataframe(pairs[pairs["fold_id"] == fold_choice], hide_index=True)

    st.header("EVALUATION-ONLY DIAGNOSTIC")
    st.warning(
        "Test-window features compared with training features. Distances only: no p-values, no "
        "T2 judgment, and never a reason to remove or change a feature. No PSI guide lines are "
        "drawn: the usual 0.10/0.25 conventions are not calibrated for these windows."
    )
    for statistic, term in (
        ("psi", "PSI"),
        ("ks_distance", "KS distance"),
        ("standardized_mean_difference", "Standardized mean difference"),
    ):
        _chart(figures.drift_heatmap(stats, statistic), f"drift_{statistic}")
        st.caption(glossary.define(term))
    _download(stats, "feature_stats", view)


def nulls(view: RunView) -> None:
    st.title("Nulls, controls and the leakage canary")
    _method_notice(view)
    metrics = view.tables["metrics"]
    safe_metrics = metrics[metrics["role"] != UNSAFE_ROLE]
    roles = st.columns(3)
    roles[0].info("**Null**: " + glossary.define("Null model"))
    roles[1].info("**Control**: " + glossary.define("Control"))
    roles[2].info("**Canary**: " + glossary.define("Canary"))
    models = _models(view)
    _chart(
        figures.rank_ic_forest(
            safe_metrics,
            models,
            methods=list(figures.METHOD_NAMES),
            decision_method=view.decision_method,
            rates=_rates(view),
        ),
        "null_forest",
    )
    st.caption(
        "Filled black marker: the decision method (mean of fold means). Open markers: "
        "corroborating methods with their stored simulated sizes. Models are grouped by role, "
        "never ranked."
    )
    st.dataframe(
        _metric_table(
            safe_metrics[
                (safe_metrics["metric"] == "mean_rank_ic") & safe_metrics["fold_id"].isna()
            ]
        ),
        hide_index=True,
    )

    st.header("Selection control: fixed rankings")
    observed_label = "pooled mean IC over dates (the value T2 compares with its nulls)"
    observed = {}
    for model in models.itertuples(index=False):
        row = figures.pooled_row(safe_metrics, model.name, "mean_rank_ic", "newey_west")
        if row is not None and row["status"] == "ok":
            observed[model.name] = (float(row["estimate"]), model.role)
    draws = view.tables["null_draws"]
    _chart(figures.static_ordering_strip(draws, observed, observed_label), "static_orderings")
    shares = safe_metrics[safe_metrics["metric"] == "static_ordering_share_at_least"]
    st.caption(
        glossary.define("Static ordering")
        + " Stored share of fixed rankings at least as good as each model (an exact share of a "
        "complete enumeration, not a p-value): "
        + ", ".join(
            f"{r.model} {r.estimate:.2f}"
            for r in shares.itertuples(index=False)
            if pd.notna(r.estimate)
        )
    )
    curves = view.tables["curves"]
    for model in models[models["role"] == "control"].itertuples(index=False):
        if ((curves["model"] == model.name) & (curves["curve"] == "rank_position_occupancy")).any():
            _chart(figures.rank_position_occupancy(curves, model.name), f"occupancy_{model.name}")

    permutation_models = draws.loc[draws["null_kind"] == "block_permutation", "model"].unique()
    if len(permutation_models):
        st.header("Block-permutation null")
        st.caption(glossary.define("Block-permutation null"))
        for name in sorted(permutation_models):
            row = figures.pooled_row(
                safe_metrics, name, "mean_rank_ic_vs_block_permutation", "null_distribution"
            )
            value = observed.get(name, (None, None))[0]
            _chart(
                figures.null_draws_histogram(draws, name, value, observed_label),
                f"permutation_{name}",
            )
            if row is not None and pd.notna(row["p_value"]):
                st.caption(
                    f"{name}: stored one-sided p = {row['p_value']:.2f} from {int(row['n'])} "
                    f"draws (the smallest possible value is 1/(draws+1) = "
                    f"{1 / (int(row['n']) + 1):.2f})."
                )

    stale_rows = safe_metrics[safe_metrics["metric"].str.startswith("rank_ic_minus_stale_")]
    if len(stale_rows):
        st.header("Timeliness control: stale features")
        st.caption(glossary.define("Stale-feature control"))
        for name in sorted(stale_rows["model"].unique()):
            _chart(
                figures.stale_comparison(safe_metrics, name, _method(view), view.decision_method),
                f"stale_{name}",
            )

    st.header("Leakage canary")
    canaries = view.models[view.models["role"] == "canary"]["name"].tolist()
    unsafe = view.models[view.models["role"] == UNSAFE_ROLE]["name"].tolist()
    with st.container(border=True):
        st.warning(
            "Positive control. The UNSAFE reference trains WITHOUT purging on purpose: its "
            "skill is leakage by construction, which proves the measurement would notice "
            "leakage. It is never a strategy, and this panel is the only place it is drawn."
        )
        for reference in view.models[view.models["role"] == UNSAFE_ROLE].itertuples(index=False):
            st.caption(
                f"Recorded for the unsafe reference: eligible as a candidate "
                f"**{reference.eligible_as_candidate}** · output kind **{reference.output_kind}**."
            )
        if canaries and unsafe:
            _chart(
                figures.canary_panel(
                    metrics, canaries[0], unsafe[0], _method(view), view.decision_method
                ),
                "canary",
            )
        else:
            st.caption("No canary stored for this run.")
    _download(safe_metrics, "metrics", view)


def validation(view: RunView) -> None:
    st.title("Validation results")
    metrics = view.tables["metrics"]
    curves = view.tables["curves"]
    models = _selected_models(view)
    if models.empty:
        st.info("Select at least one model in the sidebar.")
        return
    names = models["name"].tolist()
    name = st.selectbox("Model", names, index=_default_index(view, names), key="validation_model")
    model = view.models[view.models["name"] == name].iloc[0]
    st.caption(f"{style.legend_name(name, model['role'])} · output kind: {model['output_kind']}")
    _method_notice(view)
    method = _method(view)
    method_name = figures.method_label(method, view.decision_method)
    cards = st.columns(4)
    _metric_card(
        cards[0],
        f"Mean rank IC, {method_name}",
        figures.pooled_row(metrics, name, "mean_rank_ic", method),
        term="Rank IC",
    )
    _metric_card(
        cards[1],
        "Minimum detectable IC",
        figures.pooled_row(metrics, name, "mean_rank_ic_minimum_detectable", method),
        term="Minimum detectable effect",
    )
    _metric_card(
        cards[2],
        "Share of dates with IC > 0",
        figures.pooled_row(metrics, name, "share_dates_rank_ic_positive", "point"),
    )
    _metric_card(
        cards[3],
        "Expected share under no skill",
        figures.pooled_row(
            metrics, name, "share_dates_rank_ic_positive_expected_under_no_skill", "point"
        ),
    )
    _chart(figures.rank_ic_series(curves, name, view.folds), "ic_series")
    _chart(figures.ic_by_fold(metrics, name, view.decision_method), "ic_by_fold")
    st.subheader("Hit rate")
    _chart(figures.hit_rate_panel(metrics, models, view.decision_method), "hit_rate")
    st.caption(glossary.define("Hit rate"))
    st.subheader("Out-of-sample R² against an explicit baseline")
    baselines = view.record["spec"]["config"]["baselines"]
    baseline = st.selectbox("Baseline", baselines, key="baseline")
    forecasts = models[models["output_kind"] == "forecast"]
    if forecasts.empty:
        st.info("No forecast models selected: scores have no scale-dependent metrics.")
    else:
        _chart(figures.oos_r2_panel(metrics, forecasts, baseline, view.decision_method), "oos_r2")
    st.caption(
        glossary.define("OOS R²")
        + " T2 labels this interval a percentile bootstrap, uncalibrated; the decision test is "
        "the MSE improvement."
    )
    st.subheader("Same-date rank positions")
    curve = st.radio(
        "Outcome",
        ["rank_position_realized_minus_date_mean", "rank_position_realized"],
        horizontal=True,
        key="position_curve",
    )
    ic_row = figures.pooled_row(metrics, name, "mean_rank_ic", method)
    flat = ic_row is not None and str(ic_row["reason"]) == "constant_prediction"
    note = (
        "constant prediction on every date (stored reason): positions hold the date mean "
        "by construction"
        if flat
        else ""
    )
    _chart(
        figures.rank_position_bars(curves, name, curve, note=note),
        "rank_positions",
    )
    _chart(figures.rank_position_occupancy(curves, name), "occupancy")
    st.caption(glossary.define("Rank position"))
    st.subheader("Pooled prediction buckets")
    _chart(
        figures.pooled_deciles(curves, name, model["output_kind"]),
        "deciles",
    )
    st.caption(glossary.define("Pooled deciles"))
    st.subheader("Residuals")
    residual_rows = metrics[
        (metrics["model"] == name)
        & metrics["metric"].isin(["mean_residual", "residual_std", "mae", "rmse"])
        & metrics["fold_id"].notna()
    ]
    if residual_rows.empty:
        reason = metrics[(metrics["model"] == name) & (metrics["metric"] == "rmse")]["reason"]
        reason = reason.dropna()
        st.info(
            f"Residuals unavailable for this model ({reason.iloc[0] if len(reason) else 'not stored'})."
        )
    else:
        st.dataframe(
            residual_rows.pivot(index="fold_id", columns="metric", values="estimate").reset_index(),
            hide_index=True,
        )
        acf_series = sorted(
            curves.loc[(curves["model"] == name) & (curves["curve"] == "residual_acf"), "x_label"]
            .dropna()
            .unique()
        )
        series = st.selectbox("ACF series", acf_series, key="acf_series")
        horizon = view.record["spec"]["label"]["label_spec"]["horizon"]
        _chart(figures.residual_acf(curves, name, series, horizon), "acf")
        _chart(figures.goyal_welch(curves, name, view.folds), "goyal_welch")
        st.caption(glossary.define("Goyal-Welch curve"))
    with st.expander(f"All stored pooled metrics for {name} (table)"):
        st.dataframe(
            _metric_table(metrics[(metrics["model"] == name) & metrics["fold_id"].isna()]),
            hide_index=True,
        )
    _download(metrics[metrics["role"] != UNSAFE_ROLE], "metrics", view)


def _default_index(view: RunView, names: list[str]) -> int:
    """First model whose decision-method mean rank IC is defined (else the first)."""

    for index, name in enumerate(names):
        row = figures.pooled_row(view.tables["metrics"], name, "mean_rank_ic", _method(view))
        if row is not None and row["status"] == "ok":
            return index
    return 0


def calibration(view: RunView) -> None:
    st.title("Calibration")
    st.info(glossary.define("Calibration"))
    metrics = view.tables["metrics"]
    names = ("mz_intercept", "mz_slope", "mz_within_date_slope")
    table = metrics[
        metrics["metric"].isin(names) & metrics["fold_id"].isna() & (metrics["role"] != UNSAFE_ROLE)
    ]
    st.dataframe(_metric_table(table), hide_index=True)
    st.caption(
        glossary.define("Mincer-Zarnowitz")
        + " Intervals use Driscoll-Kraay errors with normal critical values; their size has "
        "not been simulated."
    )
    forecasts = _selected_models(view)
    forecasts = forecasts[forecasts["output_kind"] == "forecast"]
    if forecasts.empty:
        st.info("No forecast models selected; scores are rankings, not calibrated forecasts.")
        return
    options = forecasts["name"].tolist()
    defined = [
        index
        for index, name in enumerate(options)
        if (row := figures.pooled_row(metrics, name, "mz_slope", "driscoll_kraay")) is not None
        and row["status"] == "ok"
    ]
    name = st.selectbox(
        "Forecast model", options, index=defined[0] if defined else 0, key="calibration_model"
    )
    intercept = figures.pooled_row(metrics, name, "mz_intercept", "driscoll_kraay")
    slope = figures.pooled_row(metrics, name, "mz_slope", "driscoll_kraay")
    ok = (
        intercept is not None
        and slope is not None
        and intercept["status"] == "ok"
        and slope["status"] == "ok"
    )
    if not ok:
        st.caption(
            "Mincer-Zarnowitz line not drawn: "
            f"{slope['reason'] if slope is not None else 'not stored'}."
        )
    _chart(
        figures.prediction_vs_realized(
            view.tables["predictions"],
            name,
            float(intercept["estimate"]) if ok else None,
            float(slope["estimate"]) if ok else None,
        ),
        "prediction_scatter",
    )
    _chart(
        figures.pooled_deciles(view.tables["curves"], name, "forecast"),
        "calibration_deciles",
    )
    residuals = metrics[
        (metrics["model"] == name)
        & metrics["metric"].isin(["mean_residual", "mae", "rmse", "cross_sectional_r2"])
        & metrics["fold_id"].isna()
    ]
    st.dataframe(_metric_table(residuals), hide_index=True)


def uncertainty(view: RunView) -> None:
    st.title("Statistical uncertainty")
    st.markdown(
        f"**Decision method: {view.decision_method or 'unavailable'}** "
        f"({view.decision_method_source}). T2 fixes fold-block t as its decision method in "
        "advance (it came closest to nominal size in T2's design simulations). Newey-West and "
        "the circular block bootstrap are shown as corroborating evidence with their nominal "
        "intervals and this run's stored simulated sizes."
    )
    metrics = view.tables["metrics"]
    models = _selected_models(view)
    if models.empty:
        st.info("Select at least one model in the sidebar.")
        return
    columns = st.columns(2)
    names = models["name"].tolist()
    name = columns[0].selectbox(
        "Model", names, index=_default_index(view, names), key="uncertainty_model"
    )
    candidates = sorted(
        metrics.loc[
            (metrics["model"] == name)
            & metrics["fold_id"].isna()
            & metrics["inference_method"].isin(figures.METHOD_NAMES),
            "metric",
        ].unique()
    )
    metric = columns[1].selectbox(
        "Statistic",
        candidates,
        index=candidates.index("mean_rank_ic") if "mean_rank_ic" in candidates else 0,
        key="uncertainty_metric",
    )
    rates = _rates(view)
    _chart(
        figures.uncertainty_forest(metrics, name, metric, view.decision_method, rates),
        "uncertainty_forest",
    )
    rows = metrics[
        (metrics["model"] == name) & (metrics["metric"] == metric) & metrics["fold_id"].isna()
    ]
    st.dataframe(_metric_table(rows), hide_index=True)
    st.caption(
        glossary.define("HAC / Newey-West")
        + " "
        + glossary.define("Block bootstrap")
        + " "
        + glossary.define("Effective sample size")
    )
    calibration_record = view.record.get("inference_calibration") or {}
    threshold = view.record["spec"]["config"].get("max_false_positive_rate")
    st.subheader("Simulated false-positive rates (stored with this run)")
    if rates:
        _chart(figures.false_positive_rates(rates, threshold, view.decision_method), "fpr")
        lines = [
            f"{figures.METHOD_NAMES.get(method, method)}: "
            + ", ".join(f"{signal} {rates[signal][method]:.1%}" for signal in rates)
            for method in (*figures.METHOD_NAMES, "iid")
            if all(method in rates[signal] for signal in rates)
        ]
        st.caption(
            f"{calibration_record.get('reps')} no-skill replicates per signal at "
            f"{calibration_record.get('n_dates')} dates; nominal level 5%, stored threshold "
            f"{threshold}. Stored rates: " + "; ".join(lines) + ". The naive iid standard "
            "error is listed for contrast only. Rates above the threshold mean that method "
            "over-rejects for that signal."
        )
    else:
        st.info("No inference calibration stored in this run.")


def warnings_page(view: RunView) -> None:
    st.title("Research warnings")
    st.caption("Quoted from stored outputs. The dashboard never turns a warning into a failure.")
    notices = research_notices(view)
    for level, writer in (("blocking", st.error), ("warning", st.warning), ("notice", st.info)):
        for notice in [n for n in notices if n.level == level]:
            writer(f"**{notice.title}.** {notice.detail} _(source: {notice.source})_")
            if notice.rows is not None and len(notice.rows):
                st.dataframe(notice.rows, hide_index=True)


def glossary_page() -> None:
    st.title("Glossary")
    for term, text in glossary.TERMS.items():
        st.markdown(f"**{term}.** {text}")


def render(page) -> None:
    """Run a page body against the selected run (the sidebar is drawn by the app)."""

    view = st.session_state.get("_view")
    if view is None:
        st.info("No displayable run is selected.")
        return
    page(view)
