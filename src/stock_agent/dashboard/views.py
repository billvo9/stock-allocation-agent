"""
Streamlit page bodies. Everything shown comes from a loaded RunView; this
module lays out stored values, stored statuses and stored reasons, and
explains them. It computes no statistics: the only derived series (the
date-normalized Goyal-Welch advantage and descriptive filtered summaries)
come from evaluation.py and are drawn only after they reproduce T2's
stored values. Layout uses the shared components (components.py).

The unsafe canary reference (a deliberately leaky positive control) is
drawn only in the canary panel of the Nulls & controls page; everywhere
else it is filtered out by role.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import streamlit as st

from stock_agent.dashboard import components as ui
from stock_agent.dashboard import evaluation, figures, glossary, regimes, style
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
        ui.notice(
            "notice",
            f"No runs under {root / 'runs'}. Create one with scripts/run_null_diagnostics.py.",
            where=st.sidebar,
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
        ui.notice("blocking", f"Refused: {refused.reason}", where=st.sidebar)
        ui.notice("blocking", f"This run is not displayed ({refused.reason}). {refused.detail}")
        return None
    status = _status(view)
    ui.pill(status.level, where=st.sidebar, live=True)
    asof = view.evaluation_asof.date() if view.evaluation_asof is not None else "not recorded"
    st.sidebar.caption(
        f"Mode: {view.record['spec']['lockbox']['mode']} · evaluation as-of {asof} · "
        f"lockbox from {view.holdout_start.date()}"
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
        ui.notice(
            "blocking",
            f"Decision method {view.decision_method_source}: headline rows are not selected "
            "for this run.",
        )


def _rates(view: RunView) -> dict:
    return (view.record.get("inference_calibration") or {}).get("rates") or {}


def _chart(figure, key: str) -> None:
    ui.chart(figure, key)


def _download(frame: pd.DataFrame, name: str, view: RunView) -> None:
    ui.download(frame, name, view.run_id)


def _metric_table(rows: pd.DataFrame) -> pd.DataFrame:
    return rows.loc[:, METRIC_COLUMNS]


def _metric_card(
    column, label: str, row: pd.Series | None, *, fmt: str = "{:.3f}", term: str | None = None
):
    ui.estimate_card(column, label, row, fmt=fmt, help=glossary.define(term) if term else None)


def _day(value) -> str:
    return "—" if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d")


# --- pages ---


def overview(view: RunView) -> None:
    ui.page_header(
        "Run overview",
        eyebrow="Run",
        purpose="What this run is, whether its stored checks hold, and what it was evaluated on.",
    )
    spec = view.record["spec"]
    checks = view.tables["checks"]
    status = _status(view)
    ui.status_banner(status.level, status.reasons)
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
        ui.notice(
            "notice",
            f"Output schema {view.schema_version}: fields this run did not record are shown as "
            "not recorded, or as derived from stored rows where labelled. "
            + "; ".join(view.legacy_gaps)
            + ".",
        )
    ui.stat_row(
        [
            ui.Stat("Folds", len(view.folds)),
            ui.Stat("Scored dates", scored["date"].nunique()),
            ui.Stat("Scored rows per model", int(scored.groupby("model").size().max())),
            ui.Stat("Symbols", len(spec["universe"]["symbols"])),
            ui.Stat("Active research warnings", status.counts["research_warnings"]),
        ]
    )
    _evaluation_sample(view)

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
            ui.notice("notice", f"Recorded universe note: {universe['note']}")
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
    ui.page_header(
        "Fold construction",
        eyebrow="Construction",
        purpose=(
            "Read from the stored fold table; folds are never rebuilt here. The purge gap is "
            "drawn between the last training row and the test start; its exact dates are not "
            "stored, only the number of purged rows."
        ),
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
    ui.page_header(
        "Data and feature diagnostics",
        eyebrow="Construction",
        purpose="What each fold's fit sees, and how test windows differ from it.",
    )
    stats = view.tables["feature_stats"]
    training = stats[stats["scope"] == "training"]
    feature_names = sorted(f for f in training["feature"].unique() if f != "__design__")
    ui.pill("NOTICE", "Training scope")
    ui.section(
        "Training-scope diagnostics",
        "Computed by T2 on each fold's training rows only: what a fold's fit sees.",
    )
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

    ui.pill("WARNING", "Evaluation only")
    ui.section("Evaluation-only diagnostics")
    ui.notice(
        "warning",
        "Test-window features compared with training features. Distances only: no p-values, no "
        "T2 judgment, and never a reason to remove or change a feature. No PSI guide lines are "
        "drawn: the usual 0.10/0.25 conventions are not calibrated for these windows.",
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
    ui.page_header(
        "Nulls, controls and the leakage canary",
        eyebrow="Evidence",
        purpose="What no skill, selection effects and leakage look like on this run.",
    )
    _stored_results_notice(view)
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

    ui.section("Selection control: fixed rankings")
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
        ui.section("Block-permutation null")
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
        ui.section("Timeliness control: stale features")
        st.caption(glossary.define("Stale-feature control"))
        for name in sorted(stale_rows["model"].unique()):
            _chart(
                figures.stale_comparison(safe_metrics, name, _method(view), view.decision_method),
                f"stale_{name}",
            )

    ui.section("Leakage canary")
    canaries = view.models[view.models["role"] == "canary"]["name"].tolist()
    unsafe = view.models[view.models["role"] == UNSAFE_ROLE]["name"].tolist()
    with ui.card("canary"):
        ui.notice(
            "warning",
            "Positive control. The UNSAFE reference trains WITHOUT purging on purpose: its "
            "skill is leakage by construction, which proves the measurement would notice "
            "leakage. It is never a strategy, and this panel is the only place it is drawn.",
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
    ui.page_header(
        "Validation results",
        eyebrow="Evidence",
        purpose="Out-of-sample ranking, sign and error metrics as stored by T2.",
    )
    _stored_results_notice(view)
    metrics = view.tables["metrics"]
    curves = view.tables["curves"]
    models = _selected_models(view)
    if models.empty:
        ui.notice("notice", "Select at least one model in the sidebar.")
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
    st.subheader("Out-of-sample R² against a named comparator")
    labels = {c.name: c.label for c in evaluation.comparators(view.record, view.models)}
    baseline = st.selectbox(
        "Comparator", list(labels), format_func=labels.__getitem__, key="baseline"
    )
    forecasts = models[models["output_kind"] == "forecast"]
    if forecasts.empty:
        ui.notice("notice", "No forecast models selected: scores have no scale-dependent metrics.")
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
        ui.notice(
            "notice",
            f"Residuals unavailable for this model ({reason.iloc[0] if len(reason) else 'not stored'}).",
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
    st.page_link(
        "page_scripts/forecast_error.py",
        label="Forecast error over time (Goyal-Welch) against recorded comparators",
    )
    with st.expander(f"All stored pooled metrics for {name} (table)"):
        st.dataframe(
            _metric_table(metrics[(metrics["model"] == name) & metrics["fold_id"].isna()]),
            hide_index=True,
        )
    _download(metrics[metrics["role"] != UNSAFE_ROLE], "metrics", view)


def _scored_pending(view: RunView) -> pd.DataFrame:
    """Pending stored predictions of registered models (the unsafe reference stays in its panel)."""

    pending = view.pending
    return pending[pending["role"] != UNSAFE_ROLE] if len(pending) else pending


def _evaluation_sample(view: RunView) -> None:
    """Evaluation as-of, mature and pending stored predictions (pending rows are listed)."""

    predictions = view.tables["predictions"]
    scored = predictions[predictions["role"] != UNSAFE_ROLE]
    pending = _scored_pending(view)
    asof = view.evaluation_asof
    with ui.card("evaluation-sample"):
        st.markdown("**Evaluation sample**")
        ui.stat_row(
            [
                ui.Stat(
                    "Evaluation as-of",
                    "not recorded" if asof is None else str(asof.date()),
                    help=glossary.define("Evaluation as-of"),
                ),
                ui.Stat("Mature predictions", len(scored) - len(pending)),
                ui.Stat(
                    "Pending predictions",
                    len(pending),
                    help=glossary.define("Pending forecast"),
                ),
            ]
        )
        st.caption(
            f"Model-rows over the registered models. Source of the as-of: "
            f"{evaluation.EVALUATION_ASOF_SOURCE}, the last session of the truncated development "
            "frame. Only mature predictions (label ended on or before it) enter the series the "
            "dashboard derives."
        )
        if len(pending):
            with st.expander(f"{len(pending)} pending predictions"):
                st.dataframe(pending, hide_index=True)


def _stored_results_notice(view: RunView) -> None:
    """On pages of T2's stored results: say when those results include pending rows."""

    pending = _scored_pending(view)
    if len(pending):
        ui.notice(
            "blocking",
            f"T2's stored results on this page include {len(pending)} predictions whose labels "
            "were not realized by the evaluation as-of; the run is blocked. The pending rows "
            "are listed on the overview.",
        )


GW_MODEL_KEY = "gw_model"
GW_COMPARATORS_KEY = "gw_comparators"
ROLE_PREFERENCE = ("candidate", "canary", "control", "null")
# The estimand behind each method's stored mse_improvement estimate.
D_ESTIMANDS = {
    "fold_block_t": "mean of per-fold means of d_t",
    "newey_west": "mean of d_t over dates",
    "circular_block_bootstrap": "mean of d_t over dates",
}


def _forecast_models(view: RunView) -> pd.DataFrame:
    models = _models(view)
    return models[models["output_kind"] == "forecast"]


def _label_lead(view: RunView) -> int:
    spec = view.record["spec"]["label"]["label_spec"]
    return int(spec["horizon"]) + int(spec["entry_lag"])


def _sample_filter(series: pd.DataFrame, regime_table: pd.DataFrame, reasons: dict) -> tuple:
    """Reader-chosen descriptive filter. Returns (SampleFilter, regime definition or None)."""

    first, last = series["date"].min().date(), series["date"].max().date()
    with st.expander("Filter the sample (descriptive only)", expanded=False):
        st.caption(glossary.define("Descriptive filter"))
        start, end = st.slider(
            "Forecast dates",
            min_value=first,
            max_value=last,
            value=(first, last),
            format="YYYY-MM-DD",
            key="gw_dates",
        )
        definitions = {d.key: d for d in regimes.DEFINITIONS}
        key = st.selectbox(
            "Regime",
            [None, *definitions],
            format_func=lambda k: "none" if k is None else definitions[k].label,
            key="gw_regime",
        )
        value = None
        if key is not None:
            definition = definitions[key]
            if definition.point_in_time:
                st.caption(f"Point-in-time. {definition.description}")
            else:
                ui.notice("warning", f"Retrospective (ex post) labels. {definition.description}")
            if key in reasons:
                ui.notice("notice", f"Not available for this run: {reasons[key]}.")
            else:
                values = sorted(pd.unique(regime_table[key]))
                value = st.selectbox("Regime value", values, key="gw_regime_value")
    chosen = evaluation.SampleFilter(
        start=None if start == first else pd.Timestamp(start, tz="UTC"),
        end=None if end == last else pd.Timestamp(end, tz="UTC"),
        regime=key,
        regime_value=value,
    )
    return chosen, (definitions[key] if key is not None else None)


def forecast_error(view: RunView) -> None:
    ui.page_header(
        "Forecast error (Goyal-Welch)",
        eyebrow="Evidence",
        purpose=(
            "Squared-error advantage of a forecast model over recorded comparators, on "
            "identical, mature observations. Forecast error is not portfolio performance."
        ),
    )
    _forecast_error_body(view)
    ui.future_state(
        "portfolio-equity",
        "Portfolio equity curve",
        "Defined in T5: execution timing, allocation rules, transaction costs and risk "
        "constraints are not specified yet, so no equity or return path is shown.",
    )


def _forecast_error_body(view: RunView) -> None:
    label = view.record["spec"]["label"]
    st.caption(
        f"Label `{label['label_id']}`. Which comparator is natural depends on the label: the "
        "expanding mean of training labels for raw returns, zero for excess returns."
    )
    forecasts = _forecast_models(view)
    if forecasts.empty:
        ui.notice("notice", "This run has no forecast models; scores have no error scale.")
        return
    names = forecasts["name"].tolist()
    roles = dict(zip(forecasts["name"], forecasts["role"], strict=True))
    default = min(
        range(len(names)),
        key=lambda i: (
            ROLE_PREFERENCE.index(roles[names[i]]) if roles[names[i]] in ROLE_PREFERENCE else 9,
            i,
        ),
    )
    model = st.selectbox(
        "Forecast model",
        names,
        index=default,
        format_func=lambda n: style.legend_name(n, roles[n]),
        key=GW_MODEL_KEY,
    )
    if "candidate" not in roles.values():
        st.caption(
            "This run registers no candidate model: every forecast here is a null, control or "
            "canary, so none of these curves is evidence of skill."
        )
    _stored_results_notice(view)
    available = [c for c in evaluation.comparators(view.record, view.models) if c.name != model]
    omitted = evaluation.omitted_comparators(view.record, view.models)
    if not available:
        ui.notice("notice", f"No recorded comparator other than {model} itself.")
        return
    by_name = {c.name: c for c in available}
    nulls = [c.name for c in available if c.kind == evaluation.NULL_BASELINE]
    chosen_names = st.multiselect(
        "Comparators",
        list(by_name),
        default=nulls,  # never the selection control by default
        format_func=lambda n: by_name[n].label,
        key=GW_COMPARATORS_KEY,
        help=glossary.define("Comparator"),
    )
    with st.expander("Comparator identities"):
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "comparator": c.name,
                        "kind": c.kind,
                        "recorded note": c.note,
                        "how it is formed (dashboard)": evaluation.FORMED,
                    }
                    for c in available
                ]
            ),
            hide_index=True,
        )
        if omitted:
            st.caption(
                "Recorded baselines not offered: "
                + "; ".join(f"{name} ({reason})" for name, reason in omitted)
                + "."
            )
    if not chosen_names:
        ui.notice("notice", "Select at least one comparator.")
        return

    predictions, metrics, curves = (view.tables[t] for t in ("predictions", "metrics", "curves"))
    asof = view.evaluation_asof
    drawn = []
    for name in chosen_names:
        comparator = by_name[name]
        try:
            series = evaluation.squared_error_advantage(predictions, model, name, asof)
        except ValueError as error:
            ui.notice("blocking", f"vs {comparator.label}: not drawn ({error}).")
            continue
        check = evaluation.reproduce_t2(series, metrics, curves, model, name)
        if not check.ok:
            ui.notice("blocking", f"vs {comparator.label}: not drawn. {check.detail}")
            continue
        drawn.append((comparator, series))
    if not drawn:
        return
    pending = int((view.pending["model"] == model).sum()) if len(view.pending) else 0
    x_title = f"forecast date (label ends {_label_lead(view)} sessions later)"

    regime_table, reasons = regimes.regime_table(view.tables["inputs"], drawn[0][1]["date"])
    sample, definition = _sample_filter(drawn[0][1], regime_table, reasons)
    note = "full sample" if sample.is_full_sample else "filtered sample (descriptive)"
    if definition is not None and not definition.point_in_time and not sample.is_full_sample:
        note += ", retrospective regime"
    marked = [(c, frame.assign(included=sample.mask(frame, regime_table))) for c, frame in drawn]
    shown = [(c, frame[frame["included"]].drop(columns="included")) for c, frame in marked]
    _chart(
        figures.goyal_welch_date_normalized(
            marked, model, view.folds, sample_note=note, pending=pending, x_title=x_title
        ),
        "gw_primary",
    )
    st.caption(
        glossary.define("Date-normalized advantage")
        + " Each comparator's full-sample series reproduced T2's stored values before it was "
        "drawn; a filtered curve re-accumulates those values over the chosen dates only."
    )

    if sample.is_full_sample:
        ui.section("Stored inference (T2, full sample)")
        method = _method(view)
        for comparator, _ in drawn:
            columns = st.columns(2)
            ui.estimate_card(
                columns[0],
                f"d_t vs {comparator.label}",
                figures.pooled_row(metrics, model, f"mse_improvement_vs_{comparator.name}", method),
                method=(
                    f"{figures.method_label(method, view.decision_method)}: "
                    f"{D_ESTIMANDS.get(method, 'stored estimate')}"
                ),
            )
            ui.estimate_card(
                columns[1],
                f"OOS R² vs {comparator.label}",
                figures.pooled_row(
                    metrics, model, f"oos_r2_vs_{comparator.name}", "circular_block_bootstrap"
                ),
                method="percentile interval, uncalibrated (observation-weighted)",
                help=glossary.define("OOS R²"),
            )
    else:
        ui.notice(
            "notice",
            "Filtered inference is not precomputed: T2 computed dependence-aware intervals for "
            "the full sample only. The numbers below are descriptive counts and averages.",
        )
        ic = evaluation.rank_ic_dates(curves, predictions, model, asof)
        ic = ic[(ic["maturity"] == evaluation.MATURE) & (ic["status"] == "ok")]
        ic = ic[sample.mask(ic.assign(fold_id=ic["fold_id"].astype("Int64")), regime_table)]
        rows = []
        for comparator, frame in shown:
            summary = evaluation.descriptive_summary(frame)
            rows.append(
                {
                    "comparator": comparator.label,
                    "dates": summary["dates"],
                    "observations": summary["observations"],
                    "mean d_t": summary["mean_d"],
                    "share of dates model better": summary["share_dates_model_better"],
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption(
            "Mean stored daily rank IC over the same mature dates (descriptive): "
            + (
                f"{ic['value'].mean():.3f} over {len(ic)} dates."
                if len(ic)
                else "no defined dates."
            )
        )
        st.caption(_overlap_note(view, metrics, model, drawn[0][0].name))

    ui.section("Per date")
    focus = st.selectbox(
        "Comparator for the per-date view",
        [c.name for c, _ in shown],
        format_func=lambda n: by_name[n].label,
        key="gw_focus",
    )
    frame = next(f for c, f in shown if c.name == focus)
    _chart(
        figures.squared_error_advantage_by_date(frame, by_name[focus], model, x_title=x_title),
        "gw_daily",
    )

    with st.expander("Observation-weighted cumulative (secondary diagnostic, stored by T2)"):
        _chart(
            figures.goyal_welch_observation_weighted(
                curves, model, view.folds, [c for c, _ in drawn]
            ),
            "gw_observation_weighted",
        )
        st.caption(glossary.define("Observation-weighted cumulative") + " Always the full sample.")


def _overlap_note(view: RunView, metrics: pd.DataFrame, model: str, comparator: str) -> str:
    """Why a slice of k dates holds far fewer independent outcomes than k (stored n, n_eff)."""

    row = figures.pooled_row(metrics, model, f"mse_improvement_vs_{comparator}", "newey_west")
    lead = _label_lead(view)
    text = (
        f"Each label spans {lead} sessions after its forecast date, so neighbouring dates share "
        "most of their outcome window, and a slice is scored on outcomes realized up to "
        f"{lead} sessions after its last date."
    )
    if row is not None and pd.notna(row["n_eff"]):
        text += (
            f" For the full sample T2 stored an effective sample size of {row['n_eff']:.0f} "
            f"for {int(row['n'])} dates."
        )
    return text


def _default_index(view: RunView, names: list[str]) -> int:
    """First model whose decision-method mean rank IC is defined (else the first)."""

    for index, name in enumerate(names):
        row = figures.pooled_row(view.tables["metrics"], name, "mean_rank_ic", _method(view))
        if row is not None and row["status"] == "ok":
            return index
    return 0


def calibration(view: RunView) -> None:
    ui.page_header(
        "Calibration",
        eyebrow="Evidence",
        purpose="Whether forecasts are right on average and in scale (forecast models only).",
    )
    _stored_results_notice(view)
    ui.notice("notice", glossary.define("Calibration"))
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
        ui.notice(
            "notice", "No forecast models selected; scores are rankings, not calibrated forecasts."
        )
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
    ui.page_header(
        "Statistical uncertainty",
        eyebrow="Evidence",
        purpose="Every stored inference method for one statistic, with its simulated size.",
    )
    _stored_results_notice(view)
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
        ui.notice("notice", "Select at least one model in the sidebar.")
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
    ui.section("Simulated false-positive rates (stored with this run)")
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
        ui.notice("notice", "No inference calibration stored in this run.")


def warnings_page(view: RunView) -> None:
    ui.page_header(
        "Research warnings",
        eyebrow="Run",
        purpose="Quoted from stored outputs. The dashboard never turns a warning into a failure.",
    )
    notices = research_notices(view)
    for level in ("blocking", "warning", "notice"):
        for notice in [n for n in notices if n.level == level]:
            ui.notice(level, f"**{notice.title}.** {notice.detail} _(source: {notice.source})_")
            if notice.rows is not None and len(notice.rows):
                st.dataframe(notice.rows, hide_index=True)


def glossary_page() -> None:
    ui.page_header("Glossary", eyebrow="Reference", purpose="Definitions used across the pages.")
    for term, text in glossary.TERMS.items():
        st.markdown(f"**{term}.** {text}")


def render(page) -> None:
    """Run a page body against the selected run (the sidebar is drawn by the app)."""

    view = st.session_state.get("_view")
    if view is None:
        ui.notice("notice", "No displayable run is selected.")
        return
    page(view)
