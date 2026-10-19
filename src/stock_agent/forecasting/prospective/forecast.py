"""
Prospective forecast records, their scores and the cumulative report.

Forecast (`record_forecast`). For one as-of session (the last complete
session, `forecast_close`):

1. The frozen bundle and its registry entries are verified: every file
   against its manifest, every entry's artifacts against their hashes, the
   entry's bundle reference against this bundle, and the computed stage is
   at most `development_evaluated`. The bundle's zero-volatility rule must
   be its config's (`zero_vol_rule_mismatch`).
2. The timing rules hold (window.py): as_of >= H_end; created_at at or after
   the close's assumed availability (18:00 America/New_York); created_at
   before the record deadline (09:30 America/New_York on the next weekday).
   The ledger opens with its registration (registration_record.py;
   `not_registered` otherwise), the models and the running code are exactly
   what it registered (`registration_mismatch`), and the close is after the
   registration time (`before_registration`). The first record, session 1,
   also needs every fund's complete window from spot-checked batches
   (`not_first_eligible_close`); nothing is recorded for a session that is
   not eligible.
3. The window (`load_inference_window(store_root, assets, forecast_close,
   lookback, retrieved_by=created_at)`: lookback daily returns, so
   lookback + 1 closes) is checked by window.coerce_window: nothing after the
   close, no batch retrieved after created_at, sessions equal to the
   exchange calendar.
4. An asset with a close missing in its window is refused
   (`incomplete_inference_window`), never median-filled. For every other
   asset, features are computed at the close from the window's F5.1 total
   returns with the training formulas (series.py), and each frozen fit
   predicts.
5. One "forecast" entry is appended to the ledger (with the window's
   batches, missing closes, revisions and completeness, and any refused
   assets, its registration and its session number and phase), refused if
   the ledger already holds a forecast or a refusal for this as-of session
   (`duplicate_forecast`: a session is recorded once, and a refusal counts as
   its record). When every asset is refused, a "refusal" entry is appended
   instead: never a forecast.

Per asset, horizon and target a row holds the point forecast, the 10% and
90% band ends, the historical-mean null, the model id, version and entry
hash, the computed stage, the development band coverage and the fixed
labels. Units: total log return over the h sessions after entry (entry is
the `entry_lag`-th session after as_of), and log daily realized volatility.

Score (`score_matured`). Once the exit session (entry + h) has closed and its
close is available, the outcome is computed from the prospective store
through the same loader, and one "score" entry records, per matured row,
the outcome, whether it lies inside the band, its error and the null's
error, with its variant (bundle, config hash, model id and version) and the
outcome window's lineage. The window runs from the earliest pending as-of
session to the scoring close, read with retrieved_by = scored_at. Forecast
entries are never edited; a row is scored once. Scoring uses the
registration's zero-volatility rule and refuses code, bundles or configs
other than the registered ones.

Zero realized volatility (zero_vol.py). One treatment everywhere: a forecast
records the rule its bundle was trained under, and scoring refuses any other
rule (`zero_vol_rule_not_trained`); a ledger is never scored under two rules
(`zero_vol_rule_mismatch`). The rule in force is `zero_vol.CURRENT` (a zero
outcome is outside the band); the registered config names it, and `freeze`,
the job and its CLI cannot select the unapproved research option
`zero_vol.B2`. Each forecast row records its unadjusted as-of close, and
B2's floor is computed from that recorded value, so a later revision of the
close cannot move it. Every scored volatility row records the rule,
sigma_raw, measured_zero, floored, sigma_floor and the as-of close used.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from stock_agent.forecasting.prospective import (
    cadence,
    frozen,
    ledger,
    registration_record,
    series,
    window,
    zero_vol,
)
from stock_agent.forecasting.prospective import (
    environment as job_environment,
)
from stock_agent.forecasting.prospective.config import parse_config
from stock_agent.forecasting.prospective.refusal import ForecastRefused
from stock_agent.model_diagnostics import inference as inf
from stock_agent.model_diagnostics.record import sha256_json
from stock_agent.registry import EntryRef, RegistryReader
from stock_agent.registry.stages import STAGES

# 1.1: the bundle's zero-volatility rule; rows' as-of close. 1.2: the registration and session.
# 1.3: the environment check.
FORECAST_SCHEMA_VERSION = "1.3"
# 1.1: volatility rows record the zero-volatility rule and flags. 1.2: the registration.
# 1.3: the environment check.
SCORE_SCHEMA_VERSION = "1.3"
ALLOWED_STAGES = STAGES[:2]  # engineering_only, development_evaluated; never higher
UNITS = {
    "total_log_return": "total log return over the horizon (F5.1, distributions reinvested)",
    "log_realized_volatility": "log of daily realized volatility over the horizon",
}
Loader = Callable[..., object]


def display_percent(target: str, value: float | None) -> float | None:
    """A record value in the panel's percent units: log return x 100, or daily volatility %."""

    if value is None or not math.isfinite(value):
        return None
    return 100.0 * (value if target == "total_log_return" else math.exp(value))


@dataclass(frozen=True, eq=False)
class Models:
    bundle: frozen.Bundle
    entries: Mapping[str, Mapping[str, object]]  # "<target>:h<h>" -> model identity and stage

    @property
    def config(self):
        return parse_config(self.bundle.document["config"])

    @property
    def zero_vol_rule(self) -> str:
        return training_rule(self.bundle)


def training_rule(bundle: frozen.Bundle) -> str:
    """
    The zero-volatility rule every fit of the bundle was trained under. A fit
    frozen before the rule was recorded used the rule in force.
    """

    return registration_record.bundle_rule(bundle)


def load_models(bundle_path: str | Path, registry: RegistryReader) -> Models:
    """The verified frozen bundle and the identity and stage of each registered fit."""

    bundle = frozen.read_bundle(bundle_path)
    config = parse_config(bundle.document["config"])
    if config.sha256 != bundle.document["config_sha256"]:
        raise ForecastRefused("models_unverified", "the bundle's config does not match its hash")
    entries = {}
    for key, identity in bundle.document["registry"].items():
        ref = EntryRef(identity["model_id"], identity["version"])
        entry = registry.entry(ref)
        frozen_refs = [a for a in entry.artifacts if a.kind == frozen.BUNDLE_KIND]
        if (
            len(frozen_refs) != 1
            or frozen_refs[0].sha256 != bundle.manifest_sha256
            or Path(frozen_refs[0].path).name != bundle.bundle_id
        ):
            raise ForecastRefused("models_unverified", f"{ref} does not register this bundle")
        if not registry.artifacts_verified(entry):
            raise ForecastRefused("models_unverified", f"{ref}: an artifact does not verify")
        if entry.config.get("experiment_config_sha256") != config.sha256:
            raise ForecastRefused("models_unverified", f"{ref} was registered with another config")
        stage = registry.assessment(ref).stage
        if stage not in ALLOWED_STAGES:
            raise ForecastRefused("stage", f"{ref} computes {stage}, above {ALLOWED_STAGES[-1]}")
        entries[key] = {
            "model_id": ref.model_id,
            "model_version": ref.version,
            "entry_sha256": entry.entry_sha256,
            "registered_at": entry.created_at,
            "stage": stage,
        }
    # Refuses a bundle whose fits disagree on the rule, or name another rule than its config.
    registration_record.checked_rule(bundle)
    return Models(bundle, entries)


def timing(config, as_of: pd.Timestamp) -> dict[str, pd.Timestamp]:
    zone = config.prospective.timezone
    return {
        "close_available_at": window.local_instant(
            as_of, config.prospective.close_available_local, zone
        ),
        "record_deadline": window.local_instant(
            window.next_weekday(as_of), config.prospective.record_deadline_local, zone
        ),
    }


def _features_at_close(win: window.InferenceWindow, asset: str) -> tuple[bool, np.ndarray]:
    returns = series.daily_log_returns(
        win.sessions, win.prices, win.distributions, win.splits, asset
    )
    frame = series.asset_frame(returns, asset)
    last = frame.iloc[-1]
    if last["date"] != win.forecast_close:
        raise ForecastRefused("window", f"{asset}: features do not end at the forecast close")
    values = last[list(series.FEATURES)].to_numpy(dtype=np.float64)
    return bool(last["close_observed"]), values


def forecast_rows(
    models: Models, win: window.InferenceWindow
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """
    Every (asset, horizon, target) row of one forecast, from the window alone,
    and the assets refused because a close is missing in their window
    (`incomplete_inference_window`): their features are never median-filled.
    """

    config = models.config
    bundle = models.bundle
    rows: list[dict[str, object]] = []
    incomplete = win.incomplete_assets(config.assets)
    refusals = [
        {"asset": asset, "reason": window.INCOMPLETE, "sessions_without_close": sessions}
        for asset, sessions in incomplete.items()
    ]
    for asset in config.assets:
        if asset in incomplete:
            continue
        observed, x = _features_at_close(win, asset)
        if not observed:  # incomplete_assets lists every missing close
            raise ForecastRefused("window", f"{asset}: no close at the forecast close")
        imputed = [name for name, value in zip(series.FEATURES, x, strict=True) if np.isnan(value)]
        # The unadjusted as-of close, recorded so that B2's floor is fixed at the decision time.
        as_of_close = float(
            zero_vol.as_of_closes(pd.Series([win.forecast_close]), win.prices, asset)[0]
        )
        for horizon in config.horizons:
            fits = {t: bundle.fit(asset, horizon, t) for t in config.targets}
            volatility = frozen.ridge_from_json(fits["log_realized_volatility"]["model"])
            v_hat = float(volatility.predict(x[None, :])[0])
            for target in config.targets:
                fit = fits[target]
                identity = models.entries[f"{target}:h{horizon}"]
                band = fit["band"]
                base = {
                    "asset": asset,
                    "horizon": horizon,
                    "target": target,
                    "entry_lag": config.entry_lag,
                    "units": UNITS[target],
                    **identity,
                    "development": fit["development"],
                    "band_method": band["method"],
                    "band_level": band["level"],
                }
                if target == "log_realized_volatility":
                    point, scale = v_hat, 1.0
                else:
                    model = frozen.ridge_from_json(fit["model"])
                    point = float(model.predict(x[None, :])[0])
                    scale = math.exp(v_hat) * math.sqrt(horizon)
                rows.append(
                    {
                        **base,
                        "status": "ok",
                        "reason": None,
                        "point": point,
                        "band_low": point + scale * band["lower"],
                        "band_high": point + scale * band["upper"],
                        "band_scale": scale,
                        "null": fit["null"]["value"],
                        "null_model": fit["null"]["model"],
                        # A complete window can still leave a feature undefined (no price
                        # movement, or a total return the approved formulas cannot give, such
                        # as cash on a split date without its share basis): imputed as in
                        # training and listed here.
                        "imputed_features": imputed,
                        "as_of_close": as_of_close,
                    }
                )
                rows[-1]["display_percent"] = {
                    key: display_percent(target, rows[-1][key])
                    for key in ("point", "band_low", "band_high", "null")
                }
    return rows, refusals


RECORD_KINDS = ("forecast", "refusal")


def _guard(as_of: str, bundle_id: str, registration_sha256: str, records: int):
    """
    Under the append lock: the same registration, no record since the read,
    and no earlier record (forecast or refusal) for the session.
    """

    def guard(entries: tuple[ledger.LedgerEntry, ...]) -> None:
        if not entries or entries[0].entry_sha256 != registration_sha256:
            raise ForecastRefused("ledger_changed", "the ledger's registration changed; run again")
        if sum(entry.kind in RECORD_KINDS for entry in entries) != records:
            raise ForecastRefused(
                "ledger_changed", "a record was appended while this forecast ran; run again"
            )
        for entry in entries:
            # One record per session: a refusal counts as the session's record (section 3),
            # so neither a second forecast nor a forecast after a refusal is appended.
            if entry.kind in RECORD_KINDS and entry.payload.get("as_of_session") == as_of:
                raise ForecastRefused(
                    "duplicate_forecast",
                    f"the ledger already holds a {entry.kind} for {as_of} (entry "
                    f"{entry.sequence}, bundle {bundle_id}); a session is recorded once",
                )

    return guard


def registered(entries: Sequence[ledger.LedgerEntry]) -> registration_record.Registration:
    """The ledger's registration; refused (`not_registered`) when it has none."""

    found = registration_record.current(entries)
    if found is None:
        raise ForecastRefused(
            "not_registered",
            "the ledger holds no registration entry; nothing is recorded before registration",
        )
    return found


def first_eligible_refusal(win: window.InferenceWindow, assets: Sequence[str]) -> str | None:
    """
    Why the window's close cannot be session 1 (section 4, condition 2), or
    None: every fund needs its complete window, from batches whose
    distributions were spot-checked against the owner's anchors
    (window.session_one_gaps, the readiness report's test too).
    """

    gaps = window.session_one_gaps(win, assets)
    if gaps["incomplete"]:
        return f"the window is incomplete for {sorted(gaps['incomplete'])}"
    if gaps["no_batch_lineage"]:
        return "the window names no batch"
    if gaps["unchecked_batches"]:
        return (
            f"batches {gaps['unchecked_batches']} record no spot check against the owner's anchors"
        )
    return None


def session_of(
    records: Sequence[ledger.LedgerEntry], as_of: pd.Timestamp, calendar: window.Calendar
) -> dict[str, object]:
    """Session 1 is the first record's as-of session; numbers count calendar sessions."""

    first = (
        min(pd.Timestamp(entry.payload["as_of_session"], tz="UTC") for entry in records)
        if records
        else as_of
    )
    number = len(calendar(first, as_of))
    return {"number": number, "phase": cadence.phase_of(number), "first_session": str(first.date())}


def record_forecast(
    models: Models,
    *,
    loader: Loader,
    store_root: object,
    forecast_close: object,
    ledger_path: str | Path,
    created_at: pd.Timestamp,
    code: Mapping[str, object],
    calendar: window.Calendar,
    environment: Mapping[str, object],
) -> ledger.LedgerEntry:
    """Check the timing and registration, read the window, forecast, append one entry."""

    config = models.config
    rule = registration_record.checked_rule(models.bundle)
    as_of = window.session_stamp(forecast_close, "forecast_close")
    created = pd.Timestamp(created_at)
    if created.tzinfo is None:
        raise ForecastRefused("timing", "created_at must be timezone-aware")
    created = created.tz_convert("UTC")
    if as_of < config.h_end:
        raise ForecastRefused(
            "before_h_end",
            f"{as_of.date()} is before H_end {config.h_end.date()}: forecasts are prospective only",
        )
    times = timing(config, as_of)
    if created < times["close_available_at"]:
        raise ForecastRefused(
            "close_not_available",
            f"the {as_of.date()} close counts as available from {times['close_available_at']}",
        )
    if created >= times["record_deadline"]:
        raise ForecastRefused(
            "past_record_deadline",
            f"a forecast as of {as_of.date()} must be recorded before {times['record_deadline']}",
        )
    entries = ledger.read_ledger(ledger_path)
    found = registered(entries)
    registration_record.check_record(found.record, models)
    registration_record.check_code(found.record, code)
    checked_environment = job_environment.check(
        models.bundle.document.get("environment_manifest"), environment
    )
    if times["close_available_at"] <= found.registered_at:
        raise ForecastRefused(
            "before_registration",
            f"the {as_of.date()} close (available {times['close_available_at']}) is not after the "
            f"registration time {found.registered_at.isoformat()}: it cannot be session 1",
        )
    if created < pd.Timestamp(found.entry.recorded_at):
        raise ForecastRefused(
            "before_registration",
            f"created_at {created.isoformat()} precedes the registration entry "
            f"({found.entry.recorded_at}): nothing is recorded before the registration",
        )
    for identity in models.entries.values():
        if pd.Timestamp(identity["registered_at"]) > created:
            raise ForecastRefused("not_frozen", "the models were registered after this forecast")
    records = [entry for entry in entries if entry.kind in RECORD_KINDS]
    lookback = config.prospective.lookback_sessions  # daily returns: lookback + 1 closes
    result = loader(store_root, list(config.assets), as_of, lookback=lookback, retrieved_by=created)
    win = window.coerce_window(
        result,
        assets=config.assets,
        forecast_close=as_of,
        lookback=lookback,
        calendar=calendar,
        retrieved_by=created,
    )
    if not records:
        why = first_eligible_refusal(win, config.assets)
        if why is not None:
            raise ForecastRefused(
                "not_first_eligible_close",
                f"{as_of.date()} is not the first eligible close: {why}; nothing is recorded "
                "for it, and the next session is considered (section 4)",
            )
    rows, refusals = forecast_rows(models, win)
    bundle = models.bundle
    as_of_text = str(as_of.date())
    payload = {
        "record_type": "prospective_forecast",
        "forecast_schema_version": FORECAST_SCHEMA_VERSION,
        "forecast_id": sha256_json({"bundle": bundle.bundle_id, "as_of": as_of_text})[:16],
        "as_of_session": as_of_text,
        "created_at": created.isoformat(),
        "close_available_at": times["close_available_at"].isoformat(),
        "record_deadline": times["record_deadline"].isoformat(),
        "bundle": {
            "bundle_id": bundle.bundle_id,
            "manifest_sha256": bundle.manifest_sha256,
            "config_sha256": bundle.document["config_sha256"],
            "run_id": bundle.document["run_id"],
            "status": bundle.document["status"],
            "zero_volatility_rule": rule,
        },
        "registration": found.reference(),
        "session": session_of(records, as_of, calendar),
        # The bundle's manifest travels with the record (chained), so scoring can apply the
        # same check without the bundle.
        "environment": {
            **checked_environment,
            "manifest": models.bundle.document.get("environment_manifest"),
        },
        "data": {
            "data_class": bundle.document["data_class"],
            "window_fingerprint": win.fingerprint,
            "calendar": win.calendar,
            "checked_against": window.calendar_id(calendar),
            "retrieved_by": created.isoformat(),
            "lookback_returns": lookback,
            "closes": len(win.sessions),
            **win.lineage(),
            "first_session": str(win.sessions[0].date()),
            "training_end": bundle.document["training_data"]["training_end"],
        },
        "conventions": {
            "timezone": config.prospective.timezone,
            "close_available_local": config.prospective.close_available_local,
            "record_deadline_local": config.prospective.record_deadline_local,
            "close_available": (
                f"{config.prospective.close_available_local} {config.prospective.timezone} on "
                "the session (assumed descriptive convention; not proof of publication time)"
            ),
            "record_deadline": (
                f"{config.prospective.record_deadline_local} {config.prospective.timezone} on "
                "the next weekday after as_of"
            ),
            "target_window": "the h sessions after the entry session (entry_lag after as_of)",
        },
        "labels": list(config.labels),
        "code": dict(code),
        "rows": rows,
        "refusals": refusals,
    }
    guard = _guard(as_of_text, bundle.bundle_id, found.entry.entry_sha256, len(records))
    if not rows:  # every asset refused: a refusal, never a forecast
        refusal = {**payload, "record_type": "prospective_refusal"}
        refusal.pop("rows")
        return ledger.append_entry(
            ledger_path, "refusal", refusal, recorded_at=created, guard=guard
        )
    return ledger.append_entry(ledger_path, "forecast", payload, recorded_at=created, guard=guard)


# --- scoring ---


def row_key(entry_sha256: str, row: Mapping[str, object]) -> tuple:
    return (entry_sha256, row["asset"], int(row["horizon"]), row["target"])


def scored_keys(entries: Sequence[ledger.LedgerEntry]) -> set[tuple]:
    keys = set()
    for entry in entries:
        if entry.kind == "score":
            for score in entry.payload["scores"]:
                keys.add(
                    (
                        score["forecast_entry_sha256"],
                        score["asset"],
                        score["horizon"],
                        score["target"],
                    )
                )
    return keys


def pending_rows(entries: Sequence[ledger.LedgerEntry]) -> list[tuple[ledger.LedgerEntry, dict]]:
    done = scored_keys(entries)
    return [
        (entry, row)
        for entry in entries
        if entry.kind == "forecast"
        for row in entry.payload["rows"]
        if row["status"] == "ok" and row_key(entry.entry_sha256, row) not in done
    ]


def _outcome(
    win: window.InferenceWindow,
    asset: str,
    as_of,
    entry_lag: int,
    horizon: int,
    rule: str = zero_vol.DEFAULT,
    recorded_close: float | None = None,
):
    sessions = list(win.sessions)
    if as_of not in sessions:
        return None, "as_of_outside_window"
    position = sessions.index(as_of)
    exit_position = position + entry_lag + horizon
    if exit_position >= len(sessions):
        return None, "not_matured"
    returns = series.daily_log_returns(
        win.sessions, win.prices, win.distributions, win.splits, asset
    )
    g = returns["log_return"].to_numpy(np.float64)
    if not np.isfinite(g[position + entry_lag + 1 : exit_position + 1]).all():
        return None, "outcome_inputs_unavailable"
    # The as-of close the forecast recorded fixes B2's floor; the window's own close is
    # compared with it, so a revision is visible but never moves the floor.
    closes = zero_vol.as_of_closes(returns["date"], win.prices, asset)
    window_close = float(closes[position])
    if recorded_close is None and rule != zero_vol.CURRENT:
        raise ForecastRefused(
            "as_of_close_not_recorded", f"{asset}: the forecast holds no as-of close for the floor"
        )
    used_close = window_close if recorded_close is None else float(recorded_close)
    closes[position] = used_close
    # The training labels' arithmetic (series.labels_from_log_returns and
    # zero_vol.volatility_labels under the same rule), at the as-of row.
    labels = {
        "total_log_return": series.labels_from_log_returns(
            returns["date"],
            returns["log_return"],
            horizon=horizon,
            entry_lag=entry_lag,
            target="total_log_return",
        ),
        "log_realized_volatility": zero_vol.volatility_labels(
            returns["date"],
            returns["log_return"],
            closes,
            horizon=horizon,
            entry_lag=entry_lag,
            rule=rule,
        ),
    }
    values = {}
    for target, frame in labels.items():
        value = float(frame["target_return"].iloc[position])
        values[target] = value if np.isfinite(value) else None
    flags = labels["log_realized_volatility"].iloc[position]
    sigma_floor = float(flags["sigma_floor"])
    diagnostics = {
        "zero_vol_rule": rule,
        "sigma_raw": float(flags["sigma_raw"]),
        "measured_zero": bool(flags["measured_zero"]),
        "floored": bool(flags["floored"]),
        "sigma_floor": sigma_floor if np.isfinite(sigma_floor) else None,
        "as_of_close": used_close if np.isfinite(used_close) else None,
        "as_of_close_revised": (
            window_close != used_close
            if recorded_close is not None and np.isfinite(window_close)
            else None
        ),
    }
    flat = values["log_realized_volatility"] is None and diagnostics["measured_zero"]
    return {
        "zero_volatility": flat,
        "entry_session": sessions[position + entry_lag],
        "exit_session": sessions[exit_position],
        "volatility_diagnostics": diagnostics,
        **values,
    }, None


def score_matured(
    *,
    ledger_path: str | Path,
    loader: Loader,
    store_root: object,
    score_close: object,
    scored_at: pd.Timestamp,
    code: Mapping[str, object],
    calendar: window.Calendar,
    environment: Mapping[str, object],
    zero_vol_rule: str | None = None,
) -> tuple[ledger.LedgerEntry | None, list[dict[str, object]]]:
    """
    Score every matured, unscored forecast row (one "score" entry), and
    return the rows left pending with their reason (not_matured,
    close_not_available, outcome_inputs_unavailable, as_of_outside_window).

    The ledger must open with its registration (`not_registered`), and the
    running `code` must be the registered code (`registration_mismatch`).
    `zero_vol_rule` is the registered rule unless research code passes
    another; scoring refuses any rule other than the one every pending
    forecast's bundle was trained under (`zero_vol_rule_not_trained`), a
    ledger already scored under another rule (`zero_vol_rule_mismatch`), so
    two rules are never pooled, and any rule other than the registration's
    (`zero_vol_rule_mismatch`).
    """

    entries = ledger.read_ledger(ledger_path)
    found = registration_record.current(entries)
    if zero_vol_rule is None:
        zero_vol_rule = zero_vol.DEFAULT if found is None else found.rule
    zero_vol.check_rule(zero_vol_rule)
    _refuse_other_rule(entries, zero_vol_rule)
    pending = pending_rows(entries)
    if not pending:
        return None, []
    for entry in {e.entry_sha256: e for e, _ in pending}.values():
        # Forecasts recorded before the rule was recorded came from bundles trained under
        # the rule in force.
        trained = entry.payload["bundle"].get("zero_volatility_rule", zero_vol.CURRENT)
        if trained != zero_vol_rule:
            raise ForecastRefused(
                "zero_vol_rule_not_trained",
                f"entry {entry.sequence} comes from a bundle trained under {trained!r}; it is "
                f"never scored under {zero_vol_rule!r} (one treatment in training and scoring)",
            )
    if zero_vol_rule != found.rule:
        raise ForecastRefused(
            "zero_vol_rule_mismatch",
            f"the registration names {found.rule!r}; this ledger is never scored under "
            f"{zero_vol_rule!r}",
        )
    registration_record.check_code(found.record, code)
    # The bundle's environment: every forecast entry carries its manifest.
    checked_environment = _scoring_environment(entries, environment)
    for entry in {e.entry_sha256: e for e, _ in pending}.values():
        bundle = entry.payload["bundle"]
        if (bundle.get("manifest_sha256"), bundle.get("config_sha256")) != (
            found.record["bundle"]["manifest_sha256"],
            found.record["config_sha256"],
        ):
            raise ForecastRefused(
                "registration_mismatch",
                f"entry {entry.sequence} comes from another bundle or config than the "
                "registration's",
            )
    close = window.session_stamp(score_close, "score_close")
    stamp = pd.Timestamp(scored_at).tz_convert("UTC")
    earliest = min(pd.Timestamp(entry.payload["as_of_session"], tz="UTC") for entry, _ in pending)
    if earliest > close:
        raise ForecastRefused(
            "score_close", "the scoring close precedes every pending as-of session"
        )
    # The window runs from the earliest pending as-of session to the scoring close.
    lookback = len(calendar(earliest, close)) - 1
    if lookback < 1:
        return None, [
            {"forecast_entry_sha256": e.entry_sha256, "asset": r["asset"], "reason": "not_matured"}
            for e, r in pending
        ]
    assets = sorted({row["asset"] for _, row in pending})
    result = loader(store_root, assets, close, lookback=lookback, retrieved_by=stamp)
    win = window.coerce_window(
        result,
        assets=assets,
        forecast_close=close,
        lookback=lookback,
        calendar=calendar,
        retrieved_by=stamp,
    )
    scores, left = [], []
    for entry, row in pending:
        as_of = pd.Timestamp(entry.payload["as_of_session"], tz="UTC").as_unit("ns")
        outcome, reason = _outcome(
            win,
            row["asset"],
            as_of,
            int(row["entry_lag"]),
            int(row["horizon"]),
            zero_vol_rule,
            row.get("as_of_close"),
        )
        key = {
            "forecast_entry_sha256": entry.entry_sha256,
            "forecast_id": entry.payload["forecast_id"],
            **{k: entry.payload["bundle"][k] for k in ("bundle_id", "config_sha256")},
            "model_id": row["model_id"],
            "model_version": int(row["model_version"]),
            "as_of_session": entry.payload["as_of_session"],
            "asset": row["asset"],
            "horizon": int(row["horizon"]),
            "target": row["target"],
        }
        if outcome is None:
            left.append({**key, "reason": reason})
            continue
        conventions = entry.payload["conventions"]
        available = window.local_instant(
            outcome["exit_session"],
            conventions["close_available_local"],
            conventions["timezone"],
        )
        if stamp < available:
            left.append({**key, "reason": "close_not_available"})
            continue
        value = outcome[row["target"]]
        flags = (
            outcome["volatility_diagnostics"] if row["target"] == "log_realized_volatility" else {}
        )
        if value is None and outcome["zero_volatility"]:
            # Realized volatility 0: its log is -inf, below every band's lower end. It is
            # scored as outside the band, never left pending (that would bias coverage up).
            scores.append(
                {
                    **key,
                    "entry_session": str(outcome["entry_session"].date()),
                    "exit_session": str(outcome["exit_session"].date()),
                    "exit_close_available_at": available.isoformat(),
                    "outcome": None,
                    "outcome_note": "zero_realized_volatility",
                    "point": row["point"],
                    "band_low": row["band_low"],
                    "band_high": row["band_high"],
                    "null": row["null"],
                    "outcome_display_percent": 0.0,
                    "inside_band": False,
                    "error": None,
                    "null_error": None,
                    "squared_error": None,
                    "null_squared_error": None,
                    **flags,
                }
            )
            continue
        if value is None:
            left.append({**key, "reason": "outcome_inputs_unavailable"})
            continue
        scores.append(
            {
                **key,
                "entry_session": str(outcome["entry_session"].date()),
                "exit_session": str(outcome["exit_session"].date()),
                "exit_close_available_at": available.isoformat(),
                "outcome": value,
                "outcome_note": "floored_realized_volatility" if flags.get("floored") else None,
                "point": row["point"],
                "band_low": row["band_low"],
                "band_high": row["band_high"],
                "null": row["null"],
                "outcome_display_percent": display_percent(row["target"], value),
                "inside_band": bool(row["band_low"] <= value <= row["band_high"]),
                "error": value - row["point"],
                "null_error": value - row["null"],
                "squared_error": (value - row["point"]) ** 2,
                "null_squared_error": (value - row["null"]) ** 2,
                **flags,
            }
        )
    if not scores:
        return None, left

    def guard(current: tuple[ledger.LedgerEntry, ...]) -> None:
        clash = scored_keys(current) & {
            (s["forecast_entry_sha256"], s["asset"], s["horizon"], s["target"]) for s in scores
        }
        if clash:
            raise ForecastRefused("duplicate_score", f"{len(clash)} rows are already scored")

    earlier = [e.payload for e in entries if e.kind == "score"]
    cumulative = coverage_table([s for p in earlier for s in p["scores"]] + scores)
    payload = {
        "record_type": "prospective_score",
        "score_schema_version": SCORE_SCHEMA_VERSION,
        "scored_at": stamp.isoformat(),
        "score_close": str(close.date()),
        "outcome_window_fingerprint": win.fingerprint,
        "calendar": win.calendar,
        "checked_against": window.calendar_id(calendar),
        "retrieved_by": stamp.isoformat(),
        "outcome_data": win.lineage(),
        "code": dict(code),
        "registration": found.reference(),
        "environment": checked_environment,
        "scores": scores,
        "cumulative": cumulative.to_dict("records"),
    }
    written = ledger.append_entry(ledger_path, "score", payload, recorded_at=stamp, guard=guard)
    return written, left


def _scoring_environment(
    entries: Sequence[ledger.LedgerEntry], running: Mapping[str, object]
) -> dict[str, object]:
    """
    The running environment against the one the ledger's forecasts were made
    in: the bundle's manifest each forecast entry recorded (one manifest per
    ledger), checked with the forecast job's own rule.
    """

    manifests = {}
    for entry in entries:
        if entry.kind != "forecast":
            continue
        kept = entry.payload.get("environment")
        if not isinstance(kept, Mapping) or not isinstance(kept.get("manifest"), Mapping):
            raise ForecastRefused(
                "environment_unregistered",
                f"forecast entry {entry.sequence} records no environment manifest",
            )
        sha = job_environment.manifest_sha256(kept["manifest"])
        if sha != kept.get("registered_sha256"):
            raise ForecastRefused(
                "environment_invalid",
                f"forecast entry {entry.sequence}'s manifest does not match its recorded hash",
            )
        manifests[sha] = kept["manifest"]
    if len(manifests) > 1:
        raise ForecastRefused(
            "environment_mismatch", "the ledger's forecasts name more than one environment"
        )
    registered = next(iter(manifests.values()), None)
    return job_environment.check(registered, running)


def _refuse_other_rule(entries: Sequence[ledger.LedgerEntry], rule: str) -> None:
    """Refuse scoring under `rule` when earlier volatility scores used another rule."""

    for entry in entries:
        if entry.kind != "score":
            continue
        for score in entry.payload["scores"]:
            if score["target"] != "log_realized_volatility":
                continue
            # Scores written before the rule was recorded used the rule in force.
            used = score.get("zero_vol_rule", zero_vol.CURRENT)
            if used != rule:
                raise ForecastRefused(
                    "zero_vol_rule_mismatch",
                    f"entry {entry.sequence} scored volatility under {used!r}, not {rule!r}; "
                    "two zero-volatility rules are never pooled in one ledger",
                )


# --- the cumulative report ---

# One variant: a re-freeze (new config hash, bundle or model version) is never pooled.
VARIANT_COLUMNS = ("bundle_id", "config_sha256", "model_id", "model_version")
REPORT_COLUMNS = (
    *VARIANT_COLUMNS,
    "asset",
    "horizon",
    "target",
    "n_scored",
    "n_inside",
    "coverage",
    "coverage_low",
    "coverage_high",
    "mean_abs_error",
    "null_mean_abs_error",
    "caveat",
)
SMALL_N = 30


def wilson_interval(hits: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """The Wilson score interval for a proportion (treats the n records as independent)."""

    if n == 0:
        return math.nan, math.nan
    z = inf.normal_ppf(0.5 + level / 2.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def caveat(n: int, horizon: int) -> str:
    overlap = (
        f"Forecasts recorded less than {horizon} sessions apart share outcome sessions, so they "
        "are not independent and the interval is too narrow."
    )
    if n == 0:
        return "No matured forecast yet."
    if n == 1:
        return "Only 1 scored: far too few to judge calibration or skill."
    if n < SMALL_N:
        return f"Only {n} scored: far too few to judge calibration or skill. {overlap}"
    return f"{n} scored; still a short record. {overlap}"


def coverage_report(entries: Sequence[ledger.LedgerEntry]) -> pd.DataFrame:
    """Cumulative band coverage and errors per variant, asset, horizon and target."""

    return coverage_table([s for e in entries if e.kind == "score" for s in e.payload["scores"]])


def _mean_abs(values: pd.Series) -> float | None:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.abs().mean()) if len(finite) else None


def coverage_table(scores: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(scores))
    rows = []
    if not frame.empty:
        keys = [*VARIANT_COLUMNS, "asset", "horizon", "target"]
        for values, part in frame.groupby(keys, sort=True):
            variant = dict(zip(keys, values, strict=True))
            asset, horizon, target = variant["asset"], variant["horizon"], variant["target"]
            n, hits = len(part), int(part["inside_band"].sum())
            low, high = wilson_interval(hits, n)
            rows.append(
                {
                    **{k: variant[k] for k in VARIANT_COLUMNS},
                    "model_version": int(variant["model_version"]),
                    "asset": asset,
                    "horizon": int(horizon),
                    "target": target,
                    "n_scored": n,
                    "n_inside": hits,
                    "coverage": hits / n,
                    "coverage_low": low,
                    "coverage_high": high,
                    "mean_abs_error": _mean_abs(part["error"]),
                    "null_mean_abs_error": _mean_abs(part["null_error"]),
                    "caveat": caveat(n, int(horizon)),
                }
            )
    return pd.DataFrame(rows, columns=list(REPORT_COLUMNS))
