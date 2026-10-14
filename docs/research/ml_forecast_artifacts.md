# Out-of-fold forecast artifacts

Status (2026-10-14): contract 1.0 implemented on `feature/ml-registry-forecast-contracts`; merge pending.

Code: `src/stock_agent/forecasting/artifacts.py`. Tests: `tests/test_forecast_artifacts.py`.

This is the interface the RL track consumes for "PPO with forecast inputs". It is written so a consumer cannot use a forecast before it existed.

## What an artifact is

- **Scope.** One artifact holds one registered model version's out-of-fold forecasts of one target at one horizon, with **exactly one row per (date, asset)**.
- **Producer.** The experiment runner writes artifacts for candidate models. Nulls, controls and canaries get none.
- **Registry link.** An artifact is referenced by a registry entry through the SHA-256 of its `manifest.json`.

```text
reports/ml/forecasts/<model_id>-v<version:04d>-<content_sha256[:12]>/
  forecasts.parquet   the table below
  metadata.json       ForecastMetadata
  manifest.json       schema version and every file's SHA-256 (written last)
```

Write-once: the files are written to a staging directory and renamed into place, and an existing artifact is never overwritten. `read_forecast_artifact` verifies:
- every file hash;
- the schema version;
- the directory name;
- every invariant below.

## Table (schema 1.0)

| Column | Type | Meaning |
|---|---|---|
| `date` | datetime UTC, 00:00 | Session date of the features the forecast used |
| `asset` | string | Asset identifier (the panel's `symbol`) |
| `target` | string | What is forecast, e.g. `forward_return`, `excess_return`, `log_realized_volatility` |
| `horizon` | int | Label horizon in sessions |
| `prediction` | float, finite | The point forecast |
| `interval_low`, `interval_high`, `interval_level` | float | A central prediction interval, for `interval` artifacts only; NaN otherwise |
| `fold_id` | int | Outer fold that produced the row |
| `train_start`, `train_end` | datetime UTC | First and last training-row dates of that fold |
| `train_label_end_max` | datetime UTC | Latest label end among that fold's training rows |
| `test_start`, `test_end_exclusive` | datetime UTC | The fold's test window |
| `model_id`, `model_version` | string, int | The registry entry |
| `available_at` | datetime UTC | Earliest instant a consumer may use the row |

**Metadata** (`ForecastMetadata`) records:
- the model id, version and variant id, the run id and the track;
- the target, horizon and label id;
- `forecast_kind` (`point` or `interval`);
- `data_class` and `knowledge_rule`;
- the config SHA-256, the code state and the lockbox start.

Every row must agree with the metadata.

## Invariants (all enforced on write and read)

- **Out-of-fold.** `train_start <= train_end < test_start <= date < test_end_exclusive` and `train_label_end_max < test_start`. Each fold carries one test window, and windows are disjoint and ordered by `fold_id`.
- **One row per (date, asset).** Dates are session dates at 00:00 UTC.
- **Lockbox.** Every `date` and `available_at` precede 2025-01-01. `lockbox_start` may not be later than the owner's lockbox.
- **Availability.** `available_at >= knowledge_time(date)`. A forecast derives from its row's features (known at the date's knowledge time) and from a model fitted on labels that ended before `test_start`. So it is available no earlier than its inputs: "every derived value is available no earlier than the latest availability timestamp of every input" (AGENTS.md).
- **Kind.** `point` artifacts leave the interval columns NaN. `interval` artifacts need finite bounds with `low <= high` and a level in (0, 1).

## Availability rule

- **The only rule today** is `synthetic_end_of_utc_date`: a synthetic session date d becomes known at **d 23:59:59.999999Z**.
  - This is the owner's 2026-10-13 decision for synthetic fixtures only.
  - So a forecast dated d has `available_at = d 23:59:59.999999Z`.
- **Real data is refused.** A metadata object with any other data class raises, because no production availability rule is approved. OD9's `signal_cutoff_at` and open item 1 are pending.
- **This is not a trading-time claim.** The rule says when a synthetic value is known. It says nothing about when an order could be executed.

## Consuming forecasts (RL track)

```python
from stock_agent.forecasting.artifacts import read_forecast_artifact, forecasts_asof

artifact = read_forecast_artifact(path)        # verified, or ForecastArtifactError
usable = forecasts_asof(artifact.frame, decision_time)
```

- **As-of selection.** `forecasts_asof` returns, per asset, the latest row with `available_at <= decision_time`: an as-of selection, never the latest row overall. A decision just before d 23:59:59.999999Z cannot see d's forecast (tested).
- **Never align on `date` alone.** Join on `available_at <= decision_time`, so a change of knowledge rule cannot silently shift information forward.
- **Only out-of-fold rows exist.** Every forecast was produced by a model that saw only labels that matured before the forecast's test window. Its hyperparameters were chosen by nested purged validation inside that training window.
- **Leakage still possible at the next level.**
  - A policy trained on forecasts dated [a, b] and evaluated on [c, d] must still use c > b, plus a gap of at least the forecast horizon plus the entry lag. A forecast's label window overlaps the policy's reward window.
  - The forecasting design itself (feature families, grids, model classes) was chosen by a researcher who saw the whole development period. That design-level selection is not removed by out-of-fold construction. Count it as a variant (multiple testing).
- **Forecast kinds.** Use `interval` artifacts only where a risk input needs a range. Interval calibration is reported in the experiment run, not assumed.

## Versioning

- `FORECAST_SCHEMA_VERSION = "1.0"`. A reader refuses any other version.
- Additive changes will bump the minor version, and readers will accept earlier minors, as the T2 output contract does.
