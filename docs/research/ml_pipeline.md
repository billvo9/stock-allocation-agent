# ML experiment runner

Status (2026-10-14): implemented on `feature/ml-experiment-runner` (stacked on the feature-contract PR); merge pending.

Code:
- `forecasting/config.py`, `forecasting/experiment.py`, `forecasting/metrics.py`, `forecasting/provenance.py`;
- `scripts/run_experiment.py`; `configs/ml/`.

Tests: `tests/test_experiment_runner.py`, `tests/test_forecasting_metrics.py`.

The runner turns one YAML config into one reproducible, versioned experiment. It builds on the T1-T4 framework rather than replacing it:
- every model runs through `model_diagnostics.runner.run_diagnostics` (unchanged);
- so every model gets T2's purged folds, lockbox gate, nulls, controls, canary, metrics, inference and checks;
- T4's nested purged selection chooses the linear candidates' penalties.

## Running it

```bash
.venv/bin/python scripts/run_experiment.py configs/ml/equity_xs_planted_linear.yaml
```

- Outputs go under `reports/ml/` (ignored by Git); `--output-root` overrides it.
- Only synthetic data sources exist. A config naming any other source is refused: real-data experiments need a registered research panel (RES-1), an approved availability rule (OD9) and the G1 record of every real-panel scoring (OD12).
- Synthetic calendars must end before 2025-01-01.

## Configs

A config (`configs/ml/*.yaml`, schema 1.0) defines the whole experiment:

| Section | Contents |
|---|---|
| `data` | `source` (`synthetic_equity_panel` for the equity track, `synthetic_etf_panel` for the ETF track; see `ml_etf_track.md`) and its `params`, including the data seed |
| `universe` | `id` and `min_history_sessions` (frozen eligible universe) |
| `features` | `families` (contracted; see `ml_features.md`) |
| `label` | `horizon`, `entry_lag` (≥ 1: same-close labels are refused), `benchmark` (`equal_weight` or null) |
| `folds` | Outer `start` and `block_sessions`; the inner design (block, windows, minimum training sessions) |
| `models` | Linear `candidates` (subset of ridge, lasso, elastic_net), `include_ols`, `challengers` (`hist_gradient_boosting`; see `ml_challenger.md`), `increment_reference` |
| `evaluation` | Base seed, permutation draws and models, bootstrap and calibration replicates, stale lags |
| `registry` | `model_prefix` for registry ids |

**Validation and hashing.**
- **Strict parsing.** Unknown keys are errors, so a typo cannot fall back to a default.
- **Config hash.** `config_sha256` is the SHA-256 of the canonical JSON of the parsed config, independent of key order and YAML formatting (tested).
- **Configs shipped:**
  - `equity_xs_planted_linear` (a planted linear quality signal);
  - `equity_xs_null` (no signal, same design).

## What a run records

| Requirement | Where |
|---|---|
| Code SHA, dirty flag (and a diff hash when dirty) | `record.code`, ledger line |
| Config hash | `record.config_sha256`, every registry entry, every forecast artifact |
| Data fingerprints | `record.data.fingerprints`: prices, fundamentals, universe, features, labeled frame |
| Seeds | `record.seeds`: data seed, evaluation seed, the seed derivation rule |
| Label spec | `record.label`: label id, spec, availability rule, benchmark members |
| Fold geometry | `record.folds`: parameters, fold count, fold-table SHA-256, inner design; the per-fold table is in the T2 run |
| Lockbox evidence | `record.lockbox_evidence` (from T2) |
| Checks | `record.checks_summary` (from T2) |
| Nulls compared | `record.nulls` |
| Multiple testing | `record.multiple_testing`: candidates and grid points in the run; `prior_variants_in_scope`, the number of earlier registrations in the same research scope. Every earlier registration counts, even of the same variant: a rerun on new data or a new seed is another look |

### Outputs

Every output is written once, through a staging directory and a rename:

```text
reports/ml/
  runs/<run_id>/          T2 contract run (readable by model_diagnostics.artifacts.read_run)
  experiments/<run_id>/   record.json, config.json, metrics.parquet, missingness.parquet, manifest.json
  forecasts/<id>/         one out-of-fold forecast artifact per candidate (ml_forecast_artifacts.md)
  registry/               one new version per candidate (ml_registry.md)
  ledger.jsonl            one line per attempt, failures included
```

The ETF track writes no T2 contract run (it is not a cross-sectional panel). Its experiment directory holds `predictions`, `metrics`, `selection`, `checks` and `missingness` tables instead (`ml_etf_track.md`).

**Every run registers its candidates.**
- Each candidate is registered as `<model_prefix>.<model>`. Its parent is the previous version, and its research scope is `<track>:<data class>:<target>:h<horizon>:<universe id>`.
- Its evidence is derived from the experiment record by `registry.derive_evidence`.
- So every variant tried is counted (`prior_variants_in_scope`) and every run can be compared or rolled back.
- Synthetic runs stay `engineering_only` (blockers `data_class_engineering_only`, and `nulls_insufficient` below 199 draws).

**Determinism.**
- The same config and code give the same `outputs.results_sha256` under a new run id (tested). It covers the T2 results, the experiment tables and the forecast values (everything but the registry version).
- Seeds come from the config and the T2 seed rule. BLAS runs single-threaded during linear fits.

## Experiment metrics (`metrics.parquet`)

These are rows added beside T2's metrics. They use T2's columns plus `reference`; no T2 statistic changes.

| Metric | Definition | Inference |
|---|---|---|
| `rank_ic_coverage` | Share of scored dates with a defined rank IC | Point |
| `coverage_weighted_rank_ic` | Mean rank IC with undefined dates counted as 0 (the PRD rung L1 estimand) | Fold-block t (decides), Newey-West, block bootstrap |
| `rank_ic_difference` | Per-date paired difference, model − reference, both zero-filled | As above |
| `oos_r2_vs_reference` | 1 − SSE_model / SSE_reference on identical rows | Block bootstrap percentile interval |
| `mse_improvement_vs_reference` | Per-date MSE(reference) − MSE(model) | Fold-block t, Newey-West, bootstrap |
| `holm_adjusted_p` | Holm step-down adjustment of the candidates' one-sided fold-block p for coverage-weighted rank IC > 0 | Information only |
| `interval_calibration` | Not computed for point forecasts (`point_forecast`); interval forecasts get `interval_coverage_<level>` | Per-date coverage against the nominal level |

- **T2's own metrics stay unchanged.** These include rank IC, OOS R² against each registered null (`oos_r2_vs_zero`, `oos_r2_vs_per_symbol_mean`), MSE improvement, Clark-West for the registered nested pair (ols, zero), permutation nulls, the per-symbol-mean selection control, the minimum detectable IC and the simulated test sizes.
- **Clark-West** stays limited to truly nested pairs. Penalized and selected models are compared by MSE improvement and OOS R², never by Clark-West.

## Planted-signal and null results (measured, synthetic)

The full shipped configs: 30 synthetic names, 1500 sessions (2012-01 to 2017-09), h = 20, outer folds from 2014-07, 19 permutation draws.

These are **engineering checks of the pipeline on synthetic data, not evidence about markets.**

Runs `20261014T183807Z-9e1249871d2d` (planted) and `20261014T184120Z-4ca40e0dfe14` (null):
- 14 outer folds each, run from this branch's working tree before commit (recorded `git_dirty = true`; see Provenance below for the clean-tree reruns);
- fold-block t (df 13); OOS R² against the zero null (bootstrap);
- permutation p from 19 draws (resolution 0.05).

| Run | Model | Coverage-weighted rank IC (t) | Coverage | MDE | OOS R² vs zero | MSE improvement vs zero, t | Permutation p | Holm p |
|---|---|---|---|---|---|---|---|---|
| planted | ridge | 0.082 (2.97) | 1.00 | 0.083 | +0.28% | 1.18 | 0.05 | 0.016 |
| planted | lasso | 0.066 (2.31) | 0.91 | 0.100 | +0.37% | 1.22 | not run | 0.024 |
| planted | elastic_net | 0.072 (2.56) | 0.92 | 0.091 | +0.39% | 1.50 | not run | 0.024 |
| planted | ols (diagnostic) | 0.082 (3.14) | 1.00 | 0.079 | +0.24% | 0.72 | not run | n/a |
| null | ridge | 0.021 (1.16) | 1.00 | 0.056 | −0.06% | −0.89 | 0.20 | 0.33 |
| null | lasso | 0.021 (1.28) | 0.69 | 0.071 | −0.03% | −0.18 | not run | 0.33 |
| null | elastic_net | 0.021 (1.28) | 0.69 | 0.071 | −0.02% | −0.17 | not run | 0.33 |
| null | ols (diagnostic) | 0.019 (1.26) | 1.00 | 0.045 | −0.88% | −2.22 | not run | n/a |

**Provenance.**
- Both configs were re-run from commit `8585adf`: planted run `20261014T185004Z-2a6b19d242b5`, null run `20261014T185312Z-0197282e2832`. They reproduced the pre-commit runs' `results_sha256` exactly (`84e7d017…` and `8f9f66da…`).
- **Why `git_dirty` is true.** The records say `git_dirty = true` only because the session worktree holds an untracked `.venv` symlink, which `.gitignore`'s `.venv/` pattern does not match (open item 7).
- **Proof.** The recorded `git_diff_sha256` (`01217fdf…`) is the SHA-256 of an empty diff plus the untracked path `.venv`, so no source file differed from the commit.

**Planted run.**
- Every candidate recovers the planted ordering (Holm p ≤ 0.024). Clark-West for the nested pair (ols, zero) gives t 3.12.
- The ranking signal sits right at the minimum detectable effect, so a slightly weaker signal or fewer folds would be missed.
- The squared-error gains are not significant (t ≤ 1.5). See the Learning Notes on why rank IC and OOS R² can disagree.

**Null run.**
- Nothing is detected, and every leakage and instrument check passes.
- Lasso and Elastic Net choose the all-zero model on 31% of dates. Their conditional rank IC (0.029) is larger than the zero-filled one (0.021).
- Unpenalized OLS on 8 noise features is significantly *worse* than predicting zero (t −2.22): estimation noise costs squared error. That is what the penalties and nested selection guard against.

**Fast tests** (`tests/test_experiment_runner.py`; 16 names, h = 10; the planted strength s = 1e-3):
- planted: ridge coverage-weighted rank IC > 0.1 with fold-block t > 3, OOS R² vs zero > 0, Holm p < 0.01;
- null: every linear model |t| < 3, OOS R² vs zero < 1%, Holm p > 0.05, every leakage check passes.

## Finding: the T2 canary check fires on persistent expected returns

- **What happened.** On the fast-test planted panel (16 names, s = 1e-3, h = 10), T2's leakage check `canary_shows_no_skill_through_harness` (the memorizer through the harness must have |rank IC t| < 3) fired, with memorizer t ≈ 3.7. On the full planted config it stayed below 3. On the null panels it holds.
- **Why.** The memorizer predicts a row's label with the same symbol's latest *matured* training label. That is legitimate, purged information. It predicts the next label whenever expected returns persist:
  - the planted drift is constant within a quarter, because the current filing is constant;
  - even with an independent characteristic each quarter (AR coefficient 0), the memorizer kept t ≈ 3.7.
- **No leak.** The purged reference (`canary_detects_unpurged_leakage`) and every other leakage check behave as designed.
- **Implication for real data.** A forecastable signal is persistent by construction. On a real universe with genuinely persistent effects (momentum, quality), this leakage-severity check could fire without any leak, and it would block `development_evaluated`.
- **Owner follow-up (T2 contract).** Read the canary relative to a no-leak reference, for example a memorizer restricted to labels older than one block. Or move the check to `research_warning` and keep leakage detection in the unpurged positive control and the null run.
- **In this PR.** The planted test asserts that this check is the only leakage check that fails. The null test asserts that all leakage checks pass.

## Limitations

- Synthetic data only. The equity sandbox (four hindsight-selected names) is not used, and nothing is tuned on it.
- **The data class caps every entry at `engineering_only`.** On synthetic data, OOS R², rank IC and their t-statistics describe the generator, not any market.
- `prior_variants_in_scope` counts registered variants in `reports/ml`, which Git ignores. Durable counting belongs in G1 (OD12).
- The fast tests use few folds (df ≈ 7). Their thresholds have wide margins and fixed seeds, so they are deterministic, but they are not size or power estimates.
- `scripts/run_null_diagnostics.py` keeps its own provenance helpers. `forecasting/provenance.py` duplicates about 30 lines (follow-up).

## Learning Notes

1. **Problem.** Make every experiment reproducible from a file. Make every result comparable with the nulls on identical rows. Make every variant countable.
2. **Concepts.**
   - **Nested purged CV.** Inner folds inside each outer training window choose hyperparameters, and the outer folds estimate performance. Both purge labels that overlap the next test window.
   - **Coverage.** A model that is sometimes constant has a rank IC that is conditional on being active. The zero-filled mean is the honest unconditional number.
3. **Assumptions.** Fold blocks are exchangeable (fold-block t). The planted signal is linear in a point-in-time characteristic. Seeds and single-threaded BLAS make runs bit-reproducible on one platform.
4. **Formulas.**
   - Coverage-weighted IC = (1/T) Σ_t IC_t · 1{defined_t}.
   - Holm: sort p_(1) ≤ … ≤ p_(m); adjusted p_(k) = max_{j≤k} min(1, (m − j + 1) p_(j)).
5. **Example.**
   - Lasso is active on 13 of 24 folds with IC 0.0705. Its coverage-weighted IC is 13/24 × 0.0705 = 0.038 (the T4 example).
   - Holm on p = (0.01, 0.03, 0.04) gives (0.03, 0.06, 0.06).
6. **Leakage risks.**
   - Tuning on the outer test window (prevented by nested selection, as in T4).
   - Labels overlapping the test (prevented by the purge).
   - Seeds or configs edited between runs without a trace (the config hash and ledger).
   - Re-running until something looks good: `prior_variants_in_scope` makes that visible.
7. **Ways results could mislead.**
   - Conditional ICs on different date subsets.
   - A canary that fires on persistence (above).
   - Multiple configs tried on one synthetic seed.
   - Synthetic results read as market evidence.
8. **Practitioner validation.** MLflow-style tracking with immutable run ids, a registry with lineage, and pre-registered evaluation plans. White's Reality Check and Hansen's SPA when many variants compete.
9. **OOS R² vs rank IC.**
   - Rank IC measures ordering within a date. OOS R² also needs the forecast's *scale* and its date-level mean to be right.
   - With a rank IC near 0.08, the best achievable cross-sectional R² is roughly IC² ≈ 0.6%. Shrinkage (needed to avoid the OLS noise cost) pulls the forecast toward zero and leaves +0.3%.
   - Squared-error tests on 20-day returns are therefore far less powerful than rank tests. Read a significant rank IC with an insignificant MSE gain as "ordering, not magnitude".
10. **Next to learn.** The difference between Holm (family-wise error) and Benjamini-Hochberg (false discovery rate), and when each fits a research ledger.
