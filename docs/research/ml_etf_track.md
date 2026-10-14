# ETF return and risk track (synthetic)

Status (2026-10-14): implemented on `feature/ml-etf-track`; merge pending.

Code: `forecasting/etf_data.py` (series, features, targets) and `forecasting/etf.py` (the track). Configs: `configs/ml/etf_return_risk_{planted,null}.yaml`. Tests: `tests/test_etf_track.py`.

**Why a separate track.**
- The equity track asks a cross-sectional question: which names will do better than the others on the same date?
- An ETF allocation asks per asset: what return and what risk should be expected for this fund?
- The two questions need different targets, nulls and metrics. A single pooled model across six very different assets would assume one model fits all of them; this track does not.
- Real ETF data does not exist in the repository yet (the Tiingo audit is pending), so everything here is synthetic.

## Synthetic series

| Asset | Daily vol | GARCH (α, β) | Drift | Planted trend |
|---|---|---|---|---|
| EQ_US | 1.1% | 0.08, 0.90 | 3e-4 | yes |
| EQ_INTL | 1.2% | 0.07, 0.91 | 2e-4 | yes |
| BOND_AGG | 0.3% | 0.04, 0.94 | 1e-4 | no |
| BOND_SHORT | 0.1% | 0.03, 0.95 | 8e-5 | no |
| COMMODITY | 1.4% | 0.06, 0.92 | 0 | yes |
| REIT | 1.4% | 0.09, 0.89 | 2e-4 | no |

- **Returns.** r_t = μ_t + √h_t·e_t.
- **Volatility.** h_t follows GARCH(1,1) with the unconditional variance above (`volatility: garch`), or is constant (`constant`).
- **Trend.** For trend assets, μ_t = drift + s·σ·tanh(m_{t−1}), where m is the 60-session return to t−1 over σ√60: known before day t.
- **Generation.** Deterministic (legacy `RandomState`). Calendars end before 2025-01-01.
- **GARCH(1,1) only generates the series.** No model in this track is a GARCH model. The forecasters are per-asset ridge and the gradient-boosting challenger, on the HAR-style log-realized-volatility and trend features below.

## Features and targets

| Feature | Family | Contract |
|---|---|---|
| `tsmom_20d`, `tsmom_60d` | relative_technical | Trailing log return over (60-session daily vol × √k): trend against the asset's *own* risk. Not ranked across assets |
| `log_rv_5d`, `log_rv_20d`, `log_rv_60d` | risk_liquidity | log √(mean squared daily log return) over k sessions (HAR-style inputs) |

Every feature is `session_close` and backward-looking (prefix-stable, tested).

| Target | Definition | Known after |
|---|---|---|
| `forward_return` | close(exit)/close(entry) − 1 | The exit close |
| `log_realized_volatility` | log √(mean r_k² over entry < k ≤ exit) | The exit close |

Both targets are prefix-invariant (tested). Entry is `entry_lag` sessions after the row date, and exit `horizon` sessions later, as for every label.

## Protocol

For each target and each asset:
- **Folds.** Purged expanding folds on that asset's rows only, with the same outer windows for every asset (development mode, lockbox checked per asset).
- **Nulls.** `historical_mean` (the asset's expanding training-label mean, the Campbell-Thompson benchmark); `zero` for returns; `persistence` for volatility (the trailing 20-session log realized vol, a random-walk volatility forecast).
- **Models.**
  - `ols` (diagnostic; it nests the historical mean, because its intercept-only restriction *is* the historical mean);
  - `ridge` with T4's grid and nested purged selection, **chosen per asset**;
  - the gradient-boosting challenger (`ml_challenger.md`), also per asset.
- **Checks.** These are leakage checks:
  - `lockbox_closed` and `evaluation_asof_matches_folds` per asset and target;
  - `predictions_complete` per model;
  - `nested_selection_inside_training` per nested model.

  `selection_fits_converged` is an instrument check.

## Metrics

One value per date before inference. Pooled rows use the date's mean over assets. Fold-block t decides; Newey-West and the bootstrap are reported beside it.

| Metric | Models | Notes |
|---|---|---|
| `oos_r2_vs_<null>`, `mse_improvement_vs_<null>` | Every candidate and OLS against every null, pooled and per asset | Pooled OOS R² is SSE-weighted, so high-volatility assets dominate it. Read the per-asset rows |
| `qlike_improvement_vs_<null>` | Volatility target | QLIKE(σ², σ̂²) = σ²/σ̂² − log(σ²/σ̂²) − 1, robust to noise in the volatility proxy (Patton 2011) |
| `clark_west_vs_historical_mean` | **Only** `ols`, returns | The one truly nested pair |
| `oos_r2_vs_reference`, `mse_improvement_vs_reference` | The challenger against ridge and OLS | Its increment |
| `interval_coverage_<50/80/90>` | Returns, ridge | The interval is μ̂ ± z·σ̂·√h, with σ̂ = exp(volatility forecast). Its coverage is the **fold-block estimate**: the mean over outer folds of each fold's share of realized returns inside. The fold-block t tests it against the nominal level (Learning Notes). Every coverage number in this document is that estimate |

Calibration is computed only here, because only here is the forecast an interval. Rank IC is not used: ranking six heterogeneous assets is not this track's question.

**Forecast artifacts.**
- `etf.ret.ridge` is an `interval` artifact (80% intervals).
- `etf.vol.ridge` and the challenger's artifacts are `point` artifacts.
- They carry one row per (date, asset), with fold, training window and `available_at` (`ml_forecast_artifacts.md`).

## Results (measured, synthetic)

**Runs.**
- planted `20261014T193348Z-27927cb386b5` (trend, s = 0.1; GARCH) and null `20261014T193928Z-c9acd1358e35` (no trend; constant volatility);
- both from commit `c23343e`, with 2000 sessions, h = 20 and 23 outer folds per asset and target;
- every leakage check (108) and instrument check (24) passed in both;
- `git_dirty` reflects only the `.venv` symlink (`ml_pipeline.md`).

**Multiple testing.** 4 registered candidates (ridge and the challenger, for each target) and 20 grid points per run.

OOS R² of per-asset ridge against the historical mean (bootstrap 95% interval), in percent:

| Panel / target | Pooled | EQ_US | EQ_INTL | COMMODITY | BOND_AGG | BOND_SHORT | REIT |
|---|---|---|---|---|---|---|---|
| planted / volatility | **+12.4** [1.5, 21.5] | **+22.7** [6.4, 38.9] | +3.2 | +4.6 | +8.1 | −3.8 | +15.9 |
| planted / return | −0.7 [−2.7, 1.1] | +1.4 | −1.6 | −2.3 | −4.9 | −1.2 | −0.0 |
| null / volatility | −2.2 [−4.2, −0.4] | −1.2 | −1.0 | −5.8 | −2.5 | −1.2 | −2.3 |
| null / return | −1.6 [−4.6, 0.9] | −2.2 | −0.4 | −4.7 | −1.3 | −0.4 | +0.8 |

| Panel | Volatility, ridge vs historical mean: MSE t / QLIKE t | Volatility, ridge vs persistence: OOS R² (MSE t) | Return: Clark-West (ols vs mean), pooled t | 50 / 80 / 90% interval coverage, pooled |
|---|---|---|---|---|
| planted | 1.88 / 1.70 | +12.8% (2.78) | −0.29 | 0.457 / 0.753 / 0.858 |
| null | −2.29 / −2.13 | +49.9% (10.7) | −1.46 | 0.473 / 0.783 / 0.889 |

**Reading.**
- **Volatility is forecastable where it was planted.**
  - The pooled OOS R² against the historical mean is +12.4%, with a bootstrap interval excluding 0. The fold-block MSE t (1.88) is just short of 2 on 23 blocks.
  - Gains concentrate in the assets with strong clustering (EQ_US, REIT). BOND_SHORT, with almost no clustering (α = 0.03), gains nothing.
  - Ridge's chosen penalty differs per asset (modal c = 0.001 for EQ_US and COMMODITY, 1 for BOND_AGG, EQ_INTL and REIT, 100 for BOND_SHORT). That is the per-asset design working.
- **Returns are not forecastable here.**
  - The planted trend (s = 0.1) is too weak for 20-session returns over these samples. Ridge chose its strongest shrinkage (c = 100) for every asset, and no OOS R² interval excludes 0.
  - This matches the empirical literature: return predictability rarely survives out of sample.
- **Intervals** (fold-block coverage estimates).
  - On the null panel the plug-in intervals are close to nominal: 80% → 78.3%.
  - On the GARCH panel they under-cover: pooled 45.7% / 75.3% / 85.8% at 50 / 80 / 90%. The volatile assets cover 72-74% at 80%; the bonds about 80%.
  - **The under-coverage is expected from the method, not a surprise about the data.** The interval is a *plug-in*: it treats ridge's forecasts μ̂ and v̂ (log volatility) as if they were the truth.
    1. **Volatility-forecast error.** Where volatility moves (GARCH), v̂ misses log σ by an error e. The realized coverage is E[2Φ(z·e^{e}) − 1], which is below nominal for wide intervals: with sd(e) = 0.3 → 78.7%, and 0.5 → 76.8%, for a nominal 80%. The same error *raises* coverage at 50% (52.0% at sd 0.5).
    2. **Jensen bias.** exp(v̂) estimates the geometric centre of σ, which is smaller than the RMS volatility the interval needs: by a factor exp(sd²/2), so 13% too narrow at sd 0.5. This narrows every level, including 50%.
    3. **Return-forecast error.** μ̂ has no skill here (OOS R² < 0), so its error adds variance the interval omits.
  - Together these explain under-coverage at every level, and most where volatility clusters. On the null panel volatility is constant, so v̂ is nearly exact and coverage is close to nominal.
  - **A risk input from this track needs a predictive interval**: one that carries forecast uncertainty (a wider-tailed multiplier, or split-conformal calibration on development folds), calibrated per asset. It also explains the export's `warn` statuses for this run (`ml_registry.md`, "Health export").
- **Nulls matter.**
  - On the null panel ridge "beats" persistence by 50% R² (t 10.7), simply by shrinking a noisy trailing volatility toward the mean.
  - Against the historical mean, the right null under constant volatility, it is slightly *worse* (−2.2%, t −2.29): the estimation cost of fitting nothing.
- **The challenger.**
  - It was worse than per-asset ridge for returns on both panels (MSE t −2.7 and −3.5 vs ridge).
  - It gained on volatility (+10.6% vs the historical mean on the planted panel) but less than ridge (t −1.25).
  - No credible increment: with about 1,800 noisy observations per asset, the boosted trees overfit.

## Limitations

- Synthetic series only. GARCH and a tanh trend are convenient, not realistic; real fund returns have fat tails, jumps, regime changes and correlated shocks.
- The interval is a plug-in normal interval with variance h·σ̂². It ignores the error in σ̂ and μ̂, the Jensen gap between exp(v̂) and the RMS volatility, and fat tails, so under-coverage is built in wherever volatility is forecast with error (see Results). Coverage is measured, not assumed.
- `persistence` is a deliberately naive null. Beating it is necessary, not sufficient: under constant volatility, ridge beats persistence simply by shrinking toward the mean (null config). The historical mean is the null that matters for "is there predictable risk?".
- No permutation nulls yet (`permutation_draws: 0` is recorded). The null config is the null evidence.
- Fund-level evidence rules are still to be set "with the first fund-level forecast ticket" (PRD section 6). Until then, the registry's universe criterion keeps fund scopes at `engineering_only`.

## Follow-ups

- **A predictive interval that carries parameter and forecast uncertainty.** For example, split-conformal calibration of the standardized residual |y − μ̂| / (σ̂√h) on development folds, applied per asset, or a t multiplier fitted on training residuals. Then Kupiec and Christoffersen coverage tests per asset.
- **Permutation nulls** per asset, with tuning inside every draw.
- **Fund-level evidence rules** (owner decision).

## Learning Notes

1. **Problem.** Forecast each fund's return and risk with models chosen per fund, and judge each forecast against the right null.
2. **Concepts.**
   - **Volatility clustering (GARCH).** Variance today depends on yesterday's surprise and variance.
   - **HAR-style forecasts.** Average realized volatility over several windows.
   - **Campbell-Thompson OOS R².** Against the expanding historical mean.
   - **Loss functions robust to a noisy volatility proxy.** QLIKE and MSE on log vol.
   - **Interval calibration.**
3. **Assumptions.** Stationarity inside each training window; normal h-day returns for the interval; the realized-volatility label as an unbiased proxy for the true variance.
4. **Formulas.**
   - OOS R² = 1 − Σ(y − ŷ)² / Σ(y − ȳ_train)².
   - Clark-West f = 2(y − b)(ŷ − b) for the nested pair.
   - QLIKE as above.
   - Interval: μ̂ ± Φ⁻¹(0.5 + c/2)·exp(v̂)·√h.
   - Plug-in coverage when log σ is forecast with error e ~ N(0, s²): E_e[2Φ(z·e^{e}) − 1]. For z = 1.2816 (80%): s = 0.3 gives 78.7%, s = 0.5 gives 76.8%.
   - Fold-block t test of coverage = c:
     - p_b is fold b's share inside, and B the number of folds;
     - t = (mean_b p_b − c) / (sd_b(p_b)/√B), with df = B − 1.
     - Adjacent folds share overlapping 20-day labels at their edges, so the blocks are not fully independent and p is slightly optimistic. T2's simulation found the fold-block t closest to nominal size among the three methods.
5. **Example.**
   - Daily σ̂ = 1%, h = 20 and c = 80% give a half-width of 1.2816 × 1% × √20 = 5.7%.
   - Around a forecast of +0.5%, the interval is [−5.2%, +6.2%].
6. **Leakage risks guarded.**
   - Volatility labels that include the entry day's own return (they start after entry; tested).
   - Trailing features reaching the next close (prefix tests).
   - Per-asset tuning on the test window (nested inner folds per asset).
   - Pooling assets' training rows across the purge boundary (each asset has its own folds; the windows are shared).
7. **Ways results could mislead.**
   - Pooled R² dominated by the most volatile assets.
   - Beating a naive persistence null.
   - Interval coverage that is right on average but wrong in high-volatility regimes; check coverage per asset.
   - A plug-in interval read as a predictive one: it under-covers by construction wherever its inputs are forecast with error.
   - Synthetic GARCH flatters every volatility model.
8. **Practitioner validation.**
   - Diebold-Mariano on QLIKE (Patton 2011).
   - Kupiec and Christoffersen coverage tests for intervals.
   - Campbell and Thompson (2008) for return predictability against the historical mean.
   - Rolling versus expanding windows for regime change.
9. **What to learn next.** HAR-RV models (Corsi 2009), conditional coverage tests (Christoffersen 1998), and why return predictability, unlike volatility predictability, rarely survives out of sample (Welch and Goyal 2008).
