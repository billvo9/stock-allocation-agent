# Prospective ETF forecasts (VTI, AGG, SHV)

Status (2026-10-15): implemented and tested end to end on synthetic fixtures, through PR #73's real prospective store writer and `load_inference_window`, on `claude/prospective-etf-forecasts-9e5b22`; merge pending. No real data has been read: the Tiingo audit and collection run in the owner's terminal, and storing needs the owner's approval.

Code: `src/stock_agent/forecasting/prospective/`. Config: `configs/prospective/etf_prospective_v1.yaml`. Job: `scripts/forecast_prospective.py`. Panel: the Lab's Experiments page (`app/experiments.py`, `app/forecast_records.py`). Tests: `tests/test_prospective_*.py`, `tests/test_app_prospective.py`, `tests/test_forecast_prospective_script.py`.

**Forecasts are research records, never recommendations.** Every record carries the label "unvalidated, no demonstrated skill".

## What the owner decided (2026-10-14)

Recorded verbatim in `docs/PROJECT_STATE.md`, "prospective forecasting track":
- Register ETF forecasters for VTI, AGG and SHV at 5, 20 and 30 sessions, frozen before the first forecast.
- Record each forecast once, timestamped and labelled, and score it only as real outcomes arrive.
- Forecast volatility and total return, each with an 80% band, with the return forecast beside the historical-average null.
- H_end = 2026-10-14. Training, tuning and evaluation use data up to 2024-12-31 only. The last 63 sessions before a forecast date are inputs to frozen models only.

## The pipeline

| Step | Command | Reads | Writes (once) |
|---|---|---|---|
| Freeze | `forecast_prospective.py freeze --research-store <F1 store>` | The F1 research store (sessions to 2024-12-31) | `reports/ml/prospective/`: the experiment run, the frozen bundle, six registry entries |
| Forecast | `forecast_prospective.py forecast --bundle <bundle> --store-root <store> --forecast-close <date>` | The bundle and registry; the prospective store through `load_inference_window(..., retrieved_by=created_at)` | One `forecast` entry in the ledger (or a `refusal` when every asset's window is incomplete) |
| Score | `forecast_prospective.py score --store-root <store> --score-close <date>` | The ledger; the prospective store through the same loader | One `score` entry for the rows that matured |
| Report | `forecast_prospective.py report` | The ledger | Nothing |

**Owner's steps, once the data steps are done** (each in the owner's terminal; nothing here contacts a network):

1. Store the approved pre-2025 batch in the F1 research store.
2. Freeze once: `freeze --research-store <root>`. Expect six entries at `engineering_only`.
3. Once the prospective store holds the latest complete session, record the forecast as of that session. Run it between 18:00 New York that day and 09:30 New York the next weekday.
4. Keep a copy of the printed ledger head hash outside the ledger (removing the last line is undetectable otherwise). `report --expect-head <hash>` refuses a ledger that no longer holds it.
5. Later, `score --score-close <latest session>` and `report`.

The ledger defaults to `data/raw/prospective_forecasts/ledger.jsonl`. It is private and ignored by Git, and it is a sibling of any `data/raw/prospective/` store root, so a fail-closed store reader never sees it.

## Data and timing

- **Training data.** Only the F1 research store (`data/funds/reader.read_store`, the fail-closed reader). Any price, distribution or split date after 2024-12-31 is refused with a `LeakageError` before anything is computed; the frames, folds and the final refit check again.
- **Data class.** It comes from a closed map of F1 adapters: the synthetic adapters give `synthetic`, and Tiingo gives `engineering_sandbox`. An unknown or mixed source is refused. Three hand-picked existing funds form a hindsight-selected list, not a registered research panel, so real-data entries stay `engineering_only` (see "Registry").
- **Vintages.** Two unadjusted closes for one fund and session (two retrievals) are refused. Choosing among vintages needs an approved rule (F1, OD9).
- **Prospective inputs.** The forecast job reads the prospective store only through PR #73's loader, `load_inference_window(store_root, assets, forecast_close, lookback=63, *, retrieved_by)`.
  - `scripts/forecast_prospective.py` is its one reviewed consumer. It imports statically only the loader and the rules calendar (`nyse_rules.sessions`, `CALENDAR_ID`), and `tests/test_prospective_boundaries.py` checks that. The research modules never import the prospective side.
  - `retrieved_by` is the record's decision time (`created_at`; the scoring time when scoring), so only batches retrieved by then are read.
  - **`lookback_sessions: 63` is the registered meaning of 63 daily returns:** 64 closes ending at the forecast close. Nothing had been frozen when this meaning was fixed.

  `window.coerce_window` checks the result:
  - exactly lookback + 1 sessions ending at the close;
  - those sessions equal the rules-based NYSE calendar, checked independently, so a session the store skipped cannot pass for a one-session return or shift a horizon; the calendar id is recorded;
  - unadjusted closes and as-paid cash;
  - no row dated after the close and no batch retrieved after `retrieved_by` (a `LeakageError` otherwise);
  - the window's `batches`, `missing` closes, `revisions` and `complete` flag are carried into the record.
- **Incomplete windows are refused, never filled.**
  - An asset with any close missing in its window is not forecast. It is listed in the record's `refusals` with the reason `incomplete_inference_window` and the sessions without a close.
  - When every asset is incomplete, a `refusal` entry is recorded instead of a forecast.
  - A missing close is never replaced by the training median.
- **Timing conventions (assumed, recorded in every forecast).**
  - A close counts as available from 18:00 America/New_York on its session. This is the OD9 descriptive convention, not proof of vendor publication time.
  - The as-of session must be on or after H_end (2026-10-14); earlier dates would be backdated records in the sealed period.
  - `created_at` must be at or after the close's availability.
  - `created_at` must be before 09:30 America/New_York on the next weekday. No session can open earlier, since a holiday only pushes the next session later, so a record made in time cannot have seen any later session. The rule needs no calendar beyond 2024.

## Targets, features and models

Daily log total return, from the approved F5.1 total returns (`portfolio.descriptive.total_returns`): g_t = log(1 + r_t), with r_t = (f_t·P_t + D_t)/P_{t−1} − 1 on raw closes. A multi-session or unavailable return is NaN; it is never spread or filled.

| | Definition (row t, entry e = t + 1, exit x = e + h) |
|---|---|
| `total_log_return` | Σ_{k=e+1..x} g_k |
| `log_realized_volatility` | ½ log(mean_{k=e+1..x} g_k²) |
| `log_rv_5d`, `_20d`, `_60d` | ½ log(mean of g² over the last k sessions) |
| `tsmom_20d`, `tsmom_60d` | Σ of the last k g over (σ_60 √k) |

- The features are the ETF track's (`forecasting/etf_data.py`), computed on total returns. Tests show they equal the ETF track's price features when there is no cash.
- Each window is summed on its own, so a feature depends, bit for bit, only on its last 60 returns. A 63-session window gives exactly the features of the full history.
- A row is labeled only when both targets are, so the return and volatility folds are identical.
- The scored outcome uses the training labels' own arithmetic.
- **Models, per asset, horizon and target:**
  - ridge with T4's grid (c ∈ {0.001, …, 100}, α = c·n), with nested purged selection inside every outer fold;
  - the historical-mean null, the asset's expanding training-label mean. For volatility the label is log volatility, so the null shown in volatility units is a geometric mean, exp(mean log volatility), below the arithmetic mean of volatility.
- **Folds.** Purged expanding folds, the same 63-session outer windows for every asset, from 2010-01-04; inner design 4 × 63 sessions after 252.
- **The frozen refit.** One ridge fit per asset, horizon and target on every training row. Its penalty is chosen by the same nested selection on the last four inner windows before the lockbox. The null's frozen value is the mean of all training labels.
- **No pickle.** A bundle stores the fitted numbers (medians, means, scales, coefficients, intercept). A reloaded fit predicts with exactly `LinearForecast.predict`'s arithmetic (tested bit-equal).

## The 80% band (pre-registered: `split_conformal_signed_v1`)

**Why not the plug-in.** The ETF track's interval μ̂ ± z·exp(v̂)·√h treats both forecasts as exact. It under-covers wherever volatility is forecast with error: 80% nominal gave about 75% (`ml_etf_track.md`).

**The method.** Split-conformal quantiles of signed standardized out-of-fold residuals, per asset, horizon and target:

| Target | Score u | Band |
|---|---|---|
| Return | (y − μ̂) / (exp(v̂)·√h) | μ̂ + exp(v̂)·√h · [u_lo, u_hi] |
| Volatility | y − v̂ (log scale) | v̂ + [u_lo, u_hi] |

Here u_lo = u_(⌊(n+1)·0.1⌋) and u_hi = u_(⌈(n+1)·0.9⌉) are the finite-sample conformal ranks. The frozen band uses every development score, all with labels ending before 2025. Nothing from the sealed period enters.

**Alternative considered.** A predictive interval carrying parameter uncertainty (a t multiplier from training residuals). It would correct only one of the three causes of the plug-in's narrowness (parameter error). The conformal band takes its width from realized errors, so volatility-forecast error, the Jensen gap and fat tails are all inside it. It is also distribution-free and simple to audit.

**Development coverage, measured out of sample.**
- Fold b is scored with quantiles from the scores of earlier folds whose labels ended before fold b's test start (a purge). Folds with fewer than 100 earlier scores are not scored.
- Coverage is reported per asset, horizon and target as the fold-block estimate, with its fold-block standard error.
- The plug-in's coverage on the same rows is reported beside it, for the return target.
- Each forecast row records its development coverage. That is the coverage of these fold-by-fold bands, each built from earlier folds only, not of the frozen band, which uses every development score.

**Assumptions.** Split conformal's finite-sample guarantee needs exchangeable scores from a model fitted without the calibration rows. Neither holds exactly here:
- overlapping h-session labels and volatility regimes break exchangeability;
- the frozen quantiles come from the fold models' out-of-fold scores but are applied to a final refit trained on those same labels.

So 80% is a target, not a guarantee. The sequential development estimate has the same structure, which is why it is the evidence that counts. Coverage is measured, never assumed.

## Registry

- Freezing registers six versions, one per target and horizon, each holding the fits of all three assets:
  - model ids `etf_prospective_v1.{ret,vol}.h{5,20,30}.ridge`;
  - artifacts: the experiment run and the frozen bundle, each with its SHA-256.
- Each entry records the data fingerprints and batch ids, the code state and the config hash. Its config carries `prospective.status = "prospective record pending"`.
- The stage is computed from evidence, never set. The stage rules are unchanged.
  - Synthetic data and the `engineering_sandbox` class block `development_evaluated`.
  - So do a universe that is not survivorship-free and zero permutation draws.
  - The higher stages' rules are unapproved, so nothing can reach them.
- The forecast job refuses any entry whose computed stage is above `development_evaluated`. It also refuses a bundle that its entries do not register by hash, and models registered after the forecast's `created_at`.
- **Multiple testing.** Any change to the config is a new hash and a new variant. Re-freezing registers new versions in the same research scopes, and `prior_variants_in_scope` counts them.

## The ledger

- **Format.** One canonical JSON line per entry: `sequence`, `kind` (`forecast` or `score`), `recorded_at`, `payload`, `prev_entry_sha256` and `entry_sha256`, the SHA-256 of every other field.
- **What is refused on read:**
  - an edited, inserted, removed or reordered line;
  - a non-canonical line;
  - an incomplete last line.
- **Appends.** Each append takes an exclusive lock, re-verifies the whole chain, runs its duplicate check inside the lock, writes one line and fsyncs it.
- **Write-once rules.**
  - A second forecast for the same as-of session and bundle is refused (`duplicate_forecast`).
  - A row is scored once.
  - Forecast entries are never edited; a correction would be a new entry.
- **Integrity, not provenance.** The chain cannot detect a truncated tail or a consistent rewrite of the whole file: removing the last line leaves a valid chain.
  - The job prints the head hash after every append, and the owner keeps a copy outside the ledger.
  - `report --expect-head <hash>` (`ledger.check_head`) refuses a ledger that no longer holds the kept hash. That is what makes truncation detectable.

## A forecast record

Per asset × horizon × target:
- the point forecast, the 10% and 90% band ends, and the historical-mean null;
- the model id, version and entry hash, and the computed stage;
- the development coverage and the band method and level;
- `display_percent` values for the panel: log return × 100, or daily volatility in percent.

Per record:
- `as_of_session`, `created_at`, `close_available_at` and `record_deadline`;
- the bundle id, manifest hash, config hash and status;
- the window fingerprint, the calendar id, `retrieved_by`, 63 returns from 64 closes, and the window's `batches`, `missing`, `revisions` and `complete` flag;
- any refused assets (`refusals`), the code state, the conventions and the fixed labels.

## Scoring and the cumulative report

- **When a row is scored.** Once its exit session (entry + h) is in the store, as read with `retrieved_by` = the scoring time, and that close is available (18:00 New York). The outcome window runs from the earliest pending as-of session to the scoring close, and its lineage is recorded. It is recorded with its outcome, whether the outcome is inside the band, and the model's and the null's errors.
- **When a row waits.** A row whose inputs are incomplete (a missing close in its window) is left pending with its reason, never scored as missing.
- **Zero realized volatility.** If every return in a volatility outcome's window is 0 (possible for SHV, whose close moves in $0.01 steps), the log volatility is minus infinity, below every band. It is scored as outside the band, with no error, and never left pending: leaving such outcomes out would bias coverage upward.
  - Every scored volatility row also records the zero-volatility rule, `sigma_raw`, `measured_zero`, `floored` and `sigma_floor` (score schema 1.1), and a ledger is never scored under two rules.
  - The research-only floor B2 and its development check are in `docs/research/zero_vol_floor.md`. The rule in force is unchanged until the owner decides.
- **The cumulative report.** Each scoring run stores it:
  - per variant (bundle, config hash, model id and version), asset, horizon and target: n, hits, coverage and a Wilson 95% interval. A re-freeze is never pooled with earlier forecasts;
  - mean absolute errors of the model and the null;
  - a caveat in words.
- **Caveats.**
  - With fewer than 30 scores the report says it is far too few to judge calibration or skill.
  - Forecasts recorded less than h sessions apart share outcome sessions. The Wilson interval treats them as independent, so it is too narrow.

## The Lab panel

The Experiments page shows "Prospective forecasts (unvalidated)":
- **No ledger:** "No forecast recorded yet".
- **Each forecast:** the point, band and historical average, the development coverage and evidence stage, the as-of session, the labels, and a "Synthetic test data" badge when it applies.
- **Outcomes:** "Not yet known" until scored.
- **Broken ledger:** a ledger that fails its integrity check shows only that fact.

The app re-verifies the chain with its own reader, because it may not import the forecasting package; a test shows the two readers agree. The panel computes nothing, and every state is scanned for advice wording. The Lab-wide sentence "It makes no forecast" became: "Its only forecasts are the unvalidated research records on the Experiments page, which have no demonstrated skill." The Evidence page's "no model output" and "nothing is a forecast" statements were updated the same way.

## Results (measured, synthetic only)

Both runs ended with every leakage and instrument check passing and every entry `engineering_only`.

- **Test variant** (2019-2024, 126-session folds from 2021, about 9 s). Development band coverage at 80% ranged from 56% to 83% across assets, horizons and targets, on 7 scored folds each.
- **Registered config** (the full 2003-2024 geometry on synthetic data, with AGG and SHV starting at their real inception dates, 2003-09-29 and 2007-01-11). About 140 s; 60 outer folds per asset, horizon and target; 126 leakage and 36 instrument checks passed. Development coverage of the 80% band (fold-block estimate, se about 0.01-0.03):

  | Asset | Return h5 | h20 | h30 | Volatility h5 | h20 | h30 |
  |---|---|---|---|---|---|---|
  | VTI | 82.2% (plug-in 74.1%) | 81.8% (76.4%) | 82.6% (76.4%) | 80.4% | 78.0% | 78.1% |
  | AGG | 79.4% (74.3%) | 77.8% (80.7%) | 76.6% (82.3%) | 80.1% | 78.0% | 75.2% |
  | SHV | 79.9% (77.4%) | 78.1% (77.4%) | 78.9% (77.7%) | 78.4% | 77.5% | 78.2% |

  The conformal band stays within a few points of 80% in every cell. The plug-in under-covers the clustering asset (VTI, 74-76%) and over-covers AGG at long horizons: its error depends on the asset, which is why one plug-in rule cannot be calibrated for all three.

These synthetic numbers verify the pipeline. They are not evidence about the funds.

## Limitations

- No real data yet: the end-to-end tests run on a synthetic prospective store written and read by PR #73's real modules.
- The bands' nominal level is not guaranteed: overlapping labels and regime changes break exchangeability. Development coverage varies by asset and horizon and is reported, not tuned.
- The dev residuals come from expanding-window fits on less data than the frozen fit, so the frozen band may be slightly wide.
- A complete window can still leave a feature undefined (no price movement, or a total return the approved formulas cannot give, such as cash on a split date without its share basis). Such a feature is imputed with the training median, as in training, and listed per row (`imputed_features`). A missing close is never imputed: that asset is refused.
- **Flat windows in training.** A development row whose h returns are all 0 has no log volatility. Both of its labels are dropped, which biases development coverage upward, and a flat 5-session feature window is imputed as typical volatility.
  - Each fit records `zero_volatility_rows_dropped` and `training.zero_volatility_rule`; the count is 0 on the GARCH fixture.
  - For real SHV data in 2010-2015 it may not be small. `scripts/check_zero_vol_floor.py check` measures it on the stored development batch before any freeze (`docs/research/zero_vol_floor.md`, section 6).
- No permutation nulls, and no Kupiec or Christoffersen tests yet.
- Selective recording (forecasting only on some days) cannot be prevented, only seen: the ledger lists every as-of session recorded.
- The ledger shows edits but not truncation; the head hash must be kept elsewhere.
- The 18:00 availability and 09:30 deadline are conventions, not verified publication times.

## Follow-ups

- Kupiec (unconditional) and Christoffersen (independence) coverage tests per asset, once enough non-overlapping scores exist.
- Diebold-Mariano or Clark-West comparisons of the model and the null on the prospective record, after a pre-registered burn-in.
- A pre-registered forecasting cadence, so missed sessions are visible as gaps: proposed in `docs/research/prospective_registration.md` (every session, 18 forecasts), awaiting approval.
- Fund-level evidence rules (owner decision) before any fund entry can reach `development_evaluated`.
- **Owner decision before freezing on real data: flat windows.** Options considered:
  - keep dropping them, with the count reported;
  - floor realized volatility at a price tick;
  - count flat development outcomes as misses.

  The proposal is rule B2: half a tick at the as-of close, divided by √h, in training labels, calibration and scoring alike. It is implemented as a non-default research option with a development check whose acceptance criteria are fixed in advance (`docs/research/zero_vol_floor.md`). This is a target-formula choice, so it waits for approval.
- **F1 pay dates (existing open owner question).** A late-December 2024 ex-date paid in January 2025 refuses its whole F1 batch today, and training would refuse it too. The F1 decision (lock on ex-date and declaration date, or stay fail-closed) applies here unchanged.

## Learning Notes

1. **Problem.** Make forecasts that can be judged honestly later: frozen before the first forecast, recorded once, scored only by outcomes that did not exist when they were made.
2. **HAR volatility.** Corsi's (2009) heterogeneous autoregressive model regresses future realized volatility on its average over the last day, week and month: log RV_{t+h} ≈ β_0 + β_d log RV_5 + β_w log RV_20 + β_m log RV_60. It captures volatility clustering with three numbers and no GARCH likelihood. Short windows react fast; long windows anchor to the regime. Ridge shrinks the betas toward zero, which pulls the forecast toward the mean.
   - Example: last week's daily volatility 1.5%, last month's 1.0%, last quarter's 0.9%, and a long-run level of 1%. With β ≈ (0.3, 0.4, 0.2) and the intercept that returns 1% at the long-run level, the log forecast is 0.3·log 1.5 + 0.4·log 1.0 + 0.2·log 0.9 ≈ 0.10, so about 1.1%: above the long-run level, well below last week's spike.
3. **Why 5-30-session return skill is usually near zero.** The standard deviation of a 20-session equity return is about 1% × √20 ≈ 4.5%. A plausible predictable part is a few basis points a day, so R² is at most a few tenths of a percent. With about 15 years of non-overlapping 20-session periods (about 190), the standard error of an out-of-sample R² is larger than any credible true value. Welch and Goyal (2008) found that most return predictors lose to the historical mean out of sample. That is why the null is shown beside every return forecast, and why ridge shrinks toward it.
4. **Interval calibration.** An interval is calibrated when its nominal level matches the long-run share of outcomes inside it.
   - A plug-in interval treats estimated inputs as truth. If log σ is forecast with error e ~ N(0, s²), the 80% plug-in covers E[2Φ(1.2816·e^{e}) − 1]: 78.7% at s = 0.3 and 76.8% at s = 0.5.
   - Conformal prediction uses the empirical quantiles of past errors instead. With exchangeable scores, P(y ∈ band) ≥ 80% in finite samples. Time-series dependence weakens that to an approximation, which is why coverage is measured sequentially (calibrate on the past, score the next fold) and reported per asset and horizon.
5. **How a prospective record is scored.**
   - Coverage: the share of outcomes inside the band, with an interval. The Wilson interval assumes independent records; overlapping windows make the effective n smaller, so read it as too narrow.
   - Skill: the model's errors against the null's on the same outcomes. A credible difference needs many non-overlapping outcomes and a pre-registered test.
   - At one forecast every 20 sessions, a year gives about 12 independent 20-session outcomes per asset. That cannot distinguish 80% coverage from 70%: the Wilson interval for 10 of 12 is about [55%, 95%]. Prospective evidence accumulates slowly by design.
6. **Leakage risks guarded.**
   - Sealed-period sessions in training: refused on read, in the frames and in the final fit.
   - Data after the forecast close in the window: refused.
   - Backdated or late records: the H_end and deadline rules.
   - Band tuning on the sealed period: the quantiles come from development scores only.
   - A second, better-looking forecast for the same session: refused.
   - Edited records: the hash chain.
7. **Ways results could mislead.**
   - Development coverage near 80% does not mean prospective coverage will be: regimes change.
   - A few lucky early scores mean nothing.
   - Overlapping windows inflate apparent agreement.
   - The volatility model can look skilful against the historical mean simply because volatility clusters. That is real but modest, and against persistence it may vanish.
8. **Practitioner validation.** Kupiec's unconditional coverage test and Christoffersen's conditional coverage test on non-overlapping outcomes; Diebold-Mariano on QLIKE for volatility (Patton 2011); Campbell-Thompson OOS R² for returns; a pre-registered evaluation date and burn-in.
9. **What to learn next.** Split and adaptive conformal prediction for time series (Gibbs and Candès 2021, "Adaptive conformal inference under distribution shift"); HAR-RV extensions with jumps; why proper scoring rules (the pinball loss for quantiles) reward calibrated bands.

**Exercise for the owner (bounded).** With the test fixture (`tests/forecast_prospective_helpers.py`), change `band.min_calibration` to 200 and re-freeze. Count how many development folds lose their sequential band, and explain why the reported coverage becomes noisier for SHV than for VTI.
