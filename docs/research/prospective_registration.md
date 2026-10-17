# Prospective registration rules for the ETF forecasters (proposed)

Status (2026-10-17): proposed, for the owner's approval. **Nothing in this document is active, and nothing activates without the owner's explicit approval.**
- Until the owner approves sections 1–11 here, no real model is frozen or registered, no forecast is recorded and no confirmatory test is run.
- The same holds for each separate decision these rules depend on: storing the pre-2025 batch, the zero-volatility rule (`docs/research/zero_vol_floor.md`) and OD16 activation.
- Logging forecasts is never, by itself, validation.

Revised 2026-10-17 after an independent review, before any real data existed:
- the primary volatility loss is now an explicit owner decision (section 7);
- the trial-void and integrity rules are made exact (sections 5, 6 and 11);
- a checkpoint sequence rule is added (section 11);
- the power figures use the exact t-test (section 10);
- the recording deadline is flagged for confirmation (section 2).

Finalized from the lead session's draft packet of 2026-10-17. The schedule and power numbers in section 10 are computed by `src/stock_agent/forecasting/prospective/registration.py`. They are printed by `.venv/bin/python scripts/check_zero_vol_floor.py registration`, and a test checks that this document holds that output.

Scope of one registration:
- **Funds and horizons.** VTI, AGG and SHV, at h = 5, 20 and 30 sessions.
- **Targets.** Log realized volatility and total log return, each with the registered 80% split-conformal band (`split_conformal_signed_v1`) and the historical-mean null.
- **Configuration.** The registered config `etf_prospective_v1`, with H_end = 2026-10-14 and a 63-return (64-close) input window, training on F1 research-store data to 2024-12-31 only.

The owner's principles (2026-10-17, verbatim), which these rules make exact:
- "Register and freeze the model before its first real forecast."
- "Begin recording no earlier than the first eligible close after registration and verified data availability."
- "Never backdate or overwrite forecasts."
- "Treat an initial 20-session trial as a prospectively registered exclusion, not a period that can later be selectively included."
- "Use a preregistered test family and Holm correction where appropriate."
- "Account for overlapping 5-, 20- and 30-session outcomes."
- "Do not claim prospective validation merely because forecasts have been logged."
- "Keep the forecast-ledger head hash privately backed up outside the ledger. Do not commit it publicly yet."

## 1. Registration

**What is registered:**
- the frozen bundle's manifest SHA-256 and its six registry entries (model ids, versions and entry hashes);
- the config hash and the code commit;
- the zero-volatility rule;
- the commit of this document.

**What is never part of it:**
- vendor data;
- the ledger head hash (section 6).

**How it is registered:**
1. A registration pull request adds one record of those hashes. The owner merges it to `main`.
2. **The registration time is GitHub's merge time of that pull request.** It is a third-party timestamp; a local commit date can be set to anything.
3. The ledger's first entry is a registration entry that points to the registration commit and the bundle. That entry kind does not exist yet and must be built before activation (section 12).

**What counts as a new registration.** Any change to the code that produces or scores forecasts, the config, the training-data fingerprint, the band method or the zero-volatility rule.
- The earlier registration's record closes at the last session it recorded.
- It is reported on its own and never pooled with the new one.
- The new registration has its own trial.

## 2. Observation timing

- **Close availability.** A session's close counts as available from 18:00 America/New_York on that session. This is the OD9 descriptive convention, recorded in every forecast. It is not proof of the vendor's publication time.
- **Data visible to a record.** The prospective store is read with `retrieved_by` = the record's `created_at`, so only batches retrieved by the decision time are seen. Each window must hold exactly the 64 rules-calendar sessions ending at the as-of close, with nothing after it.
- **Recording window for as-of session d.** From 18:00 New York on d to 09:30 New York on the next weekday after d. The forecast job enforces this, and the registered config fixes it (`record_deadline_local: "09:30"`).
  - On a normal day the deadline is the next session's opening.
  - When the next weekday is an exchange holiday, the deadline is earlier than the next session's opening (stricter).
  - A record made in time can never have seen a later session.
  - **For the owner to confirm.** The draft packet said "09:30 New York on the next session". This document keeps the stricter rule the job already enforces, "09:30 on the next weekday". The two differ only before an exchange holiday, when the stricter rule closes the window earlier, so a forecast for the session before a holiday weekday must be recorded that evening or before 09:30 on the holiday. Keeping it needs no code change; switching to "the next session" would change the registered config, making it a new variant before any freeze (section 13, item 7).
- **The as-of session must be on or after H_end.** Earlier dates would be backdated records in the sealed period.
- **Outcomes.** A forecast's outcome is known at 18:00 New York on its exit session (entry + h, entry = as-of + 1). It is scored once, at any later scoring run, from the prospective store read with `retrieved_by` = the scoring time.

## 3. Forecast cadence

- **One record every NYSE session**, with as-of = that session's close: 3 funds × 3 horizons × 2 targets = **18 forecasts per session**. Each has a point, an 80% band and the null. No session is skipped by choice, so selective recording shows up as a gap.
- **Assets refused individually.** A fund whose 64-close window is incomplete is refused (`incomplete_inference_window`) and listed in the record. The other funds are forecast.
- **When every fund is refused**, a `refusal` entry is recorded. The session counts as recorded, with no forecast rows.
- **Missed sessions are never backfilled.** They are listed as missing (Model Health `forecast.ledger_sessions`, which expects every session) and count against C3.
- **Never overwritten.** The ledger is write-once and hash-chained, and a second forecast for the same session and bundle is refused (`duplicate_forecast`).

## 4. The first eligible close

Session 1 is the first NYSE session d for which all three hold:
1. 18:00 New York on d is after the registration time (section 1).
2. The prospective store holds a verified batch with the 64 closes ending at d for every fund, retrieved before the record, and the owner's spot checks of that batch passed.
3. The forecast is recorded inside d's recording window (section 2).

If any condition fails, d is not session 1: nothing is recorded for d and the next session is considered. From session 1 on, every session is expected.

## 5. The 20-session trial: a registered exclusion

- **Sessions 1–20 are the trial.** Their forecasts are recorded and scored for operations only: the pipeline, timing, data and Model Health.
- **They are permanently excluded from every confirmatory analysis**, whatever their results, and can never be included later.
- **The trial is the only burn-in.** The model is frozen, so no estimation burn-in exists. The confirmatory phase starts at session 21.
- **Any of these during the trial voids the registration:**
  - **A chain break:** the ledger fails its integrity check.
  - **A failed kept-head check:** the kept hash is no longer in the ledger (section 6). A degraded check is not a void if it is resolved before the next recording deadline; if it is not, it counts as failed.
  - **A refused or late session record.** For an expected session, either no `forecast` or `refusal` entry is recorded inside its recording window, or the job refuses the session as a whole. Whole-session refusals are timing, models, stage, a duplicate, or every fund refused, which writes a `refusal` entry.
    - A single fund refused for an incomplete window (`incomplete_inference_window`) is not a refused recording. It counts toward the next rule.
  - **A data gap of more than 2 consecutive sessions:** any fund without forecast rows for more than 2 consecutive expected sessions, whether from single-fund refusals, whole-session refusals or missed sessions.

  The model must then be re-registered (a new registration), and the trial restarts.

## 6. The private head-hash backup

The ledger's hash chain cannot show a removed last line. The kept head can, and it is kept privately:
1. **Two files**, outside the ledger folder and outside every repository, in a directory of mode 700 with files of mode 600, for example `~/stock-agent-private/ledger_head/`:
   - `ledger_head.txt`: its first line is the latest head hash. Model Health's `--ledger-head-backup` reads exactly that: the first line must be the 64-hex hash.
   - `ledger_head_history.log`: append-only lines of UTC time, as-of session, entry count and head hash.
2. **After every append** (forecast, refusal or score), write the printed head as the first line of `ledger_head.txt` and append one line to the history.
3. **Daily, and after each recording session, copy both files to a second medium** that only the owner controls and keeps independently of this computer, for example an external disk or an encrypted folder in the owner's private cloud storage.
   - Never a public location, and never the public repository ("Do not commit it publicly yet").
   - The owner chooses the medium when approving this section.
4. **Model Health compares the kept head with the ledger:**
   - equal is healthy;
   - an earlier entry's hash is degraded: the chain is intact, but the copy is stale. Copy the new head before the next recording deadline. A degraded state still present at that deadline counts as failed for the trial (section 5) and for C3 (section 11);
   - a hash no longer in the ledger is failed (lines were removed or the file was rewritten).
5. **Optional, only on the owner's separate approval**, because it is a third-party call: an RFC 3161 timestamp of the history file.

## 7. Metrics and the frozen nulls

- **Volatility (primary): the loss awaits the owner's decision** (the box below). Either way, y is the realized log volatility under the approved zero-volatility rule, f is the forecast, and d = L(y, null) − L(y, model).
  - Both candidate losses need a finite y. Under the rule in force a zero outcome makes y = −∞, so neither can be activated before the zero-volatility decision.

> **Owner decision (pending): the primary volatility loss.**
>
> **The issue.**
> - The forecast f is a ridge fit of log realized volatility under squared error, so it estimates E[log σ | x], and the null is a mean of log σ.
> - The draft's QLIKE, L = exp(2(y − f)) − 2(y − f) − 1 (Patton's σ²/σ̂² − ln(σ²/σ̂²) − 1 with σ̂² = exp(2f)), is consistent for E[σ² | x], not for E[log σ | x].
> - With y ~ N(μ, s²), E[σ²] = exp(2μ + 2s²), so exp(2f) sits below it by the factor exp(2s²), and QLIKE's best log forecast is f* = μ + s², not μ.
> - QLIKE is also asymmetric: it punishes a forecast that is too low more than one that is too high.
> - So d_QLIKE mixes skill with the level (Jensen) gap. A model can win C1 by sitting higher than the null rather than by forecasting better.
>
> **(a) Recommended: squared error on log σ.** L = (y − f)², so d = (y − null)² − (y − f)².
> - It is consistent with what is fitted and with the null, it is symmetric in log units, and it needs no extra estimate.
> - QLIKE on exp(2f) is reported beside it as a sensitivity, never confirmatory.
>
> **(b) QLIKE on a documented variance conversion.** The model's variance forecast is exp(2f + 2s²_model) and the null's is exp(2·null + 2s²_null).
> - The s² values are frozen per fund × horizon at registration: the model's out-of-fold development residual variance, and the variance of the null's training labels.
> - This is the standard loss for variance forecasts, but it adds two frozen estimates and relies on lognormal errors. A misestimated s² shifts the level and can still decide C1.
>
> **Worked example.** Take y ~ N(μ, s²) and a forecast f = μ + b. Then E[QLIKE] = exp(2s² − 2b) + 2b − 1 and E[SE] = s² + b². Let the null have s = 0.45 and b = 0 (it is the unconditional mean).
>
> | Forecast | E[QLIKE] | d_QLIKE vs null | E[SE] | d_SE vs null |
> |---|---|---|---|---|
> | The null (s = 0.45, b = 0) | 0.499 | — | 0.203 | — |
> | A skilful model, unbiased (s = 0.30, b = 0) | 0.197 | +0.302 | 0.090 | +0.113 |
> | No skill, moved up 0.2 log units (s = 0.45, b = 0.2) | 0.405 | **+0.094** | 0.243 | **−0.040** |
>
> The last row has no information beyond the null. QLIKE still scores it as better, because a shift of +0.2 brings it close to QLIKE's best level μ + s² = μ + 0.20. Squared error scores it as worse, as it should.
>
> Even the skilful model's QLIKE advantage mixes two things. Its 0.302 splits into:
> - 2(s²_null − s²_model) = 0.225, the information (what QLIKE would show if both forecasts sat at their best levels);
> - 0.077 from the two forecasts' different Jensen gaps, exp(2s²) − 1 − 2s², which comes from levels.
>
> Under (a) the skilful model's advantage is +0.113 (0.090 against 0.203), all of it information.
>
> The recommendation is (a). This document's status stays pending until the owner chooses. Sections 9–11 read "the primary loss" as whichever is approved.

- **Return (secondary): squared error** of the total log return, d = (y − null)² − (y − model)².
- **Calibration (secondary):** a hit or miss of each 80% band, per target.
- **Frozen training-window nulls.** Each null is the mean of all training labels of its fund, horizon and target, frozen in the bundle (`null.value`). It is never re-estimated or updated with prospective data.

## 8. Non-overlapping confirmatory outcomes, with a HAC sensitivity

- **Confirmatory outcomes**, per fund and horizon h: the forecasts recorded at sessions 21, 21 + h, 21 + 2h, and so on. Their outcome windows never overlap, and the phase is fixed now, so the subsample cannot be chosen later.
- **A missed phase session** is a missing outcome. It is not replaced by a neighbouring session, and it counts against C3.
- **Sensitivity (descriptive only, never confirmatory):** all daily post-trial forecasts, with a Newey–West (HAC) standard error at lag h − 1 (Bartlett kernel), since daily h-session outcomes overlap by up to h − 1 sessions. It is reported beside each confirmatory result.

## 9. Hypotheses, families and Holm

**The three families:**
- **Primary family P (volatility skill).** For each tested fund × horizon cell: H0: E[d] ≤ 0 against H1: E[d] > 0 (the model beats the null), with d from the approved primary loss (section 7). A one-sided t-test on the non-overlapping d.
- **Secondary family A (return skill).** The same structure on d_SE. The prior is that return skill is close to zero.
- **Secondary family B (calibration).** For each tested cell and each target: H0: coverage = 0.80, tested with an exact two-sided binomial test (equal tails). Rejection means the band is miscalibrated.

**Which cells are tested.** The family at each checkpoint is fixed now by the registered outcome counts, never by the realized data: it is every cell with at least 20 registered outcomes there (section 10).
- At 12 months: the three h = 5 cells.
- At 24 months: the six h = 5 and h = 20 cells.
- Family B has two tests per cell.
- A family member whose realized count falls below 20 (missed sessions) enters Holm with p = 1. It is never dropped.

**Holm and the level.** Each family uses Holm's step-down procedure at α = 0.025 per checkpoint. Splitting α = 0.05 equally between the two checkpoints (Bonferroni across them) keeps each family's chance of any false confirmation over both checkpoints at or below 0.05.

**Cells outside a checkpoint's family** (h = 30 at both checkpoints, h = 20 at 12 months) are reported descriptively as "insufficient evidence", never as a pass or a fail. Results at 30 sessions would need a later checkpoint, approved before it is reached.

**Alternative considered** (the draft packet): one 9-test family at every checkpoint, with untestable cells entered at p = 1. It has the same error control but less power, because the first Holm step is 0.025/9 instead of 0.025/3 at 12 months. The family above is equally prespecified, because the counts follow from the calendar and the cadence alone.

## 10. Checkpoints and honest power

**When the confirmatory analyses run.** Once each, at two prespecified closes. Views between them are descriptive and labelled "not a test".
- **C12:** post-trial session 252 (session 272).
- **C24:** post-trial session 504 (session 524).

**Registered outcomes per cell.** A forecast at session s matures at s + 1 + h, so a cell's count is the number of phase sessions s with s + 1 + h at or before the checkpoint.

The schedule below gives:
- `n_*`, the registered non-overlapping outcomes per cell;
- `tested_*`, whether the cell is in that checkpoint's family;
- `mde_*`, the minimum detectable standardized effect E[d]/s_d. That is the per-outcome mean loss improvement, in standard deviations of d, which a one-sided t-test detects with 80% power at the first Holm step (α/m; m = 3 at C12 and 6 at C24). It is computed with the exact noncentral t distribution, n − 1 degrees of freedom. The normal approximation would give 0.458, 0.348 and 0.696, slightly optimistic at these sample sizes.

| horizon | n_12_months | tested_12_months | mde_12_months | n_24_months | tested_24_months | mde_24_months |
|---|---|---|---|---|---|---|
| 5 | 50 | yes | 0.471 | 100 | yes | 0.354 |
| 20 | 12 | no | n/a | 25 | yes | 0.751 |
| 30 | 8 | no | n/a | 16 | no | n/a |

| checkpoint | horizon | n | alpha | accepted_low | accepted_high |
|---|---|---|---|---|---|
| 12_months | 5 | 50 | 0.00417 | 0.62 | 0.94 |
| 24_months | 5 | 100 | 0.00208 | 0.67 | 0.91 |
| 24_months | 20 | 25 | 0.00208 | 0.52 | 1.00 |

The second table is the calibration test. At its first Holm step (two tests per cell), the exact binomial test does not reject 80% coverage for any hit share between `accepted_low` and `accepted_high`.

What this means, honestly:
- **Volatility skill.** At 12 months, only h = 5 can be tested. To be confirmed with 80% power, the model's primary-loss advantage over the frozen mean must be about half a standard deviation of d per outcome (0.471). At 24 months the threshold is about a third of one at h = 5 (0.354) and three quarters at h = 20 (0.751).
  - Volatility clustering makes a HAR-type model beat a constant clearly at short horizons, so h = 5 has a real chance. Longer horizons may not.
  - After a pass, the development check computes the development effect sizes and the implied power per cell, for squared error on log σ and for QLIKE (`docs/research/zero_vol_floor.md`, section 6). Those are planning estimates from out-of-fold development rows, not evidence of prospective skill.
- **Return skill** is expected to be near zero (`docs/research/ml_prospective.md`, Learning Notes 3). With 50 outcomes, no realistic return effect is detectable, so a non-rejection means nothing either way.
- **Calibration.** With 50 outcomes the test rejects 80% coverage only below 62% or above 94% of hits. It catches gross miscalibration, not a band that covers 72% instead of 80%. With 25 outcomes at h = 20 it cannot reject at all on the high side. Calibration evidence accumulates slowly by design, and the descriptive coverage with its interval is reported beside the test.

## 11. Confirmation criteria C1–C4

A model is **prospectively validated for volatility forecasting of fund A at horizon h** only if, at a checkpoint where that cell is in the family, all four hold:

- **C1.** Its primary test (family P, with the approved primary loss) rejects after Holm.
- **C2.** Its 80% volatility-band coverage is not rejected (family B).
- **C3.** Integrity holds from session 1 to the checkpoint:
  - the ledger chain is intact;
  - no kept-head check failed, and every degraded check was resolved before the next recording deadline (section 6);
  - at least 95% of expected sessions are recorded;
  - nothing was backfilled or overwritten;
  - every confirmatory phase session of that cell is recorded or counted missing.
- **C4.** Nothing unregistered changed since registration: the code commit, the config hash, the bundle manifest, the zero-volatility rule and the data path.

**Return skill** needs its own rejection in family A, with C2–C4 for the return band, and is not expected.

**Checkpoints in sequence.** A cell's status is decided at the latest checkpoint where it was in the family.
- **A cell that passes at 12 months** is reported as "confirmed at 12 months".
- **At 24 months it is tested again**, on all its outcomes, as a member of that checkpoint's family whatever its earlier result.
- **If the 24-month test does not reject (C1), or C2 fails there,** the status is withdrawn and reported as "confirmed at 12 months, not confirmed at 24 months".
- **A C3 or C4 failure at any time** also withdraws it.
- **Both results stay in the record**, and neither checkpoint is re-run, pooled or replaced.
- The α split between the checkpoints (section 9) already pays for testing a cell twice.

**If any criterion fails**, the model stays at the registry's computed stage (at most `development_evaluated`). The `prospectively_validated` stage stays unreachable until the owner approves its rules (`docs/research/ml_registry.md`).

**Neither a pass nor a fail authorizes** allocation decisions, recommendations or paper trading. Each needs its own owner decision.

## 12. What is not built yet (needed before activation, each reviewed separately)

- A registration entry kind in the ledger (section 1) and the registration record file.
- Cadence-aware Model Health: today `forecast.ledger_sessions` expects every session, which matches section 3, but it does not yet know session 1, the trial or the phase sessions.
- The zero-volatility rule as a registered config key, so that the config hash states the rule (`docs/research/zero_vol_floor.md`, section 7).
- The checkpoint analysis: non-overlapping selection, the HAC sensitivity, Holm, the exact binomial test and C1–C4. `registration.py` holds only the arithmetic of this document.

## 13. Decisions for the owner

1. Approve sections 1–6: registration, observation timing, cadence, the first eligible close, the trial and the private head-hash backup (and choose the second medium).
2. Approve sections 7–10: the metrics, the frozen nulls, the non-overlapping rule, the HAC sensitivity, the families, Holm and the checkpoints.
3. Approve section 11: criteria C1–C4.
4. Separately: the zero-volatility rule (`docs/research/zero_vol_floor.md`). Both candidate primary losses depend on it.
5. Separately: storing the pre-2025 batch, and OD16 activation (the first real forecast).
6. Choose the primary volatility loss (section 7): (a) squared error on log σ, recommended; or (b) QLIKE on the documented variance conversion.
7. Confirm the recording deadline (section 2): keep "09:30 on the next weekday", stricter and already registered, or ask for "the next session", a config change and a new variant before any freeze.

## Learning Notes

1. **Problem.** A forecast record is worth something only if it was fixed before the outcome and is judged by rules fixed before the data. Pre-registration turns "the model looked good" into a test with a known false-positive rate.
2. **Why non-overlapping outcomes.** Two daily 20-session forecasts share 19 of their 20 outcome sessions, so their errors are nearly the same draw. Counting both as independent inflates n about twentyfold and makes standard errors far too small. Sampling every h-th forecast gives independent windows. Newey–West at lag h − 1 instead keeps every forecast and widens the standard error for the overlap. It is a useful cross-check, but it relies on asymptotics that 50 outcomes do not reach.
3. **Holm's step-down.** Sort the m p-values. Reject the smallest if p₍₁₎ ≤ α/m, the next if p₍₂₎ ≤ α/(m − 1), and so on, stopping at the first non-rejection. It controls the family-wise error rate like Bonferroni but is never less powerful. Example with m = 3 and α = 0.025: p = (0.004, 0.011, 0.030) rejects the first (0.004 ≤ 0.0083) and the second (0.011 ≤ 0.0125), then stops (0.030 > 0.025).
4. **Power, by example.** With n = 50 and a first-step level of 0.025/3, the t statistic of the mean of d has a noncentral t distribution with 49 degrees of freedom and noncentrality δ√50.
   - The exact minimum detectable effect is δ = 0.471.
   - The normal shortcut (z₀.₉₉₁₇ + z₀.₈)/√50 = (2.394 + 0.842)/√50 = 0.458 is a little optimistic, because it ignores that s_d is estimated.
   - A model whose mean gain is 0.3 standard deviations has power 0.37 at 12 months, and it would need about 120 independent outcomes for 80%.
5. **Consistent losses.** A loss is consistent for a quantity if its expected value is smallest when the forecast equals that quantity.
   - Squared error is consistent for the mean, so it matches a model fitted by least squares to log σ.
   - QLIKE is consistent for the mean of the variance, so it matches a variance forecast.
   - Scoring a log-volatility forecast with QLIKE without the conversion compares the forecast with the wrong quantity (section 7).
6. **Leakage risks.**
   - Backdating: the 18:00 / 09:30 window and `retrieved_by`.
   - Choosing the subsample after seeing outcomes: the fixed phase.
   - Including the trial if it looks good: the permanent exclusion.
   - Dropping inconvenient cells: p = 1, never dropped.
   - Re-estimating the null: frozen.
7. **Ways results could mislead.**
   - A pass at h = 5 says nothing about h = 30.
   - One regime (two years) is a small sample of market conditions.
   - Missed sessions are rarely random.
   - The three funds are hindsight-selected.
8. **Practitioner validation.** Diebold–Mariano tests on QLIKE (Patton 2011), Kupiec and Christoffersen coverage tests, and a pre-registered analysis plan with a fixed evaluation date. These are the same ingredients, made exact here.
9. **What to learn next.** Sequential testing with alpha spending (O'Brien–Fleming boundaries), which would allow looks at more checkpoints without inflating error, and the model confidence set (Hansen, Lunde and Nason 2011) for comparing several forecasters at once.
