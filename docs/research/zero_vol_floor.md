# Zero realized volatility: the floor decision (rule B2)

Status (2026-10-17): research only, for the owner's decision. Rule B2 is implemented as an explicitly named, non-default option (`zero_vol.B2`, `"research_b2_half_tick_floor"`). The rule in force is unchanged (`zero_vol.CURRENT`, `"current_drop_and_miss"`), and the forecast job, `freeze` and every default use it. The registered config `etf_prospective_v1` names the rule in force (`zero_volatility_rule: current_drop_and_miss`), so its hash states the rule; a test pins the new hash (`dba70fc2…`) and that the key is its only change from the previous hash (`ae5d5853…`). Nothing has been frozen and no real data has been read. The development-data check (section 6) runs once the owner approves storing the pre-2025 VTI/AGG/SHV batch.

Revised 2026-10-17 after an independent review, still before any real data existed:
- scoring now refuses any rule other than the bundle's training rule;
- each forecast row records its as-of close, which fixes the floor;
- criterion Z2 now tests the floor rather than the band (section 6; the original wording is kept there);
- the simulation code is in the repository (section 5).

Code: `src/stock_agent/forecasting/prospective/zero_vol.py` (the rules), `zero_vol_check.py` (the development check), `scripts/check_zero_vol_floor.py` (the command). Tests: `tests/test_zero_vol_floor.py`.

**Owner decision (pending, one line):** approve rule B2 (section 2) as the log-volatility target in training labels, calibration and scoring for the first frozen ETF model, provided the development check passes Z1–Z4 (section 6) — **yes / no**.

What the owner said on 2026-10-17 (`docs/PROJECT_STATE.md`): "I support investigating a prespecified positive floor for the log-volatility target, but I am not approving an arbitrary replacement value yet." "Keep measured zero volatility distinguishable from the modeling floor." "Do not use protected holdout outcomes to select the floor."

## 1. The problem: two treatments that disagree

The volatility target of row t (as-of session t, entry e = t + 1, exit x = e + h) is y = ½ log(mean g²) over the h daily log total returns g after entry. If SHV's close does not move for h sessions (its close moves in $0.01 steps, about 0.9 bp of $110), every g is 0 and y = log 0 = −∞.

Today:
- **Training and development drop the row** (both of its labels). Development coverage is then measured only on rows that moved, which biases it upward.
- **Prospective scoring counts the same outcome as a miss** (−∞ is below every band).

So the development coverage that justifies the band is measured on a different population from the one the band will be scored on. The bias is largest exactly where zeros cluster: SHV at short horizons in a zero-rate regime.

## 2. The rule (B2)

For row t with unadjusted as-of close P_asof, horizon h and tick τ = $0.01:

```text
sigma_raw      = sqrt( (1/h) · Σ_{k=e+1..x} g_k² )           measured daily volatility, ≥ 0
measured_zero  = every g_k is exactly 0                      (so sigma_raw = 0)
sigma_floor,h  = ½ · ln(1 + τ / P_asof) / √h                 the floor
floored        = sigma_raw < sigma_floor,h
y              = ln(sigma_floor,h)       if floored
               = ½ · ln(mean g²)         otherwise           (today's arithmetic, bit for bit)
```

Every row records `sigma_raw`, `measured_zero` and `floored` (and `sigma_floor`) under both rules. Under the rule in force `floored` is always false and `sigma_floor` is empty, so a measured zero is never confused with the floor.

Why this floor, and not an arbitrary value:
- **One tick, halved.** A single $0.01 move in an otherwise flat window gives sigma_raw = |ln(1 ± τ/P)|/√h. That is the smallest volatility a moving window can show. The floor is half of it, so any window with a full tick move is never floored (a test checks this for P from $20 to $500, up or down). Only measured zeros and windows whose only movement is smaller than half a tick are floored, for example a sub-tick distribution with no price move, or a floating-point residual of the ex-date arithmetic.
- **Divided by √h.** sigma_raw is a daily root mean square over h sessions, so one tick's contribution shrinks as 1/√h. A floor without the √h would bind on genuinely moving long windows (rule B1 in the simulation floored up to 68% of windows).
- **At the as-of close.** P_asof is known at the decision time, so the floor never uses the outcome window. Changing every later close leaves a row's floor unchanged (tested). The simulation's "B2_asof" variant is this rule; its "B2" used a training-period median price instead.
- **Prespecified.** It has no tuned parameter: τ is the exchange tick and ½ is fixed by the argument above. Nothing in it is chosen from outcomes, and the sealed period is never read.

## 3. Worked examples (computed by script)

Produced by `.venv/bin/python scripts/check_zero_vol_floor.py examples` with the production arithmetic: F1-shaped closes and cash go through `series.daily_log_returns` (the approved F5.1 total return) and then `zero_vol.volatility_labels`. A test checks that this table equals the script's output. Units: σ in basis points of daily volatility; y is the log daily volatility.

- **flat**: the close never moves.
- **one_tick**: the close rises $0.01 on the window's first session.
- **sub_tick_cash**: a $0.003 distribution goes ex on the window's first session and the close does not move.

`y_current` is the label under the rule in force (n/a: the row is dropped in training and scored as a miss).

| P_asof | horizon | case | sigma_floor_bp | log_sigma_floor | sigma_raw_bp | measured_zero | floored_b2 | y_current | y_b2 |
|---|---|---|---|---|---|---|---|---|---|
| 110.30 | 5 | flat | 0.2027 | -10.8063 | 0.0000 | yes | yes | n/a | -10.8063 |
| 110.30 | 5 | one_tick | 0.2027 | -10.8063 | 0.4054 | no | no | -10.1131 | -10.1131 |
| 110.30 | 5 | sub_tick_cash | 0.2027 | -10.8063 | 0.1216 | no | yes | -11.3171 | -10.8063 |
| 110.30 | 20 | flat | 0.1014 | -11.4994 | 0.0000 | yes | yes | n/a | -11.4994 |
| 110.30 | 20 | one_tick | 0.1014 | -11.4994 | 0.2027 | no | no | -10.8063 | -10.8063 |
| 110.30 | 20 | sub_tick_cash | 0.1014 | -11.4994 | 0.0608 | no | yes | -12.0102 | -11.4994 |
| 110.30 | 30 | flat | 0.0828 | -11.7022 | 0.0000 | yes | yes | n/a | -11.7022 |
| 110.30 | 30 | one_tick | 0.0828 | -11.7022 | 0.1655 | no | no | -11.0090 | -11.0090 |
| 110.30 | 30 | sub_tick_cash | 0.0828 | -11.7022 | 0.0497 | no | yes | -12.2130 | -11.7022 |
| 50.00 | 5 | flat | 0.4472 | -10.0152 | 0.0000 | yes | yes | n/a | -10.0152 |
| 50.00 | 5 | one_tick | 0.4472 | -10.0152 | 0.8943 | no | no | -9.3220 | -9.3220 |
| 50.00 | 5 | sub_tick_cash | 0.4472 | -10.0152 | 0.2683 | no | yes | -10.5259 | -10.0152 |
| 50.00 | 20 | flat | 0.2236 | -10.7083 | 0.0000 | yes | yes | n/a | -10.7083 |
| 50.00 | 20 | one_tick | 0.2236 | -10.7083 | 0.4472 | no | no | -10.0152 | -10.0152 |
| 50.00 | 20 | sub_tick_cash | 0.2236 | -10.7083 | 0.1342 | no | yes | -11.2191 | -10.7083 |
| 50.00 | 30 | flat | 0.1826 | -10.9110 | 0.0000 | yes | yes | n/a | -10.9110 |
| 50.00 | 30 | one_tick | 0.1826 | -10.9110 | 0.3651 | no | no | -10.2179 | -10.2179 |
| 50.00 | 30 | sub_tick_cash | 0.1826 | -10.9110 | 0.1095 | no | yes | -11.4218 | -10.9110 |

Reading one row by hand (P = 110.30, h = 5): ln(1 + 0.01/110.30) = 9.0658 × 10⁻⁵; half of it is 4.5329 × 10⁻⁵; divided by √5 = 2.2361 it is 2.0272 × 10⁻⁵, which is 0.2027 bp, and ln(2.0272 × 10⁻⁵) = −10.8063. The one-tick window has sigma_raw = 9.0658 × 10⁻⁵/√5 = 0.4054 bp, twice the floor, so it keeps its own label.

The sub-tick rows show why the floor also protects the rule in force. With no floor, the $0.003 cash keeps a label 0.5 log units below the floor (−11.3171 against −10.8063 at h = 5). A floating-point residual of the ex-date arithmetic (one daily return of 10⁻¹⁶ in a flat 5-session window) would give a label of about −37.6, an outlier that ridge and the conformal quantiles would both absorb. The check reports the smallest non-zero sigma_raw per fund and horizon so that this is visible.

## 4. One treatment everywhere

| Stage | Rule in force (`current_drop_and_miss`) | B2 (`research_b2_half_tick_floor`, research only) |
|---|---|---|
| Training labels (`training.training_data`) | A measured-zero row loses both labels (`zero_volatility_rows_dropped`) | y = ln(sigma_floor,h); no row dropped; the return label is kept |
| Development folds and nested selection | Zero rows are absent | Floored rows are ordinary rows |
| Conformal calibration (sequential and frozen quantiles) | Scores come from moving rows only | Floored rows' scores are included, so the band learns the point mass at the floor |
| Historical-mean null | The mean of moving rows' labels | The mean of every label, floored ones included |
| Forecast record (`forecast.record_forecast`) | `bundle.zero_volatility_rule` and each row's `as_of_close` | the same |
| Prospective scoring (`forecast.score_matured`) | outcome = None, `inside_band` = false, note `zero_realized_volatility` | outcome = ln(sigma_floor,h) at the as-of close the forecast recorded, scored like any outcome, note `floored_realized_volatility` |
| Recorded per scored volatility row | `zero_vol_rule`, `sigma_raw`, `measured_zero`, `floored` (false), `sigma_floor` (null), `as_of_close`, `as_of_close_revised` | the same flags; `floored` true when the floor binds |
| QLIKE (`training._qlike`, the registration's primary loss) | Undefined at a zero outcome (σ² = 0) | Finite everywhere |

Guards:
- **The defaults.** Every function defaults to the rule in force. The forecast job's CLI and `freeze` cannot select B2. A test runs the job's `forecast` and `score` subcommands and checks that every score records the rule in force.
- **The bundle decides the scoring rule.** A forecast records the rule its bundle was trained under (`bundle.zero_volatility_rule`). Scoring refuses any other rule (`zero_vol_rule_not_trained`), so training, calibration and scoring always share one treatment. A bundle whose fits disagree on the rule is refused when it is loaded. Bundles and forecasts from before the rule was recorded count as the rule in force.
- **No mixing.** A ledger already scored under one rule refuses scoring under the other (`zero_vol_rule_mismatch`).
- **The floor is fixed at the decision time.** Each forecast row records its unadjusted as-of close (`as_of_close`), and B2's floor is computed from that recorded value. A later revision of the close in the prospective store cannot move the floor; it shows up as `as_of_close_revised` on the score. B2 scoring refuses a forecast without a recorded close (`as_of_close_not_recorded`).
- **Lineage.** The frozen bundle's fits record `training.zero_volatility_rule`. Forecast entries are schema 1.1 (the bundle's rule and the rows' as-of close), and score entries are schema 1.1 (the volatility flags).

Unchanged by B2: the features (a flat 5-session feature window is still imputed with the training median, as today), the total-return target, the band method, the fold geometry, the sealed period, and the registered config.

## 5. Simulation evidence (SIMULATION ONLY; not development-data evidence)

Lead session study, 2026-10-17. The code is in `scripts/research/` (`zero_vol_simulation.py`, `zero_vol_simulation_tables.py`, `zero_vol_simulation_examples.py`), labelled simulation, and reproduces the study's result files byte for byte (about 15 s):

```bash
.venv/bin/python scripts/research/zero_vol_simulation.py <output directory>
.venv/bin/python scripts/research/zero_vol_simulation_tables.py <output directory>
```

`tests/test_zero_vol_simulation.py` checks that it is deterministic and that its floor equals the production formula.

The setup:
- **Latent process.** A daily log return μ + σε, with the price rounded to the cent and distributions as paid.
- **Paths.** P₀ = $110.30 or $50.00; yields of 0%, 0.5% and 4%; σ from 0.25 to 5 bp a day; random-walk, mean-reverting and stochastic-volatility variants. That is 54 scenarios × 10 seeds × h ∈ {5, 20, 30} (162 cells), plus one SHV-history-like path.
- **Forecaster.** Ridge on the same five features, α = 0.1·n fixed, with split-conformal 80% bands on separate train, calibration and test segments.
- **Rules compared.**
  - A: drop zeros in training and calibration, score zeros as misses (the rule in force).
  - C: drop zeros everywhere (what development coverage measures today).
  - B2_asof: this rule.

Results:
- **Coverage, all 162 cells.**
  - A: min 0.209, median 0.786.
  - C: min 0.745, median 0.794, max 0.847.
  - B2_asof: min 0.745, median 0.794, max 0.847.
- **Coverage in the 44 cells where more than 1% of windows are zero** (zero rates 1.1% to 74%).
  - A: mean 0.619 and min 0.209. Only 12 of 44 cells are within 0.80 ± 0.05.
  - B2_asof: mean 0.797, range 0.769 to 0.829. All 44 cells are within 0.80 ± 0.05.
  - B2_asof minus A: at least +0.008 in every cell, +0.178 on average.
- **Conditional coverage is uneven, and this is the main caveat.** In those 44 cells B2_asof covers zero rows 35% of the time on average (0% to 85%) and non-zero rows 82% (48% to 89%). A marginal conformal band cannot calibrate a point mass at the floor separately from the continuous part.
- **Band width.** B2_asof bands are wider in zero-heavy cells: 1.04 log units against 0.78 for A and C.
- **QLIKE against the latent volatility.** B2_asof minus A has median 0.000 and ranges from −1.18 (better) to +0.197. It is worse by more than 0.005 in 4 of 162 cells, all at P = $50, a 0.5% yield and σ ≤ 0.5 bp: the regime where sub-tick distributions make windows floored but not zero. With the training-median floor, up to 12% of windows in that regime were floored without being zero. That is why criterion Z3 exists.
- **SHV-history-like path** (zero rates, then hikes, zero rates, then 4.5%; trained on the early regimes, tested on the last).
  - Coverage at h = 5: A 0.632, B2 0.708. At h = 20 and h = 30 both rules give 0.591 and 0.537.
  - No zero-volatility rule fixes a regime change.

What the simulation cannot show: real SHV's tick behaviour, the distribution schedule, the vendor's rounding and actual zero frequencies. Those come from section 6.

## 6. The development-data check (the plan, fixed before any real run)

The command, run in the owner's terminal once the pre-2025 batch is stored:

```bash
PYTHONPATH=src .venv/bin/python scripts/check_zero_vol_floor.py check --research-store ~/stock-agent-private/fund_store
```

What it reads and refuses:
- **The F1 research store only**, read through its fail-closed reader. A prospective store, a directory inside one, or anything else that is not F1 batches is refused (exit 2).
- **Nothing after 2024-12-31.** Any price session, distribution date (ex, declaration, record or pay) or split dated after 2024-12-31 is refused before anything is computed (exit 2).
- **It never writes anywhere else.** It does not freeze, register, record or contact a network.
- **Output.** It writes `report.json` and `report.md` once, under a new timestamped directory of `reports/ml/prospective/zero_vol_check/` (ignored by Git).

What it reports, per fund, under both rules, with the registered config `etf_prospective_v1`:
1. **Windows.** Per fund × h × as-of year: eligible windows, measured zeros, floored windows and floored-but-not-zero windows. Per fund × h: the totals and the smallest non-zero sigma_raw.
2. **The cent grid.** Per fund × year: the share of unadjusted closes on the $0.01 grid and the worst offset. The floor's premise is the tick. If SHV's closes are not on the grid, that is reported and escalated, not fixed by changing the rule.
3. **Coverage.** The development 80% band coverage (sequential, out of fold), as the fold-block estimate and standard error with the circular block-bootstrap 95% interval:
   - under the rule in force;
   - under B2;
   - for the current bands re-scored with every measured zero in a scored fold counted as a miss. That is what prospective scoring does today.
4. **Coverage on zero and non-zero rows separately**, per rule.
5. **Coverage on non-zero rows under both rules, on the same rows** (`nonzero_coverage`): the rows that have a band under both rules and are not measured zeros, each rule's share inside its own band, and the difference. This is what Z2 reads.
6. **Band width.**
7. **Losses.** The QLIKE and squared-error improvements over the historical-mean null, under both rules.
8. **Residual shift.** On non-zero rows, B2 minus the rule in force.
9. **Bit identity.** Per fund × h, whether predictions, bands, metrics and selections are identical under both rules.

**Pre-stated acceptance criteria, evaluated in this order:**

- **Z1.** VTI, AGG and every asset-horizon without a floored window give bit-identical development predictions, bands, metrics and selections under both rules.
- **Z2.** On the volatility rows scored under both rules, B2's band coverage on non-zero rows is within 0.02 of the current rule's coverage for every asset and horizon; coverage on measured-zero rows is reported.
- **Z3.** Floored windows that are not measured zeros are at most 1% of the eligible windows of every asset and horizon.
- **Z4.** The mean residual shift (B2 minus current, log volatility) on non-zero development rows is at most 0.05 in absolute value for every asset and horizon.

The check runs under the registered config. It is written down here before any real run, so neither the thresholds nor the order can be tuned to a result.

**A reported check, never a criterion:**

- **R1.** Band calibration (reported, never a criterion): SHV volatility-band coverage under B2 (fold-block estimate) against 0.80 +- 0.05 at every horizon, and against the current bands with measured zeros as misses.

R1 is reported whatever happens to Z1–Z4, including after a stop. It never stops the run and is never a reason to adopt or reject B2.

**Revision record (2026-10-17, before any real data existed).** The first version of these criteria, merged in PR #91, had this Z2:

> SHV volatility-band coverage under B2 (fold-block estimate) is within 0.80 +- 0.05 at every horizon, and no lower than the current bands with measured zeros as misses.

An independent review found that it tests the band's calibration, not the floor. The simulation's SHV-like path, with coverage of 0.708, 0.591 and 0.537, would fail it under every rule, because no zero-volatility rule fixes a regime change. The criterion was replaced by the Z2 above, and the old test is kept as the reported check R1. No real data had been stored or read when this was decided, so the revision cannot have been tuned to a result.

**What to expect from Z2, stated before any real run.** A marginal band that has to cover floored rows widens. With a share p of floored rows that the band covers with probability c₀, keeping overall coverage near 0.80 implies non-zero coverage of about (0.80 − p·c₀)/(1 − p). That is above 0.80 by about p·(0.80 − c₀)/(1 − p).
- For c₀ = 0.35 (the simulation's average), Z2's 0.02 tolerance is reached near p = 4%.
- With small calibration sets, a few extreme floored scores can move the quantile further. On the synthetic test variant, 13 zero rows in about 870 moved SHV's 5-session non-zero coverage by +0.08.

So Z2 may fail for SHV at h = 5 if 2010–2015 had many flat 5-session windows. The window counts (item 1) show that share. A failure would mean B2, as specified, distorts the rows it does not floor. It would be reported, not repaired by changing the tolerance.

**Failure.** The first failing criterion stops the run. The criteria after it are not evaluated, and the run exits 1 with the measured numbers. A failure is reported to the owner. It is never resolved by switching rules, changing the floor or loosening a threshold: any of those would be a new, separately approved proposal.

**Pass.** It means only that B2 behaves as designed on development data. It is not approval, and the rule in force stays until the owner records a decision. After a pass the report adds the registration power estimates: development loss differentials on non-overlapping outcomes, for squared error on log volatility and for QLIKE (`docs/research/prospective_registration.md`, sections 7 and 10).

What the criteria guard against:
- **Z1:** the option changing anything where it should not bind.
- **Z2:** the floor distorting the bands of the rows it does not floor. Marginal calibration, which no zero rule can guarantee across regimes, is reported as R1.
- **Z3:** the regime where the floor would replace measured movements. The simulation found it.
- **Z4:** the floor shifting the model itself, not only the flat rows.

## 7. Limitations

- The development coverage is computed on folds from 2010 to 2024. The sealed period (2025-01-01 to H_end) is never used to choose or check the floor.
- Marginal coverage can pass while zero rows stay under-covered (section 5). The zero/non-zero split is reported so that this is visible. It is not a criterion, because no fixed threshold for it was justified in advance.
- A floored label is a modeling convention, not a measurement: ln(sigma_floor,h) says "below half a tick", not "this was the volatility".
- The rule is a registered config key, `zero_volatility_rule`, so the config hash and every bundle state which rule they used. It names the rule in force: only `zero_vol.REGISTRABLE` (the rule in force alone) may be named, frozen or registered, `freeze` and `forecast` refuse a config whose rule differs from the bundle's (`zero_vol_rule_mismatch`). `score` refuses any rule other than the registration's, which equals the config key and every fit's rule, or than each pending forecast's bundle rule. The job's CLI has no option that selects a rule. On approval of B2, B2 would join `REGISTRABLE` and the config would name it: a new registered variant before any freeze, with its own review.

## 8. Learning Notes

1. **Problem.** A log transform is undefined at zero, and a price grid makes exact zeros happen. Training, calibration and scoring must agree on what such an outcome is, or the validation measures a different population from the one that will be scored.
2. **Concept: censoring at a detection limit.** Measuring volatility from cent-rounded prices is like a lab assay with a detection limit: below half a tick the instrument cannot distinguish "no movement" from "small movement". Standard practice is to record the value as "below the limit" (a flag) and model it at a conventional value, here ln(sigma_floor,h), rather than drop it (which biases toward larger values) or call it −∞ (which no loss function can score).
3. **Assumptions.**
   - Closes are on the $0.01 grid; the check verifies this.
   - The as-of close is the right scale for the tick.
   - A marginal band is acceptable even though the floor creates a point mass.
4. **Formula.** σ_floor,h = ½·ln(1 + τ/P)/√h. For P = 110.30 and h = 20: ½ × 9.0658 × 10⁻⁵ / 4.4721 = 1.0136 × 10⁻⁵, which is 0.1014 bp and ln = −11.4994.
5. **Leakage risks.** A floor computed from the outcome window, or from a full-sample price, would use information unavailable at the decision; P_asof avoids both. Choosing τ, the ½ or the √h by looking at holdout coverage would be selection on the sealed period. All three are fixed by the argument in section 2, before any real data is read.
6. **How results could mislead.**
   - A pass on 2010–2024 SHV says nothing about a future regime.
   - Coverage near 80% can hide 35% coverage on zero rows.
   - Bit identity for VTI and AGG is a software check, not evidence of skill.
7. **Practitioner validation.** Report coverage conditional on the censoring flag. Compare with a two-part model (P(zero) and the volatility given movement). Use a proper scoring rule for intervals (the interval or pinball score) next to the coverage counts.
8. **What to learn next.**
   - Tobit and censored regression.
   - The "hurdle" (two-part) models for point masses.
   - Why QLIKE needs a strictly positive realized variance (Patton 2011, "Volatility forecast comparison using imperfect volatility proxies").

**Exercise (bounded).** In `tests/zero_vol_helpers.py`, change the flat regime's latent σ from 0.25 bp to 0.5 bp and rerun `tests/test_zero_vol_floor.py::test_the_window_counts_and_the_cent_grid`. Predict first whether SHV's h = 5 zero count rises or falls, and by roughly what factor; then check `window_totals`.
