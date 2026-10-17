# A7 decision packet: which test decides "a challenger beats B1"

Status: synthetic panels only; awaiting the owner. The pre-registration is
unchanged below.

How the ordering is evidenced, and how far:
- **Server-side.** GitHub's repository activity log
  (`repos/billvo9/stock-allocation-agent/activity`) records:
  - the branch created at `2189fe7` (the pre-registration and the code) at
    2026-10-17 21:33:07 UTC;
  - `1011056` (the results) pushed at 22:36:18 UTC;
  - PR #93 opened at 22:36:28 UTC.
- **Local only.** That the first simulation ran after 21:33:07 UTC rests on
  local evidence alone: the commit time of `2189fe7` (21:33:02 UTC) and the
  first record written about 21:47 UTC. GitHub holds no record of when a
  simulation ran.

B1 stays the Stock Lab's default throughout. This packet changes no rule,
default or stored record. It gives the owner the evidence for decision A7
(`docs/research/stock_probability_lab.md`, section 7).

## Summary

- **Recommendation: the fold-block t decides, at a one-sided family level
  of 2.5%.** It is a t-test on the 19 yearly fold means with t(18) critical
  values. Holm runs across the four challengers at 2.5%, and the
  simplest-first steps use 2.5%. The rule's structure is otherwise
  unchanged. B1 stays the default.
- **Why the fold-block t.** It is the only one of the six procedures within
  ±1 pp of 5% in all 32 pre-registered cells (worst 0.9 pp). The current
  test, Newey-West at lag 10, is 1.8 pp off: liberal (5.7–6.8%) on the
  symmetric null that the stepwise rule faces, and conservative on the
  own-history null.
- **Why 2.5% rather than 5%.** This came from simulations run after the
  choice, in review. The pre-registered criterion fixed only the procedure.
  - At 5%, the fold-block t does not hold a 5% false-selection rate
    everywhere. Holm's family-wise rate reaches 6.1% (N = 6, H = 21, about
    3 Monte Carlo SE above 5%). Under the evaluation's neighbour placebo it
    rejects 6.1–8.1% at N = 4 and 5.6–7.6% at N = 6.
  - At 2.5%, every simulated rate at N = 4 and 6 is at most 4.9%.
- **The cost is power, which is low either way.** Against a challenger
  twice as noisy as the placebo (as B3 is on the synthetic demo), at an RPS
  gain of 0.0005 and N = 4: 48% at 5%, 36% at 2.5%. At three times the
  noise (M1-like): 29% and 20%.
- **The alternative** is the fold-block t at 5%, which meets the
  pre-registered bar. Both are set out for the owner below.

## Pre-registration (written before running)

Everything in this section was fixed before the simulation was run. It uses
synthetic panels only: no real price, no real evaluation result, nothing
under `data/` or `reports/`.

### Planned conditions

- **Universe sizes:** N = 4, 6, 10 and 30 stocks. The owner's evaluable
  watchlist is about 4 to 6 names; the watchlist cap is 30.
- **History and folds:** the panel starts in 2003, the first three years are
  warm-up (the evaluation's `MIN_TRAIN_SESSIONS` = 756), and the test years
  are 2006 to 2024: 19 yearly folds of 252 sessions, T = 4,788 decision
  dates.
- **Late listings:** the first max(4, ⌈0.7N⌉) names have the full history.
  The other L names (N = 6: 1, N = 10: 3, N = 30: 9) enter the per-date
  cross-sectional mean at evenly spaced test dates ⌊kT/(L + 1)⌋,
  k = 1..L. Their estimator state is warmed up on latent history, which
  keeps the null exact; a real fresh listing's B1 is noisier.
- **Targets:** each decision date carries the five daily targets h = 1..5,
  so neighbouring dates share four of their five outcomes.
- **Returns:** r_it = σ_t (√ρ z_mt + √(1 − ρ) z_it), with standardized
  Student-t₅ shocks and ρ = 0.5.
  - **Volatility clustering:** a common GARCH(1,1) variance,
    σ²_t = ω + a σ²_(t−1) z²_m,(t−1) + b σ²_(t−1), with a = 0.06,
    b = 0.92 and an unconditional daily volatility of 2.5%.
  - The variance is common to all stocks. That keeps the null exact:
    another stock's history says as much about the shared volatility as the
    stock's own. Idiosyncratic volatility clustering would make the own
    history genuinely more informative, which is a null that favours B1.
- **Lab settings:** band τ = 0.5%, B0 prior α = 50, B1 prior κ = 40, and
  every pre-registered half-life H ∈ {21, 63, 126, 252}.

### Nulls and alternatives

- **Own-history null (the decision comparison).** Each stock has four
  challengers. Challenger j is B1 at the same half-life read from a
  "shadow" stock's history: a stock with the same return law that is never
  scored. The reference is the stock's own B1.
  - Every date's differential has mean exactly zero (the histories are
    exchangeable).
  - The imbalance of the real comparisons is kept: the stock's own B1 later
    learns from the sessions it was scored on, and the challengers never
    do.
- **Symmetric null (the stepwise rule's comparisons).** Challenger k minus
  challenger j. It is exactly antisymmetric, so it measures the tests alone.
- **Alternatives.** A location shift: a challenger's per-date differential
  minus δ, for δ = 0.0005 and 0.001 in RPS. Every procedure's standard
  error is shift-invariant, so size and power come from the same panels.

### Procedures compared (six)

All are one-sided at a nominal 5%, on the per-date differential d_t.

| Id | Procedure | Critical values |
|---|---|---|
| `nw_lag10` | Newey-West, Bartlett, lag 10 (the current decision test) | normal |
| `fold_block_t` | t-test on the 19 yearly fold means | t(18) |
| `nw_fixed_b_sqrt` | Newey-West, Bartlett, bandwidth M = ⌈1.3√T⌉ (Lazarus, Lewis, Stock and Watson 2018) | fixed-b at b = M/T |
| `nw_fixed_b_tenth` | Newey-West, Bartlett, bandwidth M = ⌈0.1T⌉ | fixed-b at b = M/T |
| `cbb_block20` | circular block bootstrap SE, block 20 (the current second view) | t(k − 1), k blocks |
| `sb_politis_white` | stationary bootstrap SE, expected block length by Politis-White (2004) with the Patton-Politis-White (2009) correction | t(⌊T/L̂⌋ − 1) |

- Fixed-b critical values come from a seeded simulation of the Bartlett
  fixed-b limit (Kiefer and Vogelsang 2005): 1,000 steps and 50,000 draws,
  each used with its negative. Its 95% and 97.5% points are checked against
  Kiefer and Vogelsang's published polynomials.
- The bootstrap variances are the exact B → ∞ limits (closed forms), not
  finite resamples.
- Holm across the four challengers and the stepwise "simplest first" rule
  are applied exactly as `evaluation._score` and `evaluation._select` apply
  them, with each procedure's critical values.

### Simulation size

- 4,000 panels per universe size, seeded (seed 20261017 + N; the fixed-b
  simulation uses seed 20261017).
- The Monte Carlo standard error of a 5% rate is about 0.34 pp.
- All six procedures are computed on the same panels.
- Code: `stock_lab.decision_tests` (the procedures),
  `stock_lab.planned_simulation` (the panels, the family rule and the
  criterion, `choose_procedure`) and `scripts/stock_lab_a7_simulation.py`.

### The selection criterion

1. **Size cells.** The one-sided rejection rate at a nominal 5%, for each
   universe size (4) and half-life (4), under each null (2): 32 cells per
   procedure. Distortion is the largest |rate − 5%| over the 32 cells.
2. **Admissible.** A distortion of at most 1.0 pp, judged on the point
   estimates.
3. **Choice.** Among admissible procedures, the highest single-comparison
   power at δ = 0.0005, averaged over the 16 own-history cells (N × H).
   - If the top two are within 1 pp of each other, take the one with the
     smaller distortion.
   - If they are still tied, take the simpler one, in this order:
     `fold_block_t`, `nw_lag10`, `nw_fixed_b_sqrt`, `nw_fixed_b_tenth`,
     `cbb_block20`, `sb_politis_white`.
4. **If none is admissible:** the procedure with the smallest distortion,
   with the shortfall stated.
5. **Reported, never used to choose:** Holm's family-wise false-selection
   rate, the stepwise rule's outcomes, and selection power with one true
   challenger.

No real historical evaluation result enters this choice. Six procedures were
tried; that count is part of the record.

## The options' equations

Every option tests one per-date differential d_t, t = 1..T: a challenger's
RPS minus B1's, averaged over the stocks present and the five daily
targets. H0: E d_t ≥ 0; the challenger is better when the mean is negative.
Each option is a t-ratio τ = d̄ / SE, and it rejects when τ < c_α (c_α < 0).
The autocovariances are γ̂_l = T⁻¹ Σ_(t>l) (d_t − d̄)(d_(t−l) − d̄).

| Option | SE² | c_α |
|---|---|---|
| (a) Fold-block t | s²_F / F, with s_F the standard deviation of the F fold means d̄_f | t_(F−1) quantile |
| Newey-West (current) | ω̂_M / T, with ω̂_M = γ̂_0 + 2 Σ_(l<M) (1 − l/M) γ̂_l and M = 11 (lag 10) | normal quantile |
| (b) Newey-West, fixed-b | ω̂_M / T, with M = ⌈1.3√T⌉ = 90 or ⌈0.1T⌉ = 479 | fixed-b quantile at b = M/T |
| (c) Circular block bootstrap | [(k − 1) V(20) + V(ℓ)] / T², with k = ⌈T/20⌉ blocks, a last block of ℓ sessions, and V(m) the variance over start dates of a circular m-session sum | t_(k−1) quantile |
| (c) Stationary bootstrap | T⁻¹ [γ̂_0 + 2 Σ_(i<T) b(i) γ̂_i], b(i) = (1 − i/T) qⁱ + (i/T) q^(T−i), q = 1 − 1/L̂ | t_(⌊T/L̂⌋−1) quantile |

- **Fixed-b.** Kiefer and Vogelsang hold b = M/T fixed as T grows. τ then
  converges to W(1)/√Q_b, with
  Q_b = (2/b)[∫₀¹ B(r)² dr − ∫₀^(1−b) B(r) B(r + b) dr] and B a Brownian
  bridge. The quantiles are wider than the normal's, and widen with b.
- **Politis-White block length.** L̂ = (Ĝ²/ĝ²)^(1/3) T^(1/3). ĝ and Ĝ are
  flat-top-weighted sums of γ̂_l and l·γ̂_l up to 2m̂ lags, where m̂ is the
  lag after which five autocorrelations in a row are insignificant
  (`decision_tests.politis_white_block`).
- **The bootstrap variances are exact.** They are the closed-form limits as
  the number of resamples grows; `circular_block_variance` is checked by
  enumeration. The evaluation's 499 resamples add a little noise.
- **Holm.** The four one-sided p-values, sorted, are compared with α/4, α/3,
  α/2 and α; rejection stops at the first failure.
- **Simplest first.** The beating challengers are taken in the order B0,
  B3, B2, M1. A later one replaces the current choice only if its
  differential against it rejects at α.

## Results

From `docs/research/stock_lab_a7_simulation.json`
(`scripts/stock_lab_a7_simulation.py`). A rerun from the same seeds
reproduced it exactly.

- The Monte Carlo standard error of a rate near 5% is 0.13–0.23 pp. It is
  smaller than the 0.34 pp of the pre-registration because each panel
  carries 4 own-history and 12 symmetric one-sided tests.
- The own-history null's mean differential was within 1.7 Monte Carlo
  standard errors of zero in every cell.
- Ranges below run over the four half-lives.

**Size, one comparison, own-history null (nominal 5%)**

| Procedure | N = 4 | N = 6 | N = 10 | N = 30 |
|---|---|---|---|---|
| `nw_lag10` (current) | 4.0–4.8% | 4.3–5.0% | 4.3–4.9% | 3.8–4.7% |
| `fold_block_t` | 4.9–5.4% | 4.6–5.8% | 4.6–5.4% | 4.1–5.2% |
| `nw_fixed_b_sqrt` | 3.7–5.1% | 3.7–5.5% | 3.7–5.2% | 3.4–5.0% |
| `nw_fixed_b_tenth` | 5.3–5.7% | 5.2–6.2% | 5.1–5.7% | 4.4–5.3% |
| `cbb_block20` | 3.8–4.7% | 3.8–4.8% | 3.8–4.8% | 3.3–4.5% |
| `sb_politis_white` | 4.0–5.3% | 3.9–5.4% | 4.0–5.3% | 3.6–5.1% |

**Size, one comparison, symmetric null**

| Procedure | N = 4 | N = 6 | N = 10 | N = 30 |
|---|---|---|---|---|
| `nw_lag10` (current) | 5.9–6.6% | 5.7–6.4% | 5.8–6.4% | 5.9–6.8% |
| `fold_block_t` | 4.9–5.2% | 4.7–5.0% | 4.8–5.1% | 5.0% |
| `nw_fixed_b_sqrt` | 4.8–5.1% | 4.6–4.9% | 4.6–5.1% | 4.9–5.2% |
| `nw_fixed_b_tenth` | 4.9–5.2% | 4.8–5.0% | 4.6–5.0% | 4.9–5.1% |
| `cbb_block20` | 5.3–5.8% | 5.0–5.7% | 5.1–5.6% | 5.3–6.0% |
| `sb_politis_white` | 5.3–5.7% | 5.1–5.6% | 5.2–5.5% | 5.4–6.0% |

**Power, one comparison, RPS gain δ = 0.0005 (mean over the half-lives)**

| Procedure | N = 4 | N = 6 | N = 10 | N = 30 |
|---|---|---|---|---|
| `nw_lag10` (current) | 94.7% | 98.2% | 99.8% | 100% |
| `fold_block_t` | 93.2% | 97.5% | 99.6% | 100% |
| `nw_fixed_b_sqrt` | 93.8% | 97.8% | 99.7% | 100% |
| `nw_fixed_b_tenth` | 93.2% | 97.5% | 99.6% | 100% |
| `cbb_block20` | 94.1% | 97.9% | 99.7% | 100% |
| `sb_politis_white` | 94.3% | 98.0% | 99.7% | 100% |

At δ = 0.001 every procedure's power is at least 99.96% in every cell.

**The family: Holm across four challengers, then simplest first**

| Procedure | Any false selection, N = 4 | N = 6 | N = 10 | N = 30 | Picks the one true challenger (last in order, δ = 0.0005), N = 4 |
|---|---|---|---|---|---|
| `nw_lag10` (current) | 3.3–4.2% | 3.4–4.7% | 3.4–3.9% | 2.9–4.2% | 71–98% |
| `fold_block_t` | 4.9–5.8% | 4.4–6.1% | 4.1–5.2% | 3.6–4.9% | 65–95% |
| `nw_fixed_b_sqrt` | 2.8–4.6% | 2.8–5.5% | 3.2–4.5% | 2.5–4.3% | 69–97% |
| `nw_fixed_b_tenth` | 4.7–5.5% | 4.4–5.6% | 4.4–5.1% | 3.8–4.5% | 64–95% |
| `cbb_block20` | 2.8–3.9% | 2.8–4.5% | 3.0–3.7% | 2.5–3.9% | 69–98% |
| `sb_politis_white` | 3.2–4.7% | 3.0–5.6% | 3.5–4.7% | 2.9–4.7% | 70–98% |

**The pre-registered choice** (`choose_procedure` on the recorded cells; reproduced in tests)

| Procedure | Distortion (worst of 32 cells) | Mean power at δ = 0.0005 | Admissible |
|---|---|---|---|
| `fold_block_t` | 0.9 pp (4.1%, N = 30, H = 252) | 97.6% | **yes (chosen)** |
| `nw_fixed_b_tenth` | 1.2 pp (6.2%, N = 6, H = 63) | 97.6% | no |
| `sb_politis_white` | 1.4 pp | 98.0% | no |
| `nw_fixed_b_sqrt` | 1.6 pp | 97.8% | no |
| `cbb_block20` | 1.7 pp | 97.9% | no |
| `nw_lag10` (current) | 1.8 pp (6.8%, symmetric, N = 30) | 98.1% | no |

**Checks run after the choice (reported only, never used to choose)**

- **Constant volatility** (`--set arch=0.0 garch=0.0`,
  `stock_lab_a7_sensitivity.json`). The fold-block t is again the only
  admissible procedure: worst distortion 0.9 pp, own-history 4.1–5.9%,
  symmetric 4.7–5.2%. Newey-West at lag 10 is 1.5 pp off.
- **The evaluation's own size check at the planned shapes**
  (`--evaluation-check`, `stock_lab_a7_evaluation_check.json`; 1,000 panels,
  Monte Carlo SE about 0.7 pp). This is `size_simulation` with its neighbour
  placebos (iid Student-t returns, ρ = 0.4, 1.5% daily volatility), at each
  universe size and 19 test years:

  | Placebo | Test | N = 4 | N = 6 | N = 10 | N = 30 |
  |---|---|---|---|---|---|
  | own history (neighbour) | Newey-West lag 10 | 6.5–8.1% | 6.4–7.6% | 5.9–6.7% | 5.6–6.2% |
  | own history (neighbour) | fold-block t | 6.1–8.1% | 5.6–7.6% | 5.4–6.5% | 4.6–5.8% |
  | own history (neighbour) | block bootstrap (20) | 6.3–7.7% | 5.7–6.9% | 5.1–6.1% | 5.1–5.4% |
  | symmetric | Newey-West lag 10 | 5.1–6.7% | 6.3–7.7% | 5.7–7.0% | 5.1–5.9% |
  | symmetric | fold-block t | 4.5–5.7% | 4.4–6.1% | 4.3–5.6% | 4.4–4.6% |
  | symmetric | block bootstrap (20) | 5.0–6.2% | 5.5–6.8% | 5.0–6.1% | 4.1–5.3% |

  On the symmetric null, the fold-block t again holds its size and
  Newey-West at lag 10 does not. On this own-history placebo, every test is
  1–3 pp liberal at N = 4 to 10.

**Power for real challengers.** The placebo challengers differ from B1 only
by sampling noise. Real challengers differ more, so their differential is
noisier.

- The true standard deviation of d̄ under the null, at H = 63, is
  1.49 × 10⁻⁴ (N = 4), 1.27 × 10⁻⁴ (N = 6), 1.03 × 10⁻⁴ (N = 10) and
  0.59 × 10⁻⁴ (N = 30). With ρ = 0.5, extra stocks help slowly.
- On the three synthetic demo stocks (`evaluation.evaluate` on
  `synthetic.demo_histories`, 777 dates), scaled to T = 4,788 dates, the
  Newey-West SE of B3 − B1 is about 2.5 × 10⁻⁴ and of M1 − B1 about
  4.4 × 10⁻⁴: about 2× and 3× the placebo's at N = 4.
- The t-ratio is scale-free, so power against δ at k times the noise equals
  the simulated power against δ/k. These are simulated, from the same
  seeds, with δ/2 and δ/3 added (`stock_lab_a7_noise_power.json` and
  `stock_lab_a7_stricter_level.json`).

**Power and false selection for the fold-block t at H = 63 (power ranges
run over the four half-lives in brackets)**

| Level | N | Power, 2× noise, δ = 0.0005 | Power, 3× noise, δ = 0.0005 | Power, 2× noise, δ = 0.001 | Power, 3× noise, δ = 0.001 |
|---|---|---|---|---|---|
| 5% | 4 | 48% (43–73%) | 29% (26–46%) | 91% (87–99%) | 67% (60–89%) |
| 5% | 6 | 59% (52–83%) | 36% (32–56%) | 97% (95–100%) | 78% (72–95%) |
| 2.5% | 4 | 36% (31–61%) | 20% (17–34%) | 84% (78–98%) | 55% (48–82%) |
| 2.5% | 6 | 46% (40–72%) | 25% (21–44%) | 93% (90–100%) | 68% (60–91%) |

**The stricter level, simulated after review (reported only; N = 4 and 6,
the evaluable watchlist; ranges over the four half-lives)**

| Fold-block t | One comparison, own-history (shadow) | Symmetric | Holm family, any false selection | One comparison, neighbour placebo |
|---|---|---|---|---|
| 5%, N = 4 | 4.9–5.4% | 4.9–5.2% | 4.9–5.8% | 6.1–8.1% |
| 5%, N = 6 | 4.6–5.8% | 4.7–5.0% | 4.4–6.1% | 5.6–7.6% |
| 2.5%, N = 4 | 2.6–2.9% | 2.3–2.6% | 2.5–3.2% | 3.3–4.9% |
| 2.5%, N = 6 | 2.3–3.2% | 2.4% | 2.3–3.4% | 3.4–4.7% |

- The 2.5% runs use `--set nominal=0.025`: Holm's levels become 0.625%,
  0.83%, 1.25% and 2.5%, and the steps 2.5%.
- The neighbour column comes from `--evaluation-check --set nominal=0.025`
  (`stock_lab_a7_evaluation_check_025.json`; 1,000 panels, Monte Carlo SE
  about 0.6 pp).
- The neighbour placebo has one challenger, so its family rate is not
  simulated. In the shadow design the family rate stayed within 0.5 pp of
  the one-comparison rate in every cell.

## Recommendation

**The fold-block t at a one-sided family level of 2.5% decides "a
challenger beats B1". B1 stays the default.**

- **Statistic.** d̄_f is the mean of the pooled daily differential over the
  test dates of yearly fold f (the evaluation's folds, 2006–2024 for names
  listed by 2003: F = 19). Then τ = mean_f(d̄_f) / (s_F / √F), with one-sided
  p = P(t_(F−1) ≤ τ).
- **Holm at 2.5%, F = 19 (t(18)), lower-tail critical values.** τ < −2.775
  at 0.625%, −2.639 at 0.83%, −2.445 at 1.25% and −2.101 at 2.5%
  (`decision_tests.PLANNED_FOLD_T_STRICT`, reproduced in tests).
- **Steps.** A more complex winner replaces the current choice only if its
  fold-block t against it gives τ < −2.101 (2.5%, unadjusted).
- **Why the stricter level is recommended.** At 5% the fold-block t keeps
  each test's own distortion within 1 pp, but it does not hold the
  false-selection rate at 5%:
  - Holm's family rate reaches 6.1%;
  - B1's learning imbalance, under the neighbour placebo, takes one
    comparison to 6–8% at N = 4 and 6.

  At 2.5% every simulated rate at N = 4 and 6 is at most 4.9%, under both
  placebos. The highest, the neighbour placebo's 4.9% (N = 4, H = 126),
  has a Monte Carlo SE of 0.7 pp, so it is "at or near 5%", not certainly
  below it. The price is power: at twice the placebo's noise, a gain of
  0.0005 is found 36% rather than 48% of the time at N = 4.
- **The alternative: the fold-block t at 5%.** It meets the pre-registered
  ±1 pp bar. Its Holm critical values are τ < −2.445 at 1.25%, −2.304 at
  1.67%, −2.101 at 2.5% and −1.734 at 5% (`decision_tests.PLANNED_FOLD_T`),
  and its steps use −1.734. Choose it to favour power and accept up to
  about 8% false selection at N = 4–6.
- **Both levels:**
  - with fewer folds, the degrees of freedom are F − 1;
  - below three folds the test is unavailable, and B1 stays;
  - Newey-West (lag 10) and the circular block bootstrap stay reported
    only.
- **Already computed.** The evaluation already records the fold-block t for
  every comparison (`fold_t_p_one_sided`). Adopting it changes which
  p-value `evaluation._score` and `_select` read, and at 2.5% the rule's
  ALPHA. It is a small follow-up, after the owner's approval.
- **For reference only (not admissible):** the runner-up, Newey-West with
  M = 479 and fixed-b critical values −2.637, −2.484, −2.246 and −1.839
  at 5%.
  - The simulated fixed-b 95% point at b = 0.1 is 1.839. Kiefer and
    Vogelsang's cubic approximation gives 1.866, and an independent
    simulation in review gave 1.841 and 1.824.
  - At b = 0.019 the two agree (1.686 and 1.687).

## Remaining limitations

1. **The learning imbalance is not a test problem.** B1 keeps learning
   from the sessions it was scored on, and a challenger may not.
   - How much that favours the challenger depends on how the challenger
     learns. With unscored shadow histories the tests sit at or below
     nominal. With the neighbour placebo every test reaches 6–8% at N = 4.
   - Real challengers (B3, M1) learn from the stock's own past in
     different ways, so their imbalance is unknown.
   - At 5%, read the nominal level as up to about 8% per comparison at
     N = 4–6. At 2.5%, the simulated rate stays at most 4.9%.
   - That holds only for the two placebos simulated. A third kind of
     challenger could differ.
   - Confirmation in the prospective ledger (A4), before any challenger's
     probabilities are shown as preferred, is a further safeguard. It was
     not simulated.
2. **Few stocks.** The evaluable watchlist is about 4 to 6 names (SKHY may
   be unsupported, and SNDK's pre-2025 history is another company's). At a
   correlation of 0.5 the cross-section adds little (above).
   - A B3-like challenger's gain of 0.0005 is found 48% of the time at 5%
     and 36% at 2.5% (N = 4).
   - An M1-like challenger's is found 29% and 20% of the time.
   - Small real gains will usually not be detected, and B1 will usually be
     kept.
3. **Few folds.** The fold-block t's sample is 19 yearly means.
   - Its t(18) critical values cost a little power.
   - One extreme year, such as 2008 or 2020, is one nineteenth of the
     evidence. That is transparent (each year's mean can be shown), but it
     is coarse.
4. **Hindsight-selected names.** The watchlist is semiconductor names that
   survived and grew, so any result is exploratory and non-confirmatory (A3
   condition). No test choice changes that.
5. **Regime shifts.** The simulations are stationary: common GARCH
   volatility or constant volatility, Student-t shocks and equal stocks.
   - Structural breaks, idiosyncratic volatility clustering and stocks of
     different volatility are not simulated.
   - A challenger whose advantage lives in a few regimes is judged on fold
     means. That is honest about concentration, but low in power.
6. **The multiple-testing budget.**
   - The pre-registered family is four challengers against B1, with Holm.
     The stepwise moves are unadjusted (the symmetric null: the fold-block
     t at 4.3–6.1%).
   - Sensitivities stay reported only.
   - This packet tried 6 procedures on 32 simulated cells. After the
     choice it ran:
     - one sensitivity (constant volatility);
     - the evaluation's size check;
     - in review, a second level (2.5%) and extra power shifts.

     All were synthetic, and no real result was seen.
   - A new challenger (for example B2's per-stock tilts), or a rerun after
     the A3 results, spends further budget and must be recorded.
7. **The simulation's own choices.** These are reasons the numbers could
   mislead.
   - The challengers are B1 placebos, and the alternatives are location
     shifts.
   - Late listings are warmed up on latent history.
   - The family's false-selection rate for the fold-block t at 5% reaches
     6.1% in one cell (N = 6, H = 21), about 3 Monte Carlo SE above 5%,
     against 5.8% for that cell's single comparison.
   - The 2.5% level was chosen after the results, so it is a post hoc
     option, not a pre-registered one.

## Owner decision

A7: adopt the fold-block t (t(F − 1) on the yearly fold means) as the test
that decides "beats B1", **at a one-sided family level of 2.5%** (Holm
across the four challengers; 2.5% for the simplest-first steps), or
alternatively at 5%. Newey-West (lag 10) and the block bootstrap would be
reported only, and B1 stays the default. Approve 2.5%, approve 5%, or
decline.

## Learning notes

- **Problem.** Decide whether a challenger's mean RPS is lower than B1's
  when the per-date differences are autocorrelated (five overlapping
  targets, slowly moving estimator state) and few.
- **HAC variance.** The variance of a mean of dependent data is the
  long-run variance over T: the sum of all autocovariances, not just γ_0.
  Newey-West estimates it with Bartlett weights up to a bandwidth M.
  - Too short an M misses persistence: the variance comes out too small,
    and the test is liberal.
  - Too long an M makes the estimate noisy.
- **Fixed-b.** With M a fixed fraction b of T, the estimate's noise does not
  vanish, and the t-ratio is not normal. Fixed-b critical values build that
  noise in. They are wider, and widen with b.
  - Example: at b = 0.1 the 5% value is 1.84, not 1.645.
  - With normal critical values, a larger M alone does not restore size.
- **Batch means.** The fold-block t is the extreme of that idea: F
  non-overlapping batch means, whose variance estimate has F − 1 degrees of
  freedom, hence t(F − 1). It ignores dependence within a fold and assumes
  little between folds.
- **Block bootstrap.** Resampling blocks of dates keeps the dependence
  inside each block, so its variance is a kernel estimate with bandwidth
  close to the block length.
  - A fixed block of 20 behaves like Newey-West at a short lag.
  - The stationary bootstrap's random block lengths give geometric weights.
  - Politis-White picks the length that balances bias against noise.
- **Size versus power.** Size is the rejection rate when the null holds;
  power is the rejection rate when it does not.
  - A test is only comparable on power once its size is right, which is why
    the criterion screens size first.
  - Here power differs little between the procedures, and size decides.
- **Ways results could mislead.**
  - A simulated size is only as good as the simulated null: the two
    placebos disagree by 1–3 pp.
  - The placebo noise is far below a real challenger's, so its power is
    optimistic.
- **Practitioner validation.** Diebold-Mariano tests with small-sample or
  fixed-b corrections, size simulations under several data-generating
  processes, and prospective confirmation.
- **Next to learn.** Lazarus, Lewis, Stock and Watson (2018) on choosing
  HAR tests; the equal-weighted cosine (EWC) estimator; Hansen's SPA test
  for several challengers.
- **Exercise.** Using `decision_tests.fixed_b_draws`, find the b at which
  the fixed-b 5% critical value reaches the t(18) value 1.734. Compare the
  resulting M with one fold (252 dates).
