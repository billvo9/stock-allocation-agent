# A7 decision packet: which test decides "a challenger beats B1"

Status: pre-registration written 2026-10-17, before any simulation in this
packet was run. Results: pending.

B1 stays the Stock Lab's default throughout. This packet changes no rule,
default or stored record. It gives the owner the evidence for decision A7
(`docs/research/stock_probability_lab.md`, section 7).

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
