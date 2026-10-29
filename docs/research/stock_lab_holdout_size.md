# How far to trust the holdout's answer: size and power of its deciding test

Status: research note, synthetic panels only (2026-10-29). It reads no data,
no store and no calendar, and it changes nothing. The registered holdout
protocol (`docs/research/stock_lab_holdout.md`, PR #130, sha256
`820d154e…856c`) is unchanged: quarterly blocks, t(6), one-sided 2.5%, one
run. The numbers below are for reading its one answer honestly. They are not
a reason to change the rule.

Record: `docs/research/stock_lab_holdout_size.json`. Code:
`scripts/research/stock_lab_holdout_size.py`. Tests:
`tests/test_stock_lab_holdout_size.py`.

## Summary

- **A "yes" can be trusted.** "Yes" means B1 beats B0. Under decision A7's
  volatility regimes the test is conservative. Suppose B1 and B0 are exactly
  equally good, with the dependence the holdout has. Then the test says "yes"
  0.6% of the time, not 2.5%. In every simulated mix shift the rate stays at
  or below 2.5% (0.6–2.4%). The rate goes above nominal only when volatility
  regimes last about two quarters: 3.4%.
- **A "no" says little.** Power is low. Take shifts that give B1 a real edge
  of 2–10 × 10⁻⁴ in RPS. The test finds them 2–11% of the time. Suppose the
  holdout drifts 20 bp a day and Increase gains about 4 pp. Then B1 beats B0
  over the 21 months in 72% of histories, but the test says so only 6% of
  the time. A pure RPS gain of 0.001 is found 29% of the time. A7 judged
  δ = 0.0005 a realistic challenger gain. That gain is found 5% of the time,
  barely above the false-positive rate.
- **Small mix shifts do not favour B1 at all.** The 252-session window makes
  B1 noisy. Its sampling noise costs about 5–6 × 10⁻⁴ in RPS. B1 overtakes
  the frozen B0 only when the cumulative shares that RPS scores, P(Decrease)
  and P(Decrease or Stable), move by roughly 2.5–3 pp or more. A 3.7 pp fall
  in Stable, spread evenly over the other two, moves each of them by only
  1.9 pp, and B1 still loses. Below the break-even, B1 is expected to lose
  to B0, and a "yes" would be a false positive.
- **Why the test is conservative.** B1 learns from each quarter's outcome
  noise and pays for it in the next quarter. Adjacent quarter means are
  therefore negatively correlated (−0.09), and the t's standard error is
  about 1.3 times too large (variance ratio 1.63). Persistent regimes push
  the other way. When they outlast a quarter, the correlation turns positive
  (+0.28), and the t becomes liberal.

## 1. What was simulated

Each replication is one synthetic history shaped like the holdout. The
design mirrors the protocol and the frozen model. Tests pin each piece to
the code the holdout runs.

| Item | Synthetic design | Matches |
|---|---|---|
| Stocks | 4 | AMD, MRVL, MU, NVDA |
| Returns | Student-t(5) shocks, pairwise correlation 0.5, common GARCH(1,1) variance: daily vol 2.5%, a = 0.06, b = 0.92 (a volatility half-life of about 34 sessions) | A7's planned panels (`planned_simulation.garch_blocks`) |
| Categories | Decrease / Stable / Increase at ±0.5% | `size_simulation.categorize`, the stable band |
| B0 | Frozen at the end of 22 development years: (counts + 50 · pooled share) / (n + 50) | `estimators.b0` at the last row (`registry._long_run`) |
| B1 | The 251 categories of the 252-session window (its first session has no return inside it), weighted λ^age with λ = 2^(−1/252), shrunk toward the frozen B0 with κ = 40 | `registry._recency_weighted` under `frozen_long_run` |
| Decisions | 443: the last development session, then every session through five before the end | `Protocol.decisions` |
| Series | Per decision, B1 − B0 RPS, averaged over the stocks and the five daily targets | the pooled daily differential (`scores.rps`) |
| Blocks | 61, 62, 64, 64, 61, 62, 69 decisions (2025Q1 … 2026Q3) | `Protocol.block_labels` on the rules calendar |
| Test | t = mean(m) / (sd(m)/√7) < −2.447 | `fold_block_t` + `one_sided_p` |

Scenarios change the law of the categories from an onset. A daily drift
moves share from Decrease to Increase. A volatility scale moves share out of
Stable. Onset "window" means the change starts with the first decision's
window (2024): the recent era differs from the long run, and B1's first
window already sees the change. Onset "holdout" means it starts at
2025-01-02, so B1 must learn it. Every scenario uses the same shocks.
Two sensitivity designs repeat the no-shift scenario. One has constant
volatility (a = b = 0). The other has longer regimes (a + b = 0.995, a
half-life of about 138 sessions).

What each column means:
- **Mix shift.** The holdout's category shares minus the frozen B0's, in
  percentage points, as Decrease / Stable / Increase.
- **True mean.** E[mean of the block means]. Negative means B1 is better on
  average.
- **B1 better (realized).** The share of histories in which B1 actually
  scored better over the 21 months.
- **Says "yes".** The test's rejection rate as registered. It is power when
  the true mean is negative, and a false-positive rate otherwise.
- **Boundary size.** The rejection rate after each history's statistic is
  recentred at the true mean. This is the false-positive rate of a world
  with the same dependence in which B1 and B0 are exactly equally good. The
  fold-block standard error does not change when a constant is added to the
  series, so the same panels give it.
- **Variance ratio.** E[se²] / Var(mean of the block means). Above 1, the
  test's se is too large (conservative).

There were 20,000 replications per design. The Monte Carlo SE of a 2.5%
rate is 0.11 pp, and of a 0.6% rate 0.05 pp.

## 2. Results

**Every scenario (nominal one-sided 2.5%)**

| Design | Scenario | Mix shift, pp (D / S / I) | True mean (×10⁻⁴) | B1 better (realized) | Says "yes" | Boundary size | Variance ratio | Adjacent-quarter corr. | Corr(mean, se) |
|---|---|---|---|---|---|---|---|---|---|
| A7 regimes | none | 0.0 / 0.0 / 0.0 | +5.28 | 12.3% | 0.06% (false positive) | 0.57% | 1.63 | −0.09 | −0.39 |
| A7 regimes | drift 5 bp | −0.9 / 0.0 / +0.9 | +4.43 | 17.6% | 0.14% (false positive) | 0.73% | 1.40 | −0.07 | −0.46 |
| A7 regimes | drift 10 bp | −1.9 / 0.0 / +1.9 | +1.89 | 32.3% | 0.66% (false positive) | 1.08% | 1.09 | −0.02 | −0.55 |
| A7 regimes | drift 15 bp | −2.8 / −0.1 / +2.8 | −2.34 | 52.5% | 2.1% (power) | 1.46% | 0.92 | +0.02 | −0.58 |
| A7 regimes | drift 20 bp | −3.7 / −0.1 / +3.8 | −8.24 | 72.0% | 5.7% (power) | 1.75% | 0.83 | +0.04 | −0.58 |
| A7 regimes | drift 30 bp | −5.4 / −0.3 / +5.7 | −25.0 | 93.9% | 22.4% (power) | 2.44% | 0.74 | +0.08 | −0.53 |
| A7 regimes | vol ×1.25 | +1.9 / −3.7 / +1.9 | +2.31 | 29.9% | 0.59% (false positive) | 1.35% | 1.17 | −0.02 | −0.41 |
| A7 regimes | vol ×1.5 | +3.1 / −6.3 / +3.1 | −3.61 | 66.5% | 5.6% (power) | 2.11% | 0.95 | +0.02 | −0.37 |
| A7 regimes | drift 20 bp and vol ×1.5 | +0.6 / −6.3 / +5.7 | −10.1 | 85.2% | 11.1% (power) | 1.92% | 0.86 | +0.03 | −0.52 |
| A7 regimes | drift 20 bp from the holdout | −3.8 / −0.1 / +4.0 | −6.72 | 68.0% | 4.3% (power) | 1.40% | 0.82 | +0.08 | −0.62 |
| A7 regimes | vol ×1.5 from the holdout | +3.3 / −6.6 / +3.3 | −2.45 | 59.7% | 3.4% (power) | 1.69% | 1.00 | +0.02 | −0.40 |
| Constant volatility | none | 0.0 / 0.0 / 0.0 | +6.34 | 7.6% | 0.01% (false positive) | 0.36% | 1.82 | −0.13 | −0.40 |
| Longer regimes | none | −0.1 / +0.2 / −0.1 | −1.34 | 42.8% | 3.8% (power) | 3.36% | 0.54 | +0.28 | −0.55 |

A drift of 20 bp a day is about 50% a year above the long-run drift. That is
large, but not unusual for these stocks in their strong years before 2025.

**Calibration: what a reported one-sided p means (no shift, recentred)**

| Nominal p threshold | t(6) critical | A7 regimes | Constant volatility | Longer regimes |
|---|---|---|---|---|
| 0.5% | −3.707 | 0.1% | 0.0% | 1.2% |
| 1% | −3.143 | 0.2% | 0.1% | 1.8% |
| 2.5% | −2.447 | 0.6% | 0.4% | 3.4% |
| 5% | −1.943 | 1.6% | 1.1% | 5.2% |
| 10% | −1.440 | 4.2% | 3.3% | 8.8% |
| Critical value giving exactly 2.5% | | −1.690 | −1.566 | −2.769 |

**Power against a pure RPS gain δ** (the no-shift dependence, recentred at −δ;
comparable with A7, whose 19 yearly folds found δ = 0.0005 36% of the time
at N = 4 and 2.5%, with twice the placebo's noise)

| δ | A7 regimes | Constant volatility | Longer regimes |
|---|---|---|---|
| 0.0005 | 5.4% | 5.7% | 5.6% |
| 0.001 | 29.0% | 34.9% | 11.4% |
| 0.002 | 86.4% | 88.5% | 51.4% |

## 3. Why the test behaves this way

Two forces act on the correlation between quarter means. One coupling acts
on the t-ratio's tails.

1. **B1's own-history learning (negative correlation).** Suppose B1 leans
   toward Decrease and a quarter brings extra Decreases. B1 gains in that
   quarter. Those outcomes also enter its window and tilt it further. The
   next quarter's outcomes do not repeat the excess, so B1 pays for the
   tilt. A good quarter is followed by a worse one. This happens even with
   constant volatility (adjacent correlation −0.13). It is the same
   asymmetry that A7's own-history placebo showed.
2. **Persistent regimes (positive correlation).** Under A7's regimes a
   volatility shock halves in about 34 sessions, well inside a quarter.
   B1 remembers a year (half-life 252), so its regime tracking barely
   carries from one quarter to the next. With a half-life of about 138
   sessions, a regime spans two or three quarters. B1's tracking then pays
   off in consecutive quarters, and the correlation turns positive (+0.28).
   This is the case the PR #130 review had in mind.
3. **Mean–se coupling.** When B1 does unusually well, its quarters are also
   unusually uneven. In those histories a strong regime or trend helps B1 in
   some quarters and not others. The estimate and its se are therefore
   negatively correlated (last column, −0.37 to −0.62). Low estimates come
   with large standard errors, so the t-ratio rarely reaches far into the
   lower tail. The estimate itself is left-skewed (skewness about −1). That
   alone would add false "yes" answers, but the coupling more than offsets
   it. This is why the large drifts stay at or below 2.5%: their variance
   ratios are below 1 (0.74 at 30 bp), yet the coupling outweighs that. In the
   longer-regime design the variance ratio is 0.54, too far below 1 for the
   coupling to offset.

## 4. Reading the one answer

- **If the answer is "yes".** In every simulated world the chance of a false
  "yes" is at most about 3.4%, and at most 2.5% unless regimes outlast
  quarters. When B1 is truly worse, as with no shift or a small one, a false
  "yes" occurs in at most 0.7% of histories. A "yes" is strong evidence that
  B1 beat B0 over this period, on these hindsight-chosen stocks, as the
  protocol's limits already say. It is not evidence of a lasting edge.
- **If the answer is "no".** "No" is the likely answer in almost every
  world simulated, including many in which B1 did beat B0 over the 21
  months. Read it as "not shown". It is not evidence that B1 is no better.
  The reported block means and the descriptive scores carry more information
  than the binary answer.
- **The p-value.** Under A7's regimes, a reported one-sided p of 2.5%
  corresponds to an actual false-positive rate of about 0.6%. A p of 10%
  corresponds to about 4%. With longer regimes the p-value is roughly right
  (2.5% → 3.4%). This is for interpretation only. The decision stays p < 2.5%.

## 5. Limits

- **The data are synthetic.** Every number depends on the data-generating
  process. Real semiconductor returns have stock-specific volatility,
  earnings jumps, regime breaks and trend changes that a common GARCH(1,1)
  with constant drift does not have. The sensitivity designs show how much
  the regime length matters: a boundary size of 0.4% with constant
  volatility, and 3.4% with longer regimes.
- **B0 is right on average in the no-shift world.** The real B0 may be
  stale in ways that are not a clean drift or volatility shift.
- **The realized shift is not known.** The power tables are conditional on
  the scenario. Which scenario the holdout resembles cannot be learned
  without reading it, and it must not be read before the one run.
- **The true mean is a Monte Carlo estimate.** Its SE is at most 1.3 × 10⁻⁵
  (Section 2's scenarios), negligible against the spread of the statistic.
- **This study was run after the protocol was approved.** It changes no rule
  and selects nothing, so it cannot be tuned toward an outcome. It is still
  one more analysis around the holdout, and that is recorded here.

## 6. Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/research/stock_lab_holdout_size.py --out docs/research/stock_lab_holdout_size.json
```

It runs in a few minutes on a laptop, uses seed 20261029 with numpy's
legacy `RandomState`, and prints the tables above. `--reps 500 --design
primary` gives a quick check.

## 7. Learning notes

**Problem.** The holdout gives one binary answer from 7 quarter means. How
often is it wrong when B1 is no better (size)? How often does it find B1 when
B1 is better (power)?

**Concept: size and power under dependence.** A t-test on B block means
assumes they are independent with a common variance σ². Suppose instead
that every pair has correlation ρ. Then

```text
Var(mean of m) = σ² (1 + (B − 1) ρ) / B,      E[sd(m)²] = σ² (1 − ρ)

variance ratio = E[se²] / Var(mean of m) = (1 − ρ) / (1 + (B − 1) ρ)
```

With B = 7, a pairwise correlation of −0.058 (A7 regimes, no shift) gives
1.058 / 0.652 = 1.62, and the simulation measured 1.63. A correlation of
+0.108 (longer regimes) gives 0.892 / 1.648 = 0.54, measured 0.54. A small
correlation matters a lot because it is multiplied by B − 1 = 6.

**Why small shifts do not favour B1 (worked example).** Take constant
volatility, where Stable is about 19%, Decrease and Increase about 40.3%
each.
- B1's window weights: W = Σ λ^j over 251 sessions = 181.5, and
  Σ λ^(2j) = 136.5, so Kish's n_eff = W² / Σ λ^(2j) = 241.
- The share B1 puts on the window is w = W / (W + κ) = 181.5 / 221.5 = 0.82.
- Each cumulative probability Q_k is noisy with variance about
  w² · p(1 − p) / n_eff = 0.672 · 0.24 / 241 = 6.7 × 10⁻⁴.
- RPS is half the sum of two squared cumulative errors, so B1's expected
  noise cost is ½ · 2 · 6.7 × 10⁻⁴ = 6.7 × 10⁻⁴. The simulation gives
  +6.3 × 10⁻⁴ with constant volatility.
- A drift that moves δ from Decrease to Increase puts both of B0's
  cumulative probabilities off by δ. B1 removes all but (1 − w) of that, so
  B1 gains about (1 − 0.18²) δ² ≈ 0.97 δ².
- Break-even: 0.97 δ² = 6.7 × 10⁻⁴ gives δ ≈ 2.6 pp. The simulation puts
  the crossing between drifts of 10 bp (1.9 pp, B1 worse) and 15 bp
  (2.8 pp, B1 better).

**Assumptions.** Synthetic GARCH(1,1) returns with a common variance;
Student-t(5) shocks; constant drift or volatility scale from an onset; B0
estimated from 22 synthetic years; no missing closes; 4 stocks always
present.

**Leakage risks.** None from data: nothing real is read. The procedural
risk is different. A study like this must not be repeated or tuned after
the holdout is read. It is fixed here, before the one run, and it changes
no rule.

**Ways the results could mislead.**
- A different regime model gives a different size. The rate ranges from
  0.4% to 3.4% across the three designs.
- "Boundary size" is the worst case at a true mean of exactly zero. Most
  plausible worlds have B1 slightly worse, where false positives are rarer.
- The power numbers assume the shift holds for the whole holdout.

**How practitioners validate this.**
- Simulate the test under several plausible processes, as here, and report
  the range rather than one number.
- Compare it with a heteroskedasticity- and autocorrelation-robust test
  (Newey-West with fixed-b critical values, or a Diebold-Mariano test).
- Check the Monte Carlo error before reading a difference.
- Pre-register the test and the level before the data are seen, as the
  protocol did.

**What to learn next.**
- Ibragimov and Müller (2010), "t-statistic based correlation and
  heterogeneity robust inference" (*Journal of Business & Economic
  Statistics*). It is the theory behind t-tests on a few group means, and
  it explains why they are often conservative.
- How estimation noise in an adaptive forecast sets a minimum detectable
  edge (the break-even above).

**Exercise.** Repeat the worked example for B1 without the window
truncation, with all its history. There W = 1 / (1 − λ) and Σ λ^(2j) =
1 / (1 − λ²). Find n_eff, w and the break-even shift. You should get about
1.6 pp. Explain why the 252-session window raises the bar B1 must clear.
