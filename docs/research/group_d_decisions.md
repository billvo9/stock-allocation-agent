# Group D: performance statistics, uncertainty and reporting (decision packet, 2026-10-17)

**Status: every decision here is PENDING OWNER APPROVAL.** Nothing in this packet is implemented. No CAGR, volatility, Sharpe, interval, participation, multiple-testing or reporting formula changes in code until the owner approves the relevant decision, and the approval is recorded with its date and quotation here, in `t5_portfolio_formulas.md` section 9 and in `docs/PROJECT_STATE.md`.

**Why now.** The owner's 2026-10-17 request: make future real-data comparisons of PPO against the baselines defensible. The synthetic comparison (`ppo_comparison.md`) reported no group D statistic because group D is unapproved: no CAGR, no Sharpe ratio, no interval and no test. A real comparison needs all of them, decided before any real result exists, so that no statistic is chosen after seeing which one flatters a strategy.

**How to read it.**
- Decision ids M1–M9 are new ("M" for metrics; D1–D8 remain the T5a ids, and group C's decisions are C1–C9 in `group_c_decisions.md`). The combined index with a suggested reply is `group_c_d_decision_index.md`.
- Formula ids (F2.4, F6.x, F7.3) refer to `t5_portfolio_formulas.md`. Each decision gives the rule, the exact equation, a worked example, the options and a recommendation, as in `t5a_open_decisions.md`.
- **Every number in bold** is computed by `scripts/group_cd_packet_numbers.py` (keys `M1.*` to `M9.*`), and so is every number in the tables. `tests/test_group_cd_packet_numbers.py` checks both directions: every computed value appears here, and every bold number here is a computed value. Numbers not in bold are stated inputs or assumptions, restatements of a computed number, or figures quoted from approved documents. The interval example reuses T2's existing bootstrap code (`model_diagnostics/inference.py`).
- **Every power, sample-size and detection figure here is a planning figure** (risk register STAT-16): computed under stated i.i.d. assumptions, with its finite-sample behaviour not yet checked by simulation.
- Evidence tags: **[A]** an assumption to approve; **[I]** from memory, unverified.

**Already decided, and not asked again.** F6.5 (drawdown, its duration and dollar size; A6) with R5c (incomplete histories show only the "largest observed decline"); F6.6 (calendar-year returns, partial years labelled; A7), including its rule that CAGR is never divided by volatility; OD8.13 ("Before taxes and transaction costs"; A8); F2.4's one-way turnover (approved in group B; M6 asks only how "annualized per calendar year" is read); F7.2 (rebalance offsets are one variant with a k-member sensitivity family; D7); D8 (a) (the PPO reward is log(1 + R_net)). The PRD's section 15 sets the direction for a learned policy (an intersection-union test against each baseline, "tested by date with fold-block inference", seeds reported as a distribution, a deflated Sharpe ledger); it is approved direction with details under review. Where a recommendation here departs from its wording (M3's headline interval; M8's effective-trial rule and DSR denominator) it says so, and so does every non-recommended option that would (M9's cash contrast).

**Notation.** R_t is a strategy's net return for the span ending at close t (from `portfolio_step`); rf_t the risk-free return for the same span (C3); e_t = R_t − rf_t; n the number of return sessions; D the calendar days of the window; b the block length and k = ⌈n/b⌉ the blocks per bootstrap replicate.

---

## M1. CAGR on calendar days (OD8.8; F6.1)

**Rule.**

```
CAGR = G^(365.25 / D) − 1,      G = W_end / W_0
```

- W_0 is the initial wealth **before** the build cost at the common first execution close, so the build (D3) is inside G. D is the calendar days from that close to the last valued close.
- **A window shorter than 365 days is not annualized.** Its period return is shown, labelled partial, as F6.6 already does for calendar years (the GIPS convention for returns under one year [I]).

**Examples.**
- G = 1.10 over 365 days and 248 sessions: **0.1000718** on calendar days. The current session basis, G^(252/248) − 1, gives **0.1016923**.
- First execution close 2021-01-04, last close 2022-12-30 (**725** days), +21% after a 5 bp build: CAGR **10.052%**. Ignoring the build would show **10.080%**.
- +5% over 182 days is shown as "+5.00%, partial". Annualizing it would show **10.29%**.

**Options.** Sessions/252 (current: wrong in 250- or 253-session years and across closures); **calendar days/365.25 (recommended)**; calendar days/365.

**Recommend:** calendar days/365.25, the build inside G, no annualization under one year.

---

## M2. Volatility and Sharpe ratio (OD8.8; F6.2)

**Rule.**

```
σ_ann = √252 × sd(R_t)                       sd with ddof = 1, over per-session net returns
SR    = √252 × mean(e_t) / sd(e_t),           e_t = R_t − rf_t   (rf from C3)
```

- Arithmetic (simple) returns. √252 whatever the year's session count: 2023 had 250 sessions, and √250/√252 − 1 = **−0.40%**, a negligible difference.
- **Zero variance is decided with a tolerance:** sd(e_t) ≤ 1e-12 per session [A] gives `unavailable` (`zero_variance`). An exact `> 0` test would not do: a 100% cash portfolio's excess returns are floating-point residue of about 1e-16 (C3), and their ratio would be a meaningless Sharpe ratio.
- **A path with a multi-session span** (a D6 gap; a sub-question):
  - (i) **recommended:** σ and SR over the single-session returns only, labelled "incomplete: n of N sessions", with no interval. This mirrors R5c, which shows drawdown over an incomplete history as a labelled "largest observed decline" rather than nothing. The omitted spans' moves can make σ too low, which the label must say;
  - (ii) `unavailable` (`incomplete_history`).

  Either way, every interval needs a per-session series, so **no interval survives a gap**, including the primary estimand's (M3, M9): its point estimate (total log growth, and CAGR from the valued endpoints) stays exact. A comparison's universe should therefore be one whose funds have no missing closes in the window; C8's registration checks that.
- Not proposed: Sortino, Calmar and the information ratio. They are reported only if approved later (the paired differences in M3 carry the same information as an information ratio).

**Examples.**
- A per-session Sharpe ratio of 0.05 is **0.794** a year. Over ten years (n = 2,520), its i.i.d. standard error (Lo 2002) is **0.316**, so the i.i.d. 95% interval is **[0.17, 1.41]**. Ten years of daily data cannot pin down a Sharpe ratio better than about ±**0.62**.
- Friday → Monday, a portfolio returns 0.60% and the risk-free return at 5% is **0.0411%** (three days), so the excess return is **0.5589%**. A weekend accrues in both terms.

**Options.** **√252 per session (recommended)**; Lo's autocorrelation-adjusted annualization (one more estimated quantity); calendar-day scaling (mixes the weekend accrual into the volatility scale).

**Recommend:** as stated, with excess returns over C3.

---

## M3. Intervals: the block bootstrap with SE and t (OD8.9; F6.3, F6.4)

**Rule (the method family).**
- Draws: T2's circular block bootstrap (`draw_circular_blocks`), blocks of b = 63 sessions, 999 replicates and a registered seed. **The same draws are used for every strategy in a comparison**, so every difference is paired.
- Interval: **estimate ± t_{k−1, 0.975} × bootstrap SE**, with k = ⌈n/b⌉. Percentile intervals are not used: they under-cover badly with few blocks (`t5_portfolio_formulas.md`, section 6).
- **CAGR:** the interval is built on mean log growth ḡ = mean(log(1 + R_t)) with `bootstrap_mean`, then mapped to CAGR endpoints by exp(ḡ × n × 365.25/D) − 1, with the calendar span held fixed.
- **Sharpe:** the SE is the standard deviation of `bootstrap_statistic`'s replicates, used with the same t rule.
- **Paired comparisons:** the difference in annualized log growth, (ḡ_a − ḡ_b) × n × 365.25/D, and the Sharpe difference, both on the same draws.
- At least **8 blocks** (504 sessions); otherwise `unavailable` (`too_few_blocks`).
- **One-sided p-values** (M8): `bootstrap_mean` returns a two-sided p₂. The one-sided p for the claimed direction is p₂/2 when the estimate has the claimed sign, else 1 − p₂/2.
- **Beside the PRD's fold-block test.** The PRD (section 15) says the primary estimand is "tested by date with fold-block inference". M3 makes the block bootstrap the headline and reports T2's `fold_block_t` over the walk-forward test folds beside it, as T2 already reports several estimates side by side. The reason: a real test window holds only a few walk-forward folds (a t with 3 or 4 degrees of freedom), against 20 bootstrap blocks in five years. **This departs from the PRD's wording and needs the owner's confirmation** (option below).
- Labels: every interval is "nominal 95%". Lo's i.i.d. SE is reported only as a sanity check.
- **Provisional settings:** b = 63, 999 replicates, the 8-block minimum and the t·SE method stay provisional until M4's registered coverage simulation fixes them. Approving M3 approves the method family.

**Example (synthetic, seed 20261017).** 1,260 sessions of i.i.d. normal log returns (mean 0.0003, sd 0.01) over **1,830** calendar days, so **251.5** sessions per calendar year; **20** blocks of 63, so t = **2.093**. The true values built into the series are shown for comparison.
- **CAGR:** **8.14%**, interval **[−4.28%, 22.17%]** (true **7.84%**).
- **Sharpe:** **0.58**, bootstrap SE **0.371**, interval **[−0.20, 1.35]** (true **0.56**). The percentile interval would be [−0.16, 1.32]; Lo's i.i.d. SE is **0.447**.
  - The bootstrap SE (0.371) differs from the i.i.d. SE (0.447) by sampling noise alone: with 20 blocks, an SE estimate is itself uncertain by roughly 1/√(2 × 19) ≈ **16%**. That is why the t critical value has k − 1 degrees of freedom, and why M4 must check coverage before the settings are fixed.
- **Paired difference** against a second series correlated 0.9 with the first: **8.04%** a year, paired SE **3.03%**, interval **[1.70%, 14.38%]** (true **2.51%**). Treating the two series as independent would give an SE of **7.90%**: pairing on the same draws is what makes a comparison informative.
- The t critical value is **2.365** at the 8-block minimum, and **1.994** at 72 blocks (a window starting with SHV in 2007, 4,527 return sessions).

**Options.** No intervals; i.i.d. (Lo) only; percentile block intervals (under-cover with few blocks); **block bootstrap SE + t as the headline, with `fold_block_t` beside it (recommended)**; `fold_block_t` as the headline, as the PRD words it (few degrees of freedom on one test window); the stationary bootstrap (random block lengths; similar, and a second code path); HAC standard errors; Ledoit and Wolf (2008), a studentized bootstrap for the Sharpe difference specifically (a candidate for M4 to compare against, not a default).

**Recommend:** the method family as stated, with the settings provisional until M4.

**Implementation note (after approval).** `bootstrap_statistic` currently returns percentile intervals (`percentile_uncalibrated`). M3 uses its replicates' SE with the t rule; that is a T5c code change.

---

## M4. The registered coverage simulation that M3 requires

**Rule.** Before any interval is described as more than nominal, and before M3's settings are fixed, T5c runs one registered simulation. Its code, seeds, design and acceptance rule are recorded in G1 before it runs.

- **Data-generating processes**, with parameters fitted to pre-2025 development data only:
  1. i.i.d. normal;
  2. i.i.d. Student-t with 4 degrees of freedom (heavy tails);
  3. GARCH(1,1) with t innovations (volatility clustering);
  4. two-state Markov regime switching, with crisis regimes of about 350 sessions, like 2008;
  5. long memory in volatility (for example a fractionally integrated log-volatility, d ≈ 0.3) [A].
- **Sample sizes:** n ∈ {504, 1,260, 2,520, 4,527}: the 8-block minimum, a five-year test window, ten years, and the SHV window.
- **Statistics:** mean log growth (and its CAGR mapping), the Sharpe ratio, and the paired differences in log growth and in Sharpe for two strategies correlated 0.5 and 0.9.
- **Methods compared:** SE + t (the candidate), percentile (reference) and Lo's i.i.d. SE (reference); block lengths b ∈ {21, 63, 126}.
- **Cells:** 5 processes × 4 sample sizes × 6 statistics (mean log growth, Sharpe, and the two paired differences at two correlations) = **120** cells per method and block length.
- **Replications: 5,000 per cell** [A], each with 999 bootstrap replicates. The Monte Carlo 95% half-width of a coverage estimate near 95% is **0.60 pp** at 5,000 replications (**0.96 pp** at 2,000; **1.35 pp** at 1,000).
- **Acceptance (registered in advance):** the candidate's coverage of nominal 95% intervals is at least 93% in **every** cell with at least 8 blocks. The requirement is joint, so per-cell pass rates overstate how easily it is met:

| True coverage in every cell | Per cell, 2,000 reps | All 120 cells, 2,000 reps | Per cell, 5,000 reps | All 120 cells, 5,000 reps |
|---|---|---|---|---|
| 95% | 100.0% | 99.5% | 100.0% | 100.0% |
| 94% | 97.1% | 2.9% | 99.8% | 82.2% |
| 92% | 5.2% | 0.0% | 0.4% | 0.0% |

  - **How the table is computed.** Per cell: the exact binomial probability that the observed coverage reaches 93%. All cells: the product of the 120 per-cell values, which **assumes the cells are independent**. They are not: the statistics of one process and sample size are computed on the same simulated paths, so the joint values are indicative only. A normal approximation instead of the exact binomial gives 84.0% and 2.6% for the two 94% entries.
  - At 2,000 replications a method with a true 94% coverage would almost surely fail somewhere by Monte Carlo noise alone; at 5,000 it passes about four times in five, while a method at 92% still fails. That is why 5,000 is recommended.
- **Compute budget.** 120 cells × 5,000 replications × 999 bootstrap replicates × 3 block lengths = **1,798,200,000** bootstrap statistics; the three methods reuse each cell's replicates. T2's block sums (`resampled_sums`) make each statistic a few array operations, but the run is long, which is one more reason to run it once, as registered.
- **A failing cell.** It is reported, and a setting may change only by the registered rule (for example the smallest block length that passes everywhere), never by looking at real-data results. Real data cannot be assigned to one simulated process, so **if any process fails, every real-data interval carries the label** "under-covers in simulation under <process> dependence", not only some of them.

**Options.**
- **(a) The design above (recommended).**
- (b) A smaller design (i.i.d. and GARCH only, as in the exploratory run), or 2,000 replications. The first misses the regime and long-memory cases that drive real drawdowns; the second makes the joint acceptance mostly a test of Monte Carlo noise.
- (c) No simulation: intervals stay "nominal, unvalidated" permanently.

**Recommend: (a).**

---

## M5. Drawdown in comparisons, and registered stress windows (F6.5, approved; reuse)

**Rule.**
- F6.5 as approved (A6), with R5c: the maximum drawdown as a point estimate, its duration in sessions and calendar days or "not recovered", its dollar size on $10,000, no interval, and a close-to-close label.
- **New here (a proposal [A]):** three stress windows, registered before any strategy is run, reported beside it for every strategy and benchmark: the return and the maximum drawdown inside each window. The proposed dates are the S&P 500's peak and trough closes [I], fixed by market history, not by any strategy's results, and to be checked against index data before registration:

| Window | Start (peak) | End (trough) | Sessions |
|---|---|---|---|
| Global financial crisis | 2007-10-09 | 2009-03-09 | **355** |
| COVID crash | 2020-02-19 | 2020-03-23 | **23** |
| 2022 rate shock | 2022-01-03 | 2022-10-12 | **195** |

A window outside a comparison's registered window is reported as `outside_window`, never shortened.

**Example (approved F6.5).** Wealth 100, 110, 99, 105, 111 gives a maximum drawdown of 10.0% and a duration of 3 sessions; on a $10,000 start, the worst decline is $1,100.

**Options.**
- **(a) F6.5 plus the three proposed peak-to-trough windows (recommended).** Fixed dates, comparable across comparisons.
- (b) F6.5 plus windows defined another way:
  - calendar years 2008, 2020 and 2022: no choice of peak or trough, but they dilute each crash with its recovery;
  - NBER recession dates (2007-12 to 2009-06; 2020-02 to 2020-04) [I]: official, but monthly and partly after the market trough;
  - a rule: the registered benchmark's k largest drawdown episodes inside the comparison's window (by F6.5), for example k = 3 with a depth of at least 10% [A]. It adapts to any window, but the windows then differ between comparisons.
- (c) F6.5 alone.
- (d) A bootstrap interval for maximum drawdown (rejected in F6.5: drawdown depends on episodes longer than a block).

**Recommend: (a).**

---

## M6. Turnover "annualized per calendar year" (F2.4, approved: the reading to confirm)

**Rule.** F2.4's one-way turnover τ = ½Σ|Δ_i| per execution (a full switch is 1) is reported as follows:
1. **Construction turnover** is reported separately. Construction ends at the first execution that reaches the build's own target, **or at the first execution of a different target, whichever comes first**. Without that bound, a policy that changes its target at every decision under a turnover limit would never "reach its target", all its trading would count as construction, and its rebalance turnover would read about 0. (The synthetic comparison's measure has no such bound; aligning it is a follow-up, not part of this decision.)
2. **Rebalance turnover** is summed within each calendar year. A partial first or last year is labelled partial and never scaled (as F6.6 does for returns).
3. The headline **"annual turnover"** is the mean of the full calendar years' rebalance sums.
4. Traded notional is reported the same way.

Synthetic runs have no calendar dates, so they keep the comparison's current practice: nothing annualized.

**Example.** A portfolio is built on 2021-03-01 (construction turnover 1.0). Rebalance turnover sums to **0.25** in 2021 (partial), **0.47** in 2022, **0.38** in 2023 and **0.10** in 2024 (partial, to 2024-06-28).
- Headline: **0.425** a year, the mean of 2022 and 2023.
- Scaling the whole **1,215**-day span instead gives **0.361**, and **0.661** with the build included, which counts a one-off purchase as recurring trading.

**Options.** **(a) Calendar-year sums and the mean of full years (recommended)**; (b) total × 365.25/D (mixes partial years and the build); (c) no annualization.

**Recommend: (a).** This confirms a reading of an approved formula; it is not a new formula.

---

## M7. The participation flag (OD8.11; F7.3)

**Rule.**

```
q_i = |Δ_i| × N × W⁻ / MDV_i            flag when q_i > p = 1% [A]
MDV_i = median over the 20 sessions ending at decision session d of (raw close × raw volume)
```

- Δ_i is the executed F2.1 trade, from the drifted weight to the target. W⁻ is pre-trade wealth per 1.0 of initial cash.
- **New here: N, a registered notional portfolio size in dollars** [A], because backtests run per 1.0 of cash. Recommended default: N = $1,000,000.
- The build is a separate full purchase, reported separately from rebalances. There is no capacity figure.
- Fewer than 20 volume sessions gives `unavailable`. MDV never includes d + 1 (T7.5). Close × volume is unchanged by a split.

**Examples (N = $1,000,000, MDV = $20m).**
- A rebalance from a drifted 45% to a 30% target trades $150,000: q = **0.75%**, not flagged. Measuring holdings instead (30% of $1m) would give **1.50%** and flag it wrongly.
- Building the same 30% position from cash buys $300,000: q = **1.50%**, flagged, and reported under the build.
- The same rebalance at N = $100,000 gives **0.075%**; at N = $10,000,000, **7.5%**. The flag scales linearly with N, which is why N must be registered.

**Data.** Needs daily volume (F1's nullable `volume`; Tiingo supplies it [I]). An ETF's exchange volume understates what it can absorb through creations and redemptions, so the flag is conservative for funds.

**Options.** **(a) The flag on traded notional at a registered N (recommended)**; (b) holdings-based (overstates rebalances); (c) a participation cap (changes the path; deferred); (d) no flag.

**Recommend: (a)**, with N = $1,000,000 unless the owner prefers another planning size.

---

## M8. Multiple testing, fed by the variant ledger

**Rule.**
1. **The primary claim** (PRD section 15): the policy beats **each** registered baseline, on the primary estimand only (the paired difference in annualized log growth, M9). It is an intersection-union test: every paired one-sided test at α = 0.025 (one-sided p from M3), and the claim holds only if all of them reject. It needs no multiplicity adjustment: if any baseline is in fact not beaten, the chance that all tests reject is at most that baseline's own α.
2. **Per-baseline statements, and the overall error rate (a sub-question).** When the intersection-union test rejects, the primary claim holds, and with it each of its components ("beats baseline j"). What happens when it fails decides the overall chance of a false claim:
   - **(i) No fallback claims (recommended).** No claim is made against any single baseline; the paired intervals are still shown, as estimates. The chance of any false claim is the intersection-union test's alone: at most α, by construction.
   - (ii) A Holm fallback: a statement about one baseline ("PPO beats inverse volatility") is made if it survives Holm's adjustment at one-sided 0.025. The two branches are disjoint, but their error rates add: at most **0.050** (2α) by the union bound, and at most **0.025625** (α + α²) when the p-values are independent; **0.0375** (1.5α) is reachable when they are negatively dependent (two baselines that are in fact not beaten). The report would have to state that bound.
   - (iii) Split α: the intersection-union test at α/2 and Holm at α/2. At most α by the union bound, with less power for both.
   - (iv) Holm alone, with "beats every baseline" only when Holm rejects every contrast. At most α, with less power for the primary claim than the intersection-union test.
3. **A headline Sharpe ratio** of a strategy chosen among several registrations evaluated on the same window carries the deflated Sharpe ratio (DSR), with N the effective number of trials from the ledger. A DSR below **0.975** (the one-sided 2.5% level the PRD uses throughout) is labelled "not distinguishable from selection among N trials". How the DSR handles dependence is a sub-question (below the equations).
4. **The effective-trial rule** (risk register STAT-5), a choice for the owner. Recommended:
   - a trial is a registered configuration whose test-window results were looked at;
   - a recorded repeat look that reproduces its previous look exactly adds no trial (like the synthetic comparison's look 2); a repeat that changed anything does;
   - the seeds of one registration count as one trial when every seed is reported and none is selected;
   - the k rebalance offsets count as one (F7.2, approved);
   - development-fold checkpoint selection (110 PPO checkpoints in the synthetic comparison) is not a trial on the test window, but it is recorded, and it counts in full if any test-window information reached it;
   - looks made before the ledger existed are counted as "≥ n" (STAT-5), never as exactly n.

   The alternative is the PRD's literal wording, "Seed, hyperparameter and checkpoint counts go into the ledger for the deflated Sharpe ratio": every seed, hyperparameter setting and checkpoint compared counts as a trial. Because the test window protects development-fold selection, that count is a conservative bound rather than the effective N. **Recommended: the headline DSR at the effective N, with the DSR at the full count shown beside it as a bound.**
5. **V**, the variance of the trials' Sharpe ratios in the DSR: the sample variance when at least five trials' Sharpe ratios are stored, otherwise the pure-noise approximation 1/n per session.
6. **Deferred:** White's Reality Check, Hansen's SPA test and the probability of backtest overfitting. They need every variant's return path stored.

**Equations** (Bailey and López de Prado 2014; per-session units).

```
DSR = Φ( (SR̂ − SR₀) √(n − 1) / √(1 − γ₃·SR̂ + (γ₄ − 1)/4 · SR̂²) )
SR₀ = √V × [ (1 − γ) Φ⁻¹(1 − 1/N) + γ Φ⁻¹(1 − 1/(N·e)) ],   γ ≈ 0.5772 (Euler's constant);  SR₀ = 0 when N = 1
```

γ₃ is skewness and γ₄ the ordinary kurtosis (3 for a normal). pandas `kurt()` returns excess kurtosis, so 3 must be added: an easy silent bug.

**Dependence (a sub-question).** The √(n − 1) term, and the variance formula under it, assume serially independent returns, which M3 rejects (daily returns cluster in volatility). Options:
- (i) The published formula with n, shown labelled "assumes independent returns".
- (ii) n replaced by an effective sample size n_eff from T2's `effective_sample_size` (Bartlett weights). It keeps the published formula, but it corrects only for autocorrelation in returns, which is small for daily returns, not for volatility clustering.
- (iii) **Recommended:** the denominator replaced by M3's block-bootstrap standard error of the Sharpe ratio (in per-session units), and V's noise approximation by that standard error squared. This adapts the published method so that it carries M3's dependence handling. The published value (i) is shown beside it.

**Examples.**
- **DSR.** An annual Sharpe ratio of 0.80 over 1,260 sessions, skewness −0.5, kurtosis 6, trials that were pure noise (V = 1/n):

| Trials N | SR₀ (annualized) | DSR |
|---|---|---|
| 1 | **0.00** | **0.961** |
| 2 | **0.23** | **0.895** |
| 10 | **0.70** | **0.584** |
| 100 | **1.13** | **0.232** |

  With ten registrations tried on one five-year window, the best of ten trials without skill is expected to reach a Sharpe ratio of about 0.70, so an observed 0.80 is weak evidence of skill (DSR 0.584, far below 0.975). Even a single trial reaches only 0.961.
- **Holm and the intersection-union test.** One-sided p-values against four baselines of 0.010, 0.020, 0.030 and 0.200:
  - read unadjusted at 0.025, they claim **2** wins;
  - Holm's first threshold is 0.025/4 = **0.00625**, so it makes **0** rejections;
  - the primary claim fails too: two p-values exceed 0.025, so under (i) no claim is made against any baseline;
  - with p = 0.02 against each of four baselines, every test rejects, so the report says "beats each of the four baselines", even though Holm alone, starting at 0.00625, would support no single-baseline claim. The two procedures answer different questions; (i) keeps only one of them.

**Options.**
- **(a) The intersection-union test for the primary claim with no fallback claims (i), and the DSR for headline Sharpe ratios with the bootstrap standard error (iii), at the effective N, with the full-count DSR beside it (recommended).**
- (b) The DSR only.
- (c) Bonferroni (valid, more conservative than Holm).
- (d) The Reality Check or SPA now (needs every variant's stored path).
- (e) Counts only (today).

**Recommend: (a).**

---

## M9. What a real PPO-versus-baseline comparison must report

**Which sessions.** A learned policy's statistics use only sessions outside its training and selection folds: its test fold. Every baseline and benchmark it is compared with is reported over **the same sessions**, on the same bootstrap draws. Full-window figures (for example the 18 years from 2007, or the 2007–09 stress window) are in-sample for a policy trained on them: they are reported for rule-based strategies only, or labelled "in-sample" for a learned one.

**Rule (the reporting contract).** For each strategy, each registered constraint set and each cost multiplier (0×, 1×, 2×):
- terminal wealth and total return; CAGR (M1) with its interval (M3); calendar-year returns (F6.6); volatility and the Sharpe ratio (M2) with its interval;
- maximum drawdown, its duration and dollar size, and the stress windows (M5);
- construction and calendar-year turnover (M6), costs paid, participation flags (M7), the build report (D5), constraint interventions and fallbacks, and the offset family's max − min spread (F7.2);
- the labels:
  - "Before taxes; after modelled transaction costs (provisional D2 rates)" on every simulated net return. A8's "Before taxes and transaction costs" remains the label of descriptive fund histories, which pay no costs; a backtest net of D2 costs needs the different wording (a label proposal, part of this decision);
  - "Development window, to 2024-12-31";
  - "Non-confirmatory" on every test result (PRD section 15, stage S1: development evidence is not confirmation).

Against each registered baseline, the benchmark **and the C1 cash account**, on the same bootstrap draws:
- **the primary estimand**: the paired difference in annualized log growth, the certainty equivalent of a log-utility investor and the D8 (a) objective, with its interval, a one-sided p-value, and the **minimum detectable effect** for that window beside it (STAT-1);
- the paired Sharpe difference with its interval;
- the intersection-union result, any per-baseline claims M8 allows, the DSR for any headline Sharpe ratio, and the ledger's trial counts (M8).

**Value against cash (a sub-question).** The PRD (section 15) says any claim of value, and any scaling up, needs both superiority over every registered baseline (the intersection-union test) and **positive value against cash**: the net certainty-equivalent return minus the T-bill return over the same dates, "judged on the point estimate and its interval", a necessary condition rather than a test.
- **(i) Follow the PRD (recommended).** A value claim is made only when the intersection-union test rejects **and** the paired difference in annualized log growth against the C1 cash account has a positive point estimate; its interval is shown beside it. Without both, the report makes no value claim.
- (ii) Add the cash account to the intersection-union baselines. That turns the PRD's necessary condition into a test, which is far stricter: showing a Sharpe ratio of 0.5 over cash takes about 31.4 years (below).
- (iii) Report the cash contrast without making it a condition. This was the first draft of this packet; it **departs from the PRD**.

**Learned policies and seeds.** The headline is a seed ensemble declared before the run (PRD section 15). Every seed is also reported, with the spread (min, quartiles, max); the best seed is never shown alone (with five seeds, the best beats the seed mean by about **1.163** seed standard deviations from selection alone). How the ensemble is formed is a sub-question:
- **(i) Averaged targets (recommended).** At each decision the ensemble's target is the mean of the seeds' targets. It runs as one registered policy through the same feasibility map (D5) and `portfolio_step`: one account, netted trades, one constraint application. That is what would be deployed.
- (ii) Separate sub-accounts, equal capital, no rebalancing between them: the ensemble's wealth is the mean of the seeds' wealth paths. Simple, but it pays each seed's trades separately, so it overstates costs against (i).
- (iii) The median seed (selection-free, but not a portfolio anyone holds), or per-seed results only (no headline).

**How uncertainty is shown.**
- Every headline estimate appears as "estimate [low, high], nominal 95%, block bootstrap", or as `unavailable` with its reason.
- A difference between strategies is shown only as a **paired** interval. Two overlapping marginal intervals are not a test, and two separate intervals hide the pairing (M3's example: SE 3.03% paired against 7.90% unpaired).
- No stars and no sorting by results: strategies appear in registered order.
- "No difference detected" is never reworded as "equivalent". Equivalence needs a pre-registered non-inferiority margin (PRD section 15).

**Minimum sample size and power (planning figures; i.i.d. normal returns; one-sided α = 0.025, the PRD's confirmatory level, which equals two-sided 0.05; power 80%; z_{0.975} + z_{0.80} = 2.8016).**

- **A Sharpe ratio against zero.** Years of daily data needed: (2.8016/SR)², so **87.2** years at SR 0.3, **31.4** at 0.5 and **7.8** at 1.0. The same formula gives **31.4** years for an information ratio of 0.5 against a baseline.
- **The smallest detectable Sharpe difference** between two strategies whose returns correlate ρ: (t_{k−1, 0.975} + t_{k−1, 0.80}) × √(2(1 − ρ)/T), with k = 4T blocks of 63 sessions, matching M3's t rule (Jobson–Korkie with Memmel's correction; the SR² terms are negligible with daily data):

| ρ | 3 years | 5 years | 10 years | 18 years |
|---|---|---|---|---|
| 0.5 | 1.78 | 1.32 | 0.91 | 0.67 |
| 0.8 | 1.12 | 0.84 | 0.57 | 0.42 |
| 0.9 | 0.79 | 0.59 | 0.41 | 0.30 |
| 0.95 | 0.56 | 0.42 | 0.29 | 0.21 |

  The t version is wider than a normal-quantile figure because a 3-year window has only 12 blocks. With normal quantiles (2.8016), ρ = 0.9 gives **0.723** over 3 years and **0.396** over 10.
  - **The PRD's planning figures differ slightly.** It gives 0.71 and 0.39. Those match **0.712** and **0.390**: the values obtained by counting 260 sessions a year (every weekday) while annualizing with √252, which understates the detectable difference by about 1.6%.
  - **Proposed PRD wording** (not edited here; the PRD is the owner's document): "about 0.40 over 10 years and 0.72 over 3 years (normal quantiles, 252 sessions a year; about 0.41 and 0.79 with t quantiles over 63-session blocks)".
- **The smallest detectable difference in annualized log growth**, for a tracking error TE between the two strategies: (t_{k−1, 0.975} + t_{k−1, 0.80}) × TE/√T, with the same k (t quantiles with k − 1 degrees of freedom stand in for both the test and the power):

| TE | 3 years | 5 years | 10 years | 18 years |
|---|---|---|---|---|
| 2% | 3.6% | 2.6% | 1.8% | 1.3% |
| 5% | 8.9% | 6.6% | 4.5% | 3.3% |
| 10% | 17.8% | 13.2% | 9.1% | 6.7% |

- **The data available.** A universe including SHV has 4,527 return sessions to 2024-12-31 (C8), about 18 years, before any warm-up or training split; that span is available in full only to rule-based strategies. A five-year test window, 2020-01-02 to 2024-12-31, has **1,257** return sessions: **20** blocks of 63.
- **Seeds.** The expected best of k seeds exceeds their mean by **0.564** (k = 2), **1.163** (5), **1.539** (10) and **1.867** (20) seed standard deviations, from selection alone.

**What this means.** On a five-year real test window, a policy whose returns correlate 0.9 with a baseline must beat it by about 0.59 in Sharpe ratio, or by about 6.6% a year in log growth at a 5% tracking error, to be detected with 80% power. Realistic edges are smaller. So a real comparison is an **estimation exercise with intervals**, not a ranking: a non-detection is the expected outcome, and the report says so in advance. Dependence beyond the i.i.d. assumptions (volatility clustering, regimes) widens every figure above.

**Options.** **(a) The contract as stated, with value against cash as the PRD states it (i) and averaged targets as the seed ensemble (recommended)**; (b) point estimates only (today's synthetic practice, which the owner's request rules out for real data); (c) the contract with sub-accounts or the median seed as the headline, or with the cash contrast as a test (ii) or as context only (iii).

**Recommend: (a).**

---

## These follow automatically once M1–M9 are approved

- The spec's tests T6.1–T6.9 and T7.1–T7.5, and the M4 simulation as its own registered run.
- The baseline re-run with T5c's one-change-at-a-time attribution (lag → cost base → build → cash rate → benchmark basis → CAGR basis).
- The Lab's comparison panel can show intervals and paired differences, with the labels above.

## Not in this packet

- Group C inputs: the cash rate, the risk-free rate and benchmarks (`group_c_decisions.md`). M2 and M3 need C3's risk-free rate before any real excess return exists.
- The PRD's confirmatory design (α-spending, futility, N_max) for prospective evidence. M8 covers development-window comparisons only, which are non-confirmatory.
- The Reality Check, SPA and the probability of backtest overfitting (M8, deferred).

## Learning notes

1. **Problem.** Report how well each strategy did, how sure we can be, and whether a difference between two strategies is more than noise or selection.
2. **Concepts.**
   - **Annualization.** CAGR compounds over calendar time; volatility scales with √(sessions); mixing the two (CAGR/σ) mixes units.
   - **Dependent-data inference.** Daily returns cluster in volatility, so i.i.d. formulas understate uncertainty; resampling whole blocks of sessions keeps the dependence inside each block.
   - **Pairing.** Two strategies on the same days share market noise; differencing first removes it (M3: SE 3.03% against 7.90%).
   - **Selection.** The best of N noisy trials looks good by construction; the deflated Sharpe ratio asks how good the best of N would look if none had skill.
3. **Assumptions.** Dependence beyond 63 sessions is weak (M4 tests this); the variance of trial Sharpe ratios is known or approximated; one historical path.
4. **Formulation.** CAGR = G^(365.25/D) − 1; SR = √252·mean(e)/sd(e); interval = estimate ± t_{k−1}·SE_boot; DSR = Φ((SR̂ − SR₀)√(n − 1)/√(1 − γ₃SR̂ + (γ₄ − 1)SR̂²/4)).
5. **Example.** M3's synthetic series: a true CAGR of 7.84% estimated as 8.14% with an interval of [−4.28%, 22.17%]: five years of daily data say little about a growth rate.
6. **Leakage.** A block length, a stress window, a notional N or a headline statistic chosen after seeing results; a test window used to pick among registrations without counting it as a trial.
7. **Ways results could mislead.** Unpaired intervals hide real differences; percentile intervals from few blocks are too narrow; a single seed or a single offset is one draw; a non-significant difference is not evidence of equivalence; an undeflated Sharpe ratio after many trials.
8. **How practitioners validate.** Coverage simulations of the interval method under realistic dependence; Ledoit and Wolf's studentized bootstrap for Sharpe differences; deflated Sharpe ratios and reality checks against the full trial ledger; pre-registration.
9. **Next to learn.** The studentized block bootstrap; the intersection-union principle; Bailey and López de Prado on the probability of backtest overfitting.

**Bounded exercise (optional, after approval).** Using `draw_circular_blocks` and `bootstrap_mean`, reproduce M3's paired SE (3.03%) and unpaired SE (7.90%) from the seed and inputs stated there, then explain in two sentences why they differ.

**Suggested reply:** "Approve M1 as stated; M2 with (i) for incomplete histories; M3 with `fold_block_t` beside the bootstrap; M4 (a) with 5,000 replications; M5 (a); M6 (a); M7 (a) with N = $1,000,000; M8 (a) with no fallback claims and the bootstrap-SE DSR; M9 (a) with value against cash as in the PRD and averaged targets as the seed ensemble", or amend any item.
