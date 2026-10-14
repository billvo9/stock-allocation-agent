# T5 portfolio readiness: financial-formula specification

**Status: every formula in this document is PENDING OD8 OWNER APPROVAL, except group A (F5.1–F5.4, F6.5, F6.6, the F4.1 fee rule and the OD8.13 label), approved on 2026-10-14 with the decisions recorded in section 9.** This is a specification for review, not an implementation. No execution, transaction-cost, cash-return, benchmark, performance-metric or reward code changes until the owner approves the relevant formulas under OD8 (`docs/product/vision_and_prd.md`, section 13), and each approval is recorded with its date and quotation.

- Written 2026-10-12 against `main` at `ad99749`. Code line references are as of that commit.
- Every numerical example is synthetic and was checked by a scratch Python run.
- An independent statistics review re-ran the examples and an exploratory coverage simulation. Its corrections are included (see "Review history" at the end). Its simulation warnings are kept in section 6.

## How to read and approve this document

- **Formula ids** (F1.1, F2.3, ...) are stable. Approvals cite them.
- **Owner decisions** OD8.1–OD8.15 group the formulas (section 9). They are approved, amended or rejected in four independent groups.
- **Group A is descriptive only.** It is what the first real page (P1-1b) needs: a fund's actual total-return history. It changes no simulator.
- **Groups B–D change the backtest simulator** (tickets T5a–T5c, section 10).

### Scope map: what P1-1b needs versus what the simulator needs

| Formulas | Subject | P1-1b (descriptive fund history) | Simulator ticket |
|---|---|---|---|
| F1.1–F1.3 | Next-close execution and drift | Not used: a fund history is not a trade | T5a |
| F2.1–F2.6 | Costs, initial build, turnover | Not used: no trading cost in a descriptive history (F5.4) | T5a |
| F3.1–F3.4 | Cash and T-bill returns | Not used: a T-bill fund is shown by its own total return | T5b |
| F4.1 | Fund total-return benchmarks; fees already in the price | **Required: the "fees are already in the price" rule only** | T5b |
| F4.2–F4.3 | `return_basis`; reference portfolios | Not used | T5b |
| F5.1–F5.3 | Total return from raw closes, distributions and splits; ex-date reinvestment; stored inputs | **Required** | T5b (as backtest inputs) |
| F5.4 | Hypothetical growth of $10,000, computed from actual prices and distributions | **Required** | Not used |
| F6.1–F6.2 | Calendar CAGR; √252 volatility and Sharpe | Optional: only if the page shows an annual rate | T5c |
| F6.3–F6.4 | Sharpe standard error; block-bootstrap intervals | Not used | T5c |
| F6.5 | Drawdown, its duration and its dollar size | **Required** ("worst decline") | T5c |
| F6.6 | Calendar-year returns | **Required** ("losing years") | T5c |
| F7.1–F7.3 | Rebalance offsets; participation flag | Not used | T5a (schedule), T5c (reporting) |
| F8.1 | Shared per-step function | Not used | T5a |
| OD8.13 | "Before taxes" label | **Required** | T5c |

P1-1b's other dependencies are not formulas: OD9's production availability rule for real prices, OD7's fund list, OD14's calendar, and a slice of F1 data. None of them is decided here.

**Notation.** Sessions t on the NYSE calendar; P_t is the raw close; r_t is the return from close t−1 to close t ("dated t"); w̃ is the drifted pre-trade weight vector; w* the target; Δ = w* − w̃; c_k the cost per dollar traded; W wealth; L the execution lag. [A] marks an assumption.

**Defects in current code (as of `ad99749`):**
1. Same-close execution (lag 0), while the label uses lag 1.
2. The cash leg is charged as a trade.
3. No initial build cost.
4. Momentum's cash earns 0 while Sharpe subtracts the T-bill rate.
5. Price-index benchmarks.
6. Compared strategies rebalance on different sessions.
7. Session-based CAGR, and no intervals.
8. Later, for PPO: the reward counts costs twice.

---

## 1. Next-close execution and drift

*Used by: T5a. Not used by P1-1b.*

**(a) Current.**
- `evaluation/baseline.py:207` takes the target dated t−1 and applies it to r_t (`:223-228`). A signal built from close t−1 therefore trades at that same close (lag 0).
- Features include their own close (`sql/rolling_features.sql:6-11, 21-25`; `baseline.py:400`).
- The initial weights are free (`:186`), and `tests/test_baseline.py:432-471` encodes lag 0.
- The label enters one close later (`features/training.py:49, 104, 214-220`; `entry_lag=1`).
- Drift (`baseline.py:49-66`) is correct.
- **Verdict:** this does not use r_{t+1}, but it is infeasible: the signal's close is the trade price. It disagrees with the label, and it leaks for inputs published after the close.

**(b) Proposed.**
- **F1.1** A decision at d (inputs available at d's cutoff) trades at the close of d+L, with L = 1. The new holdings first earn r_{d+2}. *Status: PENDING OD8 OWNER APPROVAL.*
- **F1.2** Drift: w̃_{i,t} = w_{i,t−1}(1+r_{i,t})/(1+R_{p,t}). *Status: PENDING OD8 OWNER APPROVAL.*
- **F1.3** `EXECUTION_LAG` is bound to `MODEL_LABEL_SPEC.entry_lag`. *Status: PENDING OD8 OWNER APPROVAL.*

**(c) Examples.**
- **Switch:** the portfolio holds B (flat) and decides at close d0 on 100% A, with r_A = +10% (d1) and −5% (d2). Lag 0 gives W = **1.045**; lag 1 gives **0.950**.
- **Label tie:** P = 100, 110, 121, 108.9.
  - The label at t=0 (h=2, lag 1) is 108.9/110 − 1 = **−1.00%**.
  - A lag-1 backtest that enters on decision d0 and exits on decision d2 earns 1.10 × 0.90 − 1 = **−1.00%**.
  - A lag-0 engine earns +21.0%.

**(d) Alternatives.** Lag 0 (infeasible with end-of-day data). Next open (needs split-adjusted open prices and an overnight gap). VWAP (needs intraday data).

**(e) Recommendation.** L = 1 at the close, one constant. Any other lag only as a registered sensitivity variant.

**(f) Edge cases.**
- No session d+1 in the data: not executed, with a reason.
- d+1 is a holiday: the next calendar session.
- A halted asset: keep the drifted holding, with a reason code.

**(g) Leakage.** Same-close trading on close-based signals. Benchmarks must start at the same execution close.

**(h) Tests.**
- T1.1: the switch gives 0.950 (1.045 fails).
- T1.2: the label tie gives −0.0100, equal to the label (+0.21 fails).
- T1.3: `EXECUTION_LAG == MODEL_LABEL_SPEC.entry_lag`.
- T1.4: a target on the final session produces no return row.

---

## 2. Costs, initial build and turnover

*Used by: T5a. Not used by P1-1b.*

**(a) Current.**
- Turnover is Σ|w̃ − w*| over all columns, including CASH (`environment/portfolio_math.py:21-34`): two-way, from drifted weights (`baseline.py:209-212, 246`; correct), and summed rather than annualized (`reporting.py:34`).
- Cost = 10 bp × turnover for every asset (`portfolio_math.py:49-60`; `scripts/run_all_baselines.py:122, 130, 139`), netted additively (`baseline.py:230`; `portfolio_math.py:94-95`).
- **Defects:**
  - the initial portfolio is free (`baseline.py:186`), and the benchmark costs 0 (`benchmark.py:39`);
  - the cash leg is charged: stocks → cash costs 20 bp (`tests/test_baseline.py:971-1004` asserts 99,800).

**(b) Proposed.**
- **F2.1** Δ_i = w*_i − w̃_i at the execution close. *Status: PENDING OD8 OWNER APPROVAL.*
- **F2.2** C = Σ_{i≠cash} c_{k(i)}·|Δ_i|, as a fraction of pre-trade wealth. Cash is the funding account and is free. *Status: PENDING OD8 OWNER APPROVAL.*
- **F2.3** W⁺ = W⁻(1 − C), post-trade weights = w*, W_{t+1} = W⁺(1 + R_p). The O(c²) fixed point is ignored, and this is stated. *Status: PENDING OD8 OWNER APPROVAL.*
- **F2.4** Reported one-way turnover τ = ½Σ_{all, incl. cash}|Δ_i| (a full switch = 1), annualized per calendar year, plus the traded notional Σ_{i≠cash}|Δ_i|. *Status: PENDING OD8 OWNER APPROVAL.*
- **F2.5** Every portfolio and benchmark starts as 100% cash at the first execution close and pays F2.2. No exit charge (stated). *Status: PENDING OD8 OWNER APPROVAL.*
- **F2.6** Rates [A]: single large caps 10 bp, broad equity and Treasury ETFs 5 bp, T-bill fund 2 bp, cash 0. Shown at 0×, 1× and 2×, never selected. *Status: PENDING OD8 OWNER APPROVAL.*

**(c) Examples.**
- **Rebalance:** W = 100,000, w̃ = (A .55, B .25, cash .20), w* = (.40, .40, .20), c = 10 bp.
  - Δ = (−.15, +.15, 0), so the cost base is .30, C = .0003 ($30) and τ = .15.
  - r = (+2%, −1%, +0.01%) gives R_p = 0.402% and W = **100,371.88** (additive: 100,372.00).
- **Stocks → cash:** 99,900 (current: 99,800).
- **Build 50/50 from cash:** 99,900 with τ = 1 (current: 100,000 with τ = 0).
- **Cross-class move:** 20% from an equity ETF (5 bp) to the T-bill fund (2 bp) costs 0.2×5 + 0.2×2 = **1.4 bp**; the same move into a Treasury ETF (5 bp) costs 2.0 bp.

**(d) Alternatives.** Charge the cash leg (current); cost on target weights (ignores drift); spread plus square-root impact; date-varying spreads (need data).

**(e) Recommendation.** F2.1–F2.6. Impact models wait.

**(f) Edge cases.** Δ = 0 gives C = 0; dust below 1e-12 is zeroed; C ≥ 1 raises; a forced sale on universe exit is costed; targets that do not sum to 1 are rejected.

**(g) Leakage.** Rates tuned after seeing results, or today's spreads applied to 2008. Register the rates in G1.

**(h) Tests.**
- T2.1: stocks → cash gives 99,900 (99,800 fails).
- T2.2: the build gives 99,900 and τ = 1.
- T2.3: C = .001 and R_p = .10 give 1.0989 (additive 1.0990 fails at 1e-9).
- T2.4: the rebalance gives C = .0003, τ = .15 and W = 100,371.8794. Costing from the previous target instead of drifted weights gives C = 0 and fails.
- T2.5: equity ETF → T-bill fund gives 1.4 bp; equity ETF → Treasury ETF gives 2.0 bp.

---

## 3. Cash and T-bill returns

*Used by: T5b. Not used by P1-1b.*

**(a) Current.**
- `data/rates/transform.py:68-197` takes the latest DGS3MO dated on or before the period start (`:148-155`). It accrues y × days/365 (366 in leap years), split at January 1 (`:21-65`), and accepts only `investment_basis` (`:115-120`). It is used only as the Sharpe risk-free rate (`performance.py:43-62`).
- Momentum's cash earns 0 (`run_all_baselines.py:141`; `baseline.py:366-379`).
- There is no `available_at` (`rates/schema.py:7-16`).
- **Verdict:** the accrual is correct. The zero cash leg is wrong: cash then shows an excess return of −r_f.

**(b) Proposed.**
- **F3.1** r_cash,t = y_{t−1} × days(t−1, t)/365 (366 in leap years): the existing function. *Status: PENDING OD8 OWNER APPROVAL.*
- **F3.2** The same series is the cash-leg return and the risk-free rate, so cash has an excess return of exactly 0. *Status: PENDING OD8 OWNER APPROVAL.*
- **F3.3** The rate dated t−1 prices a bill bought at t−1, so it accrues over (t−1, t] as realized return. As a decision input, it is available no earlier than the next business day [A; OD9 sets the timestamp]. *Status: PENDING OD8 OWNER APPROVAL.*
- **F3.4** Labelled "T-bill rate approximation, before fees". It ignores bill price moves (duration ≈ 0.25: a 10 bp rise costs ≈ 2.5 bp). Retail sweeps often pay less. *Status: PENDING OD8 OWNER APPROVAL.*

**(c) Examples.**
- **Convention check:** a 91-day bill at 98.75 per 100 of face value has a holding-period return of **1.2658%**.
  - Investment-basis yield 5.0772% × 91/365 = 1.2658% (exact).
  - Discount yield (DTB3-style) 4.9451%, used as if it were investment basis: 1.2329%, or −3.29 bp a quarter (≈ −13 bp a year).
  - Investment basis on /360: +1.76 bp.
- **Friday → Monday at 5%:** **0.041096%** (1/252 per session gives 0.019841%; /360 gives 0.041667%).
- **Friday 2023-12-29 → Tuesday 2024-01-02:** **0.054757%**.
- **Compounding:** daily accrual compounds to 5.1267% a year, against 5.0946% for a quarterly roll (+3 bp, accepted).

**(d) Alternatives.** A T-bill fund's total return (investable and net of fees; shorter history; needs F1 distributions). DTB3 converted by BEY = 365d/(360 − d·t) (an extra formula for no gain). 0 (current).

**(e) Recommendation.** F3.1–F3.4, with the T-bill fund only as a cross-check (tracking difference ≈ its expense ratio). Backfill DGS3MO to the earliest window start minus warm-up.

**(f) Edge cases.** No rate on or before the start: fail (exists). SIFMA-only holidays carry the previous rate. Zero or negative rates are allowed. A gap of more than 5 business days is flagged stale.

**(g) Leakage.** Using y_t, the end-of-period rate, instead of y_{t−1}. A rate signal at d sees only observations dated d−1 or earlier.

**(h) Tests.**
- T3.1: Friday → Monday gives 0.000410959 (0.000198413 and 0.000416667 both fail).
- T3.2: with Monday at 5% and Tuesday at 6%, Monday → Tuesday gives 0.000136986 (the Tuesday rate gives 0.000164384).
- T3.3: the year crossing gives 0.000547571.
- T3.4: a 100%-cash portfolio has excess returns within atol 1e-15 of 0 and Sharpe `unavailable` (floating-point residuals of (1−C)(1+R)−1 are 1e-17 to 1e-16, so exact zero is not testable). This fails with `cash_return=0`.
- T3.5: `discount_basis` is rejected (exists).

---

## 4. Total-return benchmarks

*Used by: T5b. P1-1b uses only F4.1's rule that fund fees are already in the price.*

**(a) Current.**
- The benchmarks are price indices `^GSPC`, `^IXIC` and `^DJI` (`config/assets.yaml`). For an index, `adjusted_close` = close (`sql/rolling_features.sql:6-11`).
- They are bought and held at zero cost (`evaluation/benchmark.py:12-41`), then compared with dividend-adjusted stocks (`run_all_baselines.py:144-151`).
- **Verdict:** biased by about the dividend yield, in the strategies' favour.

**(b) Proposed.**
- **F4.1** Benchmark = a fund's total return (F5.1), bought and held from the portfolio's first execution close with the F2.5 build cost. Fees are already in NAV, so the expense ratio is never subtracted. *Status: PENDING OD8 OWNER APPROVAL.*
- **F4.2** Each benchmark declares `return_basis` ∈ {`total_return_fund`, `total_return_index`, `price_only`}. `price_only` is never a benchmark, only labelled context. *Status: PENDING OD8 OWNER APPROVAL.*
- **F4.3** Reference portfolios (100% equity, 60/40, cash) run through the same engine, with their stated rebalancing. *Status: PENDING OD8 OWNER APPROVAL.*

**(c) Examples.**
- **Dividends:** with a flat price and 0.5% dividends each quarter, the total return is 1.005⁴ − 1 = **2.0151%**. A strategy at +1.00% "beats" the price index by 1.00 pt but trails the total return by **1.02 pt**.
- **Fees:** a 10.00% total-return index and a 0.09% expense ratio give a fund return of ≈ 1.10 × 0.9991 − 1 = **9.901%**.

**(d) Alternatives.**
- **Total-return index:** no fees, so it is unfair to fee-paying fund portfolios, though acceptable for stock-only research.
- **Price index plus an assumed yield:** rejected.

**(e) Recommendation.** Fund total return for every benchmark (one rule). Price indices move to labelled context.

**(f) Edge cases.**
- A fund that started after the window start: the window starts at the latest inception (registered), or a labelled proxy is used. Never splice silently.
- A benchmark missing a session: fail closed.

**(g) Leakage.** Choosing the benchmark after seeing results. Register it in G1 before the first run.

**(h) Tests.**
- T4.1: a flat-price fund paying D = 1 on P = 100 gives a period return of 1.0% (price-only gives 0).
- T4.2: a `price_only` benchmark raises `ValueError`.
- T4.3: benchmark and portfolio share the first execution date, and the benchmark's W after build is 1 − c.
- T4.4: expense-ratio metadata leaves the NAV-based returns unchanged.

---

## 5. Distributions, reinvestment and corporate actions

*Used by: P1-1b (F5.1–F5.4) and T5b (F5.1–F5.3, as backtest inputs).*

**(a) Current.**
- `daily_return` comes from Yahoo `Adj Close` (`data/yfinance_source.py:32, 110`; `sql/rolling_features.sql:6-11`). Its CRSP-style back-multiplier gives an ex-date return of P_ex/(P_{ex−1} − D) − 1 [A: vendor behaviour].
- Levels encode later actions (`model_validation/checks.py:32-35`), and every download rescales the history.
- **Verdict:** returns are approximately right, but neither reproducible nor prefix-stable.

**(b) Proposed.**
- **F5.1** r_t = (f_t·P_t + D_t)/P_{t−1} − 1 on raw closes, where:
  - f_t = shares after / shares before a split effective at t (otherwise 1);
  - D_t = the cash distribution per share held at t−1, with ex-date t (dividends, capital gains, return of capital);
  - D_t is on the pre-split basis. A distribution quoted per post-split share is converted, D_old = f_t·D_new. Every stored distribution records its share basis, and an unknown basis on a split date gives `unavailable`, never a guess.

  *Status: APPROVED 2026-10-14 (OD8 group A); see the decision record in section 9.*
- **F5.2** Reinvestment at the ex-date close, in the same asset (implicit in compounding F5.1). *Status: APPROVED 2026-10-14 (OD8 group A); see the decision record in section 9.*
- **F5.3** Store raw closes, distributions (ex-date, pay date, amount, type), split factors and retrieval times. Vendor-adjusted closes are only a cross-check (a flag, never a correction). The vendor's back-multiplier differs from F5.1 by design, by about (D/P)·r on the ex-date (0.505% vs 0.500% above), so the flag tolerance must exceed that or every ex-date is flagged. *Status: APPROVED 2026-10-14 (OD8 group A); see the decision record in section 9.*
- **F5.4** Hypothetical growth index (P1-1b): G_t = 10,000 × Π_{s=t₀+1..t}(1 + r_s), with r from F5.1.
  - **Start t₀ is fixed by rule before any chart is viewed:** the first session on or after the registered window start at which the fund has a close. Funds on one chart share t₀, the latest of their individual starts. No start date is chosen by looking.
  - No transaction or build cost is charged, and the expense ratio is never subtracted again: it is already reflected in the fund's NAV and hence its price (F4.1).

  *Status: APPROVED 2026-10-14 (OD8 group A); see the decision record in section 9.*

**Actual records versus hypothetical growth.**
- **Actual historical records:** a fund's closing prices, its declared distributions (amount, ex-date, pay date, type) and its split factors.
- **A hypothetical computation from those records: the growth index G.** It assumes:
  - every distribution is reinvested in the same fund at the ex-date close;
  - fractional shares;
  - no transaction cost and no tax;
  - no cash waiting between the ex-date and the pay date.
- **No investor's account necessarily experienced G.** Taxes on distributions, pay-date reinvestment, whole shares and trading costs all lower it.
- **P1-1b labels the two separately:**
  - prices and distributions as historical records;
  - the growth line as "Hypothetical growth of $10,000, distributions reinvested on the ex-date, before transaction costs and taxes".
- **Evidence status.** It stays `descriptive`, because it is a conventional total-return calculation over actual records, but its label must say hypothetical.
- **ETF closes are market prices, not NAV.** So G is a market-price total return. It can differ from the issuer's published NAV total return by changes in the premium or discount.

**(c) Examples.**
- **Ex-date:** P = 100 → 99.5 (ex-date, D = 1.00) → 100.495 gives r = **+0.500%**, then **+1.000%**. Price-only gives −0.500%; the vendor-adjusted series gives +0.505%.
- **Split 4:1:** 400 → 101 gives 4 × 101/400 − 1 = **+1.00%** (naive: −74.75%).
- **Split and distribution on the same date:** P_{t−1} = 400, a 4:1 split, P_t = 99.5 and a distribution of 1.00 per old share (0.25 per new share) give (4 × 99.5 + 1.00)/400 − 1 = **−0.250%**. Treating the per-new-share 0.25 as per old share gives −0.4375%, an 18.75 bp error.
- **Growth of $10,000:** the ex-date example's returns (+0.500%, then +1.000%) give 10,000 × 1.005 × 1.01 = **$10,150.50**. Price-only returns (−0.500%, +1.000%) give $10,049.50.
- **Pay date three sessions later:** the ex-date example above (P = 100 → 99.5, D = 1.00), then +1% a session for three sessions. Ex-date reinvestment gives 1.005 × 1.01³ = 1.035453; pay-date reinvestment (the dividend held as a non-earning receivable) gives 0.995 × 1.01³ + 0.01 = 1.035149, a difference of **3.0 bp**.

**(d) Alternatives.**
- Vendor-adjusted closes (current; simplest, unstable).
- Pay-date reinvestment: true cash timing, but needs a receivable state, for a basis-point effect.
- Reinvesting into cash.

**(e) Recommendation.** F5.1–F5.4 with ex-date reinvestment. Pay-date timing waits.

**(f) Edge cases.**
- Distribution dated on a non-session: error.
- Special dividends: same formula.
- A split and a dividend on the same date: D is per old share.
- Reverse splits: f < 1.
- A missing distribution record understates return. A fund-period whose distribution coverage is incomplete is flagged `distribution_coverage_incomplete`, never silently treated as price-only; the vendor cross-check helps find gaps.

**(g) Leakage.**
- Adjusted levels used as features (already forbidden).
- Vendor backfills of distributions: store retrieval times and test prefix stability.

**(h) Tests.**
- T5.1: the ex-date return is exactly 0.0050 (price-only −0.005 and the vendor's 0.00505 both fail at 1e-9).
- T5.2: the split gives +0.01.
- T5.3: appending a later distribution leaves earlier returns bit-identical.
- T5.4: the pay-date return equals the price return.
- T5.5: the growth example gives 10,150.50 within 1e-9 relative (price-only 10,049.50 fails); no cost is subtracted.
- T5.6: the same-date split and distribution give −0.0025 on either quoted basis; the mixed basis (−0.004375) fails; an unknown basis on a split date gives `unavailable`.
- T5.7: two funds whose first closes differ share t₀ = the later first close; t₀ is computed from the registered window start, never passed in by the caller.

---

## 6. CAGR, drawdown, Sharpe and intervals

*Used by: P1-1b (F6.5, F6.6; F6.1 only if an annual rate is shown) and T5c (all).*

**(a) Current.**
- CAGR = G^{252/n} − 1 (`evaluation/metrics.py:45-56`).
- Volatility = √252·sd (`:59-73`); Sharpe = √252·mean/sd of excess returns (`:76-100`).
- Sortino against the risk-free rate (`:103-133`); Calmar = CAGR/MDD (`:153-172`).
- Maximum drawdown includes W_0 (`baseline.py:251-259`; `metrics.py:136-150`).
- **Verdict:** standard formulas, but CAGR assumes 252 sessions a year (wrong for 250- or 253-session years, closures and other calendars). No intervals and no drawdown duration.

**(b) Proposed.**
- **F6.1** CAGR = G^{365.25/D} − 1, with G = W_end/W_0. D is the number of calendar days from the first execution close (W_0 before the build cost) to the last close. *Status: PENDING OD8 OWNER APPROVAL.*
- **F6.2** Volatility and Sharpe keep √252, with excess returns over F3.2. *Status: PENDING OD8 OWNER APPROVAL.*
- **F6.3** Sharpe standard error (Lo 2002, i.i.d.): √((1 + ½SR²)/n) per session, × √252. **A sanity check on the bootstrap only, never the headline.** At daily Sharpe the SR² term is negligible; the real risk is serial dependence, which the block bootstrap covers. Mertens' (2002) non-normal SE is not used: at daily Sharpe it moves the SE by about 0.004, and pandas `kurt()` returns excess kurtosis, an easy silent bug. *Status: PENDING OD8 OWNER APPROVAL.*
- **F6.4** Headline intervals reuse T2's circular block bootstrap (`model_diagnostics/inference.py:428`, `:533`):
  - blocks of b = 63 sessions, 999 replicates, a registered seed. The block length is chosen for volatility clustering in daily returns (not T2's 20-session label overlap) and is registered before any run;
  - **interval = estimate ± t_{k−1} × bootstrap SE**, with k = ⌈n/b⌉ blocks. This is T2's existing rule for means (`bootstrap_mean`); for Sharpe the SE comes from `bootstrap_statistic`'s replicates. Percentile intervals are not used (see the warning below);
  - **CAGR:** the interval is built on mean log growth ḡ = mean(log(1+R)) with `bootstrap_mean`, then mapped to CAGR endpoints through exp(·)−1 with the calendar span held fixed;
  - **paired comparison against the benchmark:** the difference in annualized log growth (not CAGR_p − CAGR_b), and the Sharpe difference, both on the same block draws;
  - **at least 8 blocks (504 sessions)**; otherwise `unavailable` with reason `too_few_blocks`;
  - intervals are labelled **nominal** 95%. No coverage is claimed for real returns;
  - **provisional settings:** the block length (63), replicate count (999), the 8-block minimum and the t·SE method stay provisional until the registered coverage simulation has run. Approving OD8.9 approves the method family; the settings are fixed only from that simulation's registered results.

  *Status: PENDING OD8 OWNER APPROVAL.*

  > **Simulation warning (exploratory and unregistered).** During review, a scratch simulation estimated the coverage of nominal 95% intervals using T2's `draw_circular_blocks` and `resampled_sums`, 63-session blocks and 999 replicates, on synthetic i.i.d. and GARCH(1,1) daily returns, with 1,000 replications per cell (Monte Carlo error about ±1.4 percentage points near 95%, about ±2.8 near 72%):
  >
  > | Blocks (sessions) | 2 (126) | 4 (252) | 8 (504) | 20 (1,260) | 40 (2,520) |
  > |---|---|---|---|---|---|
  > | Percentile interval, mean | 72–75% | 86% | 91% | 93–94% | 93–94% |
  > | Percentile interval, Sharpe | 72–76% | 85% | 90–92% | 92–94% | 93% |
  > | Estimate ± t(k−1)·bootstrap SE | 100% | 96–97% | 94–96% | 94–95% | 94–95% |
  >
  > - Percentile intervals built from few blocks are far too narrow.
  > - At 2 blocks the t interval over-covers only because t with 1 degree of freedom is 12.7; it is uninformative there, hence the 8-block minimum.
  > - **This does not validate 95% coverage on real data.** Real returns have heavier tails, regime changes and dependence longer than a block (the 2008 drawdown ran about 350 sessions). Before any interval is described as more than nominal, T5c must run a registered coverage simulation that includes heavy-tailed, regime-switching and long-memory scenarios.

- **F6.5** DD_t = 1 − W_t/max_{s≤t}W_s. Duration runs from the peak to the first close at or above it, in sessions and calendar days, or "not recovered". A point estimate only: maximum drawdown is not a smooth statistic, depends on the horizon, and is driven by episodes longer than a block (2008 ran about 350 sessions), so a block-bootstrap interval would mislead. Stress windows sit beside it. It is computed on daily closes, so intraday declines may have been deeper, and the label says so. For a growth index (F5.4), the worst decline is the episode with the largest percentage drawdown, and its dollar size is that episode's peak value minus its trough value. (The largest dollar decline can be a different, later episode, because later peaks are higher; the page states which one it shows.) *Status: APPROVED 2026-10-14 (OD8 group A); see the decision record in section 9.*
- **F6.6** Calendar-year return = W(last close of year)/W(last close of prior year) − 1; a partial first or last year is labelled partial and never annualized. A losing year has an unrounded return below 0; display rounding never changes the classification (−0.04% displays as −0.0% and counts as losing). Rule: CAGR is never divided by volatility (Sharpe keeps its own 252 scale; Calmar = CAGR/MDD is fine). *Status: APPROVED 2026-10-14 (OD8 group A); see the decision record in section 9.*

**(c) Examples.**
- **CAGR:** G = 1.10 over 365 days and 248 sessions gives **10.007%** (session-based: 10.169%).
- **Sharpe:** a per-session Sharpe of 0.05 with n = 2,520 gives an annual 0.794, Lo SE **0.316** and an i.i.d. 95% interval of **[0.17, 1.41]**. Even under i.i.d. assumptions, ten years of daily data cannot pin Sharpe down to better than about ±0.6.
- **Drawdown:** wealth 100, 110, 99, 105, 111 gives MDD **10.0%** and a duration of **3 sessions**. Scaled to a $10,000 start (10,000, 11,000, 9,900, 10,500, 11,100), the worst decline is **$1,100**.
- **Calendar-year returns:** W = 10,000 at the last close of 2022, 10,800 at the last close of 2023 and 10,260 at the last close of 2024 give **+8.00%** for 2023 and **−5.00%** for 2024, a losing year.

**(d) Alternatives.** Sessions/252; Lo's autocorrelation-adjusted annualization; the stationary bootstrap; HAC Sharpe; percentile or BCa intervals (undercover with few blocks); Ledoit–Wolf (2008) for Sharpe differences. Reusing the T2 code keeps one inference path.

**(e) Recommendation.** F6.1–F6.6.

**(f) Edge cases.** Zero variance: Sharpe `unavailable` (`zero_variance`). Fewer than 8 blocks (504 sessions): no intervals (`too_few_blocks`). An unrecovered final drawdown is flagged.

**(g) Leakage.** A block length or interval method chosen after seeing results; register both. Keep the risk-free rate aligned with the return period (`risk_adjusted.py:34-35`).

**(h) Tests.**
- T6.1: CAGR 0.1000718 (0.1016923 fails).
- T6.2: Lo SE with per-session SR = 0.3 and n = 120 gives 0.093318; dropping the SR² term gives 0.091287 and fails. (At SR = 0.05 the term moves the SE by only 2e-4, too little to test.)
- T6.3: the drawdown example gives MDD 0.10 and duration 3; ending at 108 instead gives "not recovered".
- T6.4: identical series give a paired result `unavailable` with reason `zero_variance` (what `bootstrap_mean` returns); for correlated series the paired interval is narrower than the unpaired one.
- T6.5: the same seed gives identical intervals; 503 sessions give `too_few_blocks`, 504 give an interval.
- T6.6: hand-built `BlockDraws` with starts 0, 63, 126, … reproduce the point estimate exactly, which catches the estimate and the replicates being annualized differently.
- T6.7: the calendar-year example gives +0.0800 and −0.0500 (using the first close of a year instead of the prior year's last close fails); a partial year is labelled and not annualized.
- T6.8: the drawdown example's worst decline is $1,100 on a $10,000 start.
- T6.9: a year at −0.0004 is classified as losing although it displays as −0.0%; a year at exactly 0 is not.

---

## 7. Rebalance-date sensitivity and participation

*Used by: T5a (F7.1–F7.2 schedule) and T5c (reporting, F7.3). Not used by P1-1b.*

**(a) Current.**
- Rebalancing counts rows (`baseline.py:202-204`).
- Equal weight starts at `rebalance_every + 1` (`:362`); inverse volatility and momentum at `rebalance_every` (`:313, :469`). The strategies compared in `run_all_baselines.py` therefore rebalance on **different sessions**, an unrecorded variant.
- No volume check and no bands.

**(b) Proposed.**
- **F7.1** Every portfolio and benchmark builds at one common first execution close. Later rebalances follow the decision sessions with (d − d_first) mod k = o, for one registered offset o shared by every compared portfolio. Otherwise a nonzero offset builds later, and n, the calendar span, the bootstrap draws and the benchmark start all vary by offset. *Status: PENDING OD8 OWNER APPROVAL.*
- **F7.2** Run all k offsets; report the registered offset's result and the max − min spread across offsets, with no per-offset intervals. The k offsets are recorded as **one variant with a k-member sensitivity family**, not k independent trials: they are nearly identical return paths, and counting k would inflate the trial count in a later deflated Sharpe (risk register STAT-5 asks for an effective-trial-count rule). They never choose the headline. Approving this changes the roadmap's T5 wording "counted as variants". *Status: PENDING OD8 OWNER APPROVAL.*
- **F7.3** Participation is a flag only, measured on **actual traded notional** at each execution:
  - q_i = |Δ_i|·W⁻/MDV_i, where Δ_i is the F2.1 trade (drifted pre-trade weight to target), W⁻ is pre-trade wealth, and MDV_i is the median of close × volume over the 20 sessions ending at the decision session d;
  - **the initial construction is a separate full purchase** (Δ_i = w*_i from 100% cash), reported separately from later rebalances;
  - flag when q_i > p, with p = 1% [A];
  - no capacity figure: one would read like a statement of investable capacity, and with large caps and broad funds the flag will rarely fire.

  *Status: PENDING OD8 OWNER APPROVAL.*

**(c) Examples.**
- **Offsets:** A alternates ±10%, B is flat, and a 50/50 portfolio rebalances every 2 sessions for 6 sessions. Rebalancing after A's up days ends at **0.98755**; after its down days, at **0.98507** (buy and hold: 0.98515).
- **Participation:** W = $1m and MDV_A = $20m. A rebalance of A from a drifted 45% to a 30% target trades $150,000: q = **0.75%**, not flagged. (Measuring holdings instead, 30% × $1m, would give 1.5% and flag it wrongly.) The initial build of A at 30% buys $300,000: q = **1.5%**, flagged, and reported as build participation.

**(d) Alternatives.** A month-end rule. Tranching (averaging across offsets; the usual remedy, as a later variant). Holdings-based participation (the earlier draft; it overstates rebalance participation). A participation cap (it changes the path; defer). No-trade bands (defer).

**(e) Recommendation.** F7.1–F7.3.

**(f) Edge cases.** k = 1 has one offset. Fewer than 20 volume sessions: `unavailable`. Zero-volume days. An ETF's dollar volume understates its capacity.

**(g) Leakage.** MDV including d+1. Raw split-adjusted volume (use close × volume, which is split-invariant). Choosing the best offset.

**(h) Tests.**
- T7.1: a shared schedule gives identical execution dates across strategies (today: 6, 11 against 5, 10), and every offset shares the build date.
- T7.2: k = 5 records one variant with a 5-member sensitivity family.
- T7.3: the offset values above.
- T7.4: the rebalance example gives q = 0.0075, not flagged (a holdings-based 0.015 fails); a rebalance with no trade in A gives q_A = 0.
- T7.4b: the build example gives q = 0.015, flagged and reported under build, not under rebalances.
- T7.5: changing volume at d+1 leaves MDV at d unchanged.

---

## 8. Shared per-step function (backtest and future PPO)

*Used by: T5a. Not used by P1-1b.*

**(a) Current.**
- The formulas are inlined in the loop at `baseline.py:199-259`, with additive costs.
- `environment/reward.py:40-45` computes log(W_t/W_{t−1}) − rate·turnover, while `update_wealth` already nets out costs (`portfolio_math.py:94-95`). **Costs would count twice.** This is out of scope (a reward is an approval-gated formula) and flagged.

**(b) Proposed.**
- **F8.1** A pure function, with no dates and no I/O, in `environment/portfolio_math.py`:

  ```
  portfolio_step(w_pre, target | None, r_next, cost_rate, cash_index)
      -> StepResult(w_post_trade, w_next, gross_return, cost,
                    net_return, turnover_one_way, traded_notional)
  ```

  1. Trade (F2.1–F2.3).
  2. Earn r_next, with cash from F3.1: 1 + R_net = (1 − C)(1 + w*ᵀr).
  3. Drift: w_next = w* ⊙ (1 + r)/(1 + w*ᵀr).

  The lag (F1.1) and the schedule (F7.1) live in the caller. The PPO reward is later log(1 + R_net), with no extra cost term. *Status: PENDING OD8 OWNER APPROVAL.*
- **Invariants:**
  - w ≥ 0, and Σw = 1 within 1e-12;
  - 0 ≤ C < 1;
  - no target, or a target equal to w_pre, gives C = 0 and pure drift;
  - R_net ≤ R_gross;
  - an all-cash portfolio returns r_cash;
  - Π(1 + R_net) = W_T/W_0;
  - a non-finite return on a held asset raises an error, and the caller records a reason code.

**(c) Example.** The §2 rebalance: C = 0.0003, R_gross = 0.402%, R_net = **0.37188%**, w_next = (**0.40637, 0.39441, 0.19922**).

**(d) Alternatives.** A stateful environment class, or keeping the loop. Either duplicates the formulas between backtest and PPO.

**(e) Recommendation.** F8.1. One function is the single place where costs and drift are defined.

**Common environment for every allocation policy.** The T5a scheduler and `portfolio_step` form the one simulator for every allocation policy: declared rules, constrained optimizers, and learned policies (reinforcement learning, with PPO as a candidate). PPO is a challenger evaluated under identical timing, costs and constraints, never an assumed improvement. `vision_and_prd.md` section 15 requires a no-forecast PPO comparison (what forecasts add) and same-input constrained-optimizer baselines (what reinforcement learning adds). The step function's contract that no observation contains r_next is what makes those comparisons fair. The PPO reward fix (flagged in part (a) of this section) is its own approval-gated ticket.

**(f) Edge cases.** The invariants above, and dust trades.

**(g) Leakage.**
- The function cannot see dates, so the lag tests belong to the caller.
- A PPO observation must never include r_next.

**(h) Tests.**
- T8.1: the example, within 1e-12.
- T8.2: a seeded property test of the invariants.
- T8.3: three chained steps equal the engine's wealth.
- T8.4 (PPO ticket): reward = log(1 + R_net). A second cost subtraction fails.

---

## 9. Owner decisions (OD8.x)

| # | Decision | Options | Recommendation |
|---|---|---|---|
| OD8.1 | Execution timing | lag 0; **next close**; next open | Next close (L=1), bound to `entry_lag` (F1) |
| OD8.2 | Cost base and turnover | cash leg charged; **risky legs only**; additive or **multiplicative** | Risky legs only, from drifted weights, multiplicative. Report one-way turnover annualized, and traded notional (F2.1–F2.4) |
| OD8.3 | Initial build and exit | none; **build**; build + exit | Build for portfolios and benchmarks; no exit charge (F2.5) |
| OD8.4 | Cost rates | one rate; **per asset class** | 10, 5, 2 and 0 bp [A], shown at 0×, 1× and 2× (F2.6) |
| OD8.5 | Cash convention | **DGS3MO investment basis, actual/365**; T-bill fund; DTB3 | DGS3MO for the cash leg and the risk-free rate; T-bill fund as a cross-check (F3) |
| OD8.6 | Benchmarks | price index; total-return index; **fund total return** | Fund total return; `return_basis` required (F4) |
| OD8.7 | Distributions | vendor-adjusted; **ex-date**; pay date | Ex-date on raw closes with split factors (F5) |
| OD8.8 | CAGR basis and calendar-year returns | sessions/252; **calendar/365.25** | Calendar CAGR; 252 for volatility and Sharpe; never CAGR/σ; calendar-year returns with partial years labelled (F6.1–F6.2, F6.6) |
| OD8.9 | Intervals | none; i.i.d.; percentile block; **block bootstrap SE + t** | Circular block bootstrap; estimate ± t_{k−1}·SE; settings (63, 999, at least 8 blocks) **provisional until the registered coverage simulation**; CAGR via mean log growth; paired against the benchmark on the same draws; Lo SE as a sanity check only; drawdown with duration and no interval (F6.3–F6.5) |
| OD8.10 | Rebalance offsets | one; all offsets as k variants; **one variant + sensitivity family** | Common build date; shared registered offset; report its result and the max − min spread; offsets count as one variant with a k-member family (changes the roadmap's wording) (F7.1–F7.2) |
| OD8.11 | Participation | none; **flag on traded notional**; flag on holdings; cap | Flag when traded notional exceeds p = 1% of the 20-session median dollar volume [A]; the build reported separately; no capacity figure (F7.3) |
| OD8.12 | Per-step contract | loop; **pure function** | F8.1 and its invariants |
| OD8.13 | Tax label | — | "Before taxes" on every return output; no tax model |
| OD8.14 | Changed test expectations | — | Approving OD8.1–8.3 approves rewriting the tests encoding lag 0, the cash-leg cost and the free build (`test_baseline.py:432, 474, 971`, and others found) |
| OD8.15 | Descriptive histories | — | Hypothetical growth index from actual prices and distributions (F5.1), labelled as hypothetical; start fixed by rule; no transaction or build cost; "before transaction costs and taxes" (F5.4) |

### Group A decision record (owner, 2026-10-14)

The owner approved OD8 group A for descriptive history only, with these decisions. Implemented in `src/stock_agent/portfolio/descriptive.py` (ticket T5-A).

| Item | Formula ids | Decision |
|---|---|---|
| A1 | F5.1 | Approved. Raw-price total returns handle split ratios and dividend amounts on consistent share bases; a distribution on a split date needs its share basis, else unavailable |
| A2 | F5.2 | Approved. Ex-date reinvestment is a hypothetical comparison convention, not actual cash-payment timing |
| A3 | F5.3 | Approved. Inputs stored as received (F1 contract) |
| A4 | F5.4 | Approved |
| A5 | F4.1 (fee rule only) | Approved. Fund expenses already in observed performance are not deducted again. The rest of F4.1 (benchmarks) stays in group C |
| A6 | F6.5 | Approved |
| A7 | F6.6 | Approved |
| A8 | OD8.13 | Approved: "Before taxes and transaction costs" |
| R1 | F5.1, F5.3 | Approved. No double counting of distributions or vendor adjustment factors; only unadjusted closes and as-paid amounts are accepted |
| R2 | F5.4, F6.5, F6.6 | Approved. A user-chosen period restarts growth at $10,000; drawdown uses peaks inside it; cut years are partial |
| R3 | Labels | "Historical market data" for the records; "Hypothetical growth of $10,000 — distributions reinvested" for the calculated series |
| R4 | F5.4 (start) | Approved. t0 = the first session, on or after the latest of the funds' first closes, at which every fund has a close |
| R5a | F5.1, F5.4 | Approved. A return across sessions without a close is an explicitly labelled multi-session return (`sessions_spanned`), never described as a one-session return; missing observations are never filled |
| R5b | F5.1 | **Not approved.** No automatic reinvestment through gaps: a distribution with its ex-date on a session without a close makes that return unavailable, and nothing chains through it, unless the economic return can be reconstructed and validated |
| R5c | F6.5 | **Thresholds not approved.** An observed drawdown over an incomplete history is distinguished from a valid full-period maximum drawdown: with any session missing, the full-period maximum is unavailable and only the "largest observed decline" is shown, labelled incomplete |
| R5d | F6.6 | Approved. A year-end without a value makes both affected years unavailable |

Not authorized by these approvals: T5a–T5c simulator formulas (groups B–D) and any PPO or RL work.

### Decision groups (how to approve)

| Group | Decisions | Unblocks | Needs first |
|---|---|---|---|
| **A. Descriptive (P1-1b)** | OD8.7 (F5.1–F5.3), OD8.15 (F5.4), F4.1's "fees are already in the price" rule, F6.5 (drawdown, duration, dollar size), F6.6 (calendar-year returns), OD8.13 ("before taxes") | **P1-1b** (growth of $10,000, worst decline, losing years) | Nothing; can be approved first |
| **B. Engine** | OD8.1, 8.2, 8.3, 8.4, 8.10, 8.12, 8.14 | T5a, and every backtest the product shows | — |
| **C. Return inputs** | OD8.5 (cash), the rest of OD8.6 (benchmarks); OD8.7 applied to backtest inputs (decided in group A) | T5b | F1 data; OD9's production availability rule |
| **D. Statistics and reporting** | OD8.8 (except F6.6), OD8.9 (except F6.5), OD8.11 | T5c | T5a (T5b for the final re-run) |

Each group can be approved, amended or rejected on its own. An approval is recorded with its date and quotation in `docs/PROJECT_STATE.md` and in this specification, and only then does code change.

## 10. Proposed tickets after approval

- **P1-1b** uses only group A. It adds no simulator code and does not wait for T5a–T5c.

- **T5a: engine** (OD8.1–8.4, 8.10, 8.12, 8.14).
  - Contents: `portfolio_step`, a scheduler with lag and offsets, the build cost, and turnover reporting; migrate `run_rebalanced_backtest`; the lag-binding test.
  - No new data. M.
- **T5b: return inputs** (OD8.5–8.7; F1 data; OD9 for `available_at`).
  - Contents: the DGS3MO backfill and the shared cash series; fund total-return benchmarks with `return_basis`; F5.1 returns from raw closes, distributions and splits; prefix-stability tests.
  - Depends on F1. M.
- **T5c: metrics and reporting** (OD8.8, 8.9, 8.11, 8.13).
  - Contents: calendar CAGR, calendar-year returns, intervals, drawdown duration, the offset spread, the participation flag, the "before taxes" label.
  - A baseline re-run with a **one-change-at-a-time attribution**: lag → cost base → build → cash rate → benchmark basis → CAGR basis, with the explained delta at each step.
  - A registered coverage simulation (heavy tails, regime switching, long memory) before any interval is described as more than nominal.
  - Depends on T5a; the final re-run also needs T5b. M.

## 11. Not in T5

- Market-impact or historical-spread models.
- No-trade bands, turnover penalties and optimizers (equal risk contribution, minimum variance).
- Covariance shrinkage (P4-1).
- Pay-date receivables.
- Next-open, VWAP or intraday execution.
- Delisting returns and the survivorship-free universe (U1).
- Deflated Sharpe, PBO and reality checks (variant counts are recorded now).
- Tax modelling.
- The PPO environment and the reward fix (flagged).
- The "gain" definition (OD8, later).
- Treasury yield-to-return models: funds only.
- Calendar dependency (OD14; F1 supplies sessions).
- Multi-currency and crypto calendars.

## 12. Learning notes

1. **Problem.** Make simulated portfolio returns executable, cost-complete and comparable, with uncertainty.
2. **Concept.** Self-financing wealth recursion; total return; day-count conventions; dependent-data inference.
3. **Assumptions.** Constant cost per dollar traded; trades fill at the close; the yield accrual approximates bill returns; dependence beyond about 63 sessions is weak (an assumption the registered coverage simulation must test).
4. **Formulation.** W_{t+1} = W_t(1 − C_t)(1 + w*ᵀr_{t+1}); r = (fP + D)/P₋₁ − 1; CAGR = G^{365.25/D} − 1; SE(SR) = √((1 + SR²/2)/n).
5. **Example.** The §2 rebalance costs $30 and returns 0.372% net.
6. **Leakage.** Same-close trading; end-of-period rates; vendor levels; volume from d+1.
7. **Misleading results.**
   - Price benchmarks overstate excess return by the dividend yield.
   - A single rebalance phase is one draw among k.
   - Sharpe intervals over 10 years are wide (about ±0.6, even under i.i.d. assumptions).
   - Percentile intervals from few blocks are too narrow (a nominal 95% covered about 72% at 2 blocks).
8. **Industry validation.** CRSP return definitions; GIPS-style net-of-cost, time-weighted returns; Lo (2002) and Politis–Romano block bootstrap; a coverage simulation of the interval method; Ledoit–Wolf (2008) for Sharpe differences; hand-computed reconciliation against a fund's published total return.
9. **Next to learn.** Bailey–López de Prado deflated Sharpe and the effective number of trials; the studentized block bootstrap; square-root impact.

**Optional owner exercise (after approval; never blocking).** Implement `portfolio_step` (F8.1) yourself. Tests T2.1–T2.4 and T8.1–T8.2 confirm it.

## Review history

- **2026-10-12, draft.** Code inspection against `ad99749`; every example computed by a scratch script.
- **2026-10-12, statistics review.** Re-ran every example and the exploratory coverage simulation in section 6. Changes:
  - headline intervals moved from percentile to estimate ± t·bootstrap SE, with an 8-block minimum, and are labelled nominal;
  - the CAGR interval moved to mean log growth;
  - Mertens' SE, the min/median/max offset table and the capacity figure were dropped;
  - every offset now shares one build date, and the offsets count as one variant family;
  - corrected: the cross-class cost example, the pay-date example's inputs, and the tolerances of T3.4, T6.2 and T6.4.
- **2026-10-13, scope map.** P1-1b's descriptive formulas are separated from the simulator's (scope map; groups A–D); F5.4 (growth index) and OD8.15 added; dollar drawdown and calendar-year examples added; the common-environment note for learned policies added to section 8.
- **2026-10-13, owner review.**
  - **Actual versus hypothetical:** actual records (prices, distributions, splits) are now separated from the hypothetical growth index.
  - **F7.3:** participation is measured on traded notional, with the initial build reported separately.
  - **Bootstrap settings:** provisional until the registered coverage simulation.
  - **Group A corrections before approval:**
    - the share basis of a distribution on a split date (F5.1, T5.6);
    - a start-date rule (F5.4, T5.7);
    - incomplete distribution coverage is flagged;
    - drawdown is close-to-close;
    - losing years are classified on unrounded returns (T6.9).
