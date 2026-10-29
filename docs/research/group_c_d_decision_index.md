# Groups C and D: decision index (2026-10-17)

**Status: APPROVED by the owner on 2026-10-26 (New York): "Approve all C1–C9 (Recommended)" and "Approve all M1–M9 (Recommended)", the options of the suggested reply below (recorded in `docs/PROJECT_STATE.md` and `t5_portfolio_formulas.md` section 9).** The packets are `group_c_decisions.md` (C1–C9: cash, T-bill returns and benchmarks) and `group_d_decisions.md` (M1–M9: statistics, uncertainty and reporting). Formula ids refer to `t5_portfolio_formulas.md`. D1–D8 remain the T5a ids.

Each group can be approved, amended or rejected on its own, and so can each item. Group C unblocks T5b (real return inputs); group D unblocks T5c (metrics and reporting). Together they are what a real-data comparison of PPO against the baselines needs before its first run. Nothing changes in code until an approval is recorded with its date and quotation.

## Group C (`group_c_decisions.md`)

| Id | Decision | Formula ids | Recommendation, in one line |
|---|---|---|---|
| C1 | Cash-account return | OD8.5; F3.1, F3.4 | DGS3MO, investment basis, accrued on actual/365 (366) from the span's start rate; labelled "T-bill rate approximation, before fees" |
| C2 | Availability, vintages and backfill | F3.3; OD9 (this series only) | As a decision input, every vintage counts from the later of the next Federal Reserve business day's H.15 posting and the end of its ALFRED `realtime_start` date (real-data use waits for OD9's cutoff); append-only vintages published through 2024-12-31; first release for booked wealth; backfill 2002-12-02 to 2024-12-30. Breaking change to the rates store |
| C3 | Risk-free rate | F3.2 | The C1 series, so cash has an excess return of 0 (up to floating-point residue; M2's tolerance) |
| C4 | Cash, D4 (a) and SHV | (D4) | Real data uses C1; synthetic keeps 0%; SHV stays a separately traded fund; no engine change |
| C5 | Benchmarks | OD8.6; F4.1 | A fund's total return (F5.1), built from cash with the D3 build cost, held, fees never deducted again |
| C6 | `return_basis` | F4.2 | Required enum; `price_only` refused as a benchmark, context only. Breaking change (intended) |
| C7 | Reference portfolios and constraint sets | F4.3 | Reference portfolios through the same engine and constraint set; benchmarks stay unconstrained yardsticks |
| C8 | Common window start | F4 (edge cases) | The latest first close of every fund and benchmark, registered, no proxy splice |
| C9 | Booked-state exception | F3.3; OD9 (this series only) | Booked wealth, weights and trade sizes use y_{t−1} at t, an explicit exception to the derived-value rule scoped to booked account state; a field carrying the rate itself stays under C2 |

## Group D (`group_d_decisions.md`)

| Id | Decision | Formula ids | Recommendation, in one line |
|---|---|---|---|
| M1 | CAGR | OD8.8; F6.1 | Calendar days/365.25, build inside G; no annualization under one year |
| M2 | Volatility and Sharpe | OD8.8; F6.2 | √252 on per-session net returns; Sharpe on excess over C3; zero variance by a 1e-12 tolerance; over a gap, labelled-partial point values and no intervals |
| M3 | Intervals | OD8.9; F6.3, F6.4 | Circular block bootstrap, estimate ± t_{k−1}·SE, paired on the same draws, at least 8 blocks, with `fold_block_t` beside it (departs from the PRD's fold-block wording); settings provisional |
| M4 | Coverage simulation | OD8.9 | One registered run (five data-generating processes, four sample sizes, 120 cells, 5,000 replications, at least 93% coverage in every cell) before settings are fixed |
| M5 | Drawdown and stress windows | F6.5 (approved) | Reuse F6.5; add three registered stress windows, proposed as peak-to-trough dates (2007–09, 2020, 2022), with alternatives |
| M6 | Turnover per calendar year | F2.4 (approved; reading) | Calendar-year rebalance sums, construction separate and bounded, headline the mean of full years |
| M7 | Participation flag | OD8.11; F7.3 | Flag at 1% of 20-session median dollar volume on traded notional, at a registered notional N = $1,000,000 |
| M8 | Multiple testing | (new; STAT-5) | Intersection-union test for the primary claim with no fallback claims (error at most α by construction); deflated Sharpe (threshold 0.975) with the bootstrap standard error, at the effective N with the full-count DSR beside it |
| M9 | Reporting contract | (new) | The statistics, paired intervals, positive value against cash as the PRD requires, minimum detectable effects, out-of-sample sessions for learned policies, non-confirmatory and net-of-cost labels, and an averaged-target seed ensemble |

**Suggested reply:** "Approve group C: C1 (a), C2 (a), C3 (a), C4 (a), C5 (a), C6 (a), C7 (a) with benchmarks outside the constraint set, C8 (a), C9 (a). Approve group D: M1 as stated; M2 with (i) for incomplete histories; M3 with `fold_block_t` beside the bootstrap; M4 (a) with 5,000 replications; M5 (a); M6 (a); M7 (a) with N = $1,000,000; M8 (a) with no fallback claims and the bootstrap-SE DSR; M9 (a) with value against cash as in the PRD and averaged targets as the seed ensemble." Or amend any item.
