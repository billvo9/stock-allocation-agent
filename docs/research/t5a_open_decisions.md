# T5a: the open financial decisions (v2, 2026-10-14)

**Status: D1-D8 APPROVED by the owner on 2026-10-15, as recorded below (D9 is not part of this packet and stays pending). One D5 clause awaits the owner's confirmation (see D5 below).**

## Owner approval record (2026-10-15, verbatim, in the lead session)

The owner approved "the following D1-D8 choices, provided they exactly match the current `docs/research/t5a_open_decisions.md` packet", and asked that a material difference stop work and be identified.

| Id | The owner's words (verbatim) | Match with this packet |
|---|---|---|
| D1 | "Next-session-close execution." | Matches D1. |
| D2 | "Costs on actual risky-asset traded notional, multiplicative accounting. Provisional assumptions of 10 bp single stocks, 5 bp broad ETFs, 2 bp SHV and zero cost for the cash funding leg. Report 0×, 1× and 2× sensitivity." | Matches D2. The rates are provisional assumptions. |
| D3 | "Begin in cash, charge initial construction and use no terminal liquidation charge for marked-to-market comparisons." | Matches D3. |
| D4 | "D4(a): Uninvested cash earns 0% for synthetic engineering. SHV is a separately traded ETF. Do not claim this is realistic interest-bearing cash economics; Group C remains necessary for real economic comparisons." | Matches D4 (a), with a narrower scope: synthetic engineering only. Group C is required before any real economic comparison. |
| D5 | "Long-only, sum-to-one, excess above a cap goes to cash, default cap 1.0 and no default turnover cap for plain baselines. Stricter constraints are explicitly registered for learned policies." | Matches D5's rules and defaults. **One clause awaits confirmation:** the packet says stricter sets are "identical for every compared policy", and the owner's words could be read as stricter sets for learned policies only. Until the owner confirms, a registered constraint set applies identically to every policy in a comparison, as this packet states. |
| D6 | "Defer a whole rebalance when any held or targeted asset lacks an executable close. A newer valid target supersedes an older deferred target. Never fabricate intermediate wealth or a return through an unresolved distribution gap." | Matches D6 as corrected on 2026-10-14. It also settles the sub-question: the newer target supersedes (`superseded_deferred_target`). |
| D7 | "Common initial build date, registered rebalance offset and consistent schedules across comparisons." | Matches D7. |
| D8 | "D8(a): PPO reward equals net log-wealth growth, without double-counting already incurred transaction costs." | Matches D8 (a). Option (b) is not approved. |

Not approved here: D9 (the Builder's buy-and-hold mix), D4 (b) (group C cash returns), D8 (b) and (c), and groups C and D of `t5_portfolio_formulas.md`.

- Recorded in the repository on 2026-10-14 with its substance unchanged; only this status block is added.
- Formula ids and option names refer to `docs/research/t5_portfolio_formulas.md` (sections 1, 2, 7 and 8).
- The T5a engine interface (`src/stock_agent/portfolio/transition.py`, `schedule.py`, `engine.py`, and `log_growth_reward` in `src/stock_agent/environment/reward.py`) raised `NotImplementedError` until this approval. It is implemented as approved in the T5a engine pull request; see `docs/research/t5a_engine.md`.
- `tests/reference/t5a_reference.py`, a test-only shares-and-cash ledger of the approved options, reproduces every number below (`tests/test_t5a_reference.py`). D4 (b) is group C and is not reproduced.
- D6 was corrected on 2026-10-14 after independent review: a rebalance now waits for every **held or targeted** asset, and the packet added a sub-question about a decision that arrives while a rebalance is deferred. The owner's 2026-10-15 answer (the newer target supersedes, `superseded_deferred_target`) is implemented in both the oracle and the engine and tested on each; it was held as a strict xfail until then.
- The approval, with its date and the owner's exact words, is recorded here and in `docs/PROJECT_STATE.md` before any formula is implemented.

The text below is the packet as approved (D1-D8 on 2026-10-15; see the approval record above). It asked only what needed the owner's judgment; everything else follows mechanically from the decisions (listed at the end). Every number below was computed by script (`t5a_v2_numbers.py`).

Notation:
- w̃ is the drifted weight vector just before the execution close, w* the post-trade weights, and Δ = w* − w̃.
- r is the next session's returns, c_i the all-in cost rate of asset i, and the last component of every vector is the cash account.

---

## D1. Execution timing (OD8.1, F1.1)

**Rule.** A decision made with information at the close of d trades at the close of d+1. The new weights first earn r_{d+2}.

**Example.** Switch from B (0% returns) to A, where A returns +10% on d+1 and −5% on d+2.
- L = 1 (next close): W = 1.00 × 0.95 = **0.950**.
- L = 0 (same close): W = 1.10 × 0.95 = **1.045**. That gains 10% the strategy could not have traded.

**Options.** L = 0 is unexecutable. Next open needs open prices, which we do not store. **Recommend L = 1, bound to the label's `entry_lag`.**

## D2. Transaction-cost base and decomposition (OD8.2, OD8.4, F2.1–F2.3, F2.6)

**Rule.**
- C = Σ_{i≠cash} c_i·|Δ_i|, with c_i = commission_i + half-spread_i + slippage_i (all per unit of traded notional).
- Applied multiplicatively: W⁺ = W⁻(1 − C), then 1 + R_net = (1 − C)(1 + w*ᵀr).
- The cash account is the funding leg and is free.

**Example.** W = 100,000; w̃ = (0.55, 0.25, 0.20); w* = (0.40, 0.40, 0.20); c = (10 bp, 10 bp, 0); r = (+2%, −1%, +0.01%).
- C = 0.0003 ($30) and R_gross = 0.402%.
- R_net = **0.37187940%**, so W = **100,371.8794**.
- Selling 100% of an asset into cash costs **99,900** with risky legs only. Charging the cash leg too would give 99,800, counting one trade twice.

**Rates (an assumption to approve, never tuned):** large single stocks 10 bp, broad equity and bond ETFs 5 bp, SHV 2 bp, cash 0. One all-in number per class, since we have no spread or slippage data to separate the parts. Every backtest is shown at 0×, 1× and 2×.

**Example.** Moving 20% from VTI to SHV costs 0.20 × 5 + 0.20 × 2 = **1.4 bp**.

**Recommend:** risky legs only, multiplicative, all-in rates as above.

## D3. Initial build and exit (OD8.3, F2.5)

**Rule.** Every portfolio and benchmark starts as 100% cash at one common first execution close and pays D2 to build. There is no exit charge.

**Example.** A 50/50 build at 10 bp costs C = 0.001, so W = **99,900**. One-way turnover is ½(0.5 + 0.5 + 1.0) = **1.0**.

**Recommend:** as stated. A free build favours high-turnover strategies' first period.

## D4. Cash and SHV are different instruments (OD8.5 in part; packet B9)

**Rule.**
- **Cash account.** Weight w_cash, trades free, earns r_cash,t.
- **SHV.** A fund with its own total return (approved F5.1) and a 2 bp trading cost, so it can lose money (duration ≈ 0.25–0.5).

**Open question: r_cash before group C (the T-bill-rate series) exists.**
- (a) **r_cash = 0, labelled "uninvested cash earns 0% (T-bill rate not yet approved)"** (recommended). This is conservative: it understates cash, so no strategy is flattered by holding it.
- (b) Wait for group C (DGS3MO, F3.1): r_cash,t = y_{t−1}·days/365. Example: 5% over Friday to Monday gives 0.000410959.
- (c) No cash account at all, with SHV as the only cash-like holding.

**Recommend:** (a) now. The ETF Lab's "cash" sleeve is SHV held as a fund. Switch to (b) when group C is approved.

## D5. Constraints, applied identically to every policy

**Rule.** Every policy's target goes through one deterministic feasibility map before `portfolio_step`. The map's changes are recorded (distance from the requested target), so no policy gains from a different API.
1. **Long only, summing to 1:** w ≥ 0 and Σw = 1. A violating rule target is an error, never silently repaired.
2. **Per-asset cap w_max** (cash exempt). Open: where does the excess go?
   - **Excess to cash (recommended: simple, never adds risk).** Target (0.70, 0.20, 0.10 cash) with w_max = 0.5 becomes **(0.50, 0.20, 0.30)**.
   - Proportional to the other assets: **(0.50, 0.333, 0.167)**.
3. **Turnover limit τ_max per rebalance.** If τ = ½Σ|w_tgt − w̃| > τ_max, trade part way: w* = w̃ + α(w_tgt − w̃), with α = τ_max/τ.
   - Example: w̃ = (0.70, 0.10, 0.20), target (0.30, 0.50, 0.20), τ = 0.40, τ_max = 0.25, so α = 0.625 and w* = **(0.45, 0.35, 0.20)**.
4. **Concentration:** the cap is the limit. The HHI is reported, not constrained.

**Open: default values.** Recommend w_max = 1.0 and no τ_max for the plain baselines (so equal weight is not altered). Stricter sets are registered per experiment, and identical for every compared policy.

**Drift above the cap between rebalances** is allowed and reported (not forced to sell). Recommend allowed.

## D6. Missing prices and unexecutable actions

**Rules (recommended):**
- **Execution.** If any **held or targeted** asset has no close at the execution session, the whole rebalance is deferred to the next session where all of them have one. Without every held asset's price, wealth, and so the trade sizes, cannot be computed. The reason `unexecutable_missing_close` is recorded, the drifted weights are held meanwhile, and nothing is filled. *(Corrected 2026-10-14 after independent review. The earlier wording, "any asset whose weight the target changes", could not be executed.)*
- **A new decision while a rebalance is deferred (open sub-question).**
  - **Recommended:** the newer target replaces the deferred one. The latest information wins, at most one target is ever pending, and the reason `superseded_deferred_target` is recorded.
  - Alternatives: execute the older target first, or drop the newer one.
- **Valuation across a gap.** Hold the share counts (no trading inside a gap) and value at the next session where every held asset has a close: W_end = W_start · Σ_i w_i·G_i, where G_i is asset i's growth over the span (the approved multi-session total return). This is exact at the span end. Intermediate wealth is reported as unavailable, never carried flat.
  - Example: W = 100 split 50/50 between A and B. A has no close on t1 and t2, and its t0→t3 return is +3%. B returns +1% per session. Then W_t3 = 50 × 1.03 + 50 × 1.01³ = **103.01505**, and W_t1 and W_t2 are unavailable.
- **Distribution in a gap (R5b).** A held fund's span return is unavailable, so the backtest or episode stops with the reason code `distribution_in_gap` rather than guessing.
- **RL.** No decision is taken on gap sessions. The span's reward is log(W_t3/W_t0) at the first complete session.

**Alternative:** partial execution of the executable legs, with the rest rescaled. It is more realistic for large portfolios, but it introduces a rescaling rule that every policy would have to share.

## D7. Rebalance schedule and offsets (OD8.10, F7.1–F7.2)

**Rule.** One common build date. Rebalance on decision sessions where (d − d_first) mod k = o, with one registered offset o shared by every compared portfolio. All k offsets are reported as one variant plus a k-member sensitivity family, and the offset never chooses the headline.

**Example** (packet B5): offsets 0 and 1 end at **0.98507** and **0.98755**. Report the registered one and the spread, 0.25%.

## D8. The RL/PPO reward (packet B8; reward changes need your approval)

**Base rule.** reward_t = log(1 + R_net,t), where R_net comes from `portfolio_step`. Costs are already inside it, so **no separate cost term**.
- Example: R_net = 0.37187940% gives reward = **0.0037118964**.
- Subtracting C again would give 0.0034118964, double counting $30. A test will fail if that happens.

**Open: how risk preference enters.**
- (a) **Log growth only, with risk limits through the D5 constraint set (recommended for the first registered PPO).** It is easy to audit, and it is the same objective as the wealth the comparison tables report.
- (b) Add a drawdown-increment penalty: reward = log(1 + R_net) − λ·max(0, DD_t − DD_{t−1}), with λ registered before training.
  - Example: a −1.5% day that deepens drawdown from 2.0% to 3.5% with λ = 2 gives −0.01511 − 0.03 = **−0.04511**.
  - This rewards something the wealth tables do not report, so it is shown as a separate variant.
- (c) Mean–variance: R_net − (γ/2)R_net².

**Recommend:** (a) as the contract, with (b) only as a separately registered variant with λ fixed in advance.

---

## These follow automatically once D1–D8 are approved

- **The drift identity** w_next = w*⊙(1 + r)/(1 + w*ᵀr) (packet B6). It is an accounting identity, not a choice.
- **The pure function** `portfolio_step` (OD8.12). It is an engineering contract: one function for backtests, optimizers, the RL environment and PPO.
- **Rewriting existing tests** that encode same-close execution, a cash-leg cost or a free build (packet B7, OD8.14).

## Not needed for T5a

- **Group C:** the cash-rate series and benchmarks.
- **Group D:** intervals, CAGR and the participation flag.

T5a runs on synthetic data and the approved total-return calculation without them.

**Suggested reply:** "Approve D1–D3, D6, D7; D2 rates as assumptions; D4 (a); D5 excess-to-cash with defaults w_max = 1, no τ_max; D8 (a)", or amend any item.
