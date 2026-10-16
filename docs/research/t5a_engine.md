# T5a: the approved portfolio engine (implementation notes)

Status (2026-10-16): implemented as approved. The owner approved T5a
decisions D1-D8 on 2026-10-15 (`t5a_open_decisions.md`, approval record);
group B of `t5_portfolio_formulas.md` (F1.1-F1.3, F2.1-F2.6, F7.1-F7.2,
F8.1) is the specification. D9, D4 (b), D8 (b) and (c), and groups C and D
are not approved and are not implemented.

Everything here runs on synthetic data. Nothing in this document, the code or
the tests is evidence about any real strategy. No strategy comparison is
published: the registered comparison (baselines against PPO) is a separate,
later change, and comparison tables stay withheld until it releases them.

## 1. One source of portfolio economics

| Piece | Location | Approval |
|---|---|---|
| Per-step trade, cost, return and drift (`portfolio_step`) | `src/stock_agent/portfolio/transition.py` | D2, D3, F2.1-F2.4, F8.1 |
| Cost rates (`CostModel`, `PROVISIONAL_RATES`, `sensitivity`) | same | D2, F2.6 (provisional) |
| Feasibility map (`feasible_target`, `Constraints`) | same | D5 |
| Lag, schedule, offsets, deferral (`execution_lag`, `decision_sessions`, `execution_plan`, `offset_family`, `next_executable_session`) | `src/stock_agent/portfolio/schedule.py` | D1, D6, D7, F1.1, F1.3, F7.1, F7.2 |
| The caller (`collect_targets`, `simulate`, `simulate_cost_sensitivity`) | `src/stock_agent/portfolio/engine.py` | D1, D3, D4 (a), D6 |
| Common build date and schedule (`build_date`, `require_common_schedule`) | `src/stock_agent/evaluation/baseline.py` | D3, D7 |
| Reward (`log_growth_reward`) | `src/stock_agent/environment/reward.py`, `src/stock_agent/rl/reward.py` | D8 (a) |
| Health input (`backtest_summary`) | `src/stock_agent/portfolio/backtest_summary.py` | agreed format v1.0 |

Every simulator calls `portfolio_step`:

- `evaluation.baseline.run_rebalanced_backtest`, the rule baselines and the
  buy-and-hold benchmark run on `engine.simulate`;
- the RL environment (`rl.env.AllocationEnv`), every policy evaluation and
  PPO training inject it (`rl.transition.APPROVED_TRANSITION`, status
  `approved`); the engineering stand-in `rl/reference_transition.py` is
  deleted. The environment's one-session execution lag is checked against
  `execution_lag()` when it is built;
- `environment/portfolio_math.py`, the pre-T5a helpers, is imported by no
  production module (a test enforces it). It stays, unused, until the owner
  approves deleting it and its tests.

`scripts/run_t5a_baselines.py` keeps the pre-T5a step,
`retired_legacy_transition`, only to reproduce the old engine in the
attribution (section 2). No module under `src/` imports or reimplements it
(tested).

### The formulas, with the owner's words

- **D1** ("Next-session-close execution."). A decision at the close of d
  trades at the close of d + 1 and first earns the return dated d + 2.
  `execution_lag()` returns `MODEL_LABEL_SPEC.entry_lag` (1).
- **D2** ("Costs on actual risky-asset traded notional, multiplicative
  accounting. Provisional assumptions of 10 bp single stocks, 5 bp broad
  ETFs, 2 bp SHV and zero cost for the cash funding leg. Report 0×, 1× and 2×
  sensitivity.")

      Δ = w* − w̃                        (from the drifted weights)
      C = Σ_{i≠cash} c_i·|Δ_i|           (the cash leg is never charged)
      1 + R_net = (1 − C)(1 + w*ᵀr)
      w_next = w* ⊙ (1 + r) / (1 + w*ᵀr)
      τ = ½ Σ_all |Δ_i|,  traded notional = Σ_{i≠cash} |Δ_i|

  The rates are `PROVISIONAL_RATES` (`single_stock` 10 bp, `broad_etf` 5 bp,
  `tbill_fund` 2 bp, `cash` 0), registered in code and never tuned;
  `CostModel.sensitivity()` gives the 0×, 1× and 2× models and
  `engine.simulate_cost_sensitivity` runs all three on the same targets.
- **D3** ("Begin in cash, charge initial construction and use no terminal
  liquidation charge for marked-to-market comparisons."). Every portfolio and
  every benchmark starts as 100% cash at the common first execution close and
  pays D2 for the build. There is no exit charge. The benchmarks are broad
  equity price indices, bought through their tradable proxy, so the build
  pays D2's broad-ETF rate (5 bp).
- **D4 (a)** ("Uninvested cash earns 0% for synthetic engineering. SHV is a
  separately traded ETF. Do not claim this is realistic interest-bearing cash
  economics; Group C remains necessary for real economic comparisons.").
  Callers supply `engine.UNINVESTED_CASH_RETURN` (0.0) and label results
  "uninvested cash earns 0% (synthetic engineering only; group C pending)"
  (`engine.CASH_RETURN_LABEL`). Because group C cash returns (D4 (b)) are not
  approved, the baselines refuse a nonzero cash return ("group C cash returns
  are not approved"). The engine itself books whatever cash return it is
  given, so its mechanics are also tested with nonzero rates.
- **D5** ("Long-only, sum-to-one, excess above a cap goes to cash, default
  cap 1.0 and no default turnover cap for plain baselines. Stricter
  constraints are explicitly registered for learned policies."). A target
  that is not long-only or does not sum to 1 (within 1e-9) raises
  `InvalidTarget`, never repaired. A weight above `w_max` is cut to the cap
  and the excess goes to cash. If the one-way turnover τ exceeds `tau_max`,
  the trade goes part way: w* = w̃ + α(w_tgt − w̃), α = τ_max/τ. The record
  (`FeasibleTarget`) keeps the requested target, the executed one, their
  one-way distance, the binding constraints and α. Defaults: `w_max = 1`, no
  `tau_max`. Drift above the cap between rebalances is kept and reported.
  The clause awaiting confirmation is handled as the packet states: one
  registered constraint set applies identically to every policy in a
  comparison (the RL environment's `EnvConfig.max_weight` and `max_turnover`
  are shared by every method in a harness run).
- **D6** ("Defer a whole rebalance when any held or targeted asset lacks an
  executable close. A newer valid target supersedes an older deferred
  target. Never fabricate intermediate wealth or a return through an
  unresolved distribution gap."). A rebalance waits for the first session
  where every held or targeted asset has a close (`unexecutable_missing_close`).
  A newer target decided meanwhile replaces it (`superseded_deferred_target`).
  Holdings are valued only where every held asset closes; a gap is held
  through with fixed share counts and booked in one `portfolio_step` call over
  the span, W_end = W_start · Σ w_i·G_i, so the packet's example gives
  103.01505 and the sessions inside the gap have wealth None. A held asset
  whose close has no available return stops the run (`distribution_in_gap`).
- **D7** ("Common initial build date, registered rebalance offset and
  consistent schedules across comparisons."). Every compared run is given one
  `build_date`; rebalance decisions are the dates d after it with
  (d − build) mod k = o, so offsets count from the common build.
  `require_common_schedule` refuses a comparison whose portfolios and
  benchmarks build on different dates or whose portfolios rebalance on
  different schedules. `offset_family` returns all k offsets as one variant
  with a k-member family; the registered offset is the result and the family
  is its sensitivity, never a choice.
- **D8 (a)** ("PPO reward equals net log-wealth growth, without
  double-counting already incurred transaction costs."). reward =
  log(1 + R_net). The penalised `calculate_reward`, which subtracted costs a
  second time and is not an approved reward, is removed.

### Interpretations pending the owner's confirmation

Each is the simplest reading consistent with the owner's words, implemented
identically in the engine, the oracle and the tests.

- **(a) A hold does not supersede.** A decision of None ("keep the drifted
  holdings") is not a target, so it does not replace a deferred target,
  which keeps waiting. Otherwise a buy-and-hold whose build is deferred would
  cancel its own build at its next (hold) decision and stay in cash.
- **(b) The turnover limit also limits the build.** One feasibility map
  applies to every target, so with `tau_max` set a build from cash is
  partial: a 50/50 build with τ_max = 0.25 executes (0.125, 0.125, 0.75 cash).
  This matches the reference oracle written to the packet. The plain
  baselines register no limit, so it does not arise there.

Other details, by the packet's own rules: weight changes below 1e-12 are dust
and no trade (spec 2 (f)); the cash entry of `cost_rate` is never charged,
whatever its value (F2.2). The RL environment decides and trades only on
sessions where every asset closes (the universe-complete form of D6, with the
packet's "no decision is taken on gap sessions"); it equals the engine's rule
whenever every fund is held or targeted, which is true of every softmax
policy, and it is stricter otherwise, identically for every method.

## 2. The baseline migration, one change at a time

`scripts/run_t5a_baselines.py --attribution` takes the equal-weight baseline
(three Gaussian random-walk assets, seed 20261015; rebalance every 5
sessions; 10 bp) and adds the approved rules one at a time. The `legacy` row
reproduces the pre-T5a `run_rebalanced_backtest` exactly (tested to 1e-12,
and the same reproduction is tested for every baseline); the `build` row is
the production baseline. Δ is the change in terminal wealth from the
previous row, in basis points of starting wealth. This attributes an engine
change; it compares no strategies.

| Step | Change | Δ (bp) |
|---|---|---|
| legacy | lag 0, cash leg charged on two-way turnover, additive, free build | (ratio 1.253623) |
| lag | next-close execution (D1) | +54.6 |
| cost_base | risky legs only, multiplicative (D2) | −0.0 |
| build | charged build from cash (D3) | −12.6 |

Equal weight's rebalance phase is offset 0 before and after D7, so the
schedule change does not move it. How to read it:

- **Lag.** The execution moves one close later and the first earned return
  is dropped. On i.i.d. synthetic returns there is no short-term reversal or
  continuation to exploit, so the sign of this step is noise here. On real
  data, same-close execution of a close-based signal is systematically
  optimistic whenever returns reverse or continue at short horizons; that,
  not this table, is why D1 matters.
- **Cost base.** Equal weight holds no cash, so moving the cost off the cash
  leg changes nothing; only the additive-to-multiplicative change moves
  wealth, by less than 1 bp. A rule that trades in and out of cash would
  move more, because the legacy engine charged both legs of every such trade.
- **Build.** The portfolio now pays 10 bp on its initial purchase: about
  −10 bp, compounded.

## 3. The `backtest_summary` health input

`scripts/run_t5a_baselines.py` writes one file per baseline to
`reports/health_inputs/backtest_summary/<run_id>.json` (ignored by Git), in
the agreed v1.0 format, with `transition_status` "approved", `synthetic`
true and `evidence_status` "synthetic_engineering_only". Each file has its
own `run_id` and the time it was generated.

| Measure | Definition | Compared with |
|---|---|---|
| `turnover` | largest one-way turnover of a single execution, build included | `max_turnover` = D5 τ_max (null when none) |
| `costs_bps` | summed C of every execution, each in bp of its pre-trade wealth; not annualized | `max_cost_bps` (null: D5 registers none) |
| `max_weight` | largest non-cash weight after a trade or at a valued close (drift shows) | `max_weight` = D5 w_max |
| `max_drawdown` | F6.5 on the simulated pre-trade wealth path; null if any close is unavailable (R5c) | `max_drawdown` (null: none registered) |

`units` carries these definitions and `assumptions` the cost rates and
multiplier, the cash label, the lag, the build date and the schedule. The
health reader ignores both (extra fields). The F2.4 calendar-year
annualization of turnover and an annualized cost figure are not used: their
calendar basis is a group D matter (F6.1), and the per-execution maximum is
the quantity D5's limit constrains.

## 4. Validation

- **Hand examples.** Every number in the packet: the rebalance (C = 0.0003,
  R_net = 0.37187940%, W = 100,371.8794, w_next = 0.40637, 0.39441,
  0.19922), stocks → cash 99,900 (99,800 fails), the build 99,900 with τ = 1,
  multiplicative 1.0989 (1.0990 fails), the cross-class 1.4 bp and 2.0 bp, the
  D5 cap and turnover examples, the D1 switch 0.950 (1.045 fails), the label
  tie −1.00%, the D6 gap 103.01505, and the D7 offsets 0.98507 and 0.98755.
- **Oracle equivalence.** `tests/reference/t5a_reference.py`, a shares-and-
  cash ledger that imports nothing from `stock_agent`, against the engine
  over 40 random gappy paths with deferrals, superseded targets, holds, caps,
  turnover limits and stops: identical wealth within 1e-12 and identical
  execution records.
- **Engine and RL environment.** The backtest engine, the RL environment and
  the oracle give one wealth path within 1e-12 on random gappy paths, with no
  constraint, a binding cap, and a binding turnover limit. A simulator with a
  cost or timing advantage is detected.
- **Execution delay.** A lag-0 engine, an engine whose holdings earn the
  execution close's return, a caller that shows a decision its next row, and
  an RL environment whose lag disagrees with `execution_lag()` are each
  detected.
- **Invariants.** Seeded property tests of F8.1 (weights on the simplex
  within 1e-12, 0 ≤ C < 1, R_net ≤ R_gross, pure drift without a trade, an
  all-cash portfolio earns the cash return, telescoping wealth, costs move
  wealth but never weights, a nonzero cash cost entry is never charged).
- **Determinism.** A simulation and a target collection replay bit for bit.
- **Mutation probes (2026-10-16).** 48 targeted mutants, one per guarantee,
  run against the T5a, baseline, benchmark, reward and RL suites: 46 killed.
  Both survivors are equivalent, because the guarantee is enforced twice
  (the held-asset half of D6's deferral, also blocked by valuation; the
  momentum baseline's cash-rate refusal, also made by `backtest_inputs`),
  and each mutant that removes both guards is killed.

## 5. Limitations

- Synthetic data only. The cost rates are provisional assumptions with no
  spread or impact data behind them, and cash earns 0%: neither is a claim
  about real trading costs or interest-bearing cash (group C).
- Trades fill at the close at the close price; the O(c²) fixed point of
  paying costs from the traded amount is ignored (F2.3).
- Benchmarks are still price indices, charged D2's broad-ETF rate to build;
  a fund total-return benchmark (F4.1) is group C.
- No performance metric beyond the descriptive drawdown is approved for
  simulations (group D: CAGR basis, intervals, participation).
- The RL environment's gap rule is stricter than the engine's when a policy
  neither holds nor targets some fund (section 1).

## 6. Learning notes

1. **Problem.** Make every simulated return executable, cost-complete and
   identical across every API that produces it: a backtest loop, an
   optimizer replay and an RL environment.
2. **Concept.** A self-financing portfolio: wealth changes only through
   returns and costs. With pre-trade weights w̃, post-trade weights w* and
   cost C as a fraction of pre-trade wealth, W⁺ = W⁻(1 − C) and
   W_next = W⁺(1 + w*ᵀr). Drift is an accounting identity: holding share
   counts fixed, each weight grows by its own return relative to the
   portfolio's.
3. **Assumptions.** Fills at the close; a constant all-in rate per unit
   traded; no market impact; 0% cash; fractional shares; no taxes.
4. **Small example.** W = 100,000; w̃ = (0.55, 0.25, 0.20 cash);
   w* = (0.40, 0.40, 0.20); 10 bp; r = (+2%, −1%, +0.01%). Δ = (−0.15, +0.15,
   0), so C = 0.001 × 0.30 = 0.0003 ($30) and τ = 0.15. Gross return
   0.4 × 2% − 0.4 × 1% + 0.2 × 0.01% = 0.402%; net
   (1 − 0.0003)(1.00402) − 1 = 0.37188%; reward log(1.0037188) = 0.0037119.
   Subtracting C again would give 0.0034119, counting $30 twice.
5. **Leakage risks.** Same-close execution of a close-based signal (lag 0);
   a caller that hands a decision its own next return; carrying wealth flat
   across a gap (a fabricated return of 0); runs that build on different
   dates; and choosing the rebalance offset or cost rate after seeing
   results.
6. **Ways results could mislead.** One rebalance phase is one draw among k,
   so a phase change alone can move a high-turnover rule by more than any
   cost assumption; report the offset family. Costs at 1× only hide how
   fragile a high-turnover rule is; 0× and 2× show it. A 0% cash rate
   understates cash-heavy rules. Synthetic random walks have no exploitable
   structure, so no ranking of rules on them means anything.
7. **Practitioner validation.** Reconcile the weights-based engine against an
   independent shares-and-cash ledger; replay identical targets through every
   API; plant timing and cost mutants and check the tests kill them; report
   cost and phase sensitivity with every result (GIPS-style net-of-cost,
   time-weighted returns).
8. **Next to learn.** Rebalance-timing luck and tranching (averaging across
   offsets); the square-root market-impact model; how a T-bill cash rate
   (group C) changes cash-heavy rules; the deflated Sharpe ratio and the
   effective number of trials when a k-member offset family counts as one
   variant.

**Bounded exercise (optional).** In `tests/test_t5a_engine.py`, write a test
that a 60/40 portfolio rebalanced every session at 2× costs ends below the
same portfolio at 1×, and predict the gap from the summed one-way turnover
before running it.
