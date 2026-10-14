# RL environment and PPO on synthetic data

Status (2026-10-14): engineering only. Nothing here is evidence that any
policy is a good investment, and no strategy comparison is published.

- The per-step economics are Track A's `portfolio_step` (spec F8.1,
  `t5_portfolio_formulas.md` section 8). Decisions D1-D8 are **pending owner
  approval**. Until they are approved and Track A's implementation merges,
  the environment runs on `rl/reference_transition.py`, a stand-in for the
  *proposed* rules, and every result carries the label
  `engineering_only_unapproved_transition`.
- The reward is the proposed D8 (a), `log(1 + R_net)`, pending approval.
- Standing owner rule: do not train or evaluate PPO on an economically
  uncorrected simulator. The `rl` package never imports
  `stock_agent.environment` (a test enforces it).

## 1. What is built (PR 1)

`src/stock_agent/rl/`, standard library plus NumPy only:

| Module | Role |
|---|---|
| `transition.py` | The `PortfolioStep` protocol, `StepResult`, `TransitionInfo` (approval label) and `check_step_result` (F8.1 invariants on any injected transition) |
| `reference_transition.py` | Engineering stand-in for the proposed D2 / F8.1 rules; delete when Track A's approved function merges |
| `reward.py` | `log(1 + R_net)`, labelled `proposed_d8a_pending_owner_approval` |
| `scenarios.py` | Versioned, seeded synthetic scenario library |
| `splits.py` | Train / dev / test folds by time and by seed; `LeakageError` |
| `observation.py` | Point-in-time snapshots, features, frozen normalization |
| `actions.py` | Logits to long-only weights; the likelihood argument |
| `env.py` | The environment: a scheduler around the injected transition |
| `replay.py` | `rollout`, episode records, exact replay of target sequences |
| `policies.py` | Buy-and-hold, equal weight, fixed balanced, inverse volatility, mean-variance |
| `optimizer.py` | Exact long-only mean-variance for small universes (no SciPy) |
| `learned.py` | Linear-softmax policy trained by the cross-entropy method |
| `safety.py` | Health gate and the deterministic fallback |

There is no new dependency. The environment follows the gymnasium
`reset` / `step` signatures without importing gymnasium; the PPO pull request
adds the thin `gymnasium.Env` wrapper together with the dependency review.

## 2. The environment

### 2.1 No economics inside

The environment decides **when** the injected transition is called and with
which inputs. Every return, cost and drift comes from

    portfolio_step(w_pre, target | None, r_next, cost_rate, cash_index) -> StepResult

and the reward from `rl.reward`. `check_step_result` validates every result
(weights on the simplex, 0 <= C < 1, R_net <= R_gross, finite values), so a
defective transition stops the episode instead of training a policy on
broken economics.

### 2.2 Timing (D1, L = 1) and gaps (D6)

The environment works on the window's **complete sessions**
c_0 < c_1 < ... (every asset has a close). One transition call per complete
session c_m trades at the close of c_m and earns each asset's span return
over (c_m, c_{m+1}].

```
close c_j       decide with information at the close of c_j
close c_{j+1}   the trade executes (cost charged on drifted weights)
(c_{j+1}, c_{j+2}]   the new weights first earn
```

- **Worked example (spec T1.1).** Start in cash. Decide at c_0 to hold 100% A.
  A returns +10% into c_1 and -5% into c_2. With L = 1 the trade fills at the
  close of c_1, so W(c_2) = 1.00 x 0.95 = **0.950**. Trading at the close of
  c_0 (L = 0) would give 1.10 x 0.95 = 1.045, a gain that could not have been
  traded. `test_a_decision_at_d_executes_at_d_plus_1_and_first_earns_d_plus_2_t1_1`
  pins 0.950.
- **Gaps.** No decision is taken on a gap session. A gap is held through
  (share counts fixed) and valued once, at the next complete session, by the
  span return G_i - 1, where G_i is the asset's growth over the span. The
  reward for that call is log(W_end / W_start), one span reward.
- **The universe-complete rule.** A trade waits for a session where every
  asset has a close. Corrected D6 (2026-10-14) requires the closes of every
  held or targeted asset; the two are equal whenever a policy holds or
  targets every asset (every softmax policy does), the stricter rule is the
  same for every compared policy, and because decisions happen only on
  complete sessions, a deferred rebalance never meets a newer decision (D6's
  open superseding sub-question does not arise here).
- **Schedule (D7).** Scheduled indices j >= o with (j - o) mod k = 0, up to
  T*, the last scheduled index <= J - 2. T* is the episode's final
  observation, on the schedule like every decision state, so a learner
  bootstraps from a state of the same kind; decisions are the scheduled
  indices before T*. Up to k - 1 trailing complete sessions of the window
  stay unbooked. Every compared policy uses the same k and o.
- **Start (D3).** 100% cash; the first execution close pays the build.

### 2.3 Episode accounting

`reset()` books the calls c_0..c_o (cash, identical for every policy) and
returns the observation at c_o. `step(target)` books the calls
c_{j+1}..c_{j'} up to the next decision j' and returns their summed log
growth: exactly what the action at c_j influences. Rewards telescope:
sum of rewards = log(W_final / W_first execution close).

The call at c_{j'} is booked before the observation at c_{j'} is built (the
transition API couples the trade at a close with the next span's earnings).
The observation reads only market rows <= c_{j'}, the post-trade weights at
c_{j'} (which do not depend on the next span's returns) and wealth marked at
closes <= c_{j'}, and `info` holds only facts known at the close of c_{j'}
(decision index and session, the last executed trade's session, cost and
turnover, and the transition and reward identity).
`test_the_observation_and_info_at_d_ignore_everything_after_d` proves both are
invariant to every return, close, feed lag and cost multiplier after c_{j'},
on the calm, missing-close, stale-feed and high-cost scenarios.

Because of the trade-and-earn transition API, the environment's own state
(`ledger`, `wealth_path`, the pending drifted weights) already contains the
span after c_{j'} when the observation is returned. A policy or learner must
never read environment state or the reward as an input: the observation is
the only decision input, and the reward is a training signal only.

The window end is a time limit (`truncated=True`, never `terminated`): the
investor would carry on, so a learner may bootstrap from the final state.
`truncated` depends on the window's remaining complete sessions, so it is an
episode-boundary signal, never an observation.

### 2.4 Replay and fairness

- `(scenario id, generator version, seed, action sequence)` gives an
  identical trajectory (`test_identical_inputs_replay_to_an_identical_ledger`).
- `test_replay_through_the_env_equals_a_plain_loop_on_the_same_transition`
  re-implements the schedule as a hand-written loop over the same transition
  and requires the wealth path to be bitwise equal on calm, gap and high-cost
  scenarios.
- Every policy runs through `replay.rollout`, the only entry point, and every
  recorded target sequence replays exactly through `ScriptedPolicy`. No
  policy can gain an execution or cost advantage from a different API.

## 3. Synthetic scenarios

An ETF-like universe: `equity_etf`, `bond_etf`, `short_treasury_fund` and a
`cash` account (last). Synthetic, illustrative parameters; never calibrated
to real funds.

| Regime | Equity drift / vol (annual) | Notes |
|---|---|---|
| calm | 6% / 15% | equity-bond correlation -0.2 |
| crash | -225% / 35% (log drift; 40-session median about -30%, under 1% of segments non-negative) | Student-t (4), bonds rally |
| high_correlation | 3% / 20% | equity-bond correlation 0.9 |
| extreme_volatility | 0% / 80% | Student-t (3), bond vol 25% |
| stale_feed | calm | the data feed lags 1-5 sessions (4% of sessions start a lag) |
| gappy | calm | 3% of fund sessions have no close |
| high_costs | calm | 5 x the base cost rates |

Scenarios repeat a regime schedule so that **every fold window contains its
stress regime** (tested): `calm`, `crash`, `high_correlation`,
`extreme_volatility`, `stale_observations`, `missing_closes`, `high_costs`,
and `mixed` (every regime in turn; the training scenario).

- **Replay.** Each draw comes from a legacy `RandomState` stream (frozen across
  NumPy versions) seeded by SHA-256 of (seed, scenario, generator version,
  purpose, segment). Golden tests pin every scenario: closes, feed lags, cost
  multipliers and regimes bitwise, and returns by per-asset sums to 1e-9
  (floats pass through platform maths libraries). An unsupported generator
  version is refused, never approximated.
- **Generator 1.1.0** (2026-10-14) made the crash regime reliably severe:
  1.0.0's crash segments were non-negative 8.9% of the time (median -22%).
  The version enters every seed, so 1.1.0 changes every draw; 1.0.0 is
  refused.
- **Returns.** Log returns are simulated and exponentiated, so every return
  exceeds -100%. The return at the close after a gap covers the whole gap.
- **Cash** earns 0 (D4 (a), pending): it understates cash, so no strategy is
  flattered by holding it. The short-Treasury fund therefore dominates cash
  in these scenarios; the registered fixed weights reflect that.
- **Costs.** Base rates are the proposed D2 rates: 5 bp for the equity and bond
  funds, 2 bp for the short-Treasury fund, 0 for cash, times the session's
  multiplier. The rate at the **execution** session is charged.
- **Stale feed versus gaps.** A gap is a market fact (no close; D6 economics).
  A stale feed is an information fact: execution still happens at true
  closes, but the observation sees returns only up to d - lag.

## 4. Folds by time and seed

`default_split_plan()`: train on `mixed`, seeds 0-15, sessions [60, 900);
dev on every scenario, seeds 100-103, [960, 1200); test on every scenario,
seeds 200-203, [1260, 1500); embargo 60 sessions. `SplitPlan` rejects
overlapping seeds and windows inside the embargo (`LeakageError`), and
`check_lookback(lookback)`, called by every trainer, requires the embargo to
cover the feature lookback and the train window to start after it.
Hyperparameters and checkpoints are chosen on dev only; the test fold is
reported once, after every choice is fixed.

## 5. Observations and leakage controls

At decision session d a policy sees:

- per fund: trailing compounded returns over 5, 20 and 60 sessions and
  sample volatility over 20 and 60 sessions, from data the feed has
  delivered (sessions <= d - obs_lag);
- the cost multiplier at d (not at the execution session d+1), when the
  scenario declares the cost regime observable;
- the post-trade weights after the close of d, and the drawdown of marked
  wealth.

Controls, each with a negative test:

| Control | Test |
|---|---|
| A `MarketSnapshot` holding any row after d cannot be built | `test_a_snapshot_with_rows_after_the_decision_session_is_rejected` |
| Features ignore every return after d | `test_features_at_d_ignore_every_return_after_d` |
| Data the feed has not delivered is invisible | `test_a_stale_feed_hides_returns_the_feed_has_not_delivered` |
| The cost regime is the one known at d | `test_the_cost_regime_is_the_one_known_at_d_not_at_execution` |
| Normalization is fitted on the train fold only | `test_normalization_is_fitted_on_the_training_fold_only` |
| Normalization never reads sessions after the train window | `test_normalization_never_reads_sessions_after_the_training_window` |
| The env's observation and info ignore every later return, close, feed lag and cost multiplier | `test_the_observation_and_info_at_d_ignore_everything_after_d` |
| A learned policy trains only with a normalizer fitted on its plan's train fold | `test_training_refuses_a_normalizer_fitted_outside_the_train_fold`, `test_training_refuses_an_identity_normalizer` |

Normalization: market features only (they do not depend on the policy, so
the statistics are fixed before any training), mean and standard deviation
over every complete decision session of the training fold, then frozen,
clipped at +/-10 and serialized with provenance (fold, seeds, window,
generator version, feature spec). A constant training feature is centered,
not divided by zero. Weights and drawdown are bounded and not normalized.

## 6. Actions and the likelihood

A learned policy outputs unconstrained logits z over all n assets, cash
included, and the environment holds w = softmax(z): long-only and summing to
1 with no projection or repair.

PPO samples z from a diagonal Gaussian and scores the ratio
pi_new(z | s) / pi_old(z | s) **on the sampled z**. The softmax is
deterministic and part of the environment, so the score-function gradient
E[grad log pi(z | s) A] is unbiased for the objective defined through
softmax(z). Scoring a projected action g(z) under pi would evaluate the
density of a point that was never sampled; projecting onto the simplex's
boundary even creates point masses the Gaussian does not have. So the rule
is: never project a sampled action and then score the projected action.

- `softmax` is shift-invariant: adding a constant to every logit never
  changes the reward. That direction gets no gradient signal, which does not
  affect the ratio. The action space is unbounded, so no clipping breaks the
  invariance.
- After approval, Track A's `feasible_target` (caps, turnover limits)
  composes with the map: w = feasible(softmax(z)). The ratio stays valid, but
  where a cap binds, moving z inside the capped region changes nothing the
  environment sees: those directions add variance and no signal.
  Evaluation should report how often caps bind.

`test_the_likelihood_is_a_density_over_logits_not_over_weights` shows two
logit vectors with identical weights and different densities.

## 7. Policies

All policies share one interface, `decide(observation) -> weights | None`,
and see the same observation at the same decisions of the same environment.

| Policy | Rule (registered before any result) |
|---|---|
| buy_and_hold | Buy 60 / 30 / 10 / 0 (equity, bond, short Treasury, cash) once |
| equal_weight | 1/3 in each fund, no cash, at every decision |
| fixed_balanced | Rebalance to 60 / 30 / 10 / 0 |
| inverse_volatility | w_i proportional to 1 / sigma_i (60 sessions), funds only |
| mean_variance | max mu'w - (5/2) w'Sigma w, long-only; sample mean and covariance over the 60-session window's complete rows, cash riskless |
| linear_softmax_cem | logit_i = b_i + beta_mom z(ret_20)_i + beta_vol z(vol_20)_i, cash 0; CEM |
| fallback_reference | Fixed balanced; hold on stale inputs |

- **Mean-variance** is solved exactly by enumerating the KKT system of every
  active-set pattern (at most 3^n small systems for n <= 8). Every candidate
  is feasible and the optimum's own pattern is among them, so the result is
  the global optimum without iteration counts or tolerances. Sixty-session
  sample means are very noisy (the standard error of a daily mean is about
  sigma / sqrt(60), several times a plausible daily premium), so this
  baseline trades a lot; that is its honest behaviour, not a defect.
  `mean_shrinkage` is a registered sensitivity, 0 by default.
- **Linear-softmax with CEM** (five parameters). CEM evaluates deterministic
  policies on the same training episodes, needs no likelihood or step size,
  and records every candidate, so the number of variants searched is known.
  REINFORCE would need a stochastic policy, a variance-reducing baseline and a
  learning-rate schedule, and its gradient estimates on financial rewards are
  far noisier than CEM's ranking at this size. Each iteration's mean
  parameters are a checkpoint; the best mean dev-fold log growth selects
  one. The test fold is never built during training (a test spies on it).
  Training takes a `SplitPlan` and refuses a normalizer not fitted on its
  train fold (or an identity one), an embargo shorter than the lookback and
  overlapping seeds. `LearnedResult.to_dict()` records the frozen normalizer
  and its provenance, the folds, the transition and reward identity with the
  engineering-only label, the environment schedule and every scenario's
  fingerprint.
- **Diagnostics are not metrics.** `EpisodeRecord.engineering_diagnostics()`
  (log growth, wealth ratio, largest observed decline, cost, turnover, all
  scaled to the first execution close) serves tests and health checks only.
  Risk and performance metrics are owner-gated formulas; none of these
  numbers may enter a strategy comparison table.

## 8. Health gate and fallback

Before a learned policy acts, `HealthGate` checks the observation and the
model's recorded health. Any reason switches the decision to the
deterministic fallback, and the reason is stored with the decision:

`stale_inputs` (feed lag > 2 sessions), `non_finite_features`,
`out_of_distribution` (a market z-score beyond 8 training standard
deviations), `model_unhealthy`, `invalid_action`, `policy_input_error`.

The fallback is the fixed balanced reference, except on stale inputs, where
it holds the current positions: the product rule is never to trade on stale
prices (`vision_and_prd.md`, section 15).

## 9. Limitations

- Synthetic data only. Regime parameters are illustrative and the shocks are
  Gaussian or Student-t with fixed correlation; real returns have volatility
  clustering, regime persistence and fat tails these scenarios only imitate.
- One path per seed; prices are exogenous (no market impact beyond the cost
  model).
- Gap returns inside a trailing window are multi-session returns, which
  inflates trailing volatility slightly in gappy regimes.
- The stand-in transition, the reward, the cost rates and the cash rate are
  all proposals pending approval.
- Distributions in gaps (D6, R5b) are not simulated: synthetic total returns
  carry no separate distribution events.

## 10. Learning notes

1. **Problem.** Build a simulator in which every allocation policy, from a
   fixed 60/30/10 rule to PPO, is evaluated under identical timing, costs and
   information, so differences come from the policies and not from the
   plumbing.
2. **Concepts.** Markov decision process; point-in-time information sets;
   self-financing wealth recursion; the score-function (likelihood-ratio)
   gradient; derivative-free optimization (CEM).
3. **Assumptions.** Trades fill at the close after the decision; prices are
   exogenous; costs are proportional to traded notional; the data-feed lag is
   known at decision time (its age is part of the observation).
4. **Formulation.** State s_j = (features at c_j, w_post(c_j), drawdown).
   Action z in R^n, w = softmax(z). Reward
   r_j = sum over m in (j, j'] of log(1 + R_net,m), so that
   sum_j r_j = log(W_T / W_1): maximizing the undiscounted sum maximizes log
   terminal wealth, the Kelly growth objective.
5. **Small example.** The T1.1 switch: 0.950 with next-close execution versus
   1.045 with same-close execution. A 9.5% difference from timing alone.
6. **Leakage risks.** Same-close trading; features computed from the close
   being traded; normalization fitted on all data; cost regimes read at the
   execution session; checkpoint selection on test data. Each has a test.
7. **How results could mislead.** A policy trained and evaluated on one
   synthetic generator learns that generator's quirks (for example, the
   short-Treasury fund dominating zero-yield cash). One good seed is luck;
   seeds are reported as a distribution (PR 2). A completed training run is
   an engineering result, never evidence of investment value.
8. **Practitioner validation.** Replay against an independent loop; invariance
   tests under perturbed future data; stress scenarios; seed distributions;
   comparison against simple baselines under identical costs.
9. **Learn next.** The PPO clipped objective and generalized advantage
   estimation (PR 2); why log utility is myopically optimal for i.i.d.
   returns, which makes a myopic mean-variance optimizer a strong baseline.

**Bounded exercise (optional).** Write a policy that holds 100% of the
short-Treasury fund and run it with `replay.rollout` on the `calm` scenario's
dev window. Before running, predict its terminal wealth from the regime's
2% annual drift and 0.6% volatility over the window's sessions, then compare.
