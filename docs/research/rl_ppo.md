# RL environment and PPO on synthetic data

Status (2026-10-15): synthetic engineering. Nothing here is evidence that
any policy is a good investment, and no strategy comparison is published.

- The per-step economics are Track A's `portfolio_step` (spec F8.1,
  `t5_portfolio_formulas.md` section 8), approved by the owner on 2026-10-15
  (T5a D1-D8, `t5a_open_decisions.md`). The environment and every script
  inject it (`rl.transition.APPROVED_TRANSITION`), so every result carries
  the transition status `approved`. The engineering stand-in used before the
  approval (`rl/reference_transition.py`, label
  `engineering_only_unapproved_transition`) is retired.
- Every target passes through the engine's D5 feasibility map
  (`feasible_target`) with the registered constraint set
  (`EnvConfig.max_weight`, `EnvConfig.max_turnover`; defaults cap 1.0 and no
  turnover limit), identically for every method.
- The reward is D8 (a), `log(1 + R_net)`, approved; (b) and (c) are not.
- Uninvested cash earns 0% (D4 (a), approved for synthetic engineering only;
  group C pending), and the cost rates are D2's provisional assumptions.
- Standing owner rule: do not train or evaluate PPO on an economically
  uncorrected simulator. The `rl` package never imports
  `stock_agent.environment` (a test enforces it).

## 1. What is built (PR 1)

`src/stock_agent/rl/`, standard library plus NumPy only:

| Module | Role |
|---|---|
| `transition.py` | The approved `portfolio_step` and its `StepResult` (re-exported from `portfolio.transition`), `PortfolioStep` (the engine's `Transition` protocol), `TransitionInfo` (approval label, `APPROVED_TRANSITION`) and `check_step_result` (F8.1 invariants on any injected transition) |
| `reward.py` | `log(1 + R_net)`, D8 (a), labelled `approved` |
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

PR 1 adds no dependency: the environment follows the gymnasium
`reset` / `step` signatures without importing gymnasium.

PR 2 (PPO) adds, with the learning libraries confined to two modules (a
test enforces it):

| Module | Role |
|---|---|
| `config.py` | Validated, versioned YAML run configuration (`configs/rl/`); unknown keys are errors; `sha256()` identifies it |
| `gym_env.py` | `gymnasium.Env` wrapper: fund logits in, the same `AllocationEnv` underneath; seeded episode sampling inside the train fold |
| `ppo.py` | stable-baselines3 PPO training per seed, dev-fold checkpoint selection, checkpoints with frozen normalization and hashes |
| `health.py` | The `rl_training_health` JSON report (no learning library needed) |
| `harness.py` | The comparison harness, fairness checks, and the publication rule |
| `forecasts.py` | The forecast-input interface (interface only until Track B's artifacts exist) |

Scripts: `scripts/train_ppo.py` and `scripts/evaluate_rl.py` (section 9.3).

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
- **Independent oracle (PR 2).** `tests/test_rl_oracle_replay.py` drives the
  environment through Track A's replay harness (`tests/reference/replay.py`)
  and matches the wealth path of Track A's oracle, a loop-based
  shares-and-cash ledger of the recommended D1-D8 rules that imports nothing
  from `stock_agent`, within 1e-12 on random paths with missing closes and
  holds. A planted cost defect is detected.

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

A learned policy outputs one logit per fund; the cash account is the
reference asset with its logit pinned at 0. With z the n logits (cash's
inserted as 0), the environment holds w = softmax(z): long-only and summing
to 1 with no projection or repair.

PPO samples a from a diagonal Gaussian and scores the ratio
pi_new(a | s) / pi_old(a | s) **on the sampled a** (stable-baselines3 stores
the raw sample in its rollout buffer and scores that; a test checks our
closed-form Gaussian density against its value). The map
f(a) = softmax(pad_cash(clip(a))) is deterministic and part of the
environment, so the score-function gradient E[grad log pi(a | s) A] is
unbiased for the objective defined through f(a). Scoring a projected action
g(a) under pi would evaluate the density of a point that was never sampled;
projecting onto the simplex's boundary even creates point masses the
Gaussian does not have. So the rule is: never project a sampled action and
then score the projected action.

- **Why cash is pinned.** softmax is shift-invariant: with n free logits,
  adding a constant to every logit never changes the weights, so that
  direction gets no gradient signal and can drift. stable-baselines3 requires
  a finite action box and clips to it, and clipping is not shift-invariant, so
  a drifting common mode would eventually distort the weights. Pinning cash
  removes the direction: n - 1 logits identify the weights exactly.
- **The bound.** Fund logits are clipped to [-10, 10]. The clip binds only at
  saturation (a fund-to-cash weight ratio beyond e^10, about 22,000). There
  the weights barely move and the action has no gradient signal, exactly like
  a binding cap.
- **After approval**, Track A's `feasible_target` (caps, turnover limits)
  composes with the map: w = feasible(softmax(z)). The ratio stays valid, but
  where a cap binds, moving z inside the capped region changes nothing the
  environment sees: those directions add variance and no signal. Evaluation
  should report how often caps bind.

`test_the_likelihood_is_a_density_over_logits_not_over_weights` shows two
logit vectors with identical weights and different densities, and
`test_clipped_samples_keep_their_raw_logits_and_likelihoods` checks the
library: under a wide policy many sampled logits exceed the bound, SB3 stores
the raw samples, the environment receives the clipped ones, and re-scoring
the stored raw actions reproduces the stored log probabilities.

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
  and its provenance, the folds, the transition and reward identity with
  their approval status, the environment schedule and every scenario's
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

## 9. PPO training (PR 2)

### 9.1 Library and dependency review

PPO comes from stable-baselines3 (SB3) on gymnasium and CPU torch, pinned
exactly, each released at least two weeks before 2026-10-14:

| Package | Pin | Released | License |
|---|---|---|---|
| stable-baselines3 | 2.9.0 | 2026-06-15 | MIT |
| gymnasium | 1.3.0 | 2026-04-22 (1.4.0 is 9 days old, excluded) | MIT |
| torch | 2.14.0 | 2026-09-02 (2.14.1 is exactly 14 days old, excluded as borderline) | Apache-2.0 |

How torch is installed (`requirements.txt`, one manifest for local and CI
gates, so every PPO test runs in CI):
- **macOS:** `torch==2.14.0 ; sys_platform == "darwin"` from PyPI (the arm64
  wheel, 127 MB, PyPI sha256 `c1f844f1...`).
- **Linux x86_64 (CI):** a direct URL to PyTorch's CPU-only wheel
  `torch-2.14.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl` (196 MB) with its
  published sha256 (`a09987c9...`) as the URL fragment, which pip verifies. Its
  metadata (hash checked against the index) declares no CUDA, cuDNN, NCCL or
  triton dependency. SB3 2.9.0's `torch>=2.8,<3.0` accepts `2.14.0+cpu`.
- **Why not the alternatives.** PyPI's Linux torch pulls the CUDA 13 stack
  (several GB per CI run). A global `--extra-index-url` would let a second
  index serve any unpinned package: the dependency-confusion pattern of the
  2022 torchtriton incident. A separate optional manifest would let CI skip
  the PPO tests, breaking identical local and CI gates.
- **Deployment note.** No torch line matches Linux on other architectures
  (for example aarch64) or Intel macOS (PyTorch publishes no Intel macOS
  wheels for 2.14). There, installing SB3 would pull PyPI's default torch
  build; such a deployment needs its own pinned line first.

- **Value.** A maintained, widely used PPO whose loss, advantage estimation
  and logging are tested by others. A hand-written PPO is a classic source of
  silent bugs (advantage sign, normalization, value clipping, bootstrap at
  time limits). SB3 stores the raw sampled action and scores it (checked by a
  test), which is what the likelihood argument in section 6 requires.
- **Alternatives.**
  - A NumPy PPO in this repository: no dependency, but more code to verify;
    a good owner exercise later, against SB3 as a reference.
  - CleanRL: single-file scripts, not a library; we would copy and own the
    code.
  - RLlib (Ray): far heavier, distributed-first.
  - TorchRL: a more complex API.
  - JAX-based PPO: needs JAX.
- **Maintenance.** Exact pins. Any upgrade is an owner decision. SB3 releases a
  few times a year; torch quarterly.
- **Compute.** CPU only, one thread (for reproducibility). The smoke run
  trains three seeds and the CEM baseline in about 85 seconds, and evaluates
  in about 10 seconds, on an Apple-silicon laptop. Install size: the macOS
  arm64 torch wheel is 127 MB (590 MB installed); the Linux CPU wheel is
  196 MB with no GPU libraries.
- **Security.** SB3 model files contain pickled objects, and loading one
  can execute code. Only files this pipeline wrote are loaded, through
  `load_ppo_checkpoint`, which verifies the SHA-256 recorded at training time.
  That proves integrity, not trusted provenance. There are no network calls
  at run time. The Linux wheel comes from one exact URL with a verified
  hash, not from a second package index, so no other package can be served
  from PyTorch's host.
- **Deployment.** The libraries are confined to `rl/gym_env.py` and
  `rl/ppo.py` (a test enforces it), and importing the rest of `rl` loads none
  of them, so the product app never pays for them.

### 9.2 Training procedure

`ppo.train_ppo_seed` trains one seed:

1. **Seeding.** Python, NumPy and torch are seeded and torch uses one CPU
   thread. Each vectorized environment copy is seeded `seed + index`, and
   episodes (scenario, seed and window inside the train fold) are drawn from
   the environment's own generator.
2. **Policy.** An MLP (tanh) diagonal Gaussian over the fund logits, with
   separate policy and value networks.
3. **Selection.** The deterministic policy (the Gaussian mean) is evaluated
   on every dev-fold episode before training, every `eval_every` timesteps and
   at the end. The best mean dev log growth is the checkpoint (an untrained
   policy can win, and is then reported as such). The test fold is never built
   during training (a test spies on it).
4. **Reward.** The environment's `log(1 + R_net)` summed per decision, with
   no scaling or extra terms. PPO normalizes advantages per batch.
5. **Discount.** gamma = 0.99 per decision (five sessions in the registered
   configs) is a variance-reduction choice; the undiscounted sum is the log
   wealth objective.
6. **Truncation.** The window end is truncation, not termination, so SB3
   bootstraps the value there.

### 9.3 Reproducibility

```bash
.venv/bin/python scripts/train_ppo.py --config configs/rl/ppo_smoke.yaml --output-root models/rl
```

```bash
.venv/bin/python scripts/evaluate_rl.py --run-dir models/rl/<run_id> --fold test
```

Each run directory (ignored by Git) holds:
- `run.json`: configuration and its SHA-256, Git SHA and dirty flag, library
  versions, transition and reward identity, normalizer provenance;
- per seed, `checkpoint.json`: frozen normalizer, feature spec, scenario
  generator version, selection record, model hash;
- `training_health.json` (the detailed report) and `cem_baseline.json` (the
  CEM baseline with its normalizer, folds, identity labels and scenario
  fingerprints).

The Model Health Center's input, in the format agreed with the lead (v1.0),
goes to `reports/health_inputs/rl_training_health/<run_id>.json` (ignored by
Git; `--health-inputs-root` changes the root). A run that raises still writes
it, with status `failed` and every check `not_run`.

Evaluation provenance and accounting:
- `load_ppo_checkpoint` refuses a checkpoint from another scenario generator
  version or (when given) another config hash, and warns loudly when the
  current code SHA differs from the training SHA; the evaluation report
  records it.
- Every evaluation is write-once: `evaluation_<fold>_<n>.json`, numbered per
  fold, plus one line appended to `evaluations.jsonl`, so every look at the
  test fold is counted in the run directory.
- Training and evaluation refuse a normalizer not fitted on the plan's train
  fold, with this feature spec and this generator version.
- The gym wrapper accepts a requested evaluation episode only from its own
  fold and window (`LeakageError` otherwise).

The same configuration, code SHA and library versions give identical
parameters (`test_the_same_seed_replays_to_identical_parameters...`).
Seeds are always reported as a distribution (`seed_spread`), never the best
seed alone.

### 9.4 Training-health JSON (`rl_training_health` 1.0.0)

For the Model Health Center and the deployment gate. Per seed:
- reward statistics before any value bootstrap (n, mean, std, min, max,
  non-finite count);
- approximate KL, clip fraction, entropy, value loss, explained variance and
  action standard deviation (first, last, mean, max over updates);
- the largest absolute parameter and the count of non-finite parameters;
- the imputed-input share;
- the dev selection record.

Status:
- **unhealthy:** no updates, a non-finite reward, loss, parameter or dev
  score, or a parameter above `max_param_abs`;
- **warning:** an approximate-KL spike, a high clip fraction, or collapsed
  action noise;
- **healthy** otherwise.

The run is as bad as its worst seed. `model_is_healthy(report, seed)` feeds
`GuardedPolicy(model_healthy=...)`, so an unhealthy model never acts.

`health_input_record` condenses the report into the agreed v1.0 input:
the common envelope (`artifact`, `schema_version`, `generated_at` with
microseconds, `code_sha`, `git_dirty`, `synthetic`, `run_id`, `algorithm`,
`status`, `transition_status`, `seeds`, `scenario_version`) and eight checks,
each with Track C's own status, value, unit, threshold and detail:

| Check | Value | Status |
|---|---|---|
| `nan_or_inf` | non-finite rewards, parameters and dev scores, plus seeds with a non-finite loss | fail above 0 |
| `exploding_values` | largest absolute parameter | fail above `max_param_abs` |
| `approx_kl` | largest per-seed maximum | warn above `kl_warning` |
| `clip_fraction` | largest per-seed mean | warn above `clip_fraction_warning` |
| `entropy_collapse` | smallest final action standard deviation | warn below `min_action_std` |
| `value_loss` | largest last-to-first value-loss ratio | warn above 10, fail if non-finite |
| `reward_trend` | median over seeds of final minus initial dev score | warn below 0 (a training diagnostic, not a comparison) |
| `seed_spread` | standard deviation of the selected dev scores | warn with fewer than 3 seeds |

CEM and PPO dev scores are training diagnostics. They stay in the run
directory's detailed reports and never enter a published view; the health
input carries only the spread and the median trend, never the scores
themselves.

### 9.5 Comparison harness and the publication rule

`harness.run_comparison` runs every method on every (scenario, seed) of one
fold, with the same window, schedule, costs, transition and frozen
normalizer, and `check_fairness` verifies that the records agree on
everything but the policy. The methods are:
- buy-and-hold, equal weight, fixed balanced, inverse volatility,
  mean-variance and the CEM linear-softmax baseline;
- PPO per seed without forecasts, raw and guarded (as groups);
- PPO with forecasts, reported unavailable until Track B's versioned
  out-of-fold forecast artifacts exist (`forecasts.require_certified` refuses
  any provider that is not out-of-fold and point-in-time).

Until the transition is owner-approved, `engineering_report()` contains no
performance figure of any method, and `comparison_table()` raises
`UnapprovedTransitionError`. Once approved, the table reports each method's
mean episode log growth (the sum of the D8 rewards) with trade and fallback
counts; risk and performance metrics join only from the approved modules,
and episode diagnostics never do.

### 9.6 Stress tests

The dev and test folds contain every stress scenario: crash, high
correlation, stale observations, extreme volatility, high costs, missing
closes. Tests run a trained policy through each (finite, positive wealth;
valid weights; the guard engages on stale inputs). A completed run is an
engineering result, never evidence that PPO is investment-ready.

## 10. Limitations

- Synthetic data only. Regime parameters are illustrative and the shocks are
  Gaussian or Student-t with fixed correlation; real returns have volatility
  clustering, regime persistence and fat tails these scenarios only imitate.
- One path per seed; prices are exogenous (no market impact beyond the cost
  model).
- Gap returns inside a trailing window are multi-session returns, which
  inflates trailing volatility slightly in gappy regimes.
- The cost rates are D2's provisional assumptions and the 0% cash rate is
  D4 (a), approved for synthetic engineering only: neither is a claim about
  real trading costs or interest-bearing cash (group C pending).
- The environment's gap rule is the universe-complete form of D6: it decides
  and trades only on sessions where every asset has a close. That equals the
  engine's held-or-targeted rule whenever every fund is held or targeted (all
  softmax policies) and is stricter otherwise, identically for every method.
- Distributions in gaps (D6, R5b) are not simulated: synthetic total returns
  carry no separate distribution events.

## 11. Learning notes

1. **Problem.** Build a simulator in which every allocation policy, from a
   fixed 60/30/10 rule to PPO, is evaluated under identical timing, costs and
   information, so differences come from the policies and not from the
   plumbing.
2. **Concepts.** Markov decision process; point-in-time information sets;
   self-financing wealth recursion; the score-function (likelihood-ratio)
   gradient; derivative-free optimization (CEM).
3. **Assumptions.** Trades fill at the close after the decision; prices are
   exogenous; costs are proportional to traded notional; the data feed's lag
   is known at decision time (the health gate reads it).
4. **Formulation.** State s_j = (features at c_j, w_post(c_j), drawdown).
   Action a in [-10, 10]^(n-1) (fund logits), w = softmax(pad_cash(a)). Reward
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

### PPO learning notes (PR 2)

1. **The clipped surrogate.** With r_t(theta) = pi_theta(a_t | s_t) /
   pi_old(a_t | s_t) and advantage A_t, PPO maximizes

       L(theta) = E[ min( r_t A_t, clip(r_t, 1 - eps, 1 + eps) A_t ) ]

   For A_t > 0 the gain stops once r_t > 1 + eps; for A_t < 0 it stops once
   r_t < 1 - eps. A step can never profit from moving the policy far from
   the one that collected the data, a cheap stand-in for a trust region.
   - Small example: A = 2 and eps = 0.2. At r = 1.5 the unclipped term is 3.0,
     the clipped one 1.2 x 2 = 2.4, so the objective counts 2.4 and the
     gradient through r is zero.
   - The clip fraction (the share of samples with |r - 1| > eps) and the
     approximate KL in the health report say how hard the clip is working.
     Both near zero mean small updates; a KL spike means the update left the
     region where the surrogate is trustworthy.
2. **Generalized advantage estimation (GAE).** With
   delta_t = r_t + gamma V(s_{t+1}) - V(s_t),

       A_t = sum_k (gamma lambda)^k delta_{t+k}

   lambda = 0 uses only the one-step TD error (low variance, biased by V);
   lambda = 1 is the Monte-Carlo return minus V (unbiased, high variance).
   lambda = 0.95 is the usual compromise. Financial rewards are mostly noise,
   so the value function explains little (watch explained variance), and
   advantages stay noisy whatever lambda is.
3. **Why the likelihood lives on logits.** See section 6: the ratio is a
   ratio of Gaussian densities at the sampled fund logits. The softmax, the
   cash pin and the clip belong to the environment.
4. **Seed variance.**
   - Two PPO runs that differ only in the seed can end with different
     policies and dev scores. That spread is part of the result.
   - Report every seed, as a distribution (`seed_spread`: values, mean,
     standard deviation, median, range). Picking the best seed is selection on
     noise: with k seeds, the expected best of k equally good runs exceeds
     their mean by roughly sigma_seed x E[max of k standard normals] (about
     0.85 sigma for k = 3).
   - Every seed, hyperparameter set and checkpoint evaluated is a variant in
     the multiple-testing ledger.
5. **What the smoke run is for.** At the smoke budget PPO barely moves from
   its initialization, and two of three seeds selected the untrained
   checkpoint on dev. That checks the plumbing (seeding, selection, health,
   checkpoints, harness) and says nothing about what a longer run would learn.
6. **Validation practice.** Identical-seed replay; dev-only selection;
   a toy environment where the right answer is known (equity always earns
   0.4% per session: the trained policy must overweight it); stress
   scenarios; a no-forecast PPO and same-input optimizers as baselines; seed
   distributions; and, only after approval, a held-out test fold read once.
7. **Learn next.** Natural policy gradients and TRPO, the theory PPO
   approximates; why advantage normalization makes PPO insensitive to reward
   scale; the deflated Sharpe ratio for counting seeds and checkpoints as
   trials.

**Bounded exercise (optional).** In a copy of `configs/rl/ppo_smoke.yaml`, set
`total_timesteps` to 200000 for one seed and train it. Before reading the
health report, predict how entropy, approximate KL and explained variance
will move; then compare.

### Integration notes (PR 2)

- **Track A (approved 2026-10-15, implemented).** The environment injects
  `stock_agent.portfolio.transition.portfolio_step`; the stand-in
  `rl/reference_transition.py` is deleted. The boundary test admits two
  project imports outside `rl`, `stock_agent.portfolio.transition` and
  `stock_agent.portfolio.schedule` (the environment checks its one-session
  lag against `execution_lag()`), and still forbids `stock_agent.environment`
  (whose retired pre-T5a helpers stay uncorrected). `rl.reward.log_growth_reward` is therefore the same one-line
  formula as `stock_agent.environment.reward.log_growth_reward`, and a test
  checks they agree. `tests/test_t5a_engine.py` replays the backtest engine,
  the environment and the oracle on the same targets and returns and finds
  one wealth path within 1e-12, with and without binding constraints.
- **Comparison tables stay withheld.** An approved transition is necessary
  but not sufficient: `run_comparison(..., release_tables=True)` is required
  too, and no caller passes it. The registered comparison (a separate, later
  change) will release tables under its own protocol, with every test-fold
  look counted; until then `evaluate_rl.py` writes engineering reports only.
- **Track B (merged contract).** Forecast artifacts are dated:
  `read_forecast_artifact(path)` and `forecasts_asof(frame, decision_time)`,
  selecting on `available_at <= decision_time`. RL scenarios count sessions,
  so PPO with forecasts needs a session-to-synthetic-date map (the approved
  synthetic end-of-UTC-date rule) and a certified provider
  (`forecasts.require_certified`); until synthetic ETF artifacts exist the
  harness reports the method as unavailable.

