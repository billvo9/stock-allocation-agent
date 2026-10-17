# The registered synthetic comparison: baselines against PPO

Status (2026-10-17): implemented; the test fold was evaluated twice (look 2,
a recorded repeat after an independent review, reproduces look 1 exactly).

> **Synthetic scenarios only — not evidence of real investment performance.**
> This is an **engineering evaluation** of the approved simulator and of the
> comparison protocol. Nothing here ranks the strategies or claims skill or
> profitability. Every strategy's registry stage is `engineering_only`; a
> synthetic run can never reach `development_evaluated`.

## 1. Approval basis

- **Simulator.** T5a D1-D8, approved by the owner on 2026-10-15
  (`t5a_open_decisions.md`), and the owner's 2026-10-17 confirmations,
  verbatim:
  - "D5: A stricter registered constraint set applies identically to every
    compared policy, not only learned policies."
  - "D6: A hold/no-target decision preserves an already deferred target. A
    new valid target supersedes it."
  - "D5: An explicitly registered turnover cap also applies to the initial
    build. Report actual executed weights, uninvested cash and any
    incomplete construction so results are not misleading."
- **This work.** The owner's authorization, relayed by the lead on
  2026-10-17: "Proceed with the approved synthetic comparison engineering."
  The approved measures are terminal wealth and its return, group A drawdown
  and recovery (F6.5), turnover (F2.4) and D2's 0x, 1x and 2x cost
  sensitivity. Group D (CAGR, Sharpe ratios, intervals) is **not** approved
  and is not computed.

## 2. The registered configuration

`configs/comparison/ppo_comparison_v1.yaml`, registered before any result
existed. Its SHA-256 (`4f45f7d08416...`) is pinned by
`tests/test_comparison_protocol.py`: a change is a new registration with a new
name, never an edit.

| Strategy | Rule |
|---|---|
| 60/30/10, held without rebalancing (`buy_and_hold`) | Builds 60% equity fund, 30% bond fund, 10% short-Treasury fund once, then holds the drifting position |
| Equal weight | One third in each fund, rebalanced at every decision |
| 60/30/10, rebalanced (`fixed_balanced`) | The same weights at every decision |
| Inverse volatility | Each fund in proportion to 1 / its 60-session volatility |
| Mean-variance | The existing exact long-only optimizer, risk aversion 5, 60-session sample estimates |
| 100% short-Treasury fund (`all_shv`) | The short-Treasury fund held as a fund (2 bp to trade, its own return); uninvested cash earns 0% (D4 (a), synthetic engineering only) |
| Linear-softmax (CEM) | The inspectable learned baseline, five parameters, 5 seeds |
| PPO without forecast inputs | stable-baselines3 PPO's deterministic policy, 5 seeds |
| PPO with the health gate | The same PPO checkpoints behind the health gate and the registered fallback |

- **Constraint sets (D5), each applied to every strategy:**
  - `plain`: the D5 defaults, cap 1.0 and no turnover limit;
  - `strict`: cap 0.5 per fund (the excess goes to uninvested cash) and a
    one-way turnover limit of 0.25 per rebalance, the build included.
- **Learned strategies train and choose their checkpoints under the set they
  are evaluated in.** A PPO trained without a turnover limit would be judged
  under one it never saw; training under the registered set is the D5 rule
  ("stricter constraints are explicitly registered for learned policies")
  applied symmetrically.
- **Schedule (D7):** a decision every 5 complete sessions at offset 0 for
  every strategy; execution one session after each decision (D1).
- **Costs (D2):** the provisional rates (5 bp the equity and bond funds, 2 bp
  the short-Treasury fund, 0 the cash account) times each scenario's cost
  regime; 1x is the run, 0x and 2x replay its decisions.
- **Folds:** `default_split_plan()`: train on `mixed` (seeds 0-15, sessions
  60-900), dev on every scenario (seeds 100-103, sessions 960-1200), test on
  every scenario (seeds 200-203, sessions 1260-1500), with a 60-session
  embargo. Every stress scenario is in dev and test: calm, crash, high
  correlation, extreme volatility, stale observations, missing closes, high
  costs and mixed.
- **Budget:** PPO 204,800 timesteps per seed (the `ppo_dev` settings), CEM 10
  iterations of 24 candidates per seed.

## 3. The fairness contract

Every strategy is evaluated through one API: `harness.run_comparison`, so
`replay.rollout` on `AllocationEnv` with the approved `portfolio_step`.

**One registered setup, derived inside the check.** For each constraint set
and cost multiplier, `comparison.fairness.RegisteredSetup` derives ONE
common setup from the registered configuration alone: the transition, lag,
D2 rates times the multiplier, schedule and offset, build date, evaluation
window and the full D5 set (cap and turnover limit). It also regenerates the
scenario data from the registered generator. Nothing the runner was given
can change what is expected. A `ContractCheck` per (set, run or replay,
multiplier) then checks **every booked step of every strategy and episode**
against that setup. Its `close()` compares all registered entries together:
- every rule and every learned seed was checked exactly once;
- all of them share the registered fingerprint (setup, booked and decision
  sessions, build close, cost rates) of every episode.

The runner also requires exactly one evaluation per registered entry and
episode, so a strategy run twice on the test fold (an uncounted look) is
refused.

*(Added 2026-10-17 after an independent review. Earlier, each strategy was
checked alone against the setup the runner passed in, so a runner that ran
PPO at another offset, over a later window, without the cap, or best of two
offsets, passed the contract.)*

| Guarantee | How it is checked |
|---|---|
| One API | The record is an `EpisodeRecord` whose identity is the registered setup's |
| One transition | Each step is replayed through `portfolio_step` from its own inputs and must agree bit for bit; wealth compounds by each net return; each reward is log(1 + R_net) (D8 (a)) |
| Timing (D1) | A decision at complete session c_j executes at c_{j+1}; no other row carries a target |
| Schedule, offset and window (D7) | The decisions and booked sessions are the registered schedule's over the registered window |
| Costs (D2) | Every row's rates are the provisional rates times the regime times the registered multiplier |
| Build (D3) | 100% cash until the registered build close, where every first decision executes |
| Constraint set (D5) | Every target's D5 record equals `feasible_target(target, w_pre, registered set)`, and the transition received it |
| Together, once | Every registered entry checked once per episode, all sharing the registered fingerprint |
| Determinism | The 1x replay of each strategy's recorded decisions reproduces its run exactly |

**Mutation probes.** `tests/test_comparison_fairness.py` runs sixteen
mutants, each a strategy with exactly one advantage or deviation, planted
through the same environment API wherever possible: another API (the
engine's `SimulationResult`), a half-cost transition, a free build,
same-close execution (lag 0), a daily schedule, another offset, a cheaper
cost model, a portfolio that starts invested, another constraint set, the
strict set without its cap, a later window, a constraint set claimed but not
applied, a skipped feasibility map, fabricated
wealth, a reward that counts the cost twice, and a missing episode. The
contract refuses every one, for its own reason, both beside an honest
strategy and checked alone (as the runner checks each strategy), and the
honest strategy alone passes. A lag other than the registered one is refused
too.

**Source mutation probes (2026-10-17).** Each probe removed one guard from a
scratch copy of the code and ran the targeted tests against it. A canary
that breaks the fairness module confirms the tests import the mutated copy.

- **The review's runner mutants:**
  1. PPO evaluated at offset 1;
  2. PPO's window starting 5 sessions later;
  3. PPO under `strict` without the 0.5 cap;
  4. the better of two offsets chosen on the test fold (an uncounted look).

  On the earlier code (`9599bc3`), mutants 1 to 3 survived: each strategy was
  checked alone against the setup the runner passed. On the current code all
  four are killed, as is a best-of-two run that brings its own data loader.
  New tests cover them: the three setup changes, the best-of-two choice, a
  strategy run twice even when the registered run is kept, and an exact
  rollout count.
- **New guards, on the current code:**
  - killed: the joint close, the once-per-entry check, the run count, the
    look log's binding to its configuration, the Lab's log check and match,
    the construction split, and the cost multiplier;
  - the one survivor is the explicit shared-fingerprint comparison, which is
    redundant: every fingerprint component is already checked per record
    against the same registered reference.
- **The 37 single-guard probes, rerun on the current code: 32 killed.** The
  five survivors are guards enforced twice:
  - the API type check, both when a strategy is added and per record;
  - the D2 base-rate check, also covered by the scenario fingerprint in the
    identity;
  - the decision-index check, beside the decision-session check;
  - the per-row cost-rate record, beside the bit-for-bit replay;
  - the frozen-seed coverage check, beside the per-scenario seed check and
    the recomputed spreads.
- **Six combined probes**, each removing every guard of one guarantee (API,
  schedule, costs, constraint set, build from cash, seed reporting), are all
  killed.

## 4. Folds, selection and the evaluation log

1. **Train** (`run_ppo_comparison.py train`). The observation normalizer is
   fitted on the train fold. Every learned member (5 PPO and 5 CEM seeds
   under each set) trains on the train fold and picks its checkpoint by mean
   dev-fold log growth. The loader the trainers get refuses every scenario
   seed of the test fold (`TrainingLoader`), so the test fold cannot be built
   while anything is chosen (`test_training_never_builds_the_test_fold`, and
   a trainer that peeks is refused). The frozen selection, `selection.json`,
   records each member's checkpoint hash and dev selection record.
2. **Evaluate** (`run_ppo_comparison.py evaluate`). The look is appended,
   **before** any evaluation data exists (so a failed run still counts), to
   the configuration's private evaluation log
   `~/stock-agent-private/comparison/looks/<config_sha256>.jsonl`.
   - The log is write-once and hash-chained: an edited line refuses the
     whole log.
   - It has one file per registered configuration, in one place per machine
     whatever `--reports-root` is, so a fresh reports directory cannot
     restart the count. The environment variable
     `STOCK_AGENT_COMPARISON_LOOKS` exists for the test suite only.
   - The test fold is evaluated once per registered configuration. A second
     look is refused unless it is recorded, counted and labelled as a
     repeat. Dev looks are unlimited and counted.
   - The Lab re-verifies this log and withholds an artifact whose look it
     does not find.
3. **No selection on test results.** Every frozen member is reported on
   every episode, and `check_no_test_selection` recomputes every seed spread
   and every learned summary from all members' rows, independently of the
   report's own code. Mutants that keep only the best test seed (in the rows,
   the distribution, the per-seed values or the summary), swap a checkpoint,
   or pick the best seed inside the runner are all refused, and nothing is
   published.

## 5. Measures (approved only)

Per strategy, scenario, scenario seed and (for learned strategies) training
seed:

- **Terminal wealth and total return**, per 1.0 of initial cash, from the
  common build close to the last valued close; before taxes.
- **Drawdown and recovery**: `portfolio.descriptive.drawdown_summary` (A6,
  F6.5) with its peak, trough, recovery session and duration in sessions. A
  path with sessions the simulator does not value (the missing-closes
  scenario) reports only its **largest observed decline**, labelled
  incomplete, never a full-period maximum drawdown.
- **Turnover (F2.4)**: one-way turnover (a full switch is 1) summed over
  every execution, split into two parts:
  - **construction turnover**: from the build until the portfolio first
    reaches its target (the build alone without a turnover limit; every
    execution if the target is never reached);
  - **rebalance turnover**: everything after, with the number of rebalance
    executions and the turnover per rebalance.

  Traded notional is split the same way, and the window's length in sessions
  is reported. **Nothing is annualized**: F2.4 annualizes per calendar year,
  and synthetic sessions carry no calendar dates.
- **Costs (D2)**: the summed cost paid as a fraction of initial wealth, and
  terminal wealth and total return with the same decisions replayed at 0x,
  1x and 2x the registered rates.
- **Constraint interventions (D5)**: cap hits, turnover-limit hits and the
  one-way distance between requested and executed targets; **fallback
  events** with their reasons; invalid targets (always 0, since an invalid
  target raises).
- **The build**: section 6.
- **Spread across seeds**: for CEM and PPO, each seed's mean terminal wealth
  over the scenario's test paths, every seed listed, with n, min, the
  quartiles (linear interpolation), the median and max. Never only the best
  seed.

**The F2.4 annualization basis.** F2.4 (approved 2026-10-15) says one-way
turnover is "annualized per calendar year". A calendar basis needs dated
sessions, and the synthetic scenarios have none, so no turnover here is
annualized. Construction and rebalance turnover are reported separately, so
a turnover-limited build is never read as rebalancing; under the `strict`
set it otherwise would be. Annualizing per calendar year waits for dated
(real) sessions; any other basis needs the owner's approval.

## 6. Build reporting (D5, confirmed 2026-10-17)

`portfolio.construction.build_report` (engine side; also on
`SimulationResult` and in every `backtest_summary`) reports, from the D5
records of the executed targets:

- the build's requested and executed weights, and the **uninvested cash** it
  leaves (the cap's excess, the part of a turnover-limited build not yet
  invested, and any cash asked for);
- `incomplete_construction`: the turnover limit bound at the build;
- the **first execution where the limit no longer bound** and the sessions
  from the build to it, or none (never fully built).

Example (hand-checked in `tests/test_comparison_measures.py`): a 50/50
equity/bond mix under the 0.25 limit executes 25%, then a third and a half of
the remaining way, and first reaches 50/50 at the fourth execution, 15
sessions after the build. A 60/30/10 portfolio held without rebalancing under
the strict set is capped to 50/30/10 with 10% cash, builds 0.25 / 0.9 of the
way (75% uninvested cash) and, never trading again, is never fully built: the
comparison reports exactly that rather than a number that looks invested.

## 7. Outputs and the Lab

- `reports/comparison/<artifact_id>/comparison.json`: the versioned artifact
  (`ppo_comparison` 1.1.0, write-once, `content_sha256` over its canonical
  JSON), with every row, the summaries, the seed spreads, the build reports,
  the fairness record (with the registered setups), the look and its log
  entry, and the labels. Version 1.1.0 (2026-10-17) splits construction from
  rebalance turnover, drops the annualized turnover, and records the look
  log and each learned member's parameter hash; the Lab shows 1.1.0 only and
  withholds a 1.0.0 file.
- `reports/health_inputs/backtest_summary/`: one summary per (fold,
  constraint set, strategy), each measure the worst over the strategy's 1x
  episodes; turnover rounded to the engine's 1e-12 dust so floating-point
  residue at a binding limit is not read as a breach; drift above the cap is
  reported, as D5 requires; constraint violations and fallback events
  included.
- **The Lab** (Experiments, "Synthetic strategy comparison (engineering
  evaluation)") shows the newest verified test-fold artifact with the labels
  "Synthetic scenarios only — not evidence of real investment performance"
  and "Engineering evaluation", each strategy's registry stage, results per
  constraint set and scenario in registered order (never sorted by a
  result), every learned seed and the build. A file outside the registered
  protocol (edited, unlabelled, not synthetic, not the test fold, a stage
  above `development_evaluated`, an unapproved transition, a failed
  contract, or a look the private evaluation log does not verify) is
  **withheld** with its reason. Without one, the panel says no
  registered comparison has been run. Everything outside this protocol keeps
  the earlier behaviour: `rl.harness` still withholds comparison tables and
  `evaluate_rl.py` writes engineering reports only.

Reproduce (from the repository root). On 2026-10-17, on a shared Apple-silicon laptop,
training took 24 minutes with 4 workers (75 CPU minutes) and the evaluation 3 minutes:

```bash
.venv/bin/python scripts/run_ppo_comparison.py train --config configs/comparison/ppo_comparison_v1.yaml --workers 4
```

```bash
.venv/bin/python scripts/run_ppo_comparison.py evaluate --config configs/comparison/ppo_comparison_v1.yaml --run-dir models/comparison/<run_id>
```

The same configuration, code and library versions give the same artifact
(`test_the_same_configuration_and_seeds_give_the_same_artifact`); the number
of training workers does not change the selection.

**Checkpoint hashes.** A PPO checkpoint file's SHA-256 differs between two
identical trainings. stable-baselines3 stores the training start time
(`start_time`, in nanoseconds) in the archive's `data` entry; the weights
(`policy.pth`) and the optimizer state are byte-identical
(`test_identical_trainings_give_identical_parameters_though_the_archives_differ`).
The file hash still proves that the evaluated file is the frozen one. Each
member also records a deterministic `parameters_sha256` over its parameters
(PPO: every tensor of the policy in name order; CEM: its canonical
parameters). Two identical trainings give the same parameter hash, and the
evaluation checks it against the frozen selection whenever the selection
records one.

## 8. Results (synthetic, engineering evaluation)

> Synthetic scenarios only — not evidence of real investment performance.
> Engineering evaluation. Nothing below ranks the strategies, tests a
> difference or claims skill: four test paths per scenario, one 240-session
> window, and no statistical procedure (group D is not approved). Where
> learned seeds' ranges overlap, no comparative statement is made.

**Looks at the test fold of `ppo_comparison_v1` (`4f45f7d08416...`).** Two
looks are recorded in the configuration's private evaluation log. Its head,
look 2's entry, is `9a78b4d28767...`, and it records 0 development-fold
looks.

1. **Look 1:** `ppo_comparison_v1-test-look001-20261017T034147659125Z`, from
   `95c2f9d` (schema 1.0.0). It was recorded first in
   `reports/comparison/evaluation_log.jsonl` and carried into the private
   log when the log moved there.
2. **Look 2**, a recorded repeat:
   `ppo_comparison_v1-test-look002-20261017T044941380492Z`, from `ed04df1`
   (schema 1.1.0, `content_sha256` `e7065c219774...`). It is the same frozen
   selection (members trained from `6a1dfb3`), evaluated again after an
   independent review changed the fairness check and the turnover fields.

   Look 2 reproduces look 1 exactly on every field both report, for all
   1,344 rows: terminal wealth, total return, drawdown, total turnover,
   traded notional, costs, cost sensitivity, interventions, fallbacks, the
   build and the checkpoints. Nothing was chosen between the looks. The
   tables below are from look 2.

- **Fairness contract:** passed. There are 42 registered entries per set
  (six rules plus three learned groups of five seeds), and each is evaluated
  exactly once per episode: 1,344 runs and 4,032 cost replays. Every record
  was checked against the registered setup of its set and multiplier, and
  each setup closed over all entries together: 1,260,336 booked steps across
  the run and replay checks.
- **Training health:** every PPO seed's training health is `healthy`.
- **Variants searched** (multiple-testing ledger):
  - CEM: 10 members, 2,400 candidates on the train fold, 100 checkpoints
    compared on dev;
  - PPO: 10 members, 110 checkpoints compared on dev;
  - 2 test looks, the second a recorded repeat.
- **Health inputs:** one `backtest_summary` per set and strategy for each
  look.

**Median terminal wealth per 1.0 of initial cash, `plain` set (1x costs)**

| Strategy | calm | crash | high corr. | extreme vol. | stale obs. | missing closes | high costs | mixed |
|---|---|---|---|---|---|---|---|---|
| 60/30/10, held without rebalancing | 1.038 | 0.881 | 0.887 | 0.922 | 1.012 | 0.978 | 1.005 | 0.928 |
| Equal weight | 1.032 | 0.945 | 0.933 | 0.975 | 1.027 | 0.998 | 1.014 | 0.977 |
| 60/30/10, rebalanced | 1.040 | 0.877 | 0.887 | 0.951 | 1.013 | 0.980 | 1.007 | 0.928 |
| Inverse volatility | 1.022 | 1.016 | 1.003 | 1.010 | 1.030 | 1.015 | 1.019 | 1.022 |
| Mean-variance | 1.006 | 1.036 | 0.947 | 0.994 | 0.994 | 1.014 | 0.906 | 1.044 |
| 100% short-Treasury fund | 1.019 | 1.019 | 1.012 | 1.015 | 1.019 | 1.018 | 1.016 | 1.022 |
| Linear-softmax (CEM) | 1.020 | 1.018 | 1.002 | 1.001 | 1.031 | 1.019 | 1.014 | 1.024 |
| PPO without forecast inputs | 1.025 | 0.992 | 0.949 | 1.020 | 1.021 | 1.008 | 1.011 | 1.019 |
| PPO with the health gate | 1.025 | 0.992 | 0.949 | 1.014 | 1.021 | 1.008 | 1.011 | 1.019 |

**Median terminal wealth per 1.0 of initial cash, `strict` set (1x costs)**

| Strategy | calm | crash | high corr. | extreme vol. | stale obs. | missing closes | high costs | mixed |
|---|---|---|---|---|---|---|---|---|
| 60/30/10, held without rebalancing | 1.009 | 0.973 | 0.973 | 0.981 | 1.004 | 0.995 | 1.002 | 0.984 |
| Equal weight | 1.033 | 0.946 | 0.935 | 0.973 | 1.026 | 0.992 | 1.012 | 0.967 |
| 60/30/10, rebalanced | 1.037 | 0.901 | 0.905 | 0.957 | 1.016 | 0.975 | 1.006 | 0.930 |
| Inverse volatility | 1.014 | 1.009 | 0.997 | 1.005 | 1.020 | 1.009 | 1.012 | 1.012 |
| Mean-variance | 1.002 | 1.013 | 0.974 | 0.992 | 0.999 | 1.002 | 0.944 | 1.002 |
| 100% short-Treasury fund | 1.010 | 1.009 | 1.006 | 1.008 | 1.009 | 1.009 | 1.008 | 1.011 |
| Linear-softmax (CEM) | 1.020 | 1.022 | 0.979 | 0.979 | 1.038 | 1.021 | 1.014 | 1.029 |
| PPO without forecast inputs | 1.024 | 0.991 | 0.948 | 1.024 | 1.017 | 1.003 | 1.010 | 1.017 |
| PPO with the health gate | 1.024 | 0.991 | 0.948 | 1.024 | 1.017 | 1.003 | 1.010 | 1.017 |

**Median maximum drawdown, `plain` set (missing closes: largest observed decline, incomplete)**

| Strategy | calm | crash | high corr. | extreme vol. | stale obs. | missing closes | high costs | mixed |
|---|---|---|---|---|---|---|---|---|
| 60/30/10, held without rebalancing | 8.0% | 19.4% | 14.4% | 23.8% | 7.2% | 9.3% | 8.5% | 17.3% |
| Equal weight | 4.4% | 10.4% | 8.7% | 14.0% | 3.7% | 4.4% | 4.6% | 9.8% |
| 60/30/10, rebalanced | 8.1% | 19.7% | 14.6% | 23.6% | 7.4% | 9.2% | 8.4% | 17.8% |
| Inverse volatility | 0.6% | 0.5% | 1.2% | 1.4% | 0.4% | 0.5% | 0.4% | 0.9% |
| Mean-variance | 6.7% | 8.1% | 11.4% | 19.6% | 6.8% | 8.1% | 12.5% | 11.8% |
| 100% short-Treasury fund | 0.2% | 0.3% | 0.5% | 0.5% | 0.3% | 0.3% | 0.3% | 0.3% |
| Linear-softmax (CEM) | 0.9% | 1.1% | 1.8% | 3.7% | 0.8% | 0.7% | 0.9% | 1.1% |
| PPO without forecast inputs | 2.5% | 4.1% | 6.5% | 8.9% | 2.2% | 2.4% | 3.6% | 4.0% |
| PPO with the health gate | 2.5% | 4.1% | 6.5% | 10.2% | 2.1% | 2.4% | 3.6% | 4.0% |

**Spread across training seeds: each seed's mean terminal wealth over the scenario's four test paths (min / median / max of 5 seeds)**

| Set | Strategy | calm | crash | high corr. | extreme vol. | stale obs. | missing closes | high costs | mixed |
|---|---|---|---|---|---|---|---|---|---|
| plain | Linear-softmax (CEM) | 1.016 / 1.022 / 1.029 | 1.009 / 1.020 / 1.035 | 0.975 / 1.000 / 1.012 | 0.945 / 1.002 / 1.017 | 1.019 / 1.027 / 1.049 | 1.013 / 1.022 / 1.035 | 1.009 / 1.016 / 1.023 | 1.021 / 1.029 / 1.052 |
| plain | PPO without forecast inputs | 1.021 / 1.024 / 1.031 | 0.971 / 0.987 / 0.995 | 0.956 / 0.960 / 0.968 | 1.021 / 1.053 / 1.062 | 1.014 / 1.018 / 1.022 | 0.997 / 1.003 / 1.011 | 0.994 / 1.002 / 1.010 | 1.018 / 1.035 / 1.046 |
| plain | PPO with the health gate | 1.021 / 1.024 / 1.031 | 0.971 / 0.987 / 0.995 | 0.956 / 0.960 / 0.968 | 1.006 / 1.037 / 1.052 | 1.015 / 1.018 / 1.024 | 0.997 / 1.003 / 1.011 | 0.994 / 1.002 / 1.010 | 1.018 / 1.035 / 1.046 |
| strict | Linear-softmax (CEM) | 1.019 / 1.024 / 1.026 | 1.003 / 1.022 / 1.032 | 0.971 / 0.987 / 0.994 | 0.965 / 0.972 / 0.996 | 1.027 / 1.032 / 1.035 | 1.019 / 1.024 / 1.030 | 1.012 / 1.014 / 1.021 | 1.027 / 1.040 / 1.041 |
| strict | PPO without forecast inputs | 1.021 / 1.024 / 1.032 | 0.974 / 0.989 / 0.996 | 0.959 / 0.960 / 0.968 | 1.022 / 1.053 / 1.056 | 1.012 / 1.020 / 1.024 | 0.998 / 1.004 / 1.011 | 0.997 / 1.004 / 1.012 | 1.017 / 1.032 / 1.044 |
| strict | PPO with the health gate | 1.021 / 1.024 / 1.032 | 0.974 / 0.989 / 0.996 | 0.959 / 0.960 / 0.968 | 1.016 / 1.043 / 1.055 | 1.014 / 1.022 / 1.024 | 0.998 / 1.004 / 1.011 | 0.997 / 1.004 / 1.012 | 1.017 / 1.032 / 1.044 |

**Cost sensitivity, `plain` set: median terminal wealth with the same decisions at 0x / 1x / 2x the registered rates**

| Strategy | calm | high costs |
|---|---|---|
| 60/30/10, held without rebalancing | 1.038 / 1.038 / 1.037 | 1.005 / 1.005 / 1.004 |
| Equal weight | 1.032 / 1.032 / 1.031 | 1.015 / 1.014 / 1.013 |
| 60/30/10, rebalanced | 1.041 / 1.040 / 1.039 | 1.008 / 1.007 / 1.006 |
| Inverse volatility | 1.022 / 1.022 / 1.021 | 1.020 / 1.019 / 1.018 |
| Mean-variance | 1.014 / 1.006 / 0.996 | 0.931 / 0.906 / 0.882 |
| 100% short-Treasury fund | 1.020 / 1.019 / 1.019 | 1.016 / 1.016 / 1.016 |
| Linear-softmax (CEM) | 1.021 / 1.020 / 1.020 | 1.017 / 1.014 / 1.012 |
| PPO without forecast inputs | 1.029 / 1.025 / 1.022 | 1.017 / 1.011 / 1.004 |
| PPO with the health gate | 1.029 / 1.025 / 1.022 | 1.017 / 1.011 / 1.004 |

**Turnover in the calm scenario (median one-way over the 240-session window; not annualized: synthetic sessions have no calendar dates)**

| Strategy | `plain`: construction | `plain`: rebalance | `plain`: per rebalance | `strict`: construction | `strict`: rebalance | `strict`: per rebalance |
|---|---|---|---|---|---|---|
| 60/30/10, held without rebalancing | 1.00 | 0.00 | — | 0.25 | 0.00 | — |
| Equal weight | 1.00 | 0.18 | 0.0038 | 1.00 | 0.16 | 0.0038 |
| 60/30/10, rebalanced | 1.00 | 0.19 | 0.0041 | 0.90 | 0.18 | 0.0042 |
| Inverse volatility | 1.00 | 0.23 | 0.0050 | 0.63 | 0.23 | 0.0052 |
| Mean-variance | 1.00 | 10.33 | 0.2246 | 0.50 | 5.51 | 0.1253 |
| 100% short-Treasury fund | 1.00 | 0.00 | 0.0000 | 0.50 | 0.01 | 0.0002 |
| Linear-softmax (CEM) | 0.94 | 0.23 | 0.0050 | 0.84 | 0.57 | 0.0129 |
| PPO without forecast inputs | 0.77 | 3.49 | 0.0758 | 0.79 | 3.43 | 0.0797 |
| PPO with the health gate | 0.77 | 3.49 | 0.0758 | 0.79 | 3.43 | 0.0797 |

**The build under the `strict` set (all 32 test paths; learned strategies over 5 seeds)**

| Strategy | Executed at the build (median: equity / bond / short-Treasury / cash) | Incomplete construction | Never fully built | Sessions until fully built (median) | Uninvested cash at the end (median) |
|---|---|---|---|---|---|
| 60/30/10, held without rebalancing | 13.9% / 8.3% / 2.8% / 75.0% | 32 of 32 | 32 of 32 | — | 76.1% |
| Equal weight | 8.3% / 8.3% / 8.3% / 75.0% | 32 of 32 | 0 of 32 | 15 | 0.0% |
| 60/30/10, rebalanced | 13.9% / 8.3% / 2.8% / 75.0% | 32 of 32 | 0 of 32 | 15 | 10.0% |
| Inverse volatility | 1.3% / 4.0% / 19.7% / 75.0% | 32 of 32 | 0 of 32 | 10 | 36.5% |
| Mean-variance | 11.0% / 0.0% / 0.0% / 75.0% | 32 of 32 | 0 of 32 | 10 | 50.0% |
| 100% short-Treasury fund | 0.0% / 0.0% / 25.0% / 75.0% | 32 of 32 | 0 of 32 | 5 | 50.0% |
| Linear-softmax (CEM) | 0.3% / 12.9% / 11.9% / 75.0% | 160 of 160 | 0 of 160 | 15 | 16.8% |
| PPO without forecast inputs | 5.6% / 9.9% / 8.9% / 75.0% | 160 of 160 | 0 of 160 | 15 | 23.4% |
| PPO with the health gate | 5.6% / 9.9% / 8.9% / 75.0% | 160 of 160 | 0 of 160 | 15 | 23.4% |

**Interventions and fallbacks over all scenarios, paths and seeds (the `plain` set has no cap or limit, so only its fallbacks are shown)**

| Set | Strategy | Cap hits | Turnover-limit hits | Fallback events | Invalid targets |
|---|---|---|---|---|---|
| plain | PPO with the health gate | 0 | 0 | 85 | 0 |
| strict | 60/30/10, held without rebalancing | 32 | 32 | 0 | 0 |
| strict | Equal weight | 0 | 109 | 0 | 0 |
| strict | 60/30/10, rebalanced | 1494 | 96 | 0 | 0 |
| strict | Inverse volatility | 1494 | 64 | 0 | 0 |
| strict | Mean-variance | 1455 | 430 | 0 | 0 |
| strict | 100% short-Treasury fund | 1494 | 43 | 0 | 0 |
| strict | Linear-softmax (CEM) | 4250 | 456 | 0 | 0 |
| strict | PPO without forecast inputs | 384 | 470 | 0 | 0 |
| strict | PPO with the health gate | 392 | 487 | 85 | 0 |

**What the numbers describe (structural facts only):**

- **The generator's quirk shows.** The short-Treasury fund drifts 2% a year
  at 0.6% volatility in every regime while cash earns 0% (D4 (a)). Holdings
  that lean on it have small median drawdowns in every scenario, crash
  included: the 100% fund at most 0.5%, inverse volatility at most 1.4%. That
  is a property of the synthetic generator (`rl_ppo.md`, learning note 7),
  not a finding about any strategy.
- **Turnover and costs.** Mean-variance trades the most: a median one-way
  rebalance turnover of 10.33 over the 240-session calm window, 0.22 per
  rebalance. Its terminal wealth moves most with the cost multiplier: in the
  high-cost scenario, the median is 0.931, 0.906 and 0.882 at 0x, 1x and 2x.
  The held 60/30/10 and the 100% fund barely trade after construction and
  barely move with costs.
- **Construction is not rebalancing.** Under `strict`, construction turnover
  is reported apart from rebalance turnover. Equal weight's rebalance
  turnover is 0.16 (`strict`) against 0.18 (`plain`): the build's
  completion, spread over four executions, is construction.
- **The health gate.** Under each set, the fallback decided 85 times: stale
  inputs and non-finite features in the stale-observations scenario, and
  out-of-distribution features in extreme volatility. The guarded and raw PPO
  rows differ only in those two scenarios.
- **The build under `strict`.** Every build from cash is incomplete: the 0.25
  limit lets a quarter of the portfolio in at the first close, leaving 75%
  uninvested. Rebalancing strategies are fully built 5 to 20 sessions later
  (medians 5 to 15). The 60/30/10 held without rebalancing never is: it ends
  with a median 76% in cash, so its `strict` row describes a mostly-cash
  portfolio, as the build table says. The cap leaves the 100% fund and
  mean-variance with about half their wealth in 0% cash at the end.
- **Seeds.** Every training seed is listed with the spread of its seeds' mean
  terminal wealth. With five seeds and four paths per scenario, these
  spreads are themselves noisy and often overlap between learned strategies,
  so no ordering of them is stated.

## 9. Limitations

- Synthetic scenarios only: Gaussian or Student-t shocks with fixed
  correlations and illustrative parameters. Nothing transfers to real funds.
- One test window of 240 sessions per scenario and four scenario seeds: far
  too little to separate strategies statistically, and no statistical test
  is run (group D).
- Uninvested cash earns 0% (D4 (a)), which penalizes every cash-heavy result,
  including the cash a cap or a partial build leaves; the cost rates are
  provisional.
- The environment values only sessions where every fund closes, so the
  missing-closes scenario has incomplete drawdown histories for every
  strategy alike.
- PPO and CEM are trained at the `ppo_dev` budget only; no hyperparameter was
  tuned, so the learned results describe this budget, not what PPO could
  learn.
- PPO with forecast inputs is not part of this comparison: no certified
  out-of-fold synthetic forecast artifact exists yet.

## 10. Learning notes

1. **Problem.** Compare rule baselines, an optimizer and learned policies so
   that any difference comes from the policies, not from the plumbing, and so
   that the test result is not quietly chosen.
2. **Fair comparison.** Fairness is a property of the evaluation, not of the
   strategies: one transition, one timing, one cost model, one schedule, one
   build date and one constraint set. The contract checks the records, not
   the intentions: a simulator bug that gave one strategy cheaper costs
   would show up as a step `portfolio_step` cannot reproduce. Practitioners
   call this a level playing field and test it with planted advantages
   (mutants), exactly as section 3 does.
3. **Overfitting to synthetic scenarios.** A learned policy trained on one
   generator learns that generator's quirks: here the short-Treasury fund's
   steady positive drift against a 0% cash account, and regime lengths that
   repeat on a fixed cycle. Holding out later windows and unseen seeds
   protects against memorizing paths, not against learning the generator. A
   result here is at best evidence the pipeline works; real-data validation
   (group C costs and cash, a point-in-time research panel) is a different
   question.
4. **Seed variance.** Two PPO runs that differ only in the seed can end with
   different policies. With k seeds, the best seed's expected score exceeds
   the mean by roughly sigma_seed times E[max of k standard normals] (about
   1.16 sigma for k = 5), from selection alone. So every seed is reported, and
   the spread (min, quartiles, max) is the result; picking the best test seed
   would be selection on noise, and a test fails if the code does it.
5. **Multiple testing.** Each seed, each dev checkpoint compared and each CEM
   candidate is a variant; the artifact counts them (`variants`). The test
   fold is meant to be looked at once per registered configuration, and
   every look is counted. Here a review changed the reporting after look 1,
   so look 2 is recorded as a repeat. It is harmless only because the frozen
   selection did not change and look 2 reproduces look 1 exactly; a repeat
   that chose anything would be selection on the test fold.
6. **Ways results could mislead.** A strict set changes every strategy's
   exposure (the cap's excess sits in 0% cash), so a gap between sets
   measures the constraints as much as the policies. One rebalance offset is
   one draw from a family (F7.2). Medians over four paths hide the paths;
   the rows keep them.
7. **Learn next.** Deflated Sharpe ratios and the effective number of trials
   (once group D is approved); block-bootstrap comparisons on the same paths;
   why myopic mean-variance is a strong baseline for log growth under i.i.d.
   returns; how cash yield (group C) changes cash-heavy results.

**Bounded exercise (optional).** Using one artifact's `rows`, compute PPO's
per-seed mean terminal wealth in the crash scenario under the plain set
yourself and check it against `seed_variability`. Then predict, before
looking, how much the best seed exceeds the median seed, using the formula in
note 4.
