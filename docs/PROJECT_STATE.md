# Project State

Changing project state only. Permanent policy lives in `AGENTS.md`.
Git is authoritative: verify everything here before relying on it.

Last reviewed: 2026-10-17 (New York; 2026-10-18 UTC)


## Last checkpoint

2026-10-17 (New York), milestone "Wave 2 merged; preparing real-data
activation" (the owner's 2026-10-17 brief "Finalize PRs and Prepare
Real-Data Activation"). PRs #78-#95 are on `main`. The Stock Lab watchlist
is AMD, MRVL, MU and NVDA, with SKHY and SNDK removed. The next evidence
step is the owner's two read-only audits, not more synthetic optimization.
No live vendor request has been made. Verified on `checkpoint-wave3` at
`79c80ef` (= `origin/main`); merge pending.

The previous checkpoint (2026-10-16, "Stock Probability & Reversal Lab",
verified on `b55efe6`) reached `main` with PR #81 (`ebba0d7`).
Live Git state (current branch, working tree, upstream) comes from Git and
the SessionStart hook, not from this file.

Recently merged to `main` (`origin/main` = `79c80ef` as of the fetch on
2026-10-18 00:40 UTC, verified with `git fetch` and the GitHub API). CI on
`79c80ef`, `c6bd633` and `9e1ce8e` succeeded. All were merged on 2026-10-17
UTC except #95 (2026-10-18 UTC).
- PR #95 (`79c80ef`): review follow-up to #91.
  - Zero-volatility scoring is bound to the bundle's rule; a mismatch is
    refused as `zero_vol_rule_mismatch`.
  - Criterion Z2 is revised; the original is kept as the reported check R1.
  - The registration rules are settled in
    `docs/research/prospective_registration.md`. Its section 13 decisions
    are not approved.
- PR #94 (`c6bd633`): the Model Health job's environment (`/bin/zsh -f`),
  checkout (a clean `main` only) and staleness (`STALE_AFTER` 26 h).
  Follow-up to #90.
- PR #93 (`9e1ce8e`): the A7 decision packet
  (`docs/research/stock_lab_a7_decision.md`). Documentation; not approved.
- PR #92 (`448c7d0`): the group C and D decision packets
  (`docs/research/group_c_d_decision_index.md`, `group_c_decisions.md`,
  `group_d_decisions.md`). Documentation; nothing approved or implemented.
- PR #91 (`8748b52`): ETF freeze readiness. It covers:
  - the research-only zero-volatility floor B2
    (`docs/research/zero_vol_floor.md`), not approved;
  - its development-data check plan;
  - the first registration rules.
- PR #90 (`8393787`): the reviewable Model Health scheduler setup
  (`scripts/setup_health_schedule.py`, a dry run by default) and monitor
  coverage. Not installed.
- PR #88 (`24972a0`): the registered synthetic comparison of eight
  strategies, PPO among them, with build reporting
  (`docs/research/ppo_comparison.md`). It is an engineering evaluation on
  synthetic scenarios, with no ranking, test or skill claim.
- PR #87 (`5843f5f`): the Stock Lab forecast record refuses backdating,
  detects truncation, and reads unauthorized skips as "not activated".
- PR #85 (`b8d7871`): the Stock Lab read-only coverage and corporate-action
  audit; it records owner decisions A1-A6.
- PRs #83 and #84 (`9344ea5`, `acffc58`; #84 duplicates #83): the Stock
  Lab fold-block t, simulated test size and stable-band sensitivity.
- PRs #82, #86 and #89 (`773523b`, `6c0f025`, `b7293f9`): dark mode for
  the Portfolio Lab; visible hatching and 3:1 radio rings in the Lab and
  the research dashboard.
- PR #81 (`ebba0d7`): Stock Lab PR 2 and PR 3 landed on `main` with the
  PR 3 review fixes. PRs #79 and #80 had merged into their stacked base
  branches, not `main`.
- PR #78 (`bc0db8b`): Stock Lab PR 1 (watchlist, Tiingo stock adapter,
  features, the Stock Probability page on fixtures).
- PR #77 (`4b9f8d0`, 2026-10-16): Model Health in the Portfolio Lab, with
  private local checks; it also carried the 2026-10-15 checkpoint.
- PR #76 (`22176f9`, 2026-10-16): the approved T5a engine (D1-D8) as the
  single source of portfolio economics; strategy comparison tables withheld
  until a registered comparison PR.

Earlier (`origin/main` = `f98668a`, verified 2026-10-15):
- CI on `f98668a` succeeded.
- A manual Model health run on `f98668a` succeeded (2026-10-15).
- The first scheduled Model health run, on `82911ac` (2026-10-15 11:30
  UTC), failed `deps.pins` with a false alarm that PR #75 fixed.

- PR #75 (`f98668a`, 2026-10-15): platform-aware dependency pins in the
  health check (environment markers, pip's comment rule, direct-URL wheel
  versions).
- PR #74 (`5a2db6f`, 2026-10-15): prospective ETF forecasts for VTI, AGG
  and SHV at 5, 20 and 30 sessions:
  - frozen, registered models with split-conformal 80% bands;
  - a hash-chained, write-once forecast ledger and a scoring job;
  - the "Prospective forecasts (unvalidated)" panel in Experiments.
  It reads the prospective store only through PR #73's
  `load_inference_window`. Synthetic tests only; no real forecast exists.
- PR #73 (`128604c`, 2026-10-15): the real ETF data path:
  - private, hand-filled distribution anchors replacing the in-code
    allowlist;
  - a rule-based NYSE calendar through 2027 that reproduces the frozen
    2003-2024 list;
  - an isolated prospective store and Tiingo prospective mode (window
    2026-07-16 to the last complete session; read-only by default);
  - the inference loader.
  It includes the owner decision record `a633723`. No live request has
  been made.
- PRs #64, #66, #69, #70, #72 (`dedc07e`, `97f5185`, `1e06142`, `7279aec`,
  `82911ac`, 2026-10-14): feature contracts, frozen universes and
  prefix-invariant labels; the reproducible experiment runner; the
  HistGradientBoosting challenger; the separate synthetic ETF return and
  risk track; the registry health export. PR #63 merged into a stacked
  branch, not `main`; #64 replaced it.
- PR #71 (`9e3decd`, 2026-10-14): PPO (stable-baselines3 2.9.0,
  gymnasium 1.3.0, torch 2.14.0 pinned, CPU wheel by URL and sha256 on
  Linux) on the synthetic environment. Engineering only; includes the
  owner decision record `eebff9a`.
- PR #68 (`e0557f9`, 2026-10-14): RL review fixes. The step info is now
  backward-looking, closing a leak of the next-session gap present on
  `main` from `6619a30` until this merge. It also adds checkpoint
  provenance.
- PR #67 (`772b9a9`, 2026-10-14): the Model Health Center (health
  contract, collectors, the daily `health.yml` workflow, runbook,
  standalone page).
- PR #65 (`f5578ea`, 2026-10-14): the Portfolio Lab redesign (six
  destinations; Builder weights and concentration only, mix history
  awaiting D9).
- PR #62 (`6619a30`, 2026-10-14): the RL environment, synthetic scenarios,
  replay and baseline policies.
- PR #61 (`88ac452`, 2026-10-14): T5a scaffolding: interfaces, a
  shares-and-cash reference oracle, and strict-xfail `portfolio_step`
  tests (formulas pending D1-D8).
- PR #60 (`c5b3175`, 2026-10-14): the model registry and the out-of-fold
  forecast artifact contract.
- PR #59 (`7ad4d44`, 2026-10-14): Tiingo checks labelled as spot checks,
  and a read-only coverage audit by default.
- PR #58 (`6b33c1b`, 2026-10-14): the Tiingo fund adapter (allowlisted
  fields, verified splits, primary-source distribution gate).
- PR #57 (`14cdf9d`, 2026-10-14): T5-A, the approved descriptive
  calculations on the Explore page; checkpoint `9b9a667`.

- PR #56 (`0df3a0c`, 2026-10-14): the Explore page on synthetic data and
  the neutral `ui/` package.

- PR #55 (`ace2e66`, 2026-10-14): F1 fund-data foundations (frozen NYSE
  calendar 2003-2024, fund-data contract, write-once private store).
- PR #54 (`c2ba633`, 2026-10-13): RES-1 cost-conscious and non-blocking
  for the ETF Portfolio Lab; checkpoint `267d421`.

- PR #53 (`1df3783`, 2026-10-13): `f238374` feat: publication contract
  v1 with a fail-closed reader (P1-1a); `e589f39` docs: checkpoint.
- PR #52 (`2c53ebd`, 2026-10-13): `903a48e` and `cddf97f` docs: the T5
  financial-formula specification (`docs/research/t5_portfolio_formulas.md`;
  every formula pending OD8).

- PR #51 (`b1f849d`, 2026-10-13): `2886db5` fix: remove the diagnostics
  staging directory when a run write fails.
- PR #49 (`ad99749`, 2026-10-12): the product direction documents
  (`docs/product/`) and the 2026-10-12 checkpoint (`eb25a5a`).
- PR #50 (`978c027`, 2026-10-12): `7cbb1c0` fix: keep the FRED bearer token
  out of release-request tracebacks; `aefeb8c` test: cover non-JSON release
  responses; keep this change Version 2 only.
- PR #48 (`7490686`, 2026-10-12): `7faeb8b` fix: keep the FRED API key out
  of request-failure errors.
- PR #47 (`62a7113`, 2026-10-12): `99e9000` chore: pin scikit-learn 1.9.1;
  `d435837` feat: T4 phase A; `e4939c6` and `37fcaba` docs; `1338674` feat:
  the T4 linear core; `99a8405` fix: T4 review findings; `3ae5c8f` docs: T4
  results and checkpoint.

- PR #46 (`ca68edc`): `84413b0` feat: research workspace with
  date-normalized Goyal-Welch views and a shared design system; `5234102`
  docs: T3.1 research workspace; `1b89ccf` docs: checkpoint.
- PR #45 (`1eda1d1`): `7594a5b` feat: decision method, interval settings
  and reference roles (schema 1.1); `a2442c6` fix: fail closed on summary
  disagreements and label schema 1.0 gaps; `505de3b` docs; `9492892` docs:
  checkpoint.
- PR #44 (`0c0304a`): `3addc34` chore: pin plotly and streamlit for the
  research dashboard; `14b7c40` fix: restrict the run reader to files
  inside the run directory; `1f56ec3` feat: read-only research dashboard
  over saved T2 runs; `b396096` docs: T3 dashboard and its T2 findings;
  `1000e4e` docs: checkpoint.
- PR #43 (`f013553`): `4e8a830` feat: dependence-aware inference for
  date-level statistics; `59570da` feat: null models, diagnostics runner and
  saved-output contract; `479d59f` docs: T2 design, evidence and learning
  notes; `3581c7e` docs: checkpoint.
- PR #42 (`51ca2a1`): `9470375` docs: refine universe and market-context
  research; `386df3f` chore: add PR handoff rule and pull request template.
- PR #41 (`d82fb90`): `18c0864` docs: design point-in-time universe and
  market-context tracks.
- PR #40 (`145c7bf`): `6f34788` feat: explicit label timing with next-close
  entry; `8d80b6e` feat: purged walk-forward model-validation foundation;
  `8f6d15b` docs: checkpoint; `d98cd92` docs: diagnostics and visualization
  plan.
- PR #39 (`da21961`): `b2e443a` chore: adopt guarded high-autonomy Claude
  workflow; `763b224` docs checkpoint; `28aaff6` chore: pin ruff 0.16.2 and
  pytest 9.1.1. The development workflow (autonomy, approval gates,
  `scripts/verify.py`) is described in AGENTS.md.
- PR #38 (`44efae1`): `1cfc901` docs: checkpoint project state after
  versioned fundamentals merge.
- PR #37 (`668a37c`): `b014e28` feat: add versioned point-in-time
  fundamentals (10 source / test files plus this file).
- PR #36 (`2025c1f`): `60fc6e7` chore: add Claude specialist agent team
  (`.claude/agents/`, eight agents).
- PR #35 (`d866fd4`): `3258e72` chore: add Claude project workflow
  automation (`.claude/settings.json`, hooks, `project-verify` and
  `project-checkpoint` skills).
- PR #34 (`0b7b1b8`): `add55fd` feat: integrate point-in-time EDGAR
  fiscal reconciliation.


Three older stashes (from `feature/macro-ingestion`,
`feature/fundamental-data-foundation`, `feature/equal-weight-baseline`)
are owner-owned and untouched.


## Current objective

Three tracks:
- **Phase 1, the Portfolio Lab:** first the rules-only Lab, then the
  first real-data forecasts.
- **The owner's 2026-10-14 brief:** the ML, RL and PPO engineering
  delegated to the agent (decisions below).
- **The owner's 2026-10-16 brief, the Stock Probability & Reversal Lab:**
  built on synthetic data (`docs/research/stock_probability_lab.md`); real
  data waits on the owner decisions A1-A6 listed there (section 7).

The immediate objective (owner, 2026-10-17): "real, traceable forecasting
evidence, not additional synthetic optimization." Two read-only audits come
first, each in the owner's terminal after the owner authorizes it:
- the Stock Lab audit of AMD, MRVL, MU and NVDA (5 requests);
- the ETF audit of VTI, AGG and SHV.

Storage, freezing, registration and activation each follow only after
their own approval.
- **ETF:** the first prospective 5/20/30-session forecast for VTI, AGG and
  SHV. It waits on:
  - the registration decisions (`docs/research/prospective_registration.md`,
    section 13);
  - the zero-volatility rule;
  - the hand-filled anchors and the storage approval;
  - the activation prerequisites in that document's section 12, which are
    not built yet.
- **T5a:** D1-D8 were approved on 2026-10-15 and the engine is
  implemented in PR #76. D9 stays pending.

- **Merged since the 2026-10-14 checkpoint:** PRs #57-#95 (the list
  above).
- **Wave 1 status:**
  - ML: runner, registry with evidence-derived stages (upper stages not
    attainable until the owner approves criteria), synthetic ETF track,
    challenger;
  - RL: environment and PPO, engineering only, labelled
    `engineering_only_unapproved_transition`, with no comparison tables;
  - Model Health: integrated into the Lab in PR #77;
  - Lab redesign;
  - T5a: the engine is implemented in PR #76 (D1-D8 approved 2026-10-15).

- **F1 foundations** (merged in PR #55) and the **Explore page** (merged in
  PR #56).
- **P1-1a, publication contract v1** (merged in PR #53, `src/stock_agent/publication/`):
  - a data-agnostic envelope, write-once writer and fail-closed reader
    sharing one `policy.validate`; every refusal carries a
    `RefusalReason`;
  - a closed per-kind policy table that admits only two synthetic kinds
    (`synthetic_descriptor` -> `descriptive`, `synthetic_rule_output` ->
    `rules_illustration`); real data is refused;
  - a restricted synthetic adapter (closed registry, integer-arithmetic
    generators) whose provenance check regenerates rows; this is an
    adapter-level protection, not a model for real-data provenance;
  - the lockbox through `MODEL_HOLDOUT_START` only; the synthetic-only
    knowledge rule d 23:59:59.999999Z;
  - hash integrity is documented as distinct from provenance; `code_sha`,
    `code_dirty` and `created_at` are unauthenticated producer claims.
- **T5-A** (merged in PR #57, `src/stock_agent/portfolio/descriptive.py`): the
  approved OD8 group A calculations, connected to the Explore page:
  total returns from raw closes, as-paid distributions and share-count
  splits (adjusted inputs refused); labelled multi-session returns across
  sessions without a close; no reinvestment through gaps; growth of
  $10,000 from the shared start; maximum drawdown only for a complete
  history, otherwise the largest observed decline labelled incomplete;
  calendar-year returns with unavailable years where a year-end value is
  missing. Run with `.venv/bin/python scripts/run_portfolio_lab.py`
  (synthetic demo data).
- **Tiingo adapter** (merged in PRs #58, #59 and #73):
  - allowlisted fields and verified splits; no adjusted levels or
    post-2024 metadata in the research store;
  - distribution "spot checks" against hand-filled private anchors, never
    committed, because of the Vanguard and BlackRock terms;
  - a read-only audit by default;
  - a separate prospective mode.
  No live request has been made.
- **RES-1 strategy** (merged in PR #54; owner direction 2026-10-13): ETF-first
  and cost-conscious. RES-1 does not block F1, P1-1b, the Portfolio
  Builder or fund-level allocation research; it remains the research
  dependency for individual-stock forecasting. A zero-purchase feasibility
  step (RES-1a) comes first. The 2026-10-13 vendor review found Sharadar
  Direct (Personal Use, Prices Full History $299/year) the only credible
  paid source for U1; none of the four sources compared gives CRSP-style
  delisting returns. No institutional (academic) access is available.
- **T5 formula specification** (merged in PR #52, documentation only):
  `docs/research/t5_portfolio_formulas.md`. Every formula is pending OD8;
  group A (descriptive, for P1-1b) is separated from the simulator groups
  B-D; an approval sheet for group A was delivered on 2026-10-13.
- **Product direction** (merged in PR #49, 2026-10-12; `docs/product/`;
  documentation only):
  - a rules-only, local Portfolio Lab as the first MVP, not a permanent
    limit;
  - architecture B, a publication pipeline with an evidence gate;
  - the validation ladder (location L0-L4, dispersion D1-D2), with explicit
    evidence gates for later forecast, uncertainty and allocation
    capabilities;
  - promotion stages S0-S4 toward investment-ready operation, with
    empirical, operational and risk-control requirements. PPO is a
    candidate allocation-policy engine, not a prerequisite or a shortcut;
  - the exact composition of absolute forecasts (the beta adjustment only
    under its documented approximation);
  - a separately governed prospective pipeline (OD16), designed, not
    activated;
  - T6 is holdout-checked and non-confirmatory; confirmatory inference uses
    only post-registration data after a burn-in. Every statistical
    procedure and power figure is provisional until simulated.

  Decisions are in `docs/product/vision_and_prd.md`, section 13, and below.
- **FRED key fixes** (PRs #48 and #50, merged 2026-10-12):
  - FRED Version 1 and Version 2 request failures raise `RuntimeError`
    naming only the series or release id and the HTTP status or exception
    type, with no chained cause, so neither the query-string key nor the
    bearer header reaches a message or traceback;
  - Version 2 transport and non-JSON failures changed from `TypeError` to
    `RuntimeError` (documented in PR #50);
  - `.gitignore` now ignores `*.log`.

  urllib3's logging of Version 1 URLs is recorded as a requirement before
  any shared or hosted deployment (`docs/product/risk_register.md`, SEC-2).

- The leakage-safe validation foundation is merged (PR #40); see
  `docs/research/model_validation.md`.
- The universe (U) and market-context (C) research is merged (PRs #41, #42);
  see `docs/research/universe_and_market_context.md`.
- T4 is merged (PR #47, 2026-10-12; `docs/research/linear_models.md`):
  - primary label: the point-in-time equal-weight excess forward return
    (each row on a date takes the latest member label end; incomplete
    dates are unlabeled; raw return stored in `inputs.target_return_raw`);
  - Ridge, Lasso and Elastic Net with penalties chosen by nested purged
    temporal validation inside each outer training fold (inner folds from
    the outer training rows only, pre-registered dimensionless grids,
    date-normalized inner MSE, one outer refit call); OLS as a
    `diagnostic` role comparator; Clark-West only for registered nested
    pairs (excess label: ols vs zero);
  - output contract 1.2: `coefficients` and `selection` tables with
    per-table versioning (1.0 and 1.1 runs stay readable);
  - checks: pre-training evaluation as-of vs folds, inner folds inside
    training, fit convergence, excess label centred, Clark-West one-sided
    (info), candidate below the per-symbol-mean selection control
    (research warning, never a gate);
  - dashboard "Linear models" page; candidate colours and markers in the
    design system;
  - the estimator core (`model_diagnostics/linear.py`, implemented under
    the owner's delegation of 2026-10-11): training-row median imputation
    and population standardization; Ridge alpha = c * n_fit, Lasso
    alpha = r * lambda_max, Elastic Net alpha = r * lambda_max / l1_ratio
    (lambda_max from each fit's own standardized rows); exact row-wise
    prediction; date-normalized inner MSE with a larger-penalty tie rule.
    The 12 phase A contract tests pass unchanged.
  - Development run `20261011T191112Z-50c1ca320e9c` (excess label, 49
    permutation draws, from `1338674` with a clean tree): schema 1.2,
    status WARNING, every leakage and instrument check passes, no pending
    predictions. Confirming run `20261011T222805Z-284566803c62` from
    `99a8405` (clean tree, after the review fixes) reproduced every stored
    table value for value.
  - Results (measured; fold-block t): no candidate shows detectable skill.
    Ridge rank IC +0.020 (t 0.36), d_t vs zero t 0.04, OOS R² +0.02%;
    Lasso and Elastic Net rank IC +0.070 (t 0.88) on only the 13 of 24
    folds where they are non-constant (Ridge scores 0.068 on those folds;
    zero-filled 0.038); OLS rank IC +0.018, d_t t -0.60; Clark-West (ols
    vs zero) t 0.37 (info). The per-symbol-mean control (rank IC +0.126,
    t 2.29) beats every candidate (selection, not skill; never a gate).
    Elastic Net duplicated Lasso's choice of r in 24 of 24 folds.
  - Reviewed independently (data-science, quantitative, test quality) on
    2026-10-11: no blockers, no leakage; should-fix items fixed in
    `99a8405` or recorded as owner decisions in `linear_models.md`.
- T3.1 (merged in PR #46) refined the dashboard before T4
  (`docs/research/diagnostics_dashboard.md`, "Maturity", "Goyal-Welch
  views", "Design system"). No T2 code changed.
  - Maturity: the evaluation as-of is the last session of the truncated
    development frame (`lockbox_evidence.max_date`). Only predictions whose
    label ended by then enter derived series; pending predictions are
    listed, and a stored pending prediction blocks the run.
  - Goyal-Welch page: the running sum of the date-normalized squared-error
    advantage d_t over identical observations, one line per recorded
    comparator with its identity; drawn only if it reproduces T2's stored
    estimates and curve. T2's observation-weighted cumulative is the
    secondary view. The per-symbol mean is a labelled selection control,
    never a default.
  - Descriptive filters (forecast dates, regimes) show counts and plain
    averages only; no inference is computed for a filtered sample.
    Point-in-time regimes use earlier-date thresholds; retrospective
    episodes are flagged ex post.
  - Design tokens drive the Streamlit theme, a CSS layer and Plotly
    layouts; shared components are ready for T4's model pages. Portfolio
    equity is a disabled future state until T5.
  - Reviewed independently by frontend, data-science and test reviewers on
    2026-10-10; every blocker and should-fix finding was addressed.
  - On the 2026-10-10 development run `20261010T192511Z-d967ec9c0ff2`
    (schema 1.1, produced from `9492892` with a clean tree): no legacy
    gaps, no record issues, status WARNING (per-symbol-mean selection
    control), no pending predictions, and all nine forecast
    model-comparator pairs reproduce T2.
- PR #45 (merged) fixed the T2 reporting contract found by T3 (output schema
  1.1, additive; `docs/research/model_diagnostics.md`, "Saved outputs"):
  - the checks summary adds per-family total / passed / failed /
    undetermined counts (the 1.0 summary omitted checks without a stored
    result);
  - the record states `decision_method`, `eligible_as_candidate` per model,
    and `reference_predictors` (the unpurged canary reference, never
    eligible); curve intervals record `ci_method`, `ci_level`, `hac_lag`;
  - the dashboard blocks a run whose summary or record disagrees with its
    check rows, refuses records that could present a non-candidate as a
    strategy, and shows 1.0 runs with every missing field labelled (never
    filled with assumed values);
  - no statistic, threshold, inference decision, label, lockbox or
    point-in-time rule changed.
- T3 (merged in PR #44) is a read-only Plotly and Streamlit dashboard over
  saved T2 runs (`src/stock_agent/dashboard/`,
  `docs/research/diagnostics_dashboard.md`):
  - it reads only verified run directories, refuses non-development runs
    and any stored date reaching the lockbox, and never fits, recomputes
    inference, or reads raw data (tested by a subprocess import check and
    20 mutation probes);
  - run status (VALID / WARNING / INVALID-BLOCKED) comes only from stored
    T2 check rows; warnings never escalate;
  - launch with `scripts/run_dashboard.py` (binds 127.0.0.1, usage
    statistics off).
- T3 findings in T2:
  - `read_run` could follow manifest paths outside a run directory. It now
    reads only the files a run may contain (owner-approved 2026-10-10,
    merged in PR #44).
  - The `checks_summary` defect and three output gaps (curve interval
    method, decision method, the unsafe reference's role and output kind)
    are fixed in PR #45 (schema 1.1). The other additive gaps remain listed
    in `docs/research/diagnostics_dashboard.md`.
- T2 (merged in PR #43) built the measurement layer
  (`src/stock_agent/model_diagnostics/`,
  `docs/research/model_diagnostics.md`):
  - nulls, controls and the canary run through the unchanged harness;
  - model-agnostic out-of-sample metrics;
  - Newey-West (lag 40), fold-block t and circular block bootstrap
    inference, with simulated false-positive rates recorded per run;
  - training- vs evaluation-scope feature diagnostics;
  - a run record and a versioned saved-output contract for T3.

  No model is fitted.
- T2 findings on the development run (2019-2024, 24 folds):
  - Nulls show no skill. The purged canary shows none either, while its
    unpurged reference shows spurious skill (rank IC 0.26).
  - The per-symbol mean reaches rank IC 0.126 (fold-block t 2.3) from
    hindsight selection alone. It is classified as a selection control, not
    a null.
  - Newey-West at lag 40 still over-rejects for persistent signals
    (simulated 7.6-9.2% at a nominal 5%). Fold-block t stays at 5.3-5.6%
    and decides the run checks.
  - Minimum detectable rank IC is about 0.16 at 4 names.
- Earlier findings (U and C research):
  - **The current 4-name universe can verify the pipeline but cannot
    support product evidence.** It is hindsight-selected and
    survivorship-biased, and its simulated minimum detectable rank IC is
    about 0.12.
  - **Planning target: about 100 point-in-time large caps** on
    survivorship-free data with history from 2010 (simulated MDE about
    0.020-0.025). This holds under the stated simulation assumptions; it is
    not a universal threshold, and it is re-estimated on the real panel
    before results are read.
  - **Context series are date-common.** They can affect cross-sectional
    ranking only through pre-registered interactions.
- T4's linear baselines are implemented; no prediction model is selected
  (none shows detectable skill on the sandbox universe).

Owner decisions (2026-10-11 and 2026-10-12, product direction, PR #49), from
the owner's answers and messages in session:
- 2026-10-11: OD1 and OD4 "Rules-only lab, personal"; OD5 "Strict: nothing
  after 2024"; OD2 "Pipeline; P1-1a, T5, F1 in parallel" (the architecture
  and starting those tickets, approving no T5 formula; OD8 is pending,
  formula by formula); OD6 and OD7 "Three references; semis deferred".
- 2026-10-11 revisions: make the investment-ready objective explicit, with
  PPO a candidate only; use the exact excess-return identity, with the beta
  adjustment only under a documented approximation; retain the 5-, 20- and
  30-session horizons; keep the sealed holdout while designing a separate
  prospective pipeline; keep the rules-only MVP as the first stage only;
  treat T6 as non-confirmatory and statistical promotion rules as
  provisional.
- OD16, direction only: "I approve the general direction of OD16: an
  independently governed prospective paper-trading pipeline, with a fixed
  historical holdout window and immutable registration/decision records."
  Activating the pipeline, choosing H_end, and the point-in-time transition
  rules each need a separate explicit approval and a documented
  pre-registration procedure.
- 2026-10-12, P1-1a, limited: "For P1-1a, record that I am comfortable with
  the proposed minimal v1 statuses `descriptive` and `rules_illustration`,
  and the interim end-of-session daily-close knowledge rule for
  synthetic-date fixtures only. Do not silently generalize this to
  production availability timestamps."
- 2026-10-13, P1-1a: "I approve Design B: a reusable, data-agnostic
  publication core with one shared validation contract, plus a strictly
  restricted synthetic-only adapter." "For synthetic fixtures only, I
  approve the explicit end-of-UTC-date convention: a synthetic date d
  becomes known at d 23:59:59.999999Z. This is not a production
  market-data timestamp rule." Also: "Do not make future real-data
  publication depend on regenerating data from synthetic specifications"
  and "Distinguish hash integrity from trusted publication provenance".
- 2026-10-13, T5: "PR #52 may be merged as a proposed formula
  specification after final CI and diff review. This does not approve any
  OD8 formula." For a future group D approval, F7.3 participation measures
  traded notional at each rebalance, with the initial construction a
  separate full purchase; bootstrap coverage settings stay provisional
  pending a registered simulation.
- 2026-10-13, feature engineering: "Approve FE-0 and FE-1 as contracts,
  synthetic tests and outcome-blind feature-build designs. Do not score
  E1-E3 on the current hindsight-selected four-stock universe or on U1
  before registration." Membership and labels stay independent of feature
  missingness. The E1-E3 pre-registration design stays pending until the
  RES-1 data source and label contracts are settled.
- 2026-10-13, RES-1: start data-source feasibility; "Do not purchase data
  or unlock the 2025+ holdout."
- 2026-10-13, RES-1 correction: "I am no longer a Georgia Tech student.
  Remove Georgia Tech student-library access and the WRDS-via-Georgia-Tech
  route from active RES-1 recommendations, owner action items and future
  prompts." "Institutional data may be considered only if a new
  independently available access arrangement is established." RES-1 is
  cost-conscious and "should not block F1, P1-1b, the Portfolio Builder or
  initial ETF allocation research". "Do not purchase data, implement a
  vendor adapter, run new real-stock scoring experiments or change the
  historical universe yet."
- 2026-10-14: "I approve VTI, AGG and SHV as the first three reference
  funds." "I approve the frozen 2003-2024 NYSE calendar for the initial US
  equity-ETF prototype, with exceptional closures tested. This does not
  establish a complete bond-market, international or future-date
  calendar." "I approve evaluating the Tiingo Starter API for private,
  local research, subject to its terms and a successful coverage audit."
  An upgrade to Power ($30/month) only with a demonstrated benefit, never
  automatically; the owner configures the key locally; no live vendor
  audit until the owner confirms the token and authorizes the restricted
  pre-2025 queries.
- 2026-10-14, OD9 (descriptive display only): "use 18:00
  America/New_York as an explicitly assumed close-availability
  convention. This is not proof of final vendor publication time and must
  not be used as evidence of historical trading executability or
  forecasting validity." OD9a (dates after the lockbox on earlier
  ex-dates) is not approved; the store stays fail-closed.
- 2026-10-14, OD8 group A: A1-A8, R1, R2, R4 and R5d approved; R3 wording,
  R5a approved (labelled multi-session returns); R5b not approved (no
  reinvestment through gaps); R5c thresholds not approved (conservative
  incompleteness). The record is in `docs/research/t5_portfolio_formulas.md`,
  section 9. The T5a-T5c simulator formulas stay unapproved (open
  decisions D1-D8, `docs/research/t5a_open_decisions.md`). PPO engineering
  is authorized by the 2026-10-14 brief below.
- 2026-10-14, accelerated ML/RL/PPO brief (owner, verbatim):
  - "I explicitly delegate implementation of the entire ML, RL and PPO
    codebase to you, including any previously owner-reserved RL algorithm
    core."
  - "Begin the RL environment interfaces and synthetic tests immediately,
    in parallel with T5a. After the T5a transition has been validated,
    connect it as the environment's single source of portfolio
    economics."
  - "A PPO implementation using a maintained library where appropriate,
    with pinned dependencies and deterministic test fixtures."
  - "Build the complete PPO engineering system now, but distinguish a
    successful training run from validated economic superiority."
  - First-wave deliverable: "PPO training that completes successfully on
    an authorized synthetic dataset."
  - Authorized: "Make routine implementation and package choices, with
    dependencies pinned and reviewed." and "Commit, push, open PRs and
    prepare CI workflows." Also: "Create a scheduled GitHub Actions
    health workflow for safe checks that can run in CI using synthetic
    fixtures and repository metadata."
  - Still owner-gated: "Merging into main." "Financial-formula or
    reward-policy changes." "Historical holdout access, H_end or
    prospective-pipeline activation." "Data purchases, licensing
    commitments, or unauthorized live network collection." "Real-money
    transactions or brokerage connections." "Relaxing point-in-time
    validation or research evidence standards." "Publishing private
    data, credentials or personal information."
  - "Do not claim that PPO is investment-ready until the necessary
    out-of-sample and prospective evidence genuinely exists."
- 2026-10-14, lead decision pending owner confirmation (not an owner
  decision): until D1-D8 are approved, PPO smoke training runs on a
  transition implementing the proposed D1-D8 rules; every artifact is
  labelled `engineering_only_unapproved_transition` and no strategy
  comparison table is published. The owner's earlier rule, "Do not train
  or evaluate PPO on an economically uncorrected simulator", is why the
  label exists. The owner may overrule this at merge.
- 2026-10-14, prospective forecasting track (owner, by selecting these
  options in the lead session; the quoted text is the option the owner
  chose):
  - Path: "Start prospective track". Register ETF forecasters at 5, 20
    and 30 sessions, frozen before the first forecast. Record each forecast
    once, timestamped, labelled "unvalidated, no demonstrated skill", and
    score it only as real outcomes arrive.
  - H_end: "H_end today, lookback OK". The sealed holdout is 2025-01-01
    to 2026-10-14 (H_end), kept for one historical evaluation. A separate
    prospective store may use the last 63 sessions before H_end only as
    inputs to frozen models, never to train, tune or evaluate. A test
    enforces that research code cannot read the prospective store.
  - Anchors: "Fill by hand". The owner copies issuer amounts into the
    private, git-ignored anchors file (issuer terms: personal use, no
    automated access). The rows the research agents gathered are not
    used as anchors.
  - Live data: "Audit + prospective". First the read-only pre-2025 Tiingo
    audit (VTI, AGG, SHV), then collection into the separate prospective
    store. Each run happens in the owner's terminal, and storing anything
    still needs the owner's OK at that step. Lead implementation choice
    (more conservative than authorized): the first collection requests
    only the 63-session lookback before H_end and later sessions; the
    rest of the sealed period is not collected until the one-time
    historical evaluation.
  - Target: "Risk and return". Forecast volatility and total return, each
    with an 80% band, with the return forecast shown next to the
    historical-average null.
- 2026-10-15, Wave 2 brief (owner, verbatim where quoted):
  - T5a: the owner approved D1-D8, "provided they exactly match the
    current `docs/research/t5a_open_decisions.md` packet". The approval is
    recorded verbatim against each D identifier in that packet (commit
    `8b5e564`).
    - D4 (a) holds for synthetic engineering only.
    - One D5 clause ("Stricter constraints are explicitly registered for
      learned policies") awaited confirmation; it was confirmed on
      2026-10-17 (below): constraint sets apply identically to every
      compared policy.
    - "Keep D9 separate for later owner review."
  - The prospective window: "My intended 63-return input window is 64
    consecutive closes ending at the forecast close. For the initial
    October 14 endpoint this means July 16 through October 14, 2026, used
    only as inputs to frozen models, not for research training, tuning or
    sealed-holdout evaluation." "This clarifies the window definition; it
    does not authorize retroactive forecast records or unrestricted
    sealed-period access."
  - Still to be decided before any real forecast is recorded, each
    escalated to the owner:
    - "Confirm the exact registration, first eligible decision time,
      burn-in and prospective confirmation rules. No forecast should be
      backdated, and no paper-trading activation should be inferred
      merely because logging code exists."
    - "Before freezing a real model, review the current treatment of
      zero-realized-volatility outcomes ... Escalate the target-formula
      decision for approval."
    - "Verify the 2025-2027 NYSE calendar against authoritative exchange
      information and document any unresolved holidays."
  - Data: "Do not store vendor data until I explicitly approve that step."
    The owner runs the read-only audit locally.
  - Monitoring: "The GitHub health workflow must use only public or
    synthetic-safe information. Private local data and real forecast
    monitoring need a separate safe local process."
  - Not authorized: "Do not make changes to live investment decisions,
    model promotion claims, training targets, historical lockbox access,
    data purchases, source terms or reward formulas outside the
    authorizations above."
  - Activation boundary (lead reconciliation, not an owner decision):
    - Decided: H_end, the window definition, the prospective path, the
      hand-filled anchors, the owner-run audit and collection, and the
      forecast targets.
    - Not decided: the registration rules, the first eligible decision
      time, burn-in, the confirmation rules, the zero-volatility target
      treatment, storing any batch, and OD16 activation (recording a real
      forecast, paper trading).
    - No real forecast may be recorded until all of these are approved.
- 2026-10-16, the Stock Probability & Reversal Lab brief (owner; relayed
  to this session by the lead session), in paraphrase:
  - build it as a lightweight, modular feature independent of the
    ML/RL/PPO system, starting with synthetic fixtures and eligible
    development history;
  - the stock universe is small and hindsight-selected, so its results never
    validate the forecaster;
  - the historical holdout stays protected; any post-2024 use for training,
    calibration or historical scoring needs a separately documented owner
    decision; recent authorized observations may serve as inference inputs;
  - the free Tiingo Starter plan is acceptable initially; before the first
    live call or real-data storage, one concise authorization request with
    the exact scope and command (given in
    `docs/research/stock_probability_lab.md`, section 7; not yet answered);
  - do not alter the PPO reward or the portfolio simulator.
- 2026-10-16, the Stock Lab decisions A1-A6 (owner; the full conditions
  are in `docs/research/stock_probability_lab.md`, section 7):
  - A1: "I approve the proposed Tiingo Starter provider and the architecture
    for a private watchlist data store", staged: "separate data inspection
    from persistence"; the read-only command runs only after the owner
    confirms the token and "explicitly authorize[s] that command"; storage
    is "a separate" approval after "a successful coverage and
    corporate-action audit"; the 30-ticker cap; "No vendor-adjusted price
    fields, unauthorized history or data redistribution."
  - A2: the trailing 252-session input window "approved in principle":
    models fitted, tuned, selected and calibrated only on eligible pre-2025
    data; protected-period observations "only as inputs to a previously
    frozen model", kept "in the separately controlled prospective/inference
    store"; "a scoped authorization for the Stock Probability Lab, not a
    general unlocking of the 2025+ holdout."
  - A3: exploratory walk-forward evaluation of the saved watchlist on
    eligible pre-2025 history, "hindsight-selected and non-confirmatory";
    every challenger against B0/B1 with proper scores, calibration,
    stock-level coverage, uncertainty and multiple-comparison treatment.
  - A4: the write-once forecast and scoring protocol approved; activation
    "contingent on successful data validation, a frozen registered model,
    verified decision timestamps and a successful end-to-end dry run";
    separate from ETF OD16 and PPO promotion.
  - A5: the version-1 definitions (±0.5% daily; ±0.5% × √5 for the
    five-session total; a Stable session ends a streak by default; the
    bull-trap setup parameters for research); the failure parameter f is to
    be confirmed "with an exact formula and numerical example before
    enabling real-data risk flags".
  - A6: recent observed prices on the private dashboard, through "an
    isolated descriptive reader", after the retrieval and storage
    permissions; "Do not silently modify the existing holdout policies".
  - Unchanged: the PPO reward, T5a portfolio economics, ETF prospective
    governance and the main model-evidence rules.
- 2026-10-17, Wave 2 follow-through brief (owner, verbatim where quoted):
  - Simulator interpretations: "I approve the following interpretations
    of the existing D1–D8 contract". They are recorded verbatim in
    `docs/research/t5a_open_decisions.md`:
    - D5: a registered constraint set applies identically to every
      compared policy;
    - D6: a hold preserves a deferred target, and a new valid target
      supersedes it;
    - D5: a registered turnover cap also applies to the initial build,
      reporting executed weights, uninvested cash and any incomplete
      construction.

    "These are confirmations of the existing approved simulator rules,
    not authorization for different financial formulas."
  - Zero volatility: "I support investigating a prespecified positive
    floor for the log-volatility target, but I am not approving an
    arbitrary replacement value yet."
    - "Keep measured zero volatility distinguishable from the modeling
      floor."
    - "Do not use protected holdout outcomes to select the floor."
    - The formula decision is pending.
  - Prospective governance principles the owner supports (not yet final
    activation approval):
    - "Register and freeze the model before its first real forecast."
    - "Begin recording no earlier than the first eligible close after
      registration and verified data availability."
    - "Never backdate or overwrite forecasts."
    - "Treat an initial 20-session trial as a prospectively registered
      exclusion, not a period that can later be selectively included."
    - "Use a preregistered test family and Holm correction where
      appropriate."
    - "Account for overlapping 5-, 20- and 30-session outcomes."

    Also:
    - "Do not claim prospective validation merely because forecasts have
      been logged."
    - "Keep the forecast-ledger head hash privately backed up outside the
      ledger. Do not commit it publicly yet."
  - Comparison: "Proceed with the approved synthetic comparison
    engineering." It covers:
    - the same transition, observation timing, costs, rebalance schedule,
      initial build and constraints for every strategy;
    - buy-and-hold, equal weight, fixed balanced, inverse volatility,
      mean-variance, CEM, PPO and an all-SHV baseline.

    "No selection on the final test set and no claims of real investment
    profitability."
  - Real data: "Keep the Tiingo read-only audit ready for my local
    authorization. Do not silently collect or store real data."
  - Stock Lab: "Treat the new daily stock-probability feature as separate
    from the ETF forecasting and PPO evidence programs." and "Keep its
    synthetic probabilities clearly distinguished from actual calibrated
    forecasts."
  - Not authorized: "Do not merge PRs, approve new financial target
    formulas, unlock holdout data, authorize paid data, or activate
    prospective investment claims on my behalf."
- 2026-10-17 (New York), the "Finalize PRs and Prepare Real-Data
  Activation" brief (owner; verbatim where quoted):
  - **Stock Lab watchlist:** "Exclude SKHY and SNDK from the Stock
    Probability & Reversal Lab." Also: "Do not fetch new vendor history
    for either excluded ticker."
    - Done on 2026-10-17 through the Lab's Stock Probability page, as two
      append-only removal events. The dry run of
      `scripts/stock_lab_fetch.py` lists AMD, MRVL, MU and NVDA (4 of at
      most 30).
    - Forecast logs, scores and audit evidence are kept.
    - The ETF universe, PPO benchmarks and historical research contracts
      are unchanged.
    - The watchlist is hindsight-selected, so its evaluation is
      exploratory.
  - **A7 (preference; implementation waits for a final confirmation):** the
    fold-block t at a one-sided 2.5% family level, with Holm across the
    registered challengers. B1 stays the default. Real evaluation results
    must not change the threshold.
  - **ETF primary loss (provisional):** squared error on log-volatility,
    with QLIKE as a sensitivity. "Do not interpret that preference as
    automatic approval." The model stays unfrozen and the 2025+ holdout is
    preserved.
  - **Groups C and D:** "Do not treat merging the decision packet as
    approving or implementing those formulas." Each group is implemented
    only after approval, in its own reviewed PR.
  - **Real Stock Lab data:**
    - "I will configure the API token locally. Do not display, log,
      request or commit it."
    - "Wait for my explicit authorization before making live Tiingo
      requests."
    - The first run is read-only: "No `--store`, no model fitting on
      sealed history and no automatic activation of forecasts".
    - Storage needs a separate authorization. Prospective five-session
      forecasts come only after a registered, trained and frozen model and
      authorized inference, kept separate from PPO evidence.
  - **Model Health:** the owner prefers a dedicated, clean monitoring
    checkout. The exact dry-run configuration, paths and installation
    steps come before any launchd change.
  - "Our immediate objective is real, traceable forecasting evidence, not
    additional synthetic optimization."
  - Not authorized: merging ("Do not auto-merge"), formula approval,
    purchases, protected outcomes for training, and trading activation.
- 2026-10-18, the ETF registration build's breaking contract changes
  (owner, by choosing these options in the lead session; the quoted text is
  the option chosen):
  - "Approve both (Recommended)":
    - `zero_volatility_rule` becomes a required key of the registered
      config. Its hash changes from `ae5d5853…` to `dba70fc2…`, and a
      config or bundle without the key is refused.
    - The forecast job and scoring refuse a ledger that does not begin
      with a registration entry.

    No real config, frozen model or ledger existed at the time.
  - "Refusal is final (Recommended)": a whole-session refusal is that
    session's one record. No forecast is recorded later for the same
    session.
  - These approve the contract changes only. They approve no section 13
    decision, no zero-volatility rule change, and no freeze, registration
    or activation.
- Pending:
  - Stock Lab, in order (open item 10):
    1. authorization of the A1 read-only audit, after the token is
       configured;
    2. the separate storage approval;
    3. the confirmation of the bull-trap f;
    4. A7, the final confirmation of the rule in
       `docs/research/stock_lab_a7_decision.md` (owner preference above);
    5. A4 activation.
  - the rest of OD3;
  - OD8 groups C-D (cash returns and benchmarks; statistics and
    reporting). The decision packets are merged (#92); nothing in them is
    approved. Group B, the D1-D8 packet, was approved on 2026-10-15
    (above); its D5 clause and two interpretations were confirmed on
    2026-10-17 (above);
  - D9 (the Builder's buy-and-hold mix), separate from group B;
  - OD9's `signal_cutoff_at` for decisions and the production
    availability rule beyond descriptive display;
  - OD10-OD15 (OD13's sub-decisions are proposed);
  - OD16: activation (recording any real forecast, paper trading) and
    the transition rules. Still pending:
    - the seven decisions in `docs/research/prospective_registration.md`
      section 13 (merged in #95): registration, timing, the trial, the
      metrics, C1-C4, the primary loss and the recording deadline;
    - the zero-volatility rule. B2 is research-only, and its
      development-data check needs the stored pre-2025 batch.

    H_end = 2026-10-14 and the prospective window are decided.
  - OD17.

Owner decision (2026-10-11, T4 phase B/C): "I explicitly authorize you to
implement the remaining `linear.py` estimator core, including
preprocessing, Ridge/Lasso/Elastic Net wrappers, prediction contracts and
nested hyperparameter-selection logic. This supersedes the earlier
task-specific requirement for me to manually implement that module."
The learning objective is kept through explanations, Learning Notes and
one optional exercise. The same brief: keep Clark-West secondary and
informational; the per-symbol mean is not a hard gate for the four-stock
universe; record the variable-universe complete-cross-section and
delisting problem as mandatory before large-universe validation, and the
row-weighted fitting versus date-weighted tuning question; do not change
leakage, label, execution or lockbox semantics without a genuine
correctness issue.

Owner decisions (2026-10-10, T4 brief and answers): primary alpha target =
point-in-time cross-sectional equal-weight excess return, raw return
preserved as secondary data (stored; metrics on the excess label);
scikit-learn pinned; Ridge, Lasso and Elastic Net through the T1/T2 harness
with hyperparameters chosen only by nested purged temporal validation
inside each outer training fold; OLS only as an optional diagnostic
comparator; reuse T2 metrics, inference, nulls and controls and the T3
dashboard; no lockbox; four names are not product evidence; the
per-symbol mean is a research warning or control, not a hard gate;
Clark-West only for truly nested comparisons; candidate colours in the
design system; cross-check evaluation_asof against fold
score_available_at before training and stop on disagreement; the owner
codes the estimator core (learning-core split; superseded 2026-10-11).
The plan approved on 2026-10-10 covers the excess availability rule, the
`diagnostic` role, the additive 1.2 tables and the run script's default
label.

Owner decisions (2026-10-10, T3.1 brief): run T2 once under schema 1.1
without evaluating the holdout; Goyal-Welch uses identical observations and
point-in-time comparators with no single hard-coded primary baseline, the
per-symbol mean kept as a selection control; the date-normalized advantage
is the primary view and the observation-weighted total secondary; only
predictions with target_end_date <= evaluation_asof contribute, pending
ones stay visible; filtering never recomputes inference in T3; regime
filters separate point-in-time from ex-post labels; keep Streamlit and
Plotly; no portfolio economics before T5; no new models, inference
methods, universe or label changes.

Owner decisions (2026-10-10, T2 reporting fix): fix the T3-found defect and
gaps in the same T2 output-contract ticket with additive fields; keep
`decision_method` an explicit record field; store interval method and level
as data; make the unsafe reference's role machine-readable; fail closed on
summary-row disagreement; do not coerce missing 1.0 fields; change no
statistics, thresholds, inference decisions, labels, lockbox or
point-in-time rules.

Owner decisions (2026-10-09 and 2026-10-10, T3):

- Pin `plotly==7.1.0` and `streamlit==1.64.0`. Streamlit excludes pyarrow
  25.0.0, so pyarrow moves to 25.0.1 and stays unpinned ("Allow 25.0.1, no
  pin").
- Keep the stricter `read_run` in the T2 reader ("Keep in T2 reader").

Owner decisions (2026-10-06):

- ML labels enter at the close after the feature date:
  `MODEL_LABEL_SPEC = LabelSpec(horizon=20, entry_lag=1)` in
  `features/training.py`. `add_forward_return_target` keeps its default
  `entry_lag=0`, so existing callers and tests are unchanged.
- Data from 2025-01-01 onward is a lockbox. Development folds end by it and
  never score labels that mature in it; holdout folds run once per
  pre-registered model.

In effect on `main` (`src/stock_agent/model_validation/`): expanding
forward-only folds; strict global purge (train iff
`target_end_date < test_start`); no embargo, justified and asserted per fold;
fit-once, train-only transforms with a fresh estimator per fold; a feature
contract plus a content-based prefix-stability check; canonical fingerprints
and per-fold metadata.


## Current EDGAR architecture

Modules under `src/stock_agent/data/fundamentals/`:

| Module | Responsibility |
|---|---|
| `edgar_source.py` | List financial filings and metadata (form, period end, filing date, acceptance time, accession, XBRL flag) |
| `edgar_fiscal.py` | `infer_edgar_fiscal_identity`: authoritative fiscal year / period / quarter from XBRL reporting periods |
| `edgar_concepts.py` | `resolve_statement_concept`: map XBRL concepts to canonical metrics |
| `edgar_periods.py` | Parse statement period columns; get direct-quarter, YTD, FY, instant values |
| `edgar_quarterly.py` | `build_edgar_quarter`: one filing to one canonical row; availability rule (`resolve_edgar_available_at`); `extract_fiscal_flow_observations`: single-filing direct / YTD / FY observations |
| `edgar_reconciliation.py` | Current- and prior-observation selection and cross-filing reconciliation (`select_current_fiscal_observation`, `select_prior_fiscal_observation`, `reconcile_fiscal_flow`, `reconcile_fiscal_flow_from_candidates`, `find_superseding_prior_observations`); optional `as_of` knowledge time |
| `edgar_history.py` | `build_edgar_history`: collect filings, index raw observations once, emit period-centric versions; returns versioned canonical frame, reconciliation lineage, diagnostics |
| `quarterly_schema.py` | Quarterly fundamental schema and validation |
| `point_in_time.py` | `align_quarterly_fundamentals_asof`: current-state change log, then one as-of join to market dates |
| `features.py` | Quarterly ratios and YoY growth (availability-guarded prior-year match) |


## EDGAR reconciliation (merged in PR #34)

- Authoritative EDGAR fiscal identity (`ec006a4`)
  drives every observation; quarters never come from calendar months.
- Deterministic cross-filing fiscal-flow reconciliation; reconciliation
  is the single authority for flow values in the history and is
  independent of filing order.
- Prior-observation selection: filtered by availability before ranking;
  latest available wins; equal-value ties take the smallest accession;
  conflicting values are unavailable (`ambiguous_prior_observation`); a
  newer amendment without a usable value is skipped and recorded.
- Fiscal-Q1 direct-value fallback for Q2 derivation
  (`prior_value_source="q1_direct"`); conflicting Q1 direct/YTD values
  are rejected; never applied to Q2/Q3 priors.
- Corrected deferred SEC availability semantics (below), with DST-safe
  06:00 America/New_York construction.
- `not_yet_available` pre-load filtering in strict and non-strict modes.
- Real-shaped MU regression fixtures (offline) and controlled real MU
  validation (below).


## Versioned EDGAR history (merged in PR #37)

Design owner-approved 2026-10-05, including the contract changes below
and the AGENTS.md interpretation that per-metric carry-forward is not a
"substitution" (no AGENTS.md change). Described as verified 2026-10-05
against the pre-commit working tree; committed as `b014e28` and merged
as `668a37c`:

- Versions are period-centric. For each fiscal period, each DISTINCT
  `available_at` of its own filings or the immediately preceding fiscal
  quarter's filings is evaluated once (same-instant events coalesced),
  using only raw filings available at or before that time.
- Per metric, the source is the latest own-period raw filing with a
  usable value; omitted metrics carry forward; same-instant conflicting
  values are unavailable (`ambiguous_current_observation`).
- A version is emitted when an own-period filing becomes available
  (always visible) or when a metric's (value, method, reason) changes.
  Fixture example: Q3 revenue 9000 at Q3, 9200 at the late Q2/A, 9700 at
  Q3/A. Historical rows never change.
- Raw SEC observations are the only reconciliation inputs; the candidate
  index is read-only and versions are outputs only.
- Version row: anchor (newest own-period filing) supplies
  `sec_accession_number`, form, filing date; `available_at` = event time;
  `availability_source` = anchor's rule if the anchor became available
  then, else the smallest-accession trigger's rule.
- Contract changes (owner-approved): canonical duplicate key adds
  `available_at`; EDGAR history unique on
  `(symbol, period_end, available_at)`; lineage key
  `(symbol, sec_accession_number, available_at, metric_name)` with new
  columns `source_accession_number`, `source_available_at`,
  `version_trigger_accessions`.
- Filings whose `period_end` conflicts with the period's earliest
  filing are rejected (`conflicting_period_end` diagnostic) and excluded
  from all inputs; strict mode raises.
- `derived_quarter_input_superseded` diagnostic retained.
- As-of: latest `period_end` known at the market date, then that
  period's latest version; a late amendment of an older period never
  becomes current. Duplicate key `(symbol, period_end, available_at)`.
- YoY: prior-year rows must be available no later than the current row;
  unknown availability is never eligible (NaN growth). Matching and
  growth formulas unchanged.
- Reconciliation formulas unchanged (checked 2026-10-06 with
  `git diff 2025c1f b014e28`): no arithmetic lines removed; the
  prior-availability guard now compares against an `as_of` knowledge
  time that defaults to, and may never precede, `current.available_at`.
- Verification: two independent reviews (test-reviewer, data-engineer)
  found no blockers; 12 of 12 mutation checks caught (backdating,
  filing-centric re-derivation, availability-only as-of, latest-overall
  prior, no-op versions, erasing omitted metrics, wrong lineage source,
  no coalescing, YoY guard removed, future prior in lineage, rejected
  filing as input, suppressed re-versions).


## EDGAR availability semantics

```text
accepted_at   SEC acceptance time
filing_date   SEC-assigned filing date
available_at  earliest timestamp this pipeline permits the information
              to be consumed
```

- `accepted_at` missing: `filing_date + 1 day` (`sec_filing_date_plus_1d`).
- Deferred filing (SEC `filing_date` later than the America/New_York date
  of `accepted_at`): `filing_date` at 06:00 America/New_York, converted
  to UTC (`sec_deferred_filing_date_6am`).
- Otherwise: `accepted_at` (`sec_acceptance_datetime`).
- `available_at` is never earlier than `accepted_at`.

06:00 ET for deferred filings is a conservative modeling proxy based on
the SEC filing rule (submissions after 5:30 p.m. ET receive the next
business day's 6:00 a.m. ET filing date and are not disseminated until
that day). It is not a measured dissemination timestamp.


## Controlled real-data validation (MU)

Seven MU XBRL filings, FY2025 Q1 through FY2026 Q3, edgartools 5.58.0,
validation only (run before PR #37; not re-run on the versioned
history):

- Fiscal identity matched MU's 52/53-week calendar.
- Adapter `accepted_at` is true UTC (API `23:52:13Z` equals SEC header
  `ACCEPTANCE-DATETIME 18:52:13` EST).
- FY2025 operating cash flow 3,244 / 3,942 / 4,609 / 5,730 USD M sums to
  the reported FY 17,525.
- Six of seven filings were accepted after 5:30 p.m. ET; their
  `available_at` moved to 06:00 ET on the SEC filing date (about 11 hours
  later). Values, methods, and selected priors were unchanged.
- Real-data prefix stability held at every knowledge time; no artifacts
  entered the repository.


## Real XBRL amendment validation (SMCI)

Owner-reported 2026-10-06: run after the PR #37 merge on Super Micro
Computer (SMCI) FY2017 filings, including amendments. Validation only;
not re-run in this session. No artifacts entered the repository
(verified 2026-10-06: clean tree, no SMCI references in tracked files).

No point-in-time correctness defect was found:

- Original versions stayed historically visible; amended versions
  became visible only at their own `available_at`.
- Later filings did not alter earlier histories.
- Downstream quarter re-derivation after amendments was correct.
- Lineage source / prior timestamps and accessions were correct.
- As-of selection chose the newest fiscal period first, then that
  period's newest version.
- Manual operating cash-flow reconciliation matched the reported FY
  values.

Coverage limitations found are tracked as open item 2.


## Last verified quality baseline

**Run.** Measured 2026-10-17 (New York) with `PYTHONPATH=$PWD/src python3
scripts/verify.py` (output to a log file, not piped). It ran on
`checkpoint-wave3` at `79c80ef` (= `origin/main`) in a session worktree.
The only change in the tree was this checkpoint's edit to this file.
`PYTHONPATH` makes the shared venv import the worktree's sources. Exit 0.

**Tools** (`.venv` pytest 9.1.1 and ruff 0.16.2 as reported by the run; the
rest as recorded 2026-10-15, not re-checked): Python 3.12.2, pandas 3.0.5,
numpy 2.5.1, pyarrow 25.0.1, streamlit 1.64.0, plotly 7.1.0, scikit-learn
1.9.1, scipy 1.18.1, torch 2.14.0, stable-baselines3 2.9.0 and gymnasium
1.3.0.

- `pytest -q`: 4724 passed (349 s)
- `python -m pytest -q`: 4724 passed (334 s)
- `ruff format --check src tests scripts .claude/hooks`: 515 files
  already formatted
- `ruff check src tests scripts .claude/hooks`: all checks passed
- No xfail remains: the 54 strict T5a `portfolio_step` xfails ended with
  the engine in PR #76.
- Reviews of #82-#95 (not re-run) are recorded in their PR descriptions.
  Review findings on #90 and #91 that arrived after those PRs merged were
  fixed in #94 and #95.
- Stock Lab reviews (2026-10-16, not re-run), notes in the session notes
  folder; every blocker and should-fix item was fixed before its commit:
  - **PR 1:** a test review (49 mutants, 38 killed at first; the gaps,
    including a split-mismatch level leak, were fixed and tested),
    security and UX reviews;
  - **PR 2:** a statistics review whose perturbation probe found no
    leakage (an unpaired one-SE rule, calibration labels and the
    Kaplan-Meier at-risk rule were fixed), and a test review (37 mutants);
  - **PR 3:** a test review (17 mutants, 12 killed) whose blocker
    (scoring on an outcome seen before the forecast after a re-download)
    is fixed in `2ac408e`.
- Independent reviews and mutation checks, 2026-10-14 and 2026-10-15
  (not re-run):
  - **T5a (#61):** the oracle numbers were re-derived with exact
    fractions. A separate weights-based implementation passed all 48
    xfail tests, and every mutant tried was caught (cash-leg cost,
    additive cost, same-close execution, free build, `r_next` visible,
    double-counted reward).
  - **RL (#62, #68):** 16 mutants, 15 killed. The survivor (a future
    `has_close` row) was killed after #68.
  - **Data path (#73):** 31 mutants: 27 killed, 2 equivalent, and 2
    fixed by follow-up tests.
  - **Forecasts (#74):** 15 mutants: 11 killed. Tests added for 2 (the
    null window and the stage cap); 1 equivalent.
  - **ML stack:** the surviving challenger-imputer mutant was killed after
    the fix.
  - **Health (#67), Lab (#65), registry export (#72):** security,
    frontend and statistics reviews; every blocker and should-fix item
    was fixed before merge.
- Mutation checks (T5-A): 11 of 11 probes killed (dropped distribution,
  ignored split ratio, span chained from the previous session,
  reinvestment through a gap, t0's own return applied, history always
  complete, year-start base, losing on a rounded value, last value instead
  of the year-end value, start without the common-close check, adjusted
  closes accepted). An independent statistical review recomputed every
  hand example; its findings were fixed before commit.
- Mutation checks (P1-1a): 18 of 18 probes killed, each run on the
  sources and restored. Probes: lockbox comparison and calls; `validate`
  removed from the writer or the reader; knowledge time at 00:00Z or
  21:00Z; the provenance check and its kind/status/limitations check;
  the existence check; failure cleanup; lax timestamp parsing; the attrs
  and index checks; weight bounds and per-date sums; the code-SHA recheck;
  `match` instead of `fullmatch`. An independent test review found one
  blocker (unhashed parquet `attrs` and a custom index passed the reader)
  and should-fix items; all were fixed before commit.
- Mutation checks (T4, not re-run): phase A 18 of 18 caught; the phase C test review
  ran 49 probes on `1338674` (22 survived); after the fixes, 22 targeted
  probes (those survivors and the new code) were all killed, each in its
  own scratch copy. Earlier mutants (T3.1: 22, schema 1.1: 21, T3: 20,
  T2: 47, PR #40: 34) were not re-run.


## Open items

1. One decision timestamp. The code has three implicit decision times:
   - macro uses all of calendar day d;
   - fundamentals and the panel validator use 00:00 UTC on d.

   Proposed: `signal_cutoff_at` = 18:00 America/New_York on session d,
   with timestamp-level `available_at` (source publication time) for all
   sources (`docs/research/universe_and_market_context.md`). It is kept
   separate from:
   - any later pre-execution event and news review;
   - `execution_at`, the close of d+1.

   This is an availability-semantics change, so it is owner-gated. Not
   leaking today; fundamentals are stale by a day.
2. Real-data coverage gaps from the SMCI validation (coverage and
   diagnostics; no point-in-time defect found), in suggested priority:
   - edgartools can mislabel 52/53-week fiscal calendars; confirm
     whether affected filings are rejected or accepted.
   - Later comparative restatements are not ingested.
   - One ambiguous concept mapping rejects the entire filing.
   - Part-III-only 10-K/As are skipped under the same reason code as
     genuine errors.
   - Current-period columns can sometimes be missing.
   - Synthetic amendment fixtures are cleaner and more staggered than
     some real filings.
3. Version rows copy non-flow columns from the anchor; extend
   per-metric carry-forward before EDGAR populates balance-sheet / EPS /
   capex columns.
4. Two fiscal periods sharing `(period_end, available_at)` raise even in
   non-strict mode (rejection rule is an owner decision).
5. YoY: no refresh when a prior-year amendment arrives later; nearest-
   match fallback when the nearest prior-year row is not yet available.
6. ML validation follow-ups:
   - **Blocker for treating results as product evidence: no security master
     and no survivorship-free prices.**
     - yfinance returns nothing for delisted tickers (XLNX, MXIM, LLTC
       verified empty).
     - Tickers are reused (SNDK) and CIKs change (MRVL in 2021).
     - The universe is 5 semiconductor names chosen with hindsight.

     The U track in `docs/research/universe_and_market_context.md` covers
     the contract, rules R1-R8, and source options. Data licence and cost
     are an owner decision.
   - The risk-free rate (FRED DGS3MO) has local history only from
     2024-01-02: `scripts/download_rates.py` hard-codes a first start of
     2024-01-01. Backfill to at least 2009 before T5 portfolio metrics. Rates
     storage also overwrites revisions and has no `available_at`.
   - Inference (addressed in T2, merged in PR #43):
     - Newey-West defaults to lag 40, beside a fold-block t and a 63-session
       block bootstrap. Simulated sizes are recorded in every run.
     - Remaining: lag 40 with normal critical values still over-rejects
       (7.6-9.2%) for persistent signals. Fixed-b critical values are the
       owner exercise. Driscoll-Kraay (Mincer-Zarnowitz) size is not yet
       simulated.
   - `requests` is imported by the FRED adapters but not pinned in
     `requirements.txt` (dependency gate).
   - `run_rebalanced_backtest` executes at the same close; add a matching
     execution lag before portfolio-level model metrics (model ticket).
   - `build_model_dataset` still includes benchmark index rows
     (DOW_JONES, NASDAQ_COMPOSITE, SP500); validation excludes them via
     `restrict_to_symbols`. Removing them upstream is a data-contract
     change (owner-gated).
   - Legacy `split_temporal_dataset` and `select_training_rows_asof` are
     kept with warning docstrings: no purge, and intraday cutoffs could
     admit unmatured labels. Nothing outside tests calls them. Two gaps:
     - there is no runtime guard (a `DeprecationWarning` would be
       non-breaking);
     - the split's defaults (validation 2025, test 2026) lie entirely
       inside the lockbox.

     Deprecating them changes contracts and tests.
   - SNDK has no rows dated before the lockbox. In holdout folds its first
     trainable label matures in mid-April 2025, so early holdout folds
     score it with no SNDK training history.
   - Audit gaps from the pre-merge review (2026-10-08). The T2 run record
     closes most of them (merged in PR #43):
     - git SHA, dirty flag and diff hash;
     - estimator spec, parameters, seeds and variant ids;
     - package versions and platform;
     - fold-table, prediction and output hashes;
     - a writer and reader with manifest hashes;
     - an append-only JSONL run ledger.

     Fitted transform parameters are stored since T4 (`coefficients`).
     Still open:
     - raw-data retrieval times (the downloaders do not record them);
     - a holdout-use record kept in Git. The ledger lives under ignored
       `reports/`, so it can be deleted; T6 needs a Git-tracked record
       (owner decision).
   - T2 follow-ups (`docs/research/model_diagnostics.md`):
     - `validate_feature_columns` does not reject `control_*` columns; only
       `ModelSpec` refuses them for candidates (contract change, T4);
     - a sharper canary check (IC in the first `horizon` sessions of each
       window);
     - a rank-based hit rate;
     - an offline test for `scripts/run_null_diagnostics.py`.
   - T3 findings for T2 (`docs/research/diagnostics_dashboard.md`):
     - fixed in PR #45 (schema 1.1): the checks-summary defect, curve
       interval method, decision method, the unsafe reference's role and
       output kind;
     - remaining additive gaps: per-fold IC reasons, constant-sign hit-rate
       and score-decile statuses, MZ covariance, per-date series,
       `write_run(extra_record)` key collisions, purge-band dates, the PSI
       floor, degenerate cross-sectional R² for date-constant forecasts, and
       the decile interval estimand.
   - T4 follow-ups (`docs/research/linear_models.md`, "Mandatory before
     large-universe validation" and "Warnings and limitations"):
     - **mandatory before large-universe validation:** variable-universe
       complete cross-sections and delistings (point-in-time membership
       independent of feature rows, delisting returns, reinvestment rule,
       per-date reason codes; label/availability semantics, owner-gated);
       row-weighted fitting versus date-weighted tuning (pre-register one
       rule; formula change, owner-gated);
     - owner decisions recorded, not implemented: within-date demeaned or
       ranked features (the date-common forecast part only adds loss under
       the excess label); a hit-rate zero tolerance (constant forecasts
       score 1e-19 noise signs); zero-filled or coverage rank-IC rows;
       Elastic Net's L2 parameterization (not scale-free; duplicated Lasso
       here); a one-standard-error rule with stored inner standard errors;
       at least 199 permutation draws (Lasso's null is centred at -0.03);
       Romano-Wolf or SPA across candidates; beta-neutral and cap-weighted
       benchmarks (T5);
     - the size of the Clark-West and MSE-difference tests is not simulated
       (T2's simulation covers rank IC only);
     - Holm correction and a deflated Sharpe from the ledger's variant
       count are not built;
     - `validate_feature_columns` still accepts `control_*` columns
       (contract change);
     - per-fold IC rows say `undefined` where per-date rows say
       `constant_prediction` (the T3 per-fold reason gap);
     - the constant-label guard in `LinearForecast.fit` is rarely reached
       (lambda_max is ~1e-34, not 0, for a constant label);
     - model names outside the fixed palettes still hash into Okabe-Ito.
   - T3.1 follow-ups (reviews of 2026-10-10): Plotly charts have no text
     alternative; some figure greys and method colours are still literals
     outside the tokens; Streamlit's own widget animations do not follow
     the dashboard's reduced-motion rule; `.claude/launch.json` (a local
     preview launch config, untracked) is in the guardrail area and was not
     committed. Closed by T4: candidate colours, Clark-West for nested
     pairs, the evaluation as-of vs `score_available_at` cross-check.
   - Dashboard dependencies: Streamlit 1.64 caps pyarrow below 26, tying
     the data layer's pyarrow to Streamlit releases; its web stack
     (starlette, uvicorn) is unpinned. The test suite runs AppTest twice
     per gate run; AppTest timeouts use wall-clock time, so a machine
     sleeping mid-run can fail them.
   - Labels are computed after `build_model_dataset` drops rows with
     missing required features. An interior gap would stretch a label's
     span. Not present today; purging stays correct because it uses the
     recorded `target_end_date`.
   - Macro `available_at` is date-level only (no time of day stored), so a
     row dated d includes releases published after that day's close. Times
     verified 2026-10-08 against Federal Reserve sources:
     - H.15 (fed funds) at 4:15 p.m. ET;
     - weekly H.6 (M2) on Thursdays at 4:30 p.m. until 2021-02-11.

     This is harmless for ML labels (entry at the close of d+1). It is still
     unsafe for `entry_lag=0` labels and the same-close backtest if they are
     combined with macro. Macro values are vintage-correct: ALFRED's full
     real-time range, with each revision kept as its own row.
7. Workflow follow-ups:
   - GitHub branch protection on `main` is **not enabled**: verified
     2026-10-13 with the GitHub API ("Branch not protected", HTTP 404). The
     local command guard is the only mechanical control against direct
     pushes and merges.
   - The `Edit(...)` ask rules (root-anchored, e.g. `Edit(/.github/**)`)
     did not prompt for guardrail files edited inside session worktrees.
     Observed 2026-10-14 for `.github/workflows/health.yml` and a local
     `.claude/launch.json`. The cause is unconfirmed: either path anchoring
     or the permission mode. A fix is proposed as a task (a guardrail
     change, owner-gated). Until then, sessions quote the owner's words for
     every guardrail edit.
   - Local branches left by reviews and wave 1 (`review-pr60` to
     `review-pr72`, `wave1-base`, `prospective-base`,
     `lead/pr71-owner-decisions`, merged `fix/*` branches). Deleting them
     needs owner approval.
   - `numpy` and `pandas` are unpinned; tests use the legacy `RandomState`
     stream and canonical fingerprints so results do not depend on them.
   - `.gitignore`'s `.venv/` pattern matches only a directory, so a
     symlinked `.venv` in a session worktree shows as untracked and could be
     committed by `git add -A`. A `.venv` pattern would cover both. (The
     `*.log` line was fixed in PR #48.)
8. P1-1a follow-ups (for P1-1b, the first real-data kind):
   - its own provenance check (input hashes against stored raw inputs)
     and the input-hash entry format; it must not use synthetic
     regeneration;
   - a per-kind minimum contract version (a 1.0 manifest must never carry
     a later kind);
   - the production availability rule (OD9) and the OD8 group A formulas;
   - extract a shared staging helper if a third writer appears (P1-1a
     copies about 20 lines of the T2 pattern);
   - published directories have mode 0o700 and files are not fsynced; a
     torn write fails the reader's hash checks.
9. Tiingo, F1 and prospective-track follow-ups (owner decisions where
   marked):
   - **Before the first prospective `--store`** (owner;
     `docs/research/prospective_data_path.md` sections 2 and 7):
     - confirm the window reading 2026-07-16 to 2026-10-14 (64 closes, 63
       returns);
     - decide the VTI gate: VTI has one ex-date in the window. Either
       every served distribution must match an anchor (at least one), or
       wait for its December ex-date;
     - check the 2025-2027 holidays, which were entered from recall,
       against NYSE's published page.
   - Distribution anchors are filled by hand into a private, git-ignored
     file (PR #73). The rows the research agents gathered on 2026-10-14
     are kept privately under `data/raw/issuer_sources/` and are not
     anchors.
   - A precision-aware anchor match (round half-even at the issuer's
     reported decimals; the current rule is 1e-6) is proposed and not
     approved.
   - F1's `AmountBasis` has no "unknown" value, so an unverified
     distribution basis cannot be stored honestly. The adapter stores
     distributions only after the spot check. Adding "unknown" is a
     contract change, owner-gated.
   - Vendor-adjusted reconciliation needs an isolated, owner-approved
     procedure (proposed in `docs/research/f1_fund_data.md`).
   - A primary-source close anchor, to detect a uniform rescaling of
     closes, is not built.
   - The ETF plug-in interval under-covers by design: development
     fold-block coverage was 0.753 at a nominal 80%. The prospective
     forecasters use pre-registered split-conformal bands instead (PR #74).
10. Stock Lab (`docs/research/stock_probability_lab.md`):
   - **Owner decisions A1-A6 were recorded on 2026-10-16** (section 7 of the
     doc; quoted under "Owner decisions"). The watchlist is AMD, MRVL, MU
     and NVDA (owner, 2026-10-17). Still pending:
     - the authorization of the read-only audit
       `scripts/stock_lab_fetch.py --authorized` (no `--store`), after the
       owner sets `TIINGO_API_KEY`;
     - then a separate storage approval (`--store`);
     - the confirmation of the bull-trap f;
     - A7, the final confirmation of the rule
       (`docs/research/stock_lab_a7_decision.md`), before the A3
       evaluation selects;
     - A4 activation, after validation, a frozen registered model, verified
       decision timestamps and a dry run.
   - To build under A2 and A6:
     - the separately controlled inference store for protected-period
       observations;
     - its frozen-model reader;
     - an isolated descriptive reader for the dashboard.

     The daily job's real update and real forecasts come after these.
     Until then the job records itself as skipped (read as "not activated",
     #87), and the store refuses real forecasts.
   - Done since 2026-10-16:
     - the fold-block t, simulated test size and stable-band sensitivity
       (#83);
     - dark mode for the Lab (#82);
     - backdating refusal and truncation detection in the forecast record
       (#87).

     Per-stock B2 tilts remain a statistical follow-up.
11. Smaller items: lineage lacks current-side tie lists; as-of change log
   trusts `period_end`; mixed availability rules at one trigger instant;
   FY consistency checks (Q1+Q2+Q3+Q4 = FY); duplicate flow logic in
   `build_edgar_quarter`; unused `edgar_history._infer_fiscal_quarter`;
   lineage lacks numeric derivation inputs.


## Next steps

1. **First prospective forecast** (owner steps in the owner's terminal;
   `docs/research/prospective_data_path.md` section 6,
   `docs/research/prospective_registration.md`):
   1. Owner: the three decisions in open item 9.
   2. Owner: the read-only pre-2025 audit:
      `scripts/audit_fund_coverage.py --authorized --funds VTI AGG SHV
      --start 2005-01-01 --end-exclusive 2025-01-01`.
   3. Owner: fill the anchors file by hand, then approve storing the
      pre-2025 batch.
   4. Lead: the zero-volatility development check
      (`scripts/check_zero_vol_floor.py check`), reported to the owner;
      then the owner decides the zero-volatility rule and the section 13
      registration decisions.
   5. Lead (engineering; breaking contract changes need owner approval,
      given 2026-10-18 for the three listed under "Owner decisions"): the
      section 12 activation prerequisites:
      - the ledger's registration entry and record file;
      - cadence-aware Model Health;
      - the zero-volatility rule as a registered config key;
      - the checkpoint analysis.
   6. Lead: train and freeze the registered forecasters on real data up
      to 2024; owner: merge the registration PR (its merge time is the
      registration time).
   7. Owner: approve the prospective collection.
   8. Lead: record forecast #1 at the first eligible close, labelled
      "unvalidated, no demonstrated skill", and score each horizon as its
      outcome matures. Sessions 1-20 are the registered trial.
2. **T5a:** D1-D8 approved 2026-10-15; the engine is merged (PR #76).
   Remaining (owner-gated): D9 and the D5 clause on learned-policy
   constraint sets, then PPO on the approved transition and the first
   registered comparison table across the baselines, the optimizer,
   simple RL and PPO.
3. **Stock Lab** (open item 10; watchlist AMD, MRVL, MU, NVDA):
   1. Owner: set `TIINGO_API_KEY` locally and authorize the read-only audit
      (`scripts/stock_lab_fetch.py --authorized`, no `--store`;
      5 requests).
   2. Lead: review the audit output for corporate actions, coverage,
      missing sessions, anomalies and suitability. Owner: approve storage
      separately.
   3. Owner: confirm A7. Lead: implement it, then the A3 exploratory
      evaluation, then freeze and register the first model.
   4. Lead: the A2 inference store and the A6 descriptive reader (both
      buildable now, without data).
   5. Owner: confirm f and approve A4 activation after a dry run.
   6. Lead: the first prospective five-session forecast, then daily
      updates, scoring and monitoring.
4. **Model Health schedule:** a dedicated, clean monitoring clone (owner
   preference, 2026-10-17). The dry run of `scripts/setup_health_schedule.py`
   comes first, and the owner installs the schedule. Not installed.
5. **Groups C and D:** each group is implemented only after its approval,
   in its own reviewed PR (`docs/research/group_c_d_decision_index.md`).
6. **Small engineering follow-ups:**
   - the guardrail worktree fix (owner-gated);
   - review-branch cleanup (owner approval).
7. FE-0 and FE-1 (approved 2026-10-13 as preparatory work only), and
   RES-1a (the zero-purchase proof of concept) when the owner approves it;
   RES-1 stays cost-conscious and never blocks the ETF Portfolio Lab.
8. ML roadmap (owner request 2026-10-08: diagnostics and visualization are
   part of the roadmap, not an afterthought). Tickets T2-T6 are in
   `docs/research/diagnostics_and_visualization_plan.md`:
   - T2: diagnostics layer, run outputs and ledger, null models (merged,
     PR #43);
   - T3: Plotly and Streamlit dashboard, run on the null models (merged,
     PR #44); the T2 reporting fix (PR #45) and the T3.1 workspace (PR #46)
     are merged;
   - T4: Ridge, Lasso, and Elastic Net, with coefficient stability
     (merged, PR #47);
   - T5: portfolio readiness;
   - T6: holdout evaluation, once per pre-registered model.

   Owner decisions taken 2026-10-10: the excess label, pinned
   scikit-learn, and the per-symbol mean is a reported control, not a
   gate. 2026-10-11: the estimator core was delegated.
9. Owner decision from open item 7: protect `main`. Verified with the
   GitHub API on 2026-10-18 UTC:
   - Classic branch protection is absent.
   - The ruleset "Protect main - PR and CI required" (active, created
     2026-10-17) has an empty branch include list, so it targets no branch.
   - It lists no required status checks.

   Owner: add `main` (the default branch) as its target and the CI check
   `test` as a required status check.
10. U and C tracks, in parallel with T2-T4 (see "Sequencing against T4" in
   `docs/research/universe_and_market_context.md`). Owner decisions:
   - product scope (broad large caps vs semiconductors only);
   - the survivorship-free data source (RES-1a first; no institutional
     access);
   - `signal_cutoff_at`;
   - calendar and vendor dependencies;
   - compliance exclusions (NS-CMIC).

   T4 runs on the current universe count as engineering exercises only.
   Results are treated as product evidence only on U1 (planning target
   about 100 point-in-time large caps from 2010, with the MDE re-estimated
   on that panel), with the V1 context layer, the T2 inference checks, and
   the T5 costs.
11. Documented follow-ups (not implemented), in
   `docs/research/universe_and_market_context.md`:
   - evaluate a two-layer architecture: a cross-sectional alpha model plus
     a market/regime model for risk and cash exposure;
   - verify the DGS3MO quote-to-period-return conversion used in Sharpe and
     excess returns;
   - gold, silver, and copper as V2 candidates through documented proxies
     (GLD, SLV, CPER or licensed COMEX futures), one representation per
     factor;
   - reserve "rejected" for licensing or data-contract constraints;
     low-priority series (WTI, silver, ETH, gold/silver) are deferred
     candidates;
   - label HYG, LQD, and IEF measures as credit-risk proxies, never as
     credit spreads.
