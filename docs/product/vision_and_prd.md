# Product vision and PRD

Status (2026-10-11): **approved direction; details under review.** On 2026-10-11 the owner approved OD1, OD2, OD4, OD5, OD6 and OD7 (section 13). The other decisions (OD3, OD8–OD17) remain recommendations until approved at their tickets. Companion documents:
- `architecture.md`
- `roadmap.md`
- `ux_blueprint.md`
- `risk_register.md`

## 1. Summary

Stock Allocation Agent becomes a **research-grade portfolio lab**: a place to discover assets, understand risk, build a portfolio under one's own constraints, and compare it with transparent model portfolios. Every number says what kind of evidence it is.

**Shortest credible path.**
1. **Rules first.** Ship a rules-only product (the MVP, "Portfolio Lab") that makes no forecast claims. It is useful because it is honest about risk, concentration and history, and it carries an evidence label on every number.
2. **Earn forecasts in parallel.** The research track earns the right to show forecasts one claim at a time: rank, then expected excess return, then distributions, then absolute returns. Each step needs a point-in-time universe, a pre-registered evaluation, the holdout and prospective tracking.

The product never gets ahead of the evidence. The rules-only MVP is the first stage of a path toward an **investment-ready** system (section 15), not a permanent limit. Every later capability (forecasts, uncertainty and model-driven allocation) has an explicit evidence gate (section 11).

## 2. Vision and positioning

**Mission.** Help a person decide what allocation is reasonable for them and *why*, by showing:
- what is known;
- what is estimated, and how uncertain it is;
- what cannot be known;
- each with its evidence.

**Differentiation.** Most screeners optimize for excitement: rankings, price targets, scores. This product optimizes for **justified trust**:
1. **Explainable decisions.** Every allocation shows its rule, inputs, binding constraints and tradeoffs.
2. **Honest uncertainty.** Ranges before points. Explicit "unavailable" and "insufficient confidence" states. No probabilities that have not been calibrated.
3. **Historically rigorous validation.**
   - Point-in-time data and universes.
   - Purged temporal validation.
   - A sealed holdout and pre-registration.
   - A ledger of every variant tried.
4. **A calm, premium experience.** Restrained design, strong typography and progressive disclosure that never hides a caveat.

**What it is not** (now):
- not a stock-picking or signal service;
- not a broker;
- not a robo-adviser;
- not personalized investment advice;
- not a source of validated return forecasts.

**Long-term objective: investment-ready operation.** A system the owner can rely on, first for their own capital and only much later, if they choose, as a public advisory product. Readiness has three parts:
- its allocations come from validated components;
- they are explainable;
- they run inside tested operational and risk controls.

**Model completion is not readiness.** A trained allocation policy is one component. For example, PPO is a candidate allocation-policy engine (`architecture.md`, section 3). Readiness is earned stage by stage through empirical, operational and risk-control requirements (section 15).

## 3. Where we are (verified 2026-10-11)

- **Built** (T4 is in PR #47, merge pending): a leakage-safe research framework.
  - Purged walk-forward validation (T1).
  - Dependence-aware diagnostics and inference (T2).
  - A read-only Streamlit research dashboard (T3, T3.1).
  - Ridge, Lasso and Elastic Net with nested purged selection on the equal-weight excess label (T4, PR #47).
- **Result:** on four hindsight-selected names, **no detectable out-of-sample skill**. The minimum detectable rank IC was 0.16–0.25, against plausible effects near 0.02–0.05.
- **Lasso's partial-coverage rank IC** (0.070 on 13 of 24 folds) is not evidence that it outperforms Ridge. On the same folds Ridge scores 0.068.
- **Missing:**
  - a security master, survivorship-free prices and membership history;
  - a trading calendar;
  - risk-free history before 2024;
  - Treasury, ETF and crypto data;
  - an optimizer or covariance model;
  - a forecast-to-weight rule (T5).
- **Backtests** execute at the same close, cash earns 0 in the momentum baseline, and the index benchmarks are price indices without dividends. All three are T5 items.

## 4. Users and jobs

| User | Job to be done | Mode |
|---|---|---|
| **The careful investor** (nontechnical) | "Show me a few sensible portfolios, what they hold, how badly they have fallen in the past, and what you cannot tell me" | Explore |
| **The hands-on allocator** | "I want these assets, at most 25% in any one, some cash. Show me the concentration, overlap, costs and worst historical outcomes, and the tradeoffs if I change the rule" | Build |
| **The researcher** (today, the owner) | "Show me the model version, data vintage, universe, method, validation results, uncertainty and limitations behind any number" | Research |

**Primary user for the next phases: the owner, locally, for personal research and education.** Anyone else using it changes the regulatory and licensing picture (section 9).

## 5. Product principles (non-negotiable)

1. **No claim beyond its evidence.**
   - No current model output is presented as a validated expected return, probability of gain, recommendation or portfolio forecast.
   - A forecast appears only at the claim level its model has reached (section 6).
2. **Point-in-time everywhere.**
   - Every published number has a knowledge time no later than its as-of.
   - Universes are point-in-time.
   - Historically selected lists are never silently mixed with investable universes.
3. **The lockbox stays sealed.**
   - The 2025+ holdout is never unlocked or repeatedly inspected to validate product concepts.
   - By default no product publication reads data dated from 2025-01-01. Those periods appear as "Sealed" until the owner fixes the holdout window at T6 pre-registration (decision OD5).
4. **The product follows declared rules, not the best score.**
   - The product never adopts whichever model scored best in development.
   - Product defaults come from declared rules (the simplest adequate model, or the baseline).
   - Variant counts are shown.
5. **Separate kinds of evidence never mix.**
   - Actual history, hypothetical backtests, paper tracking and forward forecasts are separate series with separate badges.
   - Inception dates are declared in advance. Theme constituents are never chosen with hindsight.
6. **First-class states.** Empty, pending, unavailable, insufficient confidence, stale, partial and sealed are designed states, never blank charts or zeros.
7. **Rules and constraints, not return-seeking.**
   - Construction shows "the weights your rule and constraints produce, with their historical risk".
   - It never says "maximize expected return", and never "lowest risk": once cash is allowed, minimum variance just means cash up to its cap.

## 6. Evidence model and validation ladder

**Two labels on every output.**
- **Evidence status** of each publication (`architecture.md`, section 4): descriptive, rules illustration, research only, development evidence, holdout-checked, prospectively confirmed.
- **Claim level** for return forecasts: rank, bucket, expected excess return, expected absolute return, distribution.

**Validation ladder.** Two tracks: location (L) and dispersion (D).
- Each rung is validated separately for each (universe, horizon) cell. A model validated on one universe is "not validated" on another.
- Every "greater than zero" names its test, its family and its α in a pre-registration.

| Rung | Product claim | Gate (estimand and test) | When it fails, the product shows |
|---|---|---|---|
| L0 | Rule rank ("highest 12-month momentum") | The rule's inputs are point-in-time; no forecast claim | Always "rule, not forecast" |
| L1 | Model order ("ranked higher by the model") | Coverage-weighted mean rank IC (constant dates count as zero) > 0, one-sided fold-block t after multiplicity. Paired superiority over the best pre-registered rule, or the rule is shown instead | "No validated model view"; the rule is shown |
| L2 | Attractiveness bucket ("top fifth") | A monotonicity test across buckets (Patton-Timmermann) and a top-minus-bottom spread net of T5 costs | "No differentiated view" |
| L3 | Expected excess return (%) | Within-date Mincer-Zarnowitz slope with a lower confidence bound > 0 and a pre-registered maximum interval width. The product shows the **MZ-recalibrated** forecast, never the raw one. Cross-sectional OOS R² > 0; date-normalized squared-error advantage > 0 | Bucket only |
| L4 | Absolute expected return (%) | A validated market layer (Campbell-Thompson R², Clark-West vs the expanding mean), **and** the *composed* forecast tested as a whole. Composition is the exact sum by default; the beta adjustment applies only under its documented approximation, with the market-exposure diagnostic (`architecture.md`, section 5) | "Unavailable: no market forecast". At most a labelled long-run assumption |
| D1 | Ranges, quantiles, expected shortfall | Coverage at 50/80/90% within tolerance, overall and by volatility, sector and **ex-ante-defined** regime; pinball/CRPS skill vs a volatility-scaled historical baseline; PIT uniformity; exceedance tests (Kupiec, Christoffersen). Expected shortfall additionally needs a joint VaR/ES test (Fissler-Ziegel scoring or Acerbi-Szekely) | "Historical range, uncalibrated" |
| D2 | P(outperform), then P(gain) | Brier skill vs the base rate with a date-clustered standard error; reliability and resolution. P(gain) also needs L4 and D1 on *absolute* returns | Hidden |

**Evidence required for every rung above L0:**
- a point-in-time universe (U1) on survivorship-free prices;
- the MDE pre-registered from the real panel;
- fold-block t as the decision method, its size re-simulated per horizon;
- nulls (at least 199 permutation draws) and the selection control;
- survivorship sensitivity;
- a Git-tracked record of every scoring of the real panel.

**From development evidence to the product.**
1. **Development evidence** appears in Research mode only.
2. **T6, the holdout, runs once,** for everything pre-registered by then. The status it earns is **holdout-checked**: a "no collapse" check, not a confirmation, with no promotion power.
   - With about 100 names the holdout's one-sided MDE is about 0.05 in rank IC (standard error about 0.020).
   - Plausible effects are 0.02–0.05.
   - The holdout cannot confirm a small edge.
3. **Prospectively confirmed** is the only status that unlocks Explore and Build, at the claim level reached. It requires the pre-registered confirmatory test of section 15: one-sided, over non-overlapping blocks, with α-spending at fixed look dates. Its sample is the holdout (T6 is look 0, for artifacts registered before it) plus the dates logged after registration.
   - **Power arithmetic, sampling error only, about 100 names, verified by the review:**
     - Confirming an IC of 0.03 (one-sided α = 0.025, power 0.8) needs about 1,170 dates never used for selection (about 4.6 years) for a signal whose overlap factor is about 13.4, and about 1,760 dates (about 7 years) for a persistent signal.
     - Twelve prospective months detect a *collapse* with only about 31% power at a true IC of 0.03, and 61% at 0.05.
   - **Model views are therefore years away. Risk forecasts (rung D1) come first.**
4. **Designer knowledge.** This repository began on 2026-08-08, so every design so far was made knowing what happened in 2025. The holdout guards against data-driven selection, not against that knowledge. Designs dated after 2025-01-01 carry a `designed_after_holdout_start` flag and a list of the 2025+ facts known. This is one more reason prospective evidence is the only clean confirmation.

**Never sort by a score that has not earned it.**
- A forecast spread below a pre-registered tolerance shows "no view". This catches near-constant forecasts: T4 scored constant forecasts of about ±4e-19 by their "signs".
- Estimates are reported selection-free (pre-registered), not as the best of many, to avoid the winner's curse.

## 7. Capability map

| Area | NOW (exists) | RULES (no forecast) | VALIDATED (after the ladder) | LONG-TERM |
|---|---|---|---|---|
| **Universe** | Four research names and current listings (survivorship-biased) | User watchlists, labelled "list you chose"; theme lenses dated "defined on ‹date›" | — | U1 point-in-time large caps, then S&P 500 membership (licensed); objective AI/semiconductor eligibility (SIC 3674 plus rules) |
| **Opportunity discovery** | Descriptive features in research runs | Characteristic screens (trailing return, volatility, drawdown, beta, liquidity, fundamentals), sorted by name or a chosen characteristic, never by "attractiveness" | L1/L2 relative ranks; then L3 expected excess return; then D2 P(outperform) | L4 absolute expected return; P(gain); expected downside per asset |
| **Horizons** | 20 sessions (registered label) | Lookback windows of 5, 20, 30 and 60 sessions for descriptors | 5, 20, 30 and 60 all remain possible user-facing horizons. Implementation priority: 20, then 60 and 5 (5 after costs), then 30 after a pre-registered check of its incremental value. Each horizon is its own validated cell; an unvalidated horizon shows "not validated for this horizon" | Calendar-time horizons for 24/7 assets |
| **Portfolio Builder** | EW, inverse volatility, momentum weight functions; backtest engine (same-close) | **MVP:**<br>• fixed weights, EW, inverse volatility over risky assets<br>• per-asset caps<br>• cash as a floor or fixed weight<br>• exposures and concentration<br>• historical risk and stress scenarios<br>• cost estimates<br>**Later:**<br>• sector and theme caps with fund look-through<br>• volatility target<br>• equal risk contribution and capped minimum variance (optimizer dependency) | Risk forecasts (volatility, covariance) after rung D1; alpha as a bounded, cash-neutral tilt after L3 | Calibrated forecast distributions of a whole portfolio; tax awareness |
| **Model portfolios** | — | Pre-registered reference constructions on broad funds and cash, named by construction; hypothetical backtests over one registered window ending 2024-12-31; paper tracking once OD16 is approved and the prospective pipeline (G2) is active (stage S2) | Model-driven variants as separate registered portfolios | Calibrated forecast distributions; realized track records |
| **Research transparency** | Run records, manifests, checks, the dashboard's evidence pages | Evidence ledger page; publication lineage; limitation codes | — | Public methodology notes (after counsel) |
| **Multi-asset** | DGS3MO from 2024 (evaluation only) | Treasury ETFs and cash via total returns; bond risk descriptors (duration) | Treasury and cash return forecasts (own ladder) | Crypto (24/7 calendar, licensed data, counsel); international |
| **Monitoring** | Run ledger (ignored by Git) | Freshness and stale states | Calibration and drift monitors with state triggers | Multi-year prospective record |

## 8. MVP: Portfolio Lab (rules-only)

**The first MVP, not a permanent restriction.** "No forecasts" describes this first release only. Forecast, uncertainty and model-driven allocation features are planned; each appears when its evidence gate in section 11 is met.

**Goal.** A local, single-user product, as of the end of the development period (2024-12-31) under the default lockbox boundary (OD5). It lets the owner do three things, with every number carrying its evidence badge and knowledge time, and with no forecast anywhere:
- compare pre-registered model portfolios;
- build their own portfolio under explicit constraints;
- understand concentration, overlap, costs and historical risk.

**In scope.**
1. **Publication pipeline foundation.** The envelope, writer, fail-closed reader, evidence gate, sealed-period refusal and import boundary (`architecture.md`, sections 3, 4 and 8).
2. **T5 corrections**, required before any simulated performance is shown:
   - next-close execution (lag 1);
   - a cash return from bill yields or a T-bill fund, with the DGS3MO backfill and convention check;
   - total-return benchmarks;
   - a reviewed cost model.

   These are financial formulas (owner gate).
3. **Model portfolio registry and comparison.**
   - Three reference constructions on broad funds and cash (section 10).
   - Each has pre-registered rules, a benchmark and a comparison window, committed to Git before any result is computed.
   - Hypothetical backtests end 2024-12-31; 2025 onward is "Sealed".
4. **Builder.**
   - Assets from the registry and a user watchlist (labelled "list you chose").
   - **Rules:** fixed weights; equal weight or inverse volatility over the risky assets; a per-asset cap; cash as a floor or a fixed weight.
   - **It shows:** exposures, concentration, historical volatility and drawdowns, stress scenarios, turnover and costs.
   - **A loss limit the user sets is a historical check, never a target.** For example: "this mix's worst historical decline was ‹$x›, above your limit". It is met in-sample by construction.
   - **Interaction:** discrete options and a "configurations viewed" counter. Allocation math is pure functions over published inputs.
   - **Deferred:** sector and theme caps with fund look-through, volatility targeting, equal risk contribution, minimum variance.
5. **Watchlist explorer.** Descriptive characteristics only, sorted by name by default, labelled "current-listing, not a point-in-time universe".
6. **Research mode** gains evidence pills and an Evidence ledger page.
7. **UX states:** every state in `ux_blueprint.md`, section 6, implemented and tested.

**Out of scope for the MVP.**
- Any return forecast, probability of gain or expected downside.
- Data dated from 2025-01-01 (until OD5 changes).
- Rankings by model.
- Crypto.
- Treasury forecasts.
- S&P 500 membership.
- Hosting, accounts and saved user data.
- Execution or brokerage links.
- Risk questionnaires.
- Alerts.

**Acceptance criteria.**
- Contract and integrity tests:
  - every rendered number has an evidence badge and a knowledge time;
  - no row dated in the sealed period renders;
  - no return-forecast field can render in Explore or Build;
  - unknown statuses and newer schema versions are refused;
  - `app/` imports no model, writer or raw-data code.
- Every first-class state has a test that renders it.
- Model portfolio definitions, benchmarks and inception rules are committed (pre-registration) before their first backtest run, with the ledger's variant count shown.
- The T5 formula review is approved by the owner. Backtests execute at the next close, net of costs, against total-return benchmarks, labelled "before taxes", with intervals on headline metrics.
- `python3 scripts/verify.py` passes, and AppTest covers every MVP page.
- **Comprehension by design.** Each question a reader must be able to answer maps to an on-screen element that AppTest asserts on every relevant page:

  | Question | Element |
  |---|---|
  | "Is this a forecast?" | Evidence badge and status pill |
  | "What was the worst historical decline in dollars?" | Worst-decline card on the reference amount |
  | "Why is this portfolio concentrated?" | Concentration callout |
  | "What is sealed and why?" | Sealed state on the time axis |
  | "What can't this tell you?" | "What this cannot tell you" panel |

  Testing with other people waits for the counsel gate (section 9) and uses synthetic data.

## 9. Regulatory boundary and data licensing

This is a design boundary, not legal advice. It must be checked by qualified counsel before anyone other than the owner uses the product.

- **Personalized portfolio construction is the hallmark of investment advice.** Turning a person's capital, horizon and risk preference into an allocation is tailored by design. It likely falls outside the "publisher's exclusion" (impersonal, bona fide, general and regular circulation; *Lowe v. SEC*). Any compensation (subscriptions, ads, referrals) changes the analysis. Hypothetical performance shown by an adviser falls under the SEC Marketing Rule (206(4)-1).
- **Safe-side design for the research and education tool:**
  - Builder outputs are "illustrative weights from the rule you chose". Never "recommended", "target", "buy" or "sell".
  - No risk questionnaire that maps a person to a portfolio. A risk limit is a constraint the user sets.
  - No default "recommended" portfolio; model portfolios are reference constructions.
  - Hypothetical results are labelled, net of costs, with assumptions, and with no-skill and survivorship caveats.
  - No execution, brokerage links, fees, ads or affiliate revenue.
  - Session-only inputs; nothing about the user stored or logged; no third-party analytics.
- **Counsel review is required before:**
  - anyone other than the owner uses it;
  - collecting user-specific inputs from others;
  - hosting;
  - monetization;
  - publishing model-portfolio performance;
  - alerts or paper tracking for users;
  - "model-driven allocations" shown to anyone other than the owner;
  - crypto alongside securities;
  - any brokerage relationship.

**Data licensing by stage:**

| Stage | Allowed |
|---|---|
| Personal research (now) | yfinance, FRED/ALFRED and EDGAR within their terms. Personal, non-commercial use |
| Private demo | No raw or vendor-derived data shown or downloaded; public-domain or synthetic series only; CSV downloads off |
| Commercial | Written licences covering display, redistribution and derived data; exchange fees; index (S&P) and classification (GICS) licences; attribution. Academic data (CRSP/WRDS) cannot be used, and models fitted on it may need re-fitting on commercially licensed data |

## 10. Model portfolios

Each model portfolio is a **registry entry** committed to Git before any result is computed. It records:
- the rule, assets and rebalancing;
- the cost model;
- the benchmark;
- the comparison window;
- the registration date;
- the alternatives that were considered.

**Rules for every registry entry.**
- **One comparison window for all portfolios**, set by an asset-class rule, not by the funds chosen. For example: "from the first month-end at which every asset class has a registered fund or a registered labelled proxy, plus a 12-month warm-up, to 2024-12-31".
  - Instrument choice must not pick the window. A T-bill fund launched in 2007 includes the 2008 crisis; one launched in 2009 does not.
  - Pre-registered subperiods, such as 2008 and 2022, are reported beside the full window.
  - Prices start in 2015 today, so any earlier window needs the data extension F1 and the DGS3MO backfill.
- **The hypothetical backtest and paper tracking stay separate.**
  - The backtest ends 2024-12-31 (OD5).
  - Paper tracking can start only once OD16 is approved and the prospective pipeline (G2) is active (stage S2). It is never spliced onto the backtest.
- **References are not benchmarks of themselves.**
  - 100% equity, 60/40 and cash are reference portfolios.
  - A rebalanced 60/40 is compared with a never-rebalanced 60/40.
  - A bond-heavy mix is compared with cash plus a static mix of the same equity share.
- **Names describe construction.** The intent is a subtitle.

| Brief's name | Proposed name (construction) | Rule sketch (examples, not chosen) | Compared with | MVP? |
|---|---|---|---|---|
| (reference) | **All equity, global** or **All equity, US only (chosen 2026)** | 100% of one broad equity fund. A US-only choice made in 2026, after about 15 years of US outperformance, carries that label | Itself as the equity reference; the other equity choice | Yes |
| Balanced Growth | **Broad equity and bonds, 60/40, rebalanced quarterly** | 60% equity fund (as above), 40% aggregate-bond fund | The same 60/40 never rebalanced; 100% equity | Yes |
| Capital Preservation | **Bond-heavy, short duration** | For example 20% equity, 40% 1–3-year Treasuries, 40% T-bills (portfolio duration about 0.8 years, capped through instrument choice: there is no point-in-time history of fund durations); the equity share is pre-registered | Cash (T-bill fund); a static mix with the same equity share | Yes |
| AI/Semiconductor Growth | **Semiconductor sector sleeve** | A sector fund, or later a point-in-time SIC 3674 sleeve under rules R1–R8 | Broad equity | **Deferred (recommended).** The theme was chosen in 2026, after the sector's run. The sector funds did not keep one method: one was a fixed basket until about 2011 and another changed index around 2021 (both unverified). If shown, it gets no headline CAGR or Sharpe and carries the label "theme chosen in 2026" |
| Multi-Asset Opportunistic | **Multi-asset trend rule** (renamed: "opportunistic" implies market timing) | A rule from the literature, such as 12-month time-series momentum across equity, bonds and cash | 60/40 | Later (RULES), with its own registration |

- **Fund-level choices are a data decision (OD7)** made in the first portfolio ticket. Candidate long-history funds (inception dates unverified): VTI or VT for equity, AGG for aggregate bonds, SHY for short Treasuries, BIL for T-bills. Older index mutual funds can serve as labelled proxies.
- Results are always before taxes, with distributions reinvested under a stated rule.

## 11. Later capabilities and their evidence gates

"Not now" is not a permanent restriction, but not a promise either: some gates may never be met, absolute-return forecasting (L4) in particular. Each capability below unlocks when its gate is met. The gate combines a ladder rung (section 6) with a promotion stage (section 15). Until then the product shows the capability's explicit state, such as "not validated for this horizon", never a placeholder number.

| Capability | Evidence gate | Earliest place it appears |
|---|---|---|
| **Forecasts** | | |
| Relative model ranks and buckets (L1, L2) | U1 on survivorship-free data; the T4 re-run on U1 with the T4 follow-ups resolved; T5; T6 (holdout-checked) | Research mode after development evidence. Explore and Build only after **prospective confirmation**, which is years away for plausible effects (section 6) |
| Expected excess return (L3) | Rung L3 of section 6; the Mincer-Zarnowitz-recalibrated forecast is shown | As above |
| Absolute expected return (L4) | A validated market layer, plus a validated composition (`architecture.md`, section 5) | As above |
| **Uncertainty** | | |
| Risk forecasts: volatility, covariance (D1) | Development check first. Then, to reach Explore and Build, an **equivalence test** (two one-sided tests) showing coverage within a pre-registered margin on prospective data from the G2 pipeline. A non-rejection check is not evidence of calibration: its power is about 0.34 over ten years at h = 20 against a 90% band whose true coverage is 85% | Research after the development check; Explore and Build after the prospective equivalence test. Likely the first forecast to reach the product |
| Ranges, quantiles, expected shortfall (D1) | Coverage by ex-ante regime; pinball/CRPS skill; a joint VaR/ES test; the same prospective equivalence test | As risk forecasts |
| P(outperform) (D2) | Brier skill with a date-clustered standard error, on top of L1–L3 | With L3 |
| P(gain), expected downside per asset | L4, plus D1 and D2 on absolute returns | With L4 |
| **Allocation** | | |
| Rule-based allocations (fixed, equal weight, inverse volatility, caps, cash) | Registration and T5's formula review | MVP (Explore, Build) |
| Risk-based construction from trailing estimates (equal risk contribution, capped minimum variance, volatility target) | A declared rule on trailing historical estimates; T5's formula review; an optimizer dependency for the first two | A RULES capability after the MVP |
| The same constructions fed by risk *forecasts* | Validated risk forecasts (D1, prospective equivalence) | After P4-1 and its prospective test |
| Alpha-tilted allocations | L3, plus a tracking-error budget and an optimizer | After L3 is prospectively confirmed |
| Learned allocation policy (PPO candidate) | Section 15's policy requirements, then paper trading (stage S2) | Research, then paper trading, then (stage S3) the owner's own capital |
| **Other** | | |
| Opportunity screener over a point-in-time universe | U1: survivorship-free data, security master, membership, calendar (data-source decision) | After RES-1 |
| Treasury and cash return forecasts | Their own ladder; Treasury total-return data | Long-term |
| Crypto | 24/7 calendar contract, a licensed data source, counsel | Long-term |
| Multi-user, hosted product | Counsel, data licences, authentication, privacy program, architecture C on top of B's contracts (stage S4) | Only if the owner chooses |

## 12. Research follow-ups this plan depends on

These are already recorded with T4, and are repeated here because the product depends on them:
1. **Elastic Net's L2 term is not scale-free.** It chose Lasso's penalty in 24 of 24 folds, and its rank IC equals Lasso's to four decimals. Count it as one variant until reparameterized.
2. **The hit rate counts near-zero sign noise.** Constant forecasts of about ±4e-19 were scored by their signs. A zero band is needed.
3. **The date-common forecast component only adds loss under the excess label.** Within-date demeaning or ranking of features, or date-weighted fitting.
4. **Row-weighted fitting vs date-weighted tuning** once the universe size varies (`architecture.md`, section 7).
5. **Variable-universe complete cross-sections and delistings** before large-universe validation.
6. **Lasso's partial-coverage rank IC is not evidence** that it outperforms Ridge. Compare models on the same dates.
7. **New, from this review:**
   - Index benchmarks are price indices (dividends excluded) while strategies use total returns.
   - The momentum baseline earns 0 on cash.
   - Any future non-forward validation scheme needs a post-test exclusion of h + L + e − 1 rows: horizon, feature lookback and entry lag, 40 rows at h = L = 20 and e = 1, so the first eligible training row is 41 sessions after the window. L grows to 252 if 252-session z-score features arrive with context V1.
   - Variant counts from before the Git-tracked record are lower bounds ("≥ n"): T4's scratch runs were not recorded.

## 13. Owner decisions

The gate column cites `AGENTS.md`'s approval gates. The status column records the owner's answers of 2026-10-11 in this session; every other decision remains a recommendation until approved.

| # | Decision | Recommendation | Gate | Status |
|---|---|---|---|---|
| OD1 | Positioning | A personal research and education tool, local and single-user. Counsel before any other user, hosting or monetization | Product and legal | **Approved 2026-10-11**: "Rules-only lab, personal" |
| OD2 | Architecture | B, the publication pipeline, with a separate product entry point; Streamlit for the MVP | Major architecture | **Approved 2026-10-11**: "Pipeline; P1-1a, T5, F1 in parallel" |
| OD3 | Evidence model | Evidence status and claim level; the L/D ladder of section 6. Only `prospectively_confirmed` unlocks Explore and Build. `holdout_checked` has no promotion power. Development model output stays in Research | Product and research policy | Pending |
| OD4 | MVP scope | Rules-only Portfolio Lab (section 8); no forecasts | Product policy | **Approved 2026-10-11** (with OD1) |
| OD5 | Lockbox boundary for the product | **Default:** no product publication reads data dated from 2025-01-01, for any asset; every `as_of` is at most 2024-12-31. "Dated" means information content, including holdings snapshots and price adjustments. **Later:** T6 runs once for everything pre-registered by then. At its pre-registration the owner fixes the holdout window; after T6 that window becomes development data. Each entity's evidence class is development before its registration date and prospective after it, viewed only on pre-registered review dates | Point-in-time and lockbox policy | **Approved 2026-10-11**: "Strict: nothing after 2024" |
| OD6 | Model portfolios | The three reference constructions in section 10; the semiconductor sleeve deferred; the trend rule later; names by construction; one comparison window by asset-class rule | Product, data | **Approved 2026-10-11**: "Three references; semis deferred" |
| OD7 | MVP data | Broad fund total returns (raw closes and dated distributions) and the DGS3MO backfill, from the existing personal-use sources, ending 2024-12-31; a user watchlist | Data (universe and source) | **Approved 2026-10-11** (with OD6); the funds are chosen in the first portfolio ticket |
| OD8 | T5 formulas | Next-close execution; cash convention; total-return, fund-based benchmarks; cost model, including the initial build and a stated turnover convention; distribution reinvestment timing; calendar-time CAGR; intervals on headline metrics; the "gain" definition (for later) | Financial formulas | Pending |
| OD9 | Decision clock | `signal_cutoff_at` = 18:00 ET on session d as the product's `as_of` (open item 1). Until then, an interim date-level rule for publication v1: a daily close is known at the end of its session date | Availability semantics | Pending |
| OD10 | Horizons | 5, 20, 30 and 60 sessions remain possible user-facing horizons. Implementation priority: 20 first, then 60 and 5 (5 after costs), then 30 after a pre-registered check of its incremental value (its correlation with 20 is about 0.82). Dropping any horizon is an explicit product decision | Research and product policy | Pending |
| OD11 | Themes | Theme lists are dated current-exposure lenses, never historical strategies; "AI exposure" only as a point-in-time characteristic | Product policy | Pending |
| OD12 | Pre-registration record | A Git-tracked registry for model portfolios, every real-panel scoring, holdout use, forecast-log starts and review dates | Research integrity | Pending |
| OD13 | Survivorship-free data | CRSP via WRDS (if Georgia Tech access exists) or Sharadar, for U1; unchanged from the universe document | Data and licence | Pending |
| OD14 | Dependencies | A trading calendar (or an in-repo holiday file) when needed; an optimizer only when construction needs one | Dependencies | Pending |
| OD15 | Mission wording | Consider changing `AGENTS.md`'s "explainable portfolio recommendations" to "explainable portfolio analysis" until the product may recommend | Guardrail file | Pending |
| OD16 | Prospective pipeline (amends OD5's "Later" clause) | A separately governed pipeline for prospective paper trading and later deployment (`architecture.md`, section 9).<br>• Before its first registration, fix the holdout end H_end and record it in G1. The sealed holdout becomes [2025-01-01, H_end]; T6 evaluates it once and cannot extend it.<br>• Registered artifacts only, with a registration-hash check before every decision.<br>• An append-only decision log written before the execution cutoff, its hash committed to Git; no backfill.<br>• Separate storage and entry point.<br>• Review reports only through the pipeline, never the product, on pre-registered dates.<br>• Inputs from the sealed window are withheld from every view until T6.<br>The research and product cutoff is unchanged. **Alternative:** keep the pipeline dormant until T6's pre-registration | Point-in-time and lockbox policy; data; infrastructure (scheduler); secrets | Pending |
| OD17 | Promotion stages | The stages S0–S4 and their entry and exit requirements (section 15). Each promotion is a separate owner decision | Product, risk and legal | Pending |

## 14. Recommended first implementation ticket (after approval)

**P1-1a: Publication contract v1 (no UI).**

**Why first.**
- Every later product feature depends on this boundary.
- It needs no new dependency, data source, formula change, forecast or 2025+ data.
- It turns this PRD's safeguards into tested code before any product page exists.
- It is small enough for one pull request.

**Scope:**
- **`publication/`:**
  - the version 1 envelope (`architecture.md`, section 8: the v1 minimum);
  - row status and reason codes;
  - a write-once writer (staged, atomic, SHA-256 manifest written last), reusing the run-artifact pattern;
  - a fail-closed reader.
- **Evidence status v1:** `descriptive` and `rules_illustration` only. `claim_type` is derived from `kind`. The other statuses, claim levels and the promotion path arrive with the first publication that needs them.
- **Lockbox rule (OD5 default):**
  - the writer refuses any input dated on or after `MODEL_HOLDOUT_START`, imported from `model_validation/folds.py`, never a new literal;
  - the reader refuses any row or `as_of` dated on or after it.
- **Knowledge time:** the interim date-level rule (OD9), stated in the contract.
- **Tests on synthetic fixtures only.** No real data, no page.

**Negative tests:**
- `knowledge_time` after `as_of`;
- an input or row dated on or after `MODEL_HOLDOUT_START`;
- an unknown evidence status;
- a newer minor version;
- a missing or mismatched hash;
- a second write to the same publication id;
- a dirty-tree publication requested outside Research mode.

**Not in scope:** T2 contract changes, new data, new dependencies, any page.

**Done when:** the tests pass under `python3 scripts/verify.py`, `/project-verify` is READY, and a PR is opened.

**Then P1-1b: the first real publication and page.**
- **Content:** the actual total-return history of two or three long-lived broad funds, for example an equity fund, an aggregate-bond fund and a T-bill fund, through 2024-12-31.
  - Growth of ‹$10,000›, worst decline in dollars, and losing calendar years.
  - Status `descriptive`, badge "Actual".
- **Presentation:** one Explore page, after moving `theme.py` and `components.py` into a neutral `ui/` module.
- **No rebalancing decision is involved,** so it does not wait for T5. It needs OD7 and a small slice of F1.

## 15. Long-term objective and promotion stages

**From research to investment-ready operation.** Each stage has entry requirements in three families:
- empirical: statistical evidence;
- operational: pipelines, replay, change control;
- risk control: limits, kill switches, reconciliation.

Each promotion is an owner decision (OD17). The G1 registry records promotions as append-only events.

| Stage | What runs | Who is exposed |
|---|---|---|
| **S0 Research and development** (today) | Rules, models and policies evaluated on development data (to 2024-12-31) | No one; research only |
| **S1 Holdout-checked** | One T6 run, over [2025-01-01, H_end], for everything pre-registered by then (derived from the T6 record) | No one |
| **S2 Prospective paper trading** | Registered artifacts in the separately governed prospective pipeline (OD16); decisions logged before outcomes exist; no capital | No one; paper only |
| **S3 Controlled personal deployment** | The owner's own capital, capped and ramped, through the same pipeline, with manual approval of each rebalance at first | The owner only |
| **S4 Public advisory product** (only if chosen) | A regulated service for other people | Other people |

**Entry requirements, by stage:**

| Stage | Empirical | Operational | Risk control |
|---|---|---|---|
| S1 | Pre-registered in Git: claim, rung, horizon, universe, MDE, test family and α. Development evidence at that rung. One T6 run, no collapse | Reproducible from code SHA and data hashes; the ledger records every real-panel scoring | — |
| S2 | **Registration only.** S1 is not a gate. A model or policy also needs development evidence and a registered confirmatory plan (below). Artifacts registered after T6 enter S2 without S1 and are flagged "prospective only" | The prospective pipeline active (`architecture.md`, section 9): replayable decisions, freshness monitors, an incident log, a burn-in period fixed in advance | A constraint check on every decision (caps, cash floor, turnover) |
| S3 | Per artifact class:<br>• **model claim:** prospectively confirmed by its confirmatory test;<br>• **rules-only allocation:** exact replay of every logged decision; drift between live and later-revised data vintages within tolerance; at least a pre-registered number of rebalances (for example 12) over at least one year; paper turnover and quote-based cost estimates within tolerance of the simulation. There is no return test for rules: a Sharpe ratio's standard error is about 0.75 over two years;<br>• **policy:** see "Learned allocation policy" below | Execution plan (manual or broker; an owner decision), including whether market-on-close orders are feasible with manual approval; position and cash reconciliation against broker statements on every rebalance; change control; backups; an audit trail; an **override log**, because one owner is designer, validator and operator | **Order checks before trading:**<br>• an allow-list of securities<br>• notional and price collars<br>• no margin or shorting<br>• settled cash only<br>• idempotent orders<br>• whole-share rounding<br>**Capital:** a cap set by the owner and a ramp tied to review dates; the account type registered in advance (a taxable account brings wash-sale and short-term-gain issues, worse with frequent rebalancing of the same names).<br>**Losses:** an absolute loss budget beside a drawdown trigger relative to the registered reference portfolio, both calibrated to a stated false-trigger rate, with a re-entry rule; any drawdown overlay is part of the registered backtest.<br>**Consistency:** the same rebalance cadence and no-trade bands in simulation, paper and live; a tested kill switch.<br>**Scale-up:** realized implementation shortfall against the simulated next-close price within tolerance before capital increases |
| S4 | A multi-year prospective record at every claim level advertised. Independent model validation | A registered firm with a chief compliance officer (likely needed even with one owner); licensed data; hosting with authentication and TLS; a security program; a privacy program; incident response | A compliance program (adviser registration path, the Marketing Rule, record keeping); suitability; conflicts policy; firm-level risk limits. Counsel throughout (section 9) |

**Registration lineage and changes.**
- A change is anything that alters a logged output. It creates a new registration in the same lineage, which starts again at S2.
- Refits under the registered refit procedure and schedule, and refactors proven by byte-identical log replay, are not changes.
- A lineage has one α budget, fixed at its first registration.
- Abandoned or reverted registrations stay visible and count as failed tests.

**The confirmatory test.** Each registration records:
- the estimand θ (for a model, coverage-weighted rank IC or a ladder rung's statistic; for a policy, the net certainty-equivalent difference against its reference);
- H0: θ ≤ 0 against δ = the MDE, with one-sided α (for example 0.025) and power 0.8;
- a statistic over non-overlapping h-session blocks, since prospective data has no folds;
- information time measured in effective blocks;
- Lan-DeMets O'Brien-Fleming α-spending at fixed look dates. T6 is look 0 for artifacts registered before it.
- a non-binding futility rule (stop when conditional power is below 10%);
- a maximum length N_max with no extension. N_max is the "minimum prospective period"; a claim not confirmed by then is retired.

**Exit and rollback triggers**, pre-registered per artifact class:
- **Models:** a CUSUM on non-overlapping block means of per-date IC or calibration. Overlapping labels make per-date IC autocorrelated, so the threshold is calibrated by block-bootstrap simulation to a stated in-control average run length, and its false-alarm rate is recorded.
- **Allocations:** a drawdown or loss-budget breach; realized implementation shortfall beyond tolerance.
- **Any class:** stale inputs (the default is to **hold** existing positions, never to trade on stale prices); a constraint violation; an operational incident.

A trigger reverts the artifact to the previous stage, or holds the registered reference portfolio.

**Requirements specific to a learned allocation policy (the PPO candidate):**
- **Environment fidelity.** The simulator is the corrected backtest engine (T5), through the shared per-step function:
  - point-in-time state;
  - next-close execution;
  - costs and liquidity;
  - constraints enforced by projecting the action onto the feasible set.

  It assumes prices are exogenous (no market impact beyond the cost model) and that there is only one historical path. Both assumptions are stated.
- **State and training without leakage.**
  - Forecasts in the state are out-of-fold and point-in-time as published, never in-sample fitted values.
  - Observation and reward normalization statistics are fitted per training fold, frozen, stored in the artifact and never updated live.
  - Checkpoint selection and early stopping use inner folds only.
  - Episodes end at least h sessions before each test start.
  - Training uses development data only, with the same purged walk-forward design.
- **Reward.**
  - Pre-registered, net of costs and risk-aware: for example net return minus a variance or drawdown penalty. A drawdown penalty needs drawdown in the state.
  - Checked for reward hacking, for example earning reward through turnover patterns the cost model underprices.
  - Changing `environment/reward.py` is a formula gate.
- **Baselines**, on the same investable set, with an intersection-union test (the policy must beat each one):
  - the registered rule-based allocations;
  - a myopic constrained mean-variance optimizer with the same inputs and turnover penalty;
  - a simple rule tuned with the same tuning budget;
  - the same policy with no forecast inputs.

  The primary estimand is the paired difference in the registered objective (certainty equivalent), tested by date with fold-block inference. The no-forecast comparison shows what the forecasts add; the same-input baselines show what reinforcement learning adds.
- **Robustness.**
  - A pre-declared seed ensemble is deployed, and results are reported across seeds as a distribution, never the best seed.
  - Seed, hyperparameter and checkpoint counts go into the ledger for the deflated Sharpe ratio.
  - Sensitivity to costs, rebalance dates and stress scenarios.
  - A complexity budget: development data hold only about 200 independent 20-session decisions.
  - A fallback, holding the registered reference, when the state falls outside the training distribution. State drift is an exit trigger.
- **Explainability, testable.** Each decision shows how and why its weights differ from the rule-based reference, and which constraints bind. A simple surrogate's fidelity to the policy must meet a pre-registered threshold.
- **Realism about power.** These figures were computed by the review under stated assumptions.
  - Against a rule whose returns correlate 0.9 with the policy's, at Sharpe 0.5, the smallest detectable Sharpe difference is about 0.39 over 10 years and 0.71 over 3 years.
  - Showing superiority at an active information ratio of 0.5 (one-sided α = 0.05, power 0.8) takes about 25 years; at 1.0, about 6 years.
  - So S3 for a policy is many years away. A **non-inferiority** test against the reference may justify small, capped personal capital; **superiority** is required before scaling up.
- **Ownership.** Per `AGENTS.md`, agents deliver the design, environment, tests and a bounded skeleton; the owner implements the reinforcement-learning core unless they delegate it.
- **Status.** PPO is a candidate, not a commitment. Constrained mean-variance with validated inputs, and risk-based rules, remain both alternatives and baselines.

## 16. Questions that need external verification

- **Regulation:** current SEC and state treatment of interactive tools (the internet-adviser exemption, the Marketing Rule's hypothetical-performance rules), Reg S-P and state privacy laws, and the status of crypto.
- **Terms of use:** Yahoo, Cboe (VIX), Coin Metrics, and S&P and GICS naming and data.
- **Data access:** Georgia Tech WRDS/CRSP student access.
