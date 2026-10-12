# Risk register

Status (2026-10-11): **direction approved by the owner on 2026-10-11** (OD1, OD2, OD4, OD5, OD6, OD7, and the general direction of OD16 in `vision_and_prd.md`, section 13). OD16's activation and holdout end, and the other decisions, are pending. Likelihood and impact are judgments (L = low, M = medium, H = high) for the current plan. Legal items are issues to check with qualified counsel, not legal advice. Decision numbers (OD1–OD17) refer to `vision_and_prd.md`, section 13. Risk IDs carry a category prefix (STAT, DATA, PORT, LEGAL, SEC, OPS, UX).

## Statistical

| ID | Risk | L | I | Early signal | Mitigation |
|---|---|---|---|---|---|
| STAT-1 | Underpowered results read as evidence (four names: MDE 0.16–0.25) | H | H | A "promising" estimate inside the MDE | MDE pre-registered and shown with every estimate; product claims gated by the ladder; "insufficient confidence" state |
| STAT-2 | Statistics computed on different date subsets compared as if equal (Lasso's 0.070 on 13 of 24 folds) | M | M | Coverage below 100% | Coverage shown with every partial estimate (exists since T4); same-date comparisons only |
| STAT-3 | P(gain) acting as a hidden low-volatility screen | M | H | P(gain) correlates with −σ, not with realized alpha | No P(gain) without L4 and D2; never sort by it below the location rungs |
| STAT-4 | Composing absolute forecasts from components on mismatched information sets, or missing or doubling beta exposure (illustration: about 3 percentage points at β = 1.8 under a −3% market view) | M | H | The market-exposure diagnostic's slope (+1 means exposure is missing; −1 means it is counted twice; attenuated by noise in β̂) | The exact sum by default. Formulation B only under its documented approximation and its outcome-free eligibility check, chosen by pre-registration. The diagnostic, with an ex-post control and pre-registered decision regions. The composed forecast tested as a whole (`architecture.md`, section 5). The cross-sectional invariant only guarantees the market level enters once on average; it cannot detect beta errors |
| STAT-5 | Multiple testing across horizons × universes × models × themes; repeated scoring of one development window | H | H | Variant count grows faster than evidence | A Git-tracked record of every real-panel scoring (G1); counts from before it shown as lower bounds ("≥ n"); a test family and α per cell; a fallback procedure with α weights assigned to each horizon in advance (so a failure at one horizon does not forbid claims at the others), each horizon its own test family; the RES-2 variant set frozen before U1 results are read; an effective-trial-count rule for the deflated Sharpe ratio and SPA; Romano-Wolf or SPA across models; selection-free estimates (against the winner's curse); retired variants stay visible |
| STAT-6 | Lockbox eroded informally (looking at any 2025+ information: outcomes, trailing returns, holdings snapshots, adjusted prices) | M | H | Requests to "just check recent performance" | Default boundary OD5: no product publication reads data from 2025-01-01, enforced by the writer and the reader; T6 runs once for everything pre-registered; Git-tracked holdout record |
| STAT-7 | The holdout cannot confirm a small edge (one-sided MDE provisionally about 0.05 in rank IC for 100 names), and its outcomes were known to the designer, so they are not independent evidence. Twelve prospective months detect a collapse with only about 25–70% power, depending on α and the overlap factor (provisional) | H | M | A "confirmed" label after T6 or one year | `holdout_checked` carries no promotion power and contributes nothing to the confirmatory test (a descriptive baseline only). Only `prospectively_confirmed` unlocks Explore and Build: a pre-registered one-sided test on post-registration data with α-spending. Provisionally, confirming an IC of 0.03 needs about 1,170 post-registration dates (one-sided α = 0.025, power 0.8), about 1,760 for a persistent signal |
| STAT-8 | Miscalibration, and calibration decay | M | H | Coverage or PIT drift; adaptive-conformal α at its bound | Nested-fold calibration; measured decay sets the refit cadence; product falls back to "historical range" |
| STAT-9 | Conformal or interval coverage fails on crash dates, when every name misses at once | M | H | Date-clustered coverage far below average coverage | Report date-clustered coverage; stress scenarios beside intervals |
| STAT-10 | Error maximization in optimizers fed unvalidated forecasts | M | H | Concentrated, unstable weights | No expected-return optimizer before L3; alpha only as a bounded tilt; λ = 0 when unavailable |
| STAT-11 | Date-common forecast component and row-weighted fitting bias selection under a variable universe | H | M | Squared-error advantage below the within-date advantage | Date-weighted fitting or within-date features (formula gate, RES-2) |
| STAT-12 | Embargo too short for future non-forward validation schemes | L now, M later | H | A K-fold or CPCV proposal | Post-test exclusion of about h + L + 1 sessions documented before such schemes exist |
| STAT-13 | Metric uncertainty not shown (CAGR, Sharpe, drawdown as points) | H | H | A headline number without an interval | Block-bootstrap intervals and a Sharpe interval beside every headline. For example, a Sharpe of 0.5 over 17 years has a 95% interval of about ±0.5 (i.i.d. approximation) |
| STAT-14 | Designer knowledge contamination: the repository began on 2026-08-08, so every design was made knowing what happened in 2025 | H (certain for current designs) | M | Choices that would have looked different without 2025 hindsight | `designed_after_holdout_start` flag with the 2025+ facts known; prospective evidence as the only clean confirmation; pre-registration before any prospective look |
| STAT-15 | A learned allocation policy (the PPO candidate) overfits the backtest: seed instability, reward hacking, many hyperparameters, normalization leakage, in-sample forecasts used as state | H | H | Results vary widely across seeds; gains vanish against same-input baselines, without forecast inputs, or under higher costs | The policy requirements in `vision_and_prd.md`, section 15: out-of-fold state, per-fold frozen normalization, inner-fold checkpoint selection, a seed ensemble, an intersection-union test against rule, same-input and no-forecast baselines, ledger counts, a complexity budget, then paper trading. Power is low (provisionally about 31 years to show superiority at an information ratio of 0.5, at one-sided α = 0.025). Non-inferiority permits at most a capped trial and shows no positive value |
| STAT-16 | Promotion procedures, sample sizes and power figures are used before their finite-sample behaviour is known (overlap, cross-sectional correlation, regime shifts, heavy tails) | M | H | A registration that relies on a normal approximation | **Blanket rule:** every power, sample-size, MDE and detection figure in these documents is provisional, whether or not it is labelled. Size and power of the whole sequential procedure are checked by simulation before any registration relies on them. The simulation estimates dependence from pre-2025 data only, uses small-sample critical values, and registers its code, seeds and acceptance tolerance (`vision_and_prd.md`, section 15) |

## Market data

| ID | Risk | L | I | Early signal | Mitigation |
|---|---|---|---|---|---|
| DATA-1 | Survivorship bias (yfinance has no delisted tickers) | H | H | Missing acquired names (XLNX, MXIM, LLTC) | Licensed survivorship-free source (OD13); "current-listing universe" label until then; survivorship sensitivity bound |
| DATA-2 | Ticker and CIK identity errors (SNDK is two issuers; MRVL changed CIK) | H | M | Duplicate or orphaned histories | Security master with permanent ids and interval tables |
| DATA-3 | Hindsight theme lists ("Magnificent Seven", "AI/semiconductors") | H | M | A theme backtest is requested | Themes are dated current-exposure lenses; point-in-time characteristics; registration before evidence (OD11) |
| DATA-4 | Price-index benchmarks compared with total-return strategies | H (today) | H | Benchmark `adjusted_close` equals `close` | Total-return benchmarks in T5 |
| DATA-5 | Yield series mistaken for instrument returns | M | M | Treasury "returns" computed from yield levels | Investable total-return series only; yields as context with the duration formula labelled "approximation" |
| DATA-6 | Back-adjusted vendor prices change on every download | M | M | Hash changes for unchanged history | Store raw closes plus dated distributions and factors; record retrieval time |
| DATA-7 | Stale or revised inputs (DGS3MO has no `available_at` and overwrites revisions) | M | M | Freshness monitor | `available_at` per source; vintages where revised; stale state |
| DATA-8 | Asynchronous calendars (bonds on SIFMA, crypto 24/7, foreign closes) | M | M | Same-date joins across calendars | Named calendars; availability-based joins; weekend folding |
| DATA-9 | Licensing: yfinance, exchange, S&P/GICS, VIX, Coin Metrics or academic (CRSP) data used beyond its terms | M (H if shared) | H | Any sharing or hosting | Personal use only now; licences per stage (PRD, section 9); separate lineage for academic data |
| DATA-10 | Asset-class hindsight: a US-only equity sleeve chosen in 2026, after about 15 years of US outperformance | H | M | Reference portfolios use only US funds | Offer a global equity fund, or label "US-only, chosen 2026"; record the alternatives considered in the registry |
| DATA-11 | Fund method changes, index switches, closures and mergers | M | M | A fund's index or sponsor changes | Method history in the registry; a registered proxy rule; the change is shown on the timeline |
| DATA-12 | Proxy splicing (an index or older fund stitched onto a fund's history) | M | M | A seamless line before a fund existed | Proxies labelled and drawn as their own segment; never seamless |

## Portfolio construction and simulation

| ID | Risk | L | I | Early signal | Mitigation |
|---|---|---|---|---|---|
| PORT-1 | Same-close execution flatters backtests | H for today's daily research backtests; L for quarterly fund portfolios | H | Label entry lag ≠ backtest execution lag | Next-close execution (T5) with a test tying the two |
| PORT-2 | Cash earns zero or the wrong convention (40% cash at 4–5% is about 1.6–2% a year) | H (today) | H | `cash_return=0.0` | Cash returns from bills or a T-bill fund (T5) |
| PORT-3 | Costs and liquidity understated (flat 10 bp) | M | M | High turnover strategies look best | Cost model by asset class, sensitivity shown, participation check |
| PORT-4 | Concentration and tail correlation hidden by diversification metrics (semiconductor pairwise correlation 0.58) | H | H | Effective number of holdings far below the count | Look-through exposures, effective holdings, stress scenarios, correlation in stress |
| PORT-5 | Cherry-picked inception dates or constituents | M | H | An inception chosen after results | Inception by rule, registered before the first run (G1) |
| PORT-6 | Backtest, paper and live curves spliced | M | H | A single continuous line across evidence types | Separate series and badges; a line never crosses evidence types |
| PORT-7 | Users overfit by browsing configurations | M | M | Many configurations viewed | Discrete options; a "configurations viewed" counter; no "best" sort |
| PORT-8 | The stock–bond correlation regime changes; bonds and stocks fell together in 2022 | H | H | Rising rolling stock–bond correlation | Rate-shock scenarios (2013, 2022) beside equity crashes; rolling stock–bond correlation shown; no "safe" naming |
| PORT-9 | Risk-based rules become degenerate with cash (inverse volatility divides by zero; minimum variance holds cash to its cap) | M | M | Weights collapse into cash | Cash only as a floor or a fixed weight; tests for both |
| PORT-10 | Rebalance-timing luck (results depend on the rebalance dates chosen) | M | M | Large spread across rebalance offsets | Rebalance-date sensitivity reported and counted as variants |
| PORT-11 | Tax and distribution treatment (dividend reinvestment timing, pre-tax results) | H | M | Results quoted as if after tax | Reinvestment rule stated (ex-date vs pay date); every simulation labelled "before taxes" |

## Legal, compliance and licensing (verify with counsel)

| ID | Risk | L | I | Early signal | Mitigation |
|---|---|---|---|---|---|
| LEGAL-1 | A personalized Builder treated as unregistered investment advice | H if others use it | H | Anyone other than the owner, or any compensation | Research and education design (PRD, section 9); counsel before any other user (OD1) |
| LEGAL-2 | Misleading hypothetical performance (SEC Marketing Rule if an adviser) | M | H | Performance shown to others | Labels, net of costs, assumptions, no-skill and survivorship caveats; counsel |
| LEGAL-3 | Risk questionnaires raise suitability and fiduciary expectations | M | M | A questionnaire is proposed | Risk limits are user-set constraints, not profiling |
| LEGAL-4 | Crypto rules change or conflict | M | M | Crypto in an allocation | Crypto deferred; counsel before inclusion |
| LEGAL-5 | Trademark or naming issues ("S&P 500", "Magnificent Seven") | L–M | M | Index names in product copy | Descriptive naming; licences before commercial use |
| LEGAL-6 | Restricted securities (OFAC NS-CMIC) in a universe | L | H | — | Rule R7 in every investable universe |

## Privacy and security

| ID | Risk | L | I | Early signal | Mitigation |
|---|---|---|---|---|---|
| SEC-1 | User financial inputs (capital, holdings, limits) leaked or retained | M if shared | H | Any logging or storage of inputs | Session-only inputs; no storage or logging; no third-party analytics, fonts or CDNs |
| SEC-2 | API key exposed in errors, tracebacks or logs | M | M | FRED errors chained with the request URL; `*.log` not ignored (`.gitignore` line 44); urllib3 logs Version 1 request URLs, `api_key` included (DEBUG for every request, WARNING when a retrying adapter retries) | Errors carry no URL or chained cause (PR #48 for Version 1, PR #50 for Version 2; both pending merge); the ignore line fixed in #48. **Requirement before any shared or hosted deployment:** keep `urllib3.connectionpool` above DEBUG, mount no retrying adapters for Version 1 calls, and either move Version 1 to header authentication (if FRED accepts it; unverified) or add a logging filter that redacts `api_key=` |
| SEC-3 | Streamlit exposed without authentication | L now | M | Any non-localhost binding | Localhost only; an authenticating proxy and TLS before hosting |
| SEC-4 | Supply-chain compromise (unpinned transitive packages, tag-pinned CI actions) | L–M | H | New dependencies | Hash-locked dependencies and scanning before any hosting (owner gate) |
| SEC-5 | No audit trail of what was shown | M if shared | M | — | Each view records publication ids and disclosure versions under a pseudonymous session id, never inputs |

## Operations and deployment

| ID | Risk | L | I | Early signal | Mitigation |
|---|---|---|---|---|---|
| OPS-1 | Someone looks at the prospective pipeline's outcomes outside review dates and adapts the system | M | H | Unlogged access; changes after an unscheduled look | Separate store and code path; the scorer refuses to report before a registered review date; the log's hash committed to Git at each decision; any change is a new registration in the same lineage (`architecture.md`, section 9). An access log cannot be enforced on a local machine and is not relied on |
| OPS-2 | A promoted artifact changes without re-registration | M | H | A code or parameter hash differs from its registration | Registration hash checked before every decision; mismatch blocks the decision and returns the artifact to stage S2 |
| OPS-3 | Execution errors or drift between intended and executed trades (stage S3) | M | H | Reconciliation breaks | Reconciliation on every rebalance; manual approval of each rebalance at first; a tested kill switch |
| OPS-4 | Data outages or stale inputs during live operation | M | M | Freshness monitors | Stale state blocks new decisions, and the default is to **hold** existing positions, never to trade on stale prices |
| OPS-5 | Treating model completion as readiness | M | H | A request to deploy because "the model is finished" | Promotion stages with owner approval at each step (OD17) |
| OPS-6 | Live data later revised by the vendor (vintage drift), so live decisions and their replay disagree | M | M | Replay mismatch | Store each live input with its retrieval time; measure drift against later vintages; a tolerance is an S3 entry check |
| OPS-7 | Runtime state changes a registered artifact (for example normalization statistics updated live) | M | H | The registration hash no longer matches the running state | Normalization and other state frozen in the artifact; the hash covers runtime state; a mismatch blocks the decision |

## Product design

| ID | Risk | L | I | Early signal | Mitigation |
|---|---|---|---|---|---|
| UX-1 | A backtest read as a track record | H | H | Users quote simulated returns | "Simulated" badge, separate series, "What this cannot tell you" always visible |
| UX-2 | False precision (decimal probabilities, single lines without bands) | M | H | Decimals in Explore | Whole percents in Explore; bands only for calibrated distributions |
| UX-3 | Progressive disclosure hides material caveats | M | H | A caveat only in Research | Caveats are never collapsible; mode sets depth, not permission |
| UX-4 | Nudging defaults (a "top picks" sort, green buy signals, leaderboards) | M | M | Any default sort by a score | Name sort by default; neutral evidence colours; banned-word copy tests |
| UX-5 | Empty or unavailable states rendered as zeros or blank charts | M | M | A zero where a reason belongs | First-class state components with reason codes; tests per state |
| UX-6 | Inaccessible charts | M | M | Charts without text alternatives (open today) | A summary sentence and a data table per chart; AA tokens |
| UX-7 | Outcome-promising names ("Capital Preservation", "Opportunistic") | H | M | Names describe hoped-for outcomes | Names describe construction; the intent is a subtitle |
| UX-8 | The product drifts toward whichever model scored best | M | H | A default changed after a development run | Defaults by declared rule; variant counts shown; registry |
