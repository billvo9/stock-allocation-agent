# Stock Probability & Reversal Lab

Status: experimental. PR 1 (watchlist, stock data adapter, features, the
Lab page on fixtures) is built. No probability model exists yet, and no
live request has been made. Owner brief: 2026-10-16.

The lab is a lightweight, self-contained feature for a saved list of
US-listed stocks. It is usable before the ML, RL and PPO system is
validated, and it is independent of that system: nothing in the ETF research
track, the prospective ETF pipeline, the RL environment or the portfolio
simulator imports it, and it imports none of them (tested in
`tests/test_stock_lab_boundaries.py`).

## 1. What it answers

For each watched stock, after the close of session d:

- **Direction of each of the next five sessions**: the probability that
  session d+h (h = 1..5) is an Increase, Stable or Decrease session. The
  three probabilities of each session sum to 100%. *(PR 2)*
- **The five-session total, shown separately**: the category of the
  cumulative return from the close of d to the close of d+5. It is never
  confused with any single day. *(PR 2)*
- **Streaks and reversals**, for example "after three declines in a row, how
  often did the next session rise?". *(descriptive tables in PR 1;
  conditional probabilities with intervals in PR 2)*
- **Bull-trap setups**: current conditions that resemble earlier failed
  rallies, distinct from historical rallies confirmed as failed. *(PR 2)*

## 2. Target and categories (financial definitions)

Price return of session t, split-consistent and close to close:

```text
r_t = close_t * split_factor_t / close_(t-1) - 1
```

- `close` is the raw, as-traded close.
- `split_factor_t` is the number of new shares per old share for a split
  whose ex-date is t, and 1 otherwise.
- It is a **price** return: cash dividends are excluded, so an ex-dividend
  session includes the dividend's mechanical drop. It is not the
  dividend-inclusive total return. The page shows the total return beside
  it for reference only.

Categories, with the stable band τ (0.5% by default, configurable, edges
inclusive):

```text
Increase   r_t >  +τ
Stable     -τ <= r_t <= +τ
Decrease   r_t <  -τ
```

**The same band means different things for different stocks.** At ±0.5%,
the synthetic calm demo stock (daily volatility about 0.7%) has about 60%
Stable sessions, and the trend demo stock (about 1.8%) about 26%. Under a
normal approximation:

```text
P(Stable) ≈ 2 Φ(τ / σ) - 1
```

That gives 38% at σ = 1%, 20% at 2% and 13% at 3%. The page shows each
stock's own share of Stable sessions, and the shares at ±0.25% and ±1%.

Proposed, pending the owner (A5 below):

- The five-session total uses the band τ·√5 (about 1.1%). With the daily τ
  unchanged, the Stable category of a five-session total would be almost
  empty.
- A Stable session ends a streak by default. The alternative rule, where up
  to `max_pause` Stable sessions pause a streak, is selectable. The rule in
  use is always stated next to the results.

## 3. Data semantics

### Vendor and fields

The vendor is Tiingo end of day (`stock_agent.stock_lab.tiingo`). It is
separate from the ETF research adapter and reuses only that adapter's
audited key handling and transport.

| Field | Meaning | Use |
|---|---|---|
| `open`, `high`, `low`, `close` | raw, as traded | prices, gaps |
| `volume` | raw shares | only as dollar volume `close × volume` (split-invariant) |
| `divCash` | cash dividend per share, on its ex-date only | ex-dividend flag; excluded from the price return |
| `splitFactor` | new shares per old share on the split's ex-date | split-consistent level and returns |
| `adj*` | vendor-adjusted series | **never requested; refused if served** |

### Point-in-time rules

- Adjusted history is recomputed after every later dividend or split, so
  today's adjusted series is not what anyone could have seen earlier. It is
  not a point-in-time vintage.
- The split-consistent level `L_t = close_t × ∏_{s<=t} split_factor_s` uses
  only splits up to t.
- Every retrieval is a write-once **vintage** in the private store. A
  session served differently by a later retrieval is reported as revised,
  never overwritten. Reads can be taken as of any retrieval time.
- A large split that the raw closes do not show (|log ratio| at least
  3·log 1.25, ratio outside the 25% band) makes that return unavailable
  (`split_price_mismatch`), never guessed. It also breaks the level at that
  session, so no window of prices spans it.
- A move of at least +90% or −47% (|log(1 + r)| at least log 1.9) without a
  split is flagged, never corrected.
- A session on the NYSE calendar that the vendor did not serve is stored as
  a row without prices. A gap stays a gap: a return across it is never used
  as a one-session return.
- The ticker-to-security mapping is Tiingo's view at retrieval. It is not a
  point-in-time security master (tickers are reused), and Tiingo does not
  say whether a listing is an ADR.

### Universe rules

US-listed stocks only, and never a substituted listing. The lab never maps
a foreign ticker to its US listing or ADR, one share class to another, or an
unknown ticker to a similar one.

- "BRK.B" is stored as "BRK-B", the provider's form of the same ticker. The
  entered text is kept and the rewrite is shown.
- Non-US suffixes (VOD.L) and index, currency or futures symbols are refused
  offline.

Statuses, against the provider's listing (`symbols.classify`):

| Status | Meaning |
|---|---|
| Supported | US exchange, stock, USD, recent coverage |
| Unsupported | not listed, abroad, over the counter, not USD, or not a stock, with the reason |
| Inactive | no prices in the last 20 weekdays |
| Delisted or renamed | coverage ended long ago; the provider does not say which |
| Ambiguous | more than one current eligible listing |
| Not yet checked | no listing has been checked |

### Licensing and privacy

- Tiingo Starter (free) is for private, local use, with no redistribution.
- Vendor data and the watchlist live only in the private store
  (`~/stock-agent-private/stock_lab/stock_lab.sqlite3`, mode 0o600 inside a
  0o700 directory). A store inside any Git working tree is refused unless it
  is under that tree's ignored `data/raw/`; a directory open to other users
  is refused; `*.sqlite3` files are ignored by Git.
- The store is write-once by triggers: UPDATE, DELETE and inserts onto an
  existing key abort (so INSERT OR REPLACE cannot overwrite), a recorded
  batch cannot gain rows, and reads re-check each batch's row count and
  content hash.
- The key comes from `TIINGO_API_KEY`. It is sent only to api.tiingo.com,
  never to the listing host, and never appears in a message.

## 4. Features (past-only, `stock_lab.features`)

The row for session d uses d and earlier sessions only.

- Windows are strict: an incomplete window is missing, never shortened.
- Prefix stability is tested with the project's `assert_prefix_stable`. A
  deliberately leaky feature (the next return) is shown to fail that check.
- Features are invariant to splits (tested).

The features are:

- the current run (direction, length, Stable run, pauses) and its signed
  length;
- category counts over the last 20 sessions;
- the longest rising and falling runs over the last 60;
- returns over 1/3/5/10/20/60 sessions, compounded from the session returns
  of the window (missing if any is);
- volatility over 20 and 60 sessions, and EWMA volatility (λ = 0.94);
- the volatility ratio and the 5-session change in volatility;
- the distance from the 20- and 50-session moving averages, and the slope
  of the 20-session average;
- drawdown from the 20/60/252-session high, and bounce off the 20/60-session
  low;
- momentum acceleration;
- relative dollar volume;
- the genuine open-to-previous-close gap;
- τ / σ (how wide the band is for this stock now);
- an ex-dividend flag for the session itself. A forward ex-dividend flag
  would be lookahead: Tiingo marks a dividend only on its ex-date row, so a
  forward flag needs point-in-time declaration data, which the lab does not
  have.

Market and sector context is excluded until a source with its own
availability timestamps is approved.

## 5. Statistical design (PR 2; reviewed 2026-10-16)

An independent data-science review of the design is kept in the session
notes. Its must-change items are adopted here.

### Estimators

Every probability comes from a documented, fitted estimator; none is
hard-coded.

| Id | Estimator | Notes |
|---|---|---|
| B0 | Unconditional frequencies, Dirichlet(α = 50) prior toward the pooled frequencies | expanding window |
| B1 | Exponentially weighted frequencies, half-life H (63), prior κ = 40 toward B0 | n_eff = (Σw)² / Σw² ≈ 2.885·H |
| B2 | Run-length tilt of B1: P_d(c' \| c, ℓ) ∝ B1_d(c') · exp(θ_{c,ℓ,c'}), θ = θ_pool + δ_i by penalized multinomial logit | θ = 0 is exactly B1; joint paths via the chain |
| B3 | Volatility-scaled empirical CDF: P(Stable) = F_h(τ/σ) − F_h(−τ/σ), with F_h the ECDF of z⁽ʰ⁾ = r_t / σ_(t\|t−h) shrunk toward the pooled ECDF (κ_F = 250) | horizon enters through F_h, since the EWMA forecast is flat |
| M1 | L2 multinomial logistic, one per horizon, pooled panel, volatility-normalized features plus logit(B3), penalized stock intercepts | kept only if it beats the baselines |

- The B2 design changed after review. A Dirichlet transition chain fitted on
  full-history counts converges to its long-run distribution within 2–3
  steps, so it would collapse to B0 for h ≥ 2.
- Gradient-boosted trees and HMMs are added only if walk-forward testing
  shows an incremental benefit.

### Five sessions: direct per-horizon models

- **Direct models, not path simulation.** Daily categories of different
  sessions are different variables, so any five valid marginals are jointly
  feasible, and direct estimation creates no incoherence. Path simulation
  would need a full conditional return density and compounds its
  misspecification.
- **The one cross-constraint.** With the cumulative category,
  P_cum(Increase) >= max(0, Σ_h p_h(Increase) − 4). It is asserted.
- **Full-path calibration** is scored on B2's joint path probabilities: all
  five sessions in one direction, the nine joint cells of (d+1, d+2), and
  the count of Increase sessions.
- **No predicted path.** The page never shows a "predicted path".

### Evaluation

- **Folds.** Strict chronological walk-forward folds (`make_expanding_folds`,
  development mode, lockbox 2025-01-01): yearly test windows, an expanding
  training window, and a strict purge (a row trains only if its target ended
  before the test start).
- **Fold contract.** Parameters (H, κ, λ, θ, β) are frozen at the test start.
  Estimator state (weighted counts, σ, run length) updates online with data
  up to each decision date only.
- **Scores.** Multiclass Brier score, log loss and the ranked probability
  score (RPS, the primary score, which respects Decrease < Stable <
  Increase).
- **Separate skills.** P(Moving) and P(Increase | Moving) are scored
  separately.
- **Calibration.** Reliability diagrams per class, and the calibration slope
  and intercept.
- **Breakdowns.** Results by horizon, volatility regime (point-in-time
  terciles) and stock.
- **Inference.**
  - Per-date cross-sectional mean loss differentials.
  - Newey-West with lag 10 (at least h − 1), a circular block bootstrap
    (block 20) and the fold-block t, each method's size checked by
    simulation.
  - Holm correction across the pre-registered comparisons against B1.
- **No selection on test results.** H comes from a pre-registered grid
  {21, 63, 126, 252}, selected in inner purged folds with a one-SE rule.
  Every candidate tried is recorded.
- **Sensitivity.** τ and H sensitivity is reported and never used for
  selection.

### Expected power

Daily direction is nearly unpredictable.

- **Effective breadth.** Twenty names at a pairwise correlation of 0.4
  behave like about 2.3 independent names per date.
- **Direction is out of reach.** A true 2-percentage-point edge in direction
  would need about 19,000 independent observations, roughly 30 years.
- **Volatility skill is detectable.** Expect to detect volatility (Stable
  versus Moving) skill, not direction skill. The page says so.

### Reversals and bull traps

**Reversal events**

- A streak event is a run reaching exactly k, counted once per run.
- Gain events (r > X) are declustered with a 5-session gap.
- Outcomes over 1/3/5 sessions are continuation, stabilization or reversal.

**Reversal statistics**

- Jeffreys intervals use the effective sample size.
- Kaplan-Meier durations treat open runs as right-censored; the hazard
  equals 1 − P(continue), which is tested.
- Events with n_eff < 20 show "Insufficient observations"; 20–49 are
  labelled low sample.
- Results are also shown at nearby thresholds.

**Bull traps**

- A registered rule:
  - a decline of at least 10% from the 60-session high;
  - a rebound of at least 5% off the 20-session low;
  - a close above the prior 20-session closing high;
  - failure: a close below that high × (1 − f) within W = 10 sessions.
- A present-day setup is "unresolved" until its window closes, never
  "confirmed".
- The failure rate is compared with a volatility-matched base rate over all
  declustered breakouts, with lift, precision, recall and false-alarm rate
  FP / (FP + TN). Bear traps mirror the rule.

**Outcomes stop at the cutoff**

- Event outcomes and durations are censored at 2024-12-31 in development,
  and at the decision date live.
- They are never resolved from sealed data.

## 6. Forecast artifact and ledger (PR 3)

`StockProbabilityForecast` v1 is a versioned, write-once record. Each one
carries:

- the as-of session, creation time and five target sessions;
- the three probabilities per session, and the cumulative five-session
  forecast;
- the model and feature versions;
- the data source, vintage and retrieval time;
- the evidence and calibration status;
- reversal and trap signals;
- data-quality limitations and refusal reasons.

How the ledger works:

- A recalculation is a new record that names the one it supersedes; nothing
  is overwritten.
- Scores are appended as sessions close; the original probabilities are
  never rewritten.
- A future PPO policy may read these records only through an approved
  point-in-time reader. Hindsight-reconstructed forecasts are never treated
  as historically available.
- The PPO reward and the portfolio simulator are unchanged.

## 7. Owner decisions needed

| Id | Decision | Needed before |
|---|---|---|
| A1 | Live Tiingo data (authorization request below) | the first real data |
| A2 | Models frozen on data through 2024. At forecast time, the trailing 252 sessions (reaching into the sealed period) are used **only as inference inputs**, never to train, calibrate, select or score | real forecasts |
| A3 | Walk-forward evaluation of the watchlist stocks on pre-2025 history, labelled hindsight-selected and non-confirmatory. This is an exception to the 2026-10-13 "no new real-stock scoring experiments" for this feature only | real-stock scores |
| A4 | Recording real forecasts in the private ledger and scoring them only on outcomes after each forecast's creation, separate from OD16 | PR 3 on real data |
| A5 | Definitions: τ·√5 for the five-session total; a Stable session ends a streak by default; the bull-trap parameters | final confirmation |
| A6 | Whether real prices after 2024 may be shown on the page (descriptive statistics in the sealed period). Default: no; the page stops at 2024-12-31 | showing recent real history |

### A1, the authorization request

Approve one owner-run command on this computer:

```bash
.venv/bin/python scripts/stock_lab_fetch.py --authorized --store
```

Exact scope:

- Tiingo Starter, with the key in `TIINGO_API_KEY`.
- 1 request for the supported-tickers listing (no key sent). It is used
  only to classify tickers.
- 1 prices request per supported ticker of the saved watchlist (at most 30;
  a longer list is refused before any request; the script has no option to
  add other tickers or raise the cap) for 2003-01-02 to 2024-12-31: raw open,
  high, low, close, volume, divCash and splitFactor. No adjusted fields and
  no date after 2024.
- Writes into the private store only: the whole listing (every row, kept as
  the classification's audit trail), the checks, and one write-once batch
  per ticker.

Run it without `--store` first for a read-only check, or with no flags for
the plan.

## 8. Learning notes

- **Problem.** Honest probabilities for a three-way daily outcome, and
  descriptive streak statistics, that never use the future.
- **Concepts.**
  - A proper scoring rule (Brier, log loss, RPS) is minimized only by the
    true probabilities.
  - Kish's effective sample size measures how much information weighted
    observations carry.
  - Fréchet bounds give what marginals allow for joints.
  - Kaplan-Meier estimates durations when some runs are unfinished.
- **Assumptions.** Direction is close to independent day to day;
  volatility clusters; censoring at the end of the data is unrelated to
  duration.
- **Small example.**
  - Under a normal approximation, P(Stable) = 2Φ(τ/σ) − 1.
  - At τ = 0.5% and σ = 1%, that is 2Φ(0.5) − 1 ≈ 0.383.
  - At σ = 3%, it is 2Φ(0.167) − 1 ≈ 0.132.
  - The same band is "wide" for one stock and "narrow" for another.
- **Leakage risks.**
  - A forward ex-dividend flag.
  - Outcomes resolved from sealed data.
  - Hyperparameters tuned on outer test folds.
  - Vendor-adjusted prices.
  - Watchlist selection with hindsight.
- **How results could mislead.**
  - "Stable" skill mistaken for direction skill.
  - Reading the five daily marginals as a path.
  - Small event counts shown as precise probabilities.
  - Chart-pattern names taken as validated signals.
- **How practitioners validate.** Walk-forward with nested selection,
  Diebold-Mariano and block bootstrap tests, reliability diagrams with the
  Murphy decomposition, and prospective scoring.
- **What to learn next.** The Murphy decomposition of the Brier score;
  discrete-time survival hazards; Hansen's SPA test for many comparisons.
- **Exercise.** Show that under the "breaks" rule the per-session view
  reports, at each run's last session, that run's length. That is the
  property tested in `test_every_run_row_matches_the_per_session_view_at_its_end`.
  Then explain why the "pauses" rule cannot have it for short Stable
  stretches.
