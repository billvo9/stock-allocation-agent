# Stock Probability & Reversal Lab

Status: experimental.

- PR 1 (#78) built the watchlist, the stock data adapter, the features and
  the Lab page on fixtures.
- PR 2 builds the estimators, the five-session forecasts, the reversal
  analyzer, bull and bear traps, the walk-forward evaluation and the
  modelled views.
- Briefing PR 1 adds the read-only daily premarket briefing and its
  reliability foundations (section 9).
- The Portfolio Lab's read-only Premarket Briefing page and a printed
  08:30 delivery option (section 9); nothing is installed or sent.

Probabilities exist only for the synthetic demo stocks. No live request has
been made. Owner brief: 2026-10-16.

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
| B2 | Run-length tilt of B1: P_d(c' \| c, ℓ) ∝ B1_d(c') · exp(θ_{c,ℓ,c'}), with θ pooled across stocks per state and fitted by penalized multinomial logit (per-stock δ_i: a follow-up) | θ = 0 is exactly B1; joint paths via the chain |
| B3 | Volatility-scaled empirical CDF: P(Stable) = F_h(τ/σ) − F_h(−τ/σ), with F_h the ECDF of z⁽ʰ⁾ = r_t / σ_(t\|t−h) shrunk toward the pooled ECDF (κ_F = 250) | horizon enters through F_h, since the EWMA forecast is flat |
| M1 | L2 multinomial logistic, one per horizon, pooled panel, volatility-normalized features plus logit(B3), penalized stock intercepts | kept only if it beats the baselines |

- The B2 design changed after review. A Dirichlet transition chain fitted on
  full-history counts converges to its long-run distribution within 2–3
  steps, so it would collapse to B0 for h ≥ 2.
- Per-stock tilts δ_i in B2 are not built. The only admissible evidence of
  a calibration benefit is the real development evaluation (A3, pending).
  A synthetic panel has whatever streak structure its generator put in, and
  a variant added after seeing results counts against the multiplicity
  budget.
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
- **Breakdowns.** Results by horizon, by stock, and by volatility regime:
  the stock's EWMA volatility at d against the terciles of its own
  volatility through d (an expanding, point-in-time rank).
- **Inference.**
  - Per-date cross-sectional mean loss differentials.
  - **The decision test (owner decision A7, confirmed 2026-10-19).** The
    fold-block t decides: a one-sided t-test on the test years' means,
    t(folds − 1), with t(18) on the 19 yearly folds of real history.
  - Holm runs across the pre-registered challengers (B0, B2, B3, M1) against
    B1 at a family level of 2.5%.
  - Beside it, reported only: Newey-West with lag 10 (at least h − 1) and a
    circular block bootstrap (block 20).
    `size_simulation.one_sided_tests` computes all three for every comparison.
  - Records name the rule (`decision_rule`, evaluation schema 1.3). An older
    record, decided by Newey-West at 5%, says so, and it never registers a
    challenger (`stock_lab.frozen_model`).
  - **Simulated size** (`size_simulation`, recorded with every evaluation).
    No-skill panels have the evaluation's own number of stocks and test
    years (after a three-year warm-up). Returns are independent over time
    and equicorrelated across stocks (ρ = 0.4, Student-t₅, 1.5% daily).
    The challenger is a placebo: B1 read from the next stock's history,
    at each pre-registered half-life. It is compared with two references.
    - Another stock's B1 (the previous one). The differential is
      symmetric, so its size measures the tests alone.
    - The stock's own B1. That one keeps learning from the very sessions it
      is scored on, and the placebo does not. The real comparisons have
      this imbalance too.
    - Each date's differential has mean exactly zero under both (checked:
      within one Monte Carlo SE of zero everywhere).

    At the watchlist's shape (20 stocks, 15 test years, 1,000 panels,
    Monte Carlo SE about 0.7 pp), the one-sided false-positive rate at a
    nominal 5% was:

    | Placebo | Newey-West (lag 10) | Fold-block t | Block bootstrap |
    |---|---|---|---|
    | Symmetric | 6.0–6.4% | 5.2–5.6% | 5.2–5.8% |
    | Own history | 6.5–8.1% | 6.2–8.0% | 5.9–7.2% |

    - Newey-West at lag 10 is mildly liberal: its variance is about 16%
      too small.
    - The fold-block t holds its size.
    - The learning imbalance adds 1–3 pp to every test, in the
      challenger's favour, most at H = 252.
    - On the three demo stocks over three years, every test reaches 8–10%
      under the own-history placebo, and Newey-West 6–7.5% under the
      symmetric one.
    - Which test decides was owner decision A7. Its decision packet,
      `docs/research/stock_lab_a7_decision.md`, simulates six tests under
      the planned conditions and recommends the fold-block t at a one-sided
      family level of 2.5%, which the owner confirmed on 2026-10-19.
- **Selection.**
  - Keep B1 unless a challenger beats it: a lower mean RPS with a
    Holm-adjusted one-sided fold-block t p below 0.025.
  - Then take the simplest such challenger (order B0, B1, B3, B2, M1).
  - Move to a more complex one only if it also beats the current choice on
    the same per-date differences: a one-sided fold-block t p below 0.025.
  - The steps are recorded.
- **No selection on test results.** The half-life comes from the
  pre-registered grid {21, 63, 126, 252} and M1's penalty from {0.01, 0.1,
  1}. Both are chosen inside the training years by the one-standard-error
  rule with paired errors: the standard error of the per-date difference
  from the best candidate, which removes shocks common to a date. Every
  inner score is recorded.
- **Uncertainty beside a forecast.** The track record shows how often
  development-test forecasts in the same range came true. Its Jeffreys
  interval counts distinct dates, not rows, because rows on one date move
  together.
- **Sensitivity.** Reported, never used for selection:
  - B1's score at every half-life.
  - B1 and B3 re-run at τ = 0.25% and 1% on the same folds and rows. Each
    band uses its own categories and targets, B1's half-life is chosen
    inside each fold by the same rule, and B3 uses the same
    standardized-return distributions (z = r/σ does not depend on τ).
  - RPS levels are not comparable across bands, because a wider band makes
    Stable more common. Compare B3 − B1 within a band.
  - The τ variants count as variants tried.

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
  - failure: a close below that high × (1 − f) within W = 10 sessions,
    with f = 2% (`TrapRule.failure_margin`; awaiting the owner's
    confirmation before real-data flags, below).

  The failure rule exactly, on the split-consistent price L:

  ```text
  H_d       = max(L_(d-20), ..., L_(d-1))          the prior 20-session closing high
  breakout  : L_d > H_d
  failed    : min(L_(d+1), ..., L_(d+10)) < H_d * (1 - f)
  unresolved: fewer than 10 sessions after d have closed (or a gap, or the cutoff)
  ```

  Example (f = 2%). The prior 20-session high close is H_d = 100.00, and
  the breakout session closes at 102.50. The failure level is
  100.00 × 0.98 = 98.00. If the next closes are 101.20, 99.60, 97.85, ...,
  the third (97.85 < 98.00) confirms a failed breakout, a bull trap. Had
  the lowest of the ten closes been 98.40, the breakout would have held.
  A smaller f calls more breakouts failures; for a stock with daily
  volatility σ, a 2% margin is about 2%/σ typical daily moves (two for a
  1%-volatility stock, under one for a 3% one). That is why the base rate
  is volatility-matched.
- A present-day setup is "unresolved" until its window closes, never
  "confirmed".
- Breakouts are declustered first, and every setup is one of those
  breakouts, so precision equals the setups' failure rate. The decline
  must run from the longer high down to the recent low (the high comes
  first).
- The failure rate is compared with the base rate of all breakouts,
  reweighted to the setups' own mix of volatility terciles (the matched
  base rate; the lift is against it). Precision, recall and the false-alarm
  rate FP / (FP + TN) are reported. Bear traps mirror the rule.

**Outcomes stop at the cutoff**

- Event outcomes and durations are censored at 2024-12-31 in development,
  and at the decision date live.
- They are never resolved from sealed data.

### Implementation (PR 2)

| Module | Contents |
|---|---|
| `targets` | daily targets d+1..d+5 and the five-session total (band τ√5, proposed A5); a gap or the outcome cutoff makes a target unavailable |
| `estimators` | B0, B1 (online state, Kish n_eff), B2 (the run-length tilt: per-state penalized Newton fit with log B1 as offset; chain marginals and path probabilities) and B3 (EWMA volatility with a horizon-specific ECDF shrunk to the pooled one) |
| `logistic` | M1 with scikit-learn: train-only median imputation and standardization; C from {0.01, 0.1, 1} by a purged inner split with the one-SE rule toward more shrinkage; stock indicators penalized like every coefficient |
| `scores` | Brier, log loss, RPS, the moving and direction decompositions, the Jeffreys interval, reliability bins, the calibration line, Holm |
| `reversal` | streak and move events, outcomes over 1/3/5 sessions, regimes, Kaplan-Meier durations |
| `traps` | the registered rule, statuses, base rate, lift, precision, recall and false-alarm rate, bear traps |
| `evaluation` | the protocol in the module docstring (yearly folds through 2024, development mode) and a strict-JSON record (schema 1.1 adds the fold-block t and bootstrap p per comparison, `sensitivity_band` and `size_simulation`) |
| `size_simulation` | the three one-sided tests used for every comparison, and their simulated size on no-skill panels shaped like the evaluation's (two placebos, every pre-registered half-life; seeded and cached per design) |
| `predict` and `forecast` | the frozen final fit and the checked `StockProbabilityForecast` artifact |
| `forecast_view` | the page's modelled views, computed from stored records only |

`scripts/stock_lab_evaluate.py --demo` stores the synthetic evaluation and
one demo forecast per demo stock. `--real` refuses without the A3 flag, and
stores no real forecast.

The Lab never imports `logistic`, `evaluation`, `predict` or `tiingo`
(tested). It reads their results as stored records.

**Synthetic check (machinery only, not evidence).** On the three demo
stocks, three yearly folds (2022–2024), mean RPS over the five daily
targets:

| Estimator | Mean RPS |
|---|---|
| B0 | 0.2072 |
| B1 | 0.2074 |
| B2 | 0.2068 |
| B3 | 0.2063 |
| M1 | 0.2051 |

The demo generator has volatility regimes and mild momentum, so beating B1
is expected there. On independent, identically distributed synthetic
returns, the test asserts that no streak skill is selected (B2 is not
significant after Holm).

The Model diagnostics tab shows each comparison's three p-values, and two
expanders: the simulated sizes for both placebos, and the band sensitivity.
On the demo stocks, B3 − B1 was −0.0005, −0.0009 and −0.0012 at τ = 0.25%,
0.5% and 1%, with Stable shares of 21%, 39% and 64%.

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
  is overwritten. There is at most one live forecast per stock and as-of
  session. The superseded forecast must exist for the same stock and
  session, and only the live one counts in the rolling scores. A new day's
  forecast is a new record, not a recalculation.
- A forecast is recorded when it is made, and the store refuses any other
  timing:
  - `created_at` must be within five minutes of the store's clock;
  - it cannot be earlier than any stored retrieval of its stock;
  - it must come before its first target session opens (09:30 New York).
- Scoring judges what a forecast could have seen by the later of its
  `created_at` and the time the store recorded it. A target whose sessions
  had opened by then is never scored.
- A copy of the ledger head is kept outside the database. It shows a removed
  last entry, which the hash chain alone cannot.
- Scores are appended as sessions close; the original probabilities are
  never rewritten.
- A future PPO policy may read these records only through an approved
  point-in-time reader. Hindsight-reconstructed forecasts are never treated
  as historically available.
- The PPO reward and the portfolio simulator are unchanged.

### Implementation (PR 3)

**The private store's record tables** (write-once by triggers):

- `forecasts` holds the forecasts.
- `scores` holds one row per forecast and target once its session closes:
  the category, the realized return, and the Brier score, log loss and RPS
  of the stored probabilities.
- `job_runs` holds each daily run: its status, the expected session and
  the per-stock freshness.
- `ledger_entries` is a hash chain over every forecast, score and job run.
  - Each entry is the sha256 of its canonical fields, names the previous
    entry's hash, and covers the record's own sha256.
  - `verify_ledger` re-checks the sequence, the chain, each entry against
    its record, and that no record lacks an entry.

**Scoring** (`scoring`):

- A target is scored only when every session it needs was **absent** from
  the store when the forecast was created, so it was first downloaded
  afterwards. This holds even if a later re-download supplies the session
  again. Scoring uses the forecast's own band and the exchange calendar, so
  a gap is never read as one session.
- A target whose session has not closed is pending, and nothing is written
  for it.
- A gap or an outcome seen before the forecast is recorded as unscorable,
  with the reason.

**The daily job** (`job`, `scripts/stock_lab_daily.py`):

- **Expected session:** the last NYSE session (rules calendar, passed in by
  the script) whose close is at least 20:00 New York time old.
- **Freshness:** a stock whose newest close is not the expected session
  keeps its last valid forecast, labelled stale on the page. Nothing is
  invented.
- **Without `--record-real`:**
  - the real job scores whatever is stored;
  - it records itself as `skipped` and not yet authorized, with the reason,
    and writes its Model Health input. Health reads that as not activated
    (insufficient evidence), never as a degraded job. A job that stops
    running still fails as overdue.
- **The price update** is the separate, owner-run
  `scripts/stock_lab_fetch_recent.py --authorized --store` (decisions A1 and
  A2), run first each evening. The daily job refuses `--authorized` and
  `--inference-window`.
- **The real path** (`stock_lab.evening`, built; one model activated 2026-10-20, A4) is
  described in "The real evening job" (section 7).

**The synthetic replay** (`replay`, `--demo-replay N`):

- The demo stocks' last N sessions are revealed one simulated day at a time.
- Models are fitted with every outcome censored at the replay's start.
- Each day: the close arrives, closed targets are scored, and the day's
  forecasts are made.
- The store runs on the replay's simulated clock, so every record carries
  the simulated time at which it was written, and nothing is backdated.
- This demonstrates the forecast-record-score loop without any real stock
  and without any model seeing what it is scored on.

**Model Health** (`monitoring`, `stock_agent.health.stock_lab`):

- The job writes `~/stock-agent-private/health_inputs/stock_lab_monitor/<run_id>.json`
  (envelope v1.0, write-once; outside every checkout since 2026-10-21, so the
  evening can run from a runner clone; the synthetic demo replay writes to
  `reports/health_inputs/`) with the fields Health reads.
- Five checks join the private monitor:
  - `data.stock_lab_freshness`;
  - `forecast.stock_lab_job`;
  - `forecast.stock_lab_scores` (insufficient evidence below 250 scored
    targets; degraded when worse than the baseline; tracking only; degraded
    too when the real job reports a matured target without a stored outcome,
    `missing_outcomes`, or an outcome it could not read, `outcomes_refused`);
  - `forecast.stock_lab_drift` (PSI, degraded above 0.25);
  - `forecast.stock_lab_ledger`.
- Their runbook entries are in `docs/operations/runbook.md`. Public CI
  reads only the synthetic fixtures.

**Scheduling** (`scripts/stock_lab_schedule.py`):

- It prints the unattended evening agent, `local.stock-lab.daily`, for 20:30
  New York time on weekdays, and with `--run-sh --model ID` its `run.sh`. The
  owner installs both (`docs/operations/runbook.md`, "Stock Lab unattended
  evening"). The agent starts `/bin/zsh -f
  ~/stock-agent-private/stock_lab/run.sh`. That script runs only from a clean
  `main` checkout, and each step under `env -i`:
  1. `scripts/stock_lab_fetch_recent.py --authorized --store --keychain`;
  2. only when that succeeds, `scripts/stock_lab_daily.py --record-real
     --model ID`.

  A failed price update stops the evening: no forecast is recorded, and none
  is backfilled.
- **The key at run time.** `--keychain` reads the Tiingo key from the macOS
  Keychain (`stock_agent.data.vendors.keychain`: service
  `stock-agent-tiingo`, account `stock-lab`). It runs `security
  find-generic-password ... -w` without a shell, inside the fetch's own
  process, once, just before the first request. The key never enters an
  environment variable, a command-line argument, a file, a log or a message.
  An item that cannot be read refuses the fetch before any request (exit 2).
- **Missed evenings.** Each real run records the prospective record's
  evenings since the first real forecast (`stock_lab.job.evening_record`,
  the monitor file's optional `evenings`). An evening is due once the next
  session opens. A due evening with no real forecast is missed, a permanent
  gap. Model Health degrades `forecast.stock_lab_job` (`evening_missed`)
  while a miss is among the last 10 due evenings, and lists older ones in its
  detail.
- **Notifications** (added 2026-10-21). Local macOS notifications, shown by
  `/usr/bin/osascript` and never sent off the Mac, so the owner learns of a
  failure without reading logs. Each is one of a few fixed sentences, never
  a value, price, forecast or key, and none ever changes a job's exit
  status.
  - `run.sh` shows one when it refuses the checkout, when the price update
    fails or refuses, and when the forecast step fails.
  - A run that never starts cannot notify. The optional morning check
    covers that (`--morning-check [--time HH:MM]`, 08:00 New York on
    weekdays by default, always before the 09:30 open):
    `scripts/stock_lab_morning_check.py` reads the store read-only
    (`stock_lab.morning`). When the prior session has no real forecast, it
    says the evening can still be recorded before 09:30, or, after the open,
    that it is a gap. It reads the same recorded evenings as
    `evening_record` (`stock_lab.job.recorded_evenings`) and the briefing's
    session timing, so weekends and holidays are never checked (the next
    session morning checks the same evening in time), and a late check from
    20:00 is of the evening before that day, never tonight's.
- **The wake.** The printer writes an optional `sudo pmset repeat
  wakeorpoweron MTWRF 20:25:00` (in the Mac's zone) to stderr, never runs
  it. pmset keeps one repeating pair, so it replaces the briefing's 08:25
  wake. The evening's wake matters more: a late briefing can still be read,
  but a missed evening cannot be recorded.
- **The time zone.** launchd takes the Mac's local time with no time zone,
  so the printer converts 20:30 New York to the zone given and refuses a
  zone whose clock changes on other dates than New York's, or never, as for
  the briefing (section 9).
  - In such a zone a fixed local time is 20:30 New York for part of the
    year only. It runs later for the rest: 21:30 in Europe/London (the
    weeks between the two countries' changes) or Asia/Tokyo (all summer),
    22:30 in Australia/Sydney.
  - It can also run earlier: in Antarctica/Troll, at 19:30, before the
    close counts as available (20:00).
  - A late run is not wrong: the job takes its session from the wall
    clock, and Model Health allows six hours. The printed file would
    still not do what it says, so it is refused.
  - US and Canadian zones that follow New York's changes are accepted. A
    test reads back the agent printed for every zone in the time-zone
    database: refused, or 20:30 New York on each day it would fire in 2026.
- **Private files.** It writes with a private umask (files 0600), like the
  briefing's agent and the Model Health agent. The job already creates
  the store, the monitor files and the kept ledger head 0600; the umask
  covers anything else it creates.
- **Installing.** First create the private folders, each with mode 700:
  `mkdir -p -m 700 ~/stock-agent-private ~/stock-agent-private/stock_lab
  ~/stock-agent-private/stock_lab/logs` (one command). `-m` sets the mode
  of the folders named only: one created on the way gets 0755, and the
  store refuses a `stock_lab` folder others can read.
- **An earlier printout.** An agent installed from one keeps its old
  settings until it is printed again, booted out and bootstrapped. An agent
  printed before 2026-10-20 ran only `stock_lab_daily.py` without
  `--record-real`; replace it.
- With `--briefing` it prints the 08:30 premarket-briefing agent, and with
  `--email-dry-run` a description of an email adapter (section 9).
- The job itself skips holidays through the calendar.

**The page:**

- The job's last status, a stale warning when the latest run expected a
  later session than the forecast, and the forecast history.
- The history shows each stored forecast's next-session probabilities as
  recorded, beside what happened and its score.

## 7. Owner decisions (recorded 2026-10-16)

The owner answered A1-A6 on 2026-10-16. They are given here in paraphrase,
with the conditions that bind the implementation. A7 was raised afterwards
by the simulated size checks (section 5) and was confirmed on 2026-10-19.

| Id | Decision | Conditions | Status |
|---|---|---|---|
| A1 | Tiingo Starter and the private watchlist store: approved | **Staged:** inspection and persistence are separate. The first live operation is the read-only command below, run only after the owner confirms the token is configured and authorizes that exact command. The 30-ticker cap holds. No vendor-adjusted fields, no unauthorized history, no redistribution. After a successful coverage and corporate-action audit, storage (`--store`) is a separate approval. | awaiting the token and authorization of the read-only command |
| A2 | Trailing 252-session inference window: approved in principle | Fit, tune, select and calibrate only on eligible pre-2025 data. Recent observations from the protected period serve only as inputs to a previously frozen model, kept in a separately controlled prospective/inference store. Protected outcomes are never used for retrospective selection, training or backtest claims. The sealed ETF period and its protections are unchanged. Strict as-of and retrieval-time records. A scoped authorization, not an unlocking of the 2025+ holdout. | built on fixtures: the inference store, the frozen-model reader and the recent-window fetch (below); nothing fetched or stored |
| A3 | Exploratory walk-forward evaluation of the saved watchlist on pre-2025 history: approved | Marked hindsight-selected and non-confirmatory. Every challenger is compared with B0/B1, with proper scores, calibration, stock-level coverage, uncertainty and multiple-comparison treatment. No claim of profitable trading or reliable direction. | runnable after storage (`scripts/stock_lab_evaluate.py --real --authorized-evaluation`); `--freeze` also registers the frozen model (below) |
| A4 | The write-once forecast and scoring protocol: approved | Activation needs successful data validation, a frozen registered model, verified decision timestamps and a successful end-to-end dry run. No backdating, rewriting of probabilities, or scoring of outcomes that existed before a forecast. Daily updates record skipped tickers, stale observations, model failures and missing outcomes in Model Health. Separate from ETF OD16 and PPO promotion. | protocol built; model `2b6cd0c1a52f5ecf` activated 2026-10-20 |
| A5 | Version-1 definitions: approved | Daily band ±0.5%. Five-session band ±0.5% × √5 ≈ ±1.12%, shown as different. A Stable session ends a streak by default, and the pause rule is identified whenever used. The bull-trap setup (10% drawdown, 5% rebound, prior 20-session high, 10-session window) may be used for research. **The failure margin f awaits confirmation** (formula and example in section 5) before real-data risk flags. Price returns stay distinct from total returns, and ex-dividend moves are identified. | f to confirm |
| A6 | Recent observed prices on the private dashboard: supported | Only after the retrieval and storage permissions. An isolated descriptive reader that cannot expose protected outcomes to development, training, tuning or backtesting. Recent observed history is kept distinct from research evidence, and current forecasts from completed outcomes. The holdout policies are not modified. | built on fixtures: the descriptive reader and the Recent prices tab (below), which shows an empty state until something is stored |
| A7 | Which test decides "beats B1": the fold-block t at a one-sided 2.5% family level, confirmed 2026-10-19 | t(folds − 1) on the yearly fold means; Holm across B0, B2, B3 and M1 at 2.5%; simplest-first steps (B0, B3, B2, M1) at 2.5%; B1 stays the default. Newey-West (lag 10) and the block bootstrap are reported only | implemented (`evaluation.DECISION_RULE`, evaluation schema 1.3); the evidence is in `docs/research/stock_lab_a7_decision.md` |

### A1 step 1: the read-only audit (awaiting the owner's authorization)

```bash
.venv/bin/python scripts/stock_lab_fetch.py --authorized
```

- **Before it:** save up to 30 tickers in the Lab's watchlist, set
  `TIINGO_API_KEY` in the shell, and run the command without
  `--authorized`. That prints the exact plan (the ticker list and every
  request URL) and makes no request.
- **Requests, in order, at least 2 seconds apart, at most 1 + N for N
  tickers:**
  1. `GET https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip`,
     with no key.
  2. For each watched ticker the listing classifies as a supported US stock:
     `GET https://api.tiingo.com/tiingo/daily/<TICKER>/prices?startDate=2003-01-02&endDate=2024-12-31&format=json&resampleFreq=daily&columns=date,open,high,low,close,volume,divCash,splitFactor`,
     with the key only in the Authorization header.
- **Output:** the listing's size and each ticker's classification, then each
  ticker's coverage and corporate-action audit (`stock_lab.audit`):
  - sessions served and missing;
  - null fields and low/high sanity;
  - every split with its raw and split-consistent close ratios and a
    verdict;
  - ex-dividend dates and the largest yield;
  - large moves without a split;
  - unavailable returns, and a ready / review line.

  A final summary lists which tickers are ready. No price series and no key
  is printed.
- **Local side effects:** no record is written. Reading the watchlist may
  create SQLite's `-shm`/`-wal` companion files beside the store.

### A1 step 2: storage (a separate approval, after a good audit)

```bash
.venv/bin/python scripts/stock_lab_fetch.py --authorized --store
```

- It repeats the same requests and the audit.
- It writes the listing, the checks and one write-once batch per ticker
  **whose audit is ready**. A ticker that needs review is reported and not
  stored.
- There is deliberately no override yet. A large move without a split can
  be a real move or an unrecorded split (an unrecorded 2-for-1 split reads
  as a fall of about 50%), and only the owner can tell which. If the audit
  holds any ticker for review, the storage request will list each one with
  its finding and propose how to handle it: drop it from the watchlist, or
  store it with the finding recorded.

### A2 and A6: the inference store and recent prices (built, not run)

Built on fixtures only. No request has been made and nothing is stored.

**The inference store** (`stock_agent.stock_lab.inference`):

- **Location.** A private SQLite file, `stock_lab_inference.sqlite3`, in
  its own directory: `STOCK_AGENT_STOCK_LAB_INFERENCE_HOME`, or by default
  `~/stock-agent-private/stock_lab_inference/`.
  - A location inside any Git working tree is refused, even under an
    ignored `data/raw/` (stricter than the research store).
  - A directory that holds the research store is refused.
  - The file must have exactly this name.
- **Identity.** The SQLite header carries the store's own
  `application_id`.
  - The research store refuses any file with a non-zero id. It can neither
    read the inference store nor add its own tables to it.
  - The inference store refuses any file without its id.
- **Write-once.** One batch per retrieval of one ticker, never rewritten,
  with its rows. Each batch records:
  - `retrieved_at`, when the provider served it;
  - `recorded_at`, when the store wrote it;
  - the requested sessions, the calendar id, the content hash and the data
    class.
- **What a batch must be** (each refusal has a machine-readable reason):
  - inside the protected period (sessions from 2025-01-01; older sessions
    belong to the research store);
  - exactly one row per calendar session of its request (a session the
    provider did not serve is a row without prices, so a gap stays a gap);
  - retrieved no earlier than its last session's close was available, at
    20:00 America/New_York (the daily job's convention, assumed);
  - recorded within an hour of its retrieval, and never before it. A batch
    cannot claim an earlier retrieval than it had.

**The frozen-model reader** (`inference.window.load_inference_window`):

```python
window = load_inference_window(
    store_path, "NVDA", "2026-10-16", 252,
    retrieved_by=decision_time, sessions=nyse_rules.sessions, calendar_id=CALENDAR_ID,
)
```

- **The window.** Exactly the 252 sessions ending at the as-of close, both
  ends included: 252 closes and 251 returns. That is the longest feature
  window (`features.LONGEST_WINDOW`).
- **Retrieval cut-off.** Only batches retrieved and recorded by the
  decision time are read. A re-run with the same decision time
  reproduces the inputs.
- **Vintages.** Per session, the newest eligible retrieval wins, and
  revisions are listed.
- **Refusals.**
  - a lookback other than 252;
  - a window starting before 2025;
  - a decision before the as-of close was available;
  - a short window (no padding);
  - a missing as-of close;
  - a batch on another calendar;
  - any integrity failure.
- **No retrospective decisions.** A decision at or after the next
  session's open is refused. By then the as-of close is history, and the
  "forecast" would be a reconstruction scored on outcomes that already
  exist. The reader reads no clock, so the decision time is the caller's
  claim. Two other rules keep a replay from becoming a backtest:
  - only reviewed modules may import the reader; the A4 forecast job must
    take the decision time from its clock;
  - the forecast record refuses a `created_at` away from the store's clock.

**The descriptive reader** (`inference.recent.read_recent`, decision A6):

- It returns up to 60 of the latest observed sessions: close, a
  split-consistent level for the chart, the change from the previous
  session, notes (gap, ex-dividend, split, large move, revised) and the
  lineage.
- It computes no category, feature, streak, probability or score on
  protected prices.
- Given the latest forecast's as-of session and its five target sessions,
  it marks those targets as that forecast's outcomes. Later sessions are
  not marked, since no forecast covers them.

**The page.** A **Recent prices** tab, apart from History:

- **Until something is stored:** an empty state.
- **For demo stocks:** a "real stocks only" state.
- **Once stored:** a badge ("Observed prices, not research evidence"), the
  last close, the sessions shown, the retrieval time, a chart and a table.
  The sessions a forecast covers are named as its outcomes.
- **The research history** still stops at 2024-12-31.

**Isolation** (`tests/test_stock_lab_inference_boundaries.py`):

- **Imports.** Only two consumers may import the package.
  - The Lab page may import only `read_recent` and `RecentPrices`.
  - The recent-window fetch script may import only the writer, the
    frozen-model reader and five schema names.
- **Research code.** No other module or script imports it. None names its
  module path, file, directory, environment variable or header id.
- **Runtime.** The research, evaluation, training, tuning and backtest
  entry points load nothing from it, which is checked in a subprocess.
- **Readers and writer.** The readers never load the writer, and only the
  writer writes.

**The recent-window fetch** (`scripts/stock_lab_fetch_recent.py`): off by
default, never run.

- It reads the watchlist and the listing checks from the research store,
  which it never writes.
- Exact flags only: abbreviations are refused.
- A ticker is stored only when its audit is ready and the provider served
  every session of the window. A session not served is held for the
  owner's review: a split or dividend on it would be invisible to the
  audit, and there is deliberately no override yet.
- Printed paths show the home folder as `~`.

| Command | What it does |
|---|---|
| `.venv/bin/python scripts/stock_lab_fetch_recent.py` | prints the plan: tickers, window, request URLs, store path; no request |
| `... --authorized` | one request per watched ticker whose research listing check is supported (at most 30), for the 252 NYSE sessions ending at the last complete session; prints the audit; writes nothing |
| `... --authorized --store` | also writes each ready batch to the inference store and reads its window back |

**Before the first real inference, the owner must:**

1. Run the A1 research steps: the read-only audit, then storage. The
   recent-window fetch requests only tickers that the research store's
   listing check marks supported.
2. Authorize the recent-window read-only command, which requests
   2025-and-later prices for AMD, MRVL, MU and NVDA. Then approve its
   storage, a separate step.
3. Confirm the conventions this store assumes:
   - the window is 252 closes ending at the as-of close, not 253;
   - a close is final at 20:00 New York time;
   - a decision must come before the next 09:30 open;
   - batches are recorded within an hour of retrieval.
4. Decide how a frozen model's state uses the window. Decided on 2026-10-19:
   `frozen_long_run` (below, "The frozen, registered model").
5. Decide where scoring reads its outcomes. Decided on 2026-10-19: the
   inference store, through the outcome reader (below, "The real evening
   job"); scores stay in the existing write-once ledger.
6. Approve A4 (A7 was confirmed on 2026-10-19): the frozen registered model,
   the verified decision timestamps and the end-to-end dry run. Only A4
   activates real forecasts: `--record-real` refuses a model the owner has
   not activated.
7. Keep A3 and A7 choices independent of the recent prices. The Recent
   prices tab shows protected-period outcomes to the owner, and no code can
   stop what is seen from shaping a choice made afterwards. Settle the
   model and the test, or record what was viewed and when, before
   reading the tab.


### The frozen, registered model (built, not run)

`scripts/stock_lab_evaluate.py --real --authorized-evaluation --freeze` evaluates the watchlist (A3)
and then freezes and registers one model (`stock_agent.stock_lab.registry`; the record's contract is
`stock_lab.frozen_model`). Nothing is fitted, tuned, selected or calibrated afterwards. No real model
exists yet.

**What is frozen.** One record holds everything a forecast needs. The record is checked against its
id (the sha256 of its content) when it is stored and again whenever it is read. Any change, for
example another estimator, half-life, state rule or environment, makes a new model with a new id.

- The estimator, and the B1 benchmark that is recorded beside every forecast.
- The settings, the band, B1's half-life (the fold majority), and the feature version.
- The parameters the estimator needs:
  - B0's long-run mix per stock, for the daily and the five-session categories;
  - B2's tilt;
  - B3's error distributions;
  - M1's coefficients and preprocessing.

  A part the estimator does not use is left out.
- Per stock, what followed rising and falling runs of 1 to 7 sessions in development history.
  The forecast record cites this, because on a 2025+ window every such event would be censored.
- The decile edges and shares of the drift inputs, for Model Health.
- The data fingerprints:
  - the evaluation's panel hash;
  - each stock's sessions, rows and content hash, over every stored field (evaluation schema 1.2
    records them);
  - the stored batches that supplied development sessions.
- The environment (Python's minor version, numpy, pandas and scikit-learn), the commit, and
  whether the checkout had uncommitted changes. A real model is frozen only from a clean checkout.
- A parameters hash, shared by two freezes of the same model.

**The rules.**

- **Real data stops at 2024-12-31.** A protected-period session is refused (`protected_period`),
  never trimmed. The evaluation must have censored its outcomes there.
- **The histories and settings must be the evaluation's own.** That means the same stocks, the
  same bars field by field, the same panel and the same settings. The model names the stored
  evaluation by run id and content hash.
- **Only B1 until A7.** Until the owner confirms which test decides "beats B1", only B1, the
  pre-registered default, is frozen. The evaluation's own selection is recorded and reported.
  A challenger is frozen only when the evaluation records an owner-confirmed decision rule
  (`registry.CONFIRMED_DECISION_RULES`, empty today). Then it must be exactly that selection.
- **Round trip.** The stored record is loaded back. It must reproduce the fitted model's
  probabilities, for its estimator and for B1, at every stock's last development session, within
  1e-12.
- **Loading refuses a mismatch.** A different feature version or environment is refused, and so is
  a stock outside the model.
- **The store re-checks.** The record's own check repeats the real-data and A7 rules, so the store
  refuses a hand-made record whose id was recomputed after an edit. The store also compares the
  record with the stored evaluation (settings, stocks, panel and history hashes) and refuses a
  creation time before that evaluation or after its clock.

**The state rule (decision A2, item 4: `frozen_long_run`, the owner's choice on 2026-10-19).** The parameters never change. A
frozen model's running state is a function of the frozen parameters and one forecast's inputs:

| Rule | B0, the long-run mix | From the 252-session window |
|---|---|---|
| `frozen_long_run` (the owner's choice; every real model) | frozen at 2024-12-31 | B1's recency-weighted counts, the volatility, the run, every feature |
| `window_only` (synthetic checks only) | recomputed from the window: a one-year mix | the same, and B0 |

- **What the window leaves out depends on the half-life.** The half-life is the folds' majority
  choice from {21, 63, 126, 252}. The weight beyond a 252-session window is λ²⁵² with λ = 2^(−1/H):
  1/16 at H = 63, but 1/4 at H = 126 and 1/2 at H = 252. The record keeps that share
  (`window_tail_weight`). At H = 252, B1 from a window is a materially different estimator from the
  evaluated one, under either rule.
- **The development state check.** It is reported only and never used to select. It is
  computed on development data only, on each stock's last 252 development sessions:
  - B1 from the trailing window, under each rule. For `frozen_long_run` the prior is the
    long-run mix as of that session.
  - Compared with B1 from the full history.
  - Scored by the next session's RPS where it matured, with the mean and largest probability
    differences.

  `scripts/stock_lab_evaluate.py --freeze` prints it, and the record keeps it (`state_check`).
  The two rules count as two variants tried.
- **Why `frozen_long_run` was recommended.** Its prior (κ = 40) keeps pointing at the long-run mix the
  development evaluation used, not at a one-year mix. It needs no other stock's window: under
  `window_only`, B0 pools every model stock's window, so one stale stock blocks every forecast
  (`panel_stocks_mismatch`). Whether either windowed B1 is close enough to the evaluated one is
  what the state check measures; it is not assumed.
- Rejected option: continuing the development state through every protected session. That would
  read more than the 252-session window, which is outside A2.
- The rule is recorded with the model. A real model under another rule is refused
  (`state_rule_not_approved`).


### The real evening job (built; one model activated 2026-10-20, A4)

Built on fixtures. On 2026-10-20 the owner activated one frozen model,
`2b6cd0c1a52f5ecf` (B1, half-life 252, `frozen_long_run`), for AMD, MRVL, MU and
NVDA, after storing the audited history and recent window and freezing that
model (`docs/PROJECT_STATE.md`). Its first real forecast follows the first
successful dry run after 20:00 New York. The owner's choices of 2026-10-19 (below and in
`docs/PROJECT_STATE.md`): B1 is recorded beside every forecast, and outcomes
are read from the inference store.

**One evening** (`scripts/stock_lab_daily.py --record-real --model ID`, after
`scripts/stock_lab_fetch_recent.py --authorized --store`):

1. **Activation.** The registered frozen model must be activated by the
   owner (`stock_lab.evening.ACTIVATED_MODELS`, empty: the owner's explicit
   go, recorded with their words in a reviewed change). It is loaded and
   verified (`registry.load`). `--dry-run` does everything below and records
   nothing; activation needs one successful dry run.
2. **Scoring first.** Every real forecast not yet fully scored is scored on
   outcomes from the inference store (`inference.outcomes.read_outcomes`):
   - only the as-of session and the five target sessions are read, from
     batches retrieved and recorded by the job's clock;
   - a target needing a session that any batch had already covered when the
     forecast was known is never scored (`scoring.score_forecast`);
   - a target whose session has closed by the expected session but has no
     stored outcome yet is counted as missing, for Model Health.
3. **Forecasts, per watched stock:**
   - skipped when the stock is not in the frozen model, or already has a
     forecast for the expected session (a second run never recalculates);
   - stale when its newest stored close is older than the expected session:
     its last forecast stays, labelled stale, and nothing is invented or
     backfilled;
   - refused, with the reader's reason, when its 252-session window cannot be
     read as of the job's clock (for example after the next 09:30 open, so a
     missed evening is never filled in later);
   - otherwise the frozen model's forecast from that window alone
     (`frozen_forecast.forecast_record`), recorded write-once.
4. **The job run and Model Health:** per-stock results, each stock's newest
   stored close (freshness), drift of the windows' inputs against the frozen
   development reference, and the scoring counts.

**What each real forecast records** (forecast schema 1.1):

- the frozen estimator's five daily distributions and the separate
  five-session total, at the model's bands;
- the **B1 comparator**: B1's forecast on the same window, sessions and
  bands, inside the same write-once record and ledger entry. It is never a
  forecast of its own, never live, never in a rolling score or calibration
  bucket, and never changes model-selection evidence; it exists only so that
  the forecast can be compared with B1 on exactly matched outcomes;
- the **state it was made in**: the run, momentum over 5 and 20 sessions,
  volatility over 20 sessions and its EWMA, and the volatility regime against
  the window's own terciles (descriptive, from the window through the as-of
  close);
- what followed the current run in development history (frozen with the
  model); no trap signal while the failure margin f awaits the owner (A5);
- the model version, the batches, the decision time and the window's gaps
  and revisions, and the limitation that an end-of-day forecast reflects no
  overnight news and no premarket move.

**Isolation.** `stock_lab.evening` is the one reviewed module that reads the
inference store for forecasts and scores, with named imports only
(`tests/test_stock_lab_inference_boundaries.py`). The daily script imports it
only on its real path, so loading the script still loads nothing from the
store.

**The scheduled real evening (prepared 2026-10-20, not installed).** The
printed evening agent runs the recent-window fetch with the Tiingo key read
from the macOS Keychain at run time, then `--record-real` for the activated
model (section 6, Scheduling). Still the owner's: adding the Keychain item,
installing the agent and the optional wake.


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
  - A test's size is its false-positive rate when the null is true. It is
    measured by simulation on panels where the null holds by construction,
    not assumed from the test's asymptotics.
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
  discrete-time survival hazards; Hansen's SPA test for many comparisons;
  fixed-b asymptotics for Newey-West, which explain why a short lag
  over-rejects.
- **The inference window (A2) in point-in-time terms.**
  - A decision at time d uses a batch only if its `recorded_at` ≤ d. The
    window is a function of (store, ticker, as-of, d), so the same d always
    gives the same inputs.
  - Bounding d to [close available, next open) makes "no retrospective
    decision" a rule of the reader. A loop over past as-of dates finds only
    the batches recorded inside each one's decision interval, so it can
    rebuild what a real decision would have seen and nothing later.
  - Example: the as-of is Thursday 2026-10-15. A batch retrieved at 20:30
    and recorded at 20:31 New York time is readable from 20:31 until
    Friday 09:30. A revision retrieved at 22:00 is invisible to a decision
    at 21:00.
  - What could mislead: a vendor that revises a close after 20:00. The
    reader then lists the session as revised in a later decision's window;
    it never rewrites the earlier window.
- **Exercise.** Show that under the "breaks" rule the per-session view
  reports, at each run's last session, that run's length. That is the
  property tested in `test_every_run_row_matches_the_per_session_view_at_its_end`.
  Then explain why the "pauses" rule cannot have it for short Stable
  stretches.

## 9. The daily premarket briefing (briefing PR 1, offline)

Owner brief: 2026-10-18. The goal is a trustworthy morning briefing that
measures its own reliability, not another prediction dashboard. Briefing
PR 1 is offline: it reads stored records, renders them and previews
locally. It sends nothing, schedules nothing and reads no credential.

**What it reads** (`stock_lab.briefing.report`, read-only):

- **Records.** The watchlist, the recorded forecasts (re-checked against
  their ids), their scores, the evaluation runs, the daily job's runs and
  the ledger check.
- **Read-only access.** The store is opened through `ReadOnlyRecords`,
  which has no write method, and every connection uses SQLite `mode=ro`.
  A test shows the database file's bytes are unchanged afterwards.
- **What it never does.** It never retrains, recomputes a forecast, writes
  to a ledger or runs the evening job.
- **As known when prepared.** A record counts only if it existed then: a
  forecast created and recorded by then, a score written by then, a job
  run finished by then. A recalculation replaces a forecast only once the
  recalculation itself exists.
- **One data class.** A briefing is wholly real (the watchlist, minus
  SKHY and SNDK by the owner's choice) or wholly synthetic (the demo
  stocks). The two are never mixed.

**Timing** (`stock_lab.briefing.timing`):

- **Target.** 08:30 America/New_York on an NYSE session morning, from the
  IANA zone: 13:30 UTC in winter, 12:30 UTC in summer.
- **States.** `premarket` (before 09:30 on a session day; the only state a
  scheduled delivery may use), `after_open`, `advance` (from 20:00, for
  the next session) and `closed_day` (a weekend or holiday, skipped unless
  asked for).
- **The data.** The briefing reads the prior session's forecast. The days
  between that close and the market date when the market was closed are
  named. A Monday briefing says plainly that Friday's end-of-day data holds
  nothing from the weekend.
- **Calendar.** The exchange calendar is passed in by the script, so the
  calendar module is unchanged.

**Per stock:**

- **Statuses.**
  - `fresh`: the stored probabilities are shown exactly as recorded.
  - `stale`: the newest forecast is older than the prior session. It is
    named, and its probabilities are not shown as today's.
  - `unavailable`: no forecast, a refusal, a failed ledger check, or a
    first target session that is not the market date. The reason is
    always given.
  - A probability is never filled in.
- **Contents.**
  - the next session's three probabilities, with the development track
    record of the same range;
  - the five sessions and the separate five-session total;
  - the change since the prior forecast, only at the same band;
  - the current run, the reversal statistics with counts and 90% Jeffreys
    intervals, and the bull-trap signal, all from the forecast record;
  - the close and lineage.
- **Point-in-time limits.**
  - Momentum, volatility and the volatility regime (descriptive) and the
    bear-trap check are computed from stored prices through the as-of
    close. This happens only for synthetic stocks or sessions through
    2024-12-31.
  - For a real stock after that, decision A6 forbids features on protected
    prices outside the frozen forecast job. The briefing therefore shows:
    - the state that job recorded with the forecast (momentum, volatility,
      and the regime against the forecast's own 252-session window), never
      recomputed;
    - the latest observed close, its session and its retrieval time, from
      the inference store's isolated descriptive reader. Only batches
      retrieved and recorded by the preparation time are read, so a preview
      for an earlier morning never shows a later close. The owner-run script
      passes the reader in; the report module never reads that store.
  - Research-store prices of 2025 or later are never shown as a real
    stock's current close.
  - Every briefing states that a forecast is made after the prior close
    from end-of-day prices, and reflects no overnight news and no premarket
    move.
  - Trap flags on real data are withheld until the failure margin f is
    confirmed (A5).

**Reliability** (`stock_lab.briefing.reliability`). Development and
prospective evidence are kept apart and never pooled across horizons,
stable bands, estimators or model versions.

- **Development.**
  - The stored evaluation is read as stored: the forecast estimator's
    scores against B1's on the same rows (RPS, Brier, log loss), the stored
    comparison (reported only), and the reliability bins.
  - A bin with fewer than 20 distinct dates is labelled insufficient
    evidence.
- **Prospective.** The recorded forecasts and their scores.
  - Per target definition and category, equal-width buckets give the mean
    predicted probability, the observed frequency, the count, the distinct
    forecast dates and a date-clustered 90% interval.
  - The record covers every stock with scored forecasts, including those
    no longer on the watchlist. Dropping a stock never removes its track
    record, which would be survivorship.
  - A bucket with fewer than 20 independent units (the dates, or the
    dates / 5 for the total) or an effective count below 20 is insufficient
    evidence: counts are shown, frequencies are not.
  - A rolling 63-date summary carries a small-sample warning, and Model
    Health's threshold (250 targets over 50 sessions) gates any summary.
- **The interval.** The observed frequency p = hits / rows gets a
  cluster-robust variance over forecast dates g in time order. Here
  u_g = y_g − p·m_g, with y_g the hits and m_g the rows of date g:

  ```text
  Var(p) = C / (C - 1) * max(G_0, G_0 + 2 * sum_{l=1..L} G_l) / n^2,  G_l = sum_g u_g u_(g+l)
  n_eff  = min(n, p (1 - p) / Var(p))
  ```

  - **Lags.** L is 0 for a single session's target. Within one horizon,
    each date is settled by a different session.
  - **The total.** L is 4 for the five-session total, whose dates up to
    four sessions apart share sessions. This is a truncated-kernel HAC
    variance, floored at the lag-0 term.
  - **The interval.** It is the Jeffreys interval at n_eff, which lies
    between the dates and the rows. When p is 0 or 1, or the variance is
    zero, n_eff falls back to the independent units (the cautious reading).
  - **Daily coverage.** In a seeded simulation (60 dates, ten stocks sharing
    each date's probability of 0.40 or 0.90), the interval covered the true
    0.65 in 90.8% of 2,000 panels at a nominal 90%. Counting rows as
    independent covered 65.5%.
  - **Total coverage.** For overlapping five-session totals (five stocks at
    correlation 0.5, 250 dates), the lagged variance covered 91% of 400
    panels. Without the lag terms the coverage was 58%; non-overlapping
    blocks of five dates would under-state the variance by about a third.
  - **Small panels.** With two to five stocks a day, 25 dates and a 10%
    rate, the coverage was 91%. The tests rerun smaller versions of these
    simulations.
  - **What it does not cover.** The interval reflects date-to-date noise in
    the period observed. If the true frequency moves with long calm or
    volatile stretches that the forecast ignores, the interval is too
    narrow, and a short record cannot show that.
- **Matched comparison with B1.** It uses only the same stock, as-of
  session, horizon, band and model version, with equal outcomes.
  - The difference is the mean of per-date differences.
  - Per-date differences are serially dependent, as the A7 packet measured.
    So the standard error comes from blocks of 20 consecutive dates, counted
    from the first matched date. A shorter remainder joins the last whole
    block, so a block holds 20 to 39 dates (the owner's choice, 2026-10-20).
    With T dates in K blocks, block means m_k of n_k dates and the mean
    difference d̄:

    ```text
    SE² = K / (K − 1) · Σ_k n_k² (m_k − d̄)² / T²
    ```

    This is the cluster-robust form of the next-session calibration variance,
    with blocks of dates as the clusters. With whole blocks it is the spread
    of the block means over √K. The interval is t on (K − 1) degrees of
    freedom.
  - **Coverage at a nominal 90%** (`stock_lab.comparison_coverage`; record
    `docs/research/stock_lab_comparison_coverage.json`, schema 1.1). The
    simulation ran 10,000 records per stylized cell and 4,000 per score
    cell, at 100 to 500 dates. The tests rerun a smaller version. Schema 1.1
    added the Brier and log loss cells, the break series and the flag rates
    below. The earlier cells reproduce exactly.
    - **Short dependence: 88.4–90.4%.** This covers independent dates, AR(1)
      persistence up to 0.5 with Student-t shocks, and panels of 2–5 matched
      stocks a date with cross-sectional correlation 0.5.
    - **RPS differences on simulated prices.** Four stocks, GARCH volatility,
      correlation 0.5 and B1 at half-life 63; the next session and the
      five-session total.
      - A symmetric placebo covered 89.2–91.3%.
      - Against the stock's own B1, a placebo and a volatility-scaled
        challenger covered 90.0–97.2%. That is conservative: B1 learns from
        the sessions it was scored on, so the differences revert over its
        memory.
      - The five-session totals' overlap (lag-1 autocorrelation 0.65) fits
        inside a block.
    - **Per score.** On the next-session symmetric placebo, RPS covered
      89.7–91.3%, Brier 88.4–89.4% and log loss 87.1–88.0%. The log loss runs
      low although its per-date differences are symmetric (skewness 0.00).
      Heavier tails are the likely cause; they were not measured. On the
      five-session total, the placebo's three scores covered 88.7–90.9%.
      - **Skewed log loss.** The volatility challenger's next-session log-loss
        differences are skewed (mean skewness +0.51). The interval covered
        89.9–91.7%, but its misses are lopsided: the truth fell below it in
        5.2–6.0% of records and above it in 3.1–4.2%.
    - **Longer persistence: too narrow.** Coverage was 86.0–86.9% at AR(1)
      0.75 and 79.2–81.5% at 0.9. The simulations are stationary, so, as for
      the calibration interval, they cannot rule out regime persistence in
      real differences.
    - **A mid-record shift.** Independent dates whose mean steps up by 0.5
      or 1 per-date standard deviation halfway through: the interval covered
      the record's mean in 97.9–100% of records. The shift widens it: it is
      too wide, not too narrow, for the period observed. But the difference
      was not steady, so the mean says little about the next stretch.
    - **Before the merge.** One date past whole blocks counted as a block of
      its own. At 101 dates the interval was up to 1.6 times as wide as at
      100, and covered 92.2–97.5%. With the merge it covers 89.5–93.8%.
  - **Reporting conventions (owner decision, 2026-10-20).** The date-clustered
    Jeffreys interval and this block t interval are reporting conventions
    only, never decision tests.
  - **The persistence check (owner decision, 2026-10-20).** Beside each
    score's SE, the same SE comes from blocks of 40 dates, the remainder
    again merged into the last whole block, once three such blocks exist
    (120 dates). The ratio SE40 / SE20 is near 1 for independent dates. It
    grows when per-date differences stay alike for longer than 20 dates, or
    when their mean shifts. Every block of 40 joins whole blocks of 20, so
    the ratio is bounded above: by √(5/2) ≈ 1.58 at 120 dates and 1.50 at
    200, and by up to 1.96 when a last block of 40 joins three blocks of 20.
    - **The flag.** `persistence_suspected` is raised when the RPS ratio
      exceeds 1.3. The Brier and log loss ratios are stored, never flagged.
      With a flag on each score, a caveat would appear far more often by
      chance, because the three ratios are only partly correlated. Below 120
      dates, or with a zero SE, the flag is not computed (`None`).
    - **The caveat.** The briefing adds one line after the comparison, only
      when the flag is on. For a ratio of 1.49 it reads: "Caveat (reported
      only): for RPS, the standard error from blocks of 40 dates is 1.49
      times the one from blocks of 20. The differences may stay alike for
      longer than 20 dates, or may have shifted during the record, so the
      90% intervals may be too narrow. This changes no number above and no
      decision." No number, interval or decision depends on it. The tests
      check that by building the same comparison with the flag forced on
      and forced off.
    - **False alarms** (ratio above 1.3, independent dates): 18.4% at 120
      dates, 9.2% at 200, 5.5% at 300 and 1.1% at 500. The symmetric RPS
      placebo gave 18.3%, 9.3%, 6.1% and 1.7% for the next session. The
      starting point of 1.2 gave 27.8%, 20.4%, 15.3% and 7.8% on independent
      dates. The ratio is noisy with few blocks of 40: two degrees of freedom
      at 120 dates.
    - **Detection.** The flag fired in 29.8%, 21.8%, 19.1% and 10.1% of
      AR(1) 0.9 records at 120, 200, 300 and 500 dates. That is a weak
      signal: its median ratio is only 1.07–1.15, because blocks of 40 miss
      most of that persistence too. At AR(1) 0.75 it fired in 2.8–22.1%,
      barely above the false-alarm rate. A mid-record shift is caught far
      more often: 29.5–79.6% for a shift of 0.5 standard deviations, and
      40.2–99.7% for a shift of 1 (98.6% from 300 dates). The
      volatility-scaled challenger, whose dependence is short, was flagged
      at the false-alarm rate (17.0% at 120 dates, 1.5% at 500).
    - **Reading it.** A flag is a reason to read the interval with caution,
      not evidence of a regime. No flag does not show that the differences
      are short-lived.
  - Fewer than 100 matched dates give no difference. Rows that differ in
    outcome are reported, not compared.
  - **B1's side.** Each forecast of schema 1.1 records B1 beside it (the
    owner's choice of 2026-10-19). B1's side of a scored target is that
    comparator's probabilities on the target's stored outcome, scored with
    the scorer's own functions (`reliability.comparator_observations`).
    It is matched only with the model it was recorded beside.
  - **What is shown.** The next session's comparison, for the current model:
    matched forecasts and dates, then RPS, Brier score and log loss
    differences with 90% intervals once 100 dates are matched ("t on K
    blocks of 20 or more dates"), and the persistence caveat when its flag
    is on. The briefing's JSON copy (briefing schema 1.1) also holds every
    score's SE from blocks of 40 dates, its ratio and the flag.
    "Insufficient evidence" is shown until then, and records from before
    schema 1.1 have no B1 side.
  - A comparison is reported, never a decision. Whether a challenger beats
    B1 is decided on development data only (A7).

**Rendering and delivery** (`render`, `delivery`):

- **Formats.** Plain text and one self-contained HTML page: no script,
  link, image or fetched font, and no dashboard link. The page uses the
  Lab's palette in light and dark, and every category is written in words
  beside its marker. A JSON copy keeps the structured briefing.
- **Adapters.** `DeliveryAdapter` is the interface. The only adapter,
  `PreviewAdapter`, writes the three files (mode 0o600, never overwritten)
  into `STOCK_AGENT_STOCK_LAB_BRIEFING_HOME` or
  `~/stock-agent-private/stock_lab_briefings/` (mode 0o700). That
  directory is refused inside any Git working tree and beside the research
  store.
- **Duplicate sends.** `DeliveryLog` is an append-only SQLite file in that
  directory, a regular file owned by the user. It allows one delivery per
  market date, channel and data class (claimed, then sent or failed),
  claimed in an immediate transaction.
  - An unfinished claim blocks a second send.
  - A failure records only the error's type and is never chained into
    another traceback.
  - The preview adapter writes all three files or none.
- **Previews and deliveries.** Only a briefing prepared before the open
  (`premarket`) is a delivery and uses the log. A preview prepared at any
  other time is written under its own labelled name and never uses up that
  day's delivery.

| Command | What it does |
|---|---|
| `.venv/bin/python scripts/stock_lab_briefing.py --demo` | replays 8 synthetic sessions in a temporary store and prints the synthetic briefing |
| `... --demo --preview` | also writes its HTML, text and JSON to the private directory |
| `... ` (no flag) | prints the real watchlist's briefing; today every stock is unavailable (A4) |
| `... --preview` | also writes the real briefing's files, once per market date |
| `... --at 2026-10-19T08:30-04:00 --include-closed-days` | prepares as of a time, including a closed day |

**The Lab's Premarket Briefing page** (`app/briefing.py`, read-only):

- **What it shows.** The latest briefing the script wrote on this computer,
  in the Lab's design, light and dark:
  - the notices;
  - every stock at a glance;
  - the header facts;
  - each stock in full;
  - reliability, with the persistence caveat after the comparison with B1
    when its flag is on, and the limitations.
- **Same words as the email.** The words are the renderer's own (`render`
  exposes them), and every category is written beside its marker
  ("▲ Increase").
- **One kind at a time.** The page shows real or synthetic demo briefings,
  never both. Real comes first when one exists, and a synthetic demo carries
  its badge and banner.
- **What it reads.** The written JSON copies and the duplicate-send log,
  through `briefing.stored`.
  - It never loads `delivery`, the writers (tested).
  - It prepares no briefing, runs no job, writes or creates no file and
    sends nothing (tested: the folder's bytes, modes and modification times
    are unchanged).
- **Why the written copy.** It shows exactly what was delivered at 08:30.
  Preparing a briefing again would need the NYSE calendar module, which
  loads model-validation code the Lab may not load.
- **Which briefing is the latest.** The newest market date first; among one
  date's briefings, the one prepared last. Earlier ones stay selectable.
  - Whole market dates are read, newest first, until at least 30 files of a
    kind are read; files of older dates are counted, never opened.
  - A file that cannot be read is named with a short reason, never skipped
    in silence. When it is newer than the one shown, a warning says so
    above the briefing.
  - A real briefing for an earlier day than today (New York) says so, so
    Monday's briefing is never read as Tuesday's.
- **What a readable file is.**
  - a regular file (never a link or a named pipe, opened without waiting),
    owned by this user, that no other user can open, at most 2 MB;
  - in a private folder outside any Git working tree;
  - strict JSON of this briefing schema and version, read back field by
    field and type by type (a null where a number is required is refused);
  - with a name that agrees with its contents (market date, data class and
    state), and contents the renderer can write.
  - Text from the file is shown literally: no markdown in it is rendered.
  - These rules guard against other users. Another program running as the
    owner could still swap a file between the checks and the read.
- **The delivery log.** It is opened with SQLite `mode=ro` (a missing log is
  not created), and only in rollback-journal mode, which the writer uses
  (tested): a read-only reader of a write-ahead-log database would create
  its `-wal` and `-shm` files, so such a log is refused.
  - An entry the writer never writes (a bad time, status or attempt) makes
    the log unreadable on the page, never half-read.
  - The page reports in words what the log recorded for the briefing's
    market date: delivered, failed (the error's type only), not finished,
    not recorded, or, for a preview, that date's delivery state.

**The 08:30 delivery option (printed, not installed):**

- **The agent.** `.venv/bin/python scripts/stock_lab_schedule.py --briefing`
  prints a launchd agent, `local.stock-lab.briefing`.
  - It runs `scripts/stock_lab_briefing.py --preview` at 08:30 New York on
    weekdays. That is the local preview channel, once per market date
    through the duplicate-send log.
  - Its log goes to `~/stock-agent-private/stock_lab/logs/briefing.*.log`,
    written with a private umask (files 0600).
  - It never runs on load. The script itself skips weekends and holidays.
  - It runs the code of the checkout it was printed from, so print it from
    a clean checkout of main.
- **The time zone.** launchd takes the Mac's local time with no time zone,
  so the printer converts 08:30 New York to the zone given.
  - It refuses a zone whose clock changes on other dates than New York's, or
    never (Europe/London, Asia/Tokyo, America/Phoenix, UTC). There, a fixed
    local time would run at 09:30 New York for part of the year, after the
    open.
  - US and Canadian zones that follow New York's changes are accepted and
    tested on both sides of each change.
  - The 20:30 daily agent follows the same rule (section 6, Scheduling).
- **The optional wake.** It is printed as a note on standard error, never
  run and never part of the agent file.
- **The email dry run.** `.venv/bin/python scripts/stock_lab_schedule.py
  --email-dry-run` describes an email adapter. No email code exists.
  - **What it would send.** The rendered briefing, text and HTML in one
    message, to one address, the owner's own, kept outside Git.
  - **Real only.** It sends real briefings only, never a synthetic demo.
  - **When.** Only a premarket briefing, at most once per market date, on
    its own "email" channel of the existing log. The wall clock (never
    `--at`) is checked again just before connecting: at or after 09:30 New
    York nothing is sent, and the attempt is recorded as late.
  - **The credential.** It would be read from the macOS Keychain at send
    time (`security find-generic-password`, run without a shell, under a
    generic account label, never the address). It would never be in Git,
    an environment variable, a command-line argument, a log, the delivery
    log, a preview file or an error message.
  - **Adding it.** The owner adds it once, typed at a prompt.
  - **Who else could read it.** An item the `security` tool adds can be read
    again by that tool without a prompt, so any program running as the
    owner, an agent's shell included, could read it. How to narrow that (a
    keychain of its own, or a command guard refusing
    `security find-generic-password` to agents) is an owner decision; a
    guard is a guardrail change.
  - **Public CI.** It holds no secret, because the repository is public.
  - **Tests first.** The adapter needs its offline tests before any send: a
    fake transport, the secret scan, the 09:30 refusal and the
    duplicate-send log.
- **Owner approvals, each separate and none given.**
  - the provider;
  - the address;
  - the Keychain item;
  - installing the agent;
  - the wake setting;
  - the data-licence check below.

**When the Mac is asleep (the local option's limits):**

- **Sleep.** launchd does not fire while the Mac is asleep. A job whose time
  passed during sleep runs once when the Mac next wakes (missed times are
  combined into one run, launchd.plist(5)). Its manual promises nothing
  after a shutdown, so do not count on a run then.
- **What a late run does.**
  - Before 09:30 New York, it is still the day's premarket delivery, late.
  - At or after 09:30, the briefing is `after_open`: the script writes a
    labelled late preview and never sends or records the day's delivery.
  - A Mac asleep for several days gets one run, at wake, for the session
    due then.
- **The evening job sleeps too.** The evening agent at 20:30 has the same
  limit. A forecast decided after the next 09:30 open is refused
  (`after_next_open`). The morning briefing then shows each stock as
  unavailable or stale, with the reason, and Model Health shows the missed
  evening (`evening_missed`). The evening's own wake (section 6) replaces
  this one, because pmset keeps one repeating pair.
- **A local wake.**
  - **The command.** `sudo pmset repeat wakeorpoweron MTWRF 08:25:00` (in
    the Mac's local time; the printer gives the time for the zone).
  - **Who.** It is an administrator setting, so the owner sets it.
  - **One repeat only.** pmset keeps a single repeating pair (a power-on
    and a power-off event), so this may replace a repeating schedule
    already set.
  - **Checking it.** Check it with `pmset -g sched`, before and after. Then confirm in the
    briefing log that the 08:30 run happened: a Mac can fall asleep again
    before 08:30 under its idle settings. A laptop with its lid closed or on
    battery may not wake at all; check on this Mac.
- **Seeing a missed delivery.** The Lab page shows "No delivery is recorded"
  for a premarket briefing without a delivery, and a late preview as a
  preview. A Model Health "missed delivery" check is a separate change,
  coordinated with the health owner.

**Recommended reliable deployment (pending approvals):**

- **The host.** A small private host that is always on, set to New York
  time or a zone that follows its clock changes.
  - **20:30 New York.** It runs the evening job: the recent-window fetch,
    then the daily job, once a model is activated (A4).
  - **Just after, the advance preview.** A dress rehearsal: a failure shows
    about 12 hours before the briefing is due, not at 08:30.
  - **08:30.** It prepares the premarket briefing (the same end-of-day data,
    as known then) and delivers it once, refusing at or after 09:30.
- **Why it is more dependable.** Nothing sleeps, the evening job and the
  briefing run on one machine, and the duplicate-send log keeps one
  delivery a day.
- **Before it can be used.**
  - **The vendor data licence.** Storing vendor prices on a hosted machine,
    and emailing derived numbers to oneself, must be checked against the
    terms for private use. Nothing is ever redistributed.
  - **Secrets.** The host needs its own secret store; nothing goes in
    environment files or in public CI.
  - **Approvals.** The host, its cost and the installation are owner
    approvals.
- **Not an option.** Public CI (this public repository's workflows): its
  logs are public, and it would need the vendor key.
- **Alternatives, from least to most dependable.**
  - **A Mac kept awake.** A desktop Mac, plugged in, with the wake above.
    It costs nothing and misses mornings when it sleeps or is shut down.
  - **An always-on machine at home.** A small machine running the same
    jobs.
  - **A private cloud host.** As recommended above.

**Next steps, each needing the owner's approval:**

1. **Delivery.** The 08:30 agent and the email dry run are printed (above);
   nothing is installed, connected or sent.
   - Still to do: the email adapter, after the provider, the address, the
     Keychain item and the installation are approved.
   - A private hosted sender is a separate approval, after the data-licence
     check.
   - A Model Health "missed delivery" check is coordinated with the health
     owner.
2. **Reliability.** B1 is recorded beside each forecast and compared on
   matched records (done). Forecast-change monitoring remains.
3. **Market regime.** The frozen job records momentum, volatility and the
   regime with each real forecast (done). Longer-horizon diagnostics
   (63/252 sessions) remain.
4. **Shadow models.** Research only after real prospective data exists. The
   running state is the owner's `frozen_long_run` rule (2026-10-19).
