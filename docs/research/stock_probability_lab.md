# Stock Probability & Reversal Lab

Status: experimental.

- PR 1 (#78) built the watchlist, the stock data adapter, the features and
  the Lab page on fixtures.
- PR 2 builds the estimators, the five-session forecasts, the reversal
  analyzer, bull and bear traps, the walk-forward evaluation and the
  modelled views.

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
  - Newey-West with lag 10 (at least h − 1), and a circular block bootstrap
    (block 20) as a second view.
  - Holm correction across the pre-registered comparisons against B1.
  - Beside Newey-West, reported only: the fold-block t (a t-test on the
    test years' means, t(folds − 1)) and the circular block bootstrap's
    one-sided p. Newey-West still decides; `size_simulation.one_sided_tests`
    computes all three for every comparison.
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
    - Which test decides is owner decision A7. Its decision packet,
      `docs/research/stock_lab_a7_decision.md`, simulates six tests under
      the planned conditions and recommends the fold-block t at a one-sided
      family level of 2.5% (5% as the alternative).
- **Selection.**
  - Keep B1 unless a challenger beats it (Holm).
  - Then take the simplest such challenger (order B0, B1, B3, B2, M1).
  - Move to a more complex one only if it also beats the current choice on
    the same per-date differences.
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
- **Until decisions A1, A2 and A4:**
  - the real job scores whatever is stored;
  - it records itself as `skipped` and not yet authorized, with the reason,
    and writes its Model Health input. Health reads that as not activated
    (insufficient evidence), never as a degraded job. A job that stops
    running still fails as overdue;
  - it refuses the real-update flags.
- **The price update and real forecasts** are built after those approvals.

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

- The job writes `reports/health_inputs/stock_lab_monitor/<run_id>.json`
  (envelope v1.0, write-once) with the fields Health reads.
- Five checks join the private monitor:
  - `data.stock_lab_freshness`;
  - `forecast.stock_lab_job`;
  - `forecast.stock_lab_scores` (insufficient evidence below 250 scored
    targets; degraded when worse than the baseline; tracking only);
  - `forecast.stock_lab_drift` (PSI, degraded above 0.25);
  - `forecast.stock_lab_ledger`.
- Their runbook entries are in `docs/operations/runbook.md`. Public CI
  reads only the synthetic fixtures.

**Scheduling** (`scripts/stock_lab_schedule.py`):

- It prints a launchd agent for 20:30 New York time on weekdays; the owner
  installs it.
- The job itself skips holidays through the calendar.

**The page:**

- The job's last status, a stale warning when the latest run expected a
  later session than the forecast, and the forecast history.
- The history shows each stored forecast's next-session probabilities as
  recorded, beside what happened and its score.

## 7. Owner decisions (recorded 2026-10-16)

The owner answered A1-A6 on 2026-10-16. They are given here in paraphrase,
with the conditions that bind the implementation. A7 was raised afterwards
by the simulated size checks (section 5) and is still open.

| Id | Decision | Conditions | Status |
|---|---|---|---|
| A1 | Tiingo Starter and the private watchlist store: approved | **Staged:** inspection and persistence are separate. The first live operation is the read-only command below, run only after the owner confirms the token is configured and authorizes that exact command. The 30-ticker cap holds. No vendor-adjusted fields, no unauthorized history, no redistribution. After a successful coverage and corporate-action audit, storage (`--store`) is a separate approval. | awaiting the token and authorization of the read-only command |
| A2 | Trailing 252-session inference window: approved in principle | Fit, tune, select and calibrate only on eligible pre-2025 data. Recent observations from the protected period serve only as inputs to a previously frozen model, kept in a separately controlled prospective/inference store. Protected outcomes are never used for retrospective selection, training or backtest claims. The sealed ETF period and its protections are unchanged. Strict as-of and retrieval-time records. A scoped authorization, not an unlocking of the 2025+ holdout. | to build: the inference store and reader |
| A3 | Exploratory walk-forward evaluation of the saved watchlist on pre-2025 history: approved | Marked hindsight-selected and non-confirmatory. Every challenger is compared with B0/B1, with proper scores, calibration, stock-level coverage, uncertainty and multiple-comparison treatment. No claim of profitable trading or reliable direction. | runnable after storage (`scripts/stock_lab_evaluate.py --real --authorized-evaluation`) |
| A4 | The write-once forecast and scoring protocol: approved | Activation needs successful data validation, a frozen registered model, verified decision timestamps and a successful end-to-end dry run. No backdating, rewriting of probabilities, or scoring of outcomes that existed before a forecast. Daily updates record skipped tickers, stale observations, model failures and missing outcomes in Model Health. Separate from ETF OD16 and PPO promotion. | protocol built; activation pending |
| A5 | Version-1 definitions: approved | Daily band ±0.5%. Five-session band ±0.5% × √5 ≈ ±1.12%, shown as different. A Stable session ends a streak by default, and the pause rule is identified whenever used. The bull-trap setup (10% drawdown, 5% rebound, prior 20-session high, 10-session window) may be used for research. **The failure margin f awaits confirmation** (formula and example in section 5) before real-data risk flags. Price returns stay distinct from total returns, and ex-dividend moves are identified. | f to confirm |
| A6 | Recent observed prices on the private dashboard: supported | Only after the retrieval and storage permissions. An isolated descriptive reader that cannot expose protected outcomes to development, training, tuning or backtesting. Recent observed history is kept distinct from research evidence, and current forecasts from completed outcomes. The holdout policies are not modified. | to build after A1 storage and the A2 store |
| A7 | Which test decides "beats B1": open | Newey-West (lag 10, as pre-registered; 6–8% simulated size at a nominal 5%), the fold-block t (closest to nominal in simulation, as on the ETF track), or Newey-West against a simulated critical value | packet ready (`docs/research/stock_lab_a7_decision.md`, recommending the fold-block t at a 2.5% family level, 5% as the alternative); awaiting the owner; needed before the A3 evaluation selects |

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
- **Exercise.** Show that under the "breaks" rule the per-session view
  reports, at each run's last session, that run's length. That is the
  property tested in `test_every_run_row_matches_the_per_session_view_at_its_end`.
  Then explain why the "pauses" rule cannot have it for short Stable
  stretches.
