# Product architecture

Status (2026-10-11): **direction approved by the owner on 2026-10-11** (OD1, OD2, OD4, OD5, OD6, OD7 in `vision_and_prd.md`, section 13); other decisions pending. Nothing here is implemented. Tags:
- **[V]**: verified in the repository on 2026-10-11 (file paths cited);
- **[R]**: recommendation;
- **[I]**: inference or assumption.

## 1. Starting point [V]

- **Imports point one way:** `data → features → model_validation → model_diagnostics → dashboard`. `evaluation/` (backtests and metrics) uses `agents/` (equal weight, inverse volatility, momentum) and `environment/portfolio_math.py`. `universe/eligibility.py` exists but no script uses it.
- **Run outputs are write-once, versioned and hash-manifested** (`model_diagnostics/artifacts.py`, `contract.py`). The dashboard reads only verified development runs and refuses lockbox dates (`dashboard/loader.py`). A test proves its pages import no model or raw-data code (`tests/test_dashboard_app.py`).
- **Data:**
  - yfinance prices from 2015, unadjusted plus `adjusted_close`;
  - current-only sector, industry and market-cap snapshots;
  - EDGAR fundamentals (a tested library);
  - ALFRED macro vintages;
  - DGS3MO only from 2024.
- **Missing pieces:**
  - no security master;
  - no trading calendar (sessions are observed dates);
  - no delisting, corporate-action or membership history;
  - no Treasury curve, bond total returns or crypto data;
  - no optimizer or covariance model.
- **Backtest gaps:**
  - `run_rebalanced_backtest` executes at the same close: `evaluation/baseline.py` applies the weights of step t−1 to the return of step t, and T5 plans a one-session lag.
  - Cash added by `add_cash_returns` earns a constant 0 by default, and `scripts/run_all_baselines.py` passes `cash_return=0.0`.
  - No cost is charged for the initial build.
- **The run ledger** (`reports/ledger.jsonl`) is ignored by Git.

## 2. Three architectures

### A. Monolith evolution
Add Explore and Build pages to the existing Streamlit app and compute screens, allocations and backtests inside it.

| | |
|---|---|
| Advantages | Fastest demonstration; no new contracts; one process |
| Disadvantages | Breaks T3's tested guarantee that the app computes nothing. Validation status becomes a UI convention instead of a contract. Logic drifts into Streamlit callbacks, and every frontend change risks a statistical change. The large view modules (`views.py` 1271 lines, `figures.py` 1330) grow further |
| Cost and migration risk | Low cost now; high migration risk later |

### B. Publication pipeline (recommended)
Batch jobs compute everything and write **publications**: write-once, versioned, contract-checked, hash-manifested snapshots, generalizing today's run directories. The app is a read-only viewer of publications plus pure, deterministic Builder functions over published inputs.

| | |
|---|---|
| Advantages | Generalizes a pattern that is already built and tested (`write_run` / `read_run`, schema versions, ledger). Point-in-time, lockbox and evidence-status rules are enforced once, in the writer and the reader. Every number traces to a code SHA, input hashes and a knowledge time. The publication contract survives a frontend swap |
| Disadvantages | Daily latency (fine for 5–60-session horizons). Contract governance. The Builder is limited to pure functions over published inputs. Local files serve one user |
| Cost and migration risk | Moderate cost; low migration risk (additive) |

### C. Service platform
An API, a database, authentication, a scheduler and a web frontend.

| | |
|---|---|
| Advantages | Multiple users, saved portfolios, hosting, alerts |
| Disadvantages | Crosses the dependency, authentication, cloud and deployment gates to serve outputs that are not yet validated. Large operations and security surface. Storing users' financial inputs creates privacy duties. Serving other people moves toward regulated advice (`vision_and_prd.md`, "Regulatory boundary") |
| Cost and migration risk | High cost and high risk now. Low risk later if built on B's contracts (a read-only API serializes the same publications) |

**Recommendation: B.** The bottleneck is evidence, not serving. B keeps every safeguard enforced by code and makes C a later wrapper instead of a rewrite.

## 3. Recommended layers

```text
reference data ─┐
market data ────┼─► features ─► rules ───────────┐
                │             ─► research models ─┼─► allocation ─► publication writer ─► publications/ ─► reader ─► app (Explore · Build · Research)
                │             ─► risk ────────────┘                         ▲
                └─────────────────────────────────────────────── monitoring ┘  (scores matured outcomes, appends evidence)
```

Existing packages keep their roles. A new package is created only by the first ticket that needs it.

| Layer | Holds | Where and when |
|---|---|---|
| Reference data | Security master, listing and identifier history, corporate actions, delisting events, universe membership, trading calendars (NYSE, SIFMA bonds, later a 24/7 UTC calendar) | A new `reference/` package, created by the research ticket RES-1. Required before any result counts as product evidence. The point-in-time universe is **reference data built by rules R1–R8**, not a model. `universe/eligibility.py` (exists, unused) is its seed |
| Market data | Raw daily bars with split and total-return factors; fund total returns; Treasury yields and cash; later crypto | `data/` (exists); extended additively |
| Features | Point-in-time features; regime descriptors; stock characteristics | `features/` (exists) |
| Research models | Validation, diagnostics, alpha models | `model_validation/`, `model_diagnostics/` (exist). Published only as research-only until validated |
| Allocation rules | Pure weight functions: fixed weights, equal weight, inverse volatility, caps, cash floor | `agents/` (exists: equal weight, inverse volatility, momentum). Extended there; no third home for inverse volatility. No expected-return optimizer until a return forecast is validated |
| Simulation and metrics | Backtests, costs, performance and risk metrics | `evaluation/`, `environment/portfolio_math.py` (exist); corrected by T5 |
| Risk | Historical risk descriptors and stress scenarios (MVP); risk forecasts later | Descriptors through `evaluation/` metrics. A `risk/` package only when forecasts start (P4-1) |
| Publications | Envelope, writer and fail-closed reader with the evidence gate. Wraps the T2 run contract and never changes it | A new `publication/` package (P1-1a) |
| Presentation | A product entry point separate from the research dashboard | A new `app/`, plus a neutral `ui/` module holding `theme.py` and `components.py`, moved out of `dashboard/` (P1-1b). `dashboard/__init__.py` says nothing outside it imports it |
| Monitoring | Append-only forecast log written at the signal cutoff; scoring as labels mature; freshness and drift | A `monitoring/` package when the first forecast log starts (RES-6) |

**Where this departs from the brief's proposed layers.**
- **Treasury and cash "return models" → data plus a formula.** Short-horizon bill returns follow from yields, and a bond fund's return is a total-return series. Neither needs a forecasting model until we want Treasury *forecasts*.
- **Broad-market return/regime layer → descriptive risk state first.** Market timing is hard to validate (the Goyal-Welch evidence). The layer starts as descriptive regime indicators and risk scaling. A market return *forecast* is a LONG-TERM, separately validated model (section 5).
- **Crypto model → deferred.** The 24/7 calendar belongs in reference data. The data licences are unresolved (Coin Metrics is CC BY-NC; Coinbase via FRED is restricted). The volatility and regulatory profile differ.
- **"Risk and calibrated uncertainty" → split in two.** Risk descriptors and risk forecasts (volatility, covariance, drawdown scenarios) come first. Calibrated uncertainty of *return* forecasts waits for a return model worth calibrating.

## 4. Safety by construction

**Target design.** Every publication carries:
- **`evidence_status`**: `descriptive | rules_illustration | research_only | development_evidence | holdout_checked | prospectively_confirmed`;
- **`claim_type`**, derived from the publication's `kind`: descriptor, rule output, risk forecast, return forecast, allocation, performance;
- **`claim_level`** for return forecasts: `rank | bucket | expected_excess | expected_absolute | distribution`.

`descriptive` and `rules_illustration` make no forecast claim. The four forecast statuses are ordered: `research_only < development_evidence < holdout_checked < prospectively_confirmed`. `holdout_checked` means T6 found no collapse. It is not a confirmation: the holdout's MDE (about 0.05 in rank IC for 100 names) is above plausible effects.

**One taxonomy, two levels.**
- The publication's `evidence_status` says what the publication may claim.
- Each series inside a performance publication has a `sample_type` (`actual | backtest | paper`); the UX badge shows that.
- Forecasts live only in ranking and risk publications, never in performance series.

| `evidence_status` | Typical content | Series badge (`ux_blueprint.md`, section 5) |
|---|---|---|
| `descriptive` | Facts about the past: a fund's actual total return, realized volatility, holdings | Actual |
| `rules_illustration` | Weights from a declared rule; a hypothetical backtest of that rule | Simulated (backtest); Paper (after registration) |
| `research_only`, `development_evidence`, `holdout_checked` | Model output in development, or after T6 found no collapse | Forecast (Research mode only) |
| `prospectively_confirmed` | Model output that passed the pre-registered one-sided test on holdout plus prospective data | Forecast (Explore and Build, at its claim level) |

**Rules enforced in code** [R]:
1. **Only a promotion path may set** `holdout_checked` (citing the T6 record) or `prospectively_confirmed` (citing the pre-registered test and its monitoring record). Only `prospectively_confirmed` unlocks Explore and Build.
2. **An allocation that uses forecasts is capped at its weakest forecast input.** An allocation from a declared rule alone is `rules_illustration`.
3. **The reader refuses unknown statuses and newer minor versions.** It never returns a return forecast to a mode that its status and claim level do not allow (`ux_blueprint.md`, section 3).
4. **UI components take typed records.** A forecast card cannot render a descriptor, and a descriptor card cannot render a forecast.
5. **An import test proves `app/` imports only the reader, the allocation rules and `ui/`.** Never models, raw data or the writer.
6. **Every publication records its code SHA and dirty flag.** The reader refuses dirty-tree publications outside Research mode.
7. **Lockbox boundary (decision OD5; default shown).** No publication reads data dated on or after `MODEL_HOLDOUT_START` (`model_validation/folds.py`, 2025-01-01), for any asset. Every product `as_of` is at most the last development session.
   - This is today's rule ("nothing reads the lockbox except a recorded holdout run", `docs/research/diagnostics_and_visualization_plan.md`) applied to the product.
   - The UI says "Data through 2024-12-31, the end of the development period. Later data is sealed for one final test."
   - **"Dated" means information content,** not the row's date label:
     - a holdings snapshot taken in 2025 is 2025 data;
     - a price level adjusted for a 2025 distribution carries 2025 information, so build history from raw closes plus the adjustment factors known by the `as_of`;
     - a trailing return over 2025 is a realized holdout label.
   - **What would change it.** T6 runs once, for everything pre-registered by then, and the owner fixes the holdout window at its pre-registration. After T6 that window becomes development data.
   - **From then on, each entity's evidence class is its own.** A portfolio, rule or model counts as development data before its registration date and as prospective after it. Prospective results are viewed only on pre-registered review dates.
   - **Designer knowledge.** The repository began on 2026-08-08, so every design was made knowing what happened in 2025. Designs carry a `designed_after_holdout_start` flag and a list of the 2025+ facts known. The holdout cannot remove that knowledge; prospective evidence can.

**Version 1 minimum** (P1-1a). Only `evidence_status` ∈ {`descriptive`, `rules_illustration`}, `claim_type` derived from `kind`, and the lockbox rule.
- `claim_level`, the confirmed statuses, the promotion path and the forecast cap arrive with the first publication that needs them.
- Because the reader fails closed, adding values later is a safe minor-version bump.

## 5. Composing absolute forecasts, and when they stay unavailable

The equity model predicts the **excess return over the equal-weight eligible universe** (the label):

    x_i,h(t) = r_i,h(t) − r_EW,h(t),   r_EW,h(t) = (1/|C_t|) Σ_{j∈C_t} r_j,h(t)

The identity r_i = r_EW + x_i is exact, so an absolute forecast needs a compatible forecast of the benchmark:

    E[r_i,h | F_t]   = E[r_EW,h | F_t] + E[x_i,h | F_t]
    Var[r_i,h | F_t] = Var[r_EW,h | F_t] + Var[x_i,h | F_t] + 2 Cov[r_EW,h, x_i,h | F_t]

`F_t` is the information at the signal cutoff. The covariance term is not zero in general: writing r_i = β_i·r_EW + ε_i (ε_i a residual) gives Cov(r_EW, x_i) = σ_EW²(β_i − 1).

**The excess label is not beta-neutral, so adding a market forecast to it is wrong.**
- Date-common inputs get zero coefficients under the excess label. So the excess model learns only the *unconditional* beta term (β_i − 1)·μ̄, where μ̄ is the unconditional mean of r_EW, and ignores any market view.
- **Illustrative example, not measured.** Take β_i = 1.8, μ̄ = +0.8% per 20 sessions, a market forecast m = −3%, and no residual skill (E[ε_i] = 0).
  - Correct: E[r_i] = β_i·m = 1.8 × (−3%) = −5.4%.
  - Naive sum: m + x̂_i = −3% + 0.8 × 0.8% = −2.36%, which is 3.04 percentage points too high.
  - Feeding the *excess* forecast into a beta form double-counts: β_i·m + x̂_i = −5.4% + 0.64% = −4.76%.
- **A beta-aware composition corrects the excess forecast for the market view** (a financial-formula decision, owner gate):

      E[r_i | F_t] = m + x̂_i + (β̂_i − 1)(m − μ̄)

  With the example's numbers: −3% + 0.64% + 0.8 × (−3.8%) = −5.4%. The alternative is to retarget the model to the residual r_i − β̂_i·r_EW and compose β̂_i·m + ε̂_i.
  - β̂_i is a trailing point-in-time estimate.
  - Per-date demeaning needs both the excess forecasts to average zero and β̂ to average one over C_t. Then the cross-sectional mean of the composed forecasts equals m.

**Compatibility conditions.** All must hold, or the absolute forecast is **unavailable**:
1. **Same horizon h and calendar:** sessions on the same named calendar, with the same entry lag.
2. **Same benchmark definition:** the alpha's equal-weight universe C_t is the universe whose return the market layer forecasts. A forecast of the S&P 500 (cap-weighted) does not compose with an equal-weight alpha without a mapping that is itself validated.
3. **Same knowledge time:** both components were computed from information available at the same `signal_cutoff_at`.
4. **Each component has passed its own validation ladder** at the needed claim level: expected value for E[·], a calibrated distribution for anything probabilistic.
5. **The asset is in the alpha universe.** ETFs, Treasuries, cash and crypto have no alpha forecast from the equity model.

| Situation | Product shows |
|---|---|
| No validated market forecast (true today, and likely for a long time) | Absolute return: unavailable (`no_market_forecast`). At most, a labelled **assumption** such as a long-run historical average, shown as an assumption and never as a forecast |
| Alpha validated at the rank level only | Rank or bucket; no percentage |
| Horizon, universe or knowledge time mismatch | Unavailable (`incompatible_components`) |
| Asset outside the alpha universe | Unavailable (`not_in_model_universe`) |

**Probability of gain needs a calibrated predictive distribution**, never a point forecast. P(r_i,h > g | F_t) = 1 − F̂_i(g), where F̂_i is a validated predictive distribution of the absolute return.
- **What counts as a "gain" is a formula decision:** above zero or above cash, before or after costs, with a zero band so that sign noise (T4 scored ±1e-19 forecasts as hits) is not counted.
- **It is mostly a market forecast.** With no timing skill it is roughly Φ(μ̄/σ_i), which ranks names by low volatility.
  - Illustrative, not measured: μ̄ = 1% per 20 sessions gives Φ(0.2) = 0.58 for σ = 5% and Φ(0.067) = 0.53 for σ = 15%.
  - For a stock with 52% annual volatility, σ_20 ≈ 0.52·√(20/252) ≈ 14.6%. Then μ = 1.5% gives 0.54, and an error of ±1 percentage point in μ moves it only between 0.51 and 0.57. The number is mostly base rate; any decimals beyond that are false precision.
- **The product never sorts by P(gain)** unless the location rungs of the ladder have passed. Otherwise it would be a hidden low-volatility screen.

**Expected downside** (expected shortfall, quantiles) needs the dispersion track (D1) of the ladder: coverage and exceedance tests. Until then the product shows **historical** drawdowns and scenario losses, labelled as history.

## 6. Asset classes

### US equities
- **Universe.**
  - The primary research universe is U1: about 100 point-in-time large caps from objective rules R1–R8 (`docs/research/universe_and_market_context.md`), re-estimating the MDE on the real panel.
  - S&P 500 membership is a **licensed** history (CRSP, Sharadar, Norgate). It is used as a membership lens or benchmark only if licensed, and is never reconstructed from today's list.
  - "AI and semiconductor" names enter only through objective eligibility. Examples: SIC 3674 from the filing available at the rebuild date, plus liquidity and listing rules. "AI exposure" is a point-in-time characteristic, such as the share of 10-K text or data-center segment revenue.
  - "Magnificent Seven" is a 2023 media label and a hindsight basket. It is a current exposure lens only, never a historical strategy.
- **Corporate actions.**
  - Raw unadjusted bars plus cumulative split and total-return factors.
  - Dividends as cash flows, with ex-date timing.
  - Spin-offs restart the R3 history count.
  - Ticker and CIK changes go through `listing_history` and `issuer_cik_history` (SNDK is two issuers; MRVL changed CIK in 2021).
- **Delistings and acquisitions.**
  - A delisting inside a label window uses the delisting return: merger consideration for cash or stock deals, CRSP `dlret` where licensed. Then a pre-registered reinvestment rule (into the equal-weight remainder, or cash).
  - When the return is unknown, use a documented fallback (for example −30% for a performance delisting) with a reason code.
  - The label's availability is the later of the window end and the time the delisting return became known.
  - **Dropping such dates selects on survival.** The current fail-closed excess label (any missing member → the date is unlabeled) is correct for data gaps but must not be applied to known delistings.
- **Membership changes.** Membership is decided at the rebuild date from inputs available by the signal cutoff, and takes effect at the next session. It is never removed retroactively (R8). The benchmark set C_t is the point-in-time membership, not "names with a feature row".

### Treasuries and cash
- **A yield is not a return.** A constant-maturity yield (DGS2, DGS10) describes a hypothetical par bond whose maturity never changes; no instrument holds it. A bond's holding-period return is approximately:

      R ≈ y·Δt − D·Δy + ½·C·(Δy)²   (carry, duration, convexity; plus roll-down)

  Illustrative, not measured: a 4.2% 10-year par bond has modified duration about 8.1 and convexity about 78. After a 25 bp yield rise over a month, exact repricing gives about −2.00%, plus carry 4.2%/12 ≈ +0.35%, so R ≈ −1.65%. For a fund, use the fund's own duration and yield, not a constant-maturity yield.
- **What the product uses:**
  - **Treasury exposure:** investable instruments only, as fund total returns (for example BIL, SHY, IEF, TLT) or licensed bond-index total returns. An aggregate-bond fund (for example AGG) is not Treasury exposure: it holds credit and mortgage-backed bonds.
    - Build ETF returns from raw closes plus dated distributions. Vendor-adjusted closes change on every download, so store the raw inputs.
    - Expense ratios are already in NAV, so never subtract them again.
    - Yields appear only as context features or as inputs to the formula above, labelled "approximation".
  - **Cash:** a T-bill fund's total return, or bill returns derived from DGS3MO.
    - The existing `prepare_risk_free_returns` accrues the investment-basis yield on actual/365 (366 in leap years), at the rate on or before the period start. Never mix in the discount-basis DTB3 (/360).
    - It approximates rolling T-bills, not what a retail cash sweep pays (often much less).
    - As a decision input it must lag to T+1 (H.15 publication).
    - It needs the convention check already listed in the universe document. It also needs the DGS3MO backfill (the series starts in 1981) to the earliest backtest window start minus warm-up, before T5.
  - Bond data follows the SIFMA calendar.
- **No Treasury forecasting model** until a separate ladder is passed. Bond risk (duration, rate sensitivity) is shown as descriptors.

### ETFs (for model portfolios)
- **Prices:** total returns from adjusted closes, with distribution dates.
- **Holdings:** a separate dated holdings snapshot, needed for look-through concentration and overlap.
- **Survivorship and hindsight:** an ETF chosen today because it exists today can be survivorship-biased. Prefer broad, long-lived funds, fix the inception in advance, and label the choice.

### Crypto (LONG-TERM)
- **Calendar.**
  - Trades 24/7. The product's decision clock is the NYSE `signal_cutoff_at`, so a crypto value enters row d only from a timestamped price at or before that cutoff.
  - A UTC daily close used for the same date leaks about 2 hours (`universe_and_market_context.md`).
  - Weekend moves fold into Monday for any equity-aligned portfolio.
  - Its horizon counts calendar time, not sessions. A separate calendar and label contract are required.
- **Data and licensing:** Coin Metrics community data is non-commercial; Coinbase data on FRED may not be reproduced.
- **Regulation:** crypto treatment is unsettled; counsel is needed before mixing crypto with securities in any allocation shown to others.

## 7. Cross-cutting questions

- **Trading calendars.**
  - Replace "observed dates" with named calendars: NYSE sessions with early closes, SIFMA for bonds, and later 24/7 UTC.
  - Horizons are counted in sessions of a named calendar.
  - Needs a calendar dependency or an in-repo holiday file (dependency decision).
- **Decision clock.**
  - `signal_cutoff_at` (proposed 18:00 ET on session d) is the product's `as_of`. Every input satisfies `available_at <= signal_cutoff_at`.
  - Execution is at the close of d+1, matching the label's entry lag.
  - This is an availability-semantics decision (open item 1).
- **Multi-horizon targets.**
  - Each horizon is its own label (`target_end_date` per horizon, so the purge follows automatically) and its own validated cell.
  - Overlap inflates the variance of the **mean rank IC** of a signal whose features overlap the same way by F(h) = 1 + 2Σ_{j=1}^{h−1}(j/h)²: about 3.4, 13.4, 20 and 40 at h = 5, 20, 30 and 60 (measured 13.9 at h = 20). For a mean of overlapping returns the factor is about h.
  - Folds need at least about 3h sessions per test block; h = 60 needs history from about 2010.
  - Order: 20 first, then 60 (dispersion first), then 5 after T5 costs. 30 is dropped: overlapping 20- and 30-session returns correlate at about 0.82 (√(20/30) ≈ 0.816 in theory; 0.81 measured on 2019–2024 index data), adding tests without information.
  - Horizons are gatekept in a fixed sequence to control multiple testing.
- **Row-weighted vs date-weighted training.**
  - With a variable universe, row-weighted fits let dates with more names dominate, while tuning, Goyal-Welch and rank IC weigh dates equally.
  - Recommended [R]: fit date-weighted (weights 1/N_t, with weighted standardization and λ_max), or rank or demean features within each date. Either removes the date-common forecast component that T4 found adds pure loss under the excess label (57–65% of feature variance).
  - N_t is the point-in-time membership count. It is never "names whose labels completed", which depends on survival.
  - This is a formula change (owner gate).
- **Embargo for non-forward schemes.** The current expanding forward folds never train on later rows (asserted per fold). Any future K-fold, CPCV or multi-horizon scheme needs a post-test exclusion of h + L + e − 1 rows after the test window: horizon, feature lookback and entry lag. That is 40 rows at h = L = 20 and e = 1, so the first eligible training row is 41 sessions after the window. L becomes 252 if 252-session z-score features arrive. Reconcile this with the 21-session embargo in `docs/research/model_validation.md` before such schemes exist.
- **Covariance and risk.**
  - **MVP:** historical correlation, volatility and drawdowns as descriptors, with the lookback stated.
  - **Stress scenarios** apply the realized returns of named episodes to the current weights; that needs no forecast.
    - **One fixed window per scenario,** registered with the scenario: for example 2008-09-01 to 2009-03-09, 2018-10-01 to 2018-12-24, 2020-02-19 to 2020-03-23, the 2013 taper tantrum (2013-05-01 to 2013-09-05), and 2022 (2022-01-03 to 2022-10-12).
    - **The two rate shocks matter for bond-heavy portfolios:** bonds fell with stocks in 2022.
    - **An asset that did not trade in the window** gets a labelled proxy (a registered index fund or index), or the scenario is `unavailable`, with the weight coverage shown. Never renormalize over the assets that existed.
    - Scenarios before the current price history (prices start in 2015) need the F1 data extension.
  - **Forecasts later:**
    - Ledoit-Wolf shrinkage toward constant correlation within one asset class.
    - A factor model Σ = BFBᵀ + D (market, sector, duration, credit, USD) for mixed universes on one synchronized calendar.
    - A daily EWMA with λ = 0.94 (about an 11-day half-life) has a flat term structure: it scales variance by h, while volatility mean-reverts. Use a model with mean reversion, such as GARCH or HAR, decaying toward the long-run level.
  - **Validation** like any risk forecast, on non-overlapping h-session returns: coverage and exceedance tests (Kupiec, Christoffersen).
  - **Validation power is low.** Ten years hold about 126 non-overlapping 20-session windows. A Kupiec test detects a 90% band whose true coverage is 85% with power of only about 0.34 (about 0.17 at h = 60). A risk forecast can pass a pre-registered non-rejection check, and the product shows that power beside it.
  - **Tails.** Volatility can be forecast; crash timing and drawdowns cannot. Correlations tend to rise in stress, so stress scenarios sit beside any correlation estimate. The four semiconductor names' full-sample mean pairwise correlation, 0.58 over 2019–2024, shows how concentrated that basket is; it is not itself evidence about stress.
- **Portfolio construction.**
  - **Without a validated alpha,** risk-based methods only:
    - fixed weights;
    - equal weight;
    - inverse volatility (exists, `agents/`);
    - equal risk contribution, w_i(Σw)_i = wᵀΣw/N;
    - minimum variance, with caps (uncapped, it concentrates in low-volatility names).
  - **Constraints:** long-only; position, sector and theme caps with ETF look-through; a turnover penalty κ‖w − w_prev‖₁; a cash floor; the user's own volatility or loss limit.
  - **Framing:** "the weights your rule and constraints produce, with their historical risk". Never "the highest expected return", and never "the lowest risk": once cash or a T-bill fund is allowed, minimum variance means as much cash as the cap allows.
  - **Cash enters only as a floor or a fixed weight,** never inside inverse volatility (1/0) or equal risk contribution (cash contributes no risk at any weight). Tests cover both.
  - **Alpha later, only as a bounded, cash-neutral tilt** inside a tracking-error budget TE:
    - Grinold-Kahn scaling uses residual volatility ω_i, not total σ_i (total σ pulls beta into the tilt): α̂_i = IC_lower·ω_i·z_i.
    - Remove the part of Σ⁻¹α̂ that does not sum to zero: with γ = 1ᵀΣ⁻¹α̂ / 1ᵀΣ⁻¹1, the tilt is λΣ⁻¹(α̂ − γ1), where λ = TE / √((α̂ − γ1)ᵀΣ⁻¹(α̂ − γ1)).
    - λ = 0 whenever α̂ is unavailable or IC_lower ≤ 0.
    - Long-only limits and caps then need an optimizer.
    - Unconstrained mean-variance maximizes estimation error (Michaud).
  - An optimizer library (scipy or cvxpy) is a dependency decision.
- **Benchmarks.**
  - Benchmarks must be total-return series. Today `config/assets.yaml` uses the price indices `^GSPC`, `^IXIC` and `^DJI`, whose `adjusted_close` equals `close` (dividends excluded). The strategies use dividend-adjusted closes, and `evaluation/benchmark.py` compares the two directly, which flatters the strategies by about the dividend yield.
  - Each model portfolio names its benchmark before results are seen.
  - **References are not benchmarks of themselves.** 100% equity, 60/40 and cash are reference portfolios. A rebalanced 60/40 is measured against a never-rebalanced 60/40, and a bond-heavy mix against cash plus a static mix with the same equity share.
  - **Benchmarks are fund-based where possible,** so fees match the portfolio's.
  - This, the zero cash return in `scripts/run_all_baselines.py`, and the same-close execution belong in T5.
- **Feature availability.** Every feature has `available_at`, and a derived value takes the latest of its inputs. Fundamentals use SEC acceptance times (exists). Macro uses vintages (exists, date-level). Prices are known at the close.
- **Uncertainty calibration.**
  - Nested folds: train | purge | calibrate | purge | test.
  - Adaptive conformal updated only on matured errors.
  - Platt scaling before isotonic for probabilities.
  - Refit cadence from measured calibration decay.
  - Never fitted on the lockbox.
- **Transaction costs and liquidity.**
  - MVP: costs per unit of turnover by asset class (stated bp; sensitivity shown), plus a participation check (position ≤ a stated share of median dollar volume).
  - Later: spread plus square-root impact, k·σ·√(trade / volume).
  - Every cost model is a financial formula (owner gate, T5).
- **Drift and staleness.** Freshness against each source's release calendar; smoothed PSI and missingness per feature; rolling calibration coverage; per-date IC with CUSUM. Each trigger maps to a product state (`stale`, `insufficient_confidence`, revert to rules).
- **Lockbox and prospective evidence.**
  - Fix the holdout end date at pre-registration; data after it is prospective.
  - Pre-registration and the holdout-use record live in Git (the ledger is ignored today).
  - The lockbox held about 1.75 years at h = 20 when this was written, and it grows until T6. Its one-sided MDE (about 0.05 for 100 names; about 0.066 by fold blocks) cannot confirm a small edge, so T6 is pre-registered as a "no collapse" check.
  - A prospective, hash-chained forecast log is the evidence after it, scored on pre-registered review dates with α-spending. Confirming an IC of 0.03 needs about 1,165 dates never used for selection.

## 8. Output contracts

Every publication shares one **envelope**:

| Field | Meaning |
|---|---|
| `publication_id`, `kind`, `schema_version` | Identity. The `kind` enum lists only kinds that exist; the target set is opportunity_ranking, risk_forecast, portfolio_allocation, performance_comparison |
| `as_of` | The signal cutoff the publication answers for (UTC) |
| `knowledge_time` | Latest `available_at` of any input. It must be ≤ `as_of`, or the writer refuses |
| `created_at`, `code_sha` | A dirty tree is refused for anything above `descriptive` |
| `input_hashes`, `data_vintages` | Hashes and retrieval times of every input |
| `universe_id`, `universe_rule_version`, `calendar_id` | The point-in-time universe and calendar |
| `method_id`, `method_version`, `parameters` | Rule or model and its frozen parameters |
| `claim_type`, `claim_level`, `evidence_status` | Section 4 |
| `evidence_refs` | Validation runs, pre-registration, T6 or monitoring records |
| `limitations` | Codes such as `survivorship_biased`, `current_listing_universe`, `hindsight_list`, `costs_assumed`, `no_market_forecast` |
| `supersedes`, `parents` | Lineage. Publications are write-once; a correction is a new publication |

Every row carries `status` ∈ {`ok`, `unavailable`, `insufficient_confidence`, `stale`, `pending`, `ineligible`, `sealed`}, plus a machine-readable `reason` and `detail`. Statistics are never silently dropped.

**Version 1 envelope (P1-1a), the minimum:**
- `publication_id`, `kind`, `schema_version`;
- `as_of`, `knowledge_time`, `created_at`;
- `code_sha` and the dirty flag;
- `input_hashes`, each with its retrieval time (this replaces `data_vintages`);
- `universe_id`, `method_id`, `method_version`, `parameters`;
- `evidence_status`;
- `limitations`, for example `current_listing_universe` or `observed_dates_calendar` (until a calendar exists).

Deferred until a publication needs them: `claim_level`, `evidence_refs`, `universe_rule_version`, `calendar_id`, `supersedes` and `parents`.

**Interim knowledge time.** Prices carry no `available_at` today. Until `signal_cutoff_at` is approved (OD9), version 1 uses an interim date-level rule: a daily close is known at the end of its session date. That needs the owner's approval as part of P1-1a.

### Opportunity ranking (one row per eligible asset, horizon and score)

| Field | Notes |
|---|---|
| `security_id`, `display_symbol`, `asset_class` | Permanent id from the security master; the symbol is display only |
| `horizon_sessions`, `calendar_id` | Only registered horizons |
| `score_kind` | `characteristic` (rules) or `model_rank`, `model_bucket`, `expected_excess`, `p_outperform`, each allowed only at its validated claim level |
| `score`, `rank`, `percentile`, `n_ranked`, `ties` | Ranks are computed within the same date, universe and horizon |
| `estimate_low`, `estimate_high`, `interval_level`, `interval_method` | Required for every forecast score; null for characteristics |
| `drivers` | Feature contributions or characteristic values (explanation) |
| `eligibility_status`, `eligibility_reason` | Why the asset is in or out (R1–R8 codes) |

### Risk forecast (portfolio or asset)

| Field | Notes |
|---|---|
| `subject_id`, `subject_kind` | Asset or portfolio |
| `measure` | `volatility`, `drawdown_historical`, `scenario_loss`, `var`, `expected_shortfall`, `beta`, `correlation` (covariance pairs as a long table) |
| `horizon_sessions`, `estimate`, `interval_low`, `interval_high`, `interval_level` | |
| `estimator`, `lookback_sessions`, `n_obs`, `n_eff` | `historical` (descriptor) vs `ewma`, `shrinkage`, `factor` (forecast) |
| `scenario_id`, `scenario_window`, `weight_coverage`, `proxies[]` | For historical replays, each with its registered window; proxies are labelled; coverage below 100% is shown, never renormalized |

### Portfolio allocation

| Field | Notes |
|---|---|
| `portfolio_id`, `rule_id`, `rule_version` | A model portfolio id, or a hash of a Builder profile (no personal data stored) |
| `weights[]`: `security_id`, `weight`, `cap`, `floor`, `binding` | Includes an explicit cash row; weights sum to 1 |
| `constraints[]`: `name`, `limit`, `value`, `binding` | Position, sector, theme, cash floor, turnover |
| `execution_at_rule`, `rebalance_rule` | For example the close of d+1, quarterly |
| `turnover`, `cost_model_id`, `expected_cost` | Against the previous publication |
| `exposures[]` | Asset class, sector and theme, with look-through and an as-of date |

### Performance comparison

| Field | Notes |
|---|---|
| `comparison_id`, `members[]`, `benchmark_ids[]` | |
| `series[]`: `member_id`, `date`, `wealth`, `period_return`, `sample_type`, `net_of_costs` | `sample_type` ∈ {`actual`, `backtest`, `holdout`, `paper`}. Never mixed in one series; forecasts live in a risk or ranking publication |
| `inception_date`, `inception_declared_at` | The inception was declared before the series was computed (pre-registration timestamp) |
| `summary[]`: `metric`, `estimate`, `interval_low`, `interval_high`, `method`, `n_eff` | CAGR, volatility, worst drawdown and duration, turnover, cost drag, difference vs benchmark |
| `variant_count` | From the ledger: how many variants were tried |
| `sealed_from`, `pending_periods` | The lockbox and immature periods, shown as states |

**Versioning.** A minor version may add fields. Readers refuse newer minor versions and unknown status values, as the T2 reader does today.
