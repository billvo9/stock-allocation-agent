# Point-in-time universe expansion and market-context layer

Status: research and design (2026-10-08; refined 2026-10-09). No data,
schema, or availability semantics change in this document; every such change
below is listed as an owner decision.

It extends the ML roadmap in `docs/research/diagnostics_and_visualization_plan.md`
with two tracks:
- **U**: the investable equity universe;
- **C**: market-context features.

Neither track blocks T2 (diagnostics and null models) or the first T4 run
(Ridge, Lasso, Elastic Net) as an engineering exercise. Both shape when T4
results can be treated as product evidence.

The universe sizes and detectable effects below are **planning targets under
the stated simulation assumptions**, not universal thresholds for
product-grade evidence.

Evidence tags: **[V]** verified against a primary source or local data on
2026-10-08 (sources at the end); **[S]** simulation or computation this
session; **[I]** inference or assumption.


## Part U: investable equity universe

### Why the current universe is not evidence

- **It was chosen in hindsight.** In 2026 we picked semiconductor names whose
  2015-2024 growth was NVDA 278x, MRVL 8.4x, MU 2.5x, INTC 0.7x [V, local
  data].
- **It is survivorship-biased by construction.** yfinance returns nothing for
  delisted tickers: XLNX, MXIM, and LLTC all come back empty [V].
- **It cannot detect plausible skill.** The minimum detectable mean rank IC
  is about 0.12 [S]; plausible true ICs are 0.02-0.05.

### Statistical power vs universe size

Under the null, a single date's rank IC has variance 1/(N−1). Overlapping
20-session labels inflate the variance of the mean by a factor F_overlap.
Measured on the real panel, F_overlap = 13.9 for `momentum_20d` [S] (13.35 in
theory for a 20-session rolling signal). Correlation among names inflates it
further by the design effect:

```text
SE(mean IC) ≈ sqrt( deff · F_overlap / ((N − 1) · T)  +  σ_regime² · B / T )
MDE ≈ 2.8 · SE        (5% two-sided test, 80% power)
```

- `deff` is the design effect: 1 + (names per cluster − 1) × (within-cluster
  correlation of the signal) × (within-cluster correlation of returns).
- `σ_regime² · B / T` is how much the true IC varies across market regimes.
  It does not shrink with more names.
- T is the number of test dates.

Simulated MDE for the mean rank IC under a null signal [S]:

| Universe | Effective N | 2019-2024 tests | 2012-2024 tests |
|---|---|---|---|
| Current 4 names | 4 | 0.12-0.13 | 0.085 |
| US semiconductors, 30-40 names (4 sub-industries, ρ = 0.4) | 14-16 | 0.060-0.064 | 0.041-0.046 |
| Broad large caps, 50 names (11 sectors, ρ = 0.15) | 46 | 0.035-0.042 | 0.024-0.029 |
| Broad large caps, 100 names | 83 | 0.028-0.036 | 0.020-0.025 |
| Broad large caps, 500 names | 248 | 0.015-0.028 | 0.012-0.019 |

Ranges run from sampling noise only to sampling noise plus an assumed regime
spread [I].

**Planning targets, not thresholds.** These figures depend on the simulation
assumptions:
- Gaussian factors;
- within-cluster correlations of 0.4 (semiconductors) and 0.15 (broad large
  caps);
- an assumed spread of true IC across regimes;
- the 20-session overlap modelled as a rolling sum;
- a persistent signal;
- the test-window span.

A real universe may need more or fewer names. Before any T4 result on a new
universe is read, re-estimate the MDE on that panel and pre-register it,
using the measured overlap factor and residual correlations.

**What the table shows.**
- **A narrow industry universe saturates.** Going from 30 to 40
  semiconductors leaves the standard error unchanged (0.0213 vs 0.0214) [S]:
  effective breadth follows the number of sub-industries, not the name count.
- **Under these assumptions, past about 100 names history length becomes the
  binding constraint.**
- **Comparing Ridge, Lasso, and Elastic Net with each other is underpowered
  at any size.** Their daily ICs correlate about 0.9, so the paired standard
  error is about √(2(1 − 0.9)) ≈ 0.45 times one model's, an MDE near 0.01
  [S]. Judge them on stability and against null models, not on rank order.

**Inference correction.** Under a true null, Newey-West standard errors with
lag 20 rejected 7-13% of the time at a nominal 5% [S]. T2 must use lag ≥ 40
or a 63-session block bootstrap, and must check its false-positive rate by
simulation before reporting. This supersedes "lag ≥ 20/21" in earlier
documents.

### Universe options compared

| Option | Membership source | Survivorship and selection | Breadth and power | Verdict |
|---|---|---|---|---|
| U0: current 4 names | Hand-picked in 2026 | Severe: hindsight winners, no delistings possible | MDE about 0.12 | Pipeline verification and null calibration only |
| US semiconductors (sector sleeve) | SIC 3674 from EDGAR filings, plus point-in-time liquidity rules [I] | Low *only* with delisted prices: the 2015-2022 acquisitions (Altera, Atmel, Fairchild, SanDisk, Linear, Intersil, Cavium, Microsemi, IDT, Mellanox, Cypress, Maxim, Inphi, Xilinx) are missing from yfinance | Saturates near 0.05 | Secondary sector analysis, not primary evidence |
| U1: point-in-time broad large caps, about 100 names | Own rules (point-in-time market cap or dollar-volume rank), or licensed S&P 100/500 history | Low with delisted prices and point-in-time rules | MDE about 0.020-0.025 with tests from 2012 (simulated) | **Primary research universe.** About 100 names is the planning target: under the stated assumptions it reaches an MDE near plausible effect sizes |
| U2: about 300-500 names | Same rules | Same | 0.012-0.019 | Only after a signal appears in U1 |
| U3: international semiconductors | Local listings and ADRs | Today's list (TSMC, ASML, SK Hynix...) is itself hindsight | Adds only 6-10 names | Separate data contract; deferred |

**"AI" cannot be defined historically.** Any AI basket built today is selected
on the outcome. Treat AI exposure as a point-in-time *characteristic* and
test it as a feature. Two examples:
- the share of 10-K text discussing AI, measured at acceptance time;
- data-center segment revenue share from XBRL.

**International (U3) breaks the US contract in several places.**
- Asian closes come 12-15 hours before the US close.
- It needs foreign-exchange rates and local trading calendars.
- Foreign issuers report semiannually under IFRS on 20-F/6-K, which the
  EDGAR adapter rejects today.
- ADR ratios change over time.
- Some names carry US-investor restrictions.

US-listed ADRs (TSM, ASML) fit the US *price* contract but not the
quarterly-fundamentals contract.

### Objective inclusion rules (draft for U1 and the sector sleeve)

The universe is rebuilt every 20 sessions. Every input satisfies
`available_at <= signal_cutoff_at` on the rebuild date, and membership takes
effect at the next session.

| Rule | Requirement |
|---|---|
| R1 | US primary common stock on NYSE, Nasdaq, or NYSE American, keyed by a permanent security id. ADRs, ETFs, OTC listings, preferreds, units, and warrants are excluded |
| R2 | Sector sleeve only: SIC 3674 on the latest filing available at the rebuild date. GICS history is licensed |
| R3 | At least 252 sessions since first trade. A spin-off restarts the count |
| R4 | 21-session median **unadjusted** close of at least $5. yfinance closes are split-adjusted (NVDA 2015-01-02 shows 0.503 [V]), so a floor on adjusted prices is wrong |
| R5 | 63-session median dollar volume (unadjusted close × unadjusted volume) of at least $10M; then rank by point-in-time market cap (unadjusted price × EDGAR `dei:EntityCommonStockSharesOutstanding`, known from filing acceptance) and keep the top K |
| R6 | Buffer to limit churn: enter at rank ≤ K, exit only at rank > 1.5K |
| R7 | Exclude securities under a US-investor prohibition in force at the rebuild date (OFAC NS-CMIC list) |
| R8 | Never remove membership retroactively. If a stock delists inside a label window, the label uses the delisting return; dropping that label would select on survival |

### Test cases: AMD, APLD, SMIC

These are evaluated under the rules, not added because they are interesting
today.

| | Verified facts | Under the rules |
|---|---|---|
| **AMD** (CIK 2488) | Removed from the S&P 500 effective after the close of 2013-09-20 and moved to the MidCap 400; re-added before the open of 2017-03-20; joined the S&P 100 on 2022-09-19. Moved from NYSE to Nasdaq on 2015-01-02. Acquired Xilinx on 2022-02-14 (1.7234 AMD shares per Xilinx share) [V] | **Qualifies on the rules.** Sector sleeve: fails R4 (under $5) until mid-2016 [I], then qualifies 2016-2024. U1 (top 100 by cap): not in 2016 (about $2B cap [I]), borderline in 2019, qualifies 2022-2024. **It shows why today's constituents are wrong:** a 2026 S&P 500 list would wrongly include AMD for 2014-2016, when it was a sub-$5 MidCap stock. The Xilinx merger must raise a `merger_in_window` flag for year-over-year fundamentals |
| **APLD** (Applied Digital, CIK 1144879) | SIC 7374 (computer processing, not semiconductors); fiscal year ends May 31. Former names: Applied Science Products, then Applied Blockchain. Traded on OTC Pink until a $5 IPO / uplisting to Nasdaq on 2022-04-13. Market cap about $0.17B (end 2022), $0.82B (2023), $1.7B (2024), $6.9B (2025) [V] | **Never qualifies for U1 or the semiconductor sleeve** in 2016-2024. 2016-2022 fails R1 (OTC). 2022-2023 fails R3 (too young). It is never SIC 3674, and it is far below the top 100 by cap. An "AI infrastructure" label would be hindsight. It could enter a broad 500-1000-name universe around 2025 on cap alone [I]. It fits the existing fundamentals contract: a US filer whose May fiscal year is handled by the XBRL fiscal identity |
| **SMIC** (CIK 1267482) | Hong Kong 0981 (since 2004). NYSE ADSs ("SMI") delisted in June 2019 for low volume and cost; deregistered, with no SEC filings since 2019-06-18. STAR Market 688981 since 2020-07-16. On OFAC's NS-CMIC list (EO 13959 as amended by EO 14032): listed 2021-06-03, effective 2021-08-02, so US persons may not buy or sell its publicly traded securities. Commerce Entity List since December 2020 [V; the Entity List date is from a secondary source] | **Never qualifies.** Before 2019 the US ADR was a foreign private issuer (20-F, not 10-Q), failing R1 and the fundamentals contract, and it was illiquid (the stated reason for delisting). Since 2021 it fails R7 in any universe a US investor could hold. At most it could later be a *context* series (an Asian semiconductor-cycle signal) under the international contract, with HKD/CNY prices and HK/CN calendars |

### Data contract required (prerequisite for treating results as product evidence)

There is no security master today. Identity is the current Yahoo ticker, and
EDGAR maps today's ticker to today's CIK (`edgar_source.py:59`).

**Facts that break ticker-based identity [V, local SEC data].**
- **SNDK** is two issuers:
  - SanDisk Corp, until its 2016 merger into Western Digital [I];
  - the 2025 Sandisk spin-off (Form 10-12B filed 2024-11-25 [V]).
- **MRVL changed CIK** when it reincorporated in 2021 (new CIK 1835632,
  8-K12B filed 2021-04-20). Ticker-based EDGAR cannot reach Marvell's
  filings before 2021.
- **Under Armour** has one CIK for two share classes, so a CIK is not a
  security key either.

**Additive tables needed** (half-open validity intervals; each row carries
`source`, `evidence`, `available_at`, `recorded_at`):

| Table | Holds |
|---|---|
| `security_master` | A permanent internal security id, never reused |
| `issuer_cik_history` | CIK intervals per issuer |
| `listing_history` | Ticker and exchange intervals; no two securities hold one ticker at once |
| `corporate_actions` | Splits, dividends, spin-offs, and mergers (cash or stock terms), with announced and effective times |
| `delisting_events` | Reason and delisting return, with method: `crsp_dlret`, `merger_consideration`, or `unavailable` |
| `raw_daily_bars` | Unadjusted OHLCV plus cumulative split and total-return factors |
| `universe_membership` | Universe id, rule version, interval, and an exclusion reason code: `price_below_floor`, `adv_below_threshold`, `insufficient_history`, `otc_listing`, `unsupported_filer_form`, or `compliance_restricted` |
| `trading_calendar` | NYSE sessions with open and close in UTC, including 13:00 ET early closes, plus a separate SIFMA bond calendar |

The NYSE calendar replaces today's union of observed dates. SIFMA differs
from NYSE: bond markets close on Columbus Day and Veterans Day; Good Friday
is an NYSE closure but only a bond early close [V].

**Survivorship-free price sources.** Owner decision; data is never committed.

| Source | Delisted securities | Index membership | Cost and terms [V] | Notes |
|---|---|---|---|---|
| CRSP via WRDS | Yes, with delisting returns | S&P 500 constituent file | Institutional subscription | Gold standard. **Check whether Georgia Tech provides student access** (unverified) |
| Sharadar (Nasdaq Data Link or QuantRocket) | Yes; prices from 1998 | S&P 500 from 1957 | Separate non-professional and professional licences | Cheapest point-in-time option; no CRSP-grade delisting returns |
| Norgate Platinum | Yes, from 1990 | Constituent flags | $630/year | Its update tool is Windows-based [I]; a poor fit on macOS |
| EODHD | Yes; depth varies | S&P 500 in the Fundamentals plan | About $60/month | History depth inconsistent |
| Massive (formerly Polygon) | `delisted_utc` flag | None | $29-$199/month | No membership history |
| fja05680/sp500 (GitHub, MIT licence) | No prices | S&P 500 from 1996, derived from Wikipedia | Free | Incomplete early years; dates confuse effective open vs close (AMD 2013) |

**Recommendation.** Build U1 from objective rules (no index licence needed)
on licensed survivorship-free prices with delisting returns. Prefer CRSP if
Georgia Tech access exists, otherwise Sharadar. Until then, label every
result "current-listing universe, survivorship-biased".

**How large the bias is** (simulated true null, 300 candidates, 15 years
[S]):
- With 2%/year failures and 2%/year takeovers, a survivors-only universe
  overstates equal-weight returns by +5.4 pp/year and shifts the IC by about
  +0.02.
- That is comparable to the simulated 100-name MDE, so survivorship alone can
  create or erase "significance".


## Part C: market-context layer

### Existing series (inspected 2026-10-08)

| Series | Where | Role | Notes |
|---|---|---|---|
| **DGS3MO**: 3-month Treasury constant-maturity yield, investment basis, H.15 (release 18) | `config/rates.yaml` (`USD_TREASURY_3M`); `data/rates/*`; `prepare_risk_free_returns` | **The project's risk-free rate.** Used only in evaluation (Sharpe and excess returns of the baselines); never a model feature | Simple accrual on actual days / 365 (or 366). **Local history covers only 2024-01-02 to 2026-08-28**, because `download_rates.py` hard-codes a first-download start of 2024-01-01. No `available_at`; storage overwrites revisions |
| UNRATE, CPIAUCSL, GDPC1, **FEDFUNDS**, M2SL | `config/macro.yaml`; ALFRED vintages | Macro features (not yet in the default model panel) | Vintage-correct. `available_at` is the release *date* only. FEDFUNDS is the monthly average effective fed funds rate: a policy-rate level, not the risk-free series |
| Treasury 2Y/5Y/10Y/30Y, slopes, TIPS, breakevens, VIX, credit, FX, commodities, crypto | — | — | **None exist anywhere in the repository** |

The risk-free rate is the 3-month bill yield, not a 5Y or 10Y yield. Longer
Treasury yields would be new context features, a separate concept from the
cash return.

### Signal cutoff, pre-execution review, and execution (owner decision)

Three moments are kept separate. The first version of this document called
the first one `decision_at`.

**1. `signal_cutoff_at(d)`: proposed 18:00 America/New_York on NYSE
session d,** stored in UTC (22:00Z in summer, 23:00Z in winter). It is the
latest information time for any feature or context value in row d. It falls:
- after every close, including 13:00 early closes;
- after the 16:15 ET VIX calculation end and the H.15 posting;
- after the 17:30 ET EDGAR acceptance cutoff;
- before the Tokyo open.

**2. Pre-execution review window: from `signal_cutoff_at(d)` until the
order cutoff before execution.**

Information keeps arriving after the cutoff and is **not** part of the
model's signal:
- after-hours earnings;
- overnight news and foreign sessions;
- 8:30 ET macro releases on d+1;
- the d+1 session itself.

A later review step, human or automated, may veto, defer, or resize orders
based on such events. Rules for that step:
- every review action is logged with its own timestamp, inputs, and reason;
- backtests do not simulate review decisions with information they could
  not have had point-in-time;
- an automated review rule becomes part of the strategy, with its own
  point-in-time data and validation.

**3. `execution_at(d)`: the close of session d+1,** matching
`MODEL_LABEL_SPEC` (the label return is measured from that close). Orders
must reach the closing auction before its submission cutoff, about 15:50 ET
[I]. Between the cutoff and execution there are about 22 hours, more over
weekends and holidays.

**The rule.** A value may enter row d only if
`available_at <= signal_cutoff_at(d)`.
- `available_at` is the timezone-aware moment *our source* published the
  value, not the moment the market traded.
- A derived value takes the latest `available_at` of its inputs.
- When the publication time is unknown, use the later (conservative) time
  and record the `availability_rule`.
- Values collected live also satisfy `available_at <= retrieved_at`.

**Why this matters.** Today the code has three implicit decision times:
- macro treats all of calendar day d as available;
- fundamentals cut off at 00:00 UTC on d;
- the panel validator cuts off at 00:00 UTC on d.

Unifying them on `signal_cutoff_at` is an availability-semantics change and
resolves open item 1. A new context layer can adopt `signal_cutoff_at` from
day one without touching existing contracts.

**Asynchronous markets.** Context must be aligned by availability, never by
matching dates.

| Source | Published [V] | Usable in row |
|---|---|---|
| ECB euro-area curve | Next TARGET business day, noon CET | d+1 for the value of d |
| Bank of England | By noon on the next business day | d+1 |
| Japan MOF JGB | 09:30 JST next business day (= 20:30 ET on d, after `signal_cutoff_at`) | d+1 |
| Coinbase BTC/ETH on FRED | As of 5 PM PST (20:00 ET) | d+1 |
| Coin Metrics | End of the UTC day | Row d uses the *previous* UTC day's value. A naive date join leaks about 2 hours |

### Candidate series: data facts

| Series | Source and history | Licensing | Revisions | Publication and `available_at` [V] | Calendar |
|---|---|---|---|---|---|
| UST 2Y, 5Y, 10Y, 30Y (DGS2/5/10/30) | H.15 / Treasury: DGS2 from 1976, DGS5 and DGS10 from 1962, DGS30 from 1977 (gap 2002-2006) | Public domain | ALFRED daily vintages for DGS10 from 2005 | Treasury quotes snapped at about 3:30 PM. H.15 posts at 4:15 PM showing the **prior** business day, and FRED reflects the T+1 value. Rule: value of d is usable in row d+1. Treasury's own par curve is usually out by 6:00 PM ET the same day: too close to `signal_cutoff_at` to rely on without measuring | SIFMA: "." on bond-only holidays, carried with a reason code |
| 10Y−2Y, 10Y−3M | Derived from DGS legs (FRED T10Y2Y/T10Y3M are same-day, 17:03 ET, but historical posting times are unknown) | — | — | Latest leg; both legs must share an observation date | As above |
| Cash proxy DGS3MO (DTB3 is discount basis) | H.15, from 1981 | Public domain | — | T+1 | As above |
| Real yields DFII5/DFII10; breakevens T5YIE/T10YIE | From 2003 | Public domain | — | DFII: T+1. Breakevens: same day on FRED. Derive the breakeven as DGS10 − DFII10 | As above |
| VIX | Cboe daily CSV from 1990; FRED VIXCLS (T+1 morning) | Copyright: citation required | VIXCLS vintages from 2010 | Calculation ends 3:15 PM CT (4:15 PM ET). From the Cboe file, row d is usable (rule `cboe_close_plus_buffer`) | NYSE |
| ICE BofA IG/HY OAS (BAMLC0A0CM, BAMLH0A0HYM2) | **On FRED and ALFRED only from 2023-10-09**; limited to 3 years from April 2026 | Copyrighted; ICE pre-approval required | Lost | T+1 | — |
| Credit-risk proxy (not a credit spread): HYG and LQD returns relative to IEF | yfinance ETF closes from 2007 | Vendor terms | None | 16:00 ET, row d | NYSE |
| Broad USD (DTWEXBGS) | H.10 from 2006 | Public domain | "Past releases are not revised", though ALFRED holds weekly vintages | **Weekly: Monday 4:15 PM** for the prior week, so up to about 10 days stale | NY noon rates |
| UUP (dollar ETF) | 2007 | Vendor terms | None | 16:00 ET, row d | NYSE |
| WTI and Brent spot (DCOILWTICO, DCOILBRENTEU) | EIA (LSEG prices); 1986 and 1987 | Public domain | Weekly vintages | **Weekly, Wednesday**: up to about 8 days lag | — |
| USO and BNO (oil ETFs) | 2006 and 2010 | Vendor terms | None | 16:00 ET, row d. Futures roll drag: use returns, not levels | NYSE |
| Gold and silver (LBMA) | **Removed from FRED on 2022-01-31** | IBA licence required | — | — | London auctions |
| GLD and SLV (gold and silver proxies) | 2004 and 2006 | Vendor terms | None | 16:00 ET, row d | NYSE |
| Copper proxy: CPER (futures-based ETF) or COMEX HG futures | CPER from about 2011 [I]; HG continuous futures longer [I] | CPER: vendor terms; CME futures data: licensed | None | CPER: 16:00 ET, row d. HG settlement time to be verified [I] | NYSE (CPER); CME Globex (HG) |
| Bund / euro area | ECB from 2004 (noon CET T+1); Bundesbank (same day, snapshot time unknown) | Public | — | Row d+1 | TARGET |
| UK gilts | Bank of England | BoE terms; no API | Can be revised when the bond set changes | Next business day noon: row d+1 | UK |
| Japan JGB | MOF from 1974 | — | — | 09:30 JST next business day: row d+1 | Japan |
| Korea KTB | Bank of Korea (ECOS) | API key (secrets gate) | — | **Unverified** | Korea |
| OECD monthly long-term rates (IRLTLT01...) | 1956-2000 starts | Copyright: citation required | Monthly vintages | About 2 weeks after month end | — |
| BTC and ETH | Coin Metrics community data (UTC end of day); Coinbase on FRED from 2014 / 2016 | Coin Metrics: CC BY-NC 4.0; Coinbase: reproduction prohibited without permission | Coinbase vintages from 2018 | See the asynchronous table above | 24/7; weekend moves fold into Monday |

### Candidate series: feature design

**Every context series is date-common: the same value for every stock on a
date.** With a per-date rank IC, or with an excess-over-equal-weight label
whose labels sum to zero each date, a date-common feature carries **no**
information on its own; with the excess label Ridge, Lasso, and Elastic Net
give it a coefficient of exactly zero [S]. It can matter only through a few
pre-registered **interactions** with stock characteristics (market beta,
rate beta, USD beta, volatility, momentum). With a raw label it can only do
market timing, which needs an R² of about 3-10% per 20 sessions to detect.
That is implausible at this horizon.

| Feature (V1 unless noted) | Transformation | Interaction carrying it | Redundancy | Missing policy | Leakage risk |
|---|---|---|---|---|---|
| UST 2Y and 10Y | 20- and 63-session changes (bp); 252-session trailing z-score of the level | Δ10Y × rate beta (duration-like sensitivity of long-dated growth names) | 5Y and 30Y redundant (level and slope explain most curve variance) | Bond-only holidays: carry the last value with age; no other forward-fill | Same-day H.15 values: use T+1 |
| Slopes 10Y−2Y and 10Y−3M | Level and 20-session change | × beta, × cyclicality | Overlap with the 2Y/10Y changes: keep slopes plus Δ10Y | Both legs required | Mixing observation dates |
| 10Y real yield (DFII10); breakeven = DGS10 − DFII10 | 20-session change | × duration / valuation characteristic | Nominal = real + breakeven: keep two | From 2003 | — |
| VIX | Log level, 252-session z-score, 20-session change; plus realized market volatility from existing prices | × beta, × momentum (momentum crashes tend to follow high-volatility periods) | Correlated with the credit-risk proxy | Cboe history from 1990 | FRED VIXCLS is T+1: use the Cboe close or lag it |
| Credit-risk proxy (not a spread) | HYG − IEF and LQD − IEF 20-session relative total returns. These differ from an option-adjusted spread: they mix credit with duration mismatch, fund fees, distributions, and ETF liquidity | × leverage, × idiosyncratic volatility | Partly VIX | ETFs from 2007 | Labelling it a "spread" invites misinterpretation |
| Equity-derived regime (no new data) | Universe equal-weight return, dispersion, average pairwise correlation, index realized volatility | Context for every interaction | — | Warm-up reason code | Must use only names eligible at d |
| Stock characteristics (no new data) | Trailing 252-session beta, idiosyncratic volatility, 12-1 momentum | These *are* the interaction partners | — | Warm-up | Fitted only on past data |
| V2: USD (UUP 20-session return; DTWEXBGS as a lagged cross-check) | Return, change | × USD beta, foreign revenue share | — | Weekly staleness for DTWEXBGS | DTWEXBGS daily values are published a week later |
| V2: Brent via BNO return (EIA spot as a lagged cross-check) | 20-session log return, realized volatility | × energy-input sensitivity | WTI is redundant | ETF from 2010 | EIA spot is weekly |
| V2: VIX3M/VIX ratio; semiconductor industrial production (IPG3344S, ALFRED) | Ratio level; year-over-year growth from vintages | Sector cycle × sector sleeve | — | Monthly, vintaged | Revisions: use vintages |
| V2 candidates: gold (GLD), silver (SLV), copper (CPER or HG) | 20- and 63-session log returns, realized volatility | Gold × real-yield sensitivity; copper × cyclicality and the sector sleeve; silver × industrial demand | Gold is largely real yields plus USD; silver is highly correlated with gold. One representation per factor until ablation shows incremental value | ETF inception dates; reason codes | See "Commodity proxies" |
| Defer: foreign sovereign yields | 20-session changes | × international revenue | Strongly correlated with US yields | Next-day publication | Date joins leak |
| Defer: BTC | 20-session return, realized volatility, from Coin Metrics UTC-day values | Speculative-risk appetite × high-beta names | ETH redundant; the GPU-mining link broke at the ETH Merge (2022) | 24/7, weekend folding | UTC-day boundary |

### Ranking and recommendation

**Core V1: before serious model comparison.** High point-in-time
reliability, low engineering cost, clear mechanism.
1. **Backfill DGS3MO to at least 2009.** Needed for cash returns and Sharpe in
   T5. Fix the 2024 start and record `available_at` (T+1).
2. **UST 2Y and 10Y from FRED, with slopes 10Y−2Y and 10Y−3M derived** (T+1
   rule).
3. **DFII10, and the derived breakeven** (T+1 rule).
4. **VIX from Cboe's daily file** (row d).
5. **Credit-risk proxy from HYG/LQD relative to IEF**, always labelled a
   proxy and never a credit spread. FRED's ICE spreads are unusable before
   2023 and restricted.
6. **Equity-derived regime features and stock characteristics.** These need
   no new data, are probably the most valuable, and are the interaction
   partners for everything above.

**V2: after the first linear baselines** (each as a pre-registered
ablation arm):
- USD (UUP; DTWEXBGS as a lagged cross-check);
- Brent (BNO; EIA as a lagged cross-check);
- VIX term structure;
- semiconductor industrial production (vintaged);
- gold, silver, and copper through point-in-time-compatible proxies (below).

**Deferred candidates.** These are economically plausible but low priority
now. Each needs a pre-registered hypothesis and its own ablation increment,
and none enters while a more direct representation of the same factor is
already in the model.
- German, UK, Japanese, and Korean yields: next-day publication, new
  adapters, an unknown Korean lag, low incremental value for a US universe.
  Revisit with U3.
- BTC: licence choice and UTC-boundary handling.
- ETH: redundant while BTC is in.
- WTI: redundant while Brent is in. WTI futures printed negative on
  2020-04-20, so any WTI feature needs a documented treatment and a
  returns-based proxy.
- Brent−WTI spread: reflects US oil logistics. Revisit if energy, transport,
  or chemicals names enter a broader universe.
- Gasoline, diesel, and refined products: refiner and transport economics.
  Same revisit condition.
- Gold/silver ratio: redundant while gold and silver are both candidates.
- DGS5, DGS30, and FRED's ready-made slope and breakeven series: use these
  only as cross-checks.

**Rejected: licensing or data-contract constraints only.** Rejection is
reserved for these cases; it is not used for low priority.
- ICE BofA spreads from FRED: truncated history and restricted licence.
- LBMA gold and silver via FRED: IBA licence required, removed from FRED.
  Gold and silver remain V2 candidates through proxies.
- ICE DXY: licence.
- Coinbase BTC/ETH via FRED: reproduction prohibited without permission, and
  stamped after the cutoff. Coin Metrics is the candidate source.

**Is one crude benchmark enough?** For the current and planned universes,
Brent alone represents oil-market conditions adequately as a first step. WTI
adds little beyond its correlation with Brent. Refined products and the
Brent−WTI spread matter mainly for refiners, airlines, and chemicals; they
stay deferred candidates if such names enter a broader universe.

### Commodity proxies: gold, silver, copper (V2 candidates)

Direct spot series are unavailable or poorly licensed (LBMA gold and silver
need an IBA licence; CME futures data is licensed). The candidates are
therefore liquid market proxies:
- **gold:** GLD;
- **silver:** SLV;
- **copper:** CPER, a futures-based ETF, or a COMEX HG continuous futures
  series if a licensed, timestamped source is approved.

**An ETF proxy is not the underlying commodity.** Document each of these for
every proxy:

| Aspect | What differs |
|---|---|
| Tracking error | GLD and SLV track the LBMA prices less fees; CPER tracks a copper futures index, not spot copper |
| Trading hours | ETFs close at 16:00 ET on NYSE Arca. LBMA auctions are in London (gold 10:30 and 15:00, silver 12:00). COMEX trades nearly 24 hours with an earlier daily settlement [I] |
| Fees and distributions | Expense ratios reduce returns. GLD and SLV are grantor trusts with no distributions [I]. Futures funds also earn collateral (T-bill) income |
| Futures roll | CPER, USO, and BNO roll contracts. Contango or backwardation makes ETF returns differ from spot returns, so use returns, never levels |
| `available_at` | ETF close: 16:00 ET, row d. Futures settlement: same day only from a licensed, timestamped source. LBMA prices: licence required |

**Redundancy discipline.** Use one representation per economic factor until
ablation shows incremental value:
- GLD *or* gold futures, not both;
- silver only if it adds beyond gold;
- the gold/silver ratio stays deferred while both are candidates.

Gold, silver, and copper enter the A2 ablation arm.

### Ablation design (pre-registered before any run)

**Nested arms on identical everything:**
- A0: equity-only (existing features plus equity-derived regime features and
  characteristics);
- A1: A0 + US macro and rates (yields, slopes, real yield and breakeven)
  **plus the V1 risk-regime block: VIX and the credit-risk proxy.** These
  are V1 priorities, so they enter here, with US rates, not with crypto;
- A2: A1 + global rates, FX, and commodities: USD, Brent, the gold, silver,
  and copper proxies, and foreign rates (the last only with U3);
- A3: A2 + crypto (BTC);
- placebo arms: each arm's context series circularly shifted by random
  offsets of at least 252 sessions, or shuffled in 63-session blocks. These
  give an empirical null for the improvement.

**Held identical across arms:**
- the fold table (checked by hash) and the rows (the window where every
  arm's data exists);
- the estimator (Ridge), the hyperparameter grid, and nested inner purged
  selection;
- the excess label;
- seeds, the weighting rule, and costs.

Each arm is written to the ledger *before* it runs. V2 risk-regime extras,
such as the VIX term structure, are tested as their own pre-registered
increment on top of A1, not bundled with crypto.

**Primary metric.** The paired per-date rank-IC difference of each arm over
the one before, with Newey-West lag ≥ 40 or a 63-session block bootstrap,
one-sided. Test in a fixed sequence (A1, then A2, then A3) at α = 0.05 and
stop at the first failure. This controls the family-wise error rate without
splitting α; Holm across the increments is the fallback if the order is ever
not nested.

**Secondary checks:**
- each pre-registered interaction keeps its sign in at least 75% of folds;
- Clark-West test (raw-label track only);
- net information ratio against equal weight;
- deflated Sharpe ratio using the ledger's variant count.

**Power declared in advance.** The paired standard error is about
√(2(1 − ρ)) times one model's SE, where ρ is the correlation between the two
arms' daily ICs. On U0 the ablation is declared non-informative before it
runs.


## Sequencing against T4

| Work | Runs in parallel with T2/T4 | Needed before T4 results are treated as product evidence (current planning targets) |
|---|---|---|
| T4 on U0 | Yes. Engineering exercise, recorded in the ledger, never reported as evidence | — |
| Security-master design and tables, tested on synthetic panels | Yes | Yes |
| Survivorship-free data source and licence (owner) | Yes | Yes |
| U1 at its planning target (about 100 point-in-time large caps, history from 2010), with the MDE re-estimated on the real panel and a survivorship sensitivity bound | Yes | **Yes** |
| `signal_cutoff_at` approved; context layer on timestamp-level availability | Yes | **Yes** |
| V1 context ingestion with prefix-stability tests; DGS3MO backfill | Yes | **Yes** (V1 only) |
| Equity-derived regime features and characteristics | Yes | Yes |
| T2 null models, canary, ledger; inference with calibrated false-positive rate (Newey-West lag ≥ 40 or block bootstrap) | Before T4 | **Yes** |
| Excess-label decision; T5 execution lag and costs; pre-registered decision rule; MDE computed before reading results | — | **Yes** |
| V2 context, ablation arms, U2, U3, sector sleeve | After the first baselines | No (they refine, not gate) |


## Owner decisions

1. **Product scope.**
   - A broad large-cap research universe (U1, planning target about 100
     names) is the path to product evidence under current assumptions. The
     target is re-estimated on the real panel.
   - A semiconductor-only product can detect only IC ≥ 0.05 under the same
     assumptions.
2. **Survivorship-free data:** CRSP via WRDS (check Georgia Tech access),
   Sharadar, or another source, plus its licence and cost.
3. **`signal_cutoff_at` = 18:00 ET on session d.** This is kept separate
   from the pre-execution review window and from `execution_at` (the close
   of d+1). Also decide whether to migrate the existing macro and
   fundamentals joins to it (availability-semantics gate; open item 1).
4. **Excess-over-equal-weight label** (still pending). It makes context
   features interaction-only.
5. **Dependencies and keys:**
   - a trading calendar (`exchange_calendars` or an in-repo holiday file);
   - any data-vendor client;
   - a Bank of Korea API key (secrets gate).
   - `requests` is already imported but not pinned in `requirements.txt`.
6. **Compliance:** confirm that NS-CMIC-restricted securities (e.g. SMIC)
   are excluded from every investable universe.


## Follow-ups (documented, not implemented)

1. **Two-layer architecture.** Evaluate splitting the model in two:
   - a cross-sectional excess-return (alpha) model for relative weights;
   - a separate market/regime model that sets overall risk and cash
     exposure.

   Date-common context features cannot change cross-sectional ranks except
   through interactions, but they are the natural inputs for a timing layer.
   Separating the layers keeps evaluation honest: rank IC for the first;
   timing metrics against constant exposure, drawdown, and volatility for
   the second. That keeps market timing from passing as stock selection.

   Start the timing layer with simple, explainable baselines, such as
   volatility targeting and VIX or credit-proxy risk-off rules. Timing is
   hard to validate (Goyal-Welch). It needs the DGS3MO backfill and its own
   label: universe return in excess of cash.
2. **Risk-free conversion check.** Verify how the quoted DGS3MO yield
   becomes the period return used in Sharpe and excess-return calculations.
   `prepare_risk_free_returns` (`data/rates/transform.py`) applies simple
   actual/365 (366) accrual per holding period, using the latest rate on or
   before the period start. Check:
   - the investment-basis (bond-equivalent) quote against money-market
     actual/360 and discount-basis conventions (DTB3);
   - a constant-maturity yield used as a proxy for realized bill returns;
   - compounding across periods, and weekend and holiday accrual;
   - consistency with how Sharpe ratios are annualized (252 sessions vs
     calendar days);
   - T+1 availability at the period start.

   Add a worked numerical example and tests when it is implemented.
3. **Deferred vs rejected.** Rejection is reserved for licensing or
   data-contract constraints. Economically plausible, low-priority series
   (WTI, silver, ETH, the gold/silver ratio, refined products) are deferred
   candidates, each needing a pre-registered hypothesis and an ablation
   increment.
4. **Gold, silver, copper.** V2 candidates through point-in-time-compatible
   proxies (GLD, SLV, CPER, or licensed COMEX futures). Each proxy is
   documented against its underlying (tracking error, hours, fees and
   distributions, roll, `available_at`), with one representation per factor
   until ablation shows incremental value.
5. **Signal cutoff vs review vs execution.** If a pre-execution review step
   is introduced, define its logging, its point-in-time inputs, and how
   backtests treat it before using it.
6. **Credit naming.** Measures from HYG, LQD, and IEF are labelled credit-risk
   proxies throughout, never credit spreads.


## Sources (verified 2026-10-08)

- S&P DJI press releases: AMD removed effective after the close on
  2013-09-20 (press.spglobal.com, 2013-09-11); re-added 2017-03-20
  (2017-03-10); added to the S&P 100 on 2022-09-19 (2022-09-02).
- AMD IR: Nasdaq transfer (ir.amd.com/news-events/press-releases/detail/585);
  Xilinx closing (ir.amd.com/.../detail/1047).
- SEC EDGAR submissions: APLD (CIK0001144879.json), SMIC
  (CIK0001267482.json), MRVL (CIK 1835632, local cache). APLD IPO release
  0001104659-22-045556.
- OFAC consolidated non-SDN list (cons_prim.csv), program CMIC-EO13959; OFAC
  Chinese Military Companies Sanctions page.
- WRDS "Downloading the S&P 500 constituents"; github.com/fja05680/sp500;
  norgatedata.com/stockmarketpackages.php; quantrocket.com/pricing/data/sharadar;
  massive.com/pricing; tiingo.com/about/pricing; crsp.org.
- FRED/ALFRED pages and data files for DGS3MO, DGS2, DGS5, DGS10, DGS30, DTB3,
  T10Y2Y, T10Y3M, DFII5, DFII10, T5YIE, T10YIE, VIXCLS, BAMLC0A0CM,
  BAMLH0A0HYM2, DTWEXBGS, DCOILWTICO, DCOILBRENTEU, CBBTCUSD, CBETHUSD,
  IRLTLT01DEM156N.
- federalreserve.gov/releases/h15 and /h10; Treasury yield-curve methodology
  (home.treasury.gov).
- Cboe VIX specifications and historical data; EIA spot prices (eia.gov/dnav/pet).
- FRED notice on removing IBA data (2022-01-10); lbma.org.uk precious-metal
  prices.
- ECB euro-area yield curves; Bundesbank daily term structure; Bank of
  England yield curves; MOF Japan JGB interest-rate Q&A.
- docs.coinmetrics.io (price); github.com/coinmetrics/data; Talos notice on
  Coin Metrics community data.
- sifma.org holiday schedule; nyse.com hours and calendars.
