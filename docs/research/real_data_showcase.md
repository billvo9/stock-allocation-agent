# Exploratory real-data rebalancing showcase

**Status.** Exploratory and private (owner only). The rules and measures in it
are already approved; it adds no formula. Its numbers are not evidence of
strategy skill, and they are not advice.

## Owner decisions (2026-10-26, New York)

- **Scope.** The owner approved an exploratory real-data rebalancing
  showcase: "I want more stocks and industries than only AMD, MRVL, MU and
  NVIDIA. and Yes just don't add new formula except we really need one."
- **Audience.** "Only me (Recommended)". It is a private report on the owner's
  Mac, with no redistribution, which the Tiingo free plan's terms also require.
- **Universe.** The owner approved the three universes below, as proposed,
  answering "Universe OK (Recommended)".
- **Fetch.** The owner authorized the one fetch run (32 Tiingo requests, the
  key from the Keychain, written only to the showcase store) and the local
  report: "Go, run it for me (Recommended)".
- **Boundary tests.** The owner approved two additions to
  `tests/test_stock_lab_boundaries.py`, the showcase package as a Stock Lab
  consumer and `showcase_fetch.py` as a third adapter user, answering
  "Approve (Recommended)". Nothing else was loosened, and the inference-store
  boundary is unchanged.

**Not approved, and not used:**
- new formulas;
- group C (cash returns, benchmarks) and group D (CAGR, Sharpe, intervals,
  significance tests);
- any 2025+ data, which is the sealed holdout;
- PPO, CEM or mean-variance on real data.

## What runs

There is one registered configuration,
`configs/showcase/real_rebalancing_v1.yaml`, written before any data was
fetched. It has 3 universes and 6 rules, so 18 variants. Each variant is
replayed at 0x, 1x and 2x the D2 provisional cost rates (54 simulations).
Every variant is reported, in registered order; none is chosen afterwards.

| Rule | Rebalanced every | Definition (the registered classes in `rl.policies`) |
|---|---|---|
| Buy and hold | never after the build | Equal weights at the build, then drift |
| Equal weight | 5, 21, 63 sessions | 1/n in each holding |
| Inverse volatility | 21, 63 sessions | w_i ∝ 1/σ_i, with σ_i the sample standard deviation (ddof 1) of the last 60 visible returns |

**Engine (T5a D1-D8, approved 2026-10-15).**
- Each decision sees closes only through its own session (`collect_targets`)
  and trades at the next close.
- Every portfolio builds from cash at one common close: decision index 60
  (2003-03-31 in the frozen calendar), executed at 2003-04-01. Starting there
  gives inverse volatility 60 returns before its first decision.
- Costs are charged on every trade, the build included:
  - sector ETFs pay the `broad_etf` rate, 5 bp (the 2x run, 10 bp, equals the
    single-stock rate);
  - stocks pay 10 bp.
- Long only and fully invested. The engine's funding leg gets 0% in every
  target, so no cash return ever reaches a wealth path. A test changes it to
  5% a session and checks that nothing moves.

**Returns.** F5.1 (OD8 group A), r_t = (f_t·P_t + D_t)/P_s − 1, computed from:
- raw closes P;
- Tiingo's split factor f;
- Tiingo's cash dividend D on its ex-date, labelled "vendor-reported, not
  issuer-verified".

A session the vendor did not serve stays a gap, and the next close carries a
labelled multi-session return. A dividend and a split on the same session
have no recorded share basis, so that return is unavailable, and a portfolio
holding the stock stops (`distribution_in_gap`) rather than guessing.

**Measures (approved only).**
- Growth of $10,000: the engine's pre-trade wealth path.
- Total return at the window's last session. It is unavailable if the run
  stopped earlier.
- Maximum drawdown (A6). Over an incomplete history it is labelled the
  largest observed decline.
- Calendar-year returns (A7). Part years are labelled and never annualized.
- Turnover (F2.4, one way): the build, the rebalances and the total, never
  annualized.
- Costs (D2): dollars paid, the fraction of the initial amount, and summed
  basis points, plus the final wealth at 0x and 2x.
- Each holding's own growth over the same span (F5.4, no cost), as context.

## The rebalancing-day (offset) family (F7.2)

- **Approved:** F7.2 under T5a D7 (2026-10-15). It was requested for the showcase on
  2026-10-26.
- **Registered** in `configs/showcase/real_rebalancing_v2.yaml` before it was run. That file
  is version 1 plus `offset_family: true`, and a test checks nothing else differs.
- **What runs.** Each rebalancing rule runs at every offset o in [0, k) of its cadence
  (weekly k = 5, monthly 21, quarterly 63), at 1x costs. Every offset shares the common
  build close, so the rebalancing days move, and with them the rebalance count by at most
  one and the first rebalance's distance from the build.
- **What is reported**, exactly F7.2's wording:
  - The registered offset (0) is the headline. Its run is the headline run itself, and the
    family never chooses it.
  - The family adds the max − min spread of the total return across the k offsets. Which
    offset was highest or lowest is not recorded or shown.
  - The spread is unavailable if any offset stopped early, and there are no per-offset
    intervals.
- **Counting.** A rule's k offsets are one variant with a k-member sensitivity family:
  `variants_run` stays 18, and `offset_family_simulations` records the extra runs.
- **Buy and hold** never rebalances, so the offset changes nothing for it, and it has no
  family.

## The Lab app's Strategies screen

More > Strategies is a small, read-only screen, as the owner chose: "Yes, small and
read-only (Recommended)" (2026-10-26).
- It reads only the newest saved `results.json`, with the standard library. It recomputes
  nothing and imports no showcase module, so the app's import allowlist is unchanged.
- It shows the same caveats, the growth of $10,000 for each list, and the results table with
  the rebalancing-day spread.
- It is behind the app's per-launch key, like every data route. The demo reads nothing real.
- The design is in `docs/product/lab_app_design.md`, "Strategies in detail".

## The universes

1. **Nine sector funds:** XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY.
   - They are the original Select Sector SPDRs, listed in 1998, so they cover
     the whole window.
   - A fund reinvests or sells what a spin-off gives it, so spin-offs do not
     distort its price.
   - Closes are market prices, not NAV.
2. **Eighteen sector stocks, by a written rule.** Source: the Select Sector
   SPDR Trust's annual report as of 2002-09-30 (SEC EDGAR accession
   0000950135-02-005333, filed 2002-11-27, before the window).
   - In each fund's printed "Top five holdings", take the two largest. Skip a
     company that:
     - did not trade under one ticker for all of 2003-2024; or
     - made a pro-rata spin-off or other non-cash distribution in that
       period. The data has cash dividends only, so such a distribution
       would read as a loss.
   - Materials' top five all fail, so its full schedule is read further down.
   - The result: WMT, HD; KO, PG; XOM, CVX; C, BAC; JNJ, LLY; UPS, LMT; MSFT,
     INTC; SO, D; NEM, PPG.
   - The skip list (VIA, AOL, MO, AIG, PFE, MRK, ABT, GE, MMM, IBM, DUK, EXC,
     DD, DOW, AA, IP, APD, WY, PX) is from public corporate history, as far as
     we know.
3. **The four Stock Lab semiconductors:** AMD, MRVL, MU, NVDA. They were picked
   in 2026 after they rose enormously. They are kept apart so that their
   hindsight does not leak into list 2.

**Checks on the stored data (2026-10-26).**
- XLF's 2016-09-19 distribution of real-estate fund (XLRE) shares, a
  fund-level spin-off, is in the vendor's dividend record as $4.44. F5.1
  therefore books +0.55% that day, not a false 19% loss.
- The stock list keeps companies whose separations were exchange offers:
  PG (2008, 2016), PPG (2013), LMT (2016), LLY (2019) and JNJ (2023). An
  exchange offer does not lower the price of shares kept, but an
  undersubscribed one ends in a pro-rata clean-up spin-off. No unusual
  one-day fall appears within a week of any of them; the largest is −3.7%.
- Every ticker's two worst days are known market events (2008, March 2020,
  company news). None looks like an unrecorded distribution.

**Assumptions to know.**
- The sector funds pay the approved `broad_etf` rate. Putting narrower
  sector funds in that class is an interpretation, so the 2x run (the
  single-stock rate) is shown beside it.
- Turnover and costs are reported as the registered comparison reports
  them (#88: summed, per rebalance, costs paid), not annualized. How F2.4's
  "per calendar year" is read is still open (group D, M6).

**Biases, plainly:**
- Every list was chosen in 2026.
- List 2 keeps only companies that survived under one ticker, which is
  survivorship bias, and its spin-off screen is best effort.
- List 3 is hindsight-selected by construction.

## Data path and commands

- **Store.** The showcase has its own private store,
  `~/stock-agent-private/showcase_store/showcase.sqlite3`. It reuses the
  Stock Lab's write-once store code, but it refuses:
  - a location in or under the Stock Lab research store's directory;
  - a folder (or any folder above it, below home) that holds another SQLite
    store, so it never sits beside or inside the inference store either;
  - any location inside a Git working tree.
- **Boundaries (owner-approved).** The Stock Lab's boundary tests now list the showcase as the
  one other consumer of the Stock Lab store, symbols, audit and adapter
  modules, and `showcase_fetch.py` as the third adapter user. The inference
  store stays unnamed outside its own package.
- **What is never touched:** the Stock Lab's watchlist, its stores, the
  frozen model, the evening job, the runner clone, launchd and the Keychain
  item's settings.
- **Classification.** `symbols.classify` takes an optional `asset_types`
  argument. The default (stocks) is unchanged. The showcase registers the
  sector funds as ETFs, and a ticker listed under another type is refused
  (`asset_type_not_registered`).
- **2025 on.** The config loader refuses a window past 2024-12-31. The fetch
  requests end on 2024-12-31. The frozen NYSE calendar ends there too. A
  stored 2025 session refuses the read instead of being dropped.
- **Request budget.** One run makes 32 requests: the listing and 31 prices.
  That is under the free plan's 50 an hour. The audit and the store happen in
  the same run, from the same responses, so the budget is not spent twice.
  Authorized runs are refused on weekdays from 19:30 to 20:45 New York,
  because the 20:30 evening job uses the same key.

Commands, run from the repository root:

```bash
.venv/bin/python scripts/showcase_fetch.py
```

```bash
.venv/bin/python scripts/showcase_fetch.py --authorized --store --keychain
```

```bash
.venv/bin/python scripts/showcase_run.py
```

- The first prints the plan, with no request.
- The second audits every ticker and stores the audit-ready ones. It is a
  live vendor request and a write, so it needs the owner's go.
- The third has no network. It writes `results.json`, `report.html`,
  `report.md` and the fonts into a new folder,
  `~/stock-agent-private/showcase/<run_id>/`.

## Learning Notes

- **Problem.** What did simple, pre-registered rebalancing rules do on real
  prices, after costs? And what can such a picture honestly support?
- **Concept: rebalancing as a bet on reversal.** Buy-and-hold lets winners
  grow; the portfolio drifts toward whatever rose. Rebalancing to equal
  weights sells what rose and buys what fell.
  - It helps when relative moves reverse, so assets take turns.
  - It hurts when one asset keeps rising, as the semiconductors did.
  - More frequent rebalancing trades more, and every trade pays the cost.
  - Inverse volatility adds a tilt toward the calmer holdings.
- **Assumptions.**
  - Trades happen at the close at a flat provisional cost, in fractional
    shares.
  - Dividends are reinvested on the ex-date.
  - No taxes are charged, and nothing is held in cash.
- **Small example.**
  - Start: two assets at $5,000 each. A doubles; B is flat.
  - Buy-and-hold ends at $15,000: $10,000 in A and $5,000 in B.
  - Rebalancing back to 50/50 halfway, when A is up 41.4% (√2), sells
    $1,036 of A at 10 bp (≈ $2). A's second half then earns 41.4% on $6,036
    instead of $7,071.
  - The rebalanced portfolio ends near $14,571 before costs, below
    buy-and-hold, because the trend continued.
  - With a reversal, the order flips.
- **Leakage risks, and how they are closed.**
  - Inverse volatility sees only the 60 returns through its decision close.
    A test changes every later return and checks the decision is unchanged.
  - Trades happen at the next close (D1).
  - The universes are the leak this report cannot close: they were picked
    knowing the outcome. That is why they are labelled, why the
    semiconductors are a separate list, and why nothing here supports a
    "this rule wins" reading.
- **How results could mislead.**
  - It is one path of 22 years. 2008 and the 2020s semiconductor rally
    dominate it.
  - The differences between rules are untested: no interval or significance
    test is approved.
  - A log-scale chart compresses large gaps.
  - Vendor dividends can be wrong or missing.
- **How practitioners would validate it.**
  - Point-in-time constituents, from the index's membership as of each date,
    so that delisted names are included.
  - Issuer-verified distributions, including spin-off values.
  - Many start dates, or offsets (F7.2).
  - Interval estimates, such as a block bootstrap on the difference between
    two rules (group D).
  - A benchmark and a cash leg (group C).
- **The rebalancing day matters.** The offset family shows how much a result moves when only
  the rebalancing day changes, a pure timing accident. The spread is a description, not a
  test.
- **What to learn next.**
  - Survivorship bias, and point-in-time index membership (CRSP or the
    index's own history).
  - The rebalancing premium versus momentum.
  - Why turnover times cost compounds into a drag.

## Follow-ups (not done)

- Group C and group D measures, which wait for the owner's decisions.
- Issuer-verified distributions and point-in-time index membership, for a less biased stock
  list.
