# Group C: cash, T-bill returns and benchmarks (decision packet, 2026-10-17)

**Status: APPROVED by the owner on 2026-10-26 (New York), in full: "Approve all C1–C9 (Recommended)", the recommended options of `group_c_d_decision_index.md` (C1–C6 (a), C7 (a) with benchmarks outside the constraint set, C8 (a), C9 (a)); the option included authorizing a one-time FRED download of 2003-2024 T-bill history. Recorded in `docs/PROJECT_STATE.md` and `t5_portfolio_formulas.md` section 9.** The text below is the packet as approved; each item is implemented in its own reviewed change, and its status there says so.

**Why now.** The owner's 2026-10-17 request: make future real-data comparisons of PPO against the baselines defensible, without tuning PPO against the synthetic generator again. D4 (a) (uninvested cash earns 0%) was approved on 2026-10-15 for synthetic engineering only, with the condition "Group C remains necessary for real economic comparisons". The synthetic comparison (`ppo_comparison.md`, section 9) shows why: 0% cash penalizes every cash-heavy result, including the cash a cap or a partial build leaves.

**How to read it.**
- Decision ids C1–C9 are new (C9, the booked-state exception, follows C8 so that C2 keeps its number). D1–D8 remain the T5a ids (`t5a_open_decisions.md`); group D's decisions are M1–M9 (`group_d_decisions.md`). The combined index with a suggested reply is `group_c_d_decision_index.md`.
- Formula ids (F3.x, F4.x) refer to `t5_portfolio_formulas.md` sections 3 and 4. This packet asks only what needs the owner's judgment, in the D1–D8 format: the rule, the exact equation, a worked example, the options and a recommendation.
- **Every number in bold** is computed by `scripts/group_cd_packet_numbers.py` (keys `C1.*` to `C8.*`). `tests/test_group_cd_packet_numbers.py` checks both directions: every computed value appears here, and every bold number here is a computed value. Numbers not in bold are stated inputs (a 5% rate is an illustration, not an observed value), restatements of a bold number, or figures quoted from approved documents.
- Evidence tags: **[V]** verified on 2026-10-17 from the FRED and ALFRED pages for DGS3MO and the FRED API terms of use; **[S]** from secondary sources, to be checked against the issuer; **[I]** from memory, unverified; **[A]** an assumption to approve.

**Already decided, and not asked again.** F5.1–F5.4 (total return from raw closes, distributions and splits; A1–A4); the fee rule (A5: fund expenses already in the price are never deducted again); the D2 rates (provisional); D3 ("Every portfolio and benchmark starts as 100% cash at one common first execution close and pays D2 to build", no exit charge); D4 (a) for synthetic engineering; D5 and its 2026-10-17 confirmations; D7. **OD7** (approved 2026-10-11) already covers retrieving "the DGS3MO backfill" and broad fund total returns "from the existing personal-use sources, ending 2024-12-31". What OD7 does not approve is how those inputs become returns and when they count as known: that is this packet.

**Notation.**
- t−1 and t are consecutive NYSE sessions (the frozen calendar, `calendars/nyse.py`).
- y_s is the DGS3MO yield for H.15 date s, as a decimal: the 3-month Treasury constant-maturity yield, quoted on an investment (bond-equivalent) basis.
- r_cash,t is the cash account's return booked at close t, for the span (t−1, t]. rf_t is the risk-free return for the same span. R_t is a portfolio's net return for the span.

---

## C1. The cash-account return (OD8.5; F3.1, F3.4)

**Rule.** Over each span between consecutive NYSE closes, the cash account earns the DGS3MO yield for the span's start date, accrued simply on actual calendar days.

**Equation.**

```
r_cash,t = y_{s(t−1)} × Σ_j days_j / B_j
```

- s(t−1) is the latest DGS3MO date on or before t−1 (C2 sets which observations count).
- The span (t−1, t] is split at each January 1. days_j counts the calendar days of segment j, and B_j = 366 in a leap year, else 365.
- The engine compounds per session: the cash column's return enters `portfolio_step` like any other asset's. Across a D6 gap, the cash growth is the product of the per-session accruals inside the span.
- Label (F3.4): "Cash: 3-month Treasury bill rate approximation (FRED DGS3MO), before fees".

The existing `prepare_risk_free_returns` (`data/rates/transform.py`) already computes exactly this, and the test checks that it reproduces the examples below. Approving C1 adds no new arithmetic: it approves this series as the cash leg for real data (and C3 as the risk-free rate), replacing D4 (a) there.

**Examples.**
- **Convention check.** A 91-day bill bought at 98.75 per 100 of face value returns **1.2658%** held to maturity.
  - Its investment-basis yield is (100 − 98.75)/98.75 × 365/91 = **5.0772%**, and 5.0772% × 91/365 gives back 1.2658% exactly.
  - Its discount yield (the DTB3 quote) is 1.25/100 × 360/91 = **4.9451%**. Used as if it were an investment yield, it accrues **1.2329%**: **−3.29 bp** a quarter.
  - The investment yield on actual/360 overstates by **+1.76 bp**.
  - Converting the discount yield with BEY = 365·d/(360 − d·t) recovers **5.0772%**.
- **Friday → Monday at 5%:** 0.05 × 3/365 = **0.000410959**. One 252nd of a year gives **0.000198413**; actual/360 gives **0.000416667**.
- **Friday 2023-12-29 → Tuesday 2024-01-02 at 5%:** 0.05 × (3/365 + 1/366) = **0.000547571**.
- **Compounding at a constant 5%.** 365 daily accruals give **5.1267%**; rolling a 91.25-day bill four times gives **5.0945%** (91-day bills rolled through the year give **5.0946%**, the figure in `t5_portfolio_formulas.md` section 3). On the NYSE calendar, 2023's **250** sessions (last close of 2022 to last close of 2023, 364 days) give **5.1120%**.
- **A short-Treasury fund instead of the account.** With duration ≈ 0.25 years [A], a 10 bp rise in yields costs about **−2.5 bp**. A 0.15% expense ratio [S] turns the 5.1267% year into about **4.97%**.

**Options.**
- **(a) DGS3MO, investment basis, actual/365 (366 in leap years) (recommended).**
  - Published since 1981 by the Federal Reserve Board in H.15, tagged "Public Domain: Citation Requested" on FRED [V]. Free through the FRED API, with the key the project already uses. It is the existing function, with no conversion.
  - Limits, stated in the label: a constant-maturity yield (interpolated from the Treasury curve), not a traded bill's realized return. It ignores bill price moves (duration ≈ 0.25) and compounds daily (**+3.2 bp** a year at 5% against a quarterly roll). Retail cash sweeps often pay less.
- **(b) DTB3** (the 3-month bill's secondary-market discount rate, H.15, from 1954 [I]), converted with BEY.
  - Same source and terms. It adds a conversion formula and a 91-day maturity assumption, for no gain over 2003–2024.
  - Used unconverted, it accrues about **−13.2 bp** a year too little at 5% (four quarters of −3.29 bp).
- **(c) A T-bill fund's total return as the account** (SHV from 2007-01-05 [S], 0.15% expense ratio [S]; BIL from 2007 [I]).
  - Investable and net of fees.
  - But it starts in 2007, so about four of the calendar's 22 years have no cash return. It needs F1 distributions. It can lose money (duration). And when SHV is also a fund a policy can hold, the cash account and the fund would earn the same return, one free and one costing 2 bp to trade (see C4).
- **(d) Keep 0% (D4 (a)) for real data.** It conflicts with the owner's condition on D4 (a).
- **Not proposed:** a haircut for retail sweep rates (r = y − h). It would add another assumed number; the label says "before fees" instead.

**Recommend: (a).** The T-bill fund appears only as a separately held fund (C4) and as a cross-check: its tracking difference against the account (about its expense ratio plus duration effects) is reported as a flag, never as a correction.

**Edge cases** (F3, part (f)).
- No observation on or before the span's start: fail (exists).
- A bond-market holiday on an NYSE session (Columbus Day; Veterans Day or its observed day) has no DGS3MO value: carry the previous observation, with reason `bond_market_holiday_carry` (C2's example).
- No new observation for more than 5 business days: flagged `stale_rate`.
- Zero and negative yields are allowed.

---

## C2. Point-in-time availability, vintages and the backfill (F3.3; part of OD9)

This decision covers the rate as a **decision input** and how its history is stored. Booked account state (wealth, weights and trade sizes) uses the rate under a separate rule, the exception in **C9**, so that it can be approved or rejected on its own.

**Rule.**

1. **Decision input.** A DGS3MO value used *as a value* (a feature, a signal, or a policy observation field that carries the rate itself) counts only from its availability time. For **every** stored vintage, first releases included, that is the later of two times (AGENTS.md: when availability is uncertain, take the later timestamp):

   ```
   available_at(y_s, vintage v) = max( H.15 posting, about 16:15 ET, on the next Federal Reserve
                                       business day after s,
                                       the end of v's realtime_start date )
                                                              (rule h15_or_alfred_later)
   ```

   - The posting time is consistent with FRED: the 2026-10-15 value was updated at 15:16 CDT (16:16 ET) on 2026-10-16 [V]. But FRED's `realtime_start` is a date with no time of day, so for history the later timestamp is normally **the end of the posting day**. In practice a value for date s first counts at the decision cutoff of the NYSE session after its posting day. That costs one session against the 16:15 ET posting; relaxing it would need a verified intraday posting time for the whole history (an OD9 question).
   - A row d may use a vintage only if its available_at ≤ OD9's `signal_cutoff_at(d)`. **Real-data decision use waits for OD9's production rule** (the proposed cutoff is 18:00 ET; the owner's 2026-10-14 18:00 ET convention covers descriptive display only, and the interim end-of-session rule covers synthetic fixtures only).
   - Columbus Day (a Monday) and Veterans Day (November 11, or the Friday or Monday it is observed on) are Federal Reserve holidays on which the NYSE trades. The value for the business day before such a holiday is posted on the business day after it.
2. **Revisions are new observations** (AGENTS.md). "All data are subject to revision" [V].
   - Every ALFRED vintage published **on or before 2024-12-31** is stored with its `realtime_start` (requests use `realtime_end` = 2024-12-31; the existing adapter's default of 9999-12-31 would read revisions published inside the holdout, STAT-6). `vintagedates` is requested separately, since it cannot be combined with `realtime_start`/`realtime_end`.
   - A revised value gets its own available_at by the same rule and never overwrites the earlier value.
   - **Booked wealth uses the first release** (the earliest stored vintage), so a later revision never rewrites booked history. The number of revised observations and the largest absolute revision are reported as a deterministic data-quality diagnostic.
3. **Before ALFRED.** ALFRED's DGS3MO vintages begin on 2005-06-28 [V]. The **626** sessions from 2003-01-02 to 2005-06-27 have no archived first release.
   - Their earliest vintage has `realtime_start` 2005-06-28, so the same rule makes them decision inputs only from the end of 2005-06-28 (recorded as `pre_alfred_first_vintage`). Under C8's window start (2007 for a universe with SHV) this costs nothing.
   - Booked accrual (C9) uses that earliest vintage, flagged as possibly including revisions made before 2005-06-28.

**Backfill range.** Observations dated 2002-12-02 through 2024-12-30, with vintages published through 2024-12-31.
- The frozen calendar starts at 2003-01-02. A month of earlier observations covers the carry and the stale check. 2003-01-02 to 2024-12-31 holds **5,537** NYSE sessions.
- The last span of the development window ends at the close of 2024-12-31 and uses the value for 2024-12-30. The value for 2024-12-31 is posted on 2025-01-02, inside the holdout, so it is not requested (lockbox; STAT-6).
- Today the local history starts on 2024-01-02, because `scripts/download_rates.py` hard-codes a first start of 2024-01-01, and the rates store overwrites revisions and has no `available_at`.

**Example (Columbus Day 2024; illustrative rates and `realtime_start` dates; a cutoff at or after 16:15 ET).** Thursday 2024-10-10 at 5.00%, posted Friday; Friday 2024-10-11 at 5.10%, posted Tuesday 2024-10-15. Monday 2024-10-14 is an NYSE session, but a bond-market and Federal Reserve holiday with no DGS3MO value.
- Thursday's 5.00% counts from the end of Friday, so the decision at Monday's cutoff sees it.
- Friday's 5.10% counts from the end of Tuesday (its posting day). The decision at Tuesday's cutoff still sees 5.00%; Wednesday's is the first to see 5.10%.
- The accruals booked meanwhile (C9) are **0.000418033** at Monday's close (0.051 × 3/366; 2024 is a leap year) and **0.000139344** at Tuesday's (Monday has no value, so Friday's is carried with `bond_market_holiday_carry`).

**Options.**
- **(a) As above (recommended):** the `h15_or_alfred_later` rule for every vintage; append-only vintages bounded at 2024-12-31; first release for booked wealth; the pre-ALFRED consequence; the backfill range above.
- (b) The H.15 posting time alone, ignoring `realtime_start`: one session earlier, but it relies on an intraday posting time that the history does not record, so it is not the later timestamp.
- (c) Book wealth with the latest vintage. Simpler storage, but a revision rewrites booked history, so a stored result stops being reproducible.
- (d) Treat each value as known on its own date (today's code: no `available_at`). As a decision input, that is lookahead.

**Recommend: (a).** It changes availability semantics, and it is a **breaking** change to the rates store's contract: append-only vintages replace the `(date, rate_id)` uniqueness key with `(date, rate_id, vintage_date)`, and the columns `available_at`, `vintage_date` and `availability_rule` are added. Both need approval. It decides OD9 only for this series.

**Tests at implementation.**
- Using y_t instead of y_{t−1} fails (F3 T3.2).
- A decision-input field carrying the rate that sees Friday's value at Tuesday's cutoff fails.
- A vintage whose `realtime_start` is later than its H.15 posting counts only from the end of that date.
- A revision published after a decision never reaches it, and no vintage published after 2024-12-31 is requested or stored.
- Appending later vintages leaves every earlier booked return bit-identical (prefix stability).

---

## C3. The risk-free rate for excess returns and Sharpe (F3.2)

**Rule.** The risk-free return is the cash account's return, for the same span:

```
rf_t = r_cash,t (C1)            e_t = R_t − rf_t
```

A 100% cash portfolio has an excess return of 0 every session, up to floating-point residue, and a Sharpe ratio `unavailable` (`zero_variance`). The residue is real: the engine computes (1 − C)(1 + w·r) − 1, so the excess is within about 1e-16 of 0, not exactly 0 (the spec's T3.4 uses atol 1e-15). Zero variance must therefore be decided with a registered tolerance (group D, M2), never with an exact `> 0` test, or a 100% cash portfolio would get a meaningless Sharpe ratio from noise. Weekends and holidays accrue in both terms (group D, M2's example).

**Example.** A 100% cash account through 2023 at a constant 5%.
- With rf = 0, its per-session returns vary only with the day count (one day, three over a weekend), so its annualized Sharpe ratio is **25.9**. Cash would look like the best risk-adjusted holding there is.
- With C3, its excess return is 0 in every session up to a residue of at most **1.0e-16**, and its Sharpe ratio is unavailable under M2's tolerance, as it should be.

**Options.**
- **(a) The C1 series (recommended).** Cash is neutral, and the excess return measures what a strategy adds over holding the account.
- (b) A T-bill fund's total return. The account then shows a nonzero excess (the fund's expense ratio and duration effects), and the fund's Sharpe ratio is about 0 by construction.
- (c) rf = 0. See the example: it flatters every cash-heavy result.
- (d) A 1-month rate (DGS1MO, from 2001 [I]). It is closer to an instantaneous risk-free rate, but it is a second series that differs from what the account earns.

**Recommend: (a).**

---

## C4. Cash, D4 (a) and SHV once C1 is approved

**Rule.**
- **Real data:** the cash column is C1's r_cash, with C1's label. That covers explicit cash, a cap's excess (D5) and the uninvested part of a turnover-limited build.
- **Synthetic scenarios:** D4 (a) is unchanged (0%, with its label). Synthetic sessions have no dates and no rate series. The registered synthetic comparison (`ppo_comparison_v1`) is neither re-run nor re-tuned.
- **SHV stays a fund:** its own total return (F5.1), 2 bp to trade (D2), able to lose money. It is never the cash account.
- **No engine change.** `portfolio_step` already books whatever cash return it is given (`portfolio/engine.py`). The implementation is a data input, a label and tests.

**Examples (one year at a constant 5%).**
- A portfolio held half in cash and half in a short-Treasury fund returns **2.48%** under D4 (a) and **5.05%** under C1: a gap of **2.56 percentage points**, all from the cash half. In the synthetic comparison's strict set, some portfolios held 50% to 76% in uninvested cash (`ppo_comparison.md`, section 8). That is why 0% cash cannot stand in for real economics.
- If SHV were also the cash account (C1 (c)), moving 20% from the SHV fund into the account would save **+0.4 bp** of trading cost and change nothing else. Every policy would learn to avoid the fund, a construction artifact.

**Options.**
- **(a) As stated (recommended).**
- (b) Also give synthetic scenarios a registered constant cash rate. That is a generator change, a new registration, and the repeated synthetic tuning the owner ruled out. Defer.
- (c) D4 (c): no cash account, SHV only. It conflicts with D5's excess-to-cash rule.

**Recommend: (a).**

---

## C5. Fund total-return benchmarks (OD8.6; F4.1)

**Rule.**
- A benchmark is one fund's total return computed by the approved F5.1 from raw closes, distributions and splits.
- It is bought from 100% cash at the comparison's common first execution close and pays D2's rate on the build (D3, approved). It is then held without rebalancing, with no exit charge.
- The fund's expense ratio is never subtracted (A5).
- It is registered with the comparison before any run (G1), never chosen after results. Which fund is a registration choice per comparison; the fund list is OD7's.

**Equation.**

```
W^B_{t0} = 1 − c_B                         (a one-fund build: Δ = 1)
W^B_t   = W^B_{t0} × Π_{s = t0+1..t} (1 + r^B_s),   r^B from F5.1
```

**Examples.**
- **Build.** At 5 bp, the benchmark holds **0.9995** after the build; a +10% year then gives **9.945%**.
- **Dividends.** A flat price with 0.5% distributions each quarter has a total return of **2.0151%**. A strategy at +1.00% "beats" the price index by 1.00 pt but trails the total return by **−1.02 pt**.
- **Fees.** A 10.00% total-return index and a 0.09% expense ratio give a fund return of about **9.901%**. That fund return is the benchmark; nothing more is subtracted.

**Options.**
- **(a) Fund total return (recommended).** One rule for every benchmark; investable; the fees already inside it.
- (b) A total-return index. It has no fees, so it flatters the benchmark against fee-paying fund portfolios by the expense ratio, and index total-return data are usually licensed.
- (c) A price index (today's `^GSPC`, `^IXIC`, `^DJI`). It is biased by the dividend yield in the strategies' favour. It moves to labelled context.
- (d) A price index plus an assumed yield. Rejected.

**Recommend: (a).**

---

## C6. Every benchmark declares `return_basis` (F4.2)

**Rule.** Every benchmark and reference series declares `return_basis` ∈ {`total_return_fund`, `total_return_index`, `price_only`}.
- A comparison refuses a `price_only` benchmark (`ValueError`). Such a series appears only as context, labelled "price index, distributions excluded".
- `total_return_index` is allowed only in stock-only research, labelled "index, before fees".
- A missing basis is refused.
- It is a **breaking** contract change under AGENTS.md: a required field that refuses a missing value makes previously valid benchmark configurations invalid (today's price-index benchmarks among them). That is intended, and it needs approval.

**Example.** The dividend example in C5: the same fund is 2.0151% as `total_return_fund` and 0% as `price_only`. Without the field, nothing stops the second from being compared with total-return strategies, which is how today's price-index bias arose.

**Options.** **(a) A required enum (recommended)**; (b) a free-text label; (c) no field (today).

**Recommend: (a).**

---

## C7. Reference portfolios, and benchmarks under a constraint set (F4.3)

**Rule.**
- **Reference portfolios** (for example 100% equity fund, 60/40 equity/bond funds, 100% cash account, or the synthetic comparison's 60/30/10 when registered) are policies.
  - They run through the same engine (`portfolio_step`), schedule and offset (D7), costs (D2) and build (D3).
  - They run under the same registered constraint set as every compared policy (D5, as confirmed by the owner on 2026-10-17).
  - Each states its rebalancing: held, or rebalanced on the registered schedule.
- **Benchmarks** (C5) are yardsticks, not compared policies: one fund, bought and held, never constrained. Each is reported unchanged beside every constraint set.

**Examples.**
- **60/40 after one period** in which equity returns +10% and bonds 0%: the equity weight drifts to **0.6226**. Rebalancing trades **0.0226** one way and costs **+0.226 bp** at 5 bp.
- **A benchmark forced through a 0.5 cap** would hold half in cash. In a +10% equity year with cash at a constant 5%, it returns **7.54%** instead of 9.945%. It would no longer be the market benchmark, and it would change with each constraint set.

**Options.**
- **(a) Reference portfolios inside the set; benchmarks outside it (recommended).**
- (b) Benchmarks constrained too (a literal reading of "every compared policy"). The benchmark's meaning then changes with every set.
- (c) Reference portfolios unconstrained too. That breaks D5's identical-set rule.

**Recommend: (a).** This needs the owner's reading of the D5 confirmation: benchmarks are not "compared policies".

---

## C8. The common window start (F4, part (f))

**Rule.** A comparison's window starts at the first session on or after the latest first close among every fund in its universe and every benchmark (the cash series starts in 1981). It is registered before any run, and no fund is spliced with a proxy.

**Facts.** AGG's inception is 2003-09-22 [S]; SHV's is 2007-01-05 [S]; VTI's is 2001 [I]. The first trade can follow the inception by a few days; the stored first close decides.

**Example.** For a universe including SHV, the window has **4,527** return sessions to 2024-12-31. Without SHV, from AGG's inception, it would have **5,355**. The longer window shrinks a minimum detectable effect only by the factor √(4,527/5,355) = **0.919**, that is by about 8%.

**Options.**
- **(a) The latest first close, no proxy (recommended).**
- (b) Drop SHV from the universe (the cash account covers short Treasuries) to start in late 2003. A product change, for an 8% smaller minimum detectable effect.
- (c) Splice a proxy (the cash account for SHV before 2007), labelled. The pre-2007 part would have no fees and no duration risk: a hybrid series. Not recommended.

**Recommend: (a).**

---

## C9. Booked account state: an explicit exception to the derived-value rule (F3.3; part of OD9)

Split from C2 so that it can be approved or rejected on its own.

**Rule.** r_cash,t uses y_{t−1}, the yield at which the account's bill was bought at the close of t−1. Booked wealth, the drifted weights, the trade sizes at each execution close and a policy's view of its own holdings all use that accrual at t, even when C2 says the value does not yet count as a decision input. **This is an exception to AGENTS.md's rule that a derived value is available no earlier than its latest input**, scoped to booked account state only. A field that carries the rate itself remains a decision input under C2.

**Why an exception, and why this one.**
- The bill's price at t−1 is a market observation at t−1, and Treasury's own yield curve publishes the t−1 yield on the evening of t−1, usually by 18:00 ET (`universe_and_market_context.md`, verified 2026-10-08). The account holder is not using later information; our source publishes it later.
- A strict rule cannot work for booked state at all. The value for t−1 is posted at 16:15 ET on t, after t's 16:00 close (and under C2 it counts only from the end of that day). So the wealth needed to size any trade at the close of t would never be known in time, on any day.
- Its information content is tiny: one span's accrual. In C2's example it moves wealth by 0.000418033 per unit of cash. But an exception is still an exception, so it is stated, scoped and tested.

**Options.**
- **(a) Approve the exception, scoped to booked account state (recommended).**
- (b) Strict: no exception. Decision-time state would use a provisional accrual from the latest counted rate, booked wealth would be trued up later, and trade sizing at every close would need the provisional value too. The booked path would then depend on publication timing, with two wealth numbers per session.

**Recommend: (a).**

**Tests at implementation.**
- The booked weights at Monday's close in C2's example include Friday's accrual, and pass under the exception.
- A policy observation field that carries the rate itself still fails if it sees Friday's value before C2 allows it.

---

## Data steps and licensing, per option

| Item | Needed by | Steps (none is taken by this packet) | Source and terms | Cost |
|---|---|---|---|---|
| DGS3MO, observations 2002-12-02 to 2024-12-30, with ALFRED vintages published through 2024-12-31 | C1 (a), C2, C3 | Lift `download_rates.py`'s 2024 start; retrieve through the existing Version 1 series endpoint with `realtime_end` = 2024-12-31, and `vintagedates` in a separate request (`FredDataSource`); an append-only rates store with `available_at`, `vintage_date` and `availability_rule`; the C2 rule; prefix-stability tests | FRED API; Federal Reserve Board H.15, "Public Domain: Citation Requested" [V]. The FRED API terms of use require the notice "This product uses the FRED® API but is not endorsed or certified by the Federal Reserve Bank of St. Louis." in the application, forbid implying endorsement, and leave rate limits to FRED; they do not restrict local storage [V]. Suggested citation: Board of Governors, DGS3MO, retrieved from FRED [V]. Key: `FRED_API_KEY` (exists) | Free; a few requests (the endpoint returns up to 100,000 observations per request) |
| DTB3 | C1 (b) only | As above, plus approval of the BEY conversion | As above [I] | Free |
| T-bill fund total return (SHV, BIL) | C1 (c); the C1 cross-check; SHV held as a fund | The Tiingo fund adapter (raw closes, `divCash`, splits); the private primary-source distribution anchors; the F1 store | Tiingo terms, recorded per batch in `terms_note`; vendor data is never committed (F1) | The existing key, and the owner's authorization of the request |
| Benchmark funds' total returns | C5, C7, C8 | As for SHV, for each registered fund | As above; OD7 approved fund total returns from the existing personal-use sources to 2024-12-31 | As above |
| A total-return index | C5 (b) only | A licensed index feed | Index providers license total-return data | Paid; against RES-1's cost-conscious rule |
| Price indices | Context only (C6) | Existing yfinance data | Vendor terms | Existing |

When a page shows results that use the cash rate, it carries the FRED notice and the citation (an implementation follow-up).

---

## These follow automatically once C1–C9 are approved

- A real-data counterpart of `CASH_RETURN_LABEL`; the synthetic label is unchanged.
- The spec's tests T3.1–T3.5 and T4.1–T4.4, plus C2's leakage and prefix-stability tests.
- In T5c's one-change-at-a-time attribution, the "cash rate" and "benchmark basis" steps.

## Not in this packet

- OD9's `signal_cutoff_at` and the production availability rule for prices. C2 proposes a rule for DGS3MO only.
- Group D statistics and reporting (`group_d_decisions.md`).
- Market impact, pay-date receivables, taxes.

## Learning notes

1. **Problem.** A backtest needs to know what uninvested money earns and what it is compared with. Get either wrong and cash-heavy or dividend-heavy strategies look better or worse than they are.
2. **Concepts.**
   - **Quote conventions.** A bill's discount yield (price discount on face value, actual/360) is not its investment yield (gain on price paid, actual/365). Only the investment yield accrues to the bill's true holding-period return.
   - **Constant maturity.** DGS3MO is read from a fitted yield curve at exactly three months: a rate, not a traded asset's return.
   - **Total return.** Price change plus distributions; a fund's total return is already net of its fees.
3. **Assumptions.** Daily accrual approximates rolling a bill; bill price moves are ignored; trades fill at the close; "before fees" because a real account's sweep rate may be lower.
4. **Formulation.** r_cash = y·days/365; e = R − r_cash; W^B = (1 − c)·Π(1 + r^B).
5. **Example.** The 91-day bill: 1.2658% held to maturity, matched exactly by the investment yield and missed by 3.29 bp by the discount yield.
6. **Leakage.** Using the end-of-span rate y_t; a decision that sees a rate before it was posted (the Columbus Day example); a revised rate rewriting booked wealth; a benchmark chosen after results.
7. **Ways results could mislead.** A price-index benchmark understates the market by its dividend yield; 0% cash understates every cash-heavy result; a "risk-free" rate of 0 makes cash look like the best risk-adjusted asset (Sharpe 25.9 in the example).
8. **How practitioners validate.** Reconcile the cash series against a T-bill fund's published total return (difference ≈ its expense ratio); reconcile benchmark total returns against the issuer's published total return; check point-in-time joins with negative tests.
9. **Next to learn.** Bond-equivalent yield and money-market conventions; how a Treasury yield curve is fitted; GIPS rules for benchmarks.

**Bounded exercise (optional, after approval).** By hand, compute the Columbus Day example's two accruals and the decision-input value for Monday, then check them against `prepare_risk_free_returns` with a three-row rates frame.

**Suggested reply:** "Approve C1 (a), C2 (a), C3 (a), C4 (a), C5 (a), C6 (a), C7 (a) with benchmarks outside the constraint set, C8 (a), C9 (a)", or amend any item.
