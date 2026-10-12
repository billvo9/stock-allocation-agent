# UX blueprint

Status (2026-10-11): **direction approved by the owner on 2026-10-11** (OD1, OD2, OD4, OD5, OD6, OD7, and the general direction of OD16 in `vision_and_prd.md`, section 13). OD16's activation and holdout end, and the other decisions, are pending.
- "Exists" cites code on `main` (T4 included since PR #47); everything else is proposed.
- Placeholders are written ‹like this›.
- Tiers (NOW, RULES, VALIDATED, LONG-TERM) are defined in `README.md`. Evidence status and claim type are defined in `vision_and_prd.md`.

## 1. Principles

1. **Evidence before opinion.** Every number says what kind of thing it is (actual, simulated, forecast, illustration) and what it knew, as of when.
2. **The validation status decides what may appear; the mode only decides depth.** Progressive disclosure may hide detail, never a material caveat.
3. **Construction, not promises.** Portfolios are named by how they are built ("Bond-heavy, inverse volatility"); the goal is a subtitle. "Capital Preservation" reads as a guarantee.
4. **No nudging.**
   - No default "best" sort.
   - No green buy signals.
   - No leaderboards, streaks or count-up numbers.
   - Nothing is "recommended" until the product is allowed to recommend (see "Regulatory boundary" in `vision_and_prd.md`).
5. **Calm by default.** Restrained colour, strong typography, generous space, and motion only for state changes.

## 2. Assumptions this blueprint challenges

| Assumption in the brief | Challenge | Consequence |
|---|---|---|
| The home screen explains "why this portfolio" | Today there is no portfolio the product can honestly stand behind: no forecast-to-weight rule (T5), backtests execute at the same close (`evaluation/baseline.py`), and the universe is four hindsight names | "Why" means "which transparent rule, on which inputs, what it aims for and what it gives up". Never "what the model believes" |
| Modes unlock capabilities | A mode switch is not evidence | Gating table in section 3 |
| Theme screens such as "Magnificent Seven" and "AI/Semiconductor" | Both are defined with today's knowledge. Applying today's list to history is survivorship and hindsight bias | Theme lists are current exposure lenses, dated "defined on ‹date›". They are never historical strategies |
| A Builder with live what-if sliders | Continuous sliders invite users to overfit backtests by search. The app also must never fit a model or read raw data; it may only run pure allocation functions over published inputs | Discrete options computed by those pure functions, plus a visible count of configurations viewed |
| A personalized builder from capital and risk preference | Mapping a person to a portfolio is the hallmark of personalized advice | The Builder explores rules the user chooses. It does not prescribe (see "Regulatory boundary") |
| Showing current data | Any 2025+ information spends the lockbox informally: a trailing return in 2026 is a realized holdout label, and holdings snapshots encode relative returns | By default every page is as of 2024-12-31, and later periods show the "Sealed" state (owner decision OD5) |
| Horizons of 5, 20, 30 and 60 sessions | Only the 20-session label is registered (`MODEL_LABEL_SPEC`, and from T4 its excess-return form `MODEL_EXCESS_LABEL_SPEC`, the default) | All four stay possible user-facing horizons (OD10). Descriptors work at every horizon. A model view appears only at a validated horizon, and the others show "not validated for this horizon". Dropping a horizon is an explicit product decision |

## 3. Information architecture

Three modes share one navigation shell (`st.navigation` sections). A keyboard-reachable mode switch sits in the header. A persistent footer shows the knowledge time and the evidence legend.

| Mode | Job | Pages |
|---|---|---|
| **Explore** (Simple) | Understand and compare a few portfolios; see what is known and what is not | Home; Compare portfolios; Asset explorer |
| **Build** (Builder) | Combine assets under the user's own constraints and see the tradeoffs | Portfolio Builder; Exposure and risk review |
| **Research** | Inspect models, data, validation and statistical evidence | Today's research dashboard pages (exists: `dashboard/app.py`, plus "Linear models" on T4) and a new "Evidence ledger" page |

**What each mode may show, by evidence status** (statuses are defined in `vision_and_prd.md`, section 6):

| Content | Explore | Build | Research |
|---|---|---|---|
| Facts about the past (`descriptive`) | Yes | Yes | Yes |
| Rules illustrations and hypothetical backtests (`rules_illustration`, after T5's formulas are approved under OD8) | Yes, badged | Yes, badged | Yes |
| Model output in development (`research_only`, `development_evidence`) | No | No | Yes, labelled |
| Model output after T6 found no collapse (`holdout_checked`) | No | No | Yes, labelled |
| Prospectively confirmed model output (`prospectively_confirmed`) | At the claim level reached | At the claim level reached | Yes |

A **claim level** that has not been validated (expected return, probability of gain, downside) never appears in Explore or Build, whatever the evidence status of the model behind it.

## 4. Screen concepts

### Home (Explore)

```
┌─────────────────────────────────────────────────────────────────────┐
│ [Rules illustration] [Simulated · net of costs · before taxes]       │
│                                                                     │
│ Broad equity and bonds, 60/40, rebalanced quarterly                 │
│ 60% broad equity fund · 40% aggregate-bond fund                     │
│                                                                     │
│ Why this portfolio                                                  │
│   Rule      Fixed 60/40 weights, reset every quarter; compared with │
│             the same mix never rebalanced                           │
│   Aims for  Equity growth with bond ballast                         │
│   Gives up  Upside in strong equity years; bonds can fall with      │
│             stocks when rates rise                                  │
│                                                                     │
│ Concentration                                                       │
│   ! ‹34%› of equity weight sits in 10 companies (holdings ‹date›)    │
│                                                                     │
│ History · registered window ‹start›–2024-12-31                       │
│   ‹3 of 17› calendar years lost money                               │
│   Worst decline ‹−31%› · ‹$3,100› on ‹$10,000› · recovered in ‹27 mo›│
│   [growth of $1 (log) vs 100% equity] [drawdown panel below]        │
│   After 2024-12-31: Sealed for one final test (hatched)              │
│                                                                     │
│ What this cannot tell you                     (always visible)      │
│   It does not forecast returns. Past simulated results come from    │
│   rules applied to history, not from trading.                       │
│                                                                     │
│ Knows data through 2024-12-31 close · publication ‹id› · Details →  │
└─────────────────────────────────────────────────────────────────────┘
```

- The headline is the construction. The "why" block is the rule, what it aims for and what it gives up.
- Concentration callouts come from holdings or exposures, with their own as-of date.
- Risk is shown in dollars on the user's reference amount, as historical ranges with honest denominators (calendar years, not overlapping windows).
- Headline metrics carry intervals (block bootstrap), and every simulation is labelled "before taxes".
- Holdings snapshots are dated, and under the default lockbox boundary never later than 2024-12-31.
- With no saved choice, Home shows the comparison of the pre-registered model portfolios (section 4.4) and one primary action: "Build your own".

### Asset explorer (the screener)

**MVP (tier RULES).** Descriptive, point-in-time characteristics only:
- asset class and sector, each "as of ‹date›";
- trailing return over the registered horizons;
- realized volatility;
- maximum drawdown;
- liquidity (median dollar volume);
- correlation with the benchmark;
- data coverage.

Rules:
- The default sort is by name, because a default sort is a recommendation.
- There is no "expected return" column. A header note reads "No validated forecast. These columns describe the past."
- Theme filters (semiconductors, large technology, user-defined lists) are labelled "list defined on ‹date›".

**Later (tier VALIDATED).** A "relative attractiveness" column may appear only at the claim level the ranking model reached:
- rank or percentile first;
- expected excess return only if calibrated;
- probability of gain only if its reliability is shown.

Each value carries an interval or a confidence state and the status pill of its model.

### Portfolio Builder (Build)

- **Inputs (MVP):**
  - assets (from the registry or the user's watchlist);
  - a construction rule: fixed weights, or equal weight or inverse volatility over the risky assets;
  - a per-asset cap;
  - cash as a floor or a fixed weight (never inside inverse volatility, which would divide by zero);
  - rebalance frequency;
  - a reference amount.

  **Later:** sector and theme caps with fund look-through, and a volatility target.
- **Loss limit.** A tolerable historical loss in dollars is a **check, not a target**: "this mix's worst historical decline was ‹$3,100›, above your ‹$2,500› limit". It describes one historical path and is met in-sample by construction. It is a constraint the user chooses, never a questionnaire that maps them to a portfolio.
- **Constraint feedback, inline:**
  - "4 assets capped at 20% cannot reach 100%. Raise the cap to 25% or add an asset."
  - "Cash floor 10% is binding."
- **Tradeoff view, current beside one alternative:**
  - historical volatility range;
  - worst historical decline in dollars;
  - turnover;
  - cost drag;
  - effective number of holdings;
  - sector and asset-class exposure (fund look-through later);
  - a correlation heatmap (neutral scale);
  - days to trade at the stated liquidity assumption.
- **What-if:** discrete variants, such as caps of 10%, 20% or 30% or monthly versus quarterly rebalancing. They are computed by pure allocation functions over published inputs, so the app never fits or downloads anything. A quiet counter reads "Configurations viewed: 7. The best-looking past result usually overstates the future."
- **Output wording:** "Illustrative weights from the rule you chose". Never "recommended" or "target".

### Model portfolio comparison (Explore)

- One column per pre-registered model portfolio, each with its construction badge and evidence badge.
- One registered comparison window for all portfolios, set by an asset-class rule before the comparison was built; pre-registered subperiods (such as 2008 and 2022) beside it.
- Each portfolio is compared with its registered reference (a 60/40 rebalanced quarterly vs the same mix never rebalanced), never with itself.
- Everything is net of stated costs.
- Rows:
  - exposures (asset class, sector, top holdings);
  - risk (volatility, worst drawdown and its duration, recovery);
  - turnover;
  - costs;
  - difference against the named benchmark;
  - data caveats (survivorship, current-listing universe).
- Small multiples share one log axis. 2025 onward is hatched "Sealed".
- When paper tracking exists, a separate panel shows it, starting at its own "tracking began" rule. It is never stitched onto the backtest line.

### Research (exists, extended)

The current pages stay, plus an **Evidence ledger** page. For each publication it shows:
- claim type, evidence status, model and rule version;
- universe and data vintage;
- variant count from the ledger;
- validation runs and limitations.

This is where "why can't I see a forecast?" is answered in full.

## 5. Evidence and provenance language

Hue already means identity or status (`dashboard/style.py`), so the evidence type is carried by **a text badge, a glyph and a line style**, never by colour alone.

| Evidence type | Badge | Chart treatment |
|---|---|---|
| Actual history (prices, holdings) | Actual | Solid line |
| Hypothetical backtest | Simulated | Thin solid line, labelled at its end |
| Paper tracking | Paper | Dashed, right of a "tracking began ‹date›" rule |
| Forward forecast | Forecast | Band only, right of a "today" rule |

- A line never continues across a change of evidence type.
- Evidence-status pills (descriptive, rules illustration, research only, development evidence, holdout-checked, prospectively confirmed) use neutral tokens: a proposed `EVIDENCE` set in `theme.py`. Never green, which invites over-trust.
- Series badges (Actual, Simulated, Paper, Forecast) and status pills are mapped one-to-one in `architecture.md`, section 4.
- In Research mode and the Evidence ledger, each rule, model or policy also shows its promotion stage (S0–S4, `vision_and_prd.md`, section 15). A stage is never shown as a quality score.
- Every card shows its knowledge time and data vintage from the publication. Until `signal_cutoff_at` is approved, the card shows the recorded `knowledge_time` and never infers one.

## 6. First-class states

Each state has its own component, reason code and copy. A state is never an empty chart or a zero.

| State | Example copy | Treatment |
|---|---|---|
| Empty | "No portfolio yet. Compare the model portfolios, or pick assets to build your own." | One primary action |
| Pending | "This forecast, made ‹date›, is scored after the close on ‹date›." | Dashed card (exists in spirit as `future_state`). Counted, never dropped (maturity rule exists in `dashboard/evaluation.py`) |
| Unavailable | "SNDK has no prices before 2025, so it is excluded, not set to zero." | A dash and the stored reason (exists: `estimate_card`) |
| Insufficient confidence | "Not shown. The evidence so far could detect only effects about three times larger than is realistic." | A link to Research. The wording is generated from stored metrics, not hard-coded |
| Stale | "Prices last updated ‹6› sessions ago. Figures describe ‹date›." | A header pill. Forward-looking content is blocked past each source's maximum age |
| Partial coverage | "Sector data covers ‹90%› of weight; the rest is Unclassified." | Never redistributed across known groups |
| Sealed (lockbox) | "Sealed. Data after 2024-12-31 is reserved for one final test." | A hatched axis region; no values |
| Withdrawn or superseded | "Replaced by publication ‹id› on ‹date›: ‹reason›." | Link to the newer publication. The old one stays readable |

## 7. Uncertainty for nontechnical users

**Do:**
- Show ranges before points, and translate them into dollars on the user's reference amount.
- Frame by frequency with honest denominators over non-overlapping periods ("3 of 17 calendar years").
- Draw fan bands (50% and 80%) only for calibrated distributions.
- Use whole percents in Explore and keep decimals in Research.
- Keep "What this cannot tell you" visible on every page that shows a number about the future or a simulation.

**Avoid:**
- gauges, star ratings and "confidence scores";
- red and green buy or sell signals;
- "beats the market by 3.27%";
- count-up animations, streaks, achievements and leaderboards;
- frequencies from overlapping windows presented as independent;
- a single "expected" line without its band.

## 8. Design language

Built on the existing tokens (`dashboard/theme.py`, `components.py`, `style.py`). It takes inspiration from calm, premium research tools without copying any brand.

- **Type.**
  - The platform sans (no web font is fetched), 15 px base, the existing heading scale and tabular figures.
  - Add one display size (about 2.25 rem) for the Home headline only.
  - Line length 60–80 characters for prose.
- **Space.** The 4–32 px scale, plus 48 px between Home sections. At most three stat cards per row.
- **Colour.**
  - One accent: the existing slate teal.
  - Status colours only for status; Okabe-Ito and `CANDIDATE_COLOURS` (added by T4) for identity.
  - Gains and losses stay neutral, carried by sign and words.
- **Motion.**
  - For state changes only (pending to available, a section revealed), 150–200 ms opacity or height.
  - Never tween a value; Plotly transitions stay off (exists).
  - `prefers-reduced-motion` turns all custom motion off.
- **Accessibility.**
  - WCAG AA tokens (exists and tested).
  - Keyboard focus rings (exists).
  - Severity written in words (exists).
  - **New:** a one-sentence summary and a data table for every chart, since Plotly charts have no text alternative (open item).
- **Narrow screens.** Chart heights shrink, wide tables become per-asset cards, and the mode switch collapses into a menu.

## 9. Frontend feasibility

- **Streamlit can deliver the MVP:** the modes, forms, discrete what-if options, badges (`pill`), every state above, and AppTest coverage.
- **Its limits:**
  - a full rerun per interaction;
  - coarse ARIA and focus control;
  - `data-testid` CSS hooks that can break on upgrade (1.64 is pinned);
  - no authentication;
  - weak mobile tables;
  - server-side state, so user inputs reach server memory.
- **A dedicated web frontend is justified only by** external users (with authentication), mobile-first use, direct-manipulation latency, or accessibility failures Streamlit cannot fix. Each is an owner decision (dependencies, deployment).
- **No rewrite later.** The app reads only versioned publications (the architecture's reader). Builder logic lives in pure functions, and the copy lives in data modules (like `glossary.py`). A new frontend would replace only the page and component layer.

**Interactive prototype.** The first prototype is the Explore and Build pages themselves, in Streamlit. They read publications of pre-registered rules and run only pure allocation functions, and every view carries the "Illustration" or "Simulated" badge. Machine-learning output of any kind has no place in the prototype.

## 10. UX scope of the MVP

1. **Research mode gains the evidence-status pills** and the Evidence ledger. This needs no new data.
2. **Explore:** first the actual history of two or three broad funds (P1-1b). Then the model portfolio comparison and Home, from rules-based publications (tier RULES, after T5's formulas are approved under OD8).
3. **Build:** the MVP Builder (section 4) with discrete options, and the exposure and risk review.
4. **Asset explorer** with descriptive characteristics only. It is labelled "current-listing universe" until a point-in-time universe exists.

Everything else (forecast columns, fan charts, paper-tracking panels) is specified here, so the components exist, but is shown only in its proper state until the evidence exists.
