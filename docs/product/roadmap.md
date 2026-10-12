# Research and product roadmap

Status (2026-10-11): **direction approved by the owner on 2026-10-11** (OD1, OD2, OD4, OD5, OD6, OD7, and the general direction of OD16 in `vision_and_prd.md`, section 13). OD16's activation and holdout end, and the other decisions, are pending. It extends the research roadmap in `docs/research/diagnostics_and_visualization_plan.md` (T1–T6) and the U and C tracks in `docs/research/universe_and_market_context.md`.

**Labels.**
- Owner decisions: OD1–OD17 (`vision_and_prd.md`, section 13). Promotion stages: S0–S4 (`vision_and_prd.md`, section 15).
- Ladder rungs: L0–L4 and D1–D2 (`vision_and_prd.md`, section 6).
- Product tickets: P*n*; research tickets: RES-*n*; foundations: F*n*, G*n*.
- Risk IDs carry a category prefix (`risk_register.md`).

**Shape of the plan.**
- **Two tracks run in parallel:**
  - the **product track** ships a rules-only Portfolio Lab, as of 2024-12-31 under the default lockbox boundary (OD5);
  - the **research track** earns forecasts one claim at a time.
- **The critical path to any model view in the product is data and evidence, not code:**
  1. a survivorship-free source;
  2. a point-in-time universe;
  3. a pre-registered re-run;
  4. the one-time holdout (a "no collapse" check);
  5. years of prospective data.

  Risk forecasts can reach the product much sooner than return forecasts.

No calendar dates are promised. Sizes are relative (S, M, L, in tickets the size of T3.1–T4).

## Phase 0: planning (this pull request)

- **Delivers:** the product documents in `docs/product/`.
- **Gate:** met on 2026-10-11. The owner approved OD1, OD2, OD4, OD5, OD6 and OD7, and the general direction of OD16 (not its activation or H_end). OD2 approved starting P1-1a, T5 and F1 in parallel, not any T5 formula (OD8). OD3 (the full evidence model) is confirmed with P1-1a, whose version 1 needs only its first two statuses. The rest are decided at their tickets.
- **Changes:** no code, data, contract, universe or lockbox change.

## Phase 1: foundations (start in parallel)

P1-1a, G1, T5 and F1 are independent. **T5 is the long pole for every simulated number, so its formula review work starts now.** Each formula still needs the owner's explicit approval under OD8; OD2's approval to start T5 approves none of them.

| Ticket | Delivers | Depends on | Acceptance gate | Size |
|---|---|---|---|---|
| **P1-1a** | Publication contract v1 on synthetic fixtures: envelope, write-once writer, fail-closed reader, evidence status v1, the lockbox rule, the interim knowledge-time rule (`vision_and_prd.md`, section 14) | OD2 and OD5 (approved), OD3 (confirmed with this ticket), OD9's interim rule | The negative tests in the PRD; verify passes; no T2 contract change | M |
| **G1** | Git-tracked registry: model-portfolio pre-registrations, every real-panel scoring, holdout use, forecast-log starts and review dates | OD12 | A registry entry is immutable after commit (a test); the ledger references it | S |
| **G2** | Prospective pipeline (`architecture.md`, section 9):<br>• holdout end H_end recorded in G1 first<br>• separate store and entry point<br>• registration-hash check before each decision<br>• append-only decision log with `written_at`, no backfill, latest hash committed to Git<br>• review reports only through its own entry point<br>• sealed-window inputs withheld from every view until T6<br>Design and tests only. Activation and H_end each need a separate explicit owner approval under the documented pre-registration procedure | OD16 (direction approved; activation and H_end pending), G1, OD9 (real `signal_cutoff_at`), OD14 (calendar), F2, a scheduler (infrastructure gate), keys (secrets gate) | Research and product code cannot read its store (import and path tests); no row is ever edited or backfilled; nothing inside [2025-01-01, H_end] is shown before T6 | M |
| **T5** | Portfolio readiness:<br>• next-close execution (lag 1)<br>• cash returns: the DGS3MO backfill from 1981 as needed, `available_at` T+1, the convention check, or a T-bill fund<br>• total-return, fund-based benchmarks<br>• a cost model, including the initial build and a stated two-way turnover convention<br>• distribution reinvestment timing (ex-date vs pay date)<br>• calendar-time CAGR<br>• block-bootstrap intervals and a Sharpe interval<br>• rebalance-date sensitivity counted as variants<br>• a "before taxes" label<br>• a participation check | OD8 | Each formula explicitly approved by the owner under OD8. Hand-computed examples. Baseline results re-run with every difference explained. A test ties the label's entry lag to the backtest's execution lag | L |
| **F1** | Fund and cash data: raw closes plus dated distributions, retrieval times, history before 2015 where available, labelled proxies; a session calendar (in-repo NYSE and SIFMA holiday file, or a dependency). Everything ends 2024-12-31 under OD5's default | OD7, OD14 | Prefix-stability and `available_at` tests; no vendor data committed; no row dated on or after `MODEL_HOLDOUT_START` | M |
| **F2** | Live-data adapter for the prospective pipeline only: current prices, distributions and rates with `available_at` and retrieval times, written to the pipeline's store | OD16, OD9 | Nothing outside `prospective/` can read its output (path test) | S–M |

## Phase 2: MVP, Portfolio Lab (rules only)

| Ticket | Delivers | Depends on | Acceptance gate | Size |
|---|---|---|---|---|
| **P1-1b** | The first real publication and page: actual total-return history of two or three broad funds through 2024-12-31 (status `descriptive`, badge "Actual"); `theme.py` and `components.py` moved into a neutral `ui/` module; one Explore page | P1-1a, OD7, a slice of F1 | Every state renders; badges and knowledge time asserted by AppTest | M |
| **P2-1** | Model portfolio registry entries and backtest publications over the registered comparison window (to 2024-12-31); the comparison page | P1-1a, G1, T5, F1, OD6 | Registered before the first run; one window by asset-class rule; net of costs; references compared as in the PRD; separate series per evidence type; intervals on headline metrics | M |
| **P2-2** | Builder: user-chosen assets; fixed weights, equal weight or inverse volatility over risky assets; per-asset caps; cash as a floor or fixed weight; exposures, concentration, historical risk, stress scenarios, costs; a loss limit as a historical check; discrete options and a configurations counter | P1-1b, P2-1 data, T5 | Pure functions with hand examples; cash never enters inverse volatility; inputs not stored or logged; a copy test bans "recommended", "buy" and "sell" | M–L |
| **P2-3** | Watchlist explorer: descriptive characteristics, name sort, current-listing label | P1-1b, OD7 | No forecast field can render; partial-coverage and stale states | S–M |
| **P2-4** | Research mode: evidence pills and an Evidence ledger page | P1-1a | Every publication traceable to code, data and evidence records | S |
| **P2-5** | Cross-cutting UX: every first-class state, chart text alternatives, the mode switch, narrow layouts | All of the above | Accessibility and AppTest coverage per page | S–M |

**MVP gate:** the acceptance criteria in `vision_and_prd.md`, section 8.

## Phase 3: research evidence track (starts in parallel with phase 1)

| Ticket | Delivers | Depends on | Acceptance gate |
|---|---|---|---|
| **RES-1** | Survivorship-free data source and licence; security master on synthetic panels, then real data; U1 membership by rules R1–R8; delisting returns and a reinvestment rule; `signal_cutoff_at` | OD13, OD9, a label-semantics decision (delistings) | The universe document's data contract; a survivorship sensitivity bound; the MDE re-estimated on the real panel and pre-registered |
| **RES-2** | T4 follow-ups as pre-registered variants: zero-band hit rate, coverage-weighted IC, at least 199 permutation draws, date-weighted fitting or within-date features, the Elastic Net parameterization | Owner decisions per item (formula and metric gates) | Each variant recorded (G1) before it runs. **The set is frozen before any U1 result is read** |
| **RES-3** | Context V1 (rates, VIX, credit-risk proxy, regime descriptors) with the pre-registered ablation | RES-1 (for power), OD9 | The ablation design in the universe document |
| **RES-4** | T4 re-run on U1 at h = 20, then h = 60 | RES-1, RES-2, T5 | The ladder's L1 and L2 gates in development; nulls and controls; the MDE stated before results are read; a test family and α per cell |
| **RES-5** | T6: one holdout run, for everything pre-registered by then. The holdout window is fixed at its pre-registration | RES-4, G1 | Exactly one recorded run; the decision rule fixed in advance; the status earned is `holdout_checked` (a "no collapse" check) |
| **RES-6** | Prospective model forecasts through the G2 pipeline: append-only, hash-chained, written at the signal cutoff; scored with the registered confirmatory test on post-registration data only (T6 is a non-confirmatory baseline) | RES-4, G2 (activated) | No row is ever edited; review dates honoured; the planned looks and α-spending recorded in advance |

## Phase 4: validated risk, screener, and (much later) relative views

| Ticket | Delivers | Depends on | Gate |
|---|---|---|---|
| **P4-1** | Risk forecasts: mean-reverting volatility (GARCH or HAR), shrinkage covariance, then a factor model; risk-forecast publications | T5, F1 | Ladder D1, first in development. Reaching Explore and Build needs a prospective **equivalence** test of coverage within a pre-registered margin, run through G2. A non-rejection check is not enough: its power is provisionally about 0.34–0.43 (the two reviews' setups differ: 126 non-overlapping windows, α = 0.05, a 90% band whose true coverage is 85%; about 0.17 at h = 60) |
| **P4-2** | Opportunity screener over U1 (characteristics only) | RES-1 | Point-in-time membership and eligibility reasons on every row |
| **P4-3** | Relative model views. Research mode only, until prospectively confirmed; then Explore and Build at the claim level reached | RES-5, RES-6 | `prospectively_confirmed` through the registered confirmatory test (`vision_and_prd.md`, section 15). **Provisionally, confirming an IC of 0.03 with about 100 names needs about 1,170 post-registration dates (about 4.6 years; about 7 years for a persistent signal).** Twelve months detect a collapse with only about 25–70% power, depending on α and the overlap factor (provisional). Plan for years, not months |

## Phase 5: calibrated forecasts

Each step needs its rung on the ladder, nested-fold calibration and drift monitors:
- **L3** expected excess return, shown MZ-recalibrated;
- **D2** P(outperform);
- **alpha as a bounded, cash-neutral tilt** inside a tracking-error budget (optimizer dependency, OD14).

## Phase 6: absolute forecasts and more asset classes

- **Market layer research.** Start from descriptive regimes and volatility targeting; L4 needs Campbell-Thompson and Clark-West against the expanding mean, and the composed forecast tested as a whole.
- **Composition method.** The exact sum by default; the beta adjustment only under its documented approximation, chosen by pre-registration and checked with the market-exposure diagnostic (`architecture.md`, section 5). A formula gate.
- **P(gain) and expected downside**, only after L4 and D1/D2 on absolute returns.
- **Treasury and cash forecasts**, on their own ladder.
- **Crypto**, needing a 24/7 calendar contract, a licensed source and counsel.

## Phase 7: shared or commercial readiness (only if the owner chooses)

- **Legal:** counsel on adviser status and the Marketing Rule.
- **Data:** commercial data licences; models re-fitted on commercially licensed data.
- **Platform:** hosting on architecture C over B's contracts; authentication and TLS.
- **Privacy:** a privacy program and retention rules.
- **Security:**
  - hash-locked dependencies with vulnerability scanning;
  - redaction of secrets from errors (the FRED key travelled in request URLs that error messages printed; fixed in PRs #48 and #50, pending merge);
  - **FRED Version 1 URL logging:** urllib3 logs Version 1 request URLs, `api_key` included. Keep `urllib3.connectionpool` above DEBUG, mount no retrying adapters for Version 1 calls, and move to header authentication if FRED accepts it or redact `api_key=` in a logging filter;
  - an audit trail of what was shown, without personal data.
- **Testing with others:** comprehension testing with real users, on synthetic data.

## Promotion stages (`vision_and_prd.md`, section 15)

| Stage | Reached by | Earliest |
|---|---|---|
| S0 Research and development | Today | Now |
| S1 Holdout-checked (non-confirmatory) | RES-5 (T6) | After RES-4 |
| S2 Prospective paper trading | G2 activated (OD16: separate approvals of activation and H_end); registration. S1 is not a gate. Models and policies also need development evidence and a registered confirmatory plan | Rules-only portfolios as soon as G2 and P2-1 exist; models and policies once registered |
| S3 Controlled personal deployment | Per artifact class:<br>• rules-only: exact replay, vintage-drift and rebalance-count checks;<br>• model claims: prospective confirmation;<br>• policies: non-inferiority at a pre-registered margin permits at most a capped trial and claims no value; any value claim or scale-up needs superiority over every registered baseline and positive net value against cash (provisional rules).<br>Plus S3's operational and risk controls | About a year for rules-only allocations; years for model claims and policies |
| S4 Public advisory product | Counsel, registration path, licences, security and privacy programs, a multi-year record | Only if the owner chooses |

**Starting the clock.** Evidence accrues only with time. Paper-tracking the registered rules-only portfolios as soon as G2 exists costs little, and it is the first prospective record the project will have. It needs OD16's separate approvals first (the pre-registration procedure, H_end and activation), because the holdout end must be fixed before any prospective registration.

## Critical path and parallelism

```text
Product:   P0 ─► P1-1a ─► P1-1b ───────────────► P2-3, P2-4
           P0 ─► T5 (formula review first) ─┐
           P0 ─► F1 ─────────────────────────┼─► P2-1 ─► P2-2 ─► MVP gate ─► P4-1
           P0 ─► G1 ─────────────────────────┘
           G1 ─► G2 (OD16) ─► paper tracking of registered portfolios (S2) ─► …
Research:  P0 ─► OD13 data source ─► RES-1 ─► RES-2 (frozen) / RES-3 ─► RES-4 ─► RES-5 (T6, S1) ─► RES-6 via G2 (S2, years) ─► P4-3 ─► Phase 5 ─► S3
```

- The product track reaches its MVP gate without any forecast.
- The research track is gated by the data-source decision (OD13). **Start it now:** it has the longest lead time, and the MVP does not depend on it.

## Acceptance gates for every ticket

- AGENTS.md workflow:
  - inspect, test first, `python3 scripts/verify.py`, `/project-verify`;
  - commit on a branch, open a pull request; the owner merges.
- Point-in-time and lockbox rules are unchanged unless an owner decision changes them.
- Formula, data-contract, dependency and guardrail changes wait for their approval gates.
- Learning objective: each statistical core ships with explanations and Learning Notes. One optional, bounded exercise per milestone, never blocking. Candidates:
  - fixed-b critical values;
  - a date-clustered coverage and PIT scorer;
  - Brier decomposition with Platt scaling;
  - adaptive conformal with delayed feedback;
  - a simulation of the overlap factor F(h);
  - smoothed PSI;
  - a power calculation for prospective confirmation.
