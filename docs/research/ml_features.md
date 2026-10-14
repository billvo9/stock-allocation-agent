# Feature contracts, frozen universes and invariant labels

Status (2026-10-14): implemented on `feature/ml-feature-contracts` (stacked on the registry PR); merge pending.

Code:
- `features/contracts.py`, `features/families.py`, `features/missingness.py`, `features/asof.py`;
- `features/frozen_universe.py`, `features/invariant_labels.py`;
- `forecasting/synthetic.py`.

Tests: `tests/test_feature_contracts.py`, `tests/test_frozen_universe_labels.py`.

**Scope.** This is preparatory work under the owner's 2026-10-13 approval of FE-0 and FE-1 "as contracts, synthetic tests and outcome-blind feature-build designs". Nothing here is scored on the four sandbox equities or on real data. The E1-E3 pre-registration stays pending.

## Feature contracts

Every model feature declares a `FeatureContract`.

| Field | Meaning |
|---|---|
| `family` | `relative_technical`, `fundamental_quality` or `risk_liquidity` |
| `availability` | `session_close` or `per_row_timestamp` (below) |
| `availability_column` | Required for `per_row_timestamp`: the column holding each value's `available_at` |
| `lookback_sessions` | Earlier sessions of the same symbol the value needs (warm-up) |
| `missingness` | `impute_training_median`, `native_missing` or `refuse` |
| `cross_sectional` | `none`, or `within_date_rank` (ranked among the eligible names of the same date only) |
| `inputs`, `description` | Required |

**Availability rules.** Neither rule changes an existing convention.
- **`session_close`.** Computed from the symbol's own prices and volumes through the close of the row date d, looking backward only. This is the existing `DECISION_CONVENTION`: features include the close of d, and labels enter at the close of d + entry_lag.
- **`per_row_timestamp`.** Every value carries its source observation's `available_at`, and may be used on row date d only if `available_at <= d` at 00:00 UTC. This is the existing panel rule (`features/panel.py`, open item 1). A value without a timestamp is refused.

**Validation** (`validate_feature_contracts`) never modifies data. It checks, in order:
1. **No contract, no feature.** It refuses any feature without a contract: "no availability semantics".
2. **The existing name rules.** It runs `validate_feature_columns`, so a contract cannot launder a label, key, timestamp or price level (tested).
3. **Availability.** Per-row availability must hold. A violation is a `LeakageError`.
4. **Missingness policies.** `refuse` rejects missing values. `native_missing` needs an estimator that handles NaN.

**Never drop rows.** No policy drops rows. Membership and labels stay independent of feature missingness (owner decision, 2026-10-13).

## Families

| Feature | Family | Availability | Lookback | Definition |
|---|---|---|---|---|
| `rel_return_5d` | relative_technical | session_close | 5 | 5-session return, ranked within the date: (rank − 1)/(n − 1) − 0.5 |
| `rel_momentum_20d` | relative_technical | session_close | 20 | 20-session return, ranked within the date |
| `rel_momentum_120_20d` | relative_technical | session_close | 120 | Return from 120 to 20 sessions ago, ranked within the date |
| `volatility_20d`, `volatility_60d` | risk_liquidity | session_close | 20, 60 | Standard deviation (ddof 1) of daily log returns |
| `log_amihud_20d` | risk_liquidity | session_close | 20 | log of the 20-session mean of \|log return\| / (close × volume) × 1e9 |
| `quality_gross_profitability` | fundamental_quality | per_row_timestamp | 0 | Gross profit / total assets of the current filing |
| `quality_leverage` | fundamental_quality | per_row_timestamp | 0 | Total liabilities / total assets of the current filing |

- **Ranking.** Relative features rank among the date's eligible names with finite values. They are NaN when fewer than two such names exist.
- **Current filing** (`features/asof.py`). At 00:00 UTC of d, the current filing is the latest period known by then, and that period's latest version known by then. A late amendment of an older period never becomes current. This restates `data/fundamentals/point_in_time.py` for plain observation frames. A filing accepted at 21:30 UTC on day X is used from session X + 1 (tested).
- **Amihud.** Amihud needs a price *level* times a volume. Vendor-adjusted levels embed later corporate actions, so on real data `log_amihud_20d` needs unadjusted closes and volumes. The synthetic source has no corporate actions.

**Prefix stability (tested).**
- Every family's values for dates before T are identical when built from data known before T: price rows dated before T and filings available before T.
- The repository's `assert_prefix_stable` passes the price families and catches a planted feature that reads the next close.

## Missingness diagnostics

`missingness_table(frame, contracts, folds=...)` gives every missing value one reason:

- `warmup`: a `session_close` feature whose symbol has at most `lookback_sessions` price rows by d;
- `not_yet_available`: a `per_row_timestamp` feature with no observation available yet;
- `source_missing`: the inputs existed but gave no value.

**Scopes.** Rows are counted per feature for:
- the panel (overall and per symbol);
- each fold's training rows;
- each fold's evaluation rows.

**Warnings.** A rate above `warning_rate` (default 25%) gets status `warning` and reason `high_missing_rate`. Nothing is imputed, dropped or modified (tested).

## Frozen eligible universe

**Membership.** Membership of symbol s on session d depends only on s's own price rows dated on or before d:

- **valid**: a finite, positive close;
- **history**: the valid closes up to and including d;
- **eligible**: valid and history ≥ `min_history_sessions`.

Reasons are `invalid_price` and `insufficient_history`. A symbol without a row on d (not yet listed, or delisted) is not a member.

**Properties (tested).**
- **Point-in-time.** Membership reads only earlier rows of the same symbol, so it is prefix-stable.
- **Independent of features and labels.** `assert_membership_independent` refuses a model frame whose rows differ from the eligible keys, for example one that dropped rows with missing features.
- **Frozen.** The membership table is hashed with its rule and id. `verify_universe` refuses an edited table, and a different rule gives a different hash.
- **Survivorship-free.** This is a declared property of the price source, recorded, never inferred. The synthetic panel keeps every symbol that ever existed, delisted ones included.

## Invariant labels

`frozen_universe_labels(prices, universe, LabelSpec)` labels exactly the eligible keys:

- **Raw label.** `add_forward_return_target`: entry `entry_lag` rows after d, exit `horizon` rows after entry, using the symbol's own rows.
- **Excess label.** The T4 function `add_equal_weight_excess_target`, applied to the eligible rows only, so C_t is **the frozen universe's eligible members on t**. Its availability and fail-closed rules are unchanged:
  - each row on t takes the latest member end and the earliest member start;
  - a member without a finite raw label (for example one that delisted before its exit) blanks the whole date (tested).

This differs from the T4 label only in C_t: T4 uses "names with a feature row on t". It is used by the synthetic experiment runner. **Applying it to real data needs the owner's approval** (label semantics, with delisting returns: "Mandatory before large-universe validation" in `linear_models.md`).

**Prefix invariance** (`assert_labels_prefix_invariant`). For each cutoff T, a build from only the rows dated before T gives:
1. the same label, entry and exit for every row it labels;
2. a label for every row whose full-data label ends before T.

**Tested.**
- Raw and excess labels hold this on aligned calendars and on a calendar where one symbol misses every 7th session (so its labels end later than its peers').
- Negative tests:
  - a "survivors-only" benchmark (the mean over names whose label is known) is caught;
  - a label divided by a full-sample volatility is caught.
- Prefix invariance cannot catch every survivorship error. A benchmark that silently drops a name that never gets a label looks the same at every cutoff. That is why the fail-closed rule, not this check, is the defence there.

## Synthetic equity generator

`synthetic_equity_panel(SyntheticEquitySpec)`:
- daily log returns r_it = β_i m_t + μ_it + σ_i e_it;
- quarterly filings of a persistent quality characteristic (AR(1), φ = 0.9), filed 30 to 55 days after quarter end at 21:30 UTC, with about 10% later amendments;
- late listings and delistings.

**The planted drift uses only observable information.** z_it is the within-date standardized gross profitability of the current filing on t.

| `signal` | μ_it |
|---|---|
| `none` | 0 |
| `linear` | s · z_it |
| `nonlinear` | s · (\|z_it\| − mean over the date of \|z\|) |

With the default s = 4e-4, a 20-session label has an expected excess of about 0.008 · z (linear), against a 20-session idiosyncratic standard deviation of about 0.08. The implied rank IC is about 0.1. The nonlinear drift is symmetric in z, so a linear model cannot use it and a tree ensemble can.

**Properties.**
- Leverage, volume and the price features carry no planted signal.
- The generator is deterministic (legacy `RandomState`).
- Calendars must end before 2025-01-01.
- Synthetic data verifies the pipeline. It is never evidence about markets.

## Learning Notes

1. **Problem.** Make "when is this value known?" a property every feature must declare, so a leak is a validation error, not a review comment.
2. **Concept.**
   - **Availability timestamps.** A value is usable only after the latest availability of every input it uses.
   - **As-of joins.** Take the latest observation known at the decision time, never the latest overall.
   - **Prefix stability.** Outputs for dates before T must not change when data after T arrives.
3. **Assumptions.** Raw prices without corporate actions (synthetic). 00:00 UTC of d as the fundamentals cutoff (the existing, conservative rule). The as-of rule of the EDGAR history.
4. **Formulas.**
   - Rank feature: (rank − 1)/(n − 1) − 0.5 within the date.
   - Volatility: sd(log returns, ddof 1).
   - Amihud: mean(\|r\| / (P·V)).
   - Excess label: y_it − (1/|C_t|) Σ_{j∈C_t} y_jt, with C_t the eligible members on t.
5. **Small example.**
   - A Q3 filing arrives on day 10 at 21:30 UTC, Q4 on day 40, a late Q3 amendment on day 50 and a Q4 amendment on day 60.
   - The current value is: none on day 10; Q3 on days 11 to 40; Q4 original from day 41; the Q4 amendment from day 61.
   - The Q3 amendment never becomes current (the test reproduces this).
6. **Leakage risks guarded.**
   - Filings used before acceptance.
   - Later amendments overwriting history.
   - Features reading the next close.
   - Benchmarks over survivors.
   - Normalizing with full-sample statistics.
   - Dropping rows with missing features, which changes C_t and the excess label.
7. **Ways results could mislead.**
   - Pooled correlations of persistent noise features (leverage) can look non-zero with few names. Judge features within dates and with fold-block inference.
   - Within-date ranks discard magnitude: a model on ranks forecasts ordering, not size.
8. **Practitioner validation.**
   - Point-in-time databases with vintage tables.
   - Prefix tests at many cutoffs.
   - Hold the universe fixed before looking at outcomes (Fama-French construction).
9. **What to learn next.** Gu, Kelly and Xiu (2020) on rank-normalized characteristics; how CRSP and Compustat point-in-time products record restatements; Shumway (1997) on delisting returns.
