# Stock Lab: the sealed holdout evaluation

Status: **registered and built offline. Nothing has been fetched, read or
run.** The fetch and the run each wait for the owner's separate go.

## 1. The decision

- **2026-10-26.** On the next step for the frozen Stock Lab model, the owner
  chose "Evaluate the model first, then open (Recommended)". The model is
  scored once on the protected period it never saw. After that, the period
  counts as spent and is opened for fixed-rule descriptive backtests only.
  Research code keeps refusing 2025 onward for training, tuning and
  selection.
- **2026-10-26.** The owner approved the protocol below as proposed
  ("Approve as proposed (Recommended)"). This covers quarterly blocks with
  t(6), one-sided at 2.5%. Only the pooled daily RPS of B1 against B0
  decides; the five-session total and every other item are descriptive. The
  evaluation runs once.

Both choices were made in the lead session and are recorded there. This
document registers the protocol they approved.

## 2. The registered protocol

The protocol lives in code (`stock_lab.holdout.protocol.PROTOCOL`), and its
hash is recorded with the result. `tests/test_stock_lab_holdout_protocol.py`
pins the hash below, so any change to a rule fails the gates until it is
approved again. The hash was written before any protected price was fetched.

```text
sha256 820d154e5d3a31019be692b809a9e0a953046ad8de4e38fcb2689c3fc603856c
```

```json
{"alpha":0.025,"blocks":[["2025Q1","2025-01-01"],["2025Q2","2025-04-01"],["2025Q3","2025-07-01"],["2025Q4","2025-10-01"],["2026Q1","2026-01-01"],["2026Q2","2026-04-01"],["2026Q3","2026-07-01"]],"calibration_level":0.9,"comparator":"B0","deciding_score":"rps","deciding_target":"daily_pooled","deciding_test":"fold_block_t_one_sided","fetch_start":"2024-01-02","first_decision":"2024-12-31","holdout_end":"2026-10-14","holdout_start":"2025-01-02","horizons":5,"interval_level":0.95,"model_id":"2b6cd0c1a52f5ecfc16d6c8e221956d6fd23093826618e895949ac62e7fa65e7","schema":"stock_lab_holdout_protocol","scores":["rps","brier","log_loss"],"tickers":["AMD","MRVL","MU","NVDA"],"window_sessions":252}
```

In plain words:

| Item | Rule |
| --- | --- |
| Model | `2b6cd0c1a52f5ecfc16d6c8e221956d6fd23093826618e895949ac62e7fa65e7`, used exactly as registered: B1, half-life 252, state rule `frozen_long_run`. Nothing is refit, retuned or reselected. |
| Forecasts | At each decision session d, each stock's forecast comes from the 252 sessions ending at d, through `build_panel` and `registry.probabilities`, normalized and refused without a finite volatility as `frozen_forecast.forecast_record` does: the evening job's computation. B1 counts the window's outcomes and shrinks them toward B0, the long-run mix frozen at 2024-12-31. The data path differs: the evening job's reader refuses windows reaching back before 2025, while here the first windows hold 2024 development sessions from the holdout fetch. |
| Stocks | AMD, MRVL, MU, NVDA. |
| Decisions | Every NYSE session from 2024-12-31 to 2026-10-07: 443 decisions. |
| Outcomes | 2025-01-02 to 2026-10-14 (447 sessions). Every target session falls inside this range. |
| Targets | As in development: each of the next five sessions as Decrease, Stable or Increase against ±0.5%, plus the five-session total against ±0.5%·√5. |
| Scores | RPS, Brier and log loss (lower is better). |
| Deciding test | One question: does B1 beat B0 on RPS? For each decision, average the B1 − B0 RPS over the stocks for each daily target, then over the five daily targets. This is development's pooled daily differential. Test it with A7's fold-block t, one-sided at 2.5%. The blocks are calendar quarters: 2025Q1 (which also takes 2024-12-31) through 2026Q3 (which also takes the October 2026 decisions). That gives 7 blocks and 6 degrees of freedom; the critical t is 2.447. There is one comparison, so no Holm correction. |
| Descriptive only | Brier and log loss, and the five-session total, each with the same test. Calibration buckets with the briefing's date-clustered 90% Jeffreys intervals (lag 0 for a single session, lag 4 for the total; groups never pooled). Breakdowns by stock and by year. The holdout's own base rates against the frozen B0. |
| Runs | Exactly one. The result is a write-once record in the private holdout store, and a second run is refused. Nothing goes into the repository. |

Block sizes, in decisions: 2025Q1 61, 2025Q2 62, 2025Q3 64, 2025Q4 64,
2026Q1 61, 2026Q2 62, 2026Q3 69.

## 3. The holdout path (`stock_lab.holdout`)

- **`protocol`**: the rules above. It also checks that a calendar lines up
  with them.
- **`store`**: the private holdout store, `stock_lab_holdout.sqlite3`.
  - It lives in `~/stock-agent-private/stock_lab_holdout/` or the directory
    named by `STOCK_AGENT_STOCK_LAB_HOLDOUT_HOME`.
  - It refuses a directory inside a Git working tree and a directory holding
    any other database. Its file header carries its own application id, so
    the research store refuses to open it.
  - It holds one write-once batch per stock, ever, and one evaluation, ever.
  - The bars are handed out only inside `evaluate_once`. That call refuses
    once an evaluation exists, reads and checks every stock's bars, runs the
    evaluation, and stores the sealed record before it returns. After that
    the period is spent: nothing can be added and the bars are not read
    again.
- **`evaluate`**: the run itself. It is pure: bars go in and a sealed record
  (its own content hash) comes out.
- **`report`**: the record in plain words.

Isolation is tested in `tests/test_stock_lab_holdout_boundaries.py`. Only the
two owner-run scripts import the package. Research, evaluation, training,
tuning and backtest entry points and scripts load nothing from it.

## 4. The owner's commands

The fetch makes **4 requests**, one per stock: raw daily bars for
2024-01-02 to 2026-10-14 from `api.tiingo.com`. The key comes from
`TIINGO_API_KEY` in the shell that runs the command, and is sent only in the
Authorization header.

```bash
.venv/bin/python scripts/stock_lab_holdout_fetch.py
```

prints the plan and makes no request.

```bash
.venv/bin/python scripts/stock_lab_holdout_fetch.py --authorized --store
```

fetches the bars, audits them, and stores each ready stock.

- A stock is ready when its audit passes, every session was served, and its
  2024 closes, dividends and split factors equal the research store's (the
  history the model was frozen from).
- Any other stock is held for review, not stored.
- A stock already stored is never requested again.

```bash
.venv/bin/python scripts/stock_lab_holdout_evaluate.py --run
```

is the one run. It needs a clean checkout, records the commit, and prints the
report. `--show` prints the stored result later.

## 5. What to expect, and the limits

B1 is mostly the last year's mix. Over a 252-session window with half-life
252, the recency weights sum to about 181.5, so B0's prior (κ = 40) carries
about 40 / 221.5 ≈ 18% of the weight. If a stock's category shares in
2025–2026 equal its long-run shares, B1 only adds sampling noise (about ±3
percentage points per category) and is expected to score slightly *worse*
than B0. It can win only if the mix shifts and the shift persists, for
example between volatility regimes. The expected result is a small
difference either way.

- "No" means the evidence did not reach the pre-registered 2.5% level. The
  sign and the interval are descriptive. A positive difference with an
  interval above zero describes a worse score in this period, which the
  one-sided test does not assess.
- "Undetermined" means the test was unavailable or a block lost all its
  scored decisions.
- A "yes" would be evidence against B0 for this period and these stocks,
  not evidence of a lasting edge.
- The stocks were chosen in hindsight.
- This is one 21-month period, with low power.
- Prices are raw closes and corporate actions as served on the fetch date,
  not point-in-time vintages.
- The fetch prints each stock's audit before the run. For the protected
  period, that includes counts and dates of large moves and of ex-dates.
  This is harmless to the protocol, which was hashed before any fetch, but
  it is a look at protected data.

## 6. Learning notes

**Problem.** Development chose B1 over its rivals using walk-forward folds
on 2003–2024. Every one of those choices looked at the development data,
however carefully. A sealed holdout is the one look at data that played no
part in any choice. It measures how the selected model does on new data,
which is less flattering than its development score, since that score was
part of how it was chosen.

**Why only once.** Each look at the holdout that can change a decision is a
selection step. If you look, adjust and look again, the holdout becomes
development data. That is why the protocol was fixed and hashed first, and
why the store refuses a second run. It is the same logic as the multiple-
testing count in the AGENTS.md research standards.

**Scores.** RPS for three ordered categories is

```text
RPS = ½ Σ_{k=1}^{2} (F_k − O_k)²
```

where F_k is the forecast's cumulative probability and O_k is the observed
cumulative indicator. It is a proper scoring rule: a forecaster minimizes the
expected score only by stating true beliefs. It also respects order: calling
Increase when the outcome is Decrease costs more than calling Stable.

**The test.** Let d_t be the per-decision pooled B1 − B0 difference. Its
block means are m_1 … m_7 (one per quarter), and

```text
t = mean(m) / (sd(m) / √7),  6 degrees of freedom; B1 beats B0 when t < −2.447
```

Why blocks? Neighbouring decisions share outcome sessions (each daily
outcome is scored by five decisions) and market moods, so the 443 daily
differences are far from independent. The test *assumes* that the 7
quarter means are independent and roughly normal. At each quarter boundary,
the last four decisions share outcome sessions with the next quarter, and
volatility regimes persist across quarters. Both make the t somewhat
anti-conservative. Development's size simulation covered yearly blocks, not
quarterly ones.

**Worked example.** Take block means of −0.0020, −0.0010, +0.0005, −0.0015,
−0.0008, +0.0002 and −0.0010:

- mean = −0.00080, sd = 0.00089, se = 0.00033;
- t = −2.39 and the one-sided p is 0.027;
- that is not below 2.5%, so the answer is "no", even though B1 was better
  in five of the seven quarters.

With 6 degrees of freedom, a real but small edge is easy to miss: the power
is low.

**Leakage risks, and how they are closed.**
- Each forecast reads only its window, the sessions up to d. A test
  scrambles every later price and checks that the forecast at d is
  unchanged.
- Outcomes start at d+1.
- Nothing is fitted.
- The 2024 sessions are development data, used only as inputs to the first
  windows. They are checked against the research store.
- The evening job reads the same protected period, but only as inputs to the
  frozen model, never to choose one.

**How the result could mislead.**
- One period is one draw. 2025–2026 may be unusually volatile (or calm) for
  these four stocks, which would move the stable share and the base rates.
- The four names were picked knowing they had done well, which selects for
  persistence in returns, not in categories.
- A significant result would describe this period, not a lasting edge.

**How practitioners validate.**
- Fix the protocol and register it (the hash) before looking.
- Report all pre-registered scores, not just the best one.
- Compare against the simplest credible benchmark (here B0).
- Treat the holdout as spent.
- Judge real skill on a prospective record that keeps growing, like the
  evening job's ledger.

**What to learn next.**
- Statistical power for block t-tests: how many quarters of a given edge it
  takes to detect it at 2.5%.
- The Diebold-Mariano test, the time-series standard for comparing forecast
  scores.
- Pre-registration, and the "garden of forking paths" in empirical finance.

**Small exercise.** Using the example's block means, compute how many more
quarters with the same mean and spread would push t below −2.447. Then
compare that with the t(6) critical value at 5%.
