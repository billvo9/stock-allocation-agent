"""
The registered synthetic comparison of the baselines against PPO, run through
the approved T5a simulator (D1-D8, approved 2026-10-15; the D5 and D6
confirmations of 2026-10-17).

- `protocol`: the registered configuration (strategies, constraint sets,
  cost multipliers, learned seeds, the RL run configuration) and its hash.
- `fairness`: the fairness contract every strategy is checked against, and
  the check that no test-fold result selected anything.
- `measures`: the approved measures of one episode.
- `ledger`: the write-once, hash-chained evaluation log that counts every
  look at a fold.
- `report`: the versioned comparison artifact and the `backtest_summary`
  health inputs.
- `runner`: training with dev-fold selection, then one evaluation.

Synthetic scenarios only: nothing here is evidence of real investment
performance. This package runs no network request and reads no real data.
"""
