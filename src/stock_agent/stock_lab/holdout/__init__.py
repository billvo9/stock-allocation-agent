"""
The Stock Lab's sealed holdout evaluation: one run, ever, of one frozen model
on the protected period (2025-01-02 to 2026-10-14). Owner decision of
2026-10-26 ("Evaluate the model first, then open"): the frozen model is
scored once on the period it never saw, against B0, under a protocol fixed
before any protected outcome was read. Afterwards the period counts as spent
and is opened for fixed-rule descriptive backtests only; research code keeps
refusing it for training, tuning and selection.

Modules (this file imports none of them):
    protocol  what is evaluated, on which sessions, with which scores and
              which deciding test (`PROTOCOL`, fixed in code; its hash is
              recorded with the result)
    store     the private holdout store: the fetched bars (one write-once
              batch per stock) and the one evaluation record. It hands the
              bars out only inside the run that records the evaluation, and
              never once an evaluation exists: the period is then spent
    evaluate  the run itself (pure: bars in, a record out): the frozen
              model's forecasts from each decision's 252-session window,
              exactly as the evening job computes them, scored on the
              outcomes that followed
    report    the record in plain words

Isolation (tests/test_stock_lab_holdout_boundaries.py):
    - no module outside this package imports it; only the two owner-run
      scripts do (scripts/stock_lab_holdout_fetch.py and
      scripts/stock_lab_holdout_evaluate.py);
    - research, evaluation, training, tuning and backtest code never imports
      it or names its file;
    - the store is its own SQLite file in its own directory, marked in the
      file header (`application_id`): the research store refuses to read or
      write it, and this store refuses a directory holding any other
      database;
    - nothing here touches the network or reads a clock except the store's
      injected clock.
"""
