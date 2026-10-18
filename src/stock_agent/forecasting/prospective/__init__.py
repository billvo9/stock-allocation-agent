"""
The prospective forecasting track (docs/research/ml_prospective.md): ETF
forecasters for VTI, AGG and SHV at 5, 20 and 30 sessions, frozen before the
first forecast, each forecast recorded once and scored only as real outcomes
arrive. Forecasts are research records, never recommendations.

Modules:
    config      the registered config (strict YAML, hashed)
    series      F5.1 total returns, daily log returns, features, targets
    training    F1 research store -> frames; development folds; final refit
    bands       split-conformal 80% bands and their development coverage
    frozen      frozen bundles (plain numbers, no pickle)
    freeze      develop, refit, write and register, once
    window      the inference-window contract and timing conventions
    ledger      the write-once, hash-chained forecast ledger
    registration_record
                the registration record and the ledger's registration entry
    cadence     session 1, the trial and section 5's conditions (diagnostics)
    forecast    forecast records, scoring of matured records, coverage report
    refusal     the track's machine-readable refusal
    zero_vol    zero realized volatility: the rule in force and research B2
    registration
                the arithmetic of the proposed registration rules
    synthetic   fixtures (tests and demonstrations only)
"""
