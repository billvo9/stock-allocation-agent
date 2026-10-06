"""
Leakage-safe model validation: purged expanding walk-forward folds, fold-fitted
preprocessing, an auditable fit/predict harness, and leakage checks.

Modules:
    folds          fold construction (purge, lockbox, universe restriction)
    checks         leakage invariants (feature contract, prefix stability, folds)
    preprocessing  fit-once, train-only transforms (numpy only)
    harness        walk-forward fit/predict loop with a fresh estimator per fold
    audit          canonical fingerprints, fold metadata, run manifest
"""
