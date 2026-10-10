"""
Measurement layer for model results: null models and controls run through
the validation harness, model-agnostic out-of-sample metrics, dependence-
aware uncertainty, leakage-safe feature diagnostics, and a reproducible run
record and saved-output contract.

Modules:
    controls             null, control and canary estimators; control columns
    placebo              block-permutation null and stale-feature control
    canary               the deliberately unpurged canary reference
    scoring              per-date metrics, calibration, rank positions
    inference            Newey-West, fold-block t, circular block bootstrap
    feature_diagnostics  training-scope statistics and evaluation drift
    seeds                deterministic seed derivation
    record               canonical JSON, spec hash, run id
    contract             saved table schemas and version
    runner               pure orchestration of a development run
    artifacts            the only file I/O: run directories and ledger

Dependencies point one way: model_diagnostics imports model_validation and
features.training, never the reverse, and never data loaders.
"""
