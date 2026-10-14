"""
Out-of-sample forecasting research: the experiment runner, its metrics,
versioned forecast artifacts and synthetic data (docs/research/ml_*.md).

Modules:
    artifacts   out-of-fold forecast artifact contract, writer, reader and
                the as-of selection consumers use
    config      strict YAML experiment configs and the config hash
    experiment  the runner: one config, one reproducible experiment
    metrics     coverage, zero-filled rank IC, paired increments, Holm,
                interval coverage
    nonlinear   the HistGradientBoosting challenger and its grid
    provenance  Git and environment facts for the run record
    synthetic   seeded synthetic panels with planted, known signals

Dependencies point one way: forecasting imports model_validation,
model_diagnostics, features and registry, never the reverse.
"""
