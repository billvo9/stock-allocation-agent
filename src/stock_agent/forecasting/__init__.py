"""
Out-of-sample forecasting research: versioned forecast artifacts and (in
later modules) the experiment runner, metrics and research tracks.

Modules:
    artifacts  out-of-fold forecast artifact contract, writer, reader and
               the as-of selection consumers use
    synthetic  seeded synthetic panels with planted, known signals

Dependencies point one way: forecasting imports model_validation,
model_diagnostics, features and registry, never the reverse.
"""
