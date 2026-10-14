"""
The Model Health Center: health checks, versioned health summaries and the
Model Health page (docs/operations/model_health.md, docs/operations/runbook.md).

Modules:
    model         states and precedence, Check, Alert, the versioned summary
                  and its strict reader (standard library only)
    alerts        informational alerts from checks and the previous summary
    summary       build a summary from checks; write it atomically
    context       GitHub Actions run metadata (allowlisted) and the git code state
    verification  the latest verify.py result and CI runs on main
    dependencies  pins installed at their pin; pip check
    contracts     publication, F1 fund-data and generator-hash probes (synthetic)
    data_quality  freshness, coverage, missing sessions, source discrepancies
    smoke         the monitors and readers run on planted synthetic defects
    inputs        producers' health inputs (agreed v1.0): envelope, newest file,
                  transition status, worst-of findings
    training      training health from the RL track's file
    registry      registry health through the ML track's read API
    forecast      forecast coverage and calibration from the registry export
    backtest      portfolio limits from a backtest summary
    fixtures      example producer inputs (synthetic), the registry via its API
    collect       the public-safe CI run and the private local run
    view          the page's pure view model
    page          the Streamlit page (shared `stock_agent.ui` components)
    app           the standalone page's Streamlit entry point

Boundaries:
    - Read-only towards everything it monitors: no check writes to a store,
      a registry, a model or a configuration, and alerts are informational.
    - The CI run reads only repository code and synthetic fixtures; private
      stores are read only by the local run on the owner's machine.
    - No network access anywhere in this package.
"""
