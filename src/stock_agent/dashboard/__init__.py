"""
Read-only research dashboard over saved T2 run outputs (ticket T3).

Modules:
    loader     verified, lockbox-guarded access to run directories
    status     run status and research notices from stored checks
    figures    pure Plotly figure builders over stored tables
    glossary   short definitions shown beside technical terms
    views      Streamlit page bodies (layout and explanation only)
    app        Streamlit entry point (launch with scripts/run_dashboard.py)

The design system (tokens, chart style, shared components) lives in the
neutral `stock_agent.ui` package, which the product app shares.

The dashboard never fits a model, recomputes inference, or reads raw data.
Nothing outside this package imports it.
"""
