"""
The Portfolio Lab: the product app, separate from the research dashboard.

Modules:
    history    typed view inputs: a fund's daily records and their labels
    provider   the descriptive-metrics protocol (implemented by T5-A) and the
               pending provider used until OD8 group A is approved
    synthetic  seeded synthetic demo funds (no I/O)
    labels     the page copy: caveats, badges, state messages
    view       pure view logic: periods, gaps, provider calls, formatting
    figures    pure Plotly figure builders
    explore    the Explore page "Fund history" (Streamlit)
    run        Streamlit entry point (launch with scripts/run_portfolio_lab.py)

Boundary (tested): the app imports only `stock_agent.ui`, its own modules,
pandas, numpy, plotly, streamlit and the standard library. It never imports
the research dashboard, models, raw data, features or the publication writer,
and it computes no financial formula itself.
"""
