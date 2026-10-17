"""
The Portfolio Lab: the product app, separate from the research dashboard.

Seven destinations share one top navigation (`run`, `navigation`, one page
script each in `destinations/`): Overview, Explore, Portfolio Builder,
Stock Probability, Experiments, Model Health and Evidence.

Modules:
    history       typed view inputs: a fund's daily records and their labels
    provider      the descriptive-metrics protocol, implemented by
                  `stock_agent.portfolio.descriptive.DescriptiveCalculator`
    synthetic     seeded synthetic demo funds (no I/O)
    labels        the Explore page's copy: caveats, badges, state messages
    wording       the other destinations' copy
    view          pure view logic: periods, gaps, provider calls, formatting
    mix           pure Builder logic: weights, concentration, what-if
                  weight alternatives (no performance: awaiting approval)
    status        pure data status for the Overview and Evidence pages
    figures       pure Plotly figure builders
    prose         markdown-safe writing for the newer pages
    persist       choices kept while another page is shown
    explore       the Explore page "Fund history" (Streamlit)
    overview      the Overview page (Streamlit)
    builder       the Portfolio Builder page: weights and concentration (Streamlit)
    forecast_records  the prospective forecast ledger, re-verified and laid out (pure)
    experiments   the Experiments page: recorded prospective forecasts, else an
                  honest empty state (Streamlit)
    health_summary  the Model Health page's inputs: the health summaries read
                  with the health center's strict reader, grouped into areas (pure)
    model_health  the Model Health page (Streamlit)
    evidence      the Evidence page: provenance and assumptions (Streamlit)
    stock_lab     the Stock Probability page: the saved watchlist and each stock's
                  daily history (Streamlit; experimental, `stock_agent.stock_lab`)
    stock_figures the Stock Probability page's Plotly figures (pure)
    stock_wording the Stock Probability page's copy
    navigation    the destinations' page scripts
    run           Streamlit entry point (launch with scripts/run_portfolio_lab.py)

The Lab's theme is the script-level `.streamlit/config.toml` next to run.py
(generated from `stock_agent.ui.theme`), with the bundled OFL fonts served
from `static/`.

Boundary (tested): the app imports only `stock_agent.ui`,
`stock_agent.portfolio`, the health center's summary reader and view model
(`stock_agent.health.model`, `stock_agent.health.view`, standard library
only), the Stock Lab's store and descriptive modules (never its Tiingo
adapter), its own modules, pandas, numpy, plotly, streamlit and the standard
library. It never imports the research dashboard, models, raw
data, features or the publication writer, and it computes no financial
formula itself: every return, growth, decline and calendar-year figure comes
from the approved descriptive calculations, and the Builder only restates
the user's weights. `stock_agent.portfolio` never imports the app.
"""
