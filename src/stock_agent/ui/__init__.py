"""
Neutral design system shared by the research dashboard and the product app.

Modules:
    theme       design tokens: palette, status treatments, type, space, CSS
                (the research variant and the Portfolio Lab's editorial one)
    style       chart visual language: identity palettes, markers, Plotly layout
    components  Streamlit components that lay out values and compute nothing

This package depends only on the tokens it defines, pandas, Plotly's layout
conventions and Streamlit. It never imports `stock_agent.dashboard` (or any
data, model or publication code), so either front end can use it without the
other.
"""
