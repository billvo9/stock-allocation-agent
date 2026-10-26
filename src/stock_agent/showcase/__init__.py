"""
The exploratory real-data rebalancing showcase: private, owner-only.

The owner approved it on 2026-10-26 ("I want more stocks and industries than
only AMD, MRVL, MU and NVIDIA. and Yes just don't add new formula except we
really need one."; audience "Only me"). It runs the registered baseline
rules (buy-and-hold, equal weight, inverse volatility) on real 2003-2024
prices through the approved T5a engine, and reports only the approved OD8
group A descriptive measures with D5 turnover and cost reporting.

Modules:
    config    the registered configuration (`configs/showcase/`), validated
              and hashed before any run
    store     the separate private showcase store and its isolation from
              every Stock Lab store
    returns   bars to the approved F5.1 total-return table the engine reads
    run       every registered variant through the engine, and its measures
    report    the private HTML and Markdown report

Nothing here is evidence of strategy skill or advice: the universes are
hindsight-selected and survivorship-biased (the report says so on every page).
The product app, the Stock Lab and the evening job never import this package.
"""
