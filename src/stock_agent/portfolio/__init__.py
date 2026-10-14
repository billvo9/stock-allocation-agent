"""
Portfolio analytics shared by the product app and later simulation work.

Modules:
    descriptive  result types and constants for descriptive fund metrics
                 (growth of $10,000, worst decline, calendar-year returns).
                 Types only: the formulas are OD8 group A in
                 docs/research/t5_portfolio_formulas.md, pending the owner's
                 approval, and arrive with ticket T5-A.

Dependency direction (tested): this package never imports the product app
(`stock_agent.app`), the design system (`stock_agent.ui`) or the research
dashboard, so a formula implementation here can be used by any front end.
"""
