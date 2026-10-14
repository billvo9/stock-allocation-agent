"""
Portfolio analytics shared by the product app and later simulation work.

Modules:
    descriptive  the approved OD8 group A calculations for descriptive fund
                 histories (growth of $10,000, worst decline, calendar-year
                 returns); docs/research/t5_portfolio_formulas.md, section 9.
    transition   the T5a per-step contract (`portfolio_step`, `StepResult`,
                 `CostModel`, `Constraints`, `feasible_target`, the
                 `Transition` seam) shared by backtests and the RL
                 environment.
    schedule     the T5a decision and execution schedule (lag, offsets,
                 deferral).
    engine       the T5a caller: targets and returns to a wealth path.

transition, schedule and engine are interfaces only: OD8 group B
(docs/research/t5a_open_decisions.md) awaits the owner's approval, and every
economic function raises NotImplementedError until then.

Dependency direction (tested): this package never imports the product app
(`stock_agent.app`), the design system (`stock_agent.ui`), the research
dashboard or the RL package, so a formula implementation here can be used by
any front end and by every simulator.
"""
