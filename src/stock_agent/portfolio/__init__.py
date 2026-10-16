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
    backtest_summary
                 the `backtest_summary` health input (agreed format v1.0)
                 for one engine simulation.

transition, schedule and engine implement OD8 group B (T5a D1-D8,
docs/research/t5a_open_decisions.md), approved by the owner on 2026-10-15;
docs/research/t5a_engine.md describes the implementation.

Dependency direction (tested): this package never imports the product app
(`stock_agent.app`), the design system (`stock_agent.ui`), the research
dashboard or the RL package, so a formula implementation here can be used by
any front end and by every simulator.
"""
