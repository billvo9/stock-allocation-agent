"""Model Health: the local monitor's latest summary and the daily GitHub check, read-only."""

from stock_agent.app import health_summary, model_health

model_health.render(health_summary.load)
