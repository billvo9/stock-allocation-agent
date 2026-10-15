"""Experiments: the recorded prospective forecasts, else an honest empty state."""

from stock_agent.app import experiments, forecast_records

experiments.render(forecast_records.read_records)
