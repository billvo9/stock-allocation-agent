"""Experiments: the recorded prospective forecasts and the registered synthetic comparison."""

from stock_agent.app import comparison, experiments, forecast_records

experiments.render(forecast_records.read_records, comparison.read_comparisons)
