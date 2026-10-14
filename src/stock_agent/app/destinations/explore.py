"""Explore: the fund-history page on synthetic demo data and the approved calculations."""

from stock_agent.app import explore, synthetic
from stock_agent.portfolio.descriptive import DescriptiveCalculator

explore.render(synthetic.demo_histories, DescriptiveCalculator())
