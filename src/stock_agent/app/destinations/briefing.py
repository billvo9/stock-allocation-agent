"""Premarket Briefing: the latest briefing written on this computer, read-only (experimental)."""

from stock_agent.app import briefing
from stock_agent.stock_lab.briefing.stored import default_directory, delivery_status, list_briefings

briefing.render(list_briefings, delivery_status, default_directory, briefing.new_york_today)
