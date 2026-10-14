"""Overview: what is happening and what needs attention."""

from stock_agent.app import navigation, overview, synthetic, wording

overview.render(synthetic.demo_histories, links=navigation.links(*wording.NEXT_LINKS))
