"""Portfolio Builder: weights and concentration; the mix history awaits approval."""

from stock_agent.app import builder, navigation, synthetic

builder.render(synthetic.demo_histories, links=navigation.links("explore"))
