"""
Vendor adapters for the F1 fund-data store.

Each module implements `stock_agent.data.funds.adapter.FundDataAdapter` for one
provider. Network code lives here, never in `data/funds`, whose boundary tests
ban network imports. Tests inject fake transports; no test contacts a vendor.

Modules:
    tiingo              Tiingo end-of-day prices for the research store (header
                        authentication, stdlib transport)
    tiingo_prospective  Tiingo prospective mode for the prospective store
                        (`stock_agent.data.prospective`), sessions from its window only
    anchors             the owner's private, hand-filled distribution anchors (no network)
"""
