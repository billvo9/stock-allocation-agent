"""
Vendor adapters for the F1 fund-data store.

Each module implements `stock_agent.data.funds.adapter.FundDataAdapter` for one
provider. Network code lives here, never in `data/funds`, whose boundary tests
ban network imports. Tests inject fake transports; no test contacts a vendor.

Modules:
    tiingo   Tiingo end-of-day prices (header authentication, stdlib transport)
"""
