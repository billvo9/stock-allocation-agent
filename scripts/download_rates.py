from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_agent.config import (
    load_risk_free_rate_config,
)
from stock_agent.data.rates.fred_source import (
    FredRateDataSource,
)
from stock_agent.data.rates.schema import (
    validate_rate_frame,
)
from stock_agent.data.rates.storage import (
    merge_rate_history,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "rates.yaml"

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "rates.parquet"


def main() -> None:
    rate_config = load_risk_free_rate_config(CONFIG_PATH)

    spec = rate_config.spec

    source = FredRateDataSource()

    if OUTPUT_PATH.exists():
        existing = pd.read_parquet(OUTPUT_PATH)

        start = (
            pd.to_datetime(existing["date"]).max()
            - pd.Timedelta(days=(rate_config.refresh_overlap_days))
        ).strftime("%Y-%m-%d")
    else:
        existing = pd.DataFrame()

        start = "2024-01-01"

    end = pd.Timestamp.today().strftime("%Y-%m-%d")

    incoming = source.get_rates(
        spec=spec,
        start=start,
        end=end,
    )

    if existing.empty:
        combined = incoming
    else:
        combined = merge_rate_history(
            existing=existing,
            incoming=incoming,
        )

    validate_rate_frame(combined)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(f"Saved {len(combined):,} rate rows")
    print(f"Output: {OUTPUT_PATH}")
    print()

    print("Rate date range:")
    print(
        combined["date"].min(),
        "to",
        combined["date"].max(),
    )
    print()

    print("Latest observations:")
    print(combined.tail())


if __name__ == "__main__":
    main()
