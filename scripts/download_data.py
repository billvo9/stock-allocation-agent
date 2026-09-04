from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from stock_agent.data.yfinance_source import YFinanceDataSource
from stock_agent.exceptions import StockAgentError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "prices.parquet"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def download_all_prices(
    start: str = "2024-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    config = load_config()
    source = YFinanceDataSource()

    frames: list[pd.DataFrame] = []

    if end is None:
        end = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    instruments = config["assets"] + config["benchmarks"]

    for item in instruments:
        symbol = item["symbol"]
        vendor_symbol = item["vendor_symbol"]
        currency = item.get("currency", "USD")

        print(f"Downloading {symbol} ({vendor_symbol})...")

        frame = source.get_prices(
            symbol=symbol,
            vendor_symbol=vendor_symbol,
            start=start,
            end=end,
            currency=currency,
        )

        frames.append(frame)

    if not frames:
        raise RuntimeError("No market data were downloaded.")

    combined = pd.concat(frames, ignore_index=True)

    return combined.sort_values(["symbol", "date"]).reset_index(drop=True)


def main() -> None:
    try:
        prices = download_all_prices()

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        prices.to_parquet(
            OUTPUT_PATH,
            index=False,
        )

        print()
        print(f"Saved {len(prices):,} rows to:")
        print(OUTPUT_PATH)

        print()
        print("Rows by symbol:")
        print(prices.groupby("symbol").size())

        print()
        print("Date range:")
        print(prices["date"].min(), "to", prices["date"].max())

    except StockAgentError as exc:
        print()
        print("MARKET DATA ERROR")
        print(exc)
        raise SystemExit(1) from exc

    except Exception as exc:
        print()
        print("UNEXPECTED ERROR")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
