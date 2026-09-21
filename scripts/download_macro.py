from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_agent.data.macro.config import load_macro_config
from stock_agent.data.macro.ingestion import (
    build_macro_summary,
    download_macro_data,
    save_macro_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "macro.yaml"

OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "macro_vintages.parquet"

DEFAULT_START = "2015-01-01"


def current_utc_date() -> str:
    return pd.Timestamp.now(tz="UTC").date().isoformat()


def main() -> None:
    configs = load_macro_config(CONFIG_PATH)

    frame = download_macro_data(
        configs=configs,
        start=DEFAULT_START,
        end=current_utc_date(),
    )

    output = save_macro_data(
        frame,
        OUTPUT_PATH,
    )

    print()
    print(f"Saved {len(frame):,} macro vintage rows")
    print(f"Output: {output}")

    print()
    print("Macro history:")
    print(build_macro_summary(frame))


if __name__ == "__main__":
    main()
