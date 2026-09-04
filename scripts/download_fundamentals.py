from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from stock_agent.data.fundamentals.schema import (
    FUNDAMENTAL_SNAPSHOT_COLUMNS,
    METADATA_COLUMNS,
)
from stock_agent.data.fundamentals.storage import (
    merge_metadata,
    merge_snapshot_history,
)
from stock_agent.data.fundamentals.yfinance_source import (
    YFinanceFundamentalDataSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSET_CONFIG_PATH = PROJECT_ROOT / "config" / "assets.yaml"

METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "instrument_metadata.parquet"

SNAPSHOT_PATH = PROJECT_ROOT / "data" / "raw" / "fundamental_snapshots.parquet"


def load_asset_pairs(
    config_path: Path,
) -> list[tuple[str, str]]:
    """
    Load canonical and Yahoo symbols for assets.

    Benchmarks are intentionally excluded because
    company fundamentals do not have the same
    interpretation for market indices.
    """

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file) or {}

    assets = config.get(
        "assets",
        [],
    )

    if not assets:
        raise ValueError("No assets found in configuration.")

    pairs: list[tuple[str, str]] = []
    seen_symbols: set[str] = set()

    for asset in assets:
        if not isinstance(
            asset,
            dict,
        ):
            raise TypeError("Each configured asset must be a mapping.")

        symbol = str(
            asset.get(
                "symbol",
                "",
            )
        ).strip()

        if not symbol:
            raise ValueError("Configured asset is missing a symbol.")

        # Reuse the same provider-symbol convention
        # as the existing market-price pipeline.
        provider_symbol = str(
            asset.get(
                "vendor_symbol",
                symbol,
            )
        ).strip()

        if not provider_symbol:
            raise ValueError(f"{symbol}: provider symbol cannot be empty.")

        if symbol in seen_symbols:
            raise ValueError(f"Duplicate asset symbol found in configuration: {symbol}")

        seen_symbols.add(symbol)

        pairs.append(
            (
                symbol,
                provider_symbol,
            )
        )

    return pairs


def load_existing(
    path: Path,
    columns: list[str],
) -> pd.DataFrame:
    """Load an existing parquet file or an empty frame."""

    if not path.exists():
        return pd.DataFrame(columns=columns)

    return pd.read_parquet(path)


def write_parquet_atomic(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """
    Write parquet through a temporary file.

    This reduces the chance of leaving a corrupted
    destination file if writing fails midway.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(".tmp.parquet")

    frame.to_parquet(
        temporary_path,
        index=False,
    )

    temporary_path.replace(path)


def main() -> None:
    asset_pairs = load_asset_pairs(ASSET_CONFIG_PATH)

    source = YFinanceFundamentalDataSource()

    metadata_frames: list[pd.DataFrame] = []

    snapshot_frames: list[pd.DataFrame] = []

    failures: list[tuple[str, str]] = []

    for (
        symbol,
        provider_symbol,
    ) in asset_pairs:
        print(f"Downloading fundamentals for {symbol} ({provider_symbol})...")

        try:
            bundle = source.get_company_data(
                symbol=symbol,
                provider_symbol=(provider_symbol),
            )

        except (
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            failures.append(
                (
                    symbol,
                    str(exc),
                )
            )

            print(f"WARNING: {symbol} failed: {exc}")

            continue

        metadata_frames.append(bundle.metadata)

        snapshot_frames.append(bundle.snapshot)

    if not metadata_frames:
        raise RuntimeError("No fundamental data were successfully downloaded.")

    incoming_metadata = pd.concat(
        metadata_frames,
        ignore_index=True,
    )

    incoming_snapshots = pd.concat(
        snapshot_frames,
        ignore_index=True,
    )

    existing_metadata = load_existing(
        METADATA_PATH,
        METADATA_COLUMNS,
    )

    existing_snapshots = load_existing(
        SNAPSHOT_PATH,
        FUNDAMENTAL_SNAPSHOT_COLUMNS,
    )

    merged_metadata = merge_metadata(
        existing=existing_metadata,
        incoming=incoming_metadata,
    )

    merged_snapshots = merge_snapshot_history(
        existing=existing_snapshots,
        incoming=incoming_snapshots,
    )

    write_parquet_atomic(
        merged_metadata,
        METADATA_PATH,
    )

    write_parquet_atomic(
        merged_snapshots,
        SNAPSHOT_PATH,
    )

    print()
    print(f"Saved {len(merged_metadata):,} metadata rows")
    print(f"Output: {METADATA_PATH}")

    print()
    print(f"Saved {len(merged_snapshots):,} fundamental snapshot rows")
    print(f"Output: {SNAPSHOT_PATH}")

    print()
    print("Latest snapshots:")
    print(
        merged_snapshots.sort_values("retrieved_at").tail(len(asset_pairs))[
            [
                "symbol",
                "retrieved_at",
                "market_cap",
                "forward_pe",
                "beta",
            ]
        ]
    )

    if failures:
        print()
        print("Fundamental download completed with failures:")

        for (
            symbol,
            message,
        ) in failures:
            print(f"- {symbol}: {message}")


if __name__ == "__main__":
    main()
