from __future__ import annotations

from pathlib import Path

import yaml


def load_asset_symbols(
    config_path: Path,
) -> list[str]:
    """Load portfolio asset symbols from assets.yaml."""

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    assets = config.get("assets", [])

    symbols = [asset["symbol"] for asset in assets]

    if not symbols:
        raise ValueError("No assets found in config.")

    if len(symbols) != len(set(symbols)):
        raise ValueError("Duplicate asset symbols found in config.")

    if any(not symbol for symbol in symbols):
        raise ValueError("Asset symbols cannot be empty.")

    return symbols
