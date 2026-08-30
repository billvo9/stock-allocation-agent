from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from stock_agent.data.rates.schema import (
    RateSeriesSpec,
)


@dataclass(frozen=True)
class RiskFreeRateConfig:
    provider: str
    spec: RateSeriesSpec
    refresh_overlap_days: int


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


def load_risk_free_rate_config(
    path: Path,
) -> RiskFreeRateConfig:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError("Rate configuration must be a mapping.")

    risk_free = config.get("risk_free")

    if not isinstance(risk_free, dict):
        raise TypeError("Rate configuration must contain a 'risk_free' mapping.")

    required_fields = {
        "rate_id",
        "provider",
        "provider_series_id",
        "provider_release_id",
        "currency",
        "tenor",
        "quote_convention",
    }

    missing = required_fields - set(risk_free)

    if missing:
        raise ValueError(f"Risk-free rate configuration is missing: {sorted(missing)}")

    if str(risk_free["provider"]).upper() != "FRED":
        raise ValueError("Only the FRED rate provider is currently supported.")

    refresh_overlap_days = int(
        risk_free.get(
            "refresh_overlap_days",
            7,
        )
    )

    if refresh_overlap_days < 0:
        raise ValueError("Refresh overlap days cannot be negative.")

    return RiskFreeRateConfig(
        provider=str(risk_free["provider"]).upper(),
        spec=RateSeriesSpec(
            rate_id=str(risk_free["rate_id"]),
            provider_series_id=str(risk_free["provider_series_id"]),
            provider_release_id=int(risk_free["provider_release_id"]),
            currency=str(risk_free["currency"]),
            tenor=str(risk_free["tenor"]),
            quote_convention=str(risk_free["quote_convention"]),
        ),
        refresh_overlap_days=refresh_overlap_days,
    )
