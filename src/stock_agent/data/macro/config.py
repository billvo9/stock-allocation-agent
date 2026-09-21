from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from stock_agent.data.macro.schema import MacroSeriesSpec


@dataclass(frozen=True)
class MacroSeriesConfig:
    """
    Provider-neutral configuration for one macroeconomic series.

    macro_id is the stable identifier used internally by the project.
    provider_series_id is the identifier used by the external provider.
    """

    macro_id: str
    provider: str
    provider_series_id: str
    country: str
    category: str
    expected_frequency: str
    units: str

    def to_spec(self) -> MacroSeriesSpec:
        """
        Convert configuration into the canonical retrieval specification.
        """

        return MacroSeriesSpec(
            series_id=self.macro_id,
            provider_series_id=self.provider_series_id,
            frequency=self.expected_frequency,
            units=self.units,
        )


_REQUIRED_FIELDS = {
    "macro_id",
    "provider",
    "provider_series_id",
    "country",
    "category",
    "expected_frequency",
    "units",
}


def _validate_series_entry(
    entry: object,
    index: int,
) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise TypeError(f"Macro series entry {index} must be a mapping.")

    missing = sorted(_REQUIRED_FIELDS - set(entry))

    if missing:
        raise ValueError(f"Macro series entry {index} is missing required fields: {missing}")

    for field in _REQUIRED_FIELDS:
        value = entry[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Macro series entry {index} has invalid value for {field!r}.")

    return entry


def load_macro_config(
    path: str | Path,
) -> list[MacroSeriesConfig]:
    """
    Load configured macroeconomic series.

    The YAML configuration remains provider-neutral so that future
    providers can coexist with FRED without changing downstream code.
    """

    config_path = Path(path)

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = yaml.safe_load(file)

    if not isinstance(payload, dict):
        raise TypeError("Macro configuration must contain a mapping.")

    raw_series = payload.get("series")

    if not isinstance(raw_series, list):
        raise TypeError("Macro configuration must contain a 'series' list.")

    if not raw_series:
        raise ValueError("Macro configuration must contain at least one series.")

    configs: list[MacroSeriesConfig] = []

    seen_macro_ids: set[str] = set()

    for index, raw_entry in enumerate(raw_series):
        entry = _validate_series_entry(
            raw_entry,
            index,
        )

        macro_id = entry["macro_id"].strip()

        if macro_id in seen_macro_ids:
            raise ValueError(f"Duplicate macro_id: {macro_id}")

        seen_macro_ids.add(macro_id)

        configs.append(
            MacroSeriesConfig(
                macro_id=macro_id,
                provider=entry["provider"].strip().upper(),
                provider_series_id=(entry["provider_series_id"].strip()),
                country=entry["country"].strip().upper(),
                category=entry["category"].strip(),
                expected_frequency=(entry["expected_frequency"].strip()),
                units=entry["units"].strip(),
            )
        )

    return configs
