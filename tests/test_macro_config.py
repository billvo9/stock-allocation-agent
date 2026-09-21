from __future__ import annotations

from pathlib import Path

import pytest

from stock_agent.data.macro.config import (
    MacroSeriesConfig,
    load_macro_config,
)


def _write_config(
    tmp_path: Path,
    contents: str,
) -> Path:
    path = tmp_path / "macro.yaml"
    path.write_text(
        contents,
        encoding="utf-8",
    )
    return path


def test_load_macro_config_returns_series(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series:
  - macro_id: us_unemployment_rate
    provider: FRED
    provider_series_id: UNRATE
    country: US
    category: labor
    expected_frequency: monthly
    units: percent
""",
    )

    result = load_macro_config(path)

    assert len(result) == 1

    assert result[0] == MacroSeriesConfig(
        macro_id="us_unemployment_rate",
        provider="FRED",
        provider_series_id="UNRATE",
        country="US",
        category="labor",
        expected_frequency="monthly",
        units="percent",
    )


def test_macro_config_converts_to_canonical_spec(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series:
  - macro_id: us_unemployment_rate
    provider: FRED
    provider_series_id: UNRATE
    country: US
    category: labor
    expected_frequency: monthly
    units: percent
""",
    )

    config = load_macro_config(path)[0]

    spec = config.to_spec()

    assert spec.series_id == "us_unemployment_rate"
    assert spec.provider_series_id == "UNRATE"
    assert spec.frequency == "monthly"
    assert spec.units == "percent"


def test_macro_config_normalizes_provider_and_country(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series:
  - macro_id: us_unemployment_rate
    provider: fred
    provider_series_id: UNRATE
    country: us
    category: labor
    expected_frequency: monthly
    units: percent
""",
    )

    result = load_macro_config(path)

    assert result[0].provider == "FRED"
    assert result[0].country == "US"


def test_duplicate_macro_id_raises(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series:
  - macro_id: us_unemployment_rate
    provider: FRED
    provider_series_id: UNRATE
    country: US
    category: labor
    expected_frequency: monthly
    units: percent

  - macro_id: us_unemployment_rate
    provider: OTHER
    provider_series_id: ABC
    country: US
    category: labor
    expected_frequency: monthly
    units: percent
""",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate macro_id",
    ):
        load_macro_config(path)


def test_missing_required_field_raises(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series:
  - macro_id: us_unemployment_rate
    provider: FRED
    provider_series_id: UNRATE
    country: US
    category: labor
    expected_frequency: monthly
""",
    )

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
        load_macro_config(path)


def test_series_must_be_list(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series:
  macro_id: us_unemployment_rate
""",
    )

    with pytest.raises(
        TypeError,
        match="'series' list",
    ):
        load_macro_config(path)


def test_empty_series_raises(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series: []
""",
    )

    with pytest.raises(
        ValueError,
        match="at least one series",
    ):
        load_macro_config(path)


def test_top_level_config_must_be_mapping(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
- not
- a
- mapping
""",
    )

    with pytest.raises(
        TypeError,
        match="must contain a mapping",
    ):
        load_macro_config(path)


def test_series_entry_must_be_mapping(
    tmp_path: Path,
):
    path = _write_config(
        tmp_path,
        """
series:
  - not-a-mapping
""",
    )

    with pytest.raises(
        TypeError,
        match="must be a mapping",
    ):
        load_macro_config(path)
