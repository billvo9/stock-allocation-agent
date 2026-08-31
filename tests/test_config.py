from __future__ import annotations

from pathlib import Path

import pytest

from stock_agent.config import (
    load_benchmark_symbols,
)


def test_load_benchmark_symbols(
    tmp_path: Path,
):
    config_path = tmp_path / "assets.yaml"

    config_path.write_text(
        """
benchmarks:
  - symbol: SP500
  - symbol: NASDAQ_COMPOSITE
  - symbol: DOW_JONES
""",
        encoding="utf-8",
    )

    result = load_benchmark_symbols(config_path)

    assert result == [
        "SP500",
        "NASDAQ_COMPOSITE",
        "DOW_JONES",
    ]


def test_load_benchmark_symbols_rejects_duplicates(
    tmp_path: Path,
):
    config_path = tmp_path / "assets.yaml"

    config_path.write_text(
        """
benchmarks:
  - symbol: SP500
  - symbol: SP500
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        load_benchmark_symbols(config_path)


def test_load_benchmark_symbols_rejects_empty_config(
    tmp_path: Path,
):
    config_path = tmp_path / "assets.yaml"

    config_path.write_text(
        """
benchmarks: []
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="No benchmarks",
    ):
        load_benchmark_symbols(config_path)
