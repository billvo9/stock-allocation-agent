"""The showcase's registered configuration: its rules cannot be weakened by editing the YAML."""

from __future__ import annotations

import copy

import pytest
from showcase_fixtures import REGISTERED, config, record

from stock_agent.showcase.config import (
    MAX_TICKERS,
    ConfigRefused,
    config_from_dict,
    load_config,
)


def test_the_registered_configuration_loads_with_three_universes_and_six_rules():
    registered = load_config(REGISTERED)
    assert [u.key for u in registered.universes] == [
        "sector_etfs",
        "sector_stocks",
        "semiconductors",
    ]
    assert [s.key for s in registered.strategies] == [
        "buy_and_hold",
        "equal_weight_weekly",
        "equal_weight_monthly",
        "equal_weight_quarterly",
        "inverse_volatility_monthly",
        "inverse_volatility_quarterly",
    ]
    assert [s.every for s in registered.strategies[1:]] == [5, 21, 63, 21, 63]
    assert len(registered.variants()) == 18
    assert len(registered.tickers()) == 31 <= MAX_TICKERS
    assert registered.universe("semiconductors").tickers == ("AMD", "MRVL", "MU", "NVDA")
    assert set(registered.universe("sector_etfs").tickers) == {
        "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"
    }  # fmt: skip
    assert registered.asset_classes()["XLK"] == "broad_etf"
    assert registered.asset_classes()["NVDA"] == "single_stock"
    assert str(registered.end) == "2024-12-31"


def test_the_semiconductors_are_kept_out_of_the_rule_selected_stock_list():
    registered = load_config(REGISTERED)
    stocks = set(registered.universe("sector_stocks").tickers)
    assert stocks.isdisjoint({"AMD", "MRVL", "MU", "NVDA"})
    assert len(stocks) == 18


def test_the_hash_identifies_the_configuration():
    assert config().sha256() == config().sha256()
    assert config().sha256() != config(initial_wealth=20000).sha256()


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"window": {"start": "2024-01-02", "end": "2025-01-02"}}, "sealed holdout"),
        ({"window": {"start": "2002-12-31", "end": "2024-12-31"}}, "starts before"),
        ({"cost_multipliers": [0.0, 1.0, 3.0]}, "never chosen"),
        ({"execution_lag": 0}, "entry lag"),
        ({"constraints": {"max_weight": 0.5, "max_turnover": None}}, "fully invested"),
        ({"constraints": {"max_weight": 1.0, "max_turnover": 0.25}}, "fully invested"),
        ({"first_decision": 10}, "inverse-volatility window"),
        ({"offset": 5}, "offset"),
        ({"cost_classes": {"etf": "tbill_fund", "stock": "single_stock"}}, "broad-ETF"),
        ({"protocol": "something_else"}, "protocol"),
    ],
)
def test_a_configuration_that_weakens_a_registered_rule_is_refused(changes, message):
    with pytest.raises(ConfigRefused, match=message):
        config(**changes)


def test_unknown_or_missing_keys_and_unknown_rules_are_refused():
    with pytest.raises(ConfigRefused, match="unknown keys"):
        config_from_dict({**record(), "best_of": 3})
    incomplete = record()
    del incomplete["offset"]
    with pytest.raises(ConfigRefused, match="missing keys"):
        config_from_dict(incomplete)
    strategies = copy.deepcopy(record()["strategies"])
    strategies[0]["rule"] = "mean_variance"  # not approved on real data
    with pytest.raises(ConfigRefused, match="unknown rule"):
        config(strategies=strategies)


def test_tickers_are_checked_and_capped():
    universes = copy.deepcopy(record()["universes"])
    universes[1]["members"][0]["kind"] = "etf"
    universes[1]["members"][0]["ticker"] = "XAA"  # XAA is already an etf: same kind is fine
    config(universes=universes)
    universes[1]["members"][0]["kind"] = "stock"
    with pytest.raises(ConfigRefused, match="two kinds"):
        config(universes=universes)
    universes = copy.deepcopy(record()["universes"])
    universes[0]["members"][0]["ticker"] = "VOD.L"
    with pytest.raises(ConfigRefused, match="provider's form"):
        config(universes=universes)
    many = [
        {"ticker": f"A{chr(65 + i // 26)}{chr(65 + i % 26)}", "kind": "stock", "sector": "S"}
        for i in range(MAX_TICKERS + 1)
    ]
    with pytest.raises(ConfigRefused, match="at most"):
        config(universes=[{"key": "big", "label": "Big", "selection": "x", "members": many}])


def test_version_2_is_version_1_plus_the_registered_offset_family():
    import yaml

    v1 = yaml.safe_load(REGISTERED.read_text(encoding="utf-8"))
    v2 = yaml.safe_load(
        (REGISTERED.parent / "real_rebalancing_v2.yaml").read_text(encoding="utf-8")
    )
    assert v2.pop("offset_family") is True and v2.pop("name") == "real_rebalancing_v2"
    v1.pop("name")
    assert v1 == v2  # same universes, rules, schedules, offset, window and cost levels
    assert load_config(REGISTERED).offset_family is False
    assert load_config(REGISTERED.parent / "real_rebalancing_v2.yaml").offset_family is True


def test_offset_family_must_be_a_boolean():
    with pytest.raises(ConfigRefused, match="offset_family"):
        config(offset_family="all")
