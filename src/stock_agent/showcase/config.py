"""
The showcase's registered configuration (`configs/showcase/*.yaml`).

A configuration is written down before any data is fetched or any result
exists. Loading it validates the registered rules, so a configuration cannot
weaken them:

- the window lies inside the development period, 2003-01-02 to 2024-12-31;
  2025 on is the sealed holdout and is refused;
- `cost_multipliers` are D2's 0x, 1x and 2x, never chosen;
- `execution_lag` is the label's entry lag (D1, `schedule.execution_lag`);
- every strategy is one of the registered rules, with a cadence of at least
  one session, and the inverse-volatility window fits before the build;
- the constraints are the D5 defaults: long only and fully invested, with no
  cash leg, because a cash return (group C) is not approved;
- tickers are US tickers in the provider's form, each with one kind (`etf` or
  `stock`), and there are at most `MAX_TICKERS` of them, one fetch run's scope.

`offset_family` (optional, default false) registers F7.2's rebalance-offset
sensitivity: every rebalancing rule is also run at all k offsets of its
cadence. The k runs are one variant with a k-member sensitivity family, never
k variants, and the registered `offset` alone gives the headline.

`measures` (optional) registers the group D settings approved on 2026-10-26:
M5's three stress windows and M7's participation notional, threshold and
volume window. A window may lie outside the run's span: it is then reported
`outside_window`, never shortened.

`variants()` lists every (universe, strategy) pair in registered order: the
number of variants run, recorded with the results. Every variant is
reported; none is chosen afterwards. `sha256()` identifies the configuration
in the results.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from stock_agent.calendars import nyse
from stock_agent.portfolio.schedule import execution_lag
from stock_agent.portfolio.transition import (
    BROAD_ETF,
    PROVISIONAL_RATES,
    SENSITIVITY_MULTIPLIERS,
    SINGLE_STOCK,
)

SCHEMA_VERSION = 1
PROTOCOL = "exploratory_real_data_showcase"
DEVELOPMENT_START = date(2003, 1, 2)  # the frozen NYSE calendar's first session
DEVELOPMENT_END = date(2024, 12, 31)  # 2025 on is the sealed holdout
MAX_TICKERS = 31  # one fetch run: 31 price requests and the listing, under 50 an hour
RULES = ("buy_and_hold", "equal_weight", "inverse_volatility")
KINDS = ("etf", "stock")
_NAME = re.compile(r"[a-z][a-z0-9_]*")
_TICKER = re.compile(r"[A-Z]{1,5}(-[A-Z]{1,2})?")
TOP_LEVEL = {
    "schema_version",
    "protocol",
    "name",
    "window",
    "initial_wealth",
    "first_decision",
    "execution_lag",
    "offset",
    "inverse_volatility_window",
    "cost_classes",
    "cost_multipliers",
    "constraints",
    "strategies",
    "universes",
}
OPTIONAL = {"offset_family", "measures"}


class ConfigRefused(ValueError):
    """A configuration that breaks a registered rule of the showcase."""


@dataclass(frozen=True)
class Member:
    ticker: str
    kind: str
    sector: str


@dataclass(frozen=True)
class Universe:
    key: str
    label: str
    selection: str
    members: tuple[Member, ...]

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(member.ticker for member in self.members)


@dataclass(frozen=True)
class Strategy:
    key: str
    rule: str
    every: int
    label: str
    description: str


@dataclass(frozen=True)
class Window:
    key: str
    label: str
    start: date
    end: date


@dataclass(frozen=True)
class Measures:
    """
    The registered group D settings (approved 2026-10-26): M5's stress windows
    and M7's participation flag (notional N, threshold p, MDV window).
    """

    stress_windows: tuple[Window, ...]
    notional: float
    threshold: float
    volume_window: int


def _measures(record: object) -> Measures:
    _keys(record, {"stress_windows", "participation"}, "measures")
    windows = []
    for entry in record["stress_windows"]:
        _keys(entry, {"key", "label", "start", "end"}, "a stress window")
        key = str(entry["key"])
        if not _NAME.fullmatch(key):
            raise ConfigRefused(f"stress window key {key!r} must be a lowercase word")
        first = _date(entry["start"], f"{key}.start")
        last = _date(entry["end"], f"{key}.end")
        if first >= last:
            raise ConfigRefused(f"stress window {key} must start before it ends")
        for name, day in (("start", first), ("end", last)):
            try:
                session = nyse.is_session(day)
            except nyse.CalendarRefused:
                session = False  # outside the frozen calendar: never a registered edge
            if not session:
                raise ConfigRefused(f"stress window {key}'s {name} {day} is not an NYSE session")
        windows.append(Window(key, _text(entry["label"], f"{key}.label"), first, last))
    if len({w.key for w in windows}) != len(windows):
        raise ConfigRefused("stress windows need distinct keys")
    participation = record["participation"]
    _keys(participation, {"notional", "threshold", "window"}, "participation")
    notional, threshold = participation["notional"], participation["threshold"]
    for name, value in (("notional", notional), ("threshold", threshold)):
        if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
            raise ConfigRefused(f"participation.{name} must be a positive number")
    if threshold >= 1:
        raise ConfigRefused("participation.threshold is a fraction below 1")
    return Measures(
        stress_windows=tuple(windows),
        notional=float(notional),
        threshold=float(threshold),
        volume_window=_int(participation["window"], "participation.window", 1),
    )


@dataclass(frozen=True)
class ShowcaseConfig:
    name: str
    start: date
    end: date
    initial_wealth: float
    first_decision: int
    execution_lag: int
    offset: int
    inverse_volatility_window: int
    cost_classes: dict[str, str]
    cost_multipliers: tuple[float, ...]
    strategies: tuple[Strategy, ...]
    universes: tuple[Universe, ...]
    record: dict
    offset_family: bool = False
    measures: Measures | None = None

    def tickers(self) -> tuple[str, ...]:
        """Every ticker of every universe, once, in registered order."""

        return tuple(dict.fromkeys(t for u in self.universes for t in u.tickers))

    def kinds(self) -> dict[str, str]:
        return {m.ticker: m.kind for u in self.universes for m in u.members}

    def asset_classes(self) -> dict[str, str]:
        """Each ticker's D2 asset class."""

        return {ticker: self.cost_classes[kind] for ticker, kind in self.kinds().items()}

    def variants(self) -> tuple[tuple[str, str], ...]:
        """Every (universe, strategy) run, in registered order."""

        return tuple((u.key, s.key) for u in self.universes for s in self.strategies)

    def universe(self, key: str) -> Universe:
        for universe in self.universes:
            if universe.key == key:
                return universe
        raise KeyError(key)

    def sha256(self) -> str:
        text = json.dumps(self.record, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _date(value: object, name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ConfigRefused(f"{name} must be a date, not {value!r}") from None


def _int(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigRefused(f"{name} must be an int of at least {minimum}, not {value!r}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigRefused(f"{name} must be non-empty text")
    return " ".join(value.split())


def _keys(record: dict, allowed: set[str], where: str) -> None:
    if not isinstance(record, dict):
        raise ConfigRefused(f"{where} must be a mapping")
    unknown = sorted(set(record) - allowed)
    if unknown:
        raise ConfigRefused(f"unknown keys in {where}: {unknown}")
    missing = sorted(allowed - set(record))
    if missing:
        raise ConfigRefused(f"missing keys in {where}: {missing}")


def config_from_dict(record: dict) -> ShowcaseConfig:
    """The validated configuration (module docstring), or `ConfigRefused`."""

    _keys({k: v for k, v in record.items() if k not in OPTIONAL}, TOP_LEVEL, "the configuration")
    offset_family = record.get("offset_family", False)
    if not isinstance(offset_family, bool):
        raise ConfigRefused("offset_family must be true or false")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ConfigRefused(f"unsupported schema_version {record['schema_version']!r}")
    if record["protocol"] != PROTOCOL:
        raise ConfigRefused(f"only the {PROTOCOL!r} protocol is defined")
    name = str(record["name"])
    if not _NAME.fullmatch(name):
        raise ConfigRefused(f"the name {name!r} must be a lowercase word")

    _keys(record["window"], {"start", "end"}, "window")
    start = _date(record["window"]["start"], "window.start")
    end = _date(record["window"]["end"], "window.end")
    if start < DEVELOPMENT_START:
        raise ConfigRefused(f"the window starts before {DEVELOPMENT_START}")
    if end > DEVELOPMENT_END:
        raise ConfigRefused(
            f"the window reaches past {DEVELOPMENT_END}: 2025 on is the sealed holdout"
        )
    if start >= end:
        raise ConfigRefused("the window must start before it ends")

    wealth = record["initial_wealth"]
    if isinstance(wealth, bool) or not isinstance(wealth, int | float) or not wealth > 0:
        raise ConfigRefused("initial_wealth must be a positive number")
    lag = _int(record["execution_lag"], "execution_lag", 0)
    if lag != execution_lag():
        raise ConfigRefused(f"execution_lag must be the label's entry lag, {execution_lag()}")
    first = _int(record["first_decision"], "first_decision", 0)
    window = _int(record["inverse_volatility_window"], "inverse_volatility_window", 2)
    if first < window:
        raise ConfigRefused(
            "first_decision must leave the inverse-volatility window of returns before the build"
        )

    classes = record["cost_classes"]
    _keys(classes, set(KINDS), "cost_classes")
    if classes["stock"] != SINGLE_STOCK or classes["etf"] not in (BROAD_ETF, SINGLE_STOCK):
        raise ConfigRefused("stocks pay the single-stock rate; ETFs the broad-ETF or stock rate")
    if any(klass not in PROVISIONAL_RATES for klass in classes.values()):
        raise ConfigRefused("cost classes must be D2 provisional classes")
    multipliers = tuple(float(m) for m in record["cost_multipliers"])
    if multipliers != SENSITIVITY_MULTIPLIERS:
        raise ConfigRefused(f"cost multipliers are D2's {SENSITIVITY_MULTIPLIERS}, never chosen")

    _keys(record["constraints"], {"max_weight", "max_turnover"}, "constraints")
    if record["constraints"]["max_weight"] != 1.0 or record["constraints"]["max_turnover"]:
        raise ConfigRefused(
            "the showcase is fully invested: a cap or a turnover limit would leave cash, "
            "whose return is not approved"
        )

    strategies = []
    for entry in record["strategies"]:
        _keys(entry, {"key", "rule", "every", "label", "description"}, "a strategy")
        key = str(entry["key"])
        if not _NAME.fullmatch(key):
            raise ConfigRefused(f"strategy key {key!r} must be a lowercase word")
        if entry["rule"] not in RULES:
            raise ConfigRefused(f"unknown rule {entry['rule']!r}; registered rules: {RULES}")
        strategies.append(
            Strategy(
                key=key,
                rule=entry["rule"],
                every=_int(entry["every"], f"{key}.every", 1),
                label=_text(entry["label"], f"{key}.label"),
                description=_text(entry["description"], f"{key}.description"),
            )
        )
    if not strategies or len({s.key for s in strategies}) != len(strategies):
        raise ConfigRefused("register at least one strategy, each with a distinct key")
    offset = _int(record["offset"], "offset", 0)
    if any(offset >= s.every for s in strategies):
        raise ConfigRefused("the offset must lie in [0, every) for every strategy")

    universes, kinds = [], {}
    for entry in record["universes"]:
        _keys(entry, {"key", "label", "selection", "members"}, "a universe")
        key = str(entry["key"])
        if not _NAME.fullmatch(key):
            raise ConfigRefused(f"universe key {key!r} must be a lowercase word")
        members = []
        for item in entry["members"]:
            _keys(item, {"ticker", "kind", "sector"}, f"a member of {key}")
            ticker = str(item["ticker"])
            if not _TICKER.fullmatch(ticker):
                raise ConfigRefused(f"{ticker!r} is not a US ticker in the provider's form")
            if item["kind"] not in KINDS:
                raise ConfigRefused(f"{ticker}: kind must be one of {KINDS}")
            if kinds.setdefault(ticker, item["kind"]) != item["kind"]:
                raise ConfigRefused(f"{ticker} is registered with two kinds")
            members.append(Member(ticker, item["kind"], _text(item["sector"], f"{ticker}.sector")))
        if len(members) < 2 or len({m.ticker for m in members}) != len(members):
            raise ConfigRefused(f"universe {key} needs at least two distinct members")
        universes.append(
            Universe(
                key=key,
                label=_text(entry["label"], f"{key}.label"),
                selection=_text(entry["selection"], f"{key}.selection"),
                members=tuple(members),
            )
        )
    if not universes or len({u.key for u in universes}) != len(universes):
        raise ConfigRefused("register at least one universe, each with a distinct key")
    if len(kinds) > MAX_TICKERS:
        raise ConfigRefused(f"{len(kinds)} tickers; at most {MAX_TICKERS} (one fetch run)")

    return ShowcaseConfig(
        name=name,
        start=start,
        end=end,
        initial_wealth=float(wealth),
        first_decision=first,
        execution_lag=lag,
        offset=offset,
        inverse_volatility_window=window,
        cost_classes=dict(classes),
        cost_multipliers=multipliers,
        strategies=tuple(strategies),
        universes=tuple(universes),
        record=record,
        offset_family=offset_family,
        measures=None if "measures" not in record else _measures(record["measures"]),
    )


def load_config(path: str | Path) -> ShowcaseConfig:
    record = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ConfigRefused("a showcase configuration must be a YAML mapping")
    return config_from_dict(record)
