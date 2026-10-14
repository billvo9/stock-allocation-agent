"""
Frozen eligible universes: point-in-time membership, decided before any
feature or label exists, and fixed by a content hash.

Membership of symbol s on session d depends only on s's own price rows dated
on or before d:

    valid     the row's close is finite and positive
    history   valid closes of s up to and including d
    eligible  valid and history >= rule.min_history_sessions

Reasons for ineligible rows: invalid_price, insufficient_history. A symbol
with no row on d (not yet listed, or delisted) is not a member on d.

Membership never looks at features or labels, so it cannot depend on
feature missingness or on outcomes (owner decision 2026-10-13: membership
and labels stay independent of feature missingness). Because it reads only
earlier rows, adding later data never changes an earlier membership (prefix
stability, tested).

"Frozen" means the membership table is hashed together with its rule and id
(`sha256`). An experiment records the hash, and `verify_universe` refuses a
table that no longer matches it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stock_agent.model_diagnostics.record import sha256_json
from stock_agent.model_validation.audit import LeakageError, frame_content_sha256

MEMBERSHIP_COLUMNS = ("date", "symbol", "history_sessions", "eligible", "reason")


@dataclass(frozen=True)
class UniverseRule:
    min_history_sessions: int = 60
    price_column: str = "close"

    def __post_init__(self) -> None:
        if isinstance(self.min_history_sessions, bool) or not isinstance(
            self.min_history_sessions, int
        ):
            raise TypeError("min_history_sessions must be an integer.")
        if self.min_history_sessions < 1:
            raise ValueError("min_history_sessions must be positive.")


@dataclass(frozen=True, eq=False)
class FrozenUniverse:
    universe_id: str
    rule: UniverseRule
    membership: pd.DataFrame
    sha256: str
    point_in_time: bool
    survivorship_free: bool

    def describe(self) -> dict[str, object]:
        eligible = self.membership.loc[self.membership["eligible"]]
        return {
            "universe_id": self.universe_id,
            "rule": asdict(self.rule),
            "sha256": self.sha256,
            "point_in_time": self.point_in_time,
            "survivorship_free": self.survivorship_free,
            "n_symbols": int(self.membership["symbol"].nunique()),
            "n_eligible_rows": len(eligible),
            "first_date": eligible["date"].min().isoformat() if len(eligible) else None,
            "last_date": eligible["date"].max().isoformat() if len(eligible) else None,
        }


def membership_table(prices: pd.DataFrame, rule: UniverseRule) -> pd.DataFrame:
    """Point-in-time membership of every (date, symbol) price row."""

    missing = [c for c in ("date", "symbol", rule.price_column) if c not in prices.columns]
    if missing:
        raise ValueError(f"Prices are missing columns: {missing}")
    frame = prices.loc[:, ["date", "symbol", rule.price_column]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.as_unit("ns")
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError("Prices contain duplicate (date, symbol) rows.")
    if (frame["date"] != frame["date"].dt.normalize()).any():
        raise ValueError("Price dates must be session dates at 00:00 UTC.")
    frame = frame.sort_values(["symbol", "date"], kind="mergesort")
    close = pd.to_numeric(frame[rule.price_column], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(close) & (close > 0)
    frame["history_sessions"] = (
        pd.Series(valid.astype(np.int64), index=frame.index).groupby(frame["symbol"]).cumsum()
    )
    enough = frame["history_sessions"].to_numpy() >= rule.min_history_sessions
    frame["eligible"] = valid & enough
    frame["reason"] = np.where(
        ~valid, "invalid_price", np.where(~enough, "insufficient_history", None)
    )
    frame["reason"] = frame["reason"].astype("string")
    result = frame.loc[:, list(MEMBERSHIP_COLUMNS)].sort_values(["date", "symbol"])
    return result.reset_index(drop=True)


def _hash(universe_id: str, rule: UniverseRule, membership: pd.DataFrame) -> str:
    content = frame_content_sha256(membership, ["history_sessions", "eligible", "reason"])
    return sha256_json({"universe_id": universe_id, "rule": asdict(rule), "content": content})


def freeze_universe(
    prices: pd.DataFrame,
    rule: UniverseRule,
    *,
    universe_id: str,
    survivorship_free: bool,
) -> FrozenUniverse:
    """
    Membership plus its hash. `survivorship_free` is a property of the price
    source (every symbol that ever existed is present), declared by the
    caller; it is recorded, not inferred.
    """

    membership = membership_table(prices, rule)
    return FrozenUniverse(
        universe_id=universe_id,
        rule=rule,
        membership=membership,
        sha256=_hash(universe_id, rule, membership),
        point_in_time=True,
        survivorship_free=bool(survivorship_free),
    )


def verify_universe(universe: FrozenUniverse) -> None:
    """Refuse a membership table that no longer matches its frozen hash."""

    if _hash(universe.universe_id, universe.rule, universe.membership) != universe.sha256:
        raise LeakageError(
            f"Universe {universe.universe_id!r} membership no longer matches its frozen hash."
        )


def eligible_keys(universe: FrozenUniverse) -> pd.DataFrame:
    """The eligible (date, symbol) keys, sorted, with a default index."""

    eligible = universe.membership.loc[universe.membership["eligible"], ["date", "symbol"]]
    return eligible.sort_values(["date", "symbol"]).reset_index(drop=True)


def assert_membership_independent(frame: pd.DataFrame, universe: FrozenUniverse) -> None:
    """
    The model frame holds exactly the eligible keys: no row was dropped or
    added because of a feature or a label.
    """

    expected = eligible_keys(universe)
    keys = frame.loc[:, ["date", "symbol"]].copy()
    keys["date"] = pd.to_datetime(keys["date"], utc=True).dt.as_unit("ns")
    keys = keys.sort_values(["date", "symbol"]).reset_index(drop=True)
    if len(keys) != len(expected) or not keys.equals(expected.astype(keys.dtypes.to_dict())):
        raise LeakageError(
            "The model frame's rows differ from the frozen universe's eligible keys: membership "
            "must not depend on features or labels."
        )
