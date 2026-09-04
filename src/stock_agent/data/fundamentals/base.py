from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FundamentalDataBundle:
    """Canonical fundamental data returned for one instrument."""

    metadata: pd.DataFrame
    snapshot: pd.DataFrame


class FundamentalDataSource(ABC):
    """Interface implemented by fundamental-data providers."""

    @abstractmethod
    def get_company_data(
        self,
        symbol: str,
        provider_symbol: str,
    ) -> FundamentalDataBundle:
        """Return metadata and a point-in-time snapshot."""

        raise NotImplementedError
