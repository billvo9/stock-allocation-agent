from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from stock_agent.data.macro.schema import MacroSeriesSpec


class MacroDataSource(ABC):
    """
    Provider-independent interface for macroeconomic data.

    Provider implementations are responsible for translating
    their native responses into the project's canonical
    macro schema.
    """

    @abstractmethod
    def get_observations(
        self,
        spec: MacroSeriesSpec,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """
        Return canonical macroeconomic observations.

        Implementations must ensure that each returned row
        represents information that was available at the
        recorded `available_at` timestamp.

        Historical providers that expose revisions should also
        populate `vintage_date`.
        """

        raise NotImplementedError
