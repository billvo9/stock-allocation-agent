"""
The prospective track's refusal: one exception for freeze, registration,
forecast and score, with a machine-readable `reason` and a plain `detail`.
Expected conditions (a timing rule, a mismatch with the registration) are
refusals, never repaired.
"""

from __future__ import annotations


class ForecastRefused(ValueError):
    """A freeze, registration, forecast or score was refused; `reason` is machine-readable."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


__all__ = ["ForecastRefused"]
