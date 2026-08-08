class StockAgentError(Exception):
    """Base exception for the project."""


class MarketDataError(StockAgentError):
    """Base exception for market-data failures."""


class MarketDataDownloadError(MarketDataError):
    """Raised when market data cannot be downloaded."""


class MarketDataValidationError(MarketDataError):
    """Raised when downloaded market data are invalid."""


class InsufficientHistoryError(MarketDataError):
    """Raised when there is not enough history for analysis."""