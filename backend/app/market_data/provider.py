"""Historical price provider abstraction.

Search, metadata and prices are separate concerns (see app.assets for search). A licensed
provider can be added by implementing PriceProvider; nothing else needs to change.
"""

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

# Adjusted close is split- and dividend-adjusted, so its percentage changes are total
# returns. Dividends are therefore already inside the price; never add them again.
PRICE_FIELD = "Adj Close"


class ProviderError(RuntimeError):
    """The provider failed (network, rate limit, outage). Retrying later may work."""


class NoDataError(LookupError):
    """The provider answered but has no usable price history for this symbol."""


class PriceProvider(ABC):
    name: str
    price_field: str = PRICE_FIELD
    attribution: str = ""

    @abstractmethod
    def fetch_prices(self, symbol: str, start: date) -> pd.Series:
        """Return adjusted closes indexed by tz-naive trading date, sorted, no duplicates.

        Raises NoDataError for unknown/empty symbols and ProviderError for provider failures.
        Must not fill or interpolate missing values.
        """


class YFinanceProvider(PriceProvider):
    name = "yfinance"
    attribution = (
        "Market data via the yfinance library (Yahoo Finance). Unofficial, for research and "
        "education only; data may be delayed, incomplete, adjusted or revised."
    )

    def fetch_prices(self, symbol: str, start: date) -> pd.Series:
        import yfinance as yf

        try:
            frame = yf.download(
                symbol,
                start=start.isoformat(),
                auto_adjust=False,
                progress=False,
                multi_level_index=False,
                threads=False,
            )
        except Exception as exc:  # provider internals vary; surface, never swallow
            raise ProviderError(f"{self.name} request failed for {symbol}: {exc}") from exc
        if frame is None or frame.empty or self.price_field not in frame.columns:
            raise NoDataError(f"{self.name} returned no price history for {symbol}.")
        return clean_series(frame[self.price_field], symbol)


def clean_series(raw: pd.Series, symbol: str) -> pd.Series:
    """Validate a provider series. Drops NaN rows (never fills them); rejects bad prices."""
    series = pd.to_numeric(raw, errors="coerce")
    series.index = pd.DatetimeIndex(pd.to_datetime(series.index)).tz_localize(None).normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index().dropna()
    if series.empty:
        raise NoDataError(f"No valid prices for {symbol}.")
    if (series <= 0).any():
        raise NoDataError(f"{symbol} contains non-positive prices; refusing to use it.")
    series.name = symbol
    return series.astype(float)
