from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    returns: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    frequency: str


def _normalize_download(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract a clean adjusted/close price matrix from yfinance output."""
    if raw.empty:
        raise ValueError("Yahoo Finance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        first = raw.columns.get_level_values(0)
        if "Close" in first:
            prices = raw["Close"].copy()
        elif "Adj Close" in first:
            prices = raw["Adj Close"].copy()
        else:
            raise ValueError("Could not find Close/Adj Close in Yahoo Finance output.")
    else:
        # Single ticker download.
        col = "Close" if "Close" in raw.columns else "Adj Close"
        prices = raw[[col]].copy()
        prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])

    # Reorder and retain only requested symbols returned by Yahoo.
    present = [t for t in tickers if t in prices.columns]
    if not present:
        # Some yfinance versions can return a one-column frame with a generic label.
        if len(tickers) == 1 and prices.shape[1] == 1:
            prices.columns = tickers
            present = tickers
        else:
            raise ValueError(
                "None of the requested tickers were returned by Yahoo Finance."
            )

    prices = prices[present].sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.astype(float)


def download_prices(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    tickers = list(
        dict.fromkeys(str(t).upper().strip() for t in tickers if str(t).strip())
    )
    if not tickers:
        raise ValueError("At least one ticker is required.")

    raw = yf.download(
        tickers=tickers,
        start=pd.Timestamp(start).strftime("%Y-%m-%d"),
        end=None if end is None else pd.Timestamp(end).strftime("%Y-%m-%d"),
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    return _normalize_download(raw, tickers)


def align_prices(prices: pd.DataFrame, min_coverage: float = 0.95) -> pd.DataFrame:
    """
    Keep assets with reliable data during their own live history, then use the
    common overlapping sample.

    Pre-inception NaNs are deliberately ignored when assessing coverage. This
    allows newer ETFs to join the universe without being rejected simply because
    the requested start date predates their launch.
    """
    if prices.empty:
        raise ValueError("Price matrix is empty.")

    keep = []
    for ticker in prices.columns:
        series = prices[ticker]
        valid = series.dropna()
        if len(valid) < 30:
            continue

        active = series.loc[valid.index.min() : valid.index.max()]
        active_coverage = float(active.notna().mean())
        if active_coverage >= min_coverage:
            keep.append(ticker)

    if len(keep) < 2:
        raise ValueError("Fewer than two assets have sufficient usable history.")

    aligned = prices[keep].dropna(how="any")
    if aligned.shape[0] < 30:
        raise ValueError("Insufficient common price history after alignment.")
    return aligned


#: How many calendar days the last price may sit before its period label and
#: still count as a complete period. A month can end on a weekend plus a market
#: holiday (Good Friday on the 29th, say), so three days is normal and four is
#: the widest gap a finished month shows. A week only ever loses its Friday.
#: Anything wider means the period is still in progress.
_COMPLETE_PERIOD_TOLERANCE_DAYS = {"monthly": 4, "weekly": 1}


def _resample_last(prices: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if frequency == "monthly":
        return prices.resample("ME").last()
    if frequency == "weekly":
        return prices.resample("W-FRI").last()
    if frequency == "daily":
        return prices
    raise ValueError("frequency must be one of: daily, weekly, monthly")


def final_period_is_incomplete(
    prices: pd.DataFrame, frequency: str = "monthly"
) -> bool:
    """Whether the newest month/week is still in progress.

    Resampling labels the last bucket with its period end however few days it
    holds, so a 21 September price is reported as a "30 September" month. Treating
    that stub as a full period biases every monthly statistic: it counts as a
    whole month of history in annualisation, volatility and drawdown, and it
    becomes the final out-of-sample observation. Daily data has no such stub.
    """
    frequency = frequency.lower()
    if frequency == "daily" or prices.empty:
        return False
    sampled = _resample_last(prices, frequency)
    gap_days = (sampled.index.max() - prices.index.max()).days
    return gap_days > _COMPLETE_PERIOD_TOLERANCE_DAYS[frequency]


def to_returns(
    prices: pd.DataFrame,
    frequency: str = "monthly",
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Period returns from a price matrix.

    By default the trailing period is dropped when it is still in progress (see
    :func:`final_period_is_incomplete`), so every observation is a whole month
    or week. Pass ``drop_incomplete=False`` to keep the partial stub.
    """
    frequency = frequency.lower()
    sampled = _resample_last(prices, frequency)
    if drop_incomplete and final_period_is_incomplete(prices, frequency):
        sampled = sampled.iloc[:-1]

    returns = sampled.pct_change(fill_method=None).dropna(how="any")
    if returns.empty:
        raise ValueError("No return observations were produced.")
    return returns


def load_market_data(
    tickers: Iterable[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    frequency: str = "monthly",
    min_coverage: float = 0.95,
) -> MarketData:
    prices = download_prices(tickers, start=start, end=end)
    prices = align_prices(prices, min_coverage=min_coverage)
    returns = to_returns(prices, frequency=frequency)

    return MarketData(
        prices=prices,
        returns=returns,
        start=prices.index.min(),
        end=prices.index.max(),
        frequency=frequency,
    )


def load_benchmark_returns(
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    frequency: str = "monthly",
) -> pd.Series:
    """Download one benchmark independently of the candidate common sample.

    Keeping benchmark data separate prevents a custom market prior from
    shortening the candidate ETF universe merely because the benchmark has a
    different inception date. Alignment happens only when beta is estimated.
    """
    symbol = str(ticker).upper().strip()
    if not symbol:
        raise ValueError("A market benchmark ticker is required.")

    prices = download_prices([symbol], start=start, end=end)
    if symbol not in prices.columns:
        raise ValueError(f"Yahoo Finance did not return data for benchmark {symbol}.")

    series = prices[symbol].dropna().to_frame(symbol)
    if len(series) < 30:
        raise ValueError(f"Benchmark {symbol} has insufficient usable price history.")
    returns = to_returns(series, frequency=frequency)[symbol].astype(float)
    returns.name = symbol
    return returns
