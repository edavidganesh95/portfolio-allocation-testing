from __future__ import annotations

import numpy as np
import pandas as pd

PERIODS = {"daily": 252, "weekly": 52, "monthly": 12}


def periods_per_year(frequency: str) -> int:
    try:
        return PERIODS[frequency.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported frequency: {frequency}") from exc


def annualized_return(r: pd.Series, periods: int) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    wealth = float((1.0 + r).prod())
    years = len(r) / periods
    if years <= 0 or wealth <= 0:
        return np.nan
    return wealth ** (1.0 / years) - 1.0


def annualized_volatility(r: pd.Series, periods: int) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods))


def downside_deviation(r: pd.Series, periods: int, mar: float = 0.0) -> float:
    downside = np.minimum(r - mar / periods, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods))


def sharpe_ratio(r: pd.Series, periods: int, risk_free_rate: float = 0.0) -> float:
    ann_ret = annualized_return(r, periods)
    ann_vol = annualized_volatility(r, periods)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(r: pd.Series, periods: int, risk_free_rate: float = 0.0) -> float:
    ann_ret = annualized_return(r, periods)
    dd = downside_deviation(r, periods)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float((ann_ret - risk_free_rate) / dd)


def drawdown_series(r: pd.Series) -> pd.Series:
    wealth = (1.0 + r.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    return wealth / peak - 1.0


def max_drawdown(r: pd.Series) -> float:
    dd = drawdown_series(r)
    return float(dd.min()) if not dd.empty else np.nan


def historical_var(r: pd.Series, level: float = 0.95) -> float:
    return float(-np.quantile(r.dropna(), 1 - level))


def historical_cvar(r: pd.Series, level: float = 0.95) -> float:
    clean = r.dropna()
    if clean.empty:
        return np.nan
    cutoff = np.quantile(clean, 1 - level)
    tail = clean[clean <= cutoff]
    return float(-tail.mean()) if not tail.empty else np.nan


def summary_table(
    returns: pd.DataFrame,
    frequency: str = "monthly",
    risk_free_rate: float = 0.0,
    cvar_level: float = 0.95,
) -> pd.DataFrame:
    ppy = periods_per_year(frequency)
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna()
        rows.append(
            {
                "Ticker": ticker,
                "CAGR": annualized_return(r, ppy),
                "Volatility": annualized_volatility(r, ppy),
                "Sharpe": sharpe_ratio(r, ppy, risk_free_rate),
                "Sortino": sortino_ratio(r, ppy, risk_free_rate),
                "Max Drawdown": max_drawdown(r),
                f"CVaR {int(cvar_level * 100)}%": historical_cvar(r, cvar_level),
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


def buy_and_hold_portfolio_returns(
    returns: pd.DataFrame, weights: pd.Series
) -> pd.Series:
    """Portfolio returns from one initial allocation with no rebalancing.

    Each holding compounds independently from its initial weight. The portfolio
    return in later periods therefore reflects naturally drifted weights rather
    than mechanically resetting back to the starting allocation.
    """
    clean = returns.astype(float).copy()
    w = weights.reindex(clean.columns).fillna(0.0).astype(float).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive number.")
    w = w / total

    asset_growth = (1.0 + clean).cumprod()
    wealth = asset_growth.mul(w, axis=1).sum(axis=1)
    previous = wealth.shift(1)
    previous.iloc[0] = 1.0
    out = wealth / previous - 1.0
    out.name = getattr(weights, "name", None)
    return out.astype(float)


def buy_and_hold_weight_path(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    """Drifted portfolio weights after each observed return period."""
    clean = returns.astype(float).copy()
    w = weights.reindex(clean.columns).fillna(0.0).astype(float).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive number.")
    w = w / total
    asset_values = (1.0 + clean).cumprod().mul(w, axis=1)
    wealth = asset_values.sum(axis=1).replace(0.0, np.nan)
    return asset_values.div(wealth, axis=0).fillna(0.0)


def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    weights = weights.reindex(returns.columns).fillna(0.0)
    total = float(weights.sum())
    if not np.isclose(total, 1.0):
        if total <= 0:
            raise ValueError("Portfolio weights must sum to a positive number.")
        weights = weights / total
    return returns.mul(weights, axis=1).sum(axis=1)
