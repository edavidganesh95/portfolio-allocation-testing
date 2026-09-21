from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import annualized_return, periods_per_year


def historical_cagr(returns: pd.DataFrame, frequency: str = "monthly") -> pd.Series:
    ppy = periods_per_year(frequency)
    return returns.apply(lambda s: annualized_return(s, ppy))


def _aligned_with_benchmark(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    min_observations: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    if returns.empty:
        raise ValueError("Candidate return matrix is empty.")
    if benchmark_returns is None or benchmark_returns.empty:
        raise ValueError("Market benchmark has no usable return history.")

    benchmark = pd.Series(benchmark_returns, dtype=float).rename("__market__")
    joined = returns.astype(float).join(benchmark, how="inner").dropna(how="any")
    if len(joined) < int(min_observations):
        raise ValueError(
            "The market benchmark has only "
            f"{len(joined)} overlapping return observations with the candidate universe; "
            f"at least {int(min_observations)} are required."
        )
    return joined[returns.columns], joined["__market__"]


def market_betas(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    min_observations: int = 24,
) -> pd.Series:
    """Estimate each candidate ETF's beta to a user-selected market benchmark."""
    assets, market = _aligned_with_benchmark(
        returns, benchmark_returns, min_observations=min_observations
    )
    market_var = float(market.var(ddof=1))
    if not np.isfinite(market_var) or market_var <= 1e-16:
        raise ValueError("Market benchmark variance is too small to estimate beta.")

    beta = assets.apply(lambda s: float(s.cov(market)) / market_var)
    beta.name = "Beta"
    return beta.astype(float)


def market_prior_returns(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    risk_free_rate: float = 0.0,
    min_observations: int = 24,
) -> pd.Series:
    """CAPM-style market prior implied by each ETF's benchmark beta.

    ``market_return_prior`` and ``risk_free_rate`` are annual decimal returns.
    The benchmark return history is used only to estimate beta; the user's
    long-run market return assumption supplies the expected market return.
    """
    betas = market_betas(returns, benchmark_returns, min_observations=min_observations)
    prior = float(risk_free_rate) + betas * (
        float(market_return_prior) - float(risk_free_rate)
    )
    prior.name = "Market Prior"
    return prior.astype(float)


def market_anchored_expected_returns(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    risk_free_rate: float = 0.0,
    frequency: str = "monthly",
    shrinkage: float = 0.50,
    min_observations: int = 24,
) -> pd.Series:
    """Blend historical CAGR with a benchmark-beta market prior.

    ``shrinkage=0`` leaves the historical CAGR unchanged.
    ``shrinkage=1`` uses only the CAPM-style market prior
    ``rf + beta * (market_return_prior - rf)``.

    Unlike peer-mean shrinkage, the anchor is external to the selected ETF
    universe: adding or removing a candidate does not mechanically change the
    prior for every other ETF.
    """
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be between 0 and 1")

    raw = historical_cagr(returns, frequency)
    prior = market_prior_returns(
        returns,
        benchmark_returns,
        market_return_prior=market_return_prior,
        risk_free_rate=risk_free_rate,
        min_observations=min_observations,
    ).reindex(raw.index)
    expected = (1.0 - float(shrinkage)) * raw + float(shrinkage) * prior
    expected.name = "Expected Return"
    return expected.astype(float)


def market_prior_diagnostics(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    risk_free_rate: float = 0.0,
    frequency: str = "monthly",
    shrinkage: float = 0.50,
    min_observations: int = 24,
) -> pd.DataFrame:
    """Transparent inputs behind the market-anchored expected-return estimate."""
    raw = historical_cagr(returns, frequency)
    beta = market_betas(
        returns, benchmark_returns, min_observations=min_observations
    ).reindex(raw.index)
    prior = market_prior_returns(
        returns,
        benchmark_returns,
        market_return_prior=market_return_prior,
        risk_free_rate=risk_free_rate,
        min_observations=min_observations,
    ).reindex(raw.index)
    expected = market_anchored_expected_returns(
        returns,
        benchmark_returns,
        market_return_prior=market_return_prior,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
        shrinkage=shrinkage,
        min_observations=min_observations,
    ).reindex(raw.index)
    return pd.DataFrame(
        {
            "Historical CAGR": raw,
            "Market Beta": beta,
            "Market Prior": prior,
            "Blended Expected Return": expected,
        }
    )
