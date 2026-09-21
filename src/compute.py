"""Cached compute layer.

Streamlit re-executes the whole script on every widget interaction. Without a
cache, changing which portfolios to plot in the bootstrap tab would re-solve
the multi-period refits, both frontiers and the entire walk-forward backtest —
several seconds of work whose inputs did not change.

Every expensive call therefore goes through a thin wrapper here, keyed on the
values that actually determine the result. The engine in ``src`` is untouched:
these are caches around it, so the numbers are identical and only the repeat
cost disappears.

Arguments are primitives, tuples or pandas objects so Streamlit can hash them
reliably; results are plain data so they pickle into the cache cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st

from .bootstrap import (
    BootstrapSettings,
    run_bootstrap_reduced,
)
from .data import load_benchmark_returns, load_market_data, to_returns
from .expected_returns import (
    historical_cagr,
    market_anchored_expected_returns,
    market_prior_diagnostics,
)
from .frontier import (
    cvar_frontier,
    cvar_opportunity_set,
    efficient_frontier,
    return_diversification_frontier,
    simulate_feasible_portfolios,
)
from .metrics import summary_table
from .multiperiod import (
    construction_weights_by_lookback,
    portfolio_metrics_by_lookback,
)
from .optimization import (
    diversified_maximum_sharpe,
    equal_weight,
    maximum_diversification,
    maximum_return_to_cvar,
    maximum_sharpe,
    minimum_cvar,
    minimum_variance,
    risk_parity,
)
from .risk import ledoit_wolf_covariance
from .walkforward import WalkForwardSettings, walk_forward_backtest

DATA_TTL = 6 * 60 * 60

CONSTRUCTION_ORDER = (
    "Equal Weight",
    "Minimum Variance",
    "Risk Parity",
    "Maximum Diversification",
    "Maximum Sharpe",
    "Diversified Maximum Sharpe",
    "Minimum CVaR",
    "Maximum Return / CVaR",
)


# ----------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------


@st.cache_data(ttl=DATA_TTL, show_spinner=False, max_entries=12)
def cached_market_data(
    tickers: tuple[str, ...],
    start: str,
    end: str,
    frequency: str,
):
    return load_market_data(tickers, start=start, end=end, frequency=frequency)


@st.cache_data(ttl=DATA_TTL, show_spinner=False, max_entries=12)
def cached_benchmark_returns(
    ticker: str,
    start: str,
    end: str,
    frequency: str,
) -> pd.Series:
    return load_benchmark_returns(
        ticker,
        start=start,
        end=end,
        frequency=frequency,
    )


@st.cache_data(show_spinner=False, max_entries=12)
def cached_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return to_returns(prices, frequency="monthly")


# ----------------------------------------------------------------------
# Shared model inputs
# ----------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=24)
def cached_expected_returns(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    risk_free_rate: float,
    frequency: str,
    shrinkage: float,
) -> pd.Series:
    return market_anchored_expected_returns(
        returns,
        benchmark_returns,
        market_return_prior=market_return_prior,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
        shrinkage=shrinkage,
    )


@st.cache_data(show_spinner=False, max_entries=24)
def cached_expected_return_diagnostics(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    risk_free_rate: float,
    frequency: str,
    shrinkage: float,
) -> pd.DataFrame:
    return market_prior_diagnostics(
        returns,
        benchmark_returns,
        market_return_prior=market_return_prior,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
        shrinkage=shrinkage,
    )


@st.cache_data(show_spinner=False, max_entries=24)
def cached_historical_cagr(
    returns: pd.DataFrame,
    frequency: str,
) -> pd.Series:
    return historical_cagr(returns, frequency=frequency)


@st.cache_data(show_spinner=False, max_entries=24)
def cached_covariance(
    returns: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    return ledoit_wolf_covariance(returns, frequency=frequency)


@st.cache_data(show_spinner=False, max_entries=40)
def cached_summary_table(
    returns: pd.DataFrame,
    frequency: str,
    risk_free_rate: float,
    cvar_level: float,
) -> pd.DataFrame:
    return summary_table(
        returns,
        frequency=frequency,
        risk_free_rate=risk_free_rate,
        cvar_level=cvar_level,
    )


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=16)
def cached_construction(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    frequency: str,
    shrinkage: float,
    max_weight: float,
    risk_free_rate: float,
    cvar_level: float,
    diversified_max_weight: float = 0.50,
    diversified_min_effective_assets: float = 3.0,
    diversified_max_risk_share: float = 0.50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full-sample construction methods as a method x ticker weight matrix."""
    tickers = list(returns.columns)
    mu = market_anchored_expected_returns(
        returns,
        benchmark_returns,
        market_return_prior=market_return_prior,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
        shrinkage=shrinkage,
    )
    cov = ledoit_wolf_covariance(returns, frequency=frequency)

    results = [
        equal_weight(tickers),
        minimum_variance(cov, max_weight=max_weight),
        risk_parity(cov, max_weight=max_weight),
        maximum_diversification(cov, max_weight=max_weight),
        maximum_sharpe(mu, cov, risk_free_rate=risk_free_rate, max_weight=max_weight),
        diversified_maximum_sharpe(
            mu,
            cov,
            risk_free_rate=risk_free_rate,
            max_weight=min(float(max_weight), float(diversified_max_weight)),
            min_effective_assets=float(diversified_min_effective_assets),
            max_risk_share=float(diversified_max_risk_share),
        ),
        minimum_cvar(returns, level=cvar_level, max_weight=max_weight),
        maximum_return_to_cvar(
            mu,
            returns,
            risk_free_rate=risk_free_rate,
            level=cvar_level,
            max_weight=max_weight,
        ),
    ]

    status = pd.DataFrame(
        [
            {
                "Method": result.name,
                "Success": bool(result.success),
                "Message": str(result.message),
            }
            for result in results
        ]
    )
    successful = [result for result in results if result.success]
    if not successful:
        return pd.DataFrame(columns=tickers), status

    weights = pd.concat(
        {result.name: result.weights for result in successful}, axis=1
    ).T
    # ``concat`` sorts the keys; restore the intended presentation order.
    ordered = [name for name in CONSTRUCTION_ORDER if name in weights.index]
    return weights.loc[ordered], status


@st.cache_data(show_spinner=False, max_entries=12)
def cached_all_lookback_weights(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    risk_free_rate: float,
    frequency: str,
    shrinkage: float,
    max_weight: float,
    cvar_level: float,
    diversified_max_weight: float,
    diversified_min_effective_assets: float,
    diversified_max_risk_share: float,
):
    return construction_weights_by_lookback(
        returns,
        benchmark_returns,
        market_return_prior=market_return_prior,
        risk_free_rate=risk_free_rate,
        frequency=frequency,
        shrinkage=shrinkage,
        max_weight=max_weight,
        cvar_level=cvar_level,
        diversified_max_weight=diversified_max_weight,
        min_effective_assets=diversified_min_effective_assets,
        max_risk_share=diversified_max_risk_share,
    )


@st.cache_data(show_spinner=False, max_entries=12)
def cached_multi_period_metrics(
    returns: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    frequency: str,
    risk_free_rate: float,
    cvar_level: float,
):
    portfolios = {
        str(name): portfolio_weights[name].dropna()
        for name in portfolio_weights.columns
    }
    return portfolio_metrics_by_lookback(
        returns,
        portfolios,
        frequency=frequency,
        risk_free_rate=risk_free_rate,
        cvar_level=cvar_level,
    )


# ----------------------------------------------------------------------
# Frontier
# ----------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=12)
def cached_feasible_cloud(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    max_weight: float,
    simulations: int,
    risk_free_rate: float,
    seed: int = 42,
) -> pd.DataFrame:
    return simulate_feasible_portfolios(
        expected_returns,
        cov,
        max_weight=max_weight,
        simulations=simulations,
        risk_free_rate=risk_free_rate,
        seed=seed,
    )


@st.cache_data(show_spinner=False, max_entries=16)
def cached_efficient_frontier(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    max_weight: float,
    points: int,
) -> pd.DataFrame:
    return efficient_frontier(
        expected_returns, cov, max_weight=max_weight, points=points
    )


@st.cache_data(show_spinner=False, max_entries=16)
def cached_return_diversification_frontier(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    max_weight: float,
    points: int,
) -> pd.DataFrame:
    return return_diversification_frontier(
        expected_returns, cov, max_weight=max_weight, points=points
    )


@st.cache_data(show_spinner=False, max_entries=12)
def cached_cvar_opportunity_set(
    feasible: pd.DataFrame,
    returns: pd.DataFrame,
    level: float,
) -> pd.DataFrame:
    return cvar_opportunity_set(feasible, returns, level=level)


@st.cache_data(show_spinner=False, max_entries=12)
def cached_cvar_frontier(
    expected_returns: pd.Series,
    returns: pd.DataFrame,
    max_weight: float,
    level: float,
    points: int,
) -> pd.DataFrame:
    return cvar_frontier(
        expected_returns,
        returns,
        max_weight=max_weight,
        level=level,
        points=points,
    )


# ----------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------


@dataclass
class BootstrapOutput:
    """Everything the bootstrap section displays, and nothing more.

    The raw simulation holds one wealth path per month per draw — tens of
    megabytes once several portfolios are selected. Only the summary, the
    percentile fan and the three terminal distributions are ever shown, so the
    cache stores those and discards the full paths.
    """

    summary: pd.DataFrame
    fans: dict[str, pd.DataFrame]
    terminal_wealth: dict[str, np.ndarray]
    cagr: dict[str, np.ndarray]
    max_drawdown: dict[str, np.ndarray]


@st.cache_data(show_spinner=False, max_entries=8)
def cached_bootstrap(
    monthly_returns: pd.DataFrame,
    portfolio_weights: pd.DataFrame,
    simulations: int,
    horizon_years: int,
    block_months: int,
    seed: int,
    initial_wealth: float,
    target_cagrs: tuple[float, ...] = (0.06, 0.08, 0.10),
) -> BootstrapOutput:
    settings = BootstrapSettings(
        simulations=int(simulations),
        horizon_years=int(horizon_years),
        block_months=int(block_months),
        seed=int(seed),
        initial_wealth=float(initial_wealth),
    )
    portfolios = {
        str(name): portfolio_weights[name].dropna()
        for name in portfolio_weights.columns
    }
    reduced = run_bootstrap_reduced(
        monthly_returns,
        portfolios,
        settings,
        target_cagrs=target_cagrs,
    )
    return BootstrapOutput(
        summary=reduced.summary,
        fans=reduced.fans,
        terminal_wealth=reduced.terminal_wealth,
        cagr=reduced.cagr,
        max_drawdown=reduced.max_drawdown,
    )


# ----------------------------------------------------------------------
# Walk-forward
# ----------------------------------------------------------------------


@st.cache_data(show_spinner=False, max_entries=8)
def cached_walk_forward(
    monthly_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    market_return_prior: float,
    optimizer_assets: tuple[str, ...],
    lookback_months: int | None,
    min_train_months: int,
    holding_months: int,
    risk_free_rate: float,
    expected_return_shrinkage: float,
    max_weight: float,
    cvar_level: float,
    reference_weights: pd.Series | None,
    diversified_max_weight: float = 0.50,
    diversified_min_effective_assets: float = 3.0,
    diversified_max_risk_share: float = 0.50,
):
    settings = WalkForwardSettings(
        lookback_months=lookback_months,
        min_train_months=int(min_train_months),
        holding_months=int(holding_months),
        risk_free_rate=float(risk_free_rate),
        expected_return_shrinkage=float(expected_return_shrinkage),
        market_return_prior=float(market_return_prior),
        max_weight=float(max_weight),
        cvar_level=float(cvar_level),
        diversified_max_weight=float(diversified_max_weight),
        diversified_min_effective_assets=float(diversified_min_effective_assets),
        diversified_max_risk_share=float(diversified_max_risk_share),
    )
    return walk_forward_backtest(
        monthly_returns,
        optimizer_assets=list(optimizer_assets),
        settings=settings,
        reference_weights=reference_weights,
        benchmark_returns=benchmark_returns,
    )


def weights_frame(portfolios: dict[str, pd.Series]) -> pd.DataFrame:
    """Portfolio dictionary as a hashable ticker x portfolio frame."""
    if not portfolios:
        return pd.DataFrame()
    return pd.DataFrame(
        {name: pd.Series(weights, dtype=float) for name, weights in portfolios.items()}
    )
