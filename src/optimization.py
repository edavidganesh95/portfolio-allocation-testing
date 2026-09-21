from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize

from .risk import portfolio_volatility


def _cvar_constraint_matrix(matrix: np.ndarray) -> sparse.csc_matrix:
    """Rockafellar-Uryasev tail constraints ``-r_t·w - alpha - u_t <= 0``.

    Held sparse because the tail-slack block is a T x T identity: at daily
    frequency the dense equivalent is tens of megabytes per solve, which
    dominates both memory and build time. HiGHS consumes the sparse form
    directly and solves the identical program.
    """
    t_obs = matrix.shape[0]
    return sparse.hstack(
        [
            sparse.csc_matrix(-matrix),
            sparse.csc_matrix(-np.ones((t_obs, 1))),
            -sparse.identity(t_obs, format="csc"),
        ],
        format="csc",
    )


@dataclass(frozen=True)
class OptimizationResult:
    name: str
    weights: pd.Series
    success: bool
    message: str


def _bounds(n: int, max_weight: float) -> tuple[tuple[float, float], ...]:
    return tuple((0.0, float(max_weight)) for _ in range(n))


def _constraints():
    return ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)


def _result(name: str, res, tickers: list[str]) -> OptimizationResult:
    return OptimizationResult(
        name=name,
        weights=pd.Series(res.x, index=tickers),
        success=bool(res.success),
        message=str(res.message),
    )


def equal_weight(tickers: list[str]) -> OptimizationResult:
    n = len(tickers)
    return OptimizationResult(
        name="Equal Weight",
        weights=pd.Series(np.repeat(1.0 / n, n), index=tickers),
        success=True,
        message="Closed-form baseline",
    )


def minimum_variance(cov: pd.DataFrame, max_weight: float = 1.0) -> OptimizationResult:
    tickers = list(cov.index)
    n = len(tickers)
    x0 = np.repeat(1.0 / n, n)

    res = minimize(
        lambda w: w @ cov.values @ w,
        x0,
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=_constraints(),
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return _result("Minimum Variance", res, tickers)


def maximum_sharpe(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> OptimizationResult:
    tickers = list(cov.index)
    mu = expected_returns.reindex(tickers).values
    n = len(tickers)
    x0 = np.repeat(1.0 / n, n)

    def objective(w):
        vol = portfolio_volatility(w, cov.values)
        if vol <= 0:
            return 1e6
        return -float((w @ mu - risk_free_rate) / vol)

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=_constraints(),
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return _result("Maximum Sharpe", res, tickers)


def diversified_maximum_sharpe(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 0.50,
    min_effective_assets: float = 3.0,
    max_risk_share: float = 0.50,
) -> OptimizationResult:
    """Maximise expected Sharpe subject to explicit diversification guardrails.

    The guardrails are intentionally transparent for a buy-and-hold audience:

    * no single ETF can exceed ``max_weight``;
    * the inverse-Herfindahl effective asset count must be at least
      ``min_effective_assets``;
    * no single ETF may contribute more than ``max_risk_share`` of portfolio
      variance risk.

    This is not a new return forecast. It uses the same market-anchored expected
    returns and Ledoit-Wolf covariance estimate as Maximum Sharpe, but searches
    only inside a more diversified feasible set.
    """
    tickers = list(cov.index)
    mu = expected_returns.reindex(tickers).values.astype(float)
    sigma = cov.loc[tickers, tickers].values.astype(float)
    n = len(tickers)

    if min_effective_assets < 1.0 or min_effective_assets > n:
        raise ValueError(
            "min_effective_assets must be between 1 and the number of assets"
        )
    if not 0 < max_risk_share <= 1.0:
        raise ValueError("max_risk_share must be between 0 and 1")
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError("max_weight is infeasible for the number of assets")

    # Equal weight is always a useful feasibility anchor when the requested
    # effective-count floor is <= n and max_weight permits it.
    x0 = np.repeat(1.0 / n, n)

    def objective(w):
        vol = portfolio_volatility(w, sigma)
        if vol <= 1e-12:
            return 1e6
        return -float((w @ mu - risk_free_rate) / vol)

    def effective_count_constraint(w):
        # 1 / sum(w^2) >= N_min  <=>  sum(w^2) <= 1/N_min
        return float(1.0 / min_effective_assets - np.sum(np.square(w)))

    def risk_share_constraints(w):
        variance = float(w @ sigma @ w)
        if variance <= 1e-16:
            return np.repeat(max_risk_share, n)
        # Variance contribution shares sum to one: w_i (Sigma w)_i / sigma_p^2.
        shares = w * (sigma @ w) / variance
        return max_risk_share - shares

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": effective_count_constraint},
        {"type": "ineq", "fun": risk_share_constraints},
    )

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=constraints,
        options={"maxiter": 6000, "ftol": 1e-12},
    )
    return _result("Diversified Maximum Sharpe", res, tickers)


def maximum_diversification(
    cov: pd.DataFrame,
    max_weight: float = 1.0,
) -> OptimizationResult:
    tickers = list(cov.index)
    n = len(tickers)
    x0 = np.repeat(1.0 / n, n)
    asset_vols = np.sqrt(np.diag(cov.values))

    def objective(w):
        port_vol = portfolio_volatility(w, cov.values)
        if port_vol <= 0:
            return 1e6
        diversification_ratio = float((w @ asset_vols) / port_vol)
        return -diversification_ratio

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=_constraints(),
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    return _result("Maximum Diversification", res, tickers)


def risk_parity(cov: pd.DataFrame, max_weight: float = 1.0) -> OptimizationResult:
    tickers = list(cov.index)
    n = len(tickers)
    x0 = np.repeat(1.0 / n, n)

    def objective(w):
        port_vol = portfolio_volatility(w, cov.values)
        if port_vol <= 0:
            return 1e6
        marginal = cov.values @ w / port_vol
        rc = w * marginal
        target = np.repeat(port_vol / n, n)
        return float(np.sum((rc - target) ** 2))

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=_constraints(),
        options={"maxiter": 4000, "ftol": 1e-14},
    )
    return _result("Risk Parity", res, tickers)


def _empirical_cvar_from_weights(
    weights: np.ndarray,
    returns: np.ndarray,
    level: float = 0.95,
) -> float:
    """Historical CVaR expressed as a positive loss number."""
    portfolio = returns @ weights
    cutoff = np.quantile(portfolio, 1.0 - level)
    tail = portfolio[portfolio <= cutoff]
    if tail.size == 0:
        return 0.0
    return float(max(-tail.mean(), 0.0))


def minimum_cvar(
    returns: pd.DataFrame,
    level: float = 0.95,
    max_weight: float = 1.0,
) -> OptimizationResult:
    """
    Minimise historical CVaR using the Rockafellar-Uryasev linear-program
    formulation.

    Variables are portfolio weights, VaR threshold alpha, and one non-negative
    tail slack variable per observation.
    """
    if not 0.5 < level < 1.0:
        raise ValueError("level must be between 0.5 and 1.0")

    tickers = list(returns.columns)
    matrix = returns.values.astype(float)
    t_obs, n_assets = matrix.shape

    # x = [weights(n), alpha(1), u(t)]
    c = np.zeros(n_assets + 1 + t_obs)
    c[n_assets] = 1.0
    c[n_assets + 1 :] = 1.0 / ((1.0 - level) * t_obs)

    # loss_t - alpha - u_t <= 0, where loss_t = -r_t @ w
    a_ub = _cvar_constraint_matrix(matrix)
    b_ub = np.zeros(t_obs)

    a_eq = sparse.csc_matrix(
        (
            np.ones(n_assets),
            (np.zeros(n_assets, dtype=int), np.arange(n_assets)),
        ),
        shape=(1, n_assets + 1 + t_obs),
    )
    b_eq = np.array([1.0])

    bounds = (
        [(0.0, float(max_weight))] * n_assets + [(None, None)] + [(0.0, None)] * t_obs
    )

    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if res.success:
        weights = pd.Series(res.x[:n_assets], index=tickers)
    else:
        weights = pd.Series(np.repeat(1.0 / n_assets, n_assets), index=tickers)

    return OptimizationResult(
        name="Minimum CVaR",
        weights=weights,
        success=bool(res.success),
        message=str(res.message),
    )


def maximum_return_to_cvar(
    expected_returns: pd.Series,
    returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    level: float = 0.95,
    max_weight: float = 1.0,
) -> OptimizationResult:
    """
    Maximise expected excess return divided by empirical historical CVaR.

    This is deliberately a reference construction rather than a promise of
    future tail efficiency; the later multi-period, bootstrap and walk-forward layers test
    how stable the result is.
    """
    tickers = list(returns.columns)
    mu = expected_returns.reindex(tickers).values.astype(float)
    matrix = returns.values.astype(float)
    n = len(tickers)
    x0 = np.repeat(1.0 / n, n)

    def objective(w):
        cvar = _empirical_cvar_from_weights(w, matrix, level)
        if cvar <= 1e-10:
            return 1e6
        expected_excess = float(w @ mu - risk_free_rate)
        return -expected_excess / cvar

    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=_bounds(n, max_weight),
        constraints=_constraints(),
        options={"maxiter": 4000, "ftol": 1e-12},
    )
    return _result("Maximum Return / CVaR", res, tickers)
