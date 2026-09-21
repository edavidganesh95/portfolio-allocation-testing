from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize

from .optimization import (
    _cvar_constraint_matrix,
    _empirical_cvar_from_weights,
    minimum_variance,
)
from .risk import portfolio_volatility


def validate_weight_cap(n_assets: int, max_weight: float) -> None:
    if n_assets < 2:
        raise ValueError("At least two assets are required.")
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in (0, 1].")
    if n_assets * max_weight < 1.0 - 1e-12:
        raise ValueError(
            "Maximum weight is infeasible for the selected universe. "
            f"With {n_assets} assets, max_weight must be at least "
            f"{1.0 / n_assets:.1%}."
        )


def _max_return_weights(
    expected_returns: pd.Series,
    max_weight: float,
) -> pd.Series:
    tickers = list(expected_returns.index)
    n = len(tickers)
    validate_weight_cap(n, max_weight)

    res = linprog(
        c=-expected_returns.to_numpy(float),
        A_eq=np.ones((1, n)),
        b_eq=np.array([1.0]),
        bounds=[(0.0, float(max_weight))] * n,
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"Maximum-return portfolio failed: {res.message}")
    return pd.Series(res.x, index=tickers)


def simulate_feasible_portfolios(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    max_weight: float,
    simulations: int = 5_000,
    risk_free_rate: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Random feasible long-only portfolios for visualising the opportunity set.

    The cloud is diagnostic only. The efficient frontier is solved separately
    with constrained optimisation and is not inferred from the cloud boundary.
    """
    tickers = list(cov.index)
    n = len(tickers)
    validate_weight_cap(n, max_weight)

    if simulations < 100:
        raise ValueError("simulations must be at least 100.")

    rng = np.random.default_rng(seed)

    # Exact equal-weight corner when the cap leaves no freedom.
    if np.isclose(n * max_weight, 1.0, atol=1e-12):
        weights = np.repeat(1.0 / n, n)[None, :]
        weights = np.repeat(weights, simulations, axis=0)
    else:
        accepted = []
        remaining = simulations
        attempts = 0

        while remaining > 0 and attempts < 80:
            batch_size = max(2_000, remaining * 3)
            batch = rng.dirichlet(np.ones(n), size=batch_size)
            batch = batch[(batch.max(axis=1) <= max_weight + 1e-12)]
            if len(batch):
                take = batch[:remaining]
                accepted.append(take)
                remaining -= len(take)
            attempts += 1

        if remaining > 0:
            # Fallback: convex blends between equal weight and random
            # Dirichlet draws always move toward feasibility.
            equal = np.repeat(1.0 / n, n)
            fallback = []
            while remaining > 0:
                raw = rng.dirichlet(np.ones(n), size=max(1_000, remaining))
                for row in raw:
                    if row.max() <= max_weight + 1e-12:
                        fallback.append(row)
                    else:
                        # Find the largest blend alpha in [0,1] such that
                        # equal + alpha*(row-equal) respects the cap.
                        denom = row - equal
                        alpha = 1.0
                        positive = denom > 0
                        if positive.any():
                            alpha = min(
                                1.0,
                                float(
                                    np.min(
                                        (max_weight - equal[positive]) / denom[positive]
                                    )
                                ),
                            )
                        blended = equal + max(alpha, 0.0) * denom
                        fallback.append(blended)
                    if len(fallback) >= remaining:
                        break
                if fallback:
                    break
            accepted.append(np.asarray(fallback[:remaining]))

        weights = np.vstack(accepted)[:simulations]

    mu = expected_returns.reindex(tickers).to_numpy(float)
    cov_values = cov.loc[tickers, tickers].to_numpy(float)

    exp_returns = weights @ mu
    variances = np.einsum("ij,jk,ik->i", weights, cov_values, weights)
    vol = np.sqrt(np.maximum(variances, 0.0))
    sharpe = np.divide(
        exp_returns - risk_free_rate,
        vol,
        out=np.full_like(exp_returns, np.nan),
        where=vol > 0,
    )

    out = pd.DataFrame(
        {
            "Expected Return": exp_returns,
            "Volatility": vol,
            "Sharpe": sharpe,
        }
    )
    for i, ticker in enumerate(tickers):
        out[f"w_{ticker}"] = weights[:, i]
    return out


def efficient_frontier(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    max_weight: float,
    points: int = 50,
) -> pd.DataFrame:
    """
    Exact long-only mean-variance efficient frontier under a per-asset cap.

    Targets run from the GMV portfolio's expected return to the maximum
    feasible expected return. Each point is a target-return/minimum-variance
    constrained optimisation.
    """
    tickers = list(cov.index)
    n = len(tickers)
    validate_weight_cap(n, max_weight)

    mu = expected_returns.reindex(tickers).to_numpy(float)
    cov_values = cov.loc[tickers, tickers].to_numpy(float)

    gmv = minimum_variance(cov, max_weight=max_weight)
    if not gmv.success:
        raise RuntimeError(f"GMV optimisation failed: {gmv.message}")
    gmv_return = float(gmv.weights.reindex(tickers).to_numpy() @ mu)

    max_ret_w = _max_return_weights(
        expected_returns.reindex(tickers),
        max_weight=max_weight,
    )
    max_ret = float(max_ret_w.to_numpy() @ mu)

    if max_ret <= gmv_return + 1e-12:
        targets = np.array([gmv_return])
    else:
        targets = np.linspace(gmv_return, max_ret, int(points))

    rows = []
    x0 = gmv.weights.reindex(tickers).to_numpy(float)

    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {
                "type": "eq",
                "fun": lambda w, t=target: float(w @ mu - t),
            },
        )
        res = minimize(
            lambda w: float(w @ cov_values @ w),
            x0,
            method="SLSQP",
            bounds=[(0.0, float(max_weight))] * n,
            constraints=constraints,
            options={"maxiter": 4000, "ftol": 1e-12},
        )
        if not res.success:
            continue

        w = res.x
        vol = portfolio_volatility(w, cov_values)
        row = {
            "Expected Return": float(w @ mu),
            "Volatility": float(vol),
        }
        for i, ticker in enumerate(tickers):
            row[f"w_{ticker}"] = float(w[i])
        rows.append(row)
        x0 = w

    return pd.DataFrame(rows)


def return_diversification_frontier(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    max_weight: float,
    points: int = 45,
) -> pd.DataFrame:
    """Maximum-diversification frontier across minimum expected-return targets.

    The diversification ratio is homogeneous in weights, so each frontier point
    is solved with a transformed quadratic programme rather than by taking a
    noisy outer envelope of random portfolios.

    Let ``a`` be the vector of standalone asset volatilities.  Maximising
    ``(a'w)/sqrt(w'Sigma w)`` is equivalent to minimising ``y'Sigma y`` subject
    to ``a'y = 1`` and then recovering ``w = y / sum(y)``.  Full-investment,
    long-only, per-asset-cap and minimum-return constraints all remain linear in
    the transformed variable ``y``:

    * ``y >= 0``;
    * ``y_i <= max_weight * sum(y)``;
    * ``(mu - target)'y >= 0``.

    This produces one economically interpretable boundary: for every minimum
    expected return, the returned point is the most diversified feasible
    portfolio under the project's standard long-only and weight-cap rules.
    """
    tickers = list(cov.index)
    n = len(tickers)
    validate_weight_cap(n, max_weight)
    if int(points) < 2:
        raise ValueError("points must be at least 2")

    mu = expected_returns.reindex(tickers).to_numpy(float)
    sigma = cov.loc[tickers, tickers].to_numpy(float)
    asset_vol = np.sqrt(np.diag(sigma))
    if np.any(~np.isfinite(asset_vol)) or np.any(asset_vol <= 0):
        raise ValueError("All assets must have positive finite volatility.")

    # Diversification-first endpoint under the same cap.  Imported lazily to
    # avoid broadening the module's public optimisation dependency surface.
    from .optimization import maximum_diversification

    max_div = maximum_diversification(cov.loc[tickers, tickers], max_weight=max_weight)
    if not max_div.success:
        raise RuntimeError(
            f"Maximum-diversification optimisation failed: {max_div.message}"
        )
    w_div = max_div.weights.reindex(tickers).to_numpy(float)
    min_target = float(w_div @ mu)

    max_ret_w = _max_return_weights(
        expected_returns.reindex(tickers), max_weight=max_weight
    ).to_numpy(float)
    max_target = float(max_ret_w @ mu)
    targets = (
        np.array([min_target])
        if max_target <= min_target + 1e-12
        else np.linspace(min_target, max_target, int(points))
    )

    # Transform a feasible fully-invested w into y with a'y = 1.
    def to_y(w: np.ndarray) -> np.ndarray:
        denom = float(asset_vol @ w)
        if denom <= 1e-15:
            raise ValueError("Cannot transform a zero-volatility portfolio.")
        return np.asarray(w, dtype=float) / denom

    y0 = to_y(w_div)
    rows: list[dict[str, float]] = []

    for target in targets:

        def objective(y: np.ndarray) -> float:
            return float(y @ sigma @ y)

        constraints = [
            {"type": "eq", "fun": lambda y: float(asset_vol @ y - 1.0)},
            {
                "type": "ineq",
                "fun": lambda y, t=float(target): float((mu - t) @ y),
            },
            {
                "type": "ineq",
                "fun": lambda y: float(max_weight * np.sum(y)) - y,
            },
        ]
        res = minimize(
            objective,
            y0,
            method="SLSQP",
            bounds=[(0.0, None)] * n,
            constraints=constraints,
            options={"maxiter": 6000, "ftol": 1e-12},
        )
        if not res.success:
            # The highest target can sit directly on a cap corner.  Seed that
            # target with the exact maximum-return portfolio before giving up.
            fallback = to_y(max_ret_w)
            res = minimize(
                objective,
                fallback,
                method="SLSQP",
                bounds=[(0.0, None)] * n,
                constraints=constraints,
                options={"maxiter": 6000, "ftol": 1e-12},
            )
        if not res.success:
            continue

        y = np.asarray(res.x, dtype=float)
        total = float(y.sum())
        if total <= 1e-15:
            continue
        w = y / total
        port_vol = portfolio_volatility(w, sigma)
        dr = float((w @ asset_vol) / port_vol) if port_vol > 0 else np.nan
        exp_ret = float(w @ mu)
        if exp_ret + 1e-8 < float(target):
            continue

        row: dict[str, float] = {
            "Minimum Expected Return": float(target),
            "Expected Return": exp_ret,
            "Diversification Ratio": dr,
        }
        for i, ticker in enumerate(tickers):
            row[f"w_{ticker}"] = float(w[i])
        rows.append(row)
        y0 = y

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    # Numerical solves can produce essentially identical neighbouring points;
    # collapse them so the plotted line does not visually double back.
    out = out.sort_values("Expected Return").drop_duplicates(
        subset=["Expected Return", "Diversification Ratio"]
    )
    return out.reset_index(drop=True)


def named_portfolio_points(
    portfolios: dict[str, pd.Series],
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    tickers = list(cov.index)
    mu = expected_returns.reindex(tickers).to_numpy(float)
    cov_values = cov.loc[tickers, tickers].to_numpy(float)

    rows = []
    for name, weights in portfolios.items():
        w = weights.reindex(tickers).fillna(0.0).to_numpy(float)
        total = float(w.sum())
        if total <= 0:
            continue
        w = w / total
        exp_ret = float(w @ mu)
        vol = portfolio_volatility(w, cov_values)
        sharpe = float((exp_ret - risk_free_rate) / vol) if vol > 0 else np.nan

        top_idx = np.argsort(w)[::-1]
        top_parts = [f"{tickers[i]} {w[i]:.0%}" for i in top_idx if w[i] >= 0.005][:6]

        rows.append(
            {
                "Portfolio": name,
                "Expected Return": exp_ret,
                "Volatility": vol,
                "Sharpe": sharpe,
                "Weights": " · ".join(top_parts),
            }
        )

    return pd.DataFrame(rows)


def cvar_opportunity_set(
    feasible: pd.DataFrame,
    returns: pd.DataFrame,
    level: float = 0.95,
) -> pd.DataFrame:
    tickers = list(returns.columns)
    weight_cols = [f"w_{ticker}" for ticker in tickers]
    weights = feasible[weight_cols].to_numpy(float)

    period_returns = returns[tickers].to_numpy(float) @ weights.T
    cutoff = np.quantile(period_returns, 1.0 - level, axis=0)

    # Mean of returns in each portfolio's own lower tail.
    tail_mask = period_returns <= cutoff[None, :]
    tail_sum = np.where(tail_mask, period_returns, 0.0).sum(axis=0)
    tail_count = tail_mask.sum(axis=0)
    tail_mean = np.divide(
        tail_sum,
        tail_count,
        out=np.zeros_like(tail_sum),
        where=tail_count > 0,
    )
    cvar = np.maximum(-tail_mean, 0.0)

    out = feasible.copy()
    out["CVaR"] = cvar
    return out


def minimum_cvar_for_target_return(
    expected_returns: pd.Series,
    returns: pd.DataFrame,
    target_return: float,
    level: float,
    max_weight: float,
) -> tuple[pd.Series, float] | None:
    """
    Historical minimum-CVaR portfolio subject to an annual expected-return target.
    """
    tickers = list(returns.columns)
    n = len(tickers)
    validate_weight_cap(n, max_weight)

    matrix = returns[tickers].to_numpy(float)
    mu = expected_returns.reindex(tickers).to_numpy(float)
    t_obs = len(matrix)

    # x = [w(n), alpha, u(t)]
    c = np.zeros(n + 1 + t_obs)
    c[n] = 1.0
    c[n + 1 :] = 1.0 / ((1.0 - level) * t_obs)

    a_ub = _cvar_constraint_matrix(matrix)

    a_eq = sparse.csc_matrix(
        (
            np.concatenate([np.ones(n), mu]),
            (
                np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)]),
                np.tile(np.arange(n), 2),
            ),
        ),
        shape=(2, n + 1 + t_obs),
    )
    b_eq = np.array([1.0, float(target_return)])

    bounds = [(0.0, float(max_weight))] * n + [(None, None)] + [(0.0, None)] * t_obs

    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=np.zeros(t_obs),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return None

    weights = pd.Series(res.x[:n], index=tickers)
    cvar_value = float(res.fun)
    return weights, cvar_value


def cvar_frontier(
    expected_returns: pd.Series,
    returns: pd.DataFrame,
    max_weight: float,
    level: float = 0.95,
    points: int = 35,
) -> pd.DataFrame:
    tickers = list(returns.columns)
    validate_weight_cap(len(tickers), max_weight)

    # Determine the feasible expected-return range.
    max_w = _max_return_weights(
        expected_returns.reindex(tickers),
        max_weight=max_weight,
    )
    max_ret = float(
        max_w.to_numpy() @ expected_returns.reindex(tickers).to_numpy(float)
    )

    # Start at the unconstrained min-CVaR portfolio's expected return.
    from .optimization import minimum_cvar

    min_cvar = minimum_cvar(
        returns[tickers],
        level=level,
        max_weight=max_weight,
    )
    if not min_cvar.success:
        return pd.DataFrame()

    min_ret = float(
        min_cvar.weights.reindex(tickers).to_numpy(float)
        @ expected_returns.reindex(tickers).to_numpy(float)
    )

    targets = (
        np.array([min_ret])
        if max_ret <= min_ret + 1e-12
        else np.linspace(min_ret, max_ret, int(points))
    )

    rows = []
    for target in targets:
        solved = minimum_cvar_for_target_return(
            expected_returns=expected_returns.reindex(tickers),
            returns=returns[tickers],
            target_return=float(target),
            level=level,
            max_weight=max_weight,
        )
        if solved is None:
            continue
        weights, cvar_value = solved
        row = {
            "Expected Return": float(target),
            "CVaR": float(cvar_value),
        }
        for ticker in tickers:
            row[f"w_{ticker}"] = float(weights[ticker])
        rows.append(row)

    return pd.DataFrame(rows)


def named_cvar_points(
    portfolios: dict[str, pd.Series],
    expected_returns: pd.Series,
    returns: pd.DataFrame,
    level: float = 0.95,
) -> pd.DataFrame:
    tickers = list(returns.columns)
    mu = expected_returns.reindex(tickers).to_numpy(float)
    matrix = returns[tickers].to_numpy(float)

    rows = []
    for name, weights in portfolios.items():
        w = weights.reindex(tickers).fillna(0.0).to_numpy(float)
        if float(w.sum()) <= 0:
            continue
        w = w / float(w.sum())
        rows.append(
            {
                "Portfolio": name,
                "Expected Return": float(w @ mu),
                "CVaR": _empirical_cvar_from_weights(w, matrix, level),
            }
        )

    return pd.DataFrame(rows)
