import numpy as np
import pandas as pd

from src.frontier import (
    cvar_frontier,
    cvar_opportunity_set,
    efficient_frontier,
    named_portfolio_points,
    return_diversification_frontier,
    simulate_feasible_portfolios,
    validate_weight_cap,
)


def _inputs():
    tickers = ["A", "B", "C", "D"]
    mu = pd.Series(
        [0.08, 0.06, 0.10, 0.07],
        index=tickers,
    )
    cov = pd.DataFrame(
        [
            [0.040, 0.010, 0.012, 0.008],
            [0.010, 0.022, 0.009, 0.006],
            [0.012, 0.009, 0.055, 0.010],
            [0.008, 0.006, 0.010, 0.028],
        ],
        index=tickers,
        columns=tickers,
    )
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    x = np.linspace(0, 18, len(idx))
    returns = pd.DataFrame(
        {
            "A": 0.006 + 0.030 * np.sin(x),
            "B": 0.004 + 0.018 * np.cos(x * 0.8),
            "C": 0.008 + 0.040 * np.sin(x * 1.1 + 0.2),
            "D": 0.005 + 0.022 * np.cos(x * 1.3),
        },
        index=idx,
    )
    return mu, cov, returns


def test_simulated_portfolios_respect_constraints():
    mu, cov, _ = _inputs()
    cloud = simulate_feasible_portfolios(
        mu,
        cov,
        max_weight=0.60,
        simulations=1000,
        seed=42,
    )
    weights = cloud[[c for c in cloud.columns if c.startswith("w_")]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert float(weights.max().max()) <= 0.600001
    assert len(cloud) == 1000


def test_efficient_frontier_is_nonempty_and_feasible():
    mu, cov, _ = _inputs()
    frontier = efficient_frontier(
        mu,
        cov,
        max_weight=0.60,
        points=25,
    )
    assert len(frontier) >= 10
    assert (frontier["Volatility"] > 0).all()
    assert frontier["Expected Return"].iloc[-1] >= frontier["Expected Return"].iloc[0]


def test_return_diversification_frontier_is_feasible_and_trades_return_for_dr():
    mu, cov, _ = _inputs()
    frontier = return_diversification_frontier(
        mu,
        cov,
        max_weight=0.60,
        points=25,
    )
    assert len(frontier) >= 10
    assert (frontier["Diversification Ratio"] > 1.0).all()
    assert frontier["Expected Return"].is_monotonic_increasing
    assert (
        frontier["Diversification Ratio"].iloc[-1]
        <= frontier["Diversification Ratio"].iloc[0] + 1e-8
    )
    weights = frontier[[c for c in frontier.columns if c.startswith("w_")]]
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert float(weights.max().max()) <= 0.600001
    assert (
        frontier["Expected Return"] + 1e-8 >= frontier["Minimum Expected Return"]
    ).all()


def test_named_points_returns_coordinates():
    mu, cov, _ = _inputs()
    portfolios = {
        "P1": pd.Series({"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}),
        "P2": pd.Series({"A": 0.5, "B": 0.2, "C": 0.2, "D": 0.1}),
    }
    points = named_portfolio_points(portfolios, mu, cov)
    assert list(points["Portfolio"]) == ["P1", "P2"]
    assert {"Expected Return", "Volatility", "Sharpe"}.issubset(points.columns)


def test_cvar_cloud_and_frontier():
    mu, cov, returns = _inputs()
    cloud = simulate_feasible_portfolios(
        mu,
        cov,
        max_weight=0.60,
        simulations=500,
        seed=1,
    )
    cloud = cvar_opportunity_set(cloud, returns, level=0.95)
    frontier = cvar_frontier(
        mu,
        returns,
        max_weight=0.60,
        level=0.95,
        points=20,
    )
    assert "CVaR" in cloud.columns
    assert (cloud["CVaR"] >= 0).all()
    assert not frontier.empty
    assert (frontier["CVaR"] >= 0).all()


def test_infeasible_weight_cap_is_rejected():
    try:
        validate_weight_cap(3, 0.30)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected infeasible cap to raise ValueError")
