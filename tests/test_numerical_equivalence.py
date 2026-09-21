"""Locks in the guarantee behind the performance work.

Two hot paths were rewritten for speed — the CVaR linear programs now hand
HiGHS a sparse constraint matrix, and the walk-forward holding block compounds
with a cumulative product instead of a per-month Python loop. Neither is
allowed to move a published number, so both are checked here against a
straightforward reference implementation of the original formulation.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog

from src.expected_returns import historical_cagr
from src.frontier import minimum_cvar_for_target_return
from src.optimization import minimum_cvar
from src.walkforward import _normalize_weights, _run_holding_block

# ----------------------------------------------------------------------
# Reference implementations (dense LP, per-month loop)
# ----------------------------------------------------------------------


def dense_minimum_cvar(returns, level, max_weight):
    tickers = list(returns.columns)
    matrix = returns.to_numpy(float)
    t_obs, n = matrix.shape

    cost = np.zeros(n + 1 + t_obs)
    cost[n] = 1.0
    cost[n + 1 :] = 1.0 / ((1.0 - level) * t_obs)

    a_ub = np.zeros((t_obs, n + 1 + t_obs))
    a_ub[:, :n] = -matrix
    a_ub[:, n] = -1.0
    for i in range(t_obs):
        a_ub[i, n + 1 + i] = -1.0

    a_eq = np.zeros((1, n + 1 + t_obs))
    a_eq[0, :n] = 1.0

    result = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.zeros(t_obs),
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=[(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * t_obs,
        method="highs",
    )
    return pd.Series(result.x[:n], index=tickers), float(result.fun)


def dense_minimum_cvar_for_target(expected, returns, target, level, max_weight):
    tickers = list(returns.columns)
    matrix = returns[tickers].to_numpy(float)
    mu = expected.reindex(tickers).to_numpy(float)
    t_obs, n = matrix.shape

    cost = np.zeros(n + 1 + t_obs)
    cost[n] = 1.0
    cost[n + 1 :] = 1.0 / ((1.0 - level) * t_obs)

    a_ub = np.zeros((t_obs, n + 1 + t_obs))
    a_ub[:, :n] = -matrix
    a_ub[:, n] = -1.0
    for i in range(t_obs):
        a_ub[i, n + 1 + i] = -1.0

    a_eq = np.zeros((2, n + 1 + t_obs))
    a_eq[0, :n] = 1.0
    a_eq[1, :n] = mu

    result = linprog(
        cost,
        A_ub=a_ub,
        b_ub=np.zeros(t_obs),
        A_eq=a_eq,
        b_eq=np.array([1.0, float(target)]),
        bounds=[(0.0, max_weight)] * n + [(None, None)] + [(0.0, None)] * t_obs,
        method="highs",
    )
    if not result.success:
        return None
    return pd.Series(result.x[:n], index=tickers), float(result.fun)


def loop_holding_block(test_returns, target_weights):
    columns = list(test_returns.columns)
    target = _normalize_weights(target_weights, columns)
    asset_values = target.astype(float).copy()
    monthly = []
    current = 1.0
    wealth_path = []
    for _, row in test_returns.iterrows():
        asset_values = asset_values * (1.0 + row.reindex(columns).astype(float))
        nxt = float(asset_values.sum())
        monthly.append(nxt / current - 1.0)
        wealth_path.append(nxt)
        current = nxt
    end_weights = target.copy() if current <= 0 else asset_values / current
    return (
        pd.Series(monthly, index=test_returns.index, dtype=float),
        end_weights,
        pd.Series(wealth_path, index=test_returns.index, dtype=float),
    )


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def _returns(rng, n_assets, n_obs):
    tickers = [f"A{i}" for i in range(n_assets)]
    index = pd.date_range("2012-01-31", periods=n_obs, freq="ME")
    return pd.DataFrame(
        rng.normal(0.006, 0.042, (n_obs, n_assets)), index=index, columns=tickers
    )


@pytest.mark.parametrize("level", [0.90, 0.95, 0.99])
@pytest.mark.parametrize("max_weight", [0.4, 0.7, 1.0])
def test_sparse_minimum_cvar_matches_the_dense_program(level, max_weight):
    rng = np.random.default_rng(hash((level, max_weight)) % 2**32)
    returns = _returns(rng, 6, 150)

    sparse_result = minimum_cvar(returns, level, max_weight)
    dense_weights, _dense_objective = dense_minimum_cvar(returns, level, max_weight)

    assert sparse_result.success
    np.testing.assert_allclose(
        sparse_result.weights.to_numpy(), dense_weights.to_numpy(), atol=1e-12
    )


def test_sparse_cvar_target_program_matches_the_dense_program():
    rng = np.random.default_rng(77)
    returns = _returns(rng, 5, 120)
    raw = historical_cagr(returns, "monthly")
    mu = 0.5 * raw + 0.5 * raw.mean()  # a half-shrunk return vector

    for target in np.linspace(float(mu.min()), float(mu.max()), 5):
        sparse_solution = minimum_cvar_for_target_return(
            mu, returns, float(target), 0.95, 0.7
        )
        dense_solution = dense_minimum_cvar_for_target(
            mu, returns, float(target), 0.95, 0.7
        )
        assert (sparse_solution is None) == (dense_solution is None)
        if sparse_solution is None:
            continue
        np.testing.assert_allclose(
            sparse_solution[0].to_numpy(), dense_solution[0].to_numpy(), atol=1e-12
        )
        assert sparse_solution[1] == pytest.approx(dense_solution[1], abs=1e-12)


def test_minimum_cvar_is_deterministic():
    rng = np.random.default_rng(5)
    returns = _returns(rng, 5, 90)
    first = minimum_cvar(returns, 0.95, 0.7).weights
    second = minimum_cvar(returns, 0.95, 0.7).weights
    pd.testing.assert_series_equal(first, second)


@pytest.mark.parametrize("months", [1, 3, 12])
def test_vectorised_holding_block_matches_the_loop(months):
    rng = np.random.default_rng(months)
    tickers = [f"A{i}" for i in range(5)]
    index = pd.date_range("2015-01-31", periods=months, freq="ME")
    test = pd.DataFrame(
        rng.normal(0.006, 0.05, (months, 5)), index=index, columns=tickers
    )
    target = pd.Series(rng.dirichlet(np.ones(5)), index=tickers)

    fast = _run_holding_block(test, target)
    slow = loop_holding_block(test, target)

    pd.testing.assert_series_equal(fast[0], slow[0], atol=1e-14, check_names=False)
    pd.testing.assert_series_equal(fast[1], slow[1], atol=1e-14, check_names=False)
    pd.testing.assert_series_equal(fast[2], slow[2], atol=1e-14, check_names=False)


def test_holding_block_end_weights_reflect_drift():
    tickers = ["A", "B"]
    index = pd.date_range("2015-01-31", periods=1, freq="ME")
    test = pd.DataFrame([[1.0, 0.0]], index=index, columns=tickers)
    target = pd.Series([0.5, 0.5], index=tickers)

    _series, end_weights, wealth_path = _run_holding_block(test, target)
    assert end_weights["A"] == pytest.approx(2 / 3)
    assert end_weights["B"] == pytest.approx(1 / 3)
    assert wealth_path.iloc[-1] == pytest.approx(1.5)
