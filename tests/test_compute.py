"""The cache layer must be a pure wrapper: same inputs, same numbers."""

import numpy as np
import pandas as pd
import pytest

from src import compute
from src.bootstrap import BootstrapSettings, run_bootstrap_comparison
from src.expected_returns import market_anchored_expected_returns
from src.metrics import summary_table
from src.optimization import minimum_variance
from src.risk import ledoit_wolf_covariance

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


@pytest.fixture(scope="module")
def returns():
    index = pd.date_range("2015-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        rng.normal(0.007, 0.04, (120, len(TICKERS))), index=index, columns=TICKERS
    )
    frame.index.name = "Date"
    return frame


@pytest.fixture(scope="module")
def benchmark_returns(returns):
    series = (0.65 * returns["AAA"] + 0.35 * returns["BBB"]).rename("MKT")
    return series


def test_cached_expected_returns_matches_the_engine(returns, benchmark_returns):
    pd.testing.assert_series_equal(
        compute.cached_expected_returns(
            returns, benchmark_returns, 0.08, 0.0, "monthly", 0.5
        ),
        market_anchored_expected_returns(
            returns,
            benchmark_returns,
            market_return_prior=0.08,
            risk_free_rate=0.0,
            frequency="monthly",
            shrinkage=0.5,
        ),
    )


def test_cached_covariance_matches_the_engine(returns):
    pd.testing.assert_frame_equal(
        compute.cached_covariance(returns, "monthly"),
        ledoit_wolf_covariance(returns, frequency="monthly"),
    )


def test_cached_summary_table_matches_the_engine(returns):
    pd.testing.assert_frame_equal(
        compute.cached_summary_table(returns, "monthly", 0.0, 0.95),
        summary_table(
            returns, frequency="monthly", risk_free_rate=0.0, cvar_level=0.95
        ),
    )


def test_cached_construction_preserves_presentation_order(returns, benchmark_returns):
    weights, status = compute.cached_construction(
        returns, benchmark_returns, 0.08, "monthly", 0.5, 0.7, 0.0, 0.95
    )
    assert list(weights.index) == [
        name for name in compute.CONSTRUCTION_ORDER if name in weights.index
    ]
    assert list(weights.columns) == TICKERS
    assert set(status["Method"]) == set(compute.CONSTRUCTION_ORDER)


def test_cached_construction_weights_are_investable(returns, benchmark_returns):
    weights, _status = compute.cached_construction(
        returns, benchmark_returns, 0.08, "monthly", 0.5, 0.7, 0.0, 0.95
    )
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert (weights >= -1e-9).all().all()
    assert (weights <= 0.7 + 1e-6).all().all()


def test_cached_construction_matches_a_direct_solve(returns, benchmark_returns):
    weights, _status = compute.cached_construction(
        returns, benchmark_returns, 0.08, "monthly", 0.5, 0.7, 0.0, 0.95
    )
    direct = minimum_variance(
        ledoit_wolf_covariance(returns, "monthly"), max_weight=0.7
    )
    pd.testing.assert_series_equal(
        weights.loc["Minimum Variance"],
        direct.weights,
        check_names=False,
        atol=1e-10,
    )


def test_weights_frame_round_trips_a_portfolio_dictionary():
    portfolios = {
        "Reference Portfolio": pd.Series({"AAA": 0.7, "BBB": 0.3}),
        "Robust Consensus": pd.Series({"AAA": 0.4, "CCC": 0.6}),
    }
    frame = compute.weights_frame(portfolios)
    assert list(frame.columns) == list(portfolios)
    assert frame.loc["CCC", "Robust Consensus"] == pytest.approx(0.6)
    assert np.isnan(frame.loc["CCC", "Reference Portfolio"])


def test_weights_frame_handles_no_portfolios():
    assert compute.weights_frame({}).empty


def test_cached_bootstrap_summary_matches_the_engine(returns):
    portfolios = {
        "Reference Portfolio": pd.Series(0.25, index=TICKERS),
        "Robust Consensus": pd.Series([0.4, 0.3, 0.2, 0.1], index=TICKERS),
    }
    cached = compute.cached_bootstrap(
        returns, compute.weights_frame(portfolios), 500, 5, 3, 42, 100_000.0
    )
    direct_summary, direct_outputs, _ = run_bootstrap_comparison(
        returns, portfolios, BootstrapSettings(500, 5, 3, 42, 100_000.0)
    )

    pd.testing.assert_frame_equal(cached.summary, direct_summary)
    for name in portfolios:
        np.testing.assert_allclose(
            cached.terminal_wealth[name], direct_outputs[name]["wealth"][:, -1]
        )
        np.testing.assert_allclose(cached.cagr[name], direct_outputs[name]["cagr"])
        np.testing.assert_allclose(
            cached.max_drawdown[name], direct_outputs[name]["max_drawdown"]
        )


def test_cached_bootstrap_keeps_only_what_is_displayed(returns):
    """The full wealth paths are tens of megabytes and are never shown."""
    portfolios = {"Equal Weight": pd.Series(0.25, index=TICKERS)}
    cached = compute.cached_bootstrap(
        returns, compute.weights_frame(portfolios), 500, 5, 3, 42, 100_000.0
    )
    assert set(cached.fans) == {"Equal Weight"}
    assert cached.fans["Equal Weight"].shape == (60, 6)
    assert cached.terminal_wealth["Equal Weight"].shape == (500,)
    assert not hasattr(cached, "wealth")


def test_cached_bootstrap_is_reproducible_for_a_given_seed(returns):
    portfolios = compute.weights_frame({"Equal Weight": pd.Series(0.25, index=TICKERS)})
    first = compute.cached_bootstrap(returns, portfolios, 400, 5, 3, 7, 100_000.0)
    second = compute.cached_bootstrap(returns, portfolios, 400, 5, 3, 7, 100_000.0)
    np.testing.assert_array_equal(
        first.terminal_wealth["Equal Weight"], second.terminal_wealth["Equal Weight"]
    )

    different = compute.cached_bootstrap(returns, portfolios, 400, 5, 3, 8, 100_000.0)
    assert not np.array_equal(
        first.terminal_wealth["Equal Weight"],
        different.terminal_wealth["Equal Weight"],
    )


def test_cached_walk_forward_matches_the_engine(returns, benchmark_returns):
    from src.walkforward import WalkForwardSettings, walk_forward_backtest

    reference = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    cached = compute.cached_walk_forward(
        returns,
        benchmark_returns,
        0.08,
        tuple(TICKERS),
        36,
        36,
        12,
        0.0,
        0.5,
        0.7,
        0.95,
        reference,
    )
    direct = walk_forward_backtest(
        returns,
        optimizer_assets=TICKERS,
        settings=WalkForwardSettings(
            lookback_months=36,
            min_train_months=36,
            holding_months=12,
            risk_free_rate=0.0,
            expected_return_shrinkage=0.5,
            market_return_prior=0.08,
            max_weight=0.7,
            cvar_level=0.95,
        ),
        reference_weights=reference,
        benchmark_returns=benchmark_returns,
    )
    pd.testing.assert_frame_equal(cached.returns, direct.returns)
    pd.testing.assert_frame_equal(cached.wealth, direct.wealth)
    pd.testing.assert_frame_equal(cached.block_outcomes, direct.block_outcomes)
    pd.testing.assert_frame_equal(cached.ending_weights, direct.ending_weights)
