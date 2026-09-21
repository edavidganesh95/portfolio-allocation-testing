import numpy as np
import pandas as pd
import pytest

from src.bootstrap import (
    BootstrapSettings,
    moving_block_indices,
    run_bootstrap_comparison,
    simulation_fan,
)


def _monthly_returns():
    idx = pd.date_range("2015-01-31", periods=120, freq="ME")
    x = np.linspace(0, 16, len(idx))
    return pd.DataFrame(
        {
            "A": 0.007 + 0.025 * np.sin(x),
            "B": 0.005 + 0.014 * np.cos(x * 0.7),
            "C": 0.006 + 0.020 * np.sin(x * 1.2 + 0.3),
        },
        index=idx,
    )


def test_moving_block_indices_are_reproducible():
    a = moving_block_indices(100, 60, 50, 3, 42)
    b = moving_block_indices(100, 60, 50, 3, 42)
    assert np.array_equal(a, b)
    assert a.shape == (50, 60)


def test_blocks_are_contiguous_inside_each_sampled_block():
    idx = moving_block_indices(20, 12, 5, 3, 7)
    for row in idx:
        for start in range(0, 12, 3):
            block = row[start : start + 3]
            if len(block) > 1:
                assert np.all((np.diff(block) % 20) == 1)


def test_bootstrap_comparison_outputs_expected_shapes():
    returns = _monthly_returns()
    portfolios = {
        "Equal": pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}),
        "Defensive": pd.Series({"A": 0.1, "B": 0.8, "C": 0.1}),
    }
    settings = BootstrapSettings(
        simulations=250,
        horizon_years=5,
        block_months=3,
        seed=42,
        initial_wealth=100_000,
    )
    summary, outputs, indices = run_bootstrap_comparison(returns, portfolios, settings)
    assert list(summary.index) == ["Equal", "Defensive"]
    assert "Median Terminal Wealth" in summary.columns
    assert "P(CAGR > 8%)" in summary.columns
    assert outputs["Equal"]["wealth"].shape == (250, 60)
    assert indices.shape == (250, 60)


def test_simulation_fan_shape():
    settings = BootstrapSettings(
        simulations=100,
        horizon_years=5,
        block_months=3,
        seed=1,
    )
    _, outputs, _ = run_bootstrap_comparison(
        _monthly_returns(),
        {"Equal": pd.Series({"A": 1 / 3, "B": 1 / 3, "C": 1 / 3})},
        settings,
    )
    fan = simulation_fan(outputs["Equal"])
    assert list(fan.columns) == ["Month", "5th", "25th", "Median", "75th", "95th"]
    assert len(fan) == 60


def test_bootstrap_uses_buy_and_hold_weight_drift():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    returns = pd.DataFrame({"A": [0.10] * 12, "B": [0.0] * 12}, index=idx)
    settings = BootstrapSettings(
        simulations=1, horizon_years=1, block_months=12, seed=1, initial_wealth=100.0
    )
    summary, outputs, _ = run_bootstrap_comparison(
        returns, {"P": pd.Series({"A": 0.5, "B": 0.5})}, settings
    )
    terminal = outputs["P"]["wealth"][0, -1]
    expected_bh = 100.0 * (0.5 * (1.10**12) + 0.5)
    monthly_reset = 100.0 * (1.05**12)
    assert np.isclose(terminal, expected_bh)
    assert not np.isclose(terminal, monthly_reset)


def _many_portfolios():
    rng = np.random.default_rng(5)
    tickers = ["A", "B", "C"]
    portfolios = {
        f"P{i}": pd.Series(rng.dirichlet(np.ones(3)), index=tickers) for i in range(5)
    }
    portfolios["OnlyA"] = pd.Series({"A": 1.0})  # holds a single asset
    return portfolios


def test_streaming_bootstrap_matches_the_reference_implementation():
    from src.bootstrap import run_bootstrap_reduced

    settings = BootstrapSettings(
        simulations=600, horizon_years=5, block_months=3, seed=11, initial_wealth=5e4
    )
    portfolios = _many_portfolios()
    summary, outputs, _ = run_bootstrap_comparison(
        _monthly_returns(), portfolios, settings
    )
    reduced = run_bootstrap_reduced(_monthly_returns(), portfolios, settings)

    assert list(reduced.summary.index) == list(summary.index)
    pd.testing.assert_frame_equal(reduced.summary, summary, rtol=1e-12, atol=0)
    for name, result in outputs.items():
        pd.testing.assert_frame_equal(
            reduced.fans[name], simulation_fan(result), rtol=1e-12, atol=0
        )
        assert np.allclose(reduced.terminal_wealth[name], result["wealth"][:, -1])
        assert np.allclose(reduced.cagr[name], result["cagr"])
        assert np.allclose(reduced.max_drawdown[name], result["max_drawdown"])


def test_streaming_bootstrap_is_independent_of_the_memory_budget():
    """A tiny budget forces one portfolio per pass; the answer must not change."""
    from src.bootstrap import run_bootstrap_reduced

    settings = BootstrapSettings(
        simulations=400, horizon_years=3, block_months=6, seed=3, initial_wealth=1e5
    )
    portfolios = _many_portfolios()
    roomy = run_bootstrap_reduced(
        _monthly_returns(), portfolios, settings, memory_budget_mb=1_000
    )
    cramped = run_bootstrap_reduced(
        _monthly_returns(), portfolios, settings, memory_budget_mb=0.001
    )

    pd.testing.assert_frame_equal(roomy.summary, cramped.summary, rtol=1e-12, atol=0)
    for name in portfolios:
        pd.testing.assert_frame_equal(roomy.fans[name], cramped.fans[name])
        # Not bit-identical: multiplying by one weight column or several takes a
        # different BLAS path, which can differ in the last place.
        assert np.allclose(
            roomy.terminal_wealth[name],
            cramped.terminal_wealth[name],
            rtol=1e-12,
            atol=0,
        )


def test_streaming_bootstrap_validates_its_inputs():
    from src.bootstrap import run_bootstrap_reduced

    settings = BootstrapSettings(simulations=10, horizon_years=1)
    with pytest.raises(ValueError, match="At least one portfolio"):
        run_bootstrap_reduced(_monthly_returns(), {}, settings)
    with pytest.raises(ValueError, match="No portfolio holdings"):
        run_bootstrap_reduced(
            _monthly_returns(), {"Ghost": pd.Series({"ZZZ": 1.0})}, settings
        )


def test_fan_percentiles_match_individual_percentile_calls():
    rng = np.random.default_rng(2)
    wealth = np.cumprod(1 + rng.normal(0.005, 0.04, (500, 36)), axis=1) * 1e5
    fan = simulation_fan({"wealth": wealth})
    for column, q in [
        ("5th", 5),
        ("25th", 25),
        ("Median", 50),
        ("75th", 75),
        ("95th", 95),
    ]:
        assert np.array_equal(fan[column].to_numpy(), np.percentile(wealth, q, axis=0))
