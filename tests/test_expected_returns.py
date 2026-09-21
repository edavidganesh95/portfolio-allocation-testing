import numpy as np
import pandas as pd
import pytest

from src.expected_returns import (
    historical_cagr,
    market_anchored_expected_returns,
    market_betas,
    market_prior_diagnostics,
    market_prior_returns,
)


def _sample():
    idx = pd.date_range("2018-01-31", periods=72, freq="ME")
    rng = np.random.default_rng(123)
    market = pd.Series(rng.normal(0.007, 0.035, len(idx)), index=idx, name="MKT")
    returns = pd.DataFrame(
        {
            "LOW": 0.4 * market + rng.normal(0.003, 0.01, len(idx)),
            "MKTLIKE": market,
            "HIGH": 1.4 * market + rng.normal(0.001, 0.012, len(idx)),
        },
        index=idx,
    )
    return returns, market


def test_market_like_asset_has_beta_one():
    returns, market = _sample()
    beta = market_betas(returns, market)
    assert beta["MKTLIKE"] == pytest.approx(1.0, abs=1e-12)
    assert beta["LOW"] < beta["MKTLIKE"] < beta["HIGH"]


def test_market_prior_uses_user_assumption_not_benchmark_realized_cagr():
    returns, market = _sample()
    prior = market_prior_returns(
        returns,
        market,
        market_return_prior=0.08,
        risk_free_rate=0.02,
    )
    assert prior["MKTLIKE"] == pytest.approx(0.08, abs=1e-12)


def test_shrinkage_endpoints_are_transparent():
    returns, market = _sample()
    raw = historical_cagr(returns, "monthly")
    prior = market_prior_returns(returns, market, 0.08, 0.01)

    zero = market_anchored_expected_returns(
        returns, market, 0.08, 0.01, "monthly", shrinkage=0.0
    )
    one = market_anchored_expected_returns(
        returns, market, 0.08, 0.01, "monthly", shrinkage=1.0
    )
    pd.testing.assert_series_equal(zero, raw.rename("Expected Return"))
    pd.testing.assert_series_equal(one, prior.rename("Expected Return"))


def test_existing_asset_expectations_do_not_depend_on_extra_candidate():
    returns, market = _sample()
    base = returns[["LOW", "MKTLIKE"]]
    expanded = returns[["LOW", "MKTLIKE", "HIGH"]]

    base_mu = market_anchored_expected_returns(
        base, market, 0.08, 0.0, "monthly", shrinkage=0.5
    )
    expanded_mu = market_anchored_expected_returns(
        expanded, market, 0.08, 0.0, "monthly", shrinkage=0.5
    )
    pd.testing.assert_series_equal(base_mu, expanded_mu.loc[base.columns])


def test_diagnostics_show_all_expected_return_components():
    returns, market = _sample()
    table = market_prior_diagnostics(
        returns,
        market,
        market_return_prior=0.08,
        risk_free_rate=0.0,
        frequency="monthly",
        shrinkage=0.5,
    )
    assert list(table.columns) == [
        "Historical CAGR",
        "Market Beta",
        "Market Prior",
        "Blended Expected Return",
    ]
    assert list(table.index) == list(returns.columns)


def test_insufficient_benchmark_overlap_is_rejected():
    returns, market = _sample()
    too_short = market.iloc[-12:]
    with pytest.raises(ValueError, match="overlapping return observations"):
        market_betas(returns, too_short, min_observations=24)
