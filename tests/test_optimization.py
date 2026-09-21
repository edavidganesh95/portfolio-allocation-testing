import numpy as np
import pandas as pd

from src.optimization import (
    maximum_diversification,
    maximum_return_to_cvar,
    minimum_cvar,
    minimum_variance,
    risk_parity,
)


def _cov():
    return pd.DataFrame(
        [[0.04, 0.01, 0.005], [0.01, 0.09, 0.01], [0.005, 0.01, 0.06]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )


def test_minimum_variance_weights_sum_to_one():
    res = minimum_variance(_cov(), max_weight=0.80)
    assert res.success
    assert np.isclose(res.weights.sum(), 1.0)
    assert (res.weights >= -1e-9).all()


def test_risk_parity_weights_sum_to_one():
    res = risk_parity(_cov(), max_weight=0.80)
    assert res.success
    assert np.isclose(res.weights.sum(), 1.0)


def test_maximum_diversification_weights_sum_to_one():
    res = maximum_diversification(_cov(), max_weight=0.80)
    assert res.success
    assert np.isclose(res.weights.sum(), 1.0)


def test_minimum_cvar_weights_sum_to_one():
    returns = pd.DataFrame(
        {
            "A": [0.03, -0.08, 0.02, -0.01, 0.04, -0.05, 0.01, 0.02] * 4,
            "B": [0.01, -0.02, 0.01, 0.00, 0.015, -0.01, 0.005, 0.01] * 4,
            "C": [0.04, -0.10, 0.03, -0.03, 0.05, -0.06, 0.02, 0.025] * 4,
        }
    )
    res = minimum_cvar(returns, level=0.95, max_weight=0.80)
    assert res.success
    assert np.isclose(res.weights.sum(), 1.0)
    assert (res.weights >= -1e-9).all()
    assert (res.weights <= 0.800001).all()


def test_maximum_return_to_cvar_weights_sum_to_one():
    returns = pd.DataFrame(
        {
            "A": [0.03, -0.08, 0.02, -0.01, 0.04, -0.05, 0.01, 0.02] * 4,
            "B": [0.01, -0.02, 0.01, 0.00, 0.015, -0.01, 0.005, 0.01] * 4,
            "C": [0.04, -0.10, 0.03, -0.03, 0.05, -0.06, 0.02, 0.025] * 4,
        }
    )
    mu = pd.Series({"A": 0.08, "B": 0.05, "C": 0.10})
    res = maximum_return_to_cvar(mu, returns, level=0.95, max_weight=0.80)
    assert res.success
    assert np.isclose(res.weights.sum(), 1.0)


def test_diversified_maximum_sharpe_respects_guardrails():
    from src.optimization import diversified_maximum_sharpe
    from src.risk import risk_contributions

    tickers = ["A", "B", "C", "D"]
    mu = pd.Series([0.12, 0.10, 0.09, 0.08], index=tickers)
    cov = pd.DataFrame(
        [
            [0.040, 0.012, 0.010, 0.008],
            [0.012, 0.032, 0.009, 0.007],
            [0.010, 0.009, 0.028, 0.006],
            [0.008, 0.007, 0.006, 0.025],
        ],
        index=tickers,
        columns=tickers,
    )
    result = diversified_maximum_sharpe(
        mu,
        cov,
        max_weight=0.50,
        min_effective_assets=3.0,
        max_risk_share=0.50,
    )
    assert result.success, result.message
    w = result.weights
    assert np.isclose(w.sum(), 1.0, atol=1e-7)
    assert w.max() <= 0.50 + 1e-6
    effective = 1.0 / float((w**2).sum())
    assert effective >= 3.0 - 1e-5
    rc = risk_contributions(w, cov)
    shares = rc / float(rc.sum())
    assert float(shares.max()) <= 0.50 + 1e-5
