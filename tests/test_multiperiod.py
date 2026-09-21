import numpy as np
import pandas as pd

from src.multiperiod import (
    construction_weights_by_lookback,
    portfolio_metrics_by_lookback,
)


def _returns():
    idx = pd.date_range("2018-01-31", periods=96, freq="ME")
    rng = np.random.default_rng(7)
    market = rng.normal(0.007, 0.035, len(idx))
    data = {
        "A": market + rng.normal(0.001, 0.015, len(idx)),
        "B": 0.8 * market + rng.normal(0.001, 0.018, len(idx)),
        "C": 0.6 * market + rng.normal(0.002, 0.020, len(idx)),
        "D": 0.5 * market + rng.normal(0.0005, 0.017, len(idx)),
    }
    return pd.DataFrame(data, index=idx), pd.Series(market, index=idx, name="MKT")


def test_diversified_rule_refits_on_3y_5y_and_full_but_never_1y():
    returns, benchmark = _returns()
    weights, status = construction_weights_by_lookback(
        returns,
        benchmark,
        market_return_prior=0.08,
        risk_free_rate=0.0,
        frequency="monthly",
        shrinkage=0.5,
        max_weight=0.50,
        cvar_level=0.95,
        diversified_max_weight=0.50,
        min_effective_assets=3.0,
        max_risk_share=0.50,
    )
    diversified = weights.xs("Diversified Maximum Sharpe", level="Method")
    assert list(diversified.index) == ["3Y", "5Y", "Full"]
    assert np.allclose(diversified.sum(axis=1), 1.0, atol=1e-6)
    assert "1Y" not in set(status.index.get_level_values("Lookback"))
    assert status["Success"].all()


def test_portfolio_metrics_include_1y_3y_5y_and_full():
    returns, _ = _returns()
    portfolios = {
        "P1": pd.Series([0.4, 0.3, 0.2, 0.1], index=returns.columns),
        "P2": pd.Series([0.25] * 4, index=returns.columns),
    }
    metrics = portfolio_metrics_by_lookback(
        returns,
        portfolios,
        frequency="monthly",
        risk_free_rate=0.0,
        cvar_level=0.95,
    )
    assert set(metrics.index.get_level_values("Lookback")) == {"1Y", "3Y", "5Y", "Full"}
    assert set(metrics.index.get_level_values("Portfolio")) == {"P1", "P2"}


def test_all_construction_methods_refit_across_3y_5y_full():
    from src.multiperiod import REFIT_METHOD_ORDER

    returns, benchmark = _returns()
    weights, status = construction_weights_by_lookback(
        returns,
        benchmark,
        market_return_prior=0.08,
        risk_free_rate=0.0,
        frequency="monthly",
        shrinkage=0.5,
        max_weight=0.70,
        cvar_level=0.95,
        diversified_max_weight=0.50,
        min_effective_assets=3.0,
        max_risk_share=0.50,
    )

    assert set(weights.index.get_level_values("Lookback")) == {"3Y", "5Y", "Full"}
    assert set(weights.index.get_level_values("Method")) == set(REFIT_METHOD_ORDER)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-6)
    assert status["Success"].all()


def test_lookback_weight_stability_reports_turnover_vs_full():
    from src.multiperiod import lookback_weight_stability

    idx = pd.MultiIndex.from_tuples(
        [
            ("3Y", "P1"),
            ("5Y", "P1"),
            ("Full", "P1"),
        ],
        names=["Lookback", "Method"],
    )
    weights = pd.DataFrame(
        [
            [0.5, 0.3, 0.2],
            [0.4, 0.4, 0.2],
            [0.4, 0.3, 0.3],
        ],
        index=idx,
        columns=["A", "B", "C"],
    )
    stability = lookback_weight_stability(weights)
    assert np.isclose(stability.loc["P1", "Capital Change 3Y vs Full"], 0.1)
    assert np.isclose(stability.loc["P1", "Capital Change 5Y vs Full"], 0.1)
    assert stability.loc["P1", "Actual ETFs Full"] == 3
