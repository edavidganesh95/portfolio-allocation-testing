import numpy as np
import pandas as pd

from src.metrics import (
    annualized_return,
    buy_and_hold_portfolio_returns,
    buy_and_hold_weight_path,
    historical_cvar,
    max_drawdown,
    portfolio_returns,
)


def test_annualized_return_constant_monthly():
    r = pd.Series([0.01] * 12)
    expected = (1.01**12) - 1
    assert np.isclose(annualized_return(r, 12), expected)


def test_max_drawdown_simple_path():
    r = pd.Series([0.10, -0.20, 0.10])
    # wealth: 1.10 -> 0.88 -> 0.968, peak 1.10, trough drawdown -20%
    assert np.isclose(max_drawdown(r), -0.20)


def test_cvar_is_nonnegative_loss_number():
    r = pd.Series([-0.10, -0.05, 0.0, 0.02, 0.03, 0.04] * 10)
    assert historical_cvar(r, 0.95) >= 0


def test_portfolio_returns_normalizes_weights():
    r = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]})
    w = pd.Series({"A": 1.0, "B": 1.0})
    out = portfolio_returns(r, w)
    assert np.allclose(out.values, [0.05, 0.05])


def test_multiple_portfolio_constructions_can_be_summarized():
    from src.metrics import summary_table

    r = pd.DataFrame(
        {
            "A": [0.02, -0.01, 0.03, 0.01, -0.02, 0.015],
            "B": [0.01, 0.00, 0.02, -0.01, 0.01, 0.005],
        }
    )
    p1 = portfolio_returns(r, pd.Series({"A": 0.5, "B": 0.5}))
    p2 = portfolio_returns(r, pd.Series({"A": 0.2, "B": 0.8}))
    stats = summary_table(
        pd.DataFrame({"Equal Weight": p1, "Alternative": p2}),
        frequency="monthly",
    )

    assert list(stats.index) == ["Equal Weight", "Alternative"]
    assert "CAGR" in stats.columns
    assert "Volatility" in stats.columns
    assert "Max Drawdown" in stats.columns


def test_buy_and_hold_returns_allow_weights_to_drift():
    r = pd.DataFrame({"A": [0.10, 0.10], "B": [0.00, 0.00]})
    w = pd.Series({"A": 0.5, "B": 0.5})
    out = buy_and_hold_portfolio_returns(r, w)
    # First month is 5%; after A outperforms its weight rises above 50%, so
    # the second-month portfolio return must exceed the constant-weight 5%.
    assert np.isclose(out.iloc[0], 0.05)
    assert out.iloc[1] > 0.05


def test_buy_and_hold_weight_path_drifts_toward_outperformer():
    r = pd.DataFrame({"A": [0.10, 0.10], "B": [0.00, 0.00]})
    w = pd.Series({"A": 0.5, "B": 0.5})
    path = buy_and_hold_weight_path(r, w)
    assert path.iloc[-1]["A"] > 0.5
    assert np.isclose(path.iloc[-1].sum(), 1.0)
