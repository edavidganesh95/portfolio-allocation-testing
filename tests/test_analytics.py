import numpy as np
import pandas as pd
import pytest

from src import analytics
from src.metrics import drawdown_series
from src.risk import ledoit_wolf_covariance


@pytest.fixture
def returns():
    index = pd.date_range("2018-01-31", periods=72, freq="ME")
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        rng.normal(0.007, 0.04, (72, 4)),
        index=index,
        columns=["AAA", "BBB", "CCC", "DDD"],
    )
    frame.index.name = "Date"
    return frame


def test_growth_frame_is_tidy_and_compounds(returns):
    tidy = analytics.growth_frame(returns)
    assert list(tidy.columns) == ["Date", "Series", "Growth"]
    assert len(tidy) == returns.size

    last = tidy[(tidy["Series"] == "AAA") & (tidy["Date"] == returns.index[-1])]
    expected = float((1 + returns["AAA"]).prod())
    assert last["Growth"].iloc[0] == pytest.approx(expected)


def test_growth_frame_honours_the_initial_value(returns):
    tidy = analytics.growth_frame(returns, initial=100.0)
    first = tidy[(tidy["Series"] == "AAA")].iloc[0]["Growth"]
    assert first == pytest.approx(100.0 * (1 + returns["AAA"].iloc[0]))


def test_drawdown_frame_matches_the_engine(returns):
    tidy = analytics.drawdown_frame(returns)
    engine = drawdown_series(returns["BBB"])
    got = tidy[tidy["Series"] == "BBB"].set_index("Date")["Drawdown"]
    pd.testing.assert_series_equal(
        got, engine.rename("Drawdown").rename_axis("Date"), check_freq=False
    )
    assert tidy["Drawdown"].max() <= 1e-12


def test_rolling_annual_return_window(returns):
    tidy = analytics.rolling_annual_return(returns, 12)
    per_series = len(returns) - 12 + 1
    assert len(tidy) == per_series * returns.shape[1]

    first = tidy[tidy["Series"] == "CCC"].iloc[0]
    expected = float((1 + returns["CCC"].iloc[:12]).prod() - 1)
    assert first["Rolling Return"] == pytest.approx(expected)


def test_rolling_annual_return_handles_short_samples(returns):
    assert analytics.rolling_annual_return(returns.head(5), 12).empty


def test_average_correlation_excludes_self(returns):
    corr = returns.corr()
    average = analytics.average_correlation(corr)
    expected = (corr["AAA"].sum() - 1.0) / (len(corr) - 1)
    assert average["AAA"] == pytest.approx(expected)
    assert average.is_monotonic_increasing


def test_cluster_order_is_a_permutation(returns):
    corr = returns.corr()
    order = analytics.cluster_order(corr)
    assert sorted(order) == sorted(corr.columns)


def test_cluster_order_passes_through_tiny_matrices(returns):
    corr = returns[["AAA", "BBB"]].corr()
    assert analytics.cluster_order(corr) == ["AAA", "BBB"]


def test_risk_contribution_frame_sums_to_one_per_portfolio(returns):
    cov = ledoit_wolf_covariance(returns, "monthly")
    portfolios = {
        "Equal Weight": pd.Series(0.25, index=returns.columns),
        "Concentrated": pd.Series([0.7, 0.1, 0.1, 0.1], index=returns.columns),
    }
    frame = analytics.risk_contribution_frame(portfolios, cov)
    totals = frame.groupby("Portfolio")["Risk Share"].sum()
    assert totals.round(9).eq(1.0).all()


def test_frontier_composition_strips_the_weight_prefix():
    frontier = pd.DataFrame(
        {
            "Volatility": [0.10, 0.12],
            "Expected Return": [0.05, 0.07],
            "w_AAA": [0.6, 0.4],
            "w_BBB": [0.4, 0.6],
        }
    )
    tidy = analytics.frontier_composition(frontier)
    assert set(tidy["Ticker"]) == {"AAA", "BBB"}
    assert len(tidy) == 4


def test_frontier_composition_handles_empty_input():
    assert analytics.frontier_composition(pd.DataFrame()).empty


def test_relative_wealth_is_one_against_itself(returns):
    wealth = (1 + returns).cumprod()
    tidy = analytics.relative_wealth(wealth, "AAA")
    assert "AAA" not in set(tidy["Series"])

    row = tidy[(tidy["Series"] == "BBB")].iloc[-1]
    expected = float(wealth["BBB"].iloc[-1] / wealth["AAA"].iloc[-1])
    assert row["Relative Wealth"] == pytest.approx(expected)


def test_relative_wealth_requires_the_benchmark(returns):
    wealth = (1 + returns).cumprod()
    assert analytics.relative_wealth(wealth, "ZZZ").empty


def test_histogram_frame_shares_are_bounded_and_binned():
    rng = np.random.default_rng(3)
    samples = {
        "A": rng.normal(100_000, 20_000, 5_000),
        "B": rng.normal(120_000, 25_000, 5_000),
    }
    hist = analytics.histogram_frame(samples, bins=40)
    assert set(hist["Series"]) == {"A", "B"}
    assert len(hist) == 80

    totals = hist.groupby("Series")["Share"].sum()
    # The plotted range covers the central 99%, so a little mass is excluded.
    assert (totals > 0.95).all()
    assert (totals <= 1.0 + 1e-9).all()


def test_histogram_frame_excludes_rather_than_piles_up_outliers():
    """Extreme draws must not create a false spike in the last bin."""
    rng = np.random.default_rng(11)
    values = np.concatenate(
        [rng.normal(100.0, 5.0, 2_000), np.array([5_000.0, 9_000.0])]
    )
    hist = analytics.histogram_frame({"A": values}, bins=30)

    # The axis stays on the body of the distribution ...
    assert hist["Value"].max() < 200.0
    # ... and the top bin holds no more than its neighbours.
    tail = hist.sort_values("Value")["Share"].to_numpy()
    assert tail[-1] <= tail[:-1].max()


def test_histogram_frame_falls_back_when_the_range_is_degenerate():
    values = np.concatenate([np.zeros(1_000), np.array([10.0])])
    hist = analytics.histogram_frame({"A": values}, bins=20)
    assert not hist.empty
    assert hist["Value"].max() == pytest.approx(9.75)


def test_histogram_frame_handles_empty_input():
    assert analytics.histogram_frame({}).empty


def test_percentile_markers_returns_requested_percentiles():
    samples = {"A": np.arange(0.0, 101.0)}
    markers = analytics.percentile_markers(samples, percentiles=(5.0, 50.0))
    assert set(markers["Percentile"]) == {"P5", "P50"}
    median = markers[markers["Percentile"] == "P50"]["Value"].iloc[0]
    assert median == pytest.approx(50.0)


def test_concentration_reports_effective_holdings():
    stats = analytics.concentration(pd.Series([0.25, 0.25, 0.25, 0.25]))
    assert stats["HHI"] == pytest.approx(0.25)
    assert stats["Effective Holdings"] == pytest.approx(4.0)
    assert stats["Largest Holding"] == pytest.approx(0.25)


def test_concentration_normalises_unscaled_weights():
    stats = analytics.concentration(pd.Series([2.0, 2.0]))
    assert stats["HHI"] == pytest.approx(0.5)
