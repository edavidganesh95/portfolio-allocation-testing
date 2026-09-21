"""Every chart builder must produce a valid, themed Vega-Lite spec.

The app renders ~30 charts per run; a builder that raises only when a
particular section is opened is the kind of defect these tests exist to catch.
"""

import altair as alt
import numpy as np
import pandas as pd
import pytest

from src import analytics, charts, theme

TICKERS = ["AAA", "BBB", "CCC", "DDD"]
PORTFOLIOS = ["Reference Portfolio", "Robust Consensus", "Minimum CVaR"]


@pytest.fixture(scope="module", autouse=True)
def configured():
    charts.configure_altair()


@pytest.fixture(scope="module")
def returns():
    index = pd.date_range("2018-01-31", periods=84, freq="ME")
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        rng.normal(0.007, 0.04, (84, len(TICKERS))), index=index, columns=TICKERS
    )
    frame.index.name = "Date"
    return frame


def spec(chart) -> dict:
    """Compile to a Vega-Lite spec, which is what the browser receives."""
    return chart.to_dict()


def encoded_colors(rendered: dict) -> list[str]:
    found: list[str] = []

    def walk(node):
        if isinstance(node, dict):
            scale = node.get("scale")
            if isinstance(scale, dict) and isinstance(scale.get("range"), list):
                found.extend(str(v) for v in scale["range"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(rendered.get("encoding", {}))
    for layer in rendered.get("layer", []):
        walk(layer.get("encoding", {}))
    return found


def test_configure_altair_lifts_the_row_cap():
    """A 10,000-portfolio cloud must render, not raise MaxRowsError.

    The sidebar offers 10,000 simulated feasible portfolios, which is above
    Altair's default 5,000-row guard.
    """
    cloud = pd.DataFrame(
        {"Expected Return": np.linspace(0, 0.1, 10_000), "Volatility": 0.1}
    )
    chart = (
        alt.Chart(cloud).mark_point().encode(x="Volatility:Q", y="Expected Return:Q")
    )
    rendered = chart.to_dict()
    assert len(rendered["datasets"][rendered["data"]["name"]]) == 10_000

    try:
        alt.data_transformers.enable("default", max_rows=5_000)
        with pytest.raises(alt.MaxRowsError):
            chart.to_dict()
    finally:
        charts.configure_altair()


def test_asset_color_scale_is_sorted_and_stable():
    color = charts.asset_color(["CCC", "AAA", "BBB"])
    rendered = color.to_dict()
    assert rendered["scale"]["domain"] == ["AAA", "BBB", "CCC"]
    assert rendered["scale"]["range"] == theme.asset_range(["AAA", "BBB", "CCC"])


def test_portfolio_color_scale_uses_fixed_identities():
    rendered = charts.portfolio_color(PORTFOLIOS).to_dict()
    assert rendered["scale"]["range"][0] == theme.BRASS
    assert rendered["scale"]["range"][1] == theme.NAVY


def test_growth_chart(returns):
    rendered = spec(charts.growth_chart(analytics.growth_frame(returns), TICKERS))
    assert rendered["mark"]["type"] == "line"
    assert theme.CATEGORICAL[0] in encoded_colors(rendered)


def test_growth_chart_log_scale(returns):
    rendered = spec(
        charts.growth_chart(analytics.growth_frame(returns), TICKERS, log_scale=True)
    )
    assert rendered["encoding"]["y"]["scale"]["type"] == "log"


def test_drawdown_chart(returns):
    rendered = spec(charts.drawdown_chart(analytics.drawdown_frame(returns), TICKERS))
    assert rendered["encoding"]["y"]["axis"]["format"] == "%"


def test_rolling_return_chart(returns):
    rolling = analytics.rolling_annual_return(returns, 12)
    rendered = spec(charts.rolling_return_chart(rolling, TICKERS, "12-month"))
    assert "layer" in rendered


def test_risk_return_scatter(returns):
    from src.metrics import summary_table

    stats = summary_table(returns, "monthly", 0.0, 0.95)
    rendered = spec(charts.risk_return_scatter(stats, "CVaR 95%"))
    assert "layer" in rendered

    points = rendered["layer"][0]
    # Every point is the same size: size is a fixed mark property, not an encoding,
    # so there is no size legend either.
    assert "size" not in points["encoding"]
    assert points["mark"]["size"] > 0
    tooltips = {tip["field"] for tip in points["encoding"]["tooltip"]}
    assert "CVaR 95%" in tooltips  # tail loss stays available on hover


def test_correlation_heatmap_annotates_cells(returns):
    corr = returns.corr()
    order = analytics.cluster_order(corr)
    corr_long = (
        corr.rename_axis("Asset A")
        .reset_index()
        .melt(id_vars="Asset A", var_name="Asset B", value_name="Correlation")
    )
    rendered = spec(charts.correlation_heatmap(corr_long, order))
    marks = {layer["mark"]["type"] for layer in rendered["layer"]}
    assert marks == {"rect", "text"}

    scale = rendered["layer"][0]["encoding"]["color"]["scale"]
    assert scale["domain"] == [-1, 1]
    assert scale["range"] == list(theme.DIVERGING)


def test_correlation_heatmap_without_annotation(returns):
    corr_long = (
        returns.corr()
        .rename_axis("Asset A")
        .reset_index()
        .melt(id_vars="Asset A", var_name="Asset B", value_name="Correlation")
    )
    rendered = spec(charts.correlation_heatmap(corr_long, TICKERS, annotate=False))
    assert rendered["mark"]["type"] == "rect"


def test_average_correlation_chart(returns):
    frame = (
        analytics.average_correlation(returns.corr())
        .rename("Average Correlation")
        .reset_index()
    )
    frame.columns = ["Ticker", "Average Correlation"]
    rendered = spec(charts.average_correlation_chart(frame))
    assert "layer" in rendered


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        [[0.25, 0.25, 0.25, 0.25], [0.4, 0.3, 0.2, 0.1]],
        index=["Equal Weight", "Minimum Variance"],
        columns=TICKERS,
    )


def test_weights_composition_chart_is_normalised():
    rendered = spec(
        charts.weights_composition_chart(
            analytics.weights_long(_weights()), TICKERS, list(_weights().index)
        )
    )
    assert rendered["encoding"]["x"]["stack"] == "normalize"


def test_risk_contribution_chart(returns):
    from src.risk import ledoit_wolf_covariance

    cov = ledoit_wolf_covariance(returns, "monthly")
    frame = analytics.risk_contribution_frame(
        {name: _weights().loc[name] for name in _weights().index}, cov
    )
    rendered = spec(
        charts.risk_contribution_chart(frame, TICKERS, list(_weights().index))
    )
    assert rendered["encoding"]["x"]["stack"] == "normalize"


@pytest.fixture(scope="module")
def frontier_inputs(returns):
    from src.expected_returns import historical_cagr
    from src.frontier import (
        efficient_frontier,
        named_portfolio_points,
        simulate_feasible_portfolios,
    )
    from src.risk import ledoit_wolf_covariance

    raw = historical_cagr(returns, "monthly")
    mu = 0.5 * raw + 0.5 * raw.mean()  # a half-shrunk return vector
    cov = ledoit_wolf_covariance(returns, "monthly")
    cloud = simulate_feasible_portfolios(mu, cov, 0.7, 500, 0.0, 42)
    frontier = efficient_frontier(mu, cov, 0.7, 12)
    named = named_portfolio_points(
        {name: _weights().loc[name] for name in _weights().index}, mu, cov, 0.0
    )
    return mu, cov, cloud, frontier, named


def test_frontier_chart(frontier_inputs):
    _mu, _cov, cloud, frontier, named = frontier_inputs
    rendered = spec(
        charts.frontier_chart(
            cloud[["Expected Return", "Volatility", "Sharpe"]],
            frontier,
            named,
            ["Minimum Variance"],
        )
    )
    assert len(rendered["layer"]) == 4


def test_frontier_composition_chart(frontier_inputs):
    _mu, _cov, _cloud, frontier, _named = frontier_inputs
    frame = analytics.frontier_composition(frontier)
    rendered = spec(charts.frontier_composition_chart(frame, TICKERS))
    assert rendered["mark"]["type"] == "area"


def test_sensitivity_frontier_chart(frontier_inputs):
    _mu, _cov, _cloud, frontier, _named = frontier_inputs
    frame = pd.concat(
        [
            frontier.assign(Estimator="Raw historical"),
            frontier.assign(Estimator="Market prior"),
        ],
        ignore_index=True,
    )
    rendered = spec(charts.sensitivity_frontier_chart(frame))
    assert rendered["encoding"]["color"]["scale"]["range"] == [
        theme.BRASS,
        theme.NAVY,
    ]


def test_return_diversification_frontier_chart(frontier_inputs):
    from src.diversification import (
        correlation_clusters,
        diversification_profile,
    )
    from src.frontier import return_diversification_frontier

    mu, cov, _cloud, _frontier, _named = frontier_inputs
    portfolios = {name: _weights().loc[name] for name in _weights().index}
    corr = cov.copy()
    std = np.sqrt(np.diag(cov))
    corr.loc[:, :] = cov.to_numpy() / np.outer(std, std)
    clusters = correlation_clusters(corr)
    profile = diversification_profile(portfolios, cov, corr, clusters)
    profile["Expected Return"] = [
        float(portfolios[name].reindex(mu.index).fillna(0.0).to_numpy() @ mu.to_numpy())
        for name in profile.index
    ]
    named = profile.reset_index()[
        [
            "Portfolio",
            "Expected Return",
            "Diversification Ratio",
            "Effective ETFs",
            "Effective Risk Bets",
            "Largest Risk Share",
        ]
    ]
    frontier = return_diversification_frontier(mu, cov, 0.7, 12)
    rendered = spec(
        charts.return_diversification_frontier_chart(
            frontier, named, label_names=["Minimum Variance"]
        )
    )
    assert len(rendered["layer"]) >= 3


def test_cvar_frontier_chart(returns, frontier_inputs):
    from src.frontier import cvar_frontier, cvar_opportunity_set, named_cvar_points

    mu, _cov, cloud, _frontier, _named = frontier_inputs
    tail_cloud = cvar_opportunity_set(cloud, returns, 0.95)
    tail_frontier = cvar_frontier(mu, returns, 0.7, 0.95, 8)
    tail_named = named_cvar_points(
        {name: _weights().loc[name] for name in _weights().index}, mu, returns, 0.95
    )
    rendered = spec(
        charts.cvar_frontier_chart(
            tail_cloud[["Expected Return", "CVaR"]], tail_frontier, tail_named, 0.95
        )
    )
    assert len(rendered["layer"]) == 3


@pytest.fixture(scope="module")
def bootstrap(returns):
    from src.bootstrap import (
        BootstrapSettings,
        run_bootstrap_comparison,
        simulation_fan,
    )

    portfolios = {
        "Reference Portfolio": pd.Series(0.25, index=TICKERS),
        "Robust Consensus": pd.Series([0.4, 0.3, 0.2, 0.1], index=TICKERS),
    }
    summary, outputs, _ = run_bootstrap_comparison(
        returns, portfolios, BootstrapSettings(400, 5, 3, 42, 100_000.0)
    )
    return summary, outputs, simulation_fan(outputs["Robust Consensus"])


def test_terminal_wealth_range_chart(bootstrap):
    summary, _outputs, _fan = bootstrap
    rendered = spec(charts.terminal_wealth_range_chart(summary.reset_index(), 10))
    assert len(rendered["layer"]) == 3


def test_fan_chart(bootstrap):
    _summary, _outputs, fan = bootstrap
    rendered = spec(charts.fan_chart(fan, 100_000.0))
    assert len(rendered["layer"]) >= 3


def test_distribution_chart_with_markers(bootstrap):
    _summary, outputs, _fan = bootstrap
    samples = {name: value["wealth"][:, -1] for name, value in outputs.items()}
    hist = analytics.histogram_frame(samples)
    markers = analytics.percentile_markers(samples, (5.0,))
    rendered = spec(charts.distribution_chart(hist, markers, reference=100_000.0))
    assert len(rendered["layer"]) == 3


def test_distribution_chart_without_markers(bootstrap):
    _summary, outputs, _fan = bootstrap
    hist = analytics.histogram_frame({k: v["cagr"] for k, v in outputs.items()})
    rendered = spec(
        charts.distribution_chart(hist, None, value_format=".0%", reference=0.0)
    )
    assert len(rendered["layer"]) == 2


def test_probability_chart(bootstrap):
    summary, _outputs, _fan = bootstrap
    columns = ["P(Loss at Horizon)", "P(CAGR > 6%)", "P(CAGR > 8%)"]
    frame = summary.reset_index()[["Portfolio"] + columns].melt(
        id_vars="Portfolio", var_name="Metric", value_name="Probability"
    )
    rendered = spec(charts.probability_chart(frame))
    assert rendered["encoding"]["color"]["scale"]["range"][0] == theme.NEGATIVE


@pytest.fixture(scope="module")
def walk_forward(returns):
    from src.walkforward import WalkForwardSettings, walk_forward_backtest

    long_index = pd.date_range("2012-01-31", periods=150, freq="ME")
    rng = np.random.default_rng(21)
    frame = pd.DataFrame(
        rng.normal(0.007, 0.04, (150, len(TICKERS))),
        index=long_index,
        columns=TICKERS,
    )
    frame.index.name = "Date"
    reference = pd.Series({"AAA": 0.5, "BBB": 0.3, "CCC": 0.2})
    benchmark = (0.7 * frame["AAA"] + 0.3 * frame["BBB"]).rename("MKT")
    return walk_forward_backtest(
        frame,
        TICKERS,
        WalkForwardSettings(
            lookback_months=36,
            min_train_months=36,
            holding_months=12,
            risk_free_rate=0.0,
            expected_return_shrinkage=0.5,
            market_return_prior=0.08,
            max_weight=0.7,
            cvar_level=0.95,
        ),
        reference,
        benchmark_returns=benchmark,
    )


def test_portfolio_growth_chart(walk_forward):
    names = list(walk_forward.returns.columns)
    rendered = spec(
        charts.portfolio_growth_chart(
            analytics.growth_frame(walk_forward.returns), names
        )
    )
    assert rendered["mark"]["type"] == "line"


def test_portfolio_drawdown_chart(walk_forward):
    names = list(walk_forward.returns.columns)
    rendered = spec(
        charts.portfolio_drawdown_chart(
            analytics.drawdown_frame(walk_forward.returns), names
        )
    )
    assert rendered["encoding"]["y"]["axis"]["format"] == "%"


def test_relative_wealth_chart(walk_forward):
    frame = analytics.relative_wealth(walk_forward.wealth, "Reference Portfolio")
    names = [n for n in walk_forward.returns.columns if n != "Reference Portfolio"]
    rendered = spec(charts.relative_wealth_chart(frame, names, "Reference Portfolio"))
    assert "layer" in rendered


def test_win_rate_chart(walk_forward):
    from src.walkforward import block_win_rates

    frame = block_win_rates(walk_forward, "Reference Portfolio").reset_index()
    frame.columns = ["Portfolio", "Block Win Rate"]
    rendered = spec(charts.win_rate_chart(frame))
    assert len(rendered["layer"]) == 3


def test_walk_forward_drift_summary(walk_forward):
    from src.walkforward import drift_summary

    summary = drift_summary(walk_forward)
    assert "Median Weight Drift" in summary.columns
    assert not summary.empty


def test_weight_evolution_chart(walk_forward):
    view = walk_forward.weights[
        walk_forward.weights["Method"] == "Diversified Maximum Sharpe"
    ]
    rendered = spec(charts.weight_evolution_chart(view, TICKERS))
    assert rendered["encoding"]["y"]["stack"] == "normalize"
