"""Altair chart library for Portfolio Allocation Testing.

Every on-screen visual is built here so encoding choices, colour identity and
interaction behaviour stay consistent across the six research sections. The
matplotlib mirrors used by the PDF export live in ``exporting`` and read the
same tidy frames, which is what keeps screen and print in agreement.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from . import theme

PCT_AXIS = alt.Axis(format="%")
MONEY_AXIS = alt.Axis(format="$,.0f")

TOOLTIP_PCT = ".2%"
TOOLTIP_PCT1 = ".1%"


def configure_altair() -> None:
    """Enable the Ledger theme and lift Altair's row cap.

    The opportunity-set cloud can legitimately carry 10,000 simulated
    portfolios; Altair's default 5,000-row guard would otherwise raise instead
    of rendering the chart the user asked for.
    """
    theme.register_altair_theme(enable=True)
    alt.data_transformers.enable("default", max_rows=None)


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def asset_color(
    tickers: list[str],
    field: str = "Ticker",
    title: str | None = None,
    legend: bool = True,
) -> alt.Color:
    domain = sorted(set(tickers))
    return alt.Color(
        f"{field}:N",
        title=title,
        scale=alt.Scale(domain=domain, range=theme.asset_range(domain)),
        legend=alt.Legend(title=title) if legend else None,
    )


def portfolio_color(
    names: list[str],
    field: str = "Portfolio",
    title: str | None = None,
    legend: bool = True,
) -> alt.Color:
    domain = list(dict.fromkeys(names))
    return alt.Color(
        f"{field}:N",
        title=title,
        scale=alt.Scale(domain=domain, range=theme.portfolio_range(domain)),
        legend=alt.Legend(title=title) if legend else None,
    )


def _legend_highlight(field: str) -> alt.Parameter:
    """Click a legend entry to isolate one series."""
    return alt.selection_point(fields=[field], bind="legend")


def _zero_rule(axis: str = "y") -> alt.Chart:
    frame = pd.DataFrame({"zero": [0.0]})
    encode = {"y": "zero:Q"} if axis == "y" else {"x": "zero:Q"}
    return (
        alt.Chart(frame)
        .mark_rule(color=theme.BORDER_STRONG, strokeWidth=1)
        .encode(**encode)
    )


def _reference_rule(value: float, axis: str = "y", label: str | None = None):
    frame = pd.DataFrame({"value": [float(value)], "label": [label or ""]})
    encode = {"y": "value:Q"} if axis == "y" else {"x": "value:Q"}
    rule = (
        alt.Chart(frame)
        .mark_rule(color=theme.MUTED, strokeDash=[4, 3], strokeWidth=1)
        .encode(**encode)
    )
    if not label:
        return rule
    text = (
        alt.Chart(frame)
        .mark_text(
            align="left" if axis == "y" else "center",
            baseline="bottom",
            dx=4,
            dy=-4,
            fontSize=10,
            color=theme.MUTED,
        )
        .encode(text="label:N", **encode)
    )
    return rule + text


# ----------------------------------------------------------------------
# Section 1 — return-stream diagnostics
# ----------------------------------------------------------------------


def growth_chart(
    growth: pd.DataFrame,
    tickers: list[str],
    log_scale: bool = False,
    height: int = 360,
    value_title: str = "Growth of $1",
) -> alt.Chart:
    highlight = _legend_highlight("Series")
    scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
    return (
        alt.Chart(growth)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Growth:Q", title=value_title, scale=scale),
            color=asset_color(tickers, field="Series"),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.14)),
            strokeWidth=alt.condition(highlight, alt.value(2.2), alt.value(1.1)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title="Ticker"),
                alt.Tooltip("Growth:Q", format=".2f", title="Growth of $1"),
            ],
        )
        .add_params(highlight)
        .properties(height=height)
    )


def drawdown_chart(
    drawdowns: pd.DataFrame,
    tickers: list[str],
    height: int = 320,
    label: str = "Ticker",
) -> alt.Chart:
    highlight = _legend_highlight("Series")
    return (
        alt.Chart(drawdowns)
        .mark_area(opacity=0.16, line={"strokeWidth": 1.4})
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Drawdown:Q",
                title="Drawdown from prior peak",
                axis=PCT_AXIS,
                # Vega-Lite stacks area marks by default; these curves overlay
                # one another, so stacking would sum eleven drawdowns into an
                # impossible -350%.
                stack=None,
            ),
            color=asset_color(tickers, field="Series"),
            opacity=alt.condition(highlight, alt.value(0.18), alt.value(0.03)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title=label),
                alt.Tooltip("Drawdown:Q", format=TOOLTIP_PCT1),
            ],
        )
        .add_params(highlight)
        .properties(height=height)
    )


def rolling_return_chart(
    rolling: pd.DataFrame,
    tickers: list[str],
    window_label: str,
    height: int = 320,
) -> alt.Chart:
    highlight = _legend_highlight("Series")
    lines = (
        alt.Chart(rolling)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Rolling Return:Q",
                title=f"Rolling {window_label} return",
                axis=PCT_AXIS,
            ),
            color=asset_color(tickers, field="Series"),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.12)),
            tooltip=[
                alt.Tooltip("Date:T", title="Window end"),
                alt.Tooltip("Series:N", title="Ticker"),
                alt.Tooltip("Rolling Return:Q", format=TOOLTIP_PCT1),
            ],
        )
        .add_params(highlight)
    )
    return (lines + _zero_rule("y")).properties(height=height)


def risk_return_scatter(
    stats: pd.DataFrame,
    cvar_column: str,
    height: int = 400,
) -> alt.Chart:
    """Volatility against CAGR, sized by tail loss, one point per asset."""
    frame = stats.reset_index().rename(columns={"index": "Ticker"})
    tickers = frame["Ticker"].tolist()

    base = alt.Chart(frame).encode(
        x=alt.X(
            "Volatility:Q",
            axis=PCT_AXIS,
            title="Annualised volatility",
            # Room on the right for the ticker label beside the rightmost point.
            scale=alt.Scale(zero=False, nice=True, paddingInner=0, padding=54),
        ),
        y=alt.Y(
            "CAGR:Q",
            axis=PCT_AXIS,
            title="Realised CAGR",
            scale=alt.Scale(zero=False, nice=True, padding=22),
        ),
    )
    points = base.mark_point(filled=True, opacity=0.9).encode(
        size=alt.Size(
            f"{cvar_column}:Q",
            title=cvar_column,
            scale=alt.Scale(range=[90, 620]),
            legend=alt.Legend(format=".1%"),
        ),
        color=asset_color(tickers, legend=False),
        tooltip=[
            alt.Tooltip("Ticker:N"),
            alt.Tooltip("CAGR:Q", format=TOOLTIP_PCT1),
            alt.Tooltip("Volatility:Q", format=TOOLTIP_PCT1),
            alt.Tooltip("Sharpe:Q", format=".2f"),
            alt.Tooltip("Sortino:Q", format=".2f"),
            alt.Tooltip("Max Drawdown:Q", format=TOOLTIP_PCT1),
            alt.Tooltip(f"{cvar_column}:Q", format=TOOLTIP_PCT1),
        ],
    )
    labels = base.mark_text(
        align="left",
        baseline="middle",
        dx=13,
        fontSize=11,
        fontWeight=600,
        color=theme.INK,
    ).encode(text="Ticker:N")
    return (points + labels).properties(height=height)


def correlation_heatmap(
    corr_long: pd.DataFrame,
    order: list[str],
    height: int = 430,
    annotate: bool = True,
) -> alt.Chart:
    base = alt.Chart(corr_long).encode(
        x=alt.X(
            "Asset A:N",
            title=None,
            sort=order,
            axis=alt.Axis(orient="top", labelAngle=0),
        ),
        y=alt.Y("Asset B:N", title=None, sort=order),
    )
    cells = base.mark_rect(stroke=theme.SURFACE, strokeWidth=1.5).encode(
        color=alt.Color(
            "Correlation:Q",
            scale=alt.Scale(
                domain=[-1, 1],
                range=list(theme.DIVERGING),
                interpolate="hcl",
            ),
            legend=alt.Legend(
                title="Correlation",
                format=".1f",
                gradientLength=180,
                direction="horizontal",
                orient="bottom",
            ),
        ),
        tooltip=[
            "Asset A:N",
            "Asset B:N",
            alt.Tooltip("Correlation:Q", format=".2f"),
        ],
    )
    if not annotate:
        return cells.properties(height=height)

    labels = base.mark_text(fontSize=10).encode(
        text=alt.Text("Correlation:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.Correlation) > 0.62",
            alt.value(theme.SURFACE),
            alt.value(theme.INK_SOFT),
        ),
    )
    return (cells + labels).properties(height=height)


def average_correlation_chart(
    frame: pd.DataFrame,
    height: int = 320,
) -> alt.Chart:
    """Mean correlation to the rest of the universe, lowest first."""
    tickers = frame["Ticker"].tolist()
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3, height=16)
        .encode(
            y=alt.Y("Ticker:N", title=None, sort="x"),
            x=alt.X(
                "Average Correlation:Q",
                title="Mean correlation to the rest of the universe",
                axis=alt.Axis(format=".2f"),
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            color=asset_color(tickers, legend=False),
            tooltip=[
                "Ticker:N",
                alt.Tooltip("Average Correlation:Q", format=".2f"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=5, fontSize=10, color=theme.INK_SOFT)
        .encode(
            y=alt.Y("Ticker:N", sort="x"),
            x="Average Correlation:Q",
            text=alt.Text("Average Correlation:Q", format=".2f"),
        )
    )
    return (bars + labels).properties(height=height)


# ----------------------------------------------------------------------
# Section 1 — construction methods
# ----------------------------------------------------------------------


def weights_composition_chart(
    weights_long: pd.DataFrame,
    tickers: list[str],
    method_order: list[str],
    height: int | None = None,
    category: str = "Method",
) -> alt.Chart:
    """Full-width stacked allocation bars, one row per construction method.

    Stacked rather than grouped: with a double-digit universe, grouped bars
    become 70-plus slivers, while a stacked row reads as a portfolio.
    """
    rows = max(len(method_order), 1)
    return (
        alt.Chart(weights_long)
        .mark_bar(height=alt.RelativeBandSize(0.74))
        .encode(
            y=alt.Y(f"{category}:N", title=None, sort=method_order),
            x=alt.X(
                "Weight:Q",
                title="Allocation",
                axis=PCT_AXIS,
                stack="normalize",
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            color=asset_color(tickers),
            order=alt.Order("Ticker:N"),
            tooltip=[
                alt.Tooltip(f"{category}:N"),
                "Ticker:N",
                alt.Tooltip("Weight:Q", format=TOOLTIP_PCT1),
            ],
        )
        .properties(height=height or max(210, 40 * rows))
    )


def risk_contribution_chart(
    frame: pd.DataFrame,
    tickers: list[str],
    method_order: list[str],
    height: int | None = None,
) -> alt.Chart:
    """Share of portfolio volatility contributed by each holding."""
    rows = max(len(method_order), 1)
    return (
        alt.Chart(frame)
        .mark_bar(height=alt.RelativeBandSize(0.74))
        .encode(
            y=alt.Y("Portfolio:N", title=None, sort=method_order),
            x=alt.X(
                "Risk Share:Q",
                title="Share of portfolio volatility",
                axis=PCT_AXIS,
                stack="normalize",
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            color=asset_color(tickers),
            order=alt.Order("Ticker:N"),
            tooltip=[
                "Portfolio:N",
                "Ticker:N",
                alt.Tooltip("Risk Share:Q", format=TOOLTIP_PCT1),
                alt.Tooltip("Weight:Q", format=TOOLTIP_PCT1),
            ],
        )
        .properties(height=height or max(210, 40 * rows))
    )


# ----------------------------------------------------------------------
# Section 1 — frontier
# ----------------------------------------------------------------------


def frontier_chart(
    cloud: pd.DataFrame,
    frontier: pd.DataFrame,
    named: pd.DataFrame,
    label_names: list[str],
    height: int = 500,
) -> alt.Chart:
    x = alt.X(
        "Volatility:Q",
        axis=PCT_AXIS,
        title="Expected volatility",
        scale=alt.Scale(zero=False, nice=True, padding=14),
    )
    y = alt.Y(
        "Expected Return:Q",
        axis=PCT_AXIS,
        title="Estimated expected return",
        scale=alt.Scale(zero=False, nice=True, padding=14),
    )

    cloud_layer = (
        alt.Chart(cloud)
        .mark_circle(size=16, opacity=0.16, color=theme.FAINT)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                alt.Tooltip("Volatility:Q", format=TOOLTIP_PCT),
                alt.Tooltip("Sharpe:Q", format=".2f"),
            ],
        )
    )
    frontier_layer = (
        alt.Chart(frontier)
        .mark_line(size=2.6, color=theme.NAVY)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                alt.Tooltip("Volatility:Q", format=TOOLTIP_PCT),
            ],
        )
    )
    points_layer = (
        alt.Chart(named)
        .mark_point(size=155, filled=True, stroke=theme.SURFACE, strokeWidth=1.4)
        .encode(
            x=x,
            y=y,
            color=portfolio_color(named["Portfolio"].tolist()),
            shape=alt.Shape(
                "Portfolio:N",
                legend=None,
                scale=alt.Scale(
                    domain=named["Portfolio"].tolist(),
                    range=[
                        "diamond" if name == "Reference Portfolio" else "circle"
                        for name in named["Portfolio"]
                    ],
                ),
            ),
            tooltip=[
                "Portfolio:N",
                alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                alt.Tooltip("Volatility:Q", format=TOOLTIP_PCT),
                alt.Tooltip("Sharpe:Q", format=".2f"),
                alt.Tooltip("Weights:N", title="Largest holdings"),
            ],
        )
    )
    labels_layer = (
        alt.Chart(named[named["Portfolio"].isin(label_names)])
        .mark_text(
            dx=10,
            dy=-10,
            align="left",
            fontSize=11,
            fontWeight=600,
            color=theme.INK,
        )
        .encode(x=x, y=y, text="Portfolio:N")
    )
    return (
        (cloud_layer + frontier_layer + points_layer + labels_layer)
        .properties(height=height)
        .interactive()
    )


def frontier_composition_chart(
    frame: pd.DataFrame,
    tickers: list[str],
    x_column: str = "Volatility",
    x_title: str = "Expected volatility",
    height: int = 300,
) -> alt.Chart:
    """How the solved frontier reallocates as risk is dialled up."""
    return (
        alt.Chart(frame)
        .mark_area(interpolate="monotone", opacity=0.9)
        .encode(
            x=alt.X(
                f"{x_column}:Q",
                axis=PCT_AXIS,
                title=x_title,
                scale=alt.Scale(zero=False, nice=False),
            ),
            y=alt.Y(
                "Weight:Q",
                stack="normalize",
                axis=PCT_AXIS,
                title="Frontier allocation",
            ),
            color=asset_color(tickers),
            order=alt.Order("Ticker:N"),
            tooltip=[
                alt.Tooltip(f"{x_column}:Q", format=TOOLTIP_PCT),
                "Ticker:N",
                alt.Tooltip("Weight:Q", format=TOOLTIP_PCT1),
            ],
        )
        .properties(height=height)
    )


def sensitivity_frontier_chart(
    frame: pd.DataFrame,
    height: int = 390,
) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_line(size=2.6)
        .encode(
            x=alt.X(
                "Volatility:Q",
                axis=PCT_AXIS,
                title="Expected volatility",
                scale=alt.Scale(zero=False, nice=True, padding=12),
            ),
            y=alt.Y(
                "Expected Return:Q",
                axis=PCT_AXIS,
                title="Estimated expected return",
                scale=alt.Scale(zero=False, nice=True, padding=12),
            ),
            color=alt.Color(
                "Estimator:N",
                title=None,
                scale=alt.Scale(
                    domain=["Historical E(r)", "Shrunk E(r)"],
                    range=[theme.BRASS, theme.NAVY],
                ),
            ),
            strokeDash=alt.StrokeDash(
                "Estimator:N",
                legend=None,
                scale=alt.Scale(
                    domain=["Historical E(r)", "Shrunk E(r)"],
                    range=[[5, 3], [1, 0]],
                ),
            ),
            tooltip=[
                "Estimator:N",
                alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                alt.Tooltip("Volatility:Q", format=TOOLTIP_PCT),
            ],
        )
        .properties(height=height)
    )


def cvar_frontier_chart(
    cloud: pd.DataFrame,
    frontier: pd.DataFrame,
    named: pd.DataFrame,
    cvar_level: float,
    height: int = 470,
) -> alt.Chart:
    x_title = f"Historical CVaR {cvar_level:.0%} per return period"
    x = alt.X(
        "CVaR:Q",
        axis=PCT_AXIS,
        title=x_title,
        scale=alt.Scale(zero=False, nice=True, padding=14),
    )
    y = alt.Y(
        "Expected Return:Q",
        axis=PCT_AXIS,
        title="Estimated expected return",
        scale=alt.Scale(zero=False, nice=True, padding=14),
    )

    cloud_layer = (
        alt.Chart(cloud)
        .mark_circle(size=16, opacity=0.16, color=theme.FAINT)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                alt.Tooltip("CVaR:Q", format=TOOLTIP_PCT),
            ],
        )
    )
    frontier_layer = (
        alt.Chart(frontier)
        .mark_line(size=2.6, color=theme.NAVY)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                alt.Tooltip("CVaR:Q", format=TOOLTIP_PCT),
            ],
        )
    )
    points_layer = (
        alt.Chart(named)
        .mark_point(size=155, filled=True, stroke=theme.SURFACE, strokeWidth=1.4)
        .encode(
            x=x,
            y=y,
            color=portfolio_color(named["Portfolio"].tolist()),
            tooltip=[
                "Portfolio:N",
                alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                alt.Tooltip("CVaR:Q", format=TOOLTIP_PCT),
            ],
        )
    )
    return (
        (cloud_layer + frontier_layer + points_layer)
        .properties(height=height)
        .interactive()
    )


def return_diversification_frontier_chart(
    frontier: pd.DataFrame,
    named: pd.DataFrame,
    label_names: list[str] | None = None,
    height: int = 455,
) -> alt.Chart:
    """Expected return versus diversification ratio with named portfolios.

    The line is an optimised boundary: for each minimum expected-return target it
    shows the maximum diversification ratio available under the standard
    long-only and maximum-weight constraints.  Named construction methods are
    overlaid so the reader can see how close each method sits to that boundary.
    """
    label_names = label_names or []
    x = alt.X(
        "Diversification Ratio:Q",
        title="Diversification ratio",
        scale=alt.Scale(zero=False, nice=True, padding=14),
    )
    y = alt.Y(
        "Expected Return:Q",
        axis=PCT_AXIS,
        title="Expected return E(r)",
        scale=alt.Scale(zero=False, nice=True, padding=14),
    )

    layers: list[alt.Chart] = []
    if frontier is not None and not frontier.empty:
        layers.append(
            alt.Chart(frontier)
            .mark_line(size=2.6, color=theme.NAVY)
            .encode(
                x=x,
                y=y,
                tooltip=[
                    alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                    alt.Tooltip("Diversification Ratio:Q", format=".3f"),
                    alt.Tooltip("Minimum Expected Return:Q", format=TOOLTIP_PCT),
                ],
            )
        )

    if named is not None and not named.empty:
        names = named["Portfolio"].astype(str).tolist()
        points = (
            alt.Chart(named)
            .mark_point(size=155, filled=True, stroke=theme.SURFACE, strokeWidth=1.4)
            .encode(
                x=x,
                y=y,
                color=portfolio_color(names),
                shape=alt.Shape(
                    "Portfolio:N",
                    legend=None,
                    scale=alt.Scale(
                        domain=names,
                        range=[
                            "diamond" if name == "Reference Portfolio" else "circle"
                            for name in names
                        ],
                    ),
                ),
                tooltip=[
                    "Portfolio:N",
                    alt.Tooltip("Expected Return:Q", format=TOOLTIP_PCT),
                    alt.Tooltip("Diversification Ratio:Q", format=".3f"),
                    alt.Tooltip(
                        "Effective ETFs:Q", title="Effective holdings", format=".2f"
                    ),
                    alt.Tooltip("Effective Risk Bets:Q", format=".2f"),
                    alt.Tooltip("Largest Risk Share:Q", format=TOOLTIP_PCT1),
                ],
            )
        )
        layers.append(points)

        labelled = named[named["Portfolio"].isin(label_names)]
        if not labelled.empty:
            layers.append(
                alt.Chart(labelled)
                .mark_text(
                    dx=9,
                    dy=-9,
                    align="left",
                    fontSize=10.5,
                    fontWeight=600,
                    color=theme.INK,
                )
                .encode(x=x, y=y, text="Portfolio:N")
            )

        reference = named.loc[named["Portfolio"] == "Reference Portfolio"]
        if not reference.empty:
            ref_dr = float(reference.iloc[0]["Diversification Ratio"])
            ref_er = float(reference.iloc[0]["Expected Return"])
            layers.extend(
                [
                    _reference_rule(ref_dr, axis="x", label="Current DR"),
                    _reference_rule(ref_er, axis="y", label="Current E(r)"),
                ]
            )

    if not layers:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()
    return alt.layer(*layers).properties(height=height).interactive()


def cluster_allocation_chart(
    frame: pd.DataFrame,
    portfolio_order: list[str],
    cluster_order: list[str],
    height: int | None = None,
) -> alt.Chart:
    """Stacked portfolio weights by return-correlation cluster."""
    colors = [
        theme.CATEGORICAL[i % len(theme.CATEGORICAL)] for i in range(len(cluster_order))
    ]
    return (
        alt.Chart(frame)
        .mark_bar(height=alt.RelativeBandSize(0.74))
        .encode(
            y=alt.Y("Portfolio:N", title=None, sort=portfolio_order),
            x=alt.X(
                "Weight:Q",
                title="Allocation by return cluster",
                axis=PCT_AXIS,
                stack="normalize",
                scale=alt.Scale(domain=[0, 1], nice=False),
            ),
            color=alt.Color(
                "Cluster:N",
                title=None,
                scale=alt.Scale(domain=cluster_order, range=colors),
            ),
            order=alt.Order("Cluster:N"),
            tooltip=[
                "Portfolio:N",
                "Cluster:N",
                alt.Tooltip("Weight:Q", format=TOOLTIP_PCT1),
            ],
        )
        .properties(height=height or max(190, 42 * max(len(portfolio_order), 1)))
    )


# ----------------------------------------------------------------------
# Section 3 — bootstrap simulation
# ----------------------------------------------------------------------


def terminal_wealth_range_chart(
    frame: pd.DataFrame,
    horizon_years: int,
    height: int | None = None,
) -> alt.Chart:
    order = alt.Sort(field="Median Terminal Wealth", op="max", order="descending")
    base = alt.Chart(frame).encode(y=alt.Y("Portfolio:N", title=None, sort=order))

    outer = base.mark_rule(size=2, opacity=0.42, color=theme.BORDER_STRONG).encode(
        x=alt.X(
            "5th Percentile Wealth:Q",
            title=f"Terminal wealth after {horizon_years} years",
            axis=MONEY_AXIS,
            scale=alt.Scale(zero=False, nice=True, padding=12),
        ),
        x2="95th Percentile Wealth:Q",
    )
    inner = base.mark_bar(height=11, cornerRadius=3, opacity=0.9).encode(
        x="25th Percentile Wealth:Q",
        x2="75th Percentile Wealth:Q",
        color=portfolio_color(frame["Portfolio"].tolist(), legend=False),
        tooltip=[
            "Portfolio:N",
            alt.Tooltip("5th Percentile Wealth:Q", format="$,.0f"),
            alt.Tooltip("25th Percentile Wealth:Q", format="$,.0f"),
            alt.Tooltip("Median Terminal Wealth:Q", format="$,.0f"),
            alt.Tooltip("75th Percentile Wealth:Q", format="$,.0f"),
            alt.Tooltip("95th Percentile Wealth:Q", format="$,.0f"),
        ],
    )
    median = base.mark_tick(size=19, thickness=2.4, color=theme.INK).encode(
        x="Median Terminal Wealth:Q",
        tooltip=[
            "Portfolio:N",
            alt.Tooltip("Median Terminal Wealth:Q", format="$,.0f"),
        ],
    )
    return (outer + inner + median).properties(
        height=height or max(230, 44 * len(frame))
    )


def fan_chart(
    fan: pd.DataFrame,
    initial_wealth: float,
    color: str = theme.NAVY,
    height: int = 390,
) -> alt.Chart:
    """Percentile fan as nested bands rather than five competing lines."""
    x = alt.X(
        "Month:Q",
        title="Simulated month",
        scale=alt.Scale(nice=False, zero=False),
    )
    y_title = "Portfolio wealth"

    outer = (
        alt.Chart(fan)
        .mark_area(opacity=0.16, color=color)
        .encode(
            x=x,
            y=alt.Y("5th:Q", title=y_title, axis=MONEY_AXIS),
            y2="95th:Q",
            tooltip=[
                alt.Tooltip("Month:Q"),
                alt.Tooltip("5th:Q", format="$,.0f", title="5th percentile"),
                alt.Tooltip("95th:Q", format="$,.0f", title="95th percentile"),
            ],
        )
    )
    inner = (
        alt.Chart(fan)
        .mark_area(opacity=0.30, color=color)
        .encode(
            x=x,
            y=alt.Y("25th:Q", title=y_title),
            y2="75th:Q",
            tooltip=[
                alt.Tooltip("Month:Q"),
                alt.Tooltip("25th:Q", format="$,.0f", title="25th percentile"),
                alt.Tooltip("75th:Q", format="$,.0f", title="75th percentile"),
            ],
        )
    )
    median = (
        alt.Chart(fan)
        .mark_line(size=2.2, color=color)
        .encode(
            x=x,
            y=alt.Y("Median:Q", title=y_title),
            tooltip=[
                alt.Tooltip("Month:Q"),
                alt.Tooltip("Median:Q", format="$,.0f", title="Median"),
            ],
        )
    )
    start = _reference_rule(initial_wealth, axis="y", label="Starting wealth")
    return (outer + inner + median + start).properties(height=height)


def distribution_chart(
    hist: pd.DataFrame,
    markers: pd.DataFrame | None = None,
    value_format: str = "$,.0f",
    value_title: str = "Terminal wealth",
    reference: float | None = None,
    reference_label: str | None = None,
    height: int = 340,
) -> alt.Chart:
    """Pre-binned simulated distribution, one filled curve per portfolio."""
    names = list(dict.fromkeys(hist["Series"].tolist()))
    highlight = _legend_highlight("Series")

    areas = (
        alt.Chart(hist)
        .mark_area(interpolate="monotone", opacity=0.16, line={"strokeWidth": 1.8})
        .encode(
            x=alt.X(
                "Value:Q",
                title=value_title,
                axis=alt.Axis(format=value_format),
                scale=alt.Scale(zero=False, nice=False),
            ),
            y=alt.Y(
                "Share:Q",
                title="Share of simulated paths",
                axis=alt.Axis(format=".1%"),
                stack=None,
            ),
            color=portfolio_color(names, field="Series"),
            opacity=alt.condition(highlight, alt.value(0.17), alt.value(0.03)),
            tooltip=[
                alt.Tooltip("Series:N", title="Portfolio"),
                alt.Tooltip("Value:Q", format=value_format, title=value_title),
                alt.Tooltip("Share:Q", format=".2%", title="Share of paths"),
            ],
        )
        .add_params(highlight)
    )

    layers = [areas]
    if reference is not None:
        layers.append(_reference_rule(reference, axis="x", label=reference_label))
    if markers is not None and not markers.empty:
        layers.append(
            alt.Chart(markers)
            .mark_rule(strokeDash=[3, 3], strokeWidth=1.2, opacity=0.8)
            .encode(
                x="Value:Q",
                color=portfolio_color(names, field="Series", legend=False),
                tooltip=[
                    alt.Tooltip("Series:N", title="Portfolio"),
                    "Percentile:N",
                    alt.Tooltip("Value:Q", format=value_format),
                ],
            )
        )
    return alt.layer(*layers).properties(height=height)


def probability_chart(
    frame: pd.DataFrame,
    height: int = 390,
) -> alt.Chart:
    metrics = list(dict.fromkeys(frame["Metric"].tolist()))
    ramp_colors = [
        theme.ramp(theme.SEQUENTIAL, 0.35 + 0.6 * i / max(len(metrics) - 1, 1))
        for i in range(len(metrics))
    ]
    if metrics and metrics[0].startswith("P(Loss"):
        ramp_colors[0] = theme.NEGATIVE
    return (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=2)
        .encode(
            x=alt.X("Portfolio:N", title=None, axis=alt.Axis(labelAngle=-20)),
            y=alt.Y(
                "Probability:Q",
                axis=PCT_AXIS,
                scale=alt.Scale(domain=[0, 1], nice=False),
                title=None,
            ),
            xOffset=alt.XOffset("Metric:N", sort=metrics),
            color=alt.Color(
                "Metric:N",
                title=None,
                scale=alt.Scale(domain=metrics, range=ramp_colors),
            ),
            tooltip=[
                "Portfolio:N",
                "Metric:N",
                alt.Tooltip("Probability:Q", format=TOOLTIP_PCT1),
            ],
        )
        .properties(height=height)
    )


# ----------------------------------------------------------------------
# Section 4 — walk-forward validation
# ----------------------------------------------------------------------


def portfolio_growth_chart(
    growth: pd.DataFrame,
    names: list[str],
    log_scale: bool = False,
    height: int = 410,
    value_title: str = "Growth of $1",
) -> alt.Chart:
    highlight = _legend_highlight("Series")
    scale = alt.Scale(type="log") if log_scale else alt.Scale(zero=False)
    return (
        alt.Chart(growth)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("Growth:Q", title=value_title, scale=scale),
            color=portfolio_color(names, field="Series"),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.14)),
            strokeWidth=alt.condition(highlight, alt.value(2.3), alt.value(1.1)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title="Portfolio"),
                alt.Tooltip("Growth:Q", format=".2f"),
            ],
        )
        .add_params(highlight)
        .properties(height=height)
    )


def portfolio_drawdown_chart(
    drawdowns: pd.DataFrame,
    names: list[str],
    height: int = 330,
) -> alt.Chart:
    highlight = _legend_highlight("Series")
    return (
        alt.Chart(drawdowns)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Drawdown:Q",
                title="Out-of-sample drawdown",
                axis=PCT_AXIS,
            ),
            color=portfolio_color(names, field="Series"),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.13)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title="Portfolio"),
                alt.Tooltip("Drawdown:Q", format=TOOLTIP_PCT1),
            ],
        )
        .add_params(highlight)
        .properties(height=height)
    )


def relative_wealth_chart(
    frame: pd.DataFrame,
    names: list[str],
    benchmark: str,
    height: int = 350,
) -> alt.Chart:
    highlight = _legend_highlight("Series")
    lines = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Relative Wealth:Q",
                title=f"Cumulative wealth vs {benchmark}",
                axis=alt.Axis(format=".2f"),
                scale=alt.Scale(zero=False),
            ),
            color=portfolio_color(names, field="Series"),
            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.13)),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Series:N", title="Portfolio"),
                alt.Tooltip("Relative Wealth:Q", format=".3f"),
            ],
        )
        .add_params(highlight)
    )
    parity = _reference_rule(1.0, axis="y", label=f"Matches {benchmark}")
    return (lines + parity).properties(height=height)


def win_rate_chart(
    frame: pd.DataFrame,
    threshold: float = 0.5,
    height: int = 340,
) -> alt.Chart:
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3, height=17)
        .encode(
            y=alt.Y("Portfolio:N", title=None, sort="-x"),
            x=alt.X(
                "Block Win Rate:Q",
                axis=PCT_AXIS,
                scale=alt.Scale(domain=[0, 1], nice=False),
                title="Share of out-of-sample holding blocks won",
            ),
            color=alt.condition(
                alt.datum["Block Win Rate"] >= threshold,
                alt.value(theme.POSITIVE),
                alt.value(theme.NEGATIVE),
            ),
            tooltip=[
                "Portfolio:N",
                alt.Tooltip("Block Win Rate:Q", format=".0%"),
            ],
        )
    )
    labels = (
        alt.Chart(frame)
        .mark_text(align="left", dx=5, fontSize=10, color=theme.INK_SOFT)
        .encode(
            y=alt.Y("Portfolio:N", sort="-x"),
            x="Block Win Rate:Q",
            text=alt.Text("Block Win Rate:Q", format=".0%"),
        )
    )
    rule = _reference_rule(threshold, axis="x", label="Coin flip")
    return (bars + labels + rule).properties(height=height or max(220, 32 * len(frame)))


def weight_evolution_chart(
    frame: pd.DataFrame,
    tickers: list[str],
    height: int = 370,
) -> alt.Chart:
    date_col = "Entry Date" if "Entry Date" in frame.columns else "Rebalance Date"
    return (
        alt.Chart(frame)
        .mark_area(interpolate="step-after", opacity=0.92)
        .encode(
            x=alt.X(f"{date_col}:T", title=None),
            y=alt.Y(
                "Weight:Q",
                stack="normalize",
                axis=PCT_AXIS,
                title="Starting allocation",
            ),
            color=asset_color(tickers),
            order=alt.Order("Ticker:N"),
            tooltip=[
                alt.Tooltip(f"{date_col}:T"),
                "Ticker:N",
                alt.Tooltip("Weight:Q", format=TOOLTIP_PCT1),
            ],
        )
        .properties(height=height)
    )
