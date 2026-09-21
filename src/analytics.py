"""Presentation-layer derived views.

Reshaping and summarising helpers that turn engine output into the tidy frames
the charting and export layers expect. Nothing here estimates a new parameter
or changes an optimisation result — it reads the same return series, weights
and simulation output the analysis already produced.

Keeping this separate from ``metrics``/``risk``/``optimization`` means the
quantitative engine stays untouched while the reporting layer can grow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import drawdown_series
from .risk import risk_contributions

DATE = "Date"


def _date_column(frame: pd.DataFrame | pd.Series) -> str:
    return frame.index.name or DATE


def growth_frame(returns: pd.DataFrame, initial: float = 1.0) -> pd.DataFrame:
    """Tidy growth-of-one-unit paths, one row per date and series."""
    growth = initial * (1.0 + returns).cumprod()
    name = _date_column(returns)
    return (
        growth.rename_axis(name)
        .reset_index()
        .melt(id_vars=name, var_name="Series", value_name="Growth")
        .rename(columns={name: DATE})
    )


def drawdown_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """Tidy underwater (peak-to-trough) paths for every column."""
    drawdowns = pd.DataFrame(
        {col: drawdown_series(returns[col]) for col in returns.columns},
        index=returns.index,
    )
    name = _date_column(returns)
    return (
        drawdowns.rename_axis(name)
        .reset_index()
        .melt(id_vars=name, var_name="Series", value_name="Drawdown")
        .rename(columns={name: DATE})
    )


def rolling_annual_return(
    returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Tidy rolling compounded return over ``window`` periods."""
    if window < 2 or len(returns) <= window:
        return pd.DataFrame(columns=[DATE, "Series", "Rolling Return"])

    rolled = (1.0 + returns).rolling(window).apply(np.prod, raw=True).dropna(
        how="all"
    ) - 1.0
    name = _date_column(returns)
    return (
        rolled.rename_axis(name)
        .reset_index()
        .melt(id_vars=name, var_name="Series", value_name="Rolling Return")
        .rename(columns={name: DATE})
        .dropna(subset=["Rolling Return"])
    )


def average_correlation(corr: pd.DataFrame) -> pd.Series:
    """Mean pairwise correlation of each asset to the rest of the universe.

    A high value flags a return stream that largely repeats what the rest of
    the universe already provides.
    """
    if corr.empty:
        return pd.Series(dtype=float)
    masked = corr.where(~np.eye(len(corr), dtype=bool))
    return masked.mean(axis=1).sort_values()


def cluster_order(corr: pd.DataFrame) -> list[str]:
    """Order tickers so correlated blocks sit together in a heatmap.

    Falls back to the input order if the matrix is too small or SciPy cannot
    linkage-cluster it. Pure reordering — no value is altered.
    """
    labels = list(corr.columns)
    if len(labels) < 3:
        return labels
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        distance = (1.0 - corr.to_numpy(float)) / 2.0
        np.fill_diagonal(distance, 0.0)
        distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
        linked = linkage(squareform(distance, checks=False), method="average")
        return [labels[i] for i in leaves_list(linked)]
    except Exception:
        return labels


def risk_contribution_frame(
    portfolios: dict[str, pd.Series],
    cov: pd.DataFrame,
) -> pd.DataFrame:
    """Share of portfolio volatility contributed by each holding."""
    rows = []
    for name, weights in portfolios.items():
        contribution = risk_contributions(weights, cov)
        total = float(contribution.sum())
        if total <= 0:
            continue
        share = contribution / total
        for ticker, value in share.items():
            rows.append(
                {
                    "Portfolio": name,
                    "Ticker": ticker,
                    "Risk Share": float(value),
                    "Weight": float(weights.get(ticker, 0.0)),
                }
            )
    return pd.DataFrame(rows)


def weights_long(weights: pd.DataFrame) -> pd.DataFrame:
    """Method x ticker weight matrix to tidy rows."""
    return (
        weights.rename_axis("Method")
        .reset_index()
        .melt(id_vars="Method", var_name="Ticker", value_name="Weight")
    )


def frontier_composition(
    frontier: pd.DataFrame,
    x_column: str = "Volatility",
) -> pd.DataFrame:
    """Tidy asset weights along a solved frontier.

    The frontier solver already returns per-asset weights as ``w_<ticker>``
    columns; this exposes them so the reader can see *how* composition changes
    along the curve rather than only where the curve sits.
    """
    weight_cols = [c for c in frontier.columns if c.startswith("w_")]
    if frontier.empty or not weight_cols or x_column not in frontier.columns:
        return pd.DataFrame(columns=[x_column, "Ticker", "Weight"])

    tidy = frontier[[x_column] + weight_cols].melt(
        id_vars=x_column,
        var_name="Ticker",
        value_name="Weight",
    )
    tidy["Ticker"] = tidy["Ticker"].str.removeprefix("w_")
    return tidy


def relative_wealth(
    wealth: pd.DataFrame,
    benchmark: str,
) -> pd.DataFrame:
    """Cumulative wealth ratio of every series against ``benchmark``.

    1.0 means the portfolio has tracked the benchmark exactly since the start
    of the out-of-sample record; 1.10 means it is 10% ahead.
    """
    if benchmark not in wealth.columns:
        return pd.DataFrame(columns=[DATE, "Series", "Relative Wealth"])

    base = wealth[benchmark].replace(0.0, np.nan)
    ratio = wealth.drop(columns=[benchmark]).div(base, axis=0).dropna(how="all")
    name = _date_column(wealth)
    return (
        ratio.rename_axis(name)
        .reset_index()
        .melt(id_vars=name, var_name="Series", value_name="Relative Wealth")
        .rename(columns={name: DATE})
        .dropna(subset=["Relative Wealth"])
    )


def histogram_frame(
    samples: dict[str, np.ndarray],
    bins: int = 48,
    clip_quantiles: tuple[float, float] = (0.005, 0.995),
) -> pd.DataFrame:
    """Pre-binned densities for one or more simulated distributions.

    Binning server-side keeps the browser payload at a few hundred rows
    instead of tens of thousands of raw simulation draws — the visual is
    identical and the chart renders immediately.

    The plotted range covers the central 99% of the pooled draws. Draws
    outside it are excluded rather than clipped, so the extreme bins show
    their true density instead of a false spike of piled-up outliers.
    """
    arrays = {
        name: np.asarray(values, dtype=float).ravel()
        for name, values in samples.items()
        if np.asarray(values).size
    }
    if not arrays:
        return pd.DataFrame(columns=["Series", "Value", "Density", "Share"])

    pooled = np.concatenate(list(arrays.values()))
    low = float(np.quantile(pooled, clip_quantiles[0]))
    high = float(np.quantile(pooled, clip_quantiles[1]))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low, high = float(pooled.min()), float(pooled.max())
        if high <= low:
            high = low + 1e-9

    edges = np.linspace(low, high, int(bins) + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0

    rows = []
    for name, values in arrays.items():
        inside = values[(values >= low) & (values <= high)]
        counts, _ = np.histogram(inside, bins=edges)
        total = max(values.size, 1)
        share = counts / total
        density = counts / (total * (edges[1] - edges[0]))
        for center, one_share, one_density in zip(centers, share, density, strict=True):
            rows.append(
                {
                    "Series": name,
                    "Value": float(center),
                    "Share": float(one_share),
                    "Density": float(one_density),
                }
            )
    return pd.DataFrame(rows)


def percentile_markers(
    samples: dict[str, np.ndarray],
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
) -> pd.DataFrame:
    """Percentile reference points for distribution charts."""
    rows = []
    for name, values in samples.items():
        array = np.asarray(values, dtype=float).ravel()
        if not array.size:
            continue
        for percentile in percentiles:
            rows.append(
                {
                    "Series": name,
                    "Percentile": f"P{percentile:g}",
                    "Value": float(np.percentile(array, percentile)),
                }
            )
    return pd.DataFrame(rows)


def concentration(weights: pd.Series) -> dict[str, float]:
    """Herfindahl concentration diagnostics for a weight vector."""
    values = pd.Series(weights, dtype=float).clip(lower=0.0).replace([np.inf], 0.0)
    total = float(values.sum())
    if total <= 0:
        return {"HHI": float("nan"), "Effective Holdings": float("nan")}
    normalised = values / total
    hhi = float((normalised**2).sum())
    return {
        "HHI": hhi,
        "Effective Holdings": float(1.0 / hhi) if hhi > 0 else float("nan"),
        "Largest Holding": float(normalised.max()),
        "Holdings Above 1%": float((normalised >= 0.01).sum()),
    }
