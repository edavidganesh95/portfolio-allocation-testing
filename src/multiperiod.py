"""Transparent multi-period sensitivity analysis for buy-and-hold allocation.

The section answers two separate questions:

1. **Refit robustness** — if the same construction rule is given only the last
   3 years, 5 years, or the full history, how much does its starting allocation
   change?
2. **Realised-period robustness** — if today's fixed starting allocation had
   been bought at the start of a 1Y / 3Y / 5Y / full-history window and then
   left to drift, how did the realised return/risk metrics compare?

No synthetic robustness score or mixed-objective consensus portfolio is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .diversification import HRP_NAME, hierarchical_risk_parity
from .expected_returns import market_anchored_expected_returns
from .metrics import buy_and_hold_portfolio_returns, summary_table
from .optimization import (
    diversified_maximum_sharpe,
    equal_weight,
    maximum_diversification,
    maximum_return_to_cvar,
    maximum_sharpe,
    minimum_cvar,
    minimum_variance,
    risk_parity,
)
from .risk import ledoit_wolf_covariance

REFIT_METHOD_ORDER = (
    "Equal Weight",
    "Minimum Variance",
    "Risk Parity",
    "Maximum Diversification",
    "Maximum Sharpe",
    "Diversified Maximum Sharpe",
    "Minimum CVaR",
    "Maximum Return / CVaR",
    HRP_NAME,
)


def trailing_window(frame: pd.DataFrame | pd.Series, years: int | None):
    """Return a trailing calendar-year window ending at the last observation."""
    if years is None:
        return frame.copy()
    if frame.empty:
        return frame.copy()
    end = pd.Timestamp(frame.index.max())
    start = end - pd.DateOffset(years=int(years))
    return frame.loc[frame.index > start].copy()


def construction_weights_by_lookback(
    returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    *,
    market_return_prior: float,
    risk_free_rate: float,
    frequency: str,
    shrinkage: float,
    max_weight: float,
    cvar_level: float,
    diversified_max_weight: float,
    min_effective_assets: float,
    max_risk_share: float,
    lookbacks: tuple[tuple[str, int | None], ...] = (
        ("3Y", 3),
        ("5Y", 5),
        ("Full", None),
    ),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit every applicable construction method on 3Y / 5Y / full samples.

    Returns
    -------
    weights:
        MultiIndex rows ``(Lookback, Method)`` and one column per ETF.
    status:
        MultiIndex rows ``(Lookback, Method)`` with observations, success flag,
        and solver/message detail.

    ``Equal Weight`` is included for completeness even though it is unchanged
    when the investable universe is unchanged. HRP is rebuilt from each window's
    covariance/correlation hierarchy and, as elsewhere in the project, does not
    use expected-return forecasts.
    """
    weight_rows: dict[tuple[str, str], pd.Series] = {}
    status_rows: list[dict] = []

    for label, years in lookbacks:
        sample = trailing_window(returns, years).dropna(how="any")
        benchmark = (
            pd.Series(benchmark_returns, dtype=float).reindex(sample.index).dropna()
        )
        sample = sample.reindex(benchmark.index).dropna(how="any")
        observations = len(sample)

        if observations < 24:
            for method in REFIT_METHOD_ORDER:
                status_rows.append(
                    {
                        "Lookback": label,
                        "Method": method,
                        "Observations": observations,
                        "Success": False,
                        "Message": "At least 24 observations are required for optimisation.",
                    }
                )
            continue

        try:
            mu = market_anchored_expected_returns(
                sample,
                benchmark,
                market_return_prior=market_return_prior,
                risk_free_rate=risk_free_rate,
                frequency=frequency,
                shrinkage=shrinkage,
            )
            cov = ledoit_wolf_covariance(sample, frequency=frequency)
            results = [
                equal_weight(list(sample.columns)),
                minimum_variance(cov, max_weight=max_weight),
                risk_parity(cov, max_weight=max_weight),
                maximum_diversification(cov, max_weight=max_weight),
                maximum_sharpe(
                    mu,
                    cov,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_weight,
                ),
                diversified_maximum_sharpe(
                    mu,
                    cov,
                    risk_free_rate=risk_free_rate,
                    max_weight=min(float(max_weight), float(diversified_max_weight)),
                    min_effective_assets=float(min_effective_assets),
                    max_risk_share=float(max_risk_share),
                ),
                minimum_cvar(sample, level=cvar_level, max_weight=max_weight),
                maximum_return_to_cvar(
                    mu,
                    sample,
                    risk_free_rate=risk_free_rate,
                    level=cvar_level,
                    max_weight=max_weight,
                ),
            ]

            for result in results:
                status_rows.append(
                    {
                        "Lookback": label,
                        "Method": result.name,
                        "Observations": observations,
                        "Success": bool(result.success),
                        "Message": str(result.message),
                    }
                )
                if result.success:
                    weight_rows[(label, result.name)] = result.weights.reindex(
                        sample.columns
                    ).fillna(0.0)

            # HRP is a diversification-only counterfactual and must be rebuilt
            # from the window-specific covariance/correlation structure.
            try:
                hrp = hierarchical_risk_parity(
                    cov,
                    sample.corr(),
                    max_weight=max_weight,
                )
                weight_rows[(label, HRP_NAME)] = hrp.reindex(sample.columns).fillna(0.0)
                status_rows.append(
                    {
                        "Lookback": label,
                        "Method": HRP_NAME,
                        "Observations": observations,
                        "Success": True,
                        "Message": "Correlation/risk hierarchy; no expected-return forecast.",
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "Lookback": label,
                        "Method": HRP_NAME,
                        "Observations": observations,
                        "Success": False,
                        "Message": str(exc),
                    }
                )

        except Exception as exc:
            # If shared model inputs fail, mark every method unavailable for the
            # window instead of returning a partially misleading comparison.
            for method in REFIT_METHOD_ORDER:
                if any(
                    row["Lookback"] == label and row["Method"] == method
                    for row in status_rows
                ):
                    continue
                status_rows.append(
                    {
                        "Lookback": label,
                        "Method": method,
                        "Observations": observations,
                        "Success": False,
                        "Message": str(exc),
                    }
                )

    if weight_rows:
        weights = (
            pd.DataFrame(weight_rows).T.reindex(columns=returns.columns).fillna(0.0)
        )
        weights.index = pd.MultiIndex.from_tuples(
            weights.index, names=["Lookback", "Method"]
        )
        lookback_order = [label for label, _ in lookbacks]
        ordered_index = [
            (lookback, method)
            for lookback in lookback_order
            for method in REFIT_METHOD_ORDER
            if (lookback, method) in weights.index
        ]
        weights = weights.reindex(ordered_index)
    else:
        weights = pd.DataFrame(columns=returns.columns)
        weights.index = pd.MultiIndex.from_arrays(
            [[], []], names=["Lookback", "Method"]
        )

    status = pd.DataFrame(status_rows)
    if not status.empty:
        status["Lookback"] = pd.Categorical(
            status["Lookback"],
            categories=[label for label, _ in lookbacks],
            ordered=True,
        )
        status["Method"] = pd.Categorical(
            status["Method"],
            categories=list(REFIT_METHOD_ORDER),
            ordered=True,
        )
        status = status.sort_values(["Lookback", "Method"]).set_index(
            ["Lookback", "Method"]
        )
    return weights, status


def lookback_weight_stability(weights: pd.DataFrame) -> pd.DataFrame:
    """Compact all-method summary of how much refitted weights move by window.

    Allocation distance is one-way turnover distance ``0.5 * sum(abs(w_a-w_b))``.
    A value of 0% means the two lookbacks produce identical starting allocations;
    100% means the capital is entirely reassigned.
    """
    if weights.empty or not isinstance(weights.index, pd.MultiIndex):
        return pd.DataFrame()

    rows: list[dict] = []
    methods = list(dict.fromkeys(weights.index.get_level_values("Method").tolist()))
    for method in methods:
        available = {}
        for label in ["3Y", "5Y", "Full"]:
            key = (label, method)
            if key in weights.index:
                available[label] = weights.loc[key].astype(float)
        if not available:
            continue
        full = available.get("Full")
        row = {"Portfolio": method}
        for label in ["3Y", "5Y", "Full"]:
            w = available.get(label)
            if w is None:
                row[f"Largest Weight {label}"] = np.nan
                row[f"Actual ETFs {label}"] = np.nan
            else:
                row[f"Largest Weight {label}"] = float(w.max())
                row[f"Actual ETFs {label}"] = int((w >= 0.0001).sum())
        if full is not None:
            for label in ["3Y", "5Y"]:
                w = available.get(label)
                row[f"Capital Change {label} vs Full"] = (
                    0.5 * float(np.abs(w - full).sum()) if w is not None else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows).set_index("Portfolio") if rows else pd.DataFrame()


def portfolio_metrics_by_lookback(
    returns: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    *,
    frequency: str,
    risk_free_rate: float,
    cvar_level: float,
    lookbacks: tuple[tuple[str, int | None], ...] = (
        ("1Y", 1),
        ("3Y", 3),
        ("5Y", 5),
        ("Full", None),
    ),
) -> pd.DataFrame:
    """Realised B&H statistics for fixed starting allocations across lookbacks."""
    rows: list[pd.DataFrame] = []
    for label, years in lookbacks:
        sample = trailing_window(returns, years).dropna(how="any")
        if len(sample) < 2:
            continue
        series = {}
        for name, weights in portfolios.items():
            usable = [
                t
                for t in weights.index
                if t in sample.columns and float(weights[t]) > 0
            ]
            if not usable:
                continue
            w = weights.reindex(usable).fillna(0.0).astype(float)
            if float(w.sum()) <= 0:
                continue
            w = w / float(w.sum())
            series[name] = buy_and_hold_portfolio_returns(sample[usable], w)
        if not series:
            continue
        frame = pd.DataFrame(series).dropna(how="all")
        metrics = summary_table(
            frame,
            frequency=frequency,
            risk_free_rate=risk_free_rate,
            cvar_level=cvar_level,
        )
        metrics.insert(0, "Lookback", label)
        metrics.insert(1, "Portfolio", metrics.index)
        rows.append(metrics.reset_index(drop=True))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).set_index(["Lookback", "Portfolio"])
