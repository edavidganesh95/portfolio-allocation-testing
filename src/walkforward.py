from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .diversification import HRP_NAME, hierarchical_risk_parity
from .expected_returns import market_anchored_expected_returns
from .metrics import annualized_return, annualized_volatility, max_drawdown
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


@dataclass(frozen=True)
class WalkForwardSettings:
    lookback_months: int | None = 36
    min_train_months: int = 36
    holding_months: int = 12
    risk_free_rate: float = 0.0
    expected_return_shrinkage: float = 0.50
    market_return_prior: float = 0.08
    max_weight: float = 0.70
    cvar_level: float = 0.95
    diversified_max_weight: float = 0.50
    diversified_min_effective_assets: float = 3.0
    diversified_max_risk_share: float = 0.50


@dataclass
class WalkForwardResult:
    # ``returns`` concatenates the independent OOS block return series so
    # existing metric helpers can inspect the same observations. It must not be
    # interpreted as one continuously rebalanced live portfolio.
    returns: pd.DataFrame
    wealth: pd.DataFrame
    weights: pd.DataFrame
    ending_weights: pd.DataFrame
    periods: pd.DataFrame
    block_outcomes: pd.DataFrame
    method_status: pd.DataFrame


CORE_METHODS = (
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


def _fit_methods(
    train_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    settings: WalkForwardSettings,
) -> dict[str, pd.Series]:
    mu = market_anchored_expected_returns(
        train_returns,
        benchmark_returns,
        market_return_prior=settings.market_return_prior,
        risk_free_rate=settings.risk_free_rate,
        frequency="monthly",
        shrinkage=settings.expected_return_shrinkage,
    )
    cov = ledoit_wolf_covariance(train_returns, frequency="monthly")

    results = [
        equal_weight(list(train_returns.columns)),
        minimum_variance(cov, max_weight=settings.max_weight),
        risk_parity(cov, max_weight=settings.max_weight),
        maximum_diversification(cov, max_weight=settings.max_weight),
        maximum_sharpe(
            mu,
            cov,
            risk_free_rate=settings.risk_free_rate,
            max_weight=settings.max_weight,
        ),
        diversified_maximum_sharpe(
            mu,
            cov,
            risk_free_rate=settings.risk_free_rate,
            max_weight=min(settings.max_weight, settings.diversified_max_weight),
            min_effective_assets=settings.diversified_min_effective_assets,
            max_risk_share=settings.diversified_max_risk_share,
        ),
        minimum_cvar(
            train_returns,
            level=settings.cvar_level,
            max_weight=settings.max_weight,
        ),
        maximum_return_to_cvar(
            mu,
            train_returns,
            risk_free_rate=settings.risk_free_rate,
            level=settings.cvar_level,
            max_weight=settings.max_weight,
        ),
    ]

    methods = {result.name: result.weights for result in results if result.success}
    # HRP is deliberately fitted from the same training-window covariance and
    # correlation matrix, but it does not use the expected-return vector.
    methods[HRP_NAME] = hierarchical_risk_parity(
        cov, train_returns.corr(), max_weight=settings.max_weight
    )

    return methods


def _normalize_weights(weights: pd.Series, columns: list[str]) -> pd.Series:
    w = weights.reindex(columns).fillna(0.0).astype(float).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive number.")
    return w / total


def _run_holding_block(
    test_returns: pd.DataFrame,
    starting_weights: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Buy once at block entry and let weights drift until block exit."""
    columns = list(test_returns.columns)
    start = _normalize_weights(starting_weights, columns)

    asset_growth = (1.0 + test_returns[columns].astype(float)).cumprod()
    asset_values = asset_growth.mul(start, axis=1)
    wealth_path = asset_values.sum(axis=1)

    previous_wealth = wealth_path.shift(1)
    previous_wealth.iloc[0] = 1.0
    series = (wealth_path / previous_wealth - 1.0).astype(float)

    end_wealth = float(wealth_path.iloc[-1])
    if end_wealth <= 0:
        end_weights = start.copy()
    else:
        end_weights = (asset_values.iloc[-1] / end_wealth).astype(float)

    return series, end_weights, wealth_path.astype(float)


def walk_forward_backtest(
    monthly_returns: pd.DataFrame,
    optimizer_assets: list[str],
    settings: WalkForwardSettings,
    reference_weights: pd.Series | None = None,
    benchmark_returns: pd.Series | None = None,
) -> WalkForwardResult:
    """Rolling entry-date validation for buy-and-hold portfolio construction.

    Every fold is an independent historical entry experiment:
      1. estimate the portfolio using only observations before the entry date;
      2. invest once at the fitted starting weights;
      3. hold through the OOS block with natural weight drift and no rebalance;
      4. move to the next entry date and repeat as a fresh experiment.

    The folds evaluate whether a *starting-allocation rule* generalizes without
    hindsight. They do not represent a live strategy that trades between folds.
    """
    if settings.min_train_months < 24:
        raise ValueError("min_train_months must be at least 24.")
    if settings.holding_months < 1:
        raise ValueError("holding_months must be positive.")

    optimizer_assets = [
        asset for asset in optimizer_assets if asset in monthly_returns.columns
    ]
    if len(optimizer_assets) < 2:
        raise ValueError("At least two optimizer assets are required.")
    if benchmark_returns is None or pd.Series(benchmark_returns).dropna().empty:
        raise ValueError(
            "Market benchmark returns are required for walk-forward fitting."
        )

    benchmark_returns = pd.Series(benchmark_returns, dtype=float).sort_index()
    returns = monthly_returns.dropna(how="any").copy()
    if len(returns) <= settings.min_train_months:
        raise ValueError("Not enough observations for walk-forward validation.")

    portfolio_names = list(CORE_METHODS)
    if reference_weights is not None:
        portfolio_names = ["Reference Portfolio"] + portfolio_names

    return_parts = {name: [] for name in portfolio_names}
    start_weight_rows: list[dict] = []
    end_weight_rows: list[dict] = []
    outcome_rows: list[dict] = []
    period_rows: list[dict] = []
    status_rows: list[dict] = []

    entry_idx = int(settings.min_train_months)
    period_id = 0

    while entry_idx < len(returns):
        train_end_idx = entry_idx
        if settings.lookback_months is None:
            train_start_idx = 0
        else:
            train_start_idx = max(0, train_end_idx - int(settings.lookback_months))

        train = returns.iloc[train_start_idx:train_end_idx][optimizer_assets].copy()
        if len(train) < settings.min_train_months:
            entry_idx += settings.holding_months
            continue

        test_end_idx = min(len(returns), entry_idx + int(settings.holding_months))
        test = returns.iloc[entry_idx:test_end_idx].copy()
        if test.empty:
            break

        train_benchmark = benchmark_returns.reindex(train.index).dropna()
        try:
            methods = _fit_methods(train, train_benchmark, settings)
        except ValueError:
            entry_idx = test_end_idx
            continue

        starting_map: dict[str, pd.Series] = {}
        if reference_weights is not None:
            starting_map["Reference Portfolio"] = reference_weights.copy()
        starting_map.update(methods)

        period_id += 1
        period_rows.append(
            {
                "Period": period_id,
                "Train Start": train.index.min(),
                "Train End": train.index.max(),
                "Test Start": test.index.min(),
                "Test End": test.index.max(),
                "Train Months": len(train),
                "Test Months": len(test),
            }
        )

        for name in portfolio_names:
            if name not in starting_map:
                status_rows.append(
                    {
                        "Period": period_id,
                        "Method": name,
                        "Success": False,
                        "Message": "Method unavailable for this training window.",
                    }
                )
                continue

            try:
                start = _normalize_weights(starting_map[name], list(test.columns))
                block_returns, end_weights, wealth_path = _run_holding_block(
                    test, start
                )
                return_parts[name].append(block_returns)

                for ticker, weight in start.items():
                    if weight > 1e-12:
                        start_weight_rows.append(
                            {
                                "Period": period_id,
                                "Method": name,
                                "Entry Date": test.index.min(),
                                "Ticker": ticker,
                                "Weight": float(weight),
                            }
                        )
                for ticker, weight in end_weights.items():
                    if weight > 1e-12:
                        end_weight_rows.append(
                            {
                                "Period": period_id,
                                "Method": name,
                                "Exit Date": test.index.max(),
                                "Ticker": ticker,
                                "Weight": float(weight),
                            }
                        )

                start_largest = float(start.max())
                end_largest = float(end_weights.max())
                drift = 0.5 * float(np.abs(end_weights - start).sum())
                outcome_rows.append(
                    {
                        "Period": period_id,
                        "Method": name,
                        "Entry Date": test.index.min(),
                        "Exit Date": test.index.max(),
                        "Holding Return": float(wealth_path.iloc[-1] - 1.0),
                        "Annualized Return": annualized_return(block_returns, 12),
                        "Volatility": annualized_volatility(block_returns, 12),
                        "Max Drawdown": max_drawdown(block_returns),
                        "Largest Start Holding": str(start.idxmax()),
                        "Start Largest Weight": start_largest,
                        "End Largest Weight": end_largest,
                        "Weight Drift": drift,
                        "Largest End Holding": str(end_weights.idxmax()),
                    }
                )
                status_rows.append(
                    {
                        "Period": period_id,
                        "Method": name,
                        "Success": True,
                        "Message": "OK",
                    }
                )
            except Exception as exc:
                status_rows.append(
                    {
                        "Period": period_id,
                        "Method": name,
                        "Success": False,
                        "Message": str(exc),
                    }
                )

        entry_idx = test_end_idx

    series_map: dict[str, pd.Series] = {}
    for name, parts in return_parts.items():
        if parts:
            series_map[name] = pd.concat(parts).sort_index()
    if not series_map:
        raise ValueError("No walk-forward portfolio completed successfully.")

    out_returns = pd.DataFrame(series_map).dropna(how="all")
    # Kept for backwards-compatible access only. Because each block is a fresh
    # entry experiment, this compounded series is not presented as a live B&H path.
    wealth = (1.0 + out_returns.fillna(0.0)).cumprod()

    return WalkForwardResult(
        returns=out_returns,
        wealth=wealth,
        weights=pd.DataFrame(start_weight_rows),
        ending_weights=pd.DataFrame(end_weight_rows),
        periods=pd.DataFrame(period_rows),
        block_outcomes=pd.DataFrame(outcome_rows),
        method_status=pd.DataFrame(status_rows),
    )


def block_win_counts(
    result: WalkForwardResult,
    benchmark: str = "Reference Portfolio",
) -> pd.DataFrame:
    """Per-method tally of independent OOS entry tests that beat the benchmark.

    ``Tests`` is the number of folds in which *both* the method and the
    benchmark produced a result. It can be smaller than the number of folds run
    when a method failed to fit in one of them, so a win count must always be
    quoted against this column rather than against the total fold count.
    """
    columns = ["Wins", "Tests", "Win Rate"]
    outcomes = result.block_outcomes
    if outcomes.empty or benchmark not in set(outcomes["Method"]):
        return pd.DataFrame(columns=columns, dtype=float)

    pivot = outcomes.pivot(index="Period", columns="Method", values="Holding Return")
    if benchmark not in pivot.columns:
        return pd.DataFrame(columns=columns, dtype=float)

    rows: dict[str, dict[str, float]] = {}
    for method in pivot.columns:
        if method == benchmark:
            continue
        pair = pivot[[method, benchmark]].dropna()
        if pair.empty:
            continue
        wins = int((pair[method] > pair[benchmark]).sum())
        tests = int(len(pair))
        rows[method] = {"Wins": wins, "Tests": tests, "Win Rate": wins / tests}
    if not rows:
        return pd.DataFrame(columns=columns, dtype=float)
    return pd.DataFrame.from_dict(rows, orient="index")[columns]


def block_win_rates(
    result: WalkForwardResult,
    benchmark: str = "Reference Portfolio",
) -> pd.Series:
    """Share of independent OOS entry experiments that beat the benchmark."""
    counts = block_win_counts(result, benchmark)
    if counts.empty:
        return pd.Series(dtype=float)
    return counts["Win Rate"].astype(float).rename("Block Win Rate")


def format_win_count(wins: float, tests: float) -> str:
    """``2 of 4 (50%)`` — the count and the rate always share one denominator."""
    tests = int(tests)
    if tests <= 0:
        return "—"
    wins = int(wins)
    return f"{wins} of {tests} ({wins / tests:.0%})"


def entry_outcome_summary(result: WalkForwardResult) -> pd.DataFrame:
    """Compact B&H validation summary across independent entry experiments."""
    if result.block_outcomes.empty:
        return pd.DataFrame()
    grouped = result.block_outcomes.groupby("Method")
    return pd.DataFrame(
        {
            "Median Annualized Return": grouped["Annualized Return"].median(),
            "Worst Annualized Return": grouped["Annualized Return"].min(),
            "Best Annualized Return": grouped["Annualized Return"].max(),
            "Median Volatility": grouped["Volatility"].median(),
            "Median Max Drawdown": grouped["Max Drawdown"].median(),
            "Blocks": grouped.size(),
        }
    )


def drift_summary(result: WalkForwardResult) -> pd.DataFrame:
    """How far buy-and-hold portfolios drift inside their OOS holding blocks."""
    if result.block_outcomes.empty:
        return pd.DataFrame()
    grouped = result.block_outcomes.groupby("Method")
    return pd.DataFrame(
        {
            "Median Start Largest Weight": grouped["Start Largest Weight"].median(),
            "Median End Largest Weight": grouped["End Largest Weight"].median(),
            "Maximum End Largest Weight": grouped["End Largest Weight"].max(),
            "Median Weight Drift": grouped["Weight Drift"].median(),
            "Blocks": grouped.size(),
        }
    )
