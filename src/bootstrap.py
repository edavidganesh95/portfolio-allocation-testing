from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BootstrapSettings:
    simulations: int = 10_000
    horizon_years: int = 10
    block_months: int = 3
    seed: int = 42
    initial_wealth: float = 100_000.0


def moving_block_indices(
    n_observations: int,
    periods: int,
    simulations: int,
    block_size: int,
    seed: int,
) -> np.ndarray:
    """Circular moving-block bootstrap indices."""
    if n_observations < 2:
        raise ValueError("At least two historical observations are required.")
    if periods < 1 or simulations < 1:
        raise ValueError("periods and simulations must be positive.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")

    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(periods / block_size))
    starts = rng.integers(0, n_observations, size=(simulations, blocks_needed))
    offsets = np.arange(block_size)
    blocks = (starts[:, :, None] + offsets[None, None, :]) % n_observations
    return blocks.reshape(simulations, -1)[:, :periods].astype(np.int32, copy=False)


def historical_portfolio_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    weights = weights.reindex(asset_returns.columns).fillna(0.0).astype(float)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    weights = weights / total
    return asset_returns.mul(weights, axis=1).sum(axis=1)


def simulate_portfolio_paths(
    historical_returns: pd.Series,
    indices: np.ndarray,
    initial_wealth: float = 100_000.0,
) -> dict[str, np.ndarray]:
    source = historical_returns.dropna().to_numpy(dtype=float)
    if source.size == 0:
        raise ValueError("Historical portfolio return series is empty.")

    sampled = source[indices]
    growth = np.cumprod(1.0 + sampled, axis=1)
    wealth = float(initial_wealth) * growth

    peak = np.maximum.accumulate(growth, axis=1)
    drawdowns = growth / peak - 1.0
    max_drawdown = np.min(drawdowns, axis=1)

    years = indices.shape[1] / 12.0
    terminal_growth = growth[:, -1]
    cagr = np.power(terminal_growth, 1.0 / years) - 1.0

    return {
        "wealth": wealth,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
    }


def summarize_simulation(
    result: dict[str, np.ndarray],
    initial_wealth: float,
    target_cagrs: tuple[float, ...] = (0.06, 0.08, 0.10),
) -> dict[str, float]:
    terminal = result["wealth"][:, -1]
    cagr = result["cagr"]
    max_dd = result["max_drawdown"]

    row = {
        "Median Terminal Wealth": float(np.median(terminal)),
        "5th Percentile Wealth": float(np.percentile(terminal, 5)),
        "25th Percentile Wealth": float(np.percentile(terminal, 25)),
        "75th Percentile Wealth": float(np.percentile(terminal, 75)),
        "95th Percentile Wealth": float(np.percentile(terminal, 95)),
        "Median CAGR": float(np.median(cagr)),
        "5th Percentile CAGR": float(np.percentile(cagr, 5)),
        "P(Loss at Horizon)": float(np.mean(terminal < initial_wealth)),
        "Median Max Drawdown": float(np.median(max_dd)),
        "5th Percentile Max Drawdown": float(np.percentile(max_dd, 5)),
    }
    for target in target_cagrs:
        row[f"P(CAGR > {target:.0%})"] = float(np.mean(cagr > target))
    return row


def simulation_fan(result: dict[str, np.ndarray]) -> pd.DataFrame:
    wealth = result["wealth"]
    # One call with all five quantiles: NumPy partitions the array once and reads
    # every percentile from it, instead of re-partitioning per percentile. The
    # values are identical to five separate calls.
    five, twenty_five, median, seventy_five, ninety_five = np.percentile(
        wealth, [5, 25, 50, 75, 95], axis=0
    )
    return pd.DataFrame(
        {
            "Month": np.arange(1, wealth.shape[1] + 1),
            "5th": five,
            "25th": twenty_five,
            "Median": median,
            "75th": seventy_five,
            "95th": ninety_five,
        }
    )


def simulate_buy_and_hold_paths(
    sampled_asset_growth: np.ndarray,
    weights: pd.Series,
    columns: list[str],
    initial_wealth: float = 100_000.0,
) -> dict[str, np.ndarray]:
    """Evaluate one initial allocation with natural weight drift.

    ``sampled_asset_growth`` has shape simulations x periods x assets and is
    shared by every portfolio in the comparison. No portfolio is rebalanced
    after inception.
    """
    w = _weight_vector(weights, columns)

    portfolio_growth = np.tensordot(sampled_asset_growth, w, axes=([2], [0]))
    wealth = float(initial_wealth) * portfolio_growth
    peak = np.maximum.accumulate(portfolio_growth, axis=1)
    drawdowns = portfolio_growth / peak - 1.0
    max_drawdown = np.min(drawdowns, axis=1)

    years = sampled_asset_growth.shape[1] / 12.0
    terminal_growth = portfolio_growth[:, -1]
    cagr = np.power(terminal_growth, 1.0 / years) - 1.0

    return {
        "wealth": wealth,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
    }


def _prepare_bootstrap_inputs(
    monthly_asset_returns: pd.DataFrame,
    portfolios: dict[str, pd.Series],
) -> tuple[list[str], pd.DataFrame]:
    """Holdings actually used by any portfolio, on their common monthly sample."""
    if not portfolios:
        raise ValueError("At least one portfolio is required.")

    active_assets: list[str] = []
    for weights in portfolios.values():
        for ticker, value in pd.Series(weights, dtype=float).items():
            if (
                float(value) > 1e-12
                and ticker in monthly_asset_returns.columns
                and ticker not in active_assets
            ):
                active_assets.append(ticker)
    if not active_assets:
        raise ValueError("No portfolio holdings are present in the return data.")

    aligned = monthly_asset_returns[active_assets].astype(float).dropna(how="any")
    if len(aligned) < 2:
        raise ValueError("At least two common monthly observations are required.")
    return active_assets, aligned


def _weight_vector(weights: pd.Series, columns: list[str]) -> np.ndarray:
    w = weights.reindex(columns).fillna(0.0).astype(float).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    return (w / total).to_numpy(dtype=float)


def run_bootstrap_comparison(
    monthly_asset_returns: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    settings: BootstrapSettings,
    target_cagrs: tuple[float, ...] = (0.06, 0.08, 0.10),
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], np.ndarray]:
    """Moving-block bootstrap for true buy-and-hold portfolios.

    Every portfolio receives the same resampled asset-return blocks. Holdings
    are purchased once at the starting weights and then compound independently;
    weights drift naturally for the full simulated horizon.

    This returns the complete wealth path of every portfolio, which makes it the
    reference implementation and the right tool for small runs and tests. Its
    memory grows as ``paths x months x assets``; the application uses
    :func:`run_bootstrap_reduced`, which produces the same numbers inside a fixed
    memory budget.
    """
    active_assets, aligned = _prepare_bootstrap_inputs(
        monthly_asset_returns, portfolios
    )

    periods = int(settings.horizon_years * 12)
    indices = moving_block_indices(
        n_observations=len(aligned),
        periods=periods,
        simulations=settings.simulations,
        block_size=settings.block_months,
        seed=settings.seed,
    )

    source = aligned.to_numpy(dtype=float)
    sampled = source[indices]
    sampled_asset_growth = np.cumprod(1.0 + sampled, axis=1)

    outputs: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for name, weights in portfolios.items():
        result = simulate_buy_and_hold_paths(
            sampled_asset_growth,
            weights,
            active_assets,
            initial_wealth=settings.initial_wealth,
        )
        outputs[name] = result
        rows.append(
            {
                "Portfolio": name,
                **summarize_simulation(
                    result,
                    initial_wealth=settings.initial_wealth,
                    target_cagrs=target_cagrs,
                ),
            }
        )

    return pd.DataFrame(rows).set_index("Portfolio"), outputs, indices


@dataclass
class BootstrapReduced:
    """What the bootstrap section shows, without the full wealth paths."""

    summary: pd.DataFrame
    fans: dict[str, pd.DataFrame]
    terminal_wealth: dict[str, np.ndarray]
    cagr: dict[str, np.ndarray]
    max_drawdown: dict[str, np.ndarray]


#: Working-memory ceiling for the streaming bootstrap, in megabytes. Chosen so a
#: default run (10,000 paths, 10 years) still finishes in a single pass while the
#: largest selectable run (25,000 paths, 20 years) stays well inside a small
#: hosted instance instead of needing roughly 1.8 GB.
DEFAULT_MEMORY_BUDGET_MB = 320.0
_CHUNK_BUDGET_BYTES = 96e6


def _reduce_paths(
    growth: np.ndarray,
    initial_wealth: float,
    target_cagrs: tuple[float, ...],
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Collapse one portfolio's growth paths to everything the section displays.

    Works in place on ``growth`` and consumes it. The maths is exactly that of
    :func:`simulate_buy_and_hold_paths`, but no second full-size wealth, drawdown
    or peak array is ever alive alongside it.
    """
    periods = growth.shape[1]
    years = periods / 12.0
    cagr = np.power(growth[:, -1], 1.0 / years) - 1.0

    # Drawdown from the running peak, reusing the peak buffer for the result.
    peak = np.maximum.accumulate(growth, axis=1)
    np.divide(growth, peak, out=peak)
    peak -= 1.0
    max_drawdown = np.min(peak, axis=1)
    del peak

    growth *= float(initial_wealth)  # growth of $1 -> wealth, in place
    terminal = np.ascontiguousarray(growth[:, -1])
    fan = simulation_fan({"wealth": growth})
    row = summarize_simulation(
        {"wealth": terminal[:, None], "cagr": cagr, "max_drawdown": max_drawdown},
        initial_wealth=initial_wealth,
        target_cagrs=target_cagrs,
    )
    return row, fan, terminal, np.ascontiguousarray(cagr), max_drawdown


def run_bootstrap_reduced(
    monthly_asset_returns: pd.DataFrame,
    portfolios: dict[str, pd.Series],
    settings: BootstrapSettings,
    target_cagrs: tuple[float, ...] = (0.06, 0.08, 0.10),
    memory_budget_mb: float = DEFAULT_MEMORY_BUDGET_MB,
) -> BootstrapReduced:
    """Moving-block bootstrap with bounded memory.

    Produces the same numbers as :func:`run_bootstrap_comparison`, but streams
    the simulation instead of materialising a ``paths x months x assets`` tensor:

    * the resampled asset paths are generated in chunks of simulations and folded
      straight into each portfolio's growth path, so the asset dimension never
      exists at full size;
    * only as many portfolios as fit in ``memory_budget_mb`` are held at once, and
      each is reduced to its summary, fan and three distributions the moment it
      is complete.

    Every portfolio still sees the same resampled blocks (the indices are drawn
    once), so relative comparisons remain path-for-path consistent. A run that
    fits the budget takes a single pass and costs what it did before; only very
    large runs need several passes over the chunks.
    """
    active_assets, aligned = _prepare_bootstrap_inputs(
        monthly_asset_returns, portfolios
    )
    periods = int(settings.horizon_years * 12)
    simulations = int(settings.simulations)
    indices = moving_block_indices(
        n_observations=len(aligned),
        periods=periods,
        simulations=simulations,
        block_size=settings.block_months,
        seed=settings.seed,
    )
    source = aligned.to_numpy(dtype=float)
    n_assets = source.shape[1]

    names = list(portfolios)
    weight_matrix = np.column_stack(
        [_weight_vector(portfolios[name], active_assets) for name in names]
    )

    path_bytes = simulations * periods * 8
    # Each portfolio in flight holds one growth path, and the percentile step
    # needs one more path-sized scratch buffer.
    group_size = max(1, int(memory_budget_mb * 1e6 // (2 * path_bytes)))
    chunk = max(1, int(_CHUNK_BUDGET_BYTES // (3 * periods * n_assets * 8)))

    rows: dict[str, dict[str, float]] = {}
    fans: dict[str, pd.DataFrame] = {}
    terminal_wealth: dict[str, np.ndarray] = {}
    cagr: dict[str, np.ndarray] = {}
    max_drawdown: dict[str, np.ndarray] = {}

    for first in range(0, len(names), group_size):
        members = list(range(first, min(first + group_size, len(names))))
        group_weights = weight_matrix[:, members]
        paths = [np.empty((simulations, periods)) for _ in members]

        for start in range(0, simulations, chunk):
            stop = min(start + chunk, simulations)
            asset_growth = np.cumprod(1.0 + source[indices[start:stop]], axis=1)
            block = np.tensordot(asset_growth, group_weights, axes=([2], [0]))
            for slot, path in enumerate(paths):
                path[start:stop] = block[:, :, slot]
            del asset_growth, block

        for slot, member in enumerate(members):
            name = names[member]
            row, fan, terminal, path_cagr, path_drawdown = _reduce_paths(
                paths[slot], settings.initial_wealth, target_cagrs
            )
            paths[slot] = None  # release before the next portfolio is reduced
            rows[name] = row
            fans[name] = fan
            terminal_wealth[name] = terminal
            cagr[name] = path_cagr
            max_drawdown[name] = path_drawdown

    summary = pd.DataFrame(
        [{"Portfolio": name, **rows[name]} for name in names]
    ).set_index("Portfolio")
    return BootstrapReduced(
        summary=summary,
        fans=fans,
        terminal_wealth=terminal_wealth,
        cagr=cagr,
        max_drawdown=max_drawdown,
    )
