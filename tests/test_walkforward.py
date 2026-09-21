import numpy as np
import pandas as pd

from src.walkforward import (
    WalkForwardSettings,
    block_win_rates,
    drift_summary,
    entry_outcome_summary,
    walk_forward_backtest,
)


def _returns():
    idx = pd.date_range("2015-01-31", periods=132, freq="ME")
    x = np.linspace(0, 20, len(idx))
    return pd.DataFrame(
        {
            "A": 0.008 + 0.025 * np.sin(x),
            "B": 0.005 + 0.012 * np.cos(x * 0.7),
            "C": 0.006 + 0.019 * np.sin(x * 1.1 + 0.2),
        },
        index=idx,
    )


def _benchmark():
    returns = _returns()
    return (0.7 * returns["A"] + 0.3 * returns["B"]).rename("MKT")


def _settings(holding_months=12):
    return WalkForwardSettings(
        lookback_months=36,
        min_train_months=36,
        holding_months=holding_months,
        max_weight=0.80,
    )


def test_walk_forward_starts_after_training_window():
    result = walk_forward_backtest(
        _returns(),
        optimizer_assets=["A", "B", "C"],
        settings=_settings(),
        benchmark_returns=_benchmark(),
    )
    assert result.returns.index.min() == _returns().index[36]
    assert len(result.periods) == 8


def test_reference_and_methods_share_oos_calendar():
    result = walk_forward_backtest(
        _returns(),
        optimizer_assets=["A", "B", "C"],
        settings=_settings(),
        reference_weights=pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}),
        benchmark_returns=_benchmark(),
    )
    assert "Reference Portfolio" in result.returns.columns
    assert "Minimum Variance" in result.returns.columns
    assert "HRP Diversification" in result.returns.columns
    assert result.returns["Reference Portfolio"].notna().sum() > 0
    assert result.returns["Minimum Variance"].notna().sum() > 0
    assert result.returns["HRP Diversification"].notna().sum() > 0


def test_walk_forward_records_natural_weight_drift_without_recurring_trades():
    result = walk_forward_backtest(
        _returns(),
        ["A", "B", "C"],
        _settings(12),
        reference_weights=pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}),
        benchmark_returns=_benchmark(),
    )
    assert not result.block_outcomes.empty
    assert "Weight Drift" in result.block_outcomes.columns
    assert (result.block_outcomes["Weight Drift"] >= 0).all()
    assert result.block_outcomes["Weight Drift"].max() > 0


def test_entry_summary_drift_summary_and_block_win_rate():
    result = walk_forward_backtest(
        _returns(),
        ["A", "B", "C"],
        _settings(),
        reference_weights=pd.Series({"A": 0.4, "B": 0.4, "C": 0.2}),
        benchmark_returns=_benchmark(),
    )
    entries = entry_outcome_summary(result)
    drift = drift_summary(result)
    wins = block_win_rates(result)

    assert "Median Annualized Return" in entries.columns
    assert "Median Weight Drift" in drift.columns
    assert not wins.empty
    assert ((wins >= 0) & (wins <= 1)).all()


def test_each_fold_is_a_fresh_entry_experiment():
    result = walk_forward_backtest(
        _returns(),
        ["A", "B", "C"],
        _settings(12),
        reference_weights=pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}),
        benchmark_returns=_benchmark(),
    )
    ref = result.block_outcomes[
        result.block_outcomes["Method"] == "Reference Portfolio"
    ]
    assert len(ref) == len(result.periods)
    # The reference starts each independent fold from the same requested weights;
    # its largest starting weight is therefore always 50%, even though endings drift.
    assert np.allclose(ref["Start Largest Weight"], 0.5)
    assert not np.allclose(ref["End Largest Weight"], 0.5)


def _outcomes(rows):
    """A WalkForwardResult carrying only the block outcomes."""
    from src.walkforward import WalkForwardResult

    frame = pd.DataFrame(rows, columns=["Period", "Method", "Holding Return"])
    return WalkForwardResult(
        returns=pd.DataFrame(),
        wealth=pd.DataFrame(),
        weights=pd.DataFrame(),
        ending_weights=pd.DataFrame(),
        periods=pd.DataFrame({"Period": sorted(frame["Period"].unique())}),
        block_outcomes=frame,
        method_status=pd.DataFrame(),
    )


def test_win_counts_quote_the_tests_a_method_actually_took_part_in():
    from src.walkforward import block_win_counts, format_win_count

    # X fails to fit in fold 3, so it has four tests, not five.
    rows = [(p, "Reference Portfolio", 0.05) for p in range(1, 6)]
    rows += [(p, "X", 0.10) for p in (1, 2, 4, 5)]
    counts = block_win_counts(_outcomes(rows))

    assert counts.loc["X", "Wins"] == 4
    assert counts.loc["X", "Tests"] == 4
    assert counts.loc["X", "Win Rate"] == 1.0
    assert format_win_count(counts.loc["X", "Wins"], counts.loc["X", "Tests"]) == (
        "4 of 4 (100%)"
    )


def test_win_counts_and_rates_agree():
    from src.walkforward import block_win_counts

    result = walk_forward_backtest(
        _returns(),
        optimizer_assets=["A", "B", "C"],
        settings=_settings(),
        reference_weights=pd.Series({"A": 0.5, "B": 0.3, "C": 0.2}),
        benchmark_returns=_benchmark(),
    )
    counts = block_win_counts(result)
    rates = block_win_rates(result)

    assert list(counts.index) == list(rates.index)
    assert np.allclose(counts["Win Rate"], rates)
    assert np.allclose(counts["Wins"] / counts["Tests"], counts["Win Rate"])
    assert (counts["Tests"] <= len(result.periods)).all()


def test_win_counts_need_a_reference():
    from src.walkforward import block_win_counts

    result = walk_forward_backtest(
        _returns(),
        optimizer_assets=["A", "B", "C"],
        settings=_settings(),
        benchmark_returns=_benchmark(),
    )
    assert block_win_counts(result).empty
    assert block_win_rates(result).empty


def test_format_win_count_handles_no_tests():
    from src.walkforward import format_win_count

    assert format_win_count(0, 0) == "—"
    assert format_win_count(2, 4) == "2 of 4 (50%)"
