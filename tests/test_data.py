"""Period-completeness handling in the return pipeline."""

import pandas as pd
import pytest

from src.data import final_period_is_incomplete, to_returns


def _prices(start: str, end: str) -> pd.DataFrame:
    index = pd.bdate_range(start, end)
    base = pd.Series(range(len(index)), index=index, dtype=float) + 100.0
    return pd.DataFrame({"A": base, "B": base * 1.01})


def test_unfinished_month_is_dropped():
    prices = _prices("2026-06-01", "2026-09-21")

    assert final_period_is_incomplete(prices, "monthly")
    returns = to_returns(prices, "monthly")
    # June, July, August are whole months; the 21 September stub is excluded.
    assert returns.index.max() == pd.Timestamp("2026-08-31")
    assert len(returns) == 2  # July and August (June has no prior month)


def test_unfinished_month_can_be_kept_on_request():
    prices = _prices("2026-06-01", "2026-09-21")

    kept = to_returns(prices, "monthly", drop_incomplete=False)
    assert kept.index.max() == pd.Timestamp("2026-09-30")
    assert len(kept) == len(to_returns(prices, "monthly")) + 1


@pytest.mark.parametrize(
    "end",
    [
        "2026-09-30",  # ordinary weekday month end
        "2026-08-31",  # Monday
        "2025-08-29",  # month ends on a Sunday; Friday is the last session
        "2024-03-28",  # Good Friday on the 29th, so Thursday is the last session
    ],
)
def test_finished_months_are_never_mistaken_for_stubs(end):
    prices = _prices("2023-11-01", end)
    assert not final_period_is_incomplete(prices, "monthly")
    assert len(to_returns(prices, "monthly")) == len(
        to_returns(prices, "monthly", drop_incomplete=False)
    )


def test_unfinished_week_is_dropped():
    midweek = _prices("2026-08-03", "2026-09-23")  # Wednesday
    assert final_period_is_incomplete(midweek, "weekly")
    assert to_returns(midweek, "weekly").index.max() == pd.Timestamp("2026-09-18")

    friday = _prices("2026-08-03", "2026-09-25")
    assert not final_period_is_incomplete(friday, "weekly")


def test_daily_data_has_no_partial_period():
    prices = _prices("2026-06-01", "2026-09-21")
    assert not final_period_is_incomplete(prices, "daily")
    assert len(to_returns(prices, "daily")) == len(prices) - 1


def test_empty_prices_are_not_reported_incomplete():
    assert not final_period_is_incomplete(pd.DataFrame(), "monthly")
