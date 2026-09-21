import numpy as np
import pandas as pd

from src.diversification import (
    HRP_NAME,
    cluster_allocation_frame,
    correlation_clusters,
    diversification_improvements,
    diversification_profile,
    hierarchical_risk_parity,
)


def _inputs():
    tickers = ["A", "B", "C", "D", "E"]
    corr = pd.DataFrame(
        [
            [1.00, 0.95, 0.25, 0.20, 0.15],
            [0.95, 1.00, 0.22, 0.18, 0.16],
            [0.25, 0.22, 1.00, 0.82, 0.30],
            [0.20, 0.18, 0.82, 1.00, 0.28],
            [0.15, 0.16, 0.30, 0.28, 1.00],
        ],
        index=tickers,
        columns=tickers,
    )
    vol = np.array([0.20, 0.19, 0.16, 0.15, 0.18])
    cov = pd.DataFrame(corr.values * np.outer(vol, vol), index=tickers, columns=tickers)
    return corr, cov


def test_hrp_is_long_only_fully_invested_and_respects_cap():
    corr, cov = _inputs()
    w = hierarchical_risk_parity(cov, corr, max_weight=0.40)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-12).all()
    assert (w <= 0.400001).all()


def test_clusters_group_highly_correlated_pairs():
    corr, _ = _inputs()
    clusters = correlation_clusters(corr, max_clusters=4)
    assert clusters["A"] == clusters["B"]
    assert clusters["C"] == clusters["D"]


def test_diversification_profile_has_plain_english_metrics():
    corr, cov = _inputs()
    clusters = correlation_clusters(corr, max_clusters=4)
    hrp = hierarchical_risk_parity(cov, corr, max_weight=0.60)
    concentrated = pd.Series({"A": 0.80, "B": 0.10, "C": 0.05, "D": 0.03, "E": 0.02})
    profile = diversification_profile(
        {"Reference Portfolio": concentrated, HRP_NAME: hrp},
        cov,
        corr,
        clusters,
    )
    assert {
        "Actual ETFs",
        "Effective ETFs",
        "Portfolio Volatility",
        "Weighted Correlation",
        "Diversification Ratio",
        "Largest Risk Contributor",
        "Largest Risk Share",
        "Dominant Cluster",
        "Largest Cluster Allocation",
    }.issubset(profile.columns)
    assert profile.loc[HRP_NAME, "Effective ETFs"] > 1.0
    assert profile.loc["Reference Portfolio", "Actual ETFs"] == 5
    assert profile.loc["Reference Portfolio", "Portfolio Volatility"] > 0


def test_cluster_allocation_sums_to_one_for_every_portfolio():
    corr, cov = _inputs()
    clusters = correlation_clusters(corr, max_clusters=4)
    hrp = hierarchical_risk_parity(cov, corr, max_weight=0.60)
    frame = cluster_allocation_frame({HRP_NAME: hrp}, clusters)
    assert np.isclose(frame.groupby("Portfolio")["Weight"].sum().iloc[0], 1.0)


def test_improvement_counter_is_bounded():
    corr, cov = _inputs()
    clusters = correlation_clusters(corr, max_clusters=4)
    hrp = hierarchical_risk_parity(cov, corr, max_weight=0.60)
    reference = pd.Series({"A": 0.80, "B": 0.10, "C": 0.05, "D": 0.03, "E": 0.02})
    profile = diversification_profile(
        {"Reference Portfolio": reference, HRP_NAME: hrp}, cov, corr, clusters
    )
    passed, total = diversification_improvements(
        profile, "Reference Portfolio", HRP_NAME
    )
    assert 0 <= passed <= total == 5


def test_effective_risk_bets_detects_concentration():
    from src.diversification import effective_risk_bets

    cov = pd.DataFrame(
        [[0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    balanced = effective_risk_bets(
        pd.Series([1 / 3, 1 / 3, 1 / 3], index=cov.index), cov
    )
    concentrated = effective_risk_bets(
        pd.Series([0.9, 0.05, 0.05], index=cov.index), cov
    )
    assert balanced > concentrated
    assert np.isclose(balanced, 3.0)
