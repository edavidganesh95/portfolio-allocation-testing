"""Return-based diversification diagnostics and Hierarchical Risk Parity (HRP).

This module deliberately stays within the project's returns-only design. It does
not inspect ETF constituents. Instead it asks whether the *return streams* behave
like distinct economic bets, and constructs an HRP counterfactual that allocates
around the observed correlation/risk hierarchy without using expected returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from .risk import portfolio_volatility, risk_contributions

HRP_NAME = "HRP Diversification"


def _normalise(weights: pd.Series, index: list[str]) -> pd.Series:
    w = weights.reindex(index).fillna(0.0).astype(float).clip(lower=0.0)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must sum to a positive value.")
    return w / total


def correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
    """Distance matrix used by the return-clustering layer.

    The transformation is monotonic in correlation and matches the ordering used
    by the app's existing correlation heatmap: perfectly correlated assets have
    distance 0, while lower correlation means greater distance.
    """
    values = (1.0 - corr.to_numpy(float)) / 2.0
    np.fill_diagonal(values, 0.0)
    values = np.clip((values + values.T) / 2.0, 0.0, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)


def correlation_linkage(corr: pd.DataFrame):
    """Average-linkage hierarchy over the correlation-distance matrix."""
    if len(corr) < 2:
        raise ValueError("At least two assets are required for clustering.")
    distance = correlation_distance(corr)
    linked = linkage(squareform(distance.values, checks=False), method="average")
    return linked, distance


def correlation_clusters(
    corr: pd.DataFrame,
    max_clusters: int = 6,
) -> pd.Series:
    """Choose a compact return-cluster partition using silhouette quality.

    Candidate cuts from 2 to ``max_clusters`` are compared on the precomputed
    correlation-distance matrix. Cluster labels are then re-numbered in leaf
    order so the same hierarchy is easy to follow in tables and charts.
    """
    labels = list(corr.columns)
    n = len(labels)
    if n == 1:
        return pd.Series(["Cluster 1"], index=labels, name="Cluster")
    if n == 2:
        return pd.Series(["Cluster 1", "Cluster 2"], index=labels, name="Cluster")

    linked, distance = correlation_linkage(corr)
    best_labels = None
    best_score = -np.inf

    for k in range(2, min(int(max_clusters), n - 1) + 1):
        raw = fcluster(linked, t=k, criterion="maxclust")
        unique = np.unique(raw)
        if len(unique) < 2 or len(unique) >= n:
            continue
        try:
            score = float(silhouette_score(distance.values, raw, metric="precomputed"))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_labels = raw

    if best_labels is None:
        best_labels = fcluster(linked, t=min(3, n - 1), criterion="maxclust")

    # Stable human-facing numbering: clusters appear in dendrogram leaf order.
    order = leaves_list(linked)
    mapping: dict[int, str] = {}
    next_id = 1
    for idx in order:
        raw_label = int(best_labels[idx])
        if raw_label not in mapping:
            mapping[raw_label] = f"Cluster {next_id}"
            next_id += 1

    return pd.Series(
        [mapping[int(label)] for label in best_labels],
        index=labels,
        name="Cluster",
    )


def _cluster_variance(cov: pd.DataFrame, assets: list[str]) -> float:
    sub = cov.loc[assets, assets]
    diag = np.diag(sub.values).astype(float)
    inv = np.divide(1.0, diag, out=np.zeros_like(diag), where=diag > 0)
    if float(inv.sum()) <= 0:
        ivp = np.repeat(1.0 / len(assets), len(assets))
    else:
        ivp = inv / inv.sum()
    return float(ivp @ sub.values @ ivp)


def _project_with_cap(weights: pd.Series, max_weight: float) -> pd.Series:
    """Nearest fully invested long-only vector respecting ``max_weight``."""
    n = len(weights)
    if n * max_weight < 1.0 - 1e-12:
        raise ValueError(
            f"Maximum weight {max_weight:.1%} is infeasible for {n} assets."
        )
    w = weights.to_numpy(float)
    if float(w.max()) <= max_weight + 1e-10:
        return weights / float(weights.sum())

    x0 = np.clip(w, 0.0, max_weight)
    if float(x0.sum()) <= 0:
        x0 = np.repeat(1.0 / n, n)
    else:
        x0 = x0 / float(x0.sum())

    res = minimize(
        lambda x: float(np.sum((x - w) ** 2)),
        x0,
        method="SLSQP",
        bounds=[(0.0, float(max_weight))] * n,
        constraints=({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},),
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not res.success:
        raise RuntimeError(f"Could not apply the HRP weight cap: {res.message}")
    return pd.Series(res.x, index=weights.index)


def hierarchical_risk_parity(
    cov: pd.DataFrame,
    corr: pd.DataFrame,
    max_weight: float = 1.0,
) -> pd.Series:
    """Construct a Hierarchical Risk Parity portfolio.

    1. Cluster assets by return correlation.
    2. Quasi-diagonalise the covariance matrix using dendrogram leaf order.
    3. Recursively split the ordered tree and allocate more capital to the
       lower-variance branch at each split.
    4. Project to the project's maximum-weight constraint if necessary.

    No expected-return estimate enters the calculation.
    """
    tickers = [ticker for ticker in corr.columns if ticker in cov.index]
    cov = cov.loc[tickers, tickers]
    corr = corr.loc[tickers, tickers]
    linked, _ = correlation_linkage(corr)
    ordered = [tickers[i] for i in leaves_list(linked)]

    weights = pd.Series(1.0, index=ordered, dtype=float)
    clusters: list[list[str]] = [ordered]

    while clusters:
        next_clusters: list[list[str]] = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            split = len(cluster) // 2
            left = cluster[:split]
            right = cluster[split:]
            next_clusters.extend([left, right])

            var_left = _cluster_variance(cov, left)
            var_right = _cluster_variance(cov, right)
            denom = var_left + var_right
            alpha = 0.5 if denom <= 0 else 1.0 - var_left / denom
            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha
        clusters = next_clusters

    weights = weights.reindex(tickers).fillna(0.0)
    weights = weights / float(weights.sum())
    return _project_with_cap(weights, max_weight=max_weight)


def weighted_average_correlation(weights: pd.Series, corr: pd.DataFrame) -> float:
    tickers = list(corr.index)
    w = _normalise(weights, tickers).to_numpy(float)
    values = corr.loc[tickers, tickers].to_numpy(float)
    outer = np.outer(w, w)
    mask = np.triu(np.ones_like(values, dtype=bool), k=1)
    denom = float(outer[mask].sum())
    if denom <= 1e-15:
        return 1.0
    return float((outer[mask] * values[mask]).sum() / denom)


def diversification_ratio(weights: pd.Series, cov: pd.DataFrame) -> float:
    tickers = list(cov.index)
    w = _normalise(weights, tickers).to_numpy(float)
    asset_vol = np.sqrt(np.diag(cov.loc[tickers, tickers].to_numpy(float)))
    port_vol = portfolio_volatility(w, cov.loc[tickers, tickers].to_numpy(float))
    if port_vol <= 0:
        return float("nan")
    return float(w @ asset_vol / port_vol)


def actual_etf_count(
    weights: pd.Series,
    universe: list[str],
    threshold: float = 0.0001,
) -> int:
    """Number of economically meaningful holdings.

    Optimisers often return tiny numerical residues rather than exact zeros.  The
    user-facing "Actual ETFs" count therefore ignores only numerical dust below 0.01%
    by default.  The threshold is deliberately explicit so the count
    is reproducible rather than dependent on floating-point noise.
    """
    w = _normalise(weights, universe)
    return int((w >= float(threshold)).sum())


def effective_etf_count(weights: pd.Series, universe: list[str]) -> float:
    w = _normalise(weights, universe).to_numpy(float)
    return float(1.0 / np.sum(w**2))


def effective_risk_bets(weights: pd.Series, cov: pd.DataFrame) -> float:
    """Inverse-Herfindahl count of portfolio variance-contribution shares.

    This is the risk analogue of ``effective_etf_count``: a portfolio whose
    volatility is effectively driven by one ETF approaches 1, while more evenly
    distributed variance contributions produce a larger value.
    """
    tickers = list(cov.index)
    w = _normalise(weights, tickers)
    sigma2 = float(
        w.to_numpy(float)
        @ cov.loc[tickers, tickers].to_numpy(float)
        @ w.to_numpy(float)
    )
    if sigma2 <= 1e-16:
        return float("nan")
    shares = (
        w.to_numpy(float)
        * (cov.loc[tickers, tickers].to_numpy(float) @ w.to_numpy(float))
        / sigma2
    )
    # With long-only equity portfolios these are normally positive. Clip tiny
    # numerical negatives before computing a concentration index.
    shares = np.clip(shares, 0.0, None)
    total = float(shares.sum())
    if total <= 1e-16:
        return float("nan")
    shares = shares / total
    return float(1.0 / np.sum(shares**2))


def cluster_allocation_frame(
    portfolios: dict[str, pd.Series],
    clusters: pd.Series,
) -> pd.DataFrame:
    tickers = list(clusters.index)
    rows = []
    for portfolio, weights in portfolios.items():
        w = _normalise(weights, tickers)
        grouped = w.groupby(clusters).sum()
        for cluster, value in grouped.items():
            rows.append(
                {
                    "Portfolio": portfolio,
                    "Cluster": str(cluster),
                    "Weight": float(value),
                }
            )
    return pd.DataFrame(rows)


def cluster_membership_frame(clusters: pd.Series) -> pd.DataFrame:
    rows = []
    for cluster, members in clusters.groupby(clusters):
        rows.append(
            {
                "Cluster": str(cluster),
                "ETFs": ", ".join(members.index.tolist()),
                "ETF Count": int(len(members)),
            }
        )
    return pd.DataFrame(rows).set_index("Cluster")


def diversification_profile(
    portfolios: dict[str, pd.Series],
    cov: pd.DataFrame,
    corr: pd.DataFrame,
    clusters: pd.Series,
) -> pd.DataFrame:
    """One-row-per-portfolio diversification diagnostics."""
    tickers = list(cov.index)
    rows = []

    for name, raw_weights in portfolios.items():
        w = _normalise(raw_weights, tickers)
        rc = risk_contributions(w, cov.loc[tickers, tickers])
        rc_total = float(rc.sum())
        risk_share = rc / rc_total if abs(rc_total) > 1e-15 else rc * 0.0
        largest_risk_asset = str(risk_share.idxmax())
        largest_risk_share = float(risk_share.max())

        cluster_weights = w.groupby(clusters.reindex(tickers)).sum()
        largest_cluster = str(cluster_weights.idxmax())
        largest_cluster_weight = float(cluster_weights.max())

        rows.append(
            {
                "Portfolio": name,
                "Actual ETFs": actual_etf_count(w, tickers),
                "Effective ETFs": effective_etf_count(w, tickers),
                "Portfolio Volatility": portfolio_volatility(
                    w.to_numpy(float), cov.loc[tickers, tickers].to_numpy(float)
                ),
                "Effective Risk Bets": effective_risk_bets(w, cov),
                "Weighted Correlation": weighted_average_correlation(w, corr),
                "Diversification Ratio": diversification_ratio(w, cov),
                "Largest Risk Contributor": largest_risk_asset,
                "Largest Risk Share": largest_risk_share,
                "Dominant Cluster": largest_cluster,
                "Largest Cluster Allocation": largest_cluster_weight,
            }
        )

    return pd.DataFrame(rows).set_index("Portfolio")


def diversification_improvements(
    profile: pd.DataFrame,
    reference: str,
    alternative: str,
) -> tuple[int, int]:
    """Count how many core diversification diagnostics improve vs reference."""
    if reference not in profile.index or alternative not in profile.index:
        return 0, 0
    ref = profile.loc[reference]
    alt = profile.loc[alternative]
    checks = [
        alt["Effective ETFs"] > ref["Effective ETFs"],
        alt["Effective Risk Bets"] > ref["Effective Risk Bets"],
        alt["Weighted Correlation"] < ref["Weighted Correlation"],
        alt["Diversification Ratio"] > ref["Diversification Ratio"],
        alt["Largest Risk Share"] < ref["Largest Risk Share"],
    ]
    return int(sum(bool(x) for x in checks)), len(checks)
