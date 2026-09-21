from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .metrics import periods_per_year


def sample_covariance(
    returns: pd.DataFrame, frequency: str = "monthly"
) -> pd.DataFrame:
    ppy = periods_per_year(frequency)
    return returns.cov() * ppy


def ledoit_wolf_covariance(
    returns: pd.DataFrame, frequency: str = "monthly"
) -> pd.DataFrame:
    ppy = periods_per_year(frequency)
    model = LedoitWolf().fit(returns.values)
    cov = model.covariance_ * ppy
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def portfolio_volatility(weights: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(weights @ cov @ weights))


def risk_contributions(weights: pd.Series, cov: pd.DataFrame) -> pd.Series:
    w = weights.reindex(cov.index).fillna(0.0).values
    sigma = portfolio_volatility(w, cov.values)
    if sigma == 0:
        return pd.Series(0.0, index=cov.index)
    marginal = cov.values @ w / sigma
    contribution = w * marginal
    return pd.Series(contribution, index=cov.index)
