"""
Shared time-series transforms used by the recession engine and the GLR
regime engine. Extracted so both call-sites use one implementation and
cannot drift.
"""

import numpy as np
import pandas as pd


def expanding_zscore(series: pd.Series, min_periods: int = 12) -> pd.Series:
    """Expanding-window z-score with division-by-zero protection.

    (x - expanding_mean) / expanding_std, ddof=1. Inf/-Inf are mapped to NaN
    by zeroing near-zero stds. Uses `min_periods` observations before
    producing any value.
    """
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    std = std.where(std > 1e-8, np.nan)
    z = (series - mean) / std
    return z.replace([np.inf, -np.inf], np.nan)


def rolling_tertile_state(
    series: pd.Series,
    window: int = 60,
    min_periods: int = 12,
    low_q: float = 0.33,
    high_q: float = 0.67,
) -> pd.Series:
    """Label each observation as 'weak' | 'neutral' | 'strong' using
    trailing rolling percentile bands.

    At date t, compares series[t] against the rolling [low_q, high_q]
    quantile bands of the last `window` observations (requires at least
    `min_periods`). Returns a pd.Series of str labels aligned to
    series.index; positions with insufficient history are NaN.
    """
    low = series.rolling(window=window, min_periods=min_periods).quantile(low_q)
    high = series.rolling(window=window, min_periods=min_periods).quantile(high_q)

    labels = pd.Series(np.nan, index=series.index, dtype=object)
    valid = series.notna() & low.notna() & high.notna()

    weak = valid & (series <= low)
    strong = valid & (series >= high)
    neutral = valid & ~weak & ~strong

    labels.loc[weak] = 'weak'
    labels.loc[neutral] = 'neutral'
    labels.loc[strong] = 'strong'
    return labels
