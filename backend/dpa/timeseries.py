"""Time-series helpers shared by quality checks and forecasting.

Seasonal adjustment (documented method used for outliers and level-shift detection):
  1. trend_t      = centred rolling median of y over 13 weeks
  2. detrended_t  = y_t - trend_t
  3. seasonal_w   = median of detrended values for ISO week-of-year w (needs >= 2 years of history;
                    otherwise seasonal = 0 and this is reported)
  4. residual_t   = detrended_t - seasonal_w(t)
  5. robust z_t   = (residual_t - median(residual)) / max(MAD/0.6745, floor)
  Outlier detection runs steps 1-5 on log(1 + demand) (multiplicative model) so each series is judged
  relative to its own level; floor = 0.05 (~5%). A week is flagged when |robust z| > 3.5
  (Iglewicz & Hoaglin, 1993) AND it differs from the expected value by at least 5 units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROBUST_Z_THRESHOLD = 3.5


def regularise_weekly(s: pd.Series, fill: float | None = 0.0) -> pd.Series:
    """Reindex a weekly series (DatetimeIndex) to a continuous weekly index between its first and last week."""
    if s.empty:
        return s
    s = s.sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="7D")
    out = s.reindex(idx)
    if fill is not None:
        out = out.fillna(fill)
    return out


def seasonal_decompose_robust(y: pd.Series, window: int = 13, log: bool = False) -> dict:
    """Robust decomposition on a weekly series with DatetimeIndex.

    log=True works on log(1+y), i.e. a multiplicative model, so that strongly seasonal or promotional
    products are judged relative to their own level. Components are returned on the working scale;
    `expected` is always returned in original units.
    """
    y_orig = y.astype(float)
    y = np.log1p(y_orig.clip(lower=0)) if log else y_orig
    trend = y.rolling(window, center=True, min_periods=max(3, window // 2)).median()
    trend = trend.bfill().ffill()
    detr = y - trend
    woy = pd.Index(y.index.isocalendar().week.astype(int).values)
    years = len(y) / 52.0
    if years >= 2:
        prof = detr.groupby(woy.values).median()
        seasonal = pd.Series(prof.reindex(woy.values).values, index=y.index).fillna(0.0)
        seasonal_used = True
    else:
        seasonal = pd.Series(0.0, index=y.index)
        seasonal_used = False
    resid = detr - seasonal
    med = resid.median()
    mad = (resid - med).abs().median()
    # Scale floor: for very smooth series the MAD can be ~0, which would turn ordinary promo weeks into
    # z-scores in the hundreds. The floor makes a flag require a deviation of at least ~17% of the
    # series level (3.5 x 5%), or 3.5 units for tiny series.
    if log:
        scale = max((mad / 0.6745) if mad == mad else 0.0, 0.05)
    else:
        level = float(np.nanmedian(np.abs(trend))) if len(trend) else 0.0
        scale = max((mad / 0.6745) if mad == mad else 0.0, 0.05 * level, 1.0)
    z = (resid - med) / scale
    fitted = trend + seasonal + med
    expected = np.expm1(fitted).clip(lower=0) if log else fitted
    return {"trend": trend, "seasonal": seasonal, "resid": resid, "robust_z": z, "scale": scale,
            "expected": expected, "seasonal_used": seasonal_used, "log": log}


def detect_level_shifts(y: pd.Series, window: int = 26, min_rel_change: float = 0.15, max_breaks: int = 3) -> list[dict]:
    """Detect sustained level shifts on the seasonally adjusted series.

    For every week t compares the mean of the `window` weeks before and after t (seasonally adjusted).
    A shift is reported when the relative change exceeds `min_rel_change` and Welch's t-statistic > 5.
    Non-maximum suppression keeps the strongest shift within each window.
    """
    if len(y) < 2 * window + 4:
        return []
    dec = seasonal_decompose_robust(y)
    sa = (y - dec["seasonal"]).astype(float).values
    n = len(sa)
    stats = []
    for t in range(window, n - window):
        a, b = sa[t - window:t], sa[t:t + window]
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(va / window + vb / window)
        tstat = (mb - ma) / se if se > 0 else 0.0
        rel = (mb - ma) / abs(ma) if ma else np.nan
        stats.append((t, tstat, rel, ma, mb))
    cands = [s for s in stats if abs(s[1]) > 5 and not np.isnan(s[2]) and abs(s[2]) >= min_rel_change]
    cands.sort(key=lambda s: -abs(s[1]))
    chosen: list[tuple] = []
    for c in cands:
        if all(abs(c[0] - k[0]) >= window for k in chosen):
            chosen.append(c)
        if len(chosen) >= max_breaks:
            break
    out = []
    for t, tstat, rel, ma, mb in sorted(chosen, key=lambda s: s[0]):
        out.append({"week": y.index[t].strftime("%Y-%m-%d"), "mean_before": float(ma), "mean_after": float(mb),
                    "change_pct": float(rel * 100), "t_stat": float(tstat)})
    return out
