"""Roughness / persistence diagnostics (build spec section 5, backtests).

These characterise a name's *texture* — whether moves tend to continue or reverse.
In this model persistence is not a separate indicator and not a band skew; it
lives inside the adaptivity of the moving averages. These functions exist to
describe it and to sanity-check fits, not to feed the levels directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def variance_ratio(close: pd.Series, q: int = 5, window: int | None = None) -> float:
    """Lo-MacKinlay variance ratio: Var(q-period return)/(q*Var(1-period return)).

    >1 trending/persistent, ~1 random walk, <1 mean-reverting.
    """
    c = pd.Series(close).astype(float).dropna()
    if window:
        c = c.iloc[-window:]
    r = np.log(c / c.shift(1)).dropna()
    if len(r) < q * 4:
        return float("nan")
    v1 = r.var(ddof=1)
    rq = np.log(c / c.shift(q)).dropna()
    vq = rq.var(ddof=1)
    if v1 <= 0:
        return float("nan")
    return float(vq / (q * v1))


def hurst_rs(close: pd.Series, min_chunk: int = 8, max_chunks: int = 12) -> float:
    """Rescaled-range Hurst exponent. 0.5 random walk, >0.5 persistent, <0.5 choppy.

    A rough proxy only — R/S is biased on short daily samples. Reported for
    texture, never used as a level.
    """
    c = pd.Series(close).astype(float).dropna()
    r = np.log(c / c.shift(1)).dropna().values
    n = len(r)
    if n < min_chunk * 4:
        return float("nan")

    sizes, s = [], min_chunk
    while s <= n // 2 and len(sizes) < max_chunks:
        sizes.append(s)
        s = int(s * 1.7)
    if len(sizes) < 3:
        return float("nan")

    xs, ys = [], []
    for size in sizes:
        rs = []
        for start in range(0, n - size + 1, size):
            chunk = r[start:start + size]
            sd = chunk.std(ddof=1)
            if sd <= 0:
                continue
            dev = np.cumsum(chunk - chunk.mean())
            rs.append((dev.max() - dev.min()) / sd)
        if rs:
            xs.append(np.log(size))
            ys.append(np.log(np.mean(rs)))
    if len(xs) < 3:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def texture(close: pd.Series) -> dict:
    """Compact persistence profile used by the scan and the calibration report."""
    return {
        "vr_5": variance_ratio(close, 5, window=252),
        "vr_15": variance_ratio(close, 15, window=252),
        "vr_63": variance_ratio(close, 63, window=504),
        "hurst": hurst_rs(close.iloc[-504:]),
    }
