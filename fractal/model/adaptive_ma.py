"""TRADE / TREND / TAIL — adaptive moving-average lines (build spec section 3.2).

These are price-based only. Volume was tested three ways in the reconstruction
(VWMA, volume-gated efficiency ratio, volume-weighted efficiency) and ruled out:
none helped and most broke fits that already worked. Do not add volume here.

KAMA (Kaufman) is the working form: an EMA whose smoothing constant is driven by
the efficiency ratio, so the effective window stretches on choppy paths and
shortens on clean ones. That adaptivity is the mechanism meant to resolve the
known residual outliers (XLF TREND wants slower, SLV TRADE wants slower).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def efficiency_ratio(close: pd.Series, n: int = 20) -> pd.Series:
    """|C[t]-C[t-n]| / sum(|dC|) over the same window. 1 = straight line, 0 = pure chop."""
    c = pd.Series(close).astype(float)
    direction = (c - c.shift(n)).abs()
    volatility = c.diff().abs().rolling(n).sum()
    er = direction / volatility.replace(0.0, np.nan)
    return er.clip(0.0, 1.0)


def kama(close: pd.Series, n: int = 20, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman adaptive moving average."""
    c = pd.Series(close).astype(float).dropna()
    er = efficiency_ratio(c, n).values
    fast_sc = 2.0 / (fast + 1.0)
    slow_sc = 2.0 / (slow + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    vals = c.values
    out = np.full(len(vals), np.nan)
    if len(vals) <= n:
        return pd.Series(out, index=c.index)

    # seed with the SMA of the first window so the recursion starts on the price scale
    out[n] = float(np.mean(vals[: n + 1]))
    for t in range(n + 1, len(vals)):
        s = sc[t]
        if not np.isfinite(s):
            s = slow_sc ** 2
        out[t] = out[t - 1] + s * (vals[t] - out[t - 1])
    return pd.Series(out, index=c.index)


def ema(close: pd.Series, span: int) -> pd.Series:
    c = pd.Series(close).astype(float).dropna()
    return c.ewm(span=max(1, int(span)), adjust=False).mean()


def sma(close: pd.Series, window: int) -> pd.Series:
    c = pd.Series(close).astype(float).dropna()
    return c.rolling(max(1, int(window))).mean()


def wma(close: pd.Series, window: int) -> pd.Series:
    c = pd.Series(close).astype(float).dropna()
    w = np.arange(1, int(window) + 1, dtype=float)
    w /= w.sum()
    return c.rolling(int(window)).apply(lambda x: float(np.dot(x, w)), raw=True)


def line(close: pd.Series, cfg: dict, kind: str = "kama") -> pd.Series:
    """Build one duration line from its params block.

    `cfg["family"]` overrides the global `kind` so each duration can use the form
    that actually fits it. The calibration lands on different families for TRADE
    and TREND, which a single global setting could not express.
    """
    family = cfg.get("family", kind)
    if family == "ema":
        return ema(close, cfg["span"])
    if family == "sma":
        return sma(close, cfg.get("window", cfg.get("span")))
    if family == "wma":
        return wma(close, cfg.get("window", cfg.get("span")))
    if family == "kama":
        return kama(close, n=int(cfg["n"]), fast=int(cfg["fast"]), slow=int(cfg["slow"]))
    raise ValueError(f"unknown line family: {family}")


def compute(close: pd.Series, params: dict) -> pd.DataFrame:
    """TRADE / TREND / TAIL as a single frame aligned to `close`."""
    lp = params["lines"]
    kind = lp.get("kind", "kama")
    out = pd.DataFrame(index=pd.Series(close).dropna().index)
    for name in ("trade", "trend", "tail"):
        cfg = lp[name]
        s = line(close, cfg, kind=kind)
        lag = int(cfg.get("lag", 0) or 0)
        out[name] = s.shift(lag) if lag else s

    # TAIL needs multi-year history; flag rather than silently trust a short series.
    min_hist = int(lp["tail"].get("min_history", 750))
    out.attrs["tail_confident"] = len(out) >= min_hist
    out.attrs["tail_bars"] = len(out)
    return out
