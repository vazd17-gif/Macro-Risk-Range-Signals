"""RISK RANGE — the probable daily high/low envelope (build spec section 3.1).

Two variants:

  spot_ewma    RANGE = C[t] * exp(+/- m*sigma[t])
               The build spec's locked form. Symmetric, centred on the last close.

  anchor_ewma  RANGE = A[t] * exp(+/- m_up/m_dn * sigma[t]),  A = EMA(close, anchor_span)
               Centred on a short EMA instead of spot, with independent up/down
               widths. Added because the published Similar Set ranges are visibly
               not spot-centred: on 2026-08-28 SLV printed [~58.4, 64.30] against a
               prior close of 62.77 (+2.4% / -7.0%), and GLD showed the same
               downside skew after a rally. See calib/ for the fit.

Both share one volatility estimator: close-to-close EWMA (RiskMetrics), *not*
Garman-Klass — that was tested and rejected in the reconstruction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_sigma(close: pd.Series, lam: float = 0.94, winsor_z: float | None = None,
               seed_bars: int = 20) -> pd.Series:
    """Close-to-close EWMA volatility, in log-return units, aligned to `close`.

    sigma[t] uses returns up to and including t, so RANGE for session t+1 is built
    from sigma[t] and close[t] — no lookahead.

    `winsor_z` clips each squared return at that many prior-sigmas before it enters
    the variance update. This is the mitigation for the known failure mode where a
    single gap day makes the range run hot (NVDA post +8.7% gave 2.94% vs a calmer
    ~1.6%). None disables it, reproducing plain RiskMetrics.
    """
    c = pd.Series(close).astype(float).dropna()
    r = np.log(c / c.shift(1)).values
    n = len(r)
    var = np.full(n, np.nan)

    # seed on the first `seed_bars` returns so the recursion does not start at zero
    k = min(seed_bars, max(2, n // 4))
    if n <= k + 1:
        return pd.Series(np.nan, index=c.index)
    v = float(np.nanvar(r[1:k + 1], ddof=1))
    if not np.isfinite(v) or v <= 0:
        v = float(np.nanmean(r[1:k + 1] ** 2)) or 1e-8
    var[k] = v

    for t in range(k + 1, n):
        rt = r[t]
        if not np.isfinite(rt):
            var[t] = var[t - 1]
            continue
        if winsor_z is not None:
            cap = winsor_z * np.sqrt(var[t - 1])
            rt = float(np.clip(rt, -cap, cap))
        var[t] = lam * var[t - 1] + (1.0 - lam) * rt * rt

    return pd.Series(np.sqrt(var), index=c.index)


def range_spot(close: pd.Series, lam: float = 0.94, m: float = 1.0,
               winsor_z: float | None = None) -> pd.DataFrame:
    """Symmetric envelope on spot. Row t holds the range for session t+1."""
    c = pd.Series(close).astype(float).dropna()
    sig = ewma_sigma(c, lam=lam, winsor_z=winsor_z)
    return pd.DataFrame({
        "sigma": sig,
        "center": c,
        "range_low": c * np.exp(-m * sig),
        "range_high": c * np.exp(+m * sig),
    })


def range_anchor(close: pd.Series, lam: float = 0.94, anchor_span: int = 8,
                 m_up: float = 1.0, m_dn: float = 1.0,
                 winsor_z: float | None = None,
                 floor: float | None = None) -> pd.DataFrame:
    """Asymmetric envelope on a short-EMA anchor.

    `floor` is a minimum half-width in log-return units (e.g. 0.015 ~= 1.5%). It
    matters for the Hedgeye target: on quiet instruments Hedgeye keeps a few
    percent of range where a pure EWMA band would collapse, so the half-width is
    max(m*sigma, floor) rather than m*sigma. None disables it (the Similar Set fit
    needs no floor).
    """
    c = pd.Series(close).astype(float).dropna()
    sig = ewma_sigma(c, lam=lam, winsor_z=winsor_z)
    anchor = c.ewm(span=max(1, int(anchor_span)), adjust=False).mean()
    hw_up = m_up * sig
    hw_dn = m_dn * sig
    if floor:
        hw_up = hw_up.clip(lower=floor)
        hw_dn = hw_dn.clip(lower=floor)
    return pd.DataFrame({
        "sigma": sig,
        "center": anchor,
        "range_low": anchor * np.exp(-hw_dn),
        "range_high": anchor * np.exp(+hw_up),
    })


def volume_features(volume: pd.Series, index, w1: int = 21, w3: int = 63):
    """Volume vs its 1-month and 3-month baselines (Hedgeye's TRADE/TREND windows).

    Returns two ratio series aligned to `index`:
      v1 = volume / SMA(volume, 21)     immediate-term (TRADE duration)
      v3 = volume / SMA(volume, 63)     intermediate-term (TREND duration)
    A ratio of 1.0 means volume is exactly on its baseline; >1 is above-average
    ("rate of change of volume relative to the baseline" in level form).
    """
    v = pd.Series(volume).reindex(index).astype(float)
    b1 = v.rolling(w1, min_periods=max(5, w1 // 3)).mean()
    b3 = v.rolling(w3, min_periods=max(10, w3 // 3)).mean()
    v1 = (v / b1).clip(lower=0.1, upper=5.0).fillna(1.0)
    v3 = (v / b3).clip(lower=0.1, upper=5.0).fillna(1.0)
    return v1, v3


def range_anchor_vol(close: pd.Series, volume: pd.Series, lam: float = 0.94,
                     anchor_span: int = 5,
                     a_up: float = 1.9, b1_up: float = 0.0, b3_up: float = 0.0,
                     a_dn: float = 1.9, b1_dn: float = 0.0, b3_dn: float = 0.0,
                     w1: int = 21, w3: int = 63, m_min: float = 0.4,
                     winsor_z: float | None = None) -> pd.DataFrame:
    """Volume-adjusted envelope: the sigma-multiplier is a function of volume.

    Hedgeye adjusts the Risk Range boundaries by the rate of change of volume
    against a 1-month (TRADE) and 3-month (TREND) baseline. This encodes exactly
    that: the multiplier on sigma is linear in both volume ratios.

        m_up_eff = max(a_up + b1_up*v1 + b3_up*v3, m_min)
        m_dn_eff = max(a_dn + b1_dn*v1 + b3_dn*v3, m_min)
        high/low = anchor * exp(+/- m_eff * sigma)

    Fitted on Hedgeye's own ranges the b coefficients come out negative:
    above-average volume goes with a tighter multiplier, because sigma has already
    widened the absolute band on high-volume days.
    """
    c = pd.Series(close).astype(float).dropna()
    sig = ewma_sigma(c, lam=lam, winsor_z=winsor_z)
    anchor = c.ewm(span=max(1, int(anchor_span)), adjust=False).mean()
    v1, v3 = volume_features(volume, c.index, w1=w1, w3=w3)

    m_up = (a_up + b1_up * v1 + b3_up * v3).clip(lower=m_min)
    m_dn = (a_dn + b1_dn * v1 + b3_dn * v3).clip(lower=m_min)
    return pd.DataFrame({
        "sigma": sig, "center": anchor, "v1": v1, "v3": v3,
        "range_low": anchor * np.exp(-m_dn * sig),
        "range_high": anchor * np.exp(+m_up * sig),
    })


def compute(close: pd.Series, params: dict, volume: pd.Series | None = None) -> pd.DataFrame:
    """Dispatch on params['range']['active']."""
    rp = params["range"]
    active = rp.get("active", "spot_ewma")
    cfg = rp[active]
    # dispatch on the config's shape, not its name, so alternate anchor profiles
    # (e.g. hedgeye_anchor) work without a new branch each time
    if "a_up" in cfg:                      # volume-adjusted profile
        if volume is None:
            raise ValueError("range profile %r needs volume; pass volume=" % active)
        return range_anchor_vol(close, volume, lam=cfg["lam"],
                                anchor_span=cfg["anchor_span"],
                                a_up=cfg["a_up"], b1_up=cfg.get("b1_up", 0.0),
                                b3_up=cfg.get("b3_up", 0.0),
                                a_dn=cfg["a_dn"], b1_dn=cfg.get("b1_dn", 0.0),
                                b3_dn=cfg.get("b3_dn", 0.0),
                                w1=cfg.get("w1", 21), w3=cfg.get("w3", 63),
                                winsor_z=cfg.get("winsor_z"))
    if "anchor_span" in cfg:
        return range_anchor(close, lam=cfg["lam"], anchor_span=cfg["anchor_span"],
                            m_up=cfg["m_up"], m_dn=cfg["m_dn"],
                            winsor_z=cfg.get("winsor_z"), floor=cfg.get("floor"))
    if "m" in cfg:
        return range_spot(close, lam=cfg["lam"], m=cfg["m"],
                          winsor_z=cfg.get("winsor_z"))
    raise ValueError(f"unknown range model shape: {active}")
