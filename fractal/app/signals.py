"""ETF Pro-style signals: where price sits in the RANGE, and what TRADE/TREND say.

The four rules mirror Hedgeye's own ETF Pro Plus update language:

  ADD LONG      price at/near the LOW end of the Risk Range and the signal is
                BULLISH TRADE and/or TREND                      -> buy
  REMOVE LONG   the ETF has BROKEN TRADE and/or TREND           -> sell
  ADD SHORT     price at/near the HIGH end of the Risk Range and the signal is
                BEARISH TRADE and/or TREND                      -> short (or avoid)
  COVER SHORT   a broken short reclaims TRADE and/or TREND      -> cover

Plus one rule taken from the Similar Set handbook rather than Hedgeye, because it
is the single most common way the "buy the low end" rule loses money:

  WATCHLIST     price is at the low end but the signal has broken, so there is
                nothing to do yet. The handbook is explicit -- do not buy the low
                end of the RANGE during a break of TRADE; wait for the TREND
                level to hold. Worth watching, not worth acting on.

"Broken" is an event, not a state: it means the line flipped within the last
`fresh_days` sessions. A name that has been bearish for a month is not a fresh
sell signal, it is simply short.

The RANGE profile defaults to `hedgeye_anchor` because these rules are Hedgeye's
and that profile is the one fitted to Hedgeye's published ranges; the Similar Set
profile draws a narrower band and would trip the edge rules more often.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices
from ..data.etf_universe import all_etfs, group_of
from ..model import adaptive_ma, range_ewma, state as state_mod
from ..model.range_ewma import volume_features

EDGE = 0.15          # "near or at" the end of the range: outer 15%
FRESH_DAYS = 3       # a break/reclaim counts as an event for this many sessions
MIN_RANGE_PCT = 2.0  # below this range width an ETF is cash-like: no signals
SETTLE_MULT = 3      # an EMA needs roughly 3x its span of history to shed its seed
VOL_W1, VOL_W3 = 21, 63     # 1-month and 3-month volume baselines (TRADE / TREND)
VOL_Z = 2.0          # |z| on log volume beyond which a session counts as an outlier

# Instruments whose whole Risk Range is narrower than MIN_RANGE_PCT do not move
# enough for these rules to mean anything - a T-bill fund "breaking TRADE" by five
# basis points is not a sell. The Hedgeye validation made the same point from the
# other side: every TREND-direction mismatch was a near-cash bond ETF where price
# sits on the line and bull/bear is noise. Such names are still reported, with
# signal None and a `cash_like` flag.

ADD_LONG = "ADD LONG"
REMOVE_LONG = "REMOVE LONG"
ADD_SHORT = "ADD SHORT"
COVER_SHORT = "COVER SHORT"
WATCHLIST = "WATCHLIST"


def classify_state(trade_bull, trend_bull):
    """Combined state from the two duration reads."""
    from ..model.state import classify
    return classify(bool(trade_bull), bool(trend_bull))


def _days_since_flip(flags: pd.Series):
    """Sessions since the boolean series last changed value, and its prior value."""
    f = flags.dropna()
    if len(f) < 2:
        return None, None
    vals = f.astype(bool).values
    last = vals[-1]
    for k in range(len(vals) - 2, -1, -1):
        if vals[k] != last:
            return len(vals) - 1 - k, bool(vals[k])
    return None, None          # never flipped in the available history


def evaluate(ticker, ohlc, params, edge=EDGE, fresh_days=FRESH_DAYS,
             min_range_pct=MIN_RANGE_PCT):
    """One ETF -> levels, range position, and any triggered signals."""
    close = ohlc["Close"].dropna()
    if len(close) < 80:
        return None

    lines = adaptive_ma.compute(close, params)
    rng = range_ewma.compute(close, params, volume=ohlc.get("Volume"))
    states = state_mod.state_series(close, lines)

    spot = float(close.iloc[-1])
    lo = float(rng["range_low"].iloc[-1])
    hi = float(rng["range_high"].iloc[-1])
    trade = lines["trade"].iloc[-1]
    trend = lines["trend"].iloc[-1]
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        return None

    pos = (spot - lo) / (hi - lo)
    trade_bull = bool(spot > trade) if np.isfinite(trade) else None
    trend_bull = bool(spot > trend) if np.isfinite(trend) else None

    d_trade, prev_trade = _days_since_flip(states.get("trade_bull", pd.Series(dtype=object)))
    d_trend, prev_trend = _days_since_flip(states.get("trend_bull", pd.Series(dtype=object)))

    broke_trade = bool(d_trade is not None and d_trade <= fresh_days and trade_bull is False)
    broke_trend = bool(d_trend is not None and d_trend <= fresh_days and trend_bull is False)
    recl_trade = bool(d_trade is not None and d_trade <= fresh_days and trade_bull is True)
    recl_trend = bool(d_trend is not None and d_trend <= fresh_days and trend_bull is True)

    at_low = pos <= edge
    at_high = pos >= 1 - edge

    # Volume against its 1-month and 3-month baselines. Volume is lognormal and its
    # scale differs by orders of magnitude across the list, so "unusual" is measured
    # as a z-score of log volume against the same 3-month window rather than as a
    # fixed percentage -- a +60% day is routine for a thin fund and remarkable for
    # a mega-ETF. The percentages are still reported, because that is how the
    # deviation is normally quoted.
    vol = vol_1m = vol_3m = v1r = v3r = vol_z = vol_z1 = np.nan
    vol_flag = ""
    volume = ohlc.get("Volume")
    if volume is not None:
        v = pd.Series(volume).reindex(close.index).astype(float)
        if len(v.dropna()) > VOL_W3:
            vol = float(v.iloc[-1]) if np.isfinite(v.iloc[-1]) else np.nan
            vol_1m = float(v.iloc[-VOL_W1:].mean())
            vol_3m = float(v.iloc[-VOL_W3:].mean())
            f1, f3 = volume_features(v, close.index, w1=VOL_W1, w3=VOL_W3)
            v1r, v3r = float(f1.iloc[-1]), float(f3.iloc[-1])
            # z of log volume against each window separately. The 3-month z is the
            # flag, because 21 observations give a noisy standard deviation; the
            # 1-month z is reported alongside because a fund whose volume has been
            # elevated for weeks reads very differently on the two windows.
            lv = np.log(v.replace(0, np.nan)).dropna()
            if np.isfinite(vol) and vol > 0:
                for win, name in ((VOL_W1, "z1"), (VOL_W3, "z3")):
                    if len(lv) <= win:
                        continue
                    w = lv.iloc[-win:]
                    sd = float(w.std(ddof=1))
                    if sd > 0:
                        z = float((np.log(vol) - w.mean()) / sd)
                        if name == "z1":
                            vol_z1 = z
                        else:
                            vol_z = z
            if np.isfinite(vol_z):
                if vol_z >= VOL_Z:
                    vol_flag = "surge"
                elif vol_z <= -VOL_Z:
                    vol_flag = "dry"
    width_pct = 100 * (hi / lo - 1)
    cash_like = width_pct < min_range_pct

    # A recently listed ETF can still be scored, but its TREND line has not had
    # enough history to shed the seed it was started from, so the level is softer
    # than it looks. Flagged rather than suppressed.
    trend_span = int(params["lines"]["trend"].get("span", 64) or 64)
    bars = len(close)
    young = bars < SETTLE_MULT * trend_span

    # An "event" is a fresh line flip; a "signal" is the action to take. They are
    # tracked separately so a reclaim somewhere mid-range cannot crowd out the
    # range-edge signals, which are the point of the report.
    event = ""
    if broke_trend or broke_trade:
        event = "broke %s" % " and ".join(
            [n for n, b in (("TREND", broke_trend), ("TRADE", broke_trade)) if b])
    elif recl_trend or recl_trade:
        event = "reclaimed %s" % " and ".join(
            [n for n, b in (("TREND", recl_trend), ("TRADE", recl_trade)) if b])

    sig, why = None, ""
    if cash_like:
        why = "range only %.2f%% wide - too tight for a signal" % width_pct
    elif broke_trend or broke_trade:
        sig, why = REMOVE_LONG, event                      # a fresh break is a sell
    elif at_low and trade_bull is False and trend_bull is False:
        sig, why = WATCHLIST, "at the low end but bearish TRADE and TREND - watch, no action yet"
    elif at_low and trade_bull is False and trend_bull:
        sig, why = WATCHLIST, "at the low end but TRADE has broken - watch for TREND to hold"
    elif at_low and (trade_bull or trend_bull):
        both = ("TRADE and TREND" if (trade_bull and trend_bull)
                else ("TRADE" if trade_bull else "TREND"))
        sig, why = ADD_LONG, "low end of RANGE, bullish %s" % both
    elif at_high and (trade_bull is False or trend_bull is False):
        both = ("TRADE and TREND" if (trade_bull is False and trend_bull is False)
                else ("TRADE" if trade_bull is False else "TREND"))
        sig, why = ADD_SHORT, "high end of RANGE, bearish %s" % both
    elif recl_trend or recl_trade:
        sig, why = COVER_SHORT, event                      # cover / watch for re-entry

    if young and sig:
        why = (why + " " if why else "") + "(short history: %d bars)" % bars

    return {
        "ticker": ticker,
        "group": group_of(ticker),
        "asof": close.index[-1].date().isoformat(),
        "spot": spot,
        "range_low": lo,
        "range_high": hi,
        "pos_in_range": float(np.clip(pos, -0.5, 1.5)),
        "range_width_pct": 100 * (hi / lo - 1),
        "pct_to_low": 100 * (lo / spot - 1),
        "pct_to_high": 100 * (hi / spot - 1),
        "trade": float(trade) if np.isfinite(trade) else np.nan,
        "trend": float(trend) if np.isfinite(trend) else np.nan,
        "trade_bull": trade_bull,
        "trend_bull": trend_bull,
        "pct_to_trade": 100 * (trade / spot - 1) if np.isfinite(trade) else np.nan,
        "pct_to_trend": 100 * (trend / spot - 1) if np.isfinite(trend) else np.nan,
        "days_since_trade_flip": d_trade,
        "days_since_trend_flip": d_trend,
        "state": states["state"].iloc[-1] if "state" in states else None,
        "signal": sig,
        "why": why,
        "event": event,
        "volume": vol,
        "vol_1m_avg": vol_1m,
        "vol_3m_avg": vol_3m,
        "vol_vs_1m_pct": (v1r - 1) * 100 if np.isfinite(v1r) else np.nan,
        "vol_vs_3m_pct": (v3r - 1) * 100 if np.isfinite(v3r) else np.nan,
        "vol_z_1m": vol_z1,
        "vol_z_3m": vol_z,
        "vol_z": vol_z,
        "vol_flag": vol_flag,
        "cash_like": bool(cash_like),
        "history_bars": int(bars),
        "young": bool(young),
        "at_low": bool(at_low),
        "at_high": bool(at_high),
        "outside_low": bool(spot < lo),
        "outside_high": bool(spot > hi),
    }


def run(tickers=None, params=None, profile="hedgeye_anchor", edge=EDGE,
        fresh_days=FRESH_DAYS, min_range_pct=MIN_RANGE_PCT, verbose=True):
    """Evaluate the ETF watchlist. Returns a DataFrame, most actionable first."""
    params = dict(params or load_params())
    if profile:
        params["range"] = dict(params["range"])
        params["range"]["active"] = profile
    tickers = tickers or all_etfs()
    prices = load_prices(tickers, params=params, verbose=verbose)

    rows, missing = [], []
    for t in tickers:
        df = prices.get(t)
        if df is None or len(df) < 80:
            missing.append(t)
            continue
        r = evaluate(t, df, params, edge=edge, fresh_days=fresh_days,
                     min_range_pct=min_range_pct)
        if r is None:
            missing.append(t)
        else:
            rows.append(r)
    if verbose and missing:
        print("[signals] no usable history: " + ", ".join(missing))

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # One report, one date. A ticker whose last bar is older than the rest is
    # stale (delisted or renamed); it is dropped from signals rather than compared
    # against everyone else's newer prices.
    asof = out["asof"].max()
    stale = out[out["asof"] < asof]
    if len(stale):
        if verbose:
            print("[signals] stale, excluded from signals: "
                  + ", ".join("%s(%s)" % (r.ticker, r.asof) for r in stale.itertuples()))
        out.loc[out["asof"] < asof, "signal"] = None
        out.loc[out["asof"] < asof, "why"] = "stale data - last bar " + stale["asof"]
    out.attrs["asof"] = asof
    order = {REMOVE_LONG: 0, ADD_LONG: 1, ADD_SHORT: 2, WATCHLIST: 3, COVER_SHORT: 4}
    out["_rank"] = out["signal"].map(order).fillna(9)
    # inside each bucket, the most extreme range position first
    out["_edge"] = np.where(out["signal"].isin([ADD_SHORT]),
                            -out["pos_in_range"], out["pos_in_range"])
    return out.sort_values(["_rank", "_edge", "ticker"]).drop(columns=["_rank", "_edge"]).reset_index(drop=True)


def buckets(df):
    """Signal name -> rows, in the order the newsletter presents them."""
    return {name: df[df["signal"] == name]
            for name in (ADD_LONG, REMOVE_LONG, ADD_SHORT, WATCHLIST, COVER_SHORT)}
