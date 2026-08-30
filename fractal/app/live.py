"""Re-price the report against live quotes without recomputing the levels.

The RANGE, TRADE and TREND levels for a session are set by the previous close and
do not move while the session trades -- that is the whole point of them. What moves
is spot, and therefore everything spot-relative: where price sits in the range,
which side of each line it is on, and so which signal is live.

So intraday work is not a recomputation, it is a re-pricing. The levels are carried
from the daily run; only the spot-dependent fields are recalculated. That is both
correct and cheap: one batch quote request instead of 191 model fits.

It also makes a distinction the daily report cannot: a line crossed *during today's
session*. The daily run only ever sees closes, so a break that happened this morning
is invisible to it until tonight. Those crossings are marked `intraday` and are the
alerts worth surfacing at the top of the screen.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import signals as S
from ..data.etf_universe import yf_symbol

# How far past a line price must travel before the move counts as a crossing,
# as a fraction of the RANGE width. At 2% of a typical 3%-wide range this is
# roughly 0.06% of spot: far enough to ignore quote jitter, close enough that a
# genuine break is flagged on the same bar it happens.
CROSS_BUFFER = 0.02


def quotes(tickers, chunk=60):
    """{display ticker: last price}. Missing names are simply absent, never guessed.

    Quotes are requested under the feed symbol, not the display ticker. VIX and
    MOVE publish as ^VIX and ^MOVE; requesting bare "MOVE" silently returns an
    unrelated listed company, which is worse than returning nothing.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import yfinance as yf

    out = {}
    tickers = list(tickers)
    feed = {t: yf_symbol(t) for t in tickers}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        syms = sorted({feed[t] for t in part})
        try:
            data = yf.download(syms, period="1d", interval="1m", progress=False,
                               group_by="ticker", threads=True)
        except Exception:
            continue
        for t in part:
            try:
                sym = feed[t]
                sub = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
                c = sub["Close"].dropna()
                if len(c):
                    out[t] = float(c.iloc[-1])
            except Exception:
                pass
    return out


def reprice(df, px=None, edge=S.EDGE):
    """Return a copy of the signal frame re-priced to live quotes.

    Levels are untouched. Spot, range position, the bull/bear reads, the state and
    the signal are recomputed, and anything that crossed a line since the close is
    flagged.
    """
    if df.empty:
        return df
    px = px if px is not None else quotes(df["ticker"].tolist())
    out = df.copy()
    out["close_spot"] = out["spot"]
    out["intraday"] = ""
    out["live"] = False

    for i, r in out.iterrows():
        spot = px.get(r["ticker"])
        if spot is None or not np.isfinite(spot) or spot <= 0:
            continue
        lo, hi = r["range_low"], r["range_high"]
        trade, trend = r["trade"], r["trend"]
        was_trade, was_trend = r["trade_bull"], r["trend_bull"]

        now_trade = bool(spot > trade) if np.isfinite(trade) else None
        now_trend = bool(spot > trend) if np.isfinite(trend) else None
        pos = (spot - lo) / (hi - lo) if (np.isfinite(lo) and np.isfinite(hi) and hi > lo) else np.nan

        # A name resting on its line flips sides on a fraction of a cent, and the
        # feed returns the last print with enough jitter to do it unprompted. Left
        # alone that churns the alert strip every ten minutes and pushes a phone
        # alert for a 0.003% move. So a crossing only counts once price has cleared
        # the line by a real distance -- scaled to the range width, because a tenth
        # of a percent is a long way in LQD and nothing at all in ARKQ.
        buf = (CROSS_BUFFER * (hi - lo)
               if (np.isfinite(lo) and np.isfinite(hi) and hi > lo) else 0.0)
        crossed = []
        for name, before, after, level in (("TREND", was_trend, now_trend, trend),
                                           ("TRADE", was_trade, now_trade, trade)):
            if before is None or after is None or bool(before) == bool(after):
                continue
            if np.isfinite(level) and abs(spot - level) <= buf:
                continue                      # still on the line, not through it
            crossed.append(("lost " if before else "reclaimed ") + name)

        at_low = np.isfinite(pos) and pos <= edge
        at_high = np.isfinite(pos) and pos >= 1 - edge

        sig, why = None, ""
        if r.get("is_index"):
            why = "volatility index - context only, not a position"
        elif r.get("cash_like"):
            why = r.get("why") or ""
        elif any(c.startswith("lost") for c in crossed):
            sig, why = S.REMOVE_LONG, " and ".join(crossed)
        elif any(c.startswith("reclaimed") for c in crossed):
            sig, why = S.COVER_SHORT, " and ".join(crossed)
        elif at_low and now_trade is False and now_trend is False:
            sig, why = S.WATCHLIST, "at the low end but bearish TRADE and TREND - watch, no action yet"
        elif at_low and now_trade is False and now_trend:
            sig, why = S.WATCHLIST, "at the low end but TRADE has broken - watch for TREND to hold"
        elif at_low and (now_trade or now_trend):
            both = ("TRADE and TREND" if (now_trade and now_trend)
                    else ("TRADE" if now_trade else "TREND"))
            sig, why = S.ADD_LONG, "low end of RANGE, bullish %s" % both
        elif at_high and (now_trade is False or now_trend is False):
            both = ("TRADE and TREND" if (now_trade is False and now_trend is False)
                    else ("TRADE" if now_trade is False else "TREND"))
            sig, why = S.ADD_SHORT, "high end of RANGE, bearish %s" % both
        else:
            sig, why = r["signal"] if not crossed else None, r.get("why") or ""

        out.at[i, "spot"] = spot
        out.at[i, "live"] = True
        out.at[i, "pos_in_range"] = float(np.clip(pos, -0.5, 1.5)) if np.isfinite(pos) else np.nan
        out.at[i, "trade_bull"] = now_trade
        out.at[i, "trend_bull"] = now_trend
        out.at[i, "at_low"] = bool(at_low)
        out.at[i, "at_high"] = bool(at_high)
        out.at[i, "pct_to_low"] = 100 * (lo / spot - 1) if np.isfinite(lo) else np.nan
        out.at[i, "pct_to_high"] = 100 * (hi / spot - 1) if np.isfinite(hi) else np.nan
        out.at[i, "pct_to_trade"] = 100 * (trade / spot - 1) if np.isfinite(trade) else np.nan
        out.at[i, "pct_to_trend"] = 100 * (trend / spot - 1) if np.isfinite(trend) else np.nan
        out.at[i, "chg_pct"] = 100 * (spot / r["close_spot"] - 1) if r["close_spot"] else np.nan
        out.at[i, "state"] = (None if (now_trade is None or now_trend is None)
                              else S.classify_state(now_trade, now_trend))
        out.at[i, "signal"] = sig
        out.at[i, "why"] = why
        out.at[i, "intraday"] = ", ".join(crossed)

    order = {S.REMOVE_LONG: 0, S.ADD_LONG: 1, S.ADD_SHORT: 2,
             S.WATCHLIST: 3, S.COVER_SHORT: 4}
    out["_r"] = out["signal"].map(order).fillna(9)
    out["_x"] = (~out["intraday"].astype(bool)).astype(int)   # today's crossings first
    out = (out.sort_values(["_x", "_r", "pos_in_range", "ticker"])
              .drop(columns=["_x", "_r"]).reset_index(drop=True))
    out.attrs["live_at"] = dt.datetime.now()
    out.attrs["n_live"] = int(out["live"].sum())
    return out
