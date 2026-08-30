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


def quotes(tickers, chunk=60, with_times=False):
    """{display ticker: last price}. Missing names are simply absent, never guessed.

    With `with_times`, returns ({ticker: price}, {ticker: timestamp}) instead. The
    timestamp is what tells a genuinely live quote from the last print of a session
    that has already closed -- see `reprice`.

    Quotes are requested under the feed symbol, not the display ticker. VIX and
    MOVE publish as ^VIX and ^MOVE; requesting bare "MOVE" silently returns an
    unrelated listed company, which is worse than returning nothing.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import yfinance as yf

    out, when = {}, {}
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
                    when[t] = c.index[-1]
            except Exception:
                pass
    return (out, when) if with_times else out


def reprice(df, px=None, edge=S.EDGE, times=None):
    """Return a copy of the signal frame re-priced to live quotes.

    Levels are untouched. Spot, range position, the bull/bear reads, the state and
    the signal are recomputed, and anything that crossed a line since the close is
    flagged.

    A crossing is only claimed when the quote comes from a session *after* the one
    the levels were built on. Outside market hours the feed keeps returning the last
    continuous print of the previous session, and that print is not the official
    close -- the closing auction sets the close, and the two differ. Comparing a
    15:59 print against flags derived from the 16:00 auction invents crossings that
    never happened: IWC printed 194.75 at 15:59 on 2026-08-28 and closed at 195.03,
    on opposite sides of a TRADE line at 194.95. Spot still updates; only the claim
    that something crossed is withheld.
    """
    if df.empty:
        return df
    if px is None:
        px, times = quotes(df["ticker"].tolist(), with_times=True)
    times = times or {}
    asof = str(df["asof"].max()) if "asof" in df else ""
    out = df.copy()
    out["close_spot"] = out["spot"]
    out["intraday"] = ""
    out["live"] = False

    for i, r in out.iterrows():
        spot = px.get(r["ticker"])
        if spot is None or not np.isfinite(spot) or spot <= 0:
            continue
        ts = times.get(r["ticker"])
        if ts is not None and asof:
            try:
                if not str(pd.Timestamp(ts).date()) > asof:
                    continue          # quote predates the close the levels came from
            except Exception:
                pass
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
        buf = S.line_band(lo, hi)
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

        # Intraday, a "break" is a line crossed since the close rather than one
        # crossed within the last few sessions, so the crossings feed the same
        # ladder the daily run uses. Everything downstream of that is identical,
        # which is the point of it living in one function.
        lost = [c for c in crossed if c.startswith("lost")]
        recl = [c for c in crossed if c.startswith("reclaimed")]

        # A line broken on an earlier close is still broken now, so the daily state
        # is carried and only overturned by an intraday move the other way.
        def _state(name):
            broke = bool(r.get("broke_" + name.lower())) or any(name in c for c in lost)
            back = bool(r.get("recl_" + name.lower())) or any(name in c for c in recl)
            if any(name in c for c in lost):
                back = False
            elif any(name in c for c in recl):
                broke = False
            return broke, back

        bt, rt = _state("TREND")
        bd, rd = _state("TRADE")
        sig, why = S.decide(
            bool(r.get("is_index")), bool(r.get("cash_like")),
            100 * (hi / lo - 1) if (np.isfinite(lo) and np.isfinite(hi) and lo > 0) else 0.0,
            bt, bd, rt, rd,
            " and ".join(crossed) or (r.get("why") or ""),
            at_low, at_high, now_trade, now_trend,
            outside_high=bool(np.isfinite(hi) and spot > hi),
            outside_low=bool(np.isfinite(lo) and spot < lo),
            # Volume is only known to the close, so intraday a break is confirmed
            # by the last completed session's volume. A name breaking out on heavy
            # volume today reads as unconfirmed until tonight, which is the
            # conservative direction to be wrong in.
            vol_surge=bool(r.get("vol_surge")),
            # The hold test asks whether an earlier break is still holding, and
            # intraday that question has not changed: the levels are still the ones
            # the daily run computed, so its own was_above/was_below carry over and
            # only `at_high` is re-read against live spot. Deriving the flag from
            # the daily signal instead lost the failed-break state a day early.
            was_above=bool(r.get("was_above")), was_below=bool(r.get("was_below")))

        # With nothing crossed and no edge read, the name keeps whatever the daily
        # run concluded -- a signal earned on closes does not evaporate because
        # spot drifted into the middle of the range at 11am.
        if sig is None and not crossed and not r.get("is_index") and not r.get("cash_like"):
            sig, why = r["signal"], r.get("why") or ""

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
        # Intraday the session's move is spot against the close the levels came
        # from, which supersedes the close-to-close figure from the daily run.
        out.at[i, "day_pct"] = out.at[i, "chg_pct"]
        out.at[i, "state"] = (None if (now_trade is None or now_trend is None)
                              else S.classify_state(now_trade, now_trend))
        out.at[i, "signal"] = sig
        out.at[i, "why"] = why
        out.at[i, "intraday"] = ", ".join(crossed)

    order = {S.REMOVE_LONG: 0, S.TRIM_LONG: 1, S.BREAKOUT: 2, S.ADD_LONG: 3,
             S.ADD_SHORT: 4, S.TRIM_SHORT: 5, S.COVER_SHORT: 6, S.WATCHLIST: 7}
    out["_r"] = out["signal"].map(order).fillna(9)
    out["_x"] = (~out["intraday"].astype(bool)).astype(int)   # today's crossings first
    out = (out.sort_values(["_x", "_r", "pos_in_range", "ticker"])
              .drop(columns=["_x", "_r"]).reset_index(drop=True))
    # With every quote stale the frame is just the daily build, so it is not
    # labelled live -- a "live" stamp over yesterday's closes is a lie.
    n_live = int(out["live"].sum())
    if n_live:
        out.attrs["live_at"] = dt.datetime.now()
    out.attrs["n_live"] = n_live
    return out
