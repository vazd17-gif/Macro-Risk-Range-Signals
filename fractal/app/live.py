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
import json
import os

import numpy as np
import pandas as pd

from . import signals as S
from ..data.loader import next_session
from ..data.etf_universe import yf_symbol


LOCKS = ("out", ".locked")


def _lock_path():
    from ..data.loader import repo_path
    return repo_path(*LOCKS)


def load_locks(session):
    """{ticker: "lost"|"reclaimed"} for TRADE, for this session only.

    A level touched during the session is a decision taken, and it stands until the
    next one. Without this the ten-minute job would unwind its own trade the moment
    price crossed back: sell at 10am on a broken TRADE, buy it back at 2pm on the
    reclaim, and do it again tomorrow. The lock is what makes the touch a trade
    rather than a running commentary on where price is.
    """
    try:
        with open(_lock_path()) as fh:
            st = json.load(fh)
        return dict(st.get("locked", {})) if st.get("session") == session else {}
    except Exception:
        return {}


def save_locks(session, locked):
    p = _lock_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump({"session": session, "locked": locked}, fh, indent=1)


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


def reprice(df, px=None, edge=S.EDGE, times=None, locks=None, persist=True):
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
    # The bands the daily run chose from the VIX. Carried, not re-derived: the
    # volatility regime is a close-to-close judgement like TREND.
    edge_buy = df.attrs.get("edge_buy") or S.EDGE_BUY
    edge_sell = df.attrs.get("edge_sell") or S.EDGE_SELL
    edge_break = df.attrs.get("edge_break") or S.EDGE_BREAK

    # A TRADE level touched earlier in this session is a trade already taken, and it
    # is not unwound if price crosses back before the close.
    # Keyed by the calendar date the job RUNS, not by next_session(asof).
    #
    # next_session rolls forward the moment the day's bar lands in the feed, so the
    # last hour of every session was writing its locks under TOMORROW's key -- and
    # the next morning inherited them. Seven names carried Monday evening's
    # crossings into Tuesday that way (BAC, COST, EWI, HYG, JNK, META, PSCC), two of
    # which opened positions on a line that had been crossed the previous day.
    #
    # The lock exists to make a touch stick for the session it happened in. That
    # session is a wall-clock day, so that is what it is keyed to.
    session = dt.date.today().isoformat()
    locked = dict(load_locks(session)) if locks is None else dict(locks)
    fresh_lock = False
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
        was_trade = r["trade_bull"]

        # TRADE is the only line read against live price. Hedgeye's own ETF Pro
        # changes bear this out: nine of ten removals had the session low dip below
        # TRADE on the day the note went out, while only one had broken on the prior
        # close -- they act on TRADE intraday. TREND and the RANGE reads stay on the
        # close, because a regime call and a range edge are close-to-close
        # judgements and re-deriving them from an 11am print is noise, not news.
        # A touch of the line counts as being on the other side of it, so the
        # crossing above and the state the ladder reads cannot disagree.
        if not np.isfinite(trade):
            now_trade = None
        elif was_trade:
            now_trade = bool(spot > trade)
        else:
            now_trade = bool(spot >= trade)
        now_trend = r["trend_bull"]
        pos = (spot - lo) / (hi - lo) if (np.isfinite(lo) and np.isfinite(hi) and hi > lo) else np.nan

        # A name resting on its line flips sides on a fraction of a cent, and the
        # feed returns the last print with enough jitter to do it unprompted. Left
        # alone that churns the alert strip every ten minutes and pushes a phone
        # alert for a 0.003% move. So a crossing only counts once price has cleared
        # the line by a real distance -- scaled to the range width, because a tenth
        # of a percent is a long way in LQD and nothing at all in ARKQ.
        # Reaching TRADE counts, not just clearing it. Hedgeye covered its UUP short
        # on a session high of 28.20 against a TRADE line of 28.20 -- price touched
        # the line and they acted; a buffer that demanded daylight past it, and a
        # strict inequality that demanded price go through it, both said nothing
        # happened. The line is the decision, so arriving at it is the event.
        crossed = []
        held = locked.get(r["ticker"])
        if held:
            # Already acted on today. It stands whatever price has done since.
            crossed.append(held + " TRADE")
            now_trade = held == "reclaimed"
        elif was_trade is not None and np.isfinite(trade):
            touched = spot <= trade if was_trade else spot >= trade
            if touched:
                way = "lost" if was_trade else "reclaimed"
                crossed.append(way + " TRADE")
                locked[r["ticker"]] = way
                fresh_lock = True

        # Read against live price, because Hedgeye adds on where price is now, not
        # where it closed. Their three adds on 2026-08-28 all sat mid-range on the
        # prior close -- 0.51 to 0.67 -- and only reached the buy zone during the
        # session, at 0.20 to 0.40. A model that froze the range read at the close
        # could not see any of them arrive.
        #
        # The RANGE itself is still fixed by the close; what moves is where price
        # sits inside it. TREND stays a close-to-close judgement.
        buy_low, sell_low, buy_high, sell_high = S.range_flags(pos, edge_buy, edge_sell)
        break_low, break_high = S.break_flags(pos, edge_break)
        at_low, at_high = buy_low, sell_high

        # Intraday, a "break" is a line crossed since the close rather than one
        # crossed within the last few sessions, so the crossings feed the same
        # ladder the daily run uses. Everything downstream of that is identical,
        # which is the point of it living in one function.
        lost = [c for c in crossed if c.startswith("lost")]
        recl = [c for c in crossed if c.startswith("reclaimed")]

        # TREND's state is whatever the close concluded -- nothing intraday can
        # change it. TRADE's is the close's, overturned by a crossing since.
        bt, rt = bool(r.get("broke_trend")), bool(r.get("recl_trend"))
        bd = bool(r.get("broke_trade")) or bool(lost)
        rd = bool(r.get("recl_trade")) or bool(recl)
        if lost:
            rd = False
        elif recl:
            bd = False
        sig, why = S.decide(
            bool(r.get("is_index")), bool(r.get("cash_like")),
            100 * (hi / lo - 1) if (np.isfinite(lo) and np.isfinite(hi) and lo > 0) else 0.0,
            bt, bd, rt, rd,
            " and ".join(crossed) or (r.get("why") or ""),
            buy_low, sell_low, buy_high, sell_high, now_trade, now_trend,
            break_low=break_low, break_high=break_high,
            trend_neutral=bool(r.get("trend_neutral")),
            # A macro reference never becomes a position, live or at the close.
            is_mac=bool(r.get("is_macro")),
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
            was_above=bool(r.get("was_above")), was_below=bool(r.get("was_below")),
            # An intraday crossing happened now, so it is age 0; anything carried
            # from the close keeps the age the daily run measured.
            trend_age=0 if (bt or rt) and not r.get("broke_trend") and not r.get("recl_trend")
                      else r.get("trend_flip_days"),
            trade_age=0 if crossed else r.get("trade_flip_days"))

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
    if persist and fresh_lock and session:
        save_locks(session, locked)
    out.attrs["locked"] = locked
    n_live = int(out["live"].sum())
    if n_live:
        out.attrs["live_at"] = dt.datetime.now()
    out.attrs["n_live"] = n_live
    return out
