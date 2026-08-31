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
from ..data.etf_universe import all_etfs, group_of, is_index, yf_symbol
from ..model import adaptive_ma, range_ewma, state as state_mod
from ..model.range_ewma import volume_features

# "Near or at" the end of the range: the outer fifth. Hedgeye's own ETF Pro adds
# sat a median 0.17 into the range at the session low, so an outer-15% band was
# tighter than the thing it is meant to reproduce and missed adds they were making.
# Used when the VIX is unreadable; otherwise the band is scaled -- see EDGE_BY_VIX.
EDGE = 0.20

# The band that counts as "at the end" of the range is set by the volatility
# regime, and it is asymmetric: the side that opens risk and the side that sheds it
# get different widths. Below 19 the model leans into buying -- a quarter of the
# range counts as a buy zone while only the outer 5% counts as a place to sell. In
# the middle it is even. Above 29 it inverts: buying needs the extreme, selling
# triggers early.
#
# The band follows the DIRECTION OF THE ACTION, not which end of the range it sits
# at. A breakout is a buy that happens at the high end, so it reads against the buy
# band; a breakdown is a sell at the low end and reads against the sell band.
EDGE_BY_VIX = ((19.0, 0.25, 0.05),    # calm: quick to buy, slow to sell
               (29.0, 0.10, 0.10),    # chop: even-handed
               (None, 0.05, 0.25))    # stress: slow to buy, quick to sell

EDGE_BUY, EDGE_SELL = 0.20, 0.20      # used only when the VIX is unreadable

# The volume-confirmed break signals get their own band, and it does not move with
# the VIX. The buy/sell bands answer "is this a good place to add to a position we
# already believe in", which should widen and narrow with the regime. A breakout or
# breakdown is a different claim -- that price has left the range on conviction --
# and that claim is only credible at the extreme, whatever the VIX is doing.
#
# Keeping them coupled had a second problem: both low-end flags read the same
# range, so narrowing m_dn to put the buy band where Hedgeye actually buys also
# armed the breakdown short. One parameter was steering two unrelated decisions.
EDGE_BREAK = 0.05


def break_flags(pos, edge_break=EDGE_BREAK):
    """(break_low, break_high) -- price at the extreme, for the volume rules."""
    return pos <= edge_break, pos >= 1 - edge_break


def edge_for_vix(level):
    """(buy band, sell band) for a VIX level, or None if the VIX is unreadable."""
    if level is None or level != level:
        return None
    for ceiling, buy, sell in EDGE_BY_VIX:
        if ceiling is None or level < ceiling:
            return buy, sell
    return None


def range_flags(pos, edge_buy, edge_sell):
    """Where price sits, judged separately for buying and for selling.

    Four flags rather than two, because at one moment the same price can be inside
    the buy zone and outside the sell zone -- that asymmetry is the whole point.
    """
    if pos is None or pos != pos:
        return False, False, False, False
    return (pos <= edge_buy,          # low enough to buy
            pos <= edge_sell,         # low enough to be a breakdown
            pos >= 1 - edge_buy,      # high enough to be a breakout
            pos >= 1 - edge_sell)     # high enough to short


FRESH_DAYS = 3       # a break/reclaim counts as an event for this many sessions
# Below this range width an instrument is cash-like and raises no signal. The
# number is a tradeability floor, not a statistical one: a range this narrow means
# the edge-to-edge move is smaller than the cost of capturing it.
#
# It was 2.0, which suppressed ten of the thirty-nine positions Hedgeye actually
# runs -- including every currency ETF. The universe sorts by width with a gap
# right here: the genuine cash proxies sit at 0.4 to 1.2 (T-bills, CLOs, high
# yield, short corporates) and the currencies at 1.4 to 1.7, with nothing between.
# 1.25 lands in that gap. It is a soft boundary and widths move, so a name near it
# will drift in and out; that is preferable to excluding a whole asset class.
MIN_RANGE_PCT = 1.25
SETTLE_MULT = 3      # an EMA needs roughly 3x its span of history to shed its seed
VOL_W1, VOL_W3 = 21, 63     # 1-month and 3-month volume baselines (TRADE / TREND)
VOL_Z = 2.0          # |z| on log volume beyond which a session counts as an outlier

# How far past a duration line price must travel before the move counts as a
# crossing rather than jitter, as a fraction of the RANGE width. Scaling to the
# range rather than to a flat percentage is what makes one number work across the
# list: it lands near 0.05% of spot on equity ETFs, where a move that small is
# noise, and under it on fixed income, where the same move is real.

# Instruments whose whole Risk Range is narrower than MIN_RANGE_PCT do not move
# enough for these rules to mean anything - a T-bill fund "breaking TRADE" by five
# basis points is not a sell. The Hedgeye validation made the same point from the
# other side: every TREND-direction mismatch was a near-cash bond ETF where price
# sits on the line and bull/bear is noise. Such names are still reported, with
# signal None and a `cash_like` flag.

ADD_LONG = "ADD LONG"
BREAKOUT = "BREAKOUT"
BREAKDOWN = "BREAKDOWN"
TRIM_LONG = "TRIM LONG"
TRIM_SHORT = "TRIM SHORT"
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


def evaluate(ticker, ohlc, params, edge_buy=EDGE_BUY, edge_sell=EDGE_SELL,
             edge_break=EDGE_BREAK, fresh_days=FRESH_DAYS, min_range_pct=MIN_RANGE_PCT):
    """One name -> levels, range position, and any triggered signal.

    A volatility index gets levels and a direction but never a signal: you cannot
    buy the VIX, and treating it as a position would put nonsense in the book. Its
    read is macro context -- bearish TRADE and TREND means volatility is falling,
    which is supportive for risk assets.
    """
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

    buy_low, sell_low, buy_high, sell_high = range_flags(pos, edge_buy, edge_sell)
    break_low, break_high = break_flags(pos, edge_break)
    # The reported flags are the actionable ones: the zone you would buy in, and the
    # zone you would short in. They are what the alert strip renders.
    at_low, at_high = buy_low, sell_high

    # The bar's own move, close to close. Carried so the book can show a daily P&L
    # without re-reading prices, and so the live re-pricer has something to
    # overwrite rather than a column that only exists intraday.
    day_pct = (100 * (spot / float(close.iloc[-2]) - 1)
               if len(close) > 1 and float(close.iloc[-2]) else np.nan)

    # Whether the previous close was already at an edge, measured against that
    # session's own range rather than today's -- the envelope moves, so comparing
    # yesterday's close to today's low would report breaks that never happened.
    # This uses the break band, not the buy/sell band, because the only thing
    # was_above/was_below feed is the break tranche -- "held" has to mean still in
    # the zone the break was called from, or the entry and its exit disagree.
    was_above = was_below = False
    if len(close) > 1:
        p_close = float(close.iloc[-2])
        p_lo = float(rng["range_low"].iloc[-2])
        p_hi = float(rng["range_high"].iloc[-2])
        if np.isfinite(p_lo) and np.isfinite(p_hi) and p_hi > p_lo:
            p_pos = (p_close - p_lo) / (p_hi - p_lo)
            was_above = p_pos >= 1 - edge_break    # yesterday's breakout zone
            was_below = p_pos <= edge_break        # yesterday's breakdown zone

    # Volume against its 1-month and 3-month baselines. Volume is lognormal and its
    # scale differs by orders of magnitude across the list, so "unusual" is measured
    # as a z-score of log volume against the same 3-month window rather than as a
    # fixed percentage -- a +60% day is routine for a thin fund and remarkable for
    # a mega-ETF. The percentages are still reported, because that is how the
    # deviation is normally quoted.
    vol = vol_1m = vol_3m = v1r = v3r = vol_z = vol_z1 = np.nan
    vol_flag = ""
    prev_surge = False
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
            lv = np.log(v.replace(0, np.nan))
            # Computed as a series rather than a single value: the failed-break
            # test needs to know whether YESTERDAY was a surge, because a break
            # only failed if there was a break, and a break needed volume.
            z1s = (lv - lv.rolling(VOL_W1).mean()) / lv.rolling(VOL_W1).std(ddof=1)
            z3s = (lv - lv.rolling(VOL_W3).mean()) / lv.rolling(VOL_W3).std(ddof=1)
            surge_s = (z1s >= VOL_Z) | (z3s >= VOL_Z)
            prev_surge = bool(surge_s.iloc[-2]) if len(surge_s) > 1 else False
            if np.isfinite(vol) and vol > 0:
                for win, name in ((VOL_W1, "z1"), (VOL_W3, "z3")):
                    if len(lv.dropna()) <= win:
                        continue
                    w = lv.dropna().iloc[-win:]
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
    # Both halves are reported when both happened. Reporting only the break left
    # EWQ reading "broke TRADE - TREND reclaimed, close the short", which contradicts
    # itself: the ladder had taken the reclaim branch while the event text still
    # described the break.
    parts = []
    if broke_trend or broke_trade:
        parts.append("broke %s" % " and ".join(
            [n for n, b in (("TREND", broke_trend), ("TRADE", broke_trade)) if b]))
    if recl_trend or recl_trade:
        parts.append("reclaimed %s" % " and ".join(
            [n for n, b in (("TREND", recl_trend), ("TRADE", recl_trade)) if b]))
    event = ", ".join(parts)

    # A range break counts as volume-confirmed if either baseline says so. The
    # 3-month window is the steadier read; the 1-month catches a name whose volume
    # has only just picked up, which is exactly the breakout case.
    vol_surge = bool(max([z for z in (vol_z, vol_z1) if np.isfinite(z)] or [0]) >= VOL_Z)

    # A break only failed if there was one, and a break needed volume. Without this
    # any name that merely sat at an edge yesterday and drifted back read as a
    # breakout add to cut -- NVDA, CIBR, IGV and BAC all did, none having broken out.
    was_above = was_above and prev_surge
    was_below = was_below and prev_surge

    sig, why = decide(is_index(ticker), cash_like, width_pct,
                      broke_trend, broke_trade, recl_trend, recl_trade, event,
                      buy_low, sell_low, buy_high, sell_high, trade_bull, trend_bull,
                      break_low=break_low, break_high=break_high,
                      outside_high=bool(spot > hi), outside_low=bool(spot < lo),
                      vol_surge=vol_surge, was_above=was_above, was_below=was_below,
                      trend_age=d_trend, trade_age=d_trade)

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
        "is_index": bool(is_index(ticker)),
        "history_bars": int(bars),
        "young": bool(young),
        "at_low": bool(at_low),
        "at_high": bool(at_high),
        "buy_low": bool(buy_low),
        "sell_low": bool(sell_low),
        "break_low": bool(break_low),
        "break_high": bool(break_high),
        "buy_high": bool(buy_high),
        "sell_high": bool(sell_high),
        "day_pct": day_pct,
        # Carried so the intraday re-pricer can see a break that happened on an
        # earlier close. Without them it only knows about lines crossed since the
        # open, and a name that broke two days ago and has since drifted to the low
        # end reads as a fresh WATCHLIST instead of the exit it still is.
        # Sessions since each line last flipped. 0 means it flipped on this bar,
        # which is what separates "has just broken" from "broke and is still within
        # the freshness window" -- the difference between one alert and three.
        "trade_flip_days": d_trade,
        "trend_flip_days": d_trend,
        "broke_trend": bool(broke_trend),
        "broke_trade": bool(broke_trade),
        "recl_trend": bool(recl_trend),
        "recl_trade": bool(recl_trade),
        "outside_low": bool(spot < lo),
        "outside_high": bool(spot > hi),
        "was_above": bool(was_above),
        "was_below": bool(was_below),
        "vol_surge": bool(vol_surge),
    }


def decide(is_idx, cash_like, width_pct, broke_trend, broke_trade,
           recl_trend, recl_trade, event, buy_low, sell_low, buy_high, sell_high,
           trade_bull, trend_bull, outside_high=False, outside_low=False,
           vol_surge=False, was_above=False, was_below=False,
           break_low=False, break_high=False,
           trend_age=None, trade_age=None):
    """(signal, why) from a name's current state. The only place the ladder lives.

    It used to be written twice -- once against closes here and once against live
    quotes in `live.reprice` -- which meant every rule change had to be made in two
    places and stayed correct only by luck.

    Three tiers, in order: TREND, then TRADE, then RANGE. TREND decides whether you
    hold at all, TRADE decides how much, and where price sits in the range only
    matters once neither line has moved. Events outrank positions in both
    directions -- a reclaim ranks with a break, not below the range reads, or the
    model adds to a short on the day that short's line was taken back.
    """
    if is_idx:
        return None, "volatility index - context only, not a position"
    if cash_like:
        return None, "range only %.2f%% wide - too tight for a signal" % width_pct

    # ---- 1. TREND ----------------------------------------------------------
    # TREND decides whether you hold the position at all, so nothing outranks it.
    # A break is the exit and a reclaim ends the short, whatever price is doing
    # inside the range at that moment.
    #
    # Ranking by duration must not become ranking by staleness. Both flags stay
    # true for FRESH_DAYS, so a TREND event from three sessions ago would otherwise
    # outrank a TRADE break happening now -- QTUM reclaimed TREND on Monday and
    # broke TRADE on Friday, and read "close the short" instead of "trim". When the
    # two conflict the more recent one wins; a tie goes to TREND.
    trend_first = (trend_age is None or trade_age is None
                   or not (broke_trade or recl_trade) or trend_age <= trade_age)
    if broke_trend and trend_first:
        return REMOVE_LONG, event + " - TREND is the position, exit it"
    if recl_trend and trend_first:
        return COVER_SHORT, event + " - TREND reclaimed, close the short"

    # ---- 2. TRADE ----------------------------------------------------------
    # TRADE breaking with TREND intact is a partial exit, which is Hedgeye's own
    # wording: sell some, keep the trend position. The book has no position sizing,
    # so it books the reduction as the lot coming off and realises the P&L -- the
    # signal says "sell some" and the book records what that was worth. TREND going
    # too is the full exit.
    if broke_trade and trend_bull:
        return TRIM_LONG, event + " with TREND still bullish - sell some, wait to buy back"
    if broke_trade:
        return REMOVE_LONG, event + " with TREND already bearish - exit"
    if recl_trade and trend_bull is False:
        return TRIM_SHORT, event + " with TREND still bearish - buy some back, wait to re-short"
    # The re-entry, and it fires on the day of the reclaim only. Every other event
    # flag stays true for FRESH_DAYS so the report keeps showing it, which is right
    # for a sell -- closing an already-closed position is a no-op -- but wrong for a
    # buy, because opening is not idempotent. Reading a three-session-old reclaim as
    # a buy told the book to open seven names it would already have owned: NVDA, USO,
    # CIBR and AMZN never traded below TRADE on the session at all.
    if recl_trade and trend_bull and trade_age == 0:
        return ADD_LONG, event + " with TREND still bullish - buy it back"
    # A reclaim on a bullish-TREND name that is not today's news is not an
    # instruction at all: the buy already happened when it reclaimed, and there is
    # no short to close. Let it fall through to where price actually sits.
    if recl_trade and not trend_bull:
        return COVER_SHORT, event + " - close the short"
    # A TREND event that yielded to a fresher TRADE one still stands behind it.
    if broke_trend:
        return REMOVE_LONG, event + " - TREND is the position, exit it"
    if recl_trend:
        return COVER_SHORT, event + " - TREND reclaimed, close the short"

    # ---- 3. RANGE ----------------------------------------------------------
    # Where price sits, which only matters once neither duration line has moved.
    # A break is read at the EDGE of the range rather than only outside it: WEAT
    # closed 4 cents inside its range high on 2 sigma of volume, which is a breakout
    # being bought, not a name to trim into. Volume decides whether being at an edge
    # means anything -- on ordinary volume it is drift, and mean-reverts more often
    # than it continues. TREND still sets direction, so price pressing the high while
    # TREND is bearish is a squeeze and reads as a short.
    if break_high and vol_surge and trend_bull:
        held = " and has held" if was_above else " - watch whether it holds"
        return BREAKOUT, "at the RANGE high on heavy volume%s" % held
    if break_low and vol_surge and trend_bull is False:
        held = " and has held" if was_below else " - watch whether it holds"
        return BREAKDOWN, "at the RANGE low on heavy volume%s" % held

    # A break that failed takes the add off. A trim rather than an exit: what is cut
    # is the breakout tranche, not the core position, which TREND still governs. The
    # direction test mirrors the trigger, or a bullish-TREND name gets told to cover
    # a breakdown add that could never have been taken.
    if was_above and not break_high and trend_bull:
        return TRIM_LONG, "broke out but failed to hold the RANGE high - cut the breakout add"
    if was_below and not break_low and trend_bull is False:
        return TRIM_SHORT, "broke down but failed to hold the RANGE low - cover the breakdown add"

    # A long is never opened against a bearish TREND. Price being cheap inside a
    # downtrend is not a buy -- the same mistake as reading a spike above the range
    # as a breakout when TREND disagrees.
    # Each edge cuts both ways, and which way depends on the direction you are
    # positioned. In a bullish name the low end is where you buy and the high end is
    # where you take some off; in a bearish one the high end is where you short and
    # the low end is where you buy some back. Only the two opening cases used to be
    # here, so a bullish name at the top of its range said nothing at all and a
    # bearish one at the bottom went on a watchlist rather than reducing the short.
    if buy_low and trend_bull and trade_bull:
        return ADD_LONG, "low end of RANGE, bullish TRADE and TREND"
    if buy_low and trend_bull:
        return WATCHLIST, "at the low end but TRADE has broken - watch for TREND to hold"
    # Opening a short needs both durations bearish, exactly as opening a long needs
    # both bullish. One of each is not a position, it is a disagreement: a bearish
    # TREND whose TRADE has been reclaimed is watched for TREND to give way, the
    # mirror of a bullish TREND whose TRADE has broken.
    if sell_high and trend_bull is False and trade_bull is False:
        return ADD_SHORT, "high end of RANGE, bearish TRADE and TREND"
    if sell_high and trend_bull is False:
        return WATCHLIST, ("at the high end but TRADE has been reclaimed - "
                           "watch for TREND to give way")
    if sell_high and trend_bull:
        return TRIM_LONG, "high end of RANGE in a bullish TREND - take some off"
    if buy_low and trend_bull is False:
        return TRIM_SHORT, "low end of RANGE in a bearish TREND - buy some back"

    # Through a range edge without the volume to back it: the weakest read here, so
    # it comes last. A duration line can sit outside the range -- GII closed with
    # TREND at 75.46 against a range high of 75.45 -- so when this sat higher up it
    # swallowed reclaims, reporting them as unconfirmed breakouts.
    if outside_high and trend_bull and not was_above:
        return WATCHLIST, "above the RANGE high but not on volume - watch for confirmation"
    if outside_low and trend_bull is False and not was_below:
        return WATCHLIST, "below the RANGE low but not on volume - watch for confirmation"
    return None, ""


# Hedgeye reads the VIX as an absolute level, not a direction: what regime you are
# in decides whether a signal is actionable at all. This is deliberately separate
# from our own bearish/bullish read of the VIX, because the two can disagree -- a
# VIX at 32 and falling is supportive by direction and defensive by level, and the
# level is the one that governs position size.
VIX_BUCKETS = (
    (20.0, "INVESTABLE", "9-19", "buy dips, normal risk", "supportive"),
    (29.0, "CHOP", "20-29", "trade the range, be aggressive on longs", "mixed"),
    (None, "DEFENSIVE", "29+", "preserve capital", "risk_off"),
)


def vix_bucket(level):
    """(name, band, guidance, stance) for a VIX level. 9-19, 20-29, 29+."""
    if level is None or level != level:
        return "", "", "", ""
    for ceiling, name, band, note, stance in VIX_BUCKETS:
        if ceiling is None or level < ceiling:
            return name, band, note, stance
    return "", "", "", ""


def vol_read(cross="", at_low=False, at_high=False):
    """(label, stance) for a volatility-index event, or ("", "") for nothing.

    VIX and MOVE never carry a position, so the ordinary reads do not apply: an
    index at the low end of its range is not a buy, and a broken TREND on it is not
    a sell. What they carry is the weather for everything else, and that inverts
    the colour convention -- falling volatility is supportive for risk assets, so it
    is the green case, while the same event on an ETF would be red.

    Because the inversion is the sort of thing a reader will not hold in their head,
    the label always says which way volatility went, and never says buy or sell.
    """
    if cross:
        down, up = "lost" in cross, "reclaimed" in cross
        if down and up:
            return cross + " - vol mixed", "mixed"
        if down:
            return cross + " - vol falling", "supportive"
        if up:
            return cross + " - vol rising", "risk_off"
        return cross, ""
    if at_high:
        return "at the high end - vol elevated", "risk_off"
    if at_low:
        return "at the low end - vol subdued", "supportive"
    return "", ""


def run(tickers=None, params=None, profile="hedgeye_anchor", edge=None,
        edge_break=EDGE_BREAK, fresh_days=FRESH_DAYS, min_range_pct=MIN_RANGE_PCT,
        verbose=True):
    """Evaluate the ETF watchlist. Returns a DataFrame, most actionable first.

    `edge` defaults to the VIX-scaled band: one band governs the whole list, because
    the volatility regime is a property of the market rather than of any one name.
    Pass a number to pin it.
    """
    params = dict(params or load_params())
    if profile:
        params["range"] = dict(params["range"])
        params["range"]["active"] = profile
    tickers = tickers or all_etfs()
    # VIX and MOVE are published under ^-prefixed symbols; everything else is its
    # own ticker. Fetch by feed symbol, report by display ticker.
    feed = {t: yf_symbol(t) for t in tickers}
    prices = load_prices(sorted(set(feed.values())), params=params, verbose=verbose)

    # One band for the whole list, set by the VIX. Falls back to the fixed EDGE if
    # the VIX is missing from this run -- a scan of three ETFs should not silently
    # change its own definition of "at the end" because it happened to omit it.
    scaled = None
    if edge is None:
        vix = prices.get(yf_symbol("VIX"))
        if vix is not None and "Close" in vix and len(vix["Close"].dropna()):
            scaled = edge_for_vix(float(vix["Close"].dropna().iloc[-1]))
        edge = scaled if scaled is not None else (EDGE_BUY, EDGE_SELL)
        if verbose:
            print("range edge: buy %.0f%% / sell %.0f%%%s"
                  % (100 * edge[0], 100 * edge[1],
                     "" if scaled is not None else "  (VIX unavailable)"))
    elif not isinstance(edge, (tuple, list)):
        edge = (float(edge), float(edge))     # a single number pins both sides
    edge_buy, edge_sell = edge

    rows, missing = [], []
    for t in tickers:
        df = prices.get(feed[t])
        if df is None or len(df) < 80:
            missing.append(t)
            continue
        r = evaluate(t, df, params, edge_buy=edge_buy, edge_sell=edge_sell,
                     edge_break=edge_break, fresh_days=fresh_days,
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
    out.attrs["edge"] = edge_buy
    out.attrs["edge_buy"] = edge_buy
    out.attrs["edge_sell"] = edge_sell
    out.attrs["edge_break"] = edge_break
    order = {REMOVE_LONG: 0, ADD_LONG: 1, ADD_SHORT: 2, WATCHLIST: 3, COVER_SHORT: 4}
    out["_rank"] = out["signal"].map(order).fillna(9)
    # inside each bucket, the most extreme range position first
    out["_edge"] = np.where(out["signal"].isin([ADD_SHORT]),
                            -out["pos_in_range"], out["pos_in_range"])
    return out.sort_values(["_rank", "_edge", "ticker"]).drop(columns=["_rank", "_edge"]).reset_index(drop=True)


# Long side first, then short, each trim beside the full action it reduces.
SIGNALS = (BREAKOUT, ADD_LONG, TRIM_LONG, REMOVE_LONG,
           BREAKDOWN, ADD_SHORT, TRIM_SHORT, COVER_SHORT, WATCHLIST)

# The report is organised by the order you would place, not by the reason it fired.
# Several reasons produce the same order -- price at the low end, a breakout, and a
# short being closed are all buys -- so they share a heading, and the per-row `why`
# keeps the reason. The signal constants stay distinct because they are dictionary
# keys and identity here has to survive the merge.
LABEL = {
    ADD_LONG: "BUY",         BREAKOUT: "BUY",
    TRIM_LONG: "SELL SOME",
    REMOVE_LONG: "SELL",
    ADD_SHORT: "SELL SHORT",  BREAKDOWN: "SELL SHORT",
    TRIM_SHORT: "BUY SOME",
    COVER_SHORT: "COVER SHORT",
    WATCHLIST: "WATCHLIST",
}
SECTIONS = ("BUY", "SELL SOME", "SELL", "SELL SHORT", "BUY SOME",
            "COVER SHORT", "WATCHLIST")


def label(sig):
    """Display heading for a signal."""
    return LABEL.get(sig, sig or "")


def members(section):
    """The signals that report under one heading, in ladder order."""
    return tuple(sig for sig in SIGNALS if LABEL.get(sig) == section)


def buckets(df):
    """Section heading -> rows, in the order the newsletter presents them."""
    lab = df["signal"].map(LABEL)
    return {name: df[lab == name] for name in SECTIONS}
