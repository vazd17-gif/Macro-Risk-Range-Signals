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

EDGE = 0.15          # "near or at" the end of the range: outer 15%
FRESH_DAYS = 3       # a break/reclaim counts as an event for this many sessions
MIN_RANGE_PCT = 2.0  # below this range width an ETF is cash-like: no signals
SETTLE_MULT = 3      # an EMA needs roughly 3x its span of history to shed its seed
VOL_W1, VOL_W3 = 21, 63     # 1-month and 3-month volume baselines (TRADE / TREND)
VOL_Z = 2.0          # |z| on log volume beyond which a session counts as an outlier

# How far past a duration line price must travel before the move counts as a
# crossing rather than jitter, as a fraction of the RANGE width. Scaling to the
# range rather than to a flat percentage is what makes one number work across the
# list: it lands near 0.05% of spot on equity ETFs, where a move that small is
# noise, and under it on fixed income, where the same move is real.
CROSS_BUFFER = 0.02

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


def evaluate(ticker, ohlc, params, edge=EDGE, fresh_days=FRESH_DAYS,
             min_range_pct=MIN_RANGE_PCT):
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

    at_low = pos <= edge
    at_high = pos >= 1 - edge

    # Whether the previous close was already at an edge, measured against that
    # session's own range rather than today's -- the envelope moves, so comparing
    # yesterday's close to today's low would report breaks that never happened.
    # This uses the same `edge` band as the trigger, so "held" means still in the
    # zone the break was called from.
    was_above = was_below = False
    if len(close) > 1:
        p_close = float(close.iloc[-2])
        p_lo = float(rng["range_low"].iloc[-2])
        p_hi = float(rng["range_high"].iloc[-2])
        if np.isfinite(p_lo) and np.isfinite(p_hi) and p_hi > p_lo:
            p_pos = (p_close - p_lo) / (p_hi - p_lo)
            was_above, was_below = p_pos >= 1 - edge, p_pos <= edge

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
    event = ""
    if broke_trend or broke_trade:
        event = "broke %s" % " and ".join(
            [n for n, b in (("TREND", broke_trend), ("TRADE", broke_trade)) if b])
    elif recl_trend or recl_trade:
        event = "reclaimed %s" % " and ".join(
            [n for n, b in (("TREND", recl_trend), ("TRADE", recl_trade)) if b])

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
                      at_low, at_high, trade_bull, trend_bull,
                      outside_high=bool(spot > hi), outside_low=bool(spot < lo),
                      vol_surge=vol_surge, was_above=was_above, was_below=was_below)

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
        # Carried so the intraday re-pricer can see a break that happened on an
        # earlier close. Without them it only knows about lines crossed since the
        # open, and a name that broke two days ago and has since drifted to the low
        # end reads as a fresh WATCHLIST instead of the exit it still is.
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
           recl_trend, recl_trade, event, at_low, at_high,
           trade_bull, trend_bull, outside_high=False, outside_low=False,
           vol_surge=False, was_above=False, was_below=False):
    """(signal, why) from a name's current state. The only place the ladder lives.

    It used to be written twice -- once against closes here and once against live
    quotes in `live.reprice` -- which meant every rule change had to be made in two
    places and stayed correct only by luck.

    Order matters, and encodes what outranks what: a broken TREND is a regime
    change and beats everything, a broken TRADE with TREND intact is a trim rather
    than an exit, and the range-edge reads come last because they describe where
    price is rather than what just happened.
    """
    if is_idx:
        return None, "volatility index - context only, not a position"
    if cash_like:
        return None, "range only %.2f%% wide - too tight for a signal" % width_pct

    # TREND is the line that decides whether you hold the position at all; TRADE
    # only decides how much. So a TREND break is the exit, and a TRADE break with
    # TREND still bullish is a trim -- calling that an exit overstated about a third
    # of the sell list. A TRADE break with TREND already bearish is not a trim
    # either: there should be no long left to trim, so it reads as the exit.
    if broke_trend:
        return REMOVE_LONG, event + " - TREND is the position, exit it"
    if broke_trade and trend_bull:
        return TRIM_LONG, event + " with TREND still bullish - trim, do not exit"
    if broke_trade:
        return REMOVE_LONG, event + " with TREND already bearish - exit"

    # A break is read at the EDGE of the range rather than only outside it. Waiting
    # for price to clear the high tick misses the move: WEAT closed 4 cents inside
    # its range high on 2 sigma of volume, which is a breakout being bought, not a
    # name to trim into. So the same outer 15% the alert strip uses marks the zone,
    # and volume decides whether being there means anything -- at the edge on
    # ordinary volume is drift, and mean-reverts more often than it continues.
    # TREND still sets direction: price pressing the high while TREND is bearish is
    # a squeeze, and still reads as a short.
    if at_high and vol_surge and trend_bull:
        held = " and has held" if was_above else " - watch whether it holds"
        return BREAKOUT, "at the RANGE high on heavy volume%s" % held
    if at_low and vol_surge and trend_bull is False:
        held = " and has held" if was_below else " - watch whether it holds"
        return BREAKDOWN, "at the RANGE low on heavy volume%s" % held

    # Right through the range edge but without the volume to back it. Not a break,
    # but the name to watch for confirmation tomorrow.
    if outside_high and trend_bull and not was_above:
        return WATCHLIST, "above the RANGE high but not on volume - watch for confirmation"
    if outside_low and trend_bull is False and not was_below:
        return WATCHLIST, "below the RANGE low but not on volume - watch for confirmation"

    # The break failed and the add taken on it comes off. A trim rather than an
    # exit: what is cut is the breakout tranche, not the core position, which TREND
    # still governs. With no position sizing in the book that is a full exit in
    # practice, which is the conservative direction to be wrong in.
    # The direction test mirrors the trigger. Without it a name whose TREND is
    # bullish was told to cover a breakdown add that could never have been taken,
    # because BREAKDOWN requires bearish TREND in the first place.
    if was_above and not at_high and trend_bull:
        return TRIM_LONG, "broke out but failed to hold the RANGE high - cut the breakout add"
    if was_below and not at_low and trend_bull is False:
        return TRIM_SHORT, "broke down but failed to hold the RANGE low - cover the breakdown add"

    if at_low and trade_bull is False and trend_bull is False:
        return WATCHLIST, "at the low end but bearish TRADE and TREND - watch, no action yet"
    if at_low and trade_bull is False and trend_bull:
        return WATCHLIST, "at the low end but TRADE has broken - watch for TREND to hold"
    if at_low and (trade_bull or trend_bull):
        both = ("TRADE and TREND" if (trade_bull and trend_bull)
                else ("TRADE" if trade_bull else "TREND"))
        return ADD_LONG, "low end of RANGE, bullish %s" % both
    if at_high and (trade_bull is False or trend_bull is False):
        both = ("TRADE and TREND" if (trade_bull is False and trend_bull is False)
                else ("TRADE" if trade_bull is False else "TREND"))
        return ADD_SHORT, "high end of RANGE, bearish %s" % both
    # The mirror image on the short side. Reclaiming TREND ends the short;
    # reclaiming TRADE while TREND is still bearish only reduces it.
    if recl_trend:
        return COVER_SHORT, event + " - TREND reclaimed, close the short"
    if recl_trade and trend_bull is False:
        return TRIM_SHORT, event + " with TREND still bearish - buy back some"
    if recl_trade:
        return COVER_SHORT, event + " - close the short"
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


def line_band(range_low, range_high):
    """Distance from a duration line inside which price counts as sitting on it."""
    if not (np.isfinite(range_low) and np.isfinite(range_high)) or range_high <= range_low:
        return 0.0
    return CROSS_BUFFER * (range_high - range_low)


def run(tickers=None, params=None, profile="hedgeye_anchor", edge=EDGE,
        fresh_days=FRESH_DAYS, min_range_pct=MIN_RANGE_PCT, verbose=True):
    """Evaluate the ETF watchlist. Returns a DataFrame, most actionable first."""
    params = dict(params or load_params())
    if profile:
        params["range"] = dict(params["range"])
        params["range"]["active"] = profile
    tickers = tickers or all_etfs()
    # VIX and MOVE are published under ^-prefixed symbols; everything else is its
    # own ticker. Fetch by feed symbol, report by display ticker.
    feed = {t: yf_symbol(t) for t in tickers}
    prices = load_prices(sorted(set(feed.values())), params=params, verbose=verbose)

    rows, missing = [], []
    for t in tickers:
        df = prices.get(feed[t])
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
    REMOVE_LONG: "SELL",     BREAKDOWN: "SELL",
    ADD_SHORT: "SELL SHORT",
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
