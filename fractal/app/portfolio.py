"""Position book: what is held, when it was added, at what price, and what to do now.

The signal engine says what the market is doing. This says what it means *for the
book actually held* -- a different question, and the one that decides whether a
signal is actionable. A break of TREND is a sell only if you own it; the low end of
the RANGE is a buy only if you do not.

The baseline is **the last completed session** -- Friday's close on a weekend run.
Every level in the report comes off that bar, so the book uses the same starting
point: a position added without an explicit price is booked at that close, on that
date. Performance is measured two ways -- since the position was opened, and over
the current session -- which are the two a reader actually acts on.

Positions live in `data/portfolio.csv`, one row per lot:

    ticker, side, entry_date, entry_price, shares, notes, status, exit_date, exit_price

Closing a lot stamps status/exit rather than deleting the row, so the book keeps
its own history.

The book is driven by the signals rather than kept by hand: `sync` opens a position
for every BUY and SELL SHORT, closes it when the ladder says SELL or COVER SHORT,
and flips it when the signal turns the other way. Several opens a day is expected
and fine -- the same ladder takes them off again when a line breaks. Trims are the
one action it does not execute: with no position sizing there is nothing to sell
some of, and treating a trim as an exit would undo the distinction between TREND
deciding whether you hold and TRADE deciding how much.

Actions are phrased as the order to place, not as a description of the position:

    LONG        buy / add long exposure
    SELL        exit the long - the line has broken
    SHORT       open or add to a short
    TRIM        sell some into strength
    COVER       buy the short back, closing it out of the book
    COVER SOME  buy back part of the short
    HOLD        nothing to do
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..data.loader import load_prices, next_session, repo_path
from . import signals as S

COLUMNS = ["ticker", "side", "entry_date", "entry_price", "shares", "units",
           "last_add", "notes", "status", "exit_date", "exit_price"]

from ..data.etf_universe import group_of

LONG, SHORT = "long", "short"
OPEN, CLOSED = "open", "closed"


# ---- position sizing (Keith McCullough's conviction framework) ----------------
# 1 unit = UNIT_PCT of capital. A position scales IN one unit at a time as the
# signal confirms, and scales OUT one unit at a time on a trim, closing only at the
# floor. This is the shape a backtest showed beats equal-weight per dollar deployed
# (+95% vs +83%, Sharpe 0.95 vs 0.88 over 3y) -- see the momentum/sizing work.
#
# Caps are per-asset-class RISK BUDGETS, expressed as a max % and converted to whole
# units. Longs run bigger than shorts by design (equities 2-6% long vs 1-3% short),
# which matches where our own signals are weakest. Every budget is a whole multiple
# of the 3% unit: shorts 3% (1 unit), commodities 6% (2), equities and fixed income
# 9% (3), FX 12% (4). Nothing rounds.
# The unit and the caps SCALE WITH THE VIX. A 3-year backtest showed fixed sizing
# leverages the book to ~198% of capital in a broad selloff (dozens of names hit
# their range lows at once and all fire buys), and 2025 -- the crash year -- was
# the model's worst at -7.9%. Sizing down with vol cut that to -2.4%, halved the
# drawdown and lifted the Sharpe from 0.00 to 0.48.
#
# Regimes by VIX: calm (<19), chop (19-29), stress (>=29). The unit shrinks with
# vol so a buy in a panic commits less, and the caps shrink with it. Calm caps are
# deliberately the LARGEST -- that is when it is safe to size up toward fully
# invested; stress caps are a third of that.
START_UNITS = 1
UNIT_BY_REGIME = {"calm": 3.0, "chop": 2.0, "stress": 1.0}
CAP_BY_REGIME = {
    "calm":   {"equity": 15.0, "commodity": 12.0, "fixed_income": 15.0,
               "fx_crypto": 18.0, "short": 3.0},
    "chop":   {"equity": 6.0,  "commodity": 4.0,  "fixed_income": 6.0,
               "fx_crypto": 8.0,  "short": 2.0},
    "stress": {"equity": 3.0,  "commodity": 2.0,  "fixed_income": 3.0,
               "fx_crypto": 4.0,  "short": 1.0},
}
UNIT_PCT = UNIT_BY_REGIME["calm"]     # legacy default (calm) for callers without a VIX

_VIX_CACHE = {}


def regime_of(vix):
    """calm / chop / stress from a VIX level. Unknown vol defaults to calm."""
    if vix is None or vix != vix:
        return "calm"
    if vix >= 29.0:
        return "stress"
    if vix >= 19.0:
        return "chop"
    return "calm"


def current_vix():
    """Latest VIX close, cached per day. Falls back to a calm 15 if unavailable."""
    import datetime as _dt
    key = _dt.date.today()
    if key not in _VIX_CACHE:
        try:
            v = load_prices(["^VIX"], verbose=False)["^VIX"]["Close"].dropna().iloc[-1]
            _VIX_CACHE[key] = float(v)
        except Exception:
            _VIX_CACHE[key] = 15.0
    return _VIX_CACHE[key]


def unit_pct(vix=None):
    """% of capital in one unit, for the VIX regime (current VIX if not given)."""
    return UNIT_BY_REGIME[regime_of(current_vix() if vix is None else vix)]


def _cap_key(ticker):
    g = group_of(ticker)
    return g if g in ("commodity", "fixed_income", "fx_crypto") else "equity"


def max_units(ticker, side, vix=None):
    """Ceiling in whole units for this name and side, in the current VIX regime."""
    reg = regime_of(current_vix() if vix is None else vix)
    cap = (CAP_BY_REGIME[reg]["short"] if side == SHORT
           else CAP_BY_REGIME[reg][_cap_key(ticker)])
    return max(1, int(round(cap / UNIT_BY_REGIME[reg])))


def size_pct(units, vix=None):
    """A position's exposure: units x the current regime's unit size."""
    return float(units) * unit_pct(vix)


def units_of(row):
    """Units on a position row; a pre-sizing row (NaN) counts as the starter unit.

    Robust to a Series, a dict, or an itertuples namedtuple.
    """
    if hasattr(row, "get"):
        u = row.get("units")
    elif hasattr(row, "units"):
        u = row.units
    else:
        try:
            u = row["units"]
        except Exception:
            u = None
    try:
        u = float(u)
    except (TypeError, ValueError):
        return START_UNITS
    return int(u) if u == u and u >= 1 else START_UNITS

# Action vocabulary: the order to place, in the same words the report uses for the
# signal that produced it. SELL LONGS flattens a long; SELL SHORT opens or adds to a
# short. They are different orders and are kept as different words.
#
# TRIM LONGS reads as a reduction and IS one as an instruction -- but the book holds
# no position size, so the lot comes off and the P&L is realised. That is why it sits
# in AUTO_CLOSE. The reason text has to say so: it read "sell some, TREND still
# holds", which describes a position that stays, against a book that was closing it.
A_LONG, A_SELL, A_SHORT, A_TRIM, A_COVER, A_COVER_SOME, A_HOLD = (
    "BUY", "SELL LONGS", "SELL SHORT", "TRIM LONGS", "COVER SHORT", "TRIM SHORTS", "HOLD")

ACTION_COLOUR = {
    A_SELL: "#ef5350", A_SHORT: "#c0392b", A_COVER: "#5c9ded",
    A_TRIM: "#d9a441", A_LONG: "#0ea37f", A_COVER_SOME: "#d9a441",
    A_HOLD: "#8b94a5",
}


def path(custom=None):
    return custom or repo_path("data", "portfolio.csv")


def load(custom=None) -> pd.DataFrame:
    p = path(custom)
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(p)
    for c in COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[COLUMNS]


def save(df: pd.DataFrame, custom=None) -> None:
    p = path(custom)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    df.to_csv(p, index=False)


def base_bar(ticker):
    """(date, close) of the last completed session for `ticker` -- the baseline bar."""
    px = load_prices([ticker.upper()], verbose=False).get(ticker.upper())
    if px is None or px.empty:
        raise ValueError("no price history for %s" % ticker.upper())
    c = px["Close"].dropna()
    return c.index[-1].date().isoformat(), float(c.iloc[-1])


def live_spot(tickers, asof=""):
    """Current price per ticker, but only when it is genuinely current.

    A quote is used only if it comes from a session after `asof`, the close the
    report was built on. Outside market hours the feed keeps returning the previous
    session's last continuous print, which is not that session's official close --
    the closing auction sets the close and the two differ. Marking a book to that
    print invents P&L on a day nothing traded: it had ENZL at +0.46% on a Sunday.

    Falls back silently per ticker; a missing quote is not a reason to fail the
    whole report, and the baseline close is the right answer when the market is shut.
    """
    from . import live as LIVE
    try:
        px, when = LIVE.quotes(list(tickers), with_times=True)
    except Exception:
        return {}
    if not asof:
        return px
    fresh = {}
    for t, v in px.items():
        ts = when.get(t)
        try:
            if ts is None or str(pd.Timestamp(ts).date()) > str(asof):
                fresh[t] = v
        except Exception:
            fresh[t] = v
    return fresh


def add_position(ticker, side, price=None, date=None, shares=None, notes="",
                 units=START_UNITS, custom=None):
    """Open a lot. Price and date default to the last completed close."""
    side = side.lower()
    if side not in (LONG, SHORT):
        raise ValueError("side must be long or short")
    ticker = ticker.upper()

    used_base = price is None or date is None
    if used_base:
        base_date, base_close = base_bar(ticker)
        price = base_close if price is None else float(price)
        date = base_date if date is None else date

    df = load(custom)
    row = {
        "ticker": ticker, "side": side, "entry_date": date,
        "entry_price": float(price), "shares": shares if shares is not None else np.nan,
        "units": int(units), "notes": notes, "status": OPEN,
        "exit_date": np.nan, "exit_price": np.nan,
    }
    new = pd.DataFrame([row], columns=COLUMNS)
    df = new if df.empty else pd.concat([df, new], ignore_index=True)
    save(df, custom)
    row["_used_base"] = used_base
    return row


def close_position(ticker, price=None, date=None, custom=None):
    df = load(custom)
    m = (df["ticker"] == ticker.upper()) & (df["status"] == OPEN)
    if not m.any():
        raise ValueError("no open position in %s" % ticker.upper())
    if price is None or date is None:
        base_date, base_close = base_bar(ticker)
        price = base_close if price is None else float(price)
        date = base_date if date is None else date
    df.loc[m, "status"] = CLOSED
    df.loc[m, "exit_date"] = date
    df.loc[m, "exit_price"] = float(price)
    save(df, custom)
    return int(m.sum()), float(price)


def open_positions(custom=None) -> pd.DataFrame:
    df = load(custom)
    return df[df["status"] == OPEN].copy()


def _action(side, sig, at_low, at_high, event):
    """The book follows the signal, which already distinguishes a trim from an exit.

    It used to re-derive that from the event text, which meant any break at all
    closed the position -- the book exited where the report only wanted a trim.
    """
    if side == LONG:
        if sig == S.REMOVE_LONG:
            return A_SELL, "%s - exit the long" % (event or "TREND broke")
        if sig == S.TRIM_LONG:
            return A_TRIM, ("%s - TREND still holds, so this is a reduction; with no "
                            "position size the lot comes off and the P&L is realised"
                            % (event or "TRADE broke"))
        if sig == S.BREAKOUT:
            return A_LONG, "broke out above the RANGE and held - add"
        # Any BUY is acted on, wherever price sits. This used to require at_low,
        # which was fine while the only way to reach ADD LONG was from the low end;
        # a re-entry on a reclaimed TRADE arrives mid-range and was silently
        # becoming a HOLD -- the book being told to buy and doing nothing.
        if sig == S.ADD_LONG:
            return A_LONG, "buy - %s" % (event or "signal is bullish")
        if at_high:
            return A_TRIM, "at the high end of the RANGE - sell some into strength"
        return A_HOLD, ""
    # short
    if sig == S.COVER_SHORT:
        return A_COVER, "%s - buy the short back to close it out" % (event or "TREND reclaimed")
    if sig == S.TRIM_SHORT:
        return A_COVER_SOME, ("%s - TREND still bearish, so this is a reduction; with no "
                              "position size the short is bought back in full"
                              % (event or "TRADE reclaimed"))
    if at_low:
        return A_COVER_SOME, ("at the low end of the RANGE - reduce the short; with no "
                              "position size it is bought back in full")
    if at_high and sig == S.ADD_SHORT:
        return A_SHORT, "at the high end and still bearish - add to the short"
    return A_HOLD, ""


# Every signal that is an order to open, and which way. This is the whole BUY and
# SELL SHORT side of the ladder, not just the volume breaks: a name at the low end
# with TREND intact has met the rules, and the rules are the position. Several a
# day is fine, because the same ladder takes them off again when a line breaks.
AUTO_OPEN = {S.ADD_LONG: LONG, S.BREAKOUT: LONG,
             S.ADD_SHORT: SHORT, S.BREAKDOWN: SHORT}

# ...and the orders that take one off. The trims are here because the book holds no
# size: "sell some" has nothing to sell some of, so the lot comes off and the P&L is
# realised. The signal still reads SELL SOME, which is the instruction; this is only
# what an unsized book can do about it, and it is why every reduction shows up as a
# closed position with a number attached rather than vanishing quietly.
AUTO_CLOSE = {S.REMOVE_LONG: LONG, S.COVER_SHORT: SHORT,
              S.TRIM_LONG: LONG, S.TRIM_SHORT: SHORT}

# Reductions. These never close the lot -- the book has no position sizing, so
# there is no size to take off -- but they are still decisions with a price and a
# P&L at the moment they were called, and that is worth recording. The position
# carries on being measured from its original entry.


def sync(sig_df: pd.DataFrame, custom=None, verbose=True, only_intraday=False):
    """Bring the book in line with today's signals. Returns (opened, closed).

    Closes run before opens so a name that has flipped can come off one side and go
    on the other in the same pass.

    Booked at the close the signal came from, so entry and exit match the bar the
    decision was made on.

    `only_intraday` restricts the pass to names that crossed a duration line during
    the current session, and books them at live spot and today's date. That is the
    one thing worth acting on before the close: a line cleared by more than the
    crossing buffer has happened, whereas a range-edge read at 11am is just where
    price is standing at 11am and says nothing about where it closes.
    """
    if sig_df.empty:
        return [], []
    if only_intraday:
        if "intraday" not in sig_df:
            return [], []
        moved = sig_df["intraday"].astype(str).str.strip()
        sig_df = sig_df[moved.ne("") & moved.ne("nan")]
        if sig_df.empty:
            if verbose:
                print("  no clean intraday break to act on")
            return [], []
    df = load(custom)
    opened, closed = [], []

    def _open_idx(tk, side=None):
        m = (df["ticker"] == tk) & (df["status"] == OPEN)
        if side is not None:
            m = m & (df["side"] == side)
        idx = df.index[m]
        return idx[-1] if len(idx) else None

    def _held_side(tk):
        i = _open_idx(tk)
        return df.at[i, "side"] if i is not None else None

    def _close_lot(idx, price, why, when, u_taken=None):
        """Close a lot outright, or peel one unit off and keep the rest open."""
        nonlocal df
        side = df.at[idx, "side"]; ent = float(df.at[idx, "entry_price"])
        u = units_of(df.loc[idx]); sign = 1.0 if side == LONG else -1.0
        pnl = 100.0 * sign * (price / ent - 1.0) if ent else float("nan")
        if u_taken is None or u_taken >= u:                       # full close
            df.at[idx, "status"] = CLOSED
            df.at[idx, "exit_date"] = when
            df.at[idx, "exit_price"] = float(price)
            taken = u
        else:                                                     # partial: peel 1u
            peel = {"ticker": df.at[idx, "ticker"], "side": side, "entry_date":
                    df.at[idx, "entry_date"], "entry_price": ent, "shares": np.nan,
                    "units": int(u_taken), "notes": "trim", "status": CLOSED,
                    "exit_date": when, "exit_price": float(price)}
            df = pd.concat([df, pd.DataFrame([peel], columns=COLUMNS)],
                           ignore_index=True)
            df.at[idx, "units"] = u - u_taken
            taken = u_taken
        closed.append({"ticker": df.at[idx, "ticker"], "price": float(price),
                       "why": why, "pnl_pct": pnl, "units": int(taken)})
        if verbose:
            tag = "" if u_taken is None else " (%du of %d)" % (int(taken), int(u))
            print("  closed %s at %.2f (%s%s) - %+.2f%% since entry"
                  % (df.at[idx, "ticker"], price, why, tag, pnl))

    for r in sig_df.itertuples():
        sig = getattr(r, "signal", None)
        tk, price = r.ticker, float(r.spot)
        when = next_session(getattr(r, "asof", "")) if only_intraday else getattr(r, "asof", "")

        side_out = AUTO_CLOSE.get(sig)                # a trim/remove for this side
        if side_out is not None:
            idx = _open_idx(tk, side_out)
            if idx is not None:
                full = sig in (S.REMOVE_LONG, S.COVER_SHORT)
                u = units_of(df.loc[idx])
                if full or u <= START_UNITS:
                    _close_lot(idx, price, sig, when)          # flatten
                else:
                    _close_lot(idx, price, sig, when, u_taken=1)  # trim one unit

        side_in = AUTO_OPEN.get(sig)                  # a buy/short for this side
        if side_in is None or _held_side(tk) == side_in and                 units_of(df.loc[_open_idx(tk, side_in)]) >= max_units(tk, side_in):
            continue                                  # nothing to do / already at cap
        idx = _open_idx(tk, side_in)
        other = _open_idx(tk)
        if idx is None and other is not None:         # holding the other way: flip
            _close_lot(other, price, "%s - flipping to %s" % (sig, side_in), when)
        if idx is not None:                           # scale IN one unit
            # ONCE PER SESSION. The live job re-syncs every few minutes, so a
            # persistent ADD signal would otherwise ladder the same position on
            # every run -- on 9 Sep 2026 four names reached 4 units in an afternoon
            # that way. last_add records the session a unit was last added; a repeat
            # add in the same session is ignored (the position is left as it is).
            if str(df.at[idx, "last_add"]) == str(when):
                pass                                   # already added this session
            else:
                u = units_of(df.loc[idx]); ent = float(df.at[idx, "entry_price"])
                df.at[idx, "entry_price"] = (u * ent + price) / (u + 1)   # blended
                df.at[idx, "units"] = u + 1
                df.at[idx, "last_add"] = str(when)
                opened.append({"ticker": tk, "side": side_in, "entry_price": price,
                               "units": u + 1, "added": 1})
                if verbose:
                    print("  added %s %s -> %du at %.2f (%s)"
                          % (side_in, tk, u + 1, price, sig))
        else:                                         # fresh position at the floor
            fresh = {"ticker": tk, "side": side_in, "entry_date": str(when) or None,
                     "entry_price": float(price), "shares": np.nan,
                     "units": START_UNITS, "last_add": str(when),
                     "notes": "auto: %s%s" % (sig,
                     " (intraday)" if only_intraday else ""), "status": OPEN,
                     "exit_date": np.nan, "exit_price": np.nan}
            df = pd.concat([df, pd.DataFrame([fresh], columns=COLUMNS)],
                           ignore_index=True)
            opened.append({"ticker": tk, "side": side_in, "entry_price": float(price),
                           "units": START_UNITS, "added": START_UNITS})
            if verbose:
                print("  opened %s %s at %.2f (%s)" % (side_in, tk, price, sig))

    save(df, custom)
    if verbose and not (opened or closed):
        print("  book already matches the signals")
    return opened, closed


def closed_on(session, custom=None):
    """Lots that came off on `session`, with what they made.

    A position leaving the book is the only moment its P&L becomes real, so it has
    to be reported rather than simply disappearing from the open list. Both surfaces
    show this; without it a reduction would look like the name had never been held.
    """
    df = load(custom)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "side", "entry_date", "entry_price",
                                     "exit_date", "exit_price", "pnl_pct", "notes"])
    out = df[(df["status"] == CLOSED) & (df["exit_date"].astype(str) == str(session))].copy()
    if out.empty:
        return out.assign(pnl_pct=[])
    sign = np.where(out["side"] == LONG, 1.0, -1.0)
    out["pnl_pct"] = 100.0 * sign * (
        out["exit_price"].astype(float) / out["entry_price"].astype(float) - 1.0)
    out["units"] = out.apply(units_of, axis=1)
    return out[["ticker", "side", "entry_date", "entry_price",
                "exit_date", "exit_price", "pnl_pct", "units", "notes"]]


def reconcile(sig_df: pd.DataFrame, custom=None, live=False) -> pd.DataFrame:
    """Open positions joined to today's signals, with P&L and an action.

    `spot` is the current price: the live quote when `live` is set and the market
    is open, otherwise the baseline close. `pnl_pct` runs from the entry price and
    `day_pct` is the position's move over the current session -- both signed by
    side, so a short that falls shows a gain.
    """
    pos = open_positions(custom)
    if pos.empty:
        return pd.DataFrame()

    s = sig_df.set_index("ticker")
    asof = str(sig_df["asof"].max()) if "asof" in sig_df else ""
    quotes = (live_spot(sorted(set(pos["ticker"]) & set(s.index)), asof=asof)
              if live else {})

    rows = []
    for p in pos.itertuples():
        if p.ticker not in s.index:
            rows.append({"ticker": p.ticker, "side": p.side, "entry_date": p.entry_date,
                         "entry_price": float(p.entry_price), "shares": p.shares,
                         "units": units_of(p), "size_pct": size_pct(units_of(p)),
                         "spot": np.nan, "pnl_pct": np.nan,
                         "day_pct": np.nan, "days_held": np.nan,
                         "range_low": np.nan, "range_high": np.nan, "pos_in_range": np.nan,
                         "trade": np.nan, "trend": np.nan, "trade_bull": None,
                         "trend_bull": None, "signal": None, "action": A_HOLD,
                         "action_why": "not in the watchlist - no signal computed",
                         "notes": p.notes})
            continue
        r = s.loc[p.ticker]
        base = float(r["spot"])                     # the baseline (last completed) close
        spot = float(quotes.get(p.ticker, base))    # current price
        entry = float(p.entry_price)
        sign = 1.0 if p.side == LONG else -1.0
        pnl = sign * (spot / entry - 1.0)
        day = sign * float(r.get("day_pct", np.nan) or np.nan)
        act, why = _action(p.side, r["signal"], bool(r["at_low"]),
                           bool(r["at_high"]), str(r.get("event") or ""))
        # SESSIONS held, not calendar days, and never negative.
        #
        # This was `(asof - entry_date).days`, which broke two ways. An intraday lot
        # carries TODAY's date while the report is anchored to the settled PRIOR
        # close, so RSP, JPM, PSP, CRWD and SNOW all read -1 days held on 3 Sep
        # 2026. And a Friday entry read on Monday counted three days when one
        # session had passed.
        #
        # Business days are the proxy for sessions -- it ignores holidays, which is
        # a day out at worst and never negative. A lot entered on or after the
        # report session reads 0: opened this session, nothing held through yet.
        try:
            a = pd.Timestamp(r["asof"]).normalize()
            e = pd.Timestamp(p.entry_date).normalize()
            held = max(0, len(pd.bdate_range(e, a)) - 1) if a >= e else 0
        except Exception:
            held = np.nan
        rows.append({
            "ticker": p.ticker, "side": p.side, "entry_date": p.entry_date,
            "entry_price": entry, "shares": p.shares,
            "units": units_of(p), "size_pct": size_pct(units_of(p)), "spot": spot,
            "pnl_pct": 100 * pnl, "day_pct": day, "days_held": held,
            "range_low": float(r["range_low"]), "range_high": float(r["range_high"]),
            "pos_in_range": float(r["pos_in_range"]),
            "trade": float(r["trade"]), "trend": float(r["trend"]),
            "trade_bull": r["trade_bull"], "trend_bull": r["trend_bull"],
            "signal": r["signal"], "action": act, "action_why": why, "notes": p.notes,
        })
    out = pd.DataFrame(rows)
    order = {A_SELL: 0, A_COVER: 1, A_SHORT: 2, A_TRIM: 3, A_COVER_SOME: 4,
             A_LONG: 5, A_HOLD: 6}
    out["_r"] = out["action"].map(order).fillna(9)
    return (out.sort_values(["_r", "pnl_pct"], ascending=[True, False])
               .drop(columns=["_r"]).reset_index(drop=True))


# ------------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser(
        description="Track long/short positions against the signals. "
                    "The baseline is the last completed close.")
    ap.add_argument("--file", default=None, help="portfolio CSV (default data/portfolio.csv)")
    ap.add_argument("--live", action="store_true",
                    help="use live intraday quotes for the current price")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="open a position (defaults to the last close)")
    a.add_argument("ticker")
    a.add_argument("side", choices=[LONG, SHORT])
    a.add_argument("--price", type=float, default=None, help="default: the last close")
    a.add_argument("--date", default=None, help="YYYY-MM-DD (default: the last close date)")
    a.add_argument("--shares", type=float, default=None)
    a.add_argument("--notes", default="")

    c = sub.add_parser("close", help="close a position (defaults to the last close)")
    c.add_argument("ticker")
    c.add_argument("--price", type=float, default=None)
    c.add_argument("--date", default=None)

    sub.add_parser("list", help="show the book")
    sub.add_parser("status", help="reconcile the book against today's signals")

    args = ap.parse_args()

    if args.cmd == "add":
        r = add_position(args.ticker, args.side, args.price, args.date,
                         args.shares, args.notes, custom=args.file)
        print("added %s %s @ %.4f on %s%s"
              % (r["side"], r["ticker"], r["entry_price"], r["entry_date"],
                 "  (booked at the last close)" if r.get("_used_base") else ""))
        return 0

    if args.cmd == "close":
        n, px = close_position(args.ticker, args.price, args.date, custom=args.file)
        print("closed %d lot(s) in %s @ %.4f" % (n, args.ticker.upper(), px))
        return 0

    if args.cmd == "list":
        df = load(args.file)
        if df.empty:
            print("portfolio is empty. add one with:\n"
                  "  python -m fractal.app.portfolio add GLD long")
            return 0
        print(df.to_string(index=False))
        return 0

    sig = S.run(verbose=False)
    rec = reconcile(sig, custom=args.file, live=args.live)
    if rec.empty:
        print("no open positions. add one with:\n"
              "  python -m fractal.app.portfolio add GLD long")
        return 0
    cols = ["ticker", "side", "entry_date", "entry_price", "spot",
            "pnl_pct", "day_pct", "days_held", "action", "action_why"]
    print("baseline: %s close\n" % sig["asof"].max())
    print(rec[cols].to_string(index=False, float_format=lambda v: "%.2f" % v))
    print("\n%d open | since entry %+.2f%% | since the baseline close %+.2f%% | %s"
          % (len(rec), rec["pnl_pct"].mean(), rec["day_pct"].mean(),
             "  ".join("%s=%d" % kv for kv in rec["action"].value_counts().items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
