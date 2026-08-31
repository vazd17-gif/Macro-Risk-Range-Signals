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

COLUMNS = ["ticker", "side", "entry_date", "entry_price", "shares",
           "notes", "status", "exit_date", "exit_price"]

LONG, SHORT = "long", "short"
OPEN, CLOSED = "open", "closed"

# Action vocabulary: the order to place. SELL exits a long; SHORT opens or adds to
# a short. They are different orders and are kept as different words.
A_LONG, A_SELL, A_SHORT, A_TRIM, A_COVER, A_COVER_SOME, A_HOLD = (
    "LONG", "SELL", "SHORT", "TRIM", "COVER", "COVER SOME", "HOLD")

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


def add_position(ticker, side, price=None, date=None, shares=None, notes="", custom=None):
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
        "notes": notes, "status": OPEN, "exit_date": np.nan, "exit_price": np.nan,
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
            return A_TRIM, "%s - sell some, TREND still holds" % (event or "TRADE broke")
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
        return A_COVER_SOME, "%s - buy back some, TREND still bearish" % (event or "TRADE reclaimed")
    if at_low:
        return A_COVER_SOME, "at the low end of the RANGE - buy back some of the short"
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
    held = open_positions(custom)
    have = dict(zip(held["ticker"], held["side"])) if not held.empty else {}
    entry = dict(zip(held["ticker"], held["entry_price"])) if not held.empty else {}
    opened, closed = [], []

    def _close(tk, price, why):
        side, ent = have.get(tk), entry.get(tk)
        n, px = close_position(tk, price=price, date=None, custom=custom)
        sign = 1.0 if side == LONG else -1.0
        pnl = (100.0 * sign * (px / float(ent) - 1.0)) if ent else float("nan")
        have.pop(tk, None)
        closed.append({"ticker": tk, "price": px, "why": why, "pnl_pct": pnl})
        if verbose:
            print("  closed %s at %.2f (%s) - %+.2f%% since entry" % (tk, px, why, pnl))

    for r in sig_df.itertuples():
        sig = getattr(r, "signal", None)
        tk, price = r.ticker, float(r.spot)
        side_out = AUTO_CLOSE.get(sig)
        if side_out and have.get(tk) == side_out:
            _close(tk, price, sig)

        side_in = AUTO_OPEN.get(sig)
        if side_in is None or have.get(tk) == side_in:
            continue
        if tk in have:                      # holding the other way: flip it
            _close(tk, price, "%s - flipping to %s" % (sig, side_in))
        # An intraday fill belongs to the session it happened in, not to the close
        # the levels came from.
        when = next_session(getattr(r, "asof", "")) if only_intraday else getattr(r, "asof", "")
        row = add_position(tk, side_in, price=price, date=str(when) or None,
                           notes="auto: %s%s" % (sig, " (intraday)" if only_intraday else ""),
                           custom=custom)
        have[tk] = side_in
        entry[tk] = row["entry_price"]
        opened.append(row)
        if verbose:
            print("  opened %s %s at %.2f (%s)" % (side_in, tk, row["entry_price"], sig))

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
    return out[["ticker", "side", "entry_date", "entry_price",
                "exit_date", "exit_price", "pnl_pct", "notes"]]


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
        try:
            held = (pd.Timestamp(r["asof"]) - pd.Timestamp(p.entry_date)).days
        except Exception:
            held = np.nan
        rows.append({
            "ticker": p.ticker, "side": p.side, "entry_date": p.entry_date,
            "entry_price": entry, "shares": p.shares, "spot": spot,
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
