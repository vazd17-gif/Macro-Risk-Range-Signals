"""Position book: what is held, when it was added, at what price, and what to do now.

The signal engine says what the market is doing. This says what it means *for the
book actually held* -- a different question, and the one that decides whether a
signal is actionable. A break of TREND is a sell only if you own it; the low end of
the RANGE is a buy only if you do not.

The baseline is **the last completed session** -- Friday's close on a weekend run.
Every level in the report comes off that bar, so the book uses the same starting
point: a position added without an explicit price is booked at that close, on that
date. Performance is then measured two ways -- from the entry price to the current
spot, and from the baseline close to the current spot -- so a position opened at
the baseline and one opened weeks ago are both measured against the same "now".

Positions live in `data/portfolio.csv`, one row per lot:

    ticker, side, entry_date, entry_price, shares, notes, status, exit_date, exit_price

Closing a lot stamps status/exit rather than deleting the row, so the book keeps
its own history.

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
import datetime as dt
import os

import numpy as np
import pandas as pd

from ..data.loader import load_prices, repo_path
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


def live_spot(tickers):
    """Current price per ticker: the live quote when the market is open, else the baseline close.

    Falls back silently per ticker -- a missing quote is not a reason to fail the
    whole report, and the baseline close is the correct answer outside market hours.
    """
    out = {}
    try:
        import yfinance as yf
        import warnings
        warnings.filterwarnings("ignore")
        data = yf.download(list(tickers), period="1d", interval="1m",
                           progress=False, group_by="ticker", threads=True)
        for t in tickers:
            try:
                sub = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                c = sub["Close"].dropna()
                if len(c):
                    out[t] = float(c.iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return out


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
    broke = event.startswith("broke")
    reclaimed = event.startswith("reclaimed")
    if side == LONG:
        if broke or sig == S.REMOVE_LONG:
            return A_SELL, "%s - exit the long" % (event or "broke the line")
        if at_high:
            return A_TRIM, "at the high end of the RANGE - sell some into strength"
        if at_low and sig == S.ADD_LONG:
            return A_LONG, "at the low end and still bullish - add"
        return A_HOLD, ""
    # short
    if reclaimed or sig == S.COVER_SHORT:
        return A_COVER, "%s - buy the short back to close it out" % (event or "reclaimed the line")
    if at_low:
        return A_COVER_SOME, "at the low end of the RANGE - buy back some of the short"
    if at_high and sig == S.ADD_SHORT:
        return A_SHORT, "at the high end and still bearish - add to the short"
    return A_HOLD, ""


def reconcile(sig_df: pd.DataFrame, custom=None, live=False) -> pd.DataFrame:
    """Open positions joined to today's signals, with P&L and an action.

    `spot` is the current price: the live quote when `live` is set and the market
    is open, otherwise the baseline close. `pnl_pct` runs from the entry price;
    `since_close_pct` runs from the baseline close, so positions opened at different times
    are still comparable against the same baseline.
    """
    pos = open_positions(custom)
    if pos.empty:
        return pd.DataFrame()

    s = sig_df.set_index("ticker")
    quotes = live_spot(sorted(set(pos["ticker"]) & set(s.index))) if live else {}

    rows = []
    for p in pos.itertuples():
        if p.ticker not in s.index:
            rows.append({"ticker": p.ticker, "side": p.side, "entry_date": p.entry_date,
                         "entry_price": float(p.entry_price), "shares": p.shares,
                         "base_close": np.nan, "spot": np.nan, "pnl_pct": np.nan,
                         "since_close_pct": np.nan, "days_held": np.nan,
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
        since_base = sign * (spot / base - 1.0)
        act, why = _action(p.side, r["signal"], bool(r["at_low"]),
                           bool(r["at_high"]), str(r.get("event") or ""))
        try:
            held = (pd.Timestamp(r["asof"]) - pd.Timestamp(p.entry_date)).days
        except Exception:
            held = np.nan
        rows.append({
            "ticker": p.ticker, "side": p.side, "entry_date": p.entry_date,
            "entry_price": entry, "shares": p.shares, "base_close": base, "spot": spot,
            "pnl_pct": 100 * pnl, "since_close_pct": 100 * since_base, "days_held": held,
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
    cols = ["ticker", "side", "entry_date", "entry_price", "base_close", "spot",
            "pnl_pct", "since_close_pct", "days_held", "action", "action_why"]
    print("baseline: %s close\n" % sig["asof"].max())
    print(rec[cols].to_string(index=False, float_format=lambda v: "%.2f" % v))
    print("\n%d open | since entry %+.2f%% | since the baseline close %+.2f%% | %s"
          % (len(rec), rec["pnl_pct"].mean(), rec["since_close_pct"].mean(),
             "  ".join("%s=%d" % kv for kv in rec["action"].value_counts().items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
