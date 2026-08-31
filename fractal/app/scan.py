"""Daily scan: one row per ticker, sortable by the setups worth looking at.

Screens implemented (build spec section 7):
  at_top / at_bottom     price in the outer decile of the RANGE
  first_break_trade      TRADE just flipped while TREND holds  -> phase 2
  cross                  TRADE and TREND crossed each other
  trend_flip             TREND flipped -> phase 5, the regime change
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices
from ..data.universe import all_tickers
from ..pipeline import compute_one
from ..model import adaptive_ma, state as state_mod

EDGE = 0.10          # "at the edge" = outer 10% of the range


def add_screens(row, hist_states, lines):
    """Flags that need yesterday as well as today."""
    pos = row.get("pos_in_range")
    row["at_top"] = bool(pos is not None and pos >= 1 - EDGE)
    row["at_bottom"] = bool(pos is not None and pos <= EDGE)

    st = hist_states.dropna(subset=["trade_bull", "trend_bull"])
    row["first_break_trade"] = False
    row["trend_flip"] = False
    row["cross"] = False
    if len(st) >= 2:
        now, prev = st.iloc[-1], st.iloc[-2]
        row["trend_flip"] = bool(now["trend_bull"] != prev["trend_bull"])
        row["first_break_trade"] = bool(
            now["trade_bull"] != prev["trade_bull"] and now["trend_bull"] == prev["trend_bull"]
        )
    lv = lines.dropna(subset=["trade", "trend"])
    if len(lv) >= 2:
        a = lv["trade"].iloc[-1] - lv["trend"].iloc[-1]
        b = lv["trade"].iloc[-2] - lv["trend"].iloc[-2]
        row["cross"] = bool(np.sign(a) != np.sign(b))
    return row


def run(tickers=None, params=None, source=None, verbose=True):
    params = params or load_params()
    tickers = tickers or all_tickers()
    prices = load_prices(tickers, params=params, source=source, verbose=verbose)

    rows, missing = [], []
    for t in tickers:
        df = prices.get(t)
        if df is None or len(df) < 80:
            missing.append(t)
            continue
        row = compute_one(t, df, params, with_texture=False)
        if "error" in row:
            missing.append(t)
            continue
        close = df["Close"].dropna()
        lines = adaptive_ma.compute(close, params)
        rows.append(add_screens(row, state_mod.state_series(close, lines), lines))

    if verbose and missing:
        print("[scan] no usable history: " + ", ".join(missing))
    out = pd.DataFrame(rows)
    return out.sort_values(["group", "ticker"]).reset_index(drop=True) if len(out) else out


def to_table(df, sort=None, only=None):
    d = df.copy()
    if only:
        d = d[d[only]] if only in d else d.iloc[0:0]
    if sort == "range":
        d = d.sort_values("pos_in_range")
    elif sort == "state":
        d = d.sort_values(["state", "ticker"])
    cols = ["ticker", "group", "asof", "spot", "range_low", "range_high",
            "pct_to_low", "pct_to_high", "pos_in_range",
            "trade", "trend", "tail", "state", "phase"]
    cols = [c for c in cols if c in d]
    return d[cols]


def main():
    ap = argparse.ArgumentParser(description="Daily fractal trend/range scan.")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--sort", choices=["range", "state", "ticker"], default="ticker")
    ap.add_argument("--only", default=None,
                    choices=["at_top", "at_bottom", "first_break_trade", "cross", "trend_flip"])
    ap.add_argument("--csv", default=None)
    ap.add_argument("--source", default=None, choices=["ib", "yahoo"])
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    df = run(tickers, source=args.source)
    if df.empty:
        print("nothing to scan")
        return

    table = to_table(df, sort=args.sort, only=args.only)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(table.to_string(index=False, float_format=lambda v: "%.3f" % v))

    flags = ["at_top", "at_bottom", "first_break_trade", "cross", "trend_flip"]
    print("\nscreens:")
    for f in flags:
        hits = df[df[f]]["ticker"].tolist() if f in df else []
        print("  %-18s %s" % (f, ", ".join(hits) if hits else "-"))
    print("\nstates: " + ", ".join("%s=%d" % (k, v)
                                   for k, v in df["state"].value_counts().items()))

    if args.csv:
        path = args.csv
        df.to_csv(path, index=False)
        print("\nwrote " + path)


if __name__ == "__main__":
    main()
