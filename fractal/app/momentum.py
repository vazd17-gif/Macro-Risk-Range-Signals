"""Cross-sectional momentum book -- a SECOND strategy beside the Risk Range model.

Why this exists. Over three years to 2026-09-04, on a universe fixed as at 2023 so
survivorship cannot flatter it, monthly-rebalanced top-decile momentum returned
+172.6% net of 10bp against SPY's +71.4%, and still +149.4% at 50bp. The Risk
Range model returned +61.4% over the same window, losing money in 2024 and 2025.
Adding the model's own TREND filter to this selection COST 30 points and nearly
doubled turnover, so it is deliberately absent.

    fixed 2023 universe, net    0bp      10bp      25bp      50bp   turn/yr
    momentum top10, monthly  +178.70%  +172.58%  +163.65%  +149.37%    358%
    momentum top20, monthly  +159.47%  +154.51%  +147.23%  +135.54%    312%
    momentum top20, weekly   +213.25%  +186.95%  +151.52%  +101.80%   1410%
    SPY                       +71.44%

MONTHLY, not weekly. Weekly looks better gross and collapses under real costs;
monthly barely moves. That gap is the whole difference between a backtest artifact
and something tradeable.

EQUAL WEIGHT, not inverse-vol. Vol-scaling cost 27 points of return to buy 0.04 of
Sharpe -- it is a risk lever, not a return one.

Two things this is NOT. It is not proprietary edge: cross-sectional momentum is one
of the most documented anomalies in finance, so treat it as exposure to a known
factor. And three years cannot show its real failure mode, the sharp reversal
(2009, March 2020) where momentum can lose 30-40% in weeks. Max drawdown measured
here is -22% and this sample contains no such episode.
"""
from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from ..data.etf_universe import all_etfs, is_index, yf_symbol
from ..data.loader import load_params, load_prices

LOOKBACK = 252          # 12 months
SKIP = 21               # ...skipping the most recent month: standard, avoids the
                        # short-term reversal that contaminates a raw 12m signal
TOP_PCT = 0.90          # top decile
MIN_BARS = LOOKBACK + SKIP + 5


def _closes(params=None, max_age_hours=None, verbose=False):
    params = params or load_params()
    names = [t for t in all_etfs() if not is_index(t)]
    feed = {t: yf_symbol(t) for t in names}
    px = load_prices(sorted(set(feed.values())), params=params, verbose=verbose,
                     max_age_hours=max_age_hours)
    out = {}
    for t in names:
        d = px.get(feed[t])
        if d is None or "Close" not in d:
            continue
        c = d["Close"].dropna()
        if len(c) >= MIN_BARS:
            out[t] = c
    return pd.DataFrame(out).sort_index()


def scores(close: pd.DataFrame) -> pd.Series:
    """12-month return skipping the last month, as of the final bar."""
    if len(close) < MIN_BARS:
        return pd.Series(dtype=float)
    a = close.shift(SKIP).iloc[-1]
    b = close.shift(LOOKBACK).iloc[-1]
    return (a / b - 1.0).dropna()


def is_rebalance_day(day, calendar) -> bool:
    """True on the last trading day of a month, which is when this rebalances."""
    day = pd.Timestamp(day)
    same_month = [d for d in calendar if d.year == day.year and d.month == day.month]
    return bool(same_month) and day == max(same_month)


def book(asof=None, top_pct=TOP_PCT, params=None, max_age_hours=None, verbose=False):
    """The names this strategy holds, equal weight, with their momentum scores."""
    close = _closes(params, max_age_hours, verbose)
    if asof is not None:
        close = close.loc[:pd.Timestamp(asof)]
    s = scores(close)
    if s.empty:
        return pd.DataFrame(columns=["ticker", "score", "weight"]), None
    cut = s.rank(pct=True) > top_pct
    picks = s[cut].sort_values(ascending=False)
    w = 100.0 / len(picks) if len(picks) else 0.0
    df = pd.DataFrame({"ticker": picks.index, "score": 100 * picks.values,
                       "weight": w})
    return df.reset_index(drop=True), close.index[-1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asof", default=None)
    ap.add_argument("--top-pct", type=float, default=TOP_PCT)
    ap.add_argument("--fresh", action="store_true", help="bypass the price cache")
    a = ap.parse_args(argv)
    df, asof = book(a.asof, a.top_pct, max_age_hours=0 if a.fresh else None,
                    verbose=True)
    if df.empty:
        print("[momentum] no usable history")
        return 1
    cal = _closes().index
    due = is_rebalance_day(asof, cal)
    print("\nmomentum book as of %s  (%d names, %.2f%% each)"
          % (asof.date(), len(df), df["weight"].iloc[0]))
    print("rebalance day: %s\n" % ("YES - last trading day of the month" if due
                                   else "no - holds until month end"))
    print("%-8s %10s" % ("ticker", "12-1 mom%"))
    for r in df.itertuples():
        print("%-8s %+9.1f%%" % (r.ticker, r.score))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
