"""Out-of-sample behaviour of the fitted RANGE (build spec section 5, backtests).

Two questions, kept separate from the label fit:

  coverage   what fraction of next-day closes land inside the envelope. This is
             the design point of the model, not a free parameter: a range fitted
             to reproduce published levels has no reason to hit any particular
             coverage, so measuring it says whether the reconstruction is a
             sensible risk band or only a curve-fit to someone else's numbers.

  breach     what happens after a touch. If the range is a mean-reversion tool,
             closes beyond an edge should be followed by reversion, and the sign
             of the forward return conditional on a breach is the test.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices
from ..data.universe import all_tickers
from ..model import range_ewma


def coverage_for(close, params, horizon=1):
    rng = range_ewma.compute(close, params)
    lo = rng["range_low"].shift(1)
    hi = rng["range_high"].shift(1)
    c = close
    ok = lo.notna() & hi.notna()
    inside = ((c >= lo) & (c <= hi))[ok]

    fwd = close.shift(-horizon) / close - 1.0
    above = (c > hi)[ok]
    below = (c < lo)[ok]
    return {
        "n": int(ok.sum()),
        "coverage": float(inside.mean()) if ok.sum() else np.nan,
        "breach_up_rate": float(above.mean()) if ok.sum() else np.nan,
        "breach_dn_rate": float(below.mean()) if ok.sum() else np.nan,
        "fwd_after_up": float(fwd[ok][above].mean()) if above.sum() else np.nan,
        "fwd_after_dn": float(fwd[ok][below].mean()) if below.sum() else np.nan,
        "fwd_uncond": float(fwd[ok].mean()) if ok.sum() else np.nan,
        "median_width_pct": float(((hi / lo - 1) * 100)[ok].median()) if ok.sum() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description="RANGE coverage and breach-reversion backtest.")
    ap.add_argument("--tickers", default=None, help="comma-separated; default = universe")
    ap.add_argument("--horizon", type=int, default=5, help="forward return horizon in days")
    ap.add_argument("--years", type=int, default=3)
    args = ap.parse_args()

    params = load_params()
    tickers = ([t.strip() for t in args.tickers.split(",")] if args.tickers
               else all_tickers())
    prices = load_prices(tickers, params=params)

    rows = []
    for t in tickers:
        df = prices.get(t)
        if df is None or len(df) < 300:
            continue
        close = df["Close"].dropna().iloc[-int(252 * args.years):]
        r = coverage_for(close, params, horizon=args.horizon)
        r["ticker"] = t
        rows.append(r)

    out = pd.DataFrame(rows).set_index("ticker")
    if out.empty:
        print("no data")
        return

    show = out[["n", "coverage", "median_width_pct", "breach_up_rate", "breach_dn_rate",
                "fwd_after_up", "fwd_after_dn", "fwd_uncond"]]
    print(show.to_string(float_format=lambda v: "%.4f" % v))
    print("\n=== aggregate over %d names, %dy, %dd forward ===" % (len(out), args.years, args.horizon))
    print("  coverage            %.1f%%  (design point for a 1-sigma band is ~68%%,"
          " for the build spec's m=1 close-basis reading ~50%%)" % (100 * out.coverage.mean()))
    print("  median range width  %.2f%%" % out.median_width_pct.median())
    print("  breaches            up %.1f%%   down %.1f%%"
          % (100 * out.breach_up_rate.mean(), 100 * out.breach_dn_rate.mean()))
    print("  %dd forward return   after up-breach %+.2f%%   after down-breach %+.2f%%"
          "   unconditional %+.2f%%"
          % (args.horizon, 100 * out.fwd_after_up.mean(),
             100 * out.fwd_after_dn.mean(), 100 * out.fwd_uncond.mean()))
    up_edge = out.fwd_after_up.mean() - out.fwd_uncond.mean()
    dn_edge = out.fwd_after_dn.mean() - out.fwd_uncond.mean()
    print("  excess vs unconditional: after up %+.2f%%   after down %+.2f%%"
          % (100 * up_edge, 100 * dn_edge))
    print("  mean reversion at the edges is %s"
          % ("present in both directions" if (up_edge < 0 and dn_edge > 0) else
             "not present in both directions"))


if __name__ == "__main__":
    main()
