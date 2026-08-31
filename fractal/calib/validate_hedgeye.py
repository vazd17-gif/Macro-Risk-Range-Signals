"""Validate the model against Hedgeye's *own* published Risk Ranges.

The build spec fitted everything to Similar Set, which is itself a reconstruction
of Hedgeye. Hedgeye publishes its actual numbers in plain text — the ETF Pro Plus
weekly report, the daily ETF Pro change notes, and the "Our Levels" block at the
bottom of every Early Look — so those are the real target, not a proxy for it.

This is a genuine out-of-sample test: none of these levels were used to fit any
parameter. For each row it computes the model's RANGE as of the prior close and
compares low, high and the bull/bear TREND direction against Hedgeye.

Reference CSVs (harvested from the user's email):
  reference/hedgeye_ranges.csv       ETF Pro Plus weekly report, 35 ETFs @ 8/21
  reference/hedgeye_early_look.csv    Early Look "Our Levels", macro @ 8/27
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices, repo_path
from ..model import adaptive_ma, range_ewma


def _prior_index(close, date):
    idx = close.index
    pos = idx.searchsorted(pd.Timestamp(date), side="right") - 1
    return pos if pos >= 30 else None


def validate(csv_path, params=None, verbose=True, price_tol=1.0):
    params = params or load_params()
    df = pd.read_csv(csv_path)
    if "yf" not in df:
        df["yf"] = df["ticker"]

    prices = load_prices(sorted(df["yf"].unique()), params=params, verbose=verbose)

    rows = []
    for r in df.itertuples():
        px = prices.get(r.yf)
        rec = {"ticker": r.ticker, "yf": r.yf, "dir": r.trend_dir}
        if px is None or len(px) < 60:
            rec["note"] = "no data"
            rows.append(rec)
            continue
        close = px["Close"].dropna()
        pos = _prior_index(close, r.prior_close_date)
        if pos is None:
            rec["note"] = "no prior bar"
            rows.append(rec)
            continue

        asof = close.index[pos]
        # price sanity: the report's recent price should match the bar we anchor on
        if hasattr(r, "recent_price") and pd.notna(getattr(r, "recent_price", np.nan)):
            rec["px_check_pct"] = 100 * (float(close.iloc[pos]) / float(r.recent_price) - 1)

        rng = range_ewma.compute(close, params, volume=px.get("Volume"))
        lines = adaptive_ma.compute(close, params)
        rv = rng.iloc[pos]
        lo, hi = float(rv["range_low"]), float(rv["range_high"])
        rec["model_low"], rec["model_high"] = lo, hi
        rec["hedge_low"], rec["hedge_high"] = float(r.rr_low), float(r.rr_high)
        rec["low_err_pct"] = 100 * (lo / r.rr_low - 1)
        rec["high_err_pct"] = 100 * (hi / r.rr_high - 1)
        rec["width_model"] = 100 * (hi / lo - 1)
        rec["width_hedge"] = 100 * (r.rr_high / r.rr_low - 1)

        # TREND direction: model says bull if prior close > TREND line
        trend = lines["trend"].iloc[pos]
        c = float(close.iloc[pos])
        model_dir = "bull" if c > trend else "bear"
        rec["model_dir"] = model_dir
        rec["dir_match"] = bool(r.trend_dir == model_dir) if r.trend_dir in ("bull", "bear") else None
        rows.append(rec)

    return pd.DataFrame(rows)


def report(res, title):
    ok = res.dropna(subset=["low_err_pct", "high_err_pct"]).copy()
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    if ok.empty:
        miss = ", ".join("%s(%s)" % (r.ticker, r.get("note", "?")) for r in res.itertuples())
        print("  no comparable rows. " + miss)
        return

    cols = ["ticker", "hedge_low", "model_low", "low_err_pct",
            "hedge_high", "model_high", "high_err_pct",
            "width_hedge", "width_model", "dir", "model_dir"]
    show = ok[cols].sort_values("high_err_pct", key=lambda s: s.abs(), ascending=False)
    print(show.to_string(index=False, float_format=lambda v: "%.2f" % v))

    le = ok["low_err_pct"].abs()
    he = ok["high_err_pct"].abs()
    both = pd.concat([le, he])
    print("\n  low edge : median |err| %.2f%%   mean %.2f%%" % (le.median(), le.mean()))
    print("  high edge: median |err| %.2f%%   mean %.2f%%" % (he.median(), he.mean()))
    print("  both     : median |err| %.2f%%   within 2%%: %d/%d   within 5%%: %d/%d"
          % (both.median(), (both <= 2).sum(), len(both), (both <= 5).sum(), len(both)))
    print("  width    : model median %.2f%%   hedgeye median %.2f%%   ratio %.2fx"
          % (ok["width_model"].median(), ok["width_hedge"].median(),
             ok["width_model"].median() / ok["width_hedge"].median()))

    dm = ok.dropna(subset=["dir_match"])
    if len(dm):
        print("  TREND direction match: %d/%d (%.0f%%)"
              % (dm["dir_match"].sum(), len(dm), 100 * dm["dir_match"].mean()))
        miss = dm[dm["dir_match"] == False]["ticker"].tolist()
        if miss:
            print("    mismatches: " + ", ".join(miss))

    if "px_check_pct" in ok:
        bad = ok[ok["px_check_pct"].abs() > 1.5]
        if len(bad):
            print("  NOTE: price mismatch (>1.5%%) on: "
                  + ", ".join("%s(%.1f%%)" % (b.ticker, b.px_check_pct) for b in bad.itertuples()))


def main():
    ap = argparse.ArgumentParser(description="Validate the model vs Hedgeye's published risk ranges.")
    ap.add_argument("--set", choices=["weekly", "early", "both"], default="both")
    ap.add_argument("--csv", default=None, help="override with a custom reference CSV")
    # Default to the profile the live report actually runs, not the params file's
    # own default -- validating a configuration nothing ships is worthless.
    ap.add_argument("--profile", default="hedgeye_anchor",
                    help="RANGE profile to validate (default: the one in production)")
    args = ap.parse_args()

    params = load_params()
    if args.profile:
        params["range"] = dict(params["range"])
        params["range"]["active"] = args.profile
        print("RANGE profile under test: %s\n" % args.profile)
    if args.csv:
        report(validate(args.csv, params), os.path.basename(args.csv))
        return
    if args.set in ("weekly", "both"):
        report(validate(repo_path("reference", "hedgeye_ranges.csv"), params),
               "ETF Pro Plus weekly report  (35 ETFs, ranges as of Fri 2026-08-21)")
    if args.set in ("early", "both"):
        report(validate(repo_path("reference", "hedgeye_early_look.csv"), params),
               "Early Look 'Our Levels'  (macro, ranges as of 2026-08-27)")
        wk = repo_path("reference", "hedgeye_early_look_week.csv")
        if os.path.exists(wk):
            report(validate(wk, params), "Early Look 'Our Levels'  (a full week of issues)")


if __name__ == "__main__":
    main()
