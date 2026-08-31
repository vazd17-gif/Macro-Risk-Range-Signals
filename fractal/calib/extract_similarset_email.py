"""Out-of-sample Similar Set check: extract a daily email's charts and validate.

The model was fitted to the 2026-08-28 Similar Set charts. The daily newsletter
emails for the other days carry the same charts (external Kit CDN images), so they
are a true out-of-sample test. This module:

1. Auto-identifies each chart's ticker. The charts carry no machine-readable label,
   but the geometry solver in extract_ranges reconciles candle pixels against a
   ticker's real OHLC and reports a fit error. Running every candidate ticker and
   keeping the one that reconciles below threshold identifies the chart with no
   manual mapping — and the same fit that identifies it also proves the date.
2. Extracts the RANGE (traced red/green lines) and the SS:TRADE / SS:TREND axis
   badges for the identified ticker.
3. Compares the extracted levels against the model computed as of the prior close.

Candidate tickers come from the newsletter's own text (it names every ticker it
charts), mapped to Yahoo symbols.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
from PIL import Image

from ..data.loader import load_params, load_prices
from ..model import adaptive_ma, range_ewma
from . import extract_chart_levels as X
from . import extract_ranges as R

# Similar Set names -> Yahoo symbols for the tickers seen across the daily emails.
SS_YF = {
    "VIX": "^VIX", "MOVE": "^MOVE", "HYG": "HYG", "SPY": "SPY", "QQQ": "QQQ",
    "IWM": "IWM", "MAGS": "MAGS", "IGV": "IGV", "NOW": "NOW", "MSFT": "MSFT",
    "PLTR": "PLTR", "SNOW": "SNOW", "AMZN": "AMZN", "XLY": "XLY", "V": "V",
    "XLV": "XLV", "QTUM": "QTUM", "SMH": "SMH", "NVDA": "NVDA", "DXY": "DX-Y.NYB",
    "UUP": "UUP", "TLT": "TLT", "USO": "USO", "COM": "COM", "HECA": "HECA",
    "GLD": "GLD", "SLV": "SLV", "PALL": "PALL", "BTC": "BTC-USD", "XRP": "XRP-USD",
    "MSTR": "MSTR", "COLO": "COLO", "XLC": "XLC", "XLF": "XLF", "SPX": "^GSPC",
}


def shape_score(top, bot, geo):
    """Correlation of the candle *pattern* (not its level) with the ticker's OHLC.

    The % fit error is not enough to identify a chart: a low-volatility instrument
    reconciles anything because a near-flat axis maps every candle to ~one price
    with tiny percent error. What actually distinguishes tickers is whether the
    sequence of highs and lows matches. This correlates the candle-implied high/low
    against the real High/Low over the aligned window; only the true ticker tracks.
    """
    a, b = geo["a"], geo["b"]
    grid, window = geo["fit_grid"], geo["window"]
    ext = R.grid_extents(top, bot, grid, max(2.0, geo["pitch"] * 0.15))
    ih = a * ext[:, 0] + b
    il = a * ext[:, 1] + b
    ah = window["High"].values
    al = window["Low"].values
    ok = np.isfinite(ih) & np.isfinite(il)
    if ok.sum() < 12:
        return -1.0
    def corr(x, y):
        x, y = x[ok], y[ok]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    return 0.5 * (corr(ih, ah) + corr(il, al))


def identify(path, prices, chart_date, max_err=0.6, min_shape=0.85):
    """Return the candidate whose candle *pattern* matches, not just its level."""
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    axis_x = X.find_axis_x(img)
    hi_y = X.trace_line(img, X.COL_RED, axis_x)
    lo_y = X.trace_line(img, X.COL_TEAL_LINE, axis_x, tol=X.TOL_TEAL)
    if hi_y is None or lo_y is None:
        return None
    drawn = np.isfinite(hi_y) & np.isfinite(lo_y)
    if drawn.sum() < 80:
        return None
    xs = np.nonzero(drawn)[0]
    x_lo, x_hi = int(xs.min()), int(xs.max())
    top, bot = R.white_extents(img, axis_x, x_hi)

    best = None            # (shape, err, tk, geo)
    for tk, px in prices.items():
        if px is None or len(px) < 120:
            continue
        bars = px[px.index <= pd.Timestamp(chart_date)]
        if len(bars) < 80:
            continue
        try:
            geo = R.solve_geometry(top, bot, bars, x_lo, x_hi, coarse=True)
        except Exception:
            geo = None
        if geo is None or geo["err"] > max_err:
            continue
        sh = shape_score(top, bot, geo)
        if best is None or sh > best[0]:
            best = (sh, geo["err"], tk, geo)
    if best is None or best[0] < min_shape:
        return None

    tk, geo = best[2], best[3]
    bars = prices[tk][prices[tk].index <= pd.Timestamp(chart_date)]
    fine = R.solve_geometry(top, bot, bars, x_lo, x_hi, coarse=False)
    use = fine if (fine is not None and shape_score(top, bot, fine) >= best[0] - 0.05) else geo
    return (tk, use, img, axis_x, hi_y, lo_y, x_hi)


def extract_levels(bundle, chart_date):
    tk, geo, img, axis_x, hi_y, lo_y, x_hi = bundle
    a, b = geo["a"], geo["b"]
    grid = geo["grid"]

    # RANGE = last drawn slot (the projected level for the session about to trade)
    x = min(int(round(grid[-1])), x_hi)
    hh = np.nanmean(hi_y[max(0, x - 1):x + 2])
    ll = np.nanmean(lo_y[max(0, x - 1):x + 2])
    rng = {"range_high": a * hh + b, "range_low": a * ll + b} if np.isfinite(hh) else None

    # SS badges -> candidate line levels (teal = TRADE/TREND). Refine axis on the
    # white "last price" badge if present, but the candle axis is already good.
    badges = X.find_badges(img, axis_x)
    ss_vals = sorted(a * y + b for (y, kind, *_ ) in badges if kind == "ss")
    return {"ticker": tk, "fit_err": geo["err"], "rng": rng, "ss_vals": ss_vals,
            "pitch": geo["pitch"]}


def run(charts_dir, chart_date, prior_close_date, candidates, params=None, verbose=True):
    params = params or load_params()
    yfs = sorted({SS_YF.get(t, t) for t in candidates})
    prices = load_prices(yfs, params=params, verbose=False)
    pmap = {y: prices.get(y) for y in yfs}

    rows = []
    used = set()
    for path in sorted(glob.glob(os.path.join(charts_dir, "*.png"))):
        im = Image.open(path)
        if im.size != (2400, 1546):
            continue
        bundle = identify(path, pmap, chart_date)
        if bundle is None:
            continue
        lv = extract_levels(bundle, chart_date)
        tk = lv["ticker"]
        if tk in used:            # keep the better-fitting chart if a dup id
            prev = next(r for r in rows if r["yf"] == tk)
            if lv["fit_err"] >= prev["fit_err"]:
                continue
            rows.remove(prev)
        used.add(tk)

        px = pmap[tk]
        close = px["Close"].dropna()
        pos = close.index.searchsorted(pd.Timestamp(prior_close_date), side="right") - 1
        if pos < 40:
            continue
        rng = range_ewma.compute(close, params, volume=px.get("Volume"))
        lines = adaptive_ma.compute(close, params)
        mr = rng.iloc[pos]
        row = {"yf": tk, "fit_err": lv["fit_err"], "asof": close.index[pos].date().isoformat(),
               "ss_range_low": lv["rng"]["range_low"] if lv["rng"] else np.nan,
               "ss_range_high": lv["rng"]["range_high"] if lv["rng"] else np.nan,
               "model_range_low": float(mr["range_low"]), "model_range_high": float(mr["range_high"]),
               "model_trade": float(lines["trade"].iloc[pos]),
               "model_trend": float(lines["trend"].iloc[pos]),
               "ss_badges": ", ".join("%.2f" % v for v in lv["ss_vals"])}
        if lv["rng"]:
            row["rlow_err"] = 100 * (row["model_range_low"] / row["ss_range_low"] - 1)
            row["rhigh_err"] = 100 * (row["model_range_high"] / row["ss_range_high"] - 1)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Validate model vs a Similar Set daily email (out of sample).")
    ap.add_argument("--charts", required=True)
    ap.add_argument("--chart-date", required=True, help="the email's date (charts drawn through the prior bar)")
    ap.add_argument("--prior-close", required=True)
    ap.add_argument("--candidates", required=True, help="comma-separated Similar Set tickers named in the email")
    args = ap.parse_args()

    cands = [t.strip().upper() for t in args.candidates.split(",")]
    df = run(args.charts, args.chart_date, args.prior_close, cands)
    if df.empty:
        print("no charts identified")
        return
    cols = ["yf", "asof", "fit_err", "ss_range_low", "model_range_low", "rlow_err",
            "ss_range_high", "model_range_high", "rhigh_err", "ss_badges",
            "model_trade", "model_trend"]
    cols = [c for c in cols if c in df]
    print(df[cols].to_string(index=False, float_format=lambda v: "%.2f" % v))
    if "rhigh_err" in df:
        e = pd.concat([df["rlow_err"].abs(), df["rhigh_err"].abs()]).dropna()
        print("\n  RANGE edge vs Similar Set (out of sample): median |err| %.2f%%   within 2%%: %d/%d"
              % (e.median(), (e <= 2).sum(), len(e)))
        print("  identified %d charts, all candle-fit < %.2f%%" % (len(df), df["fit_err"].max()))


if __name__ == "__main__":
    main()
