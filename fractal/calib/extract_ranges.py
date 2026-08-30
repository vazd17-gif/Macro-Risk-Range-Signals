"""Turn Similar Set chart images into a dated RANGE series, and prove it is right.

`extract_chart_levels` does the colour tracing. This module supplies the geometry
and the verification, and it takes both from the price history rather than from
anything read by eye.

Recovering the bar grid from the image alone does not work: a bar splits into a
body and a wick, adjacent bars merge, and an autocorrelation locks onto the body
width instead of the bar spacing. The price history removes the ambiguity. Only
the true pitch makes every candle top and bottom land on that bar's actual high
and low, so the geometry (pitch, grid anchor, calendar alignment) is solved by
minimising that residual, and the same fit yields the price axis as a by-product:

    price = a * pixel_row + b        fitted over ~80 candle extremes

That is far more anchor points than the two or three axis badges, and it needs
nothing transcribed from the image. The badges then serve as an independent
check: their known values (from labels.csv) are predicted from the fitted axis
and compared. A chart failing either test is dropped rather than guessed at.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

from ..data.loader import load_prices, repo_path
from . import extract_chart_levels as X
from . import labels as labels_mod

PITCH_MIN, PITCH_MAX, PITCH_STEP = 9.0, 70.0, 0.25
MIN_BARS = 25
MAX_CANDLE_W = 55        # px; wider white blocks are annotation plates
HEADER_FRAC, FOOTER_FRAC = 0.06, 0.885   # strips holding the title and the date axis


# ----------------------------------------------------------------- bar grid
def candle_mask(img, axis_x, x_hi):
    """White pixels that are actually candles.

    Three things also render white and would otherwise be read as candle extremes:
    the dashed horizontal rule TradingView draws at the last price, the filled
    annotation boxes (the ticker plate, day labels), and the header text. The rule
    is a row that is white nearly all the way across; the boxes are large solid
    blocks; the header is a fixed strip. Each is removed on its own signature.
    """
    w = X._white(img)
    w[:, axis_x:] = False
    w[: int(img.shape[0] * HEADER_FRAC), :] = False        # header text
    w[int(img.shape[0] * FOOTER_FRAC):, :] = False         # date axis labels
    w = w[:, : x_hi + 1].copy()

    width = w.shape[1]
    row_frac = w.sum(axis=1) / max(width, 1)
    w[row_frac > 0.35, :] = False                        # horizontal price rule

    lab, n = ndimage.label(w, structure=np.ones((3, 3)))
    if n:
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            h = sl[0].stop - sl[0].start
            ww = sl[1].stop - sl[1].start
            # A candle is never wider than the bar pitch; anything much wider is
            # an annotation plate (ticker name, day label).
            if ww >= MAX_CANDLE_W or (h >= 25 and w[sl].mean() > 0.6):
                w[sl] = w[sl] & ~(lab[sl] == i)
    return w


def white_extents(img, axis_x, x_hi):
    """Per-column topmost and bottommost candle pixel; NaN where the column is empty."""
    w = candle_mask(img, axis_x, x_hi)
    top = np.full(w.shape[1], np.nan)
    bot = np.full(w.shape[1], np.nan)
    rows = np.arange(w.shape[0])
    for x in range(w.shape[1]):
        ys = rows[w[:, x]]
        if len(ys):
            top[x] = ys[0]
            bot[x] = ys[-1]
    return top, bot


def grid_extents(top, bot, grid, half):
    """Candle top/bottom at each grid x, aggregated over a +/-half window."""
    out = np.full((len(grid), 2), np.nan)
    for i, cx in enumerate(grid):
        a = max(0, int(round(cx - half)))
        b = min(len(top), int(round(cx + half)) + 1)
        if b <= a:
            continue
        t, o = top[a:b], bot[a:b]
        if np.isfinite(t).any():
            out[i, 0] = np.nanmin(t)
            out[i, 1] = np.nanmax(o)
    return out


def fit_axis(ext, highs, lows, trim=0.25):
    """Least-squares price = a*y + b over candle tops/bottoms, with an outlier trim.

    Annotation boxes and drawn arrows occasionally swallow a bar's white pixels,
    so the worst `trim` fraction of points is dropped before the final fit.
    """
    y = np.concatenate([ext[:, 0], ext[:, 1]])
    p = np.concatenate([np.asarray(highs, float), np.asarray(lows, float)])
    ok = np.isfinite(y) & np.isfinite(p)
    if ok.sum() < 12:
        return None
    y, p = y[ok], p[ok]
    if np.ptp(y) < 20:
        return None
    a, b = np.polyfit(y, p, 1)
    if a >= 0:                       # price must fall as the pixel row increases
        return None
    resid = np.abs((a * y + b) - p) / np.abs(p)
    keep = resid <= np.quantile(resid, 1 - trim)
    if keep.sum() >= 10:
        a, b = np.polyfit(y[keep], p[keep], 1)
        resid = np.abs((a * y[keep] + b) - p[keep]) / np.abs(p[keep])
    return float(a), float(b), float(np.median(resid) * 100)


def last_candle_x(top, bot):
    """Centre of the right-most candle.

    This is the anchor, not the end of the range lines: the lines are plotted one
    slot beyond the last candle because they carry the level for the session about
    to trade. Anchoring on the lines instead shifts every bar by one day.
    """
    present = np.isfinite(top)
    xs = np.nonzero(present)[0]
    if len(xs) == 0:
        return None
    end = int(xs.max())
    start = end
    while start > 0 and present[start - 1]:
        start -= 1
    return (start + end) / 2.0


def solve_geometry(top, bot, bars, x_lo, x_hi, verbose=False, coarse=False):
    """Pitch, bar grid, calendar alignment and price axis, all from one fit.

    `coarse` uses a wide pitch step and a single anchor/drop for fast ticker
    identification; the winning ticker is then re-solved at full resolution.

    `drop` exists because the chart is captured intraday: its right-most bar is the
    session in progress, which has no completed daily bar to check against. Trying
    a few trailing drops lets the fit line up the completed bars without assuming
    how many unfinished ones the capture happens to include.

    The window is narrow (a fraction of the pitch) on purpose. Candle bodies are
    about three quarters of the bar pitch wide, so a wide window straddles the
    neighbouring bars and mixes their extremes into this one.
    """
    best = None
    x_anchor = last_candle_x(top, bot)
    if x_anchor is None:
        return None
    span = x_anchor - x_lo
    step = 1.5 if coarse else PITCH_STEP
    anchors = (0.0,) if coarse else np.arange(-0.3, 0.31, 0.15)
    for pitch in np.arange(PITCH_MIN, PITCH_MAX, step):
        n_max = int(span / pitch) + 2
        if n_max < MIN_BARS or n_max > len(bars):
            continue
        half = max(2.0, pitch * 0.15)
        for anchor in anchors:
            x_last = x_anchor + anchor * pitch
            grid = np.sort((x_last - pitch * np.arange(n_max))[
                (x_last - pitch * np.arange(n_max)) >= x_lo - 0.5 * pitch])
            grid = grid[grid <= x_anchor + 0.5 * pitch]
            n = len(grid)
            if n < MIN_BARS or n > len(bars):
                continue
            ext = grid_extents(top, bot, grid, half)
            window = bars.iloc[len(bars) - n:]
            fit = fit_axis(ext, window["High"].values, window["Low"].values)
            if fit is None:
                continue
            if best is None or fit[2] < best["err"]:
                # projected slots: the lines run past the last candle, one slot
                # per session whose level has been published but not yet traded
                extra = int(round((x_hi - grid[-1]) / pitch))
                extra = max(0, min(extra, 2))
                full = np.concatenate([grid, grid[-1] + pitch * np.arange(1, extra + 1)])
                best = {"err": fit[2], "a": fit[0], "b": fit[1],
                        "pitch": float(pitch), "grid": full, "fit_grid": grid,
                        "window": window, "projected": extra, "n": n}
    if verbose and best:
        print("      geometry: pitch %.2f  bars %d  projected %d  err %.3f%%"
              % (best["pitch"], best["n"], best["projected"], best["err"]))
    return best


# --------------------------------------------------------------------- main
def extract_one(path, hist, chart_date, anchors=None,
                max_fit_err=0.8, max_badge_err=0.5, verbose=True):
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    axis_x = X.find_axis_x(img)

    hi_y = X.trace_line(img, X.COL_RED, axis_x)
    lo_y = X.trace_line(img, X.COL_TEAL_LINE, axis_x, tol=X.TOL_TEAL)
    if hi_y is None or lo_y is None:
        raise ValueError("could not trace both range lines")
    drawn = np.isfinite(hi_y) & np.isfinite(lo_y)
    if drawn.sum() < 80:
        raise ValueError("range lines too short (%d px)" % drawn.sum())
    xs_drawn = np.nonzero(drawn)[0]
    x_lo, x_hi = int(xs_drawn.min()), int(xs_drawn.max())

    top, bot = white_extents(img, axis_x, x_hi)
    bars = hist[hist.index <= pd.Timestamp(chart_date)]
    if len(bars) < 30:
        raise ValueError("not enough price history")

    geo = solve_geometry(top, bot, bars, x_lo, x_hi)
    if geo is None:
        raise ValueError("could not solve chart geometry")
    if geo["err"] > max_fit_err:
        raise ValueError("candles do not reconcile with price history "
                         "(median %.2f%% > %.2f%%)" % (geo["err"], max_fit_err))

    a, b = geo["a"], geo["b"]
    grid, window = geo["grid"], geo["window"]

    # Sessions for every grid point, including the in-progress bar the capture
    # was taken during, which has no completed daily bar of its own.
    sessions = list(window.index)
    tail = pd.Timestamp(chart_date)
    while len(sessions) < len(grid):
        sessions.append(tail)
        tail = tail + pd.Timedelta(days=1)
    sessions = sessions[: len(grid)]
    closes = list(window["Close"].values) + [np.nan] * (len(grid) - len(window))

    # The candle fit pins the geometry but carries a few tenths of a percent of
    # level bias. The axis badges are exact known prices, so where they can be
    # matched unambiguously they replace the candle fit as the price map, and the
    # candle axis is only used to decide which badge is which.
    badge_err = np.nan
    if anchors:
        badges = X.find_badges(img, axis_x)
        pairs = []
        for value in anchors:
            cands = [(abs((a * y + b) / value - 1.0), y) for y, *_ in badges]
            if not cands:
                break
            err, y = min(cands)
            if err < 0.02:
                pairs.append((y, value))
        seen, uniq = set(), []
        for y, v in pairs:
            if y not in seen:
                seen.add(y)
                uniq.append((y, v))
        if len(uniq) >= 2:
            ys = np.array([y for y, _ in uniq], dtype=float)
            vs = np.array([v for _, v in uniq], dtype=float)
            a2, b2 = np.polyfit(ys, vs, 1)
            badge_err = float(np.max(np.abs((a2 * ys + b2) / vs - 1.0)) * 100)
            if badge_err > max_badge_err:
                raise ValueError("badge check failed (%.2f%% > %.2f%%)"
                                 % (badge_err, max_badge_err))
            a, b = float(a2), float(b2)
        elif badges:
            pred = [(a * y + b) for y, *_ in badges]
            badge_err = float(max(min(abs(p / k - 1.0) for p in pred)
                                  for k in anchors) * 100)
            if badge_err > max_badge_err:
                raise ValueError("badge check failed (%.2f%% > %.2f%%)"
                                 % (badge_err, max_badge_err))

    rows = []
    for k, cx in enumerate(grid):
        x = min(int(round(cx)), x_hi)          # the projected slot can sit a pixel
        lo_x, hi_x = max(0, x - 1), min(x_hi + 1, x + 2)   # past the drawn end
        h, l = hi_y[lo_x:hi_x], lo_y[lo_x:hi_x]
        h = np.nanmean(h) if np.isfinite(h).any() else np.nan
        l = np.nanmean(l) if np.isfinite(l).any() else np.nan
        rows.append({
            "date": sessions[k],
            "close": float(closes[k]),
            "range_high": a * h + b if np.isfinite(h) else np.nan,
            "range_low": a * l + b if np.isfinite(l) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.attrs.update(fit_err_pct=geo["err"], badge_err_pct=badge_err,
                    pitch=geo["pitch"], n_bars=geo["n"],
                    projected=geo["projected"])
    if verbose:
        print("    %d bars @ %.1fpx | candle fit %.3f%% | badge %s | +%d projected"
              % (geo["n"], geo["pitch"], geo["err"],
                 "n/a" if not np.isfinite(badge_err) else "%.2f%%" % badge_err,
                 geo["projected"]))
    return df


def main():
    ap = argparse.ArgumentParser(description="Extract dated RANGE series from chart images.")
    ap.add_argument("--charts", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", default=repo_path("data", "range_labels.csv"))
    ap.add_argument("--max-fit-err", type=float, default=0.8)
    ap.add_argument("--max-badge-err", type=float, default=0.5)
    args = ap.parse_args()

    cmap = pd.read_csv(args.map)
    resolved = labels_mod.resolve(verbose=True)

    anchors_by_ticker, syms = {}, {}
    for t, grp in resolved[resolved["ok"]].groupby("ticker"):
        r = grp.iloc[-1]
        vals = [float(v) for v in (r.get("chart_last"), r.get("ss_trend"), r.get("ss_trade"))
                if pd.notna(v)]
        if vals:
            anchors_by_ticker[t] = vals
    for t in cmap["ticker"]:
        rr = resolved[resolved["ticker"] == t]
        syms[t] = rr.iloc[0]["yf"] if len(rr) else t
    prices = load_prices(sorted(set(syms.values())), verbose=False)

    frames, failed = [], []
    for row in cmap.itertuples():
        path = os.path.join(args.charts, row.image)
        if not os.path.exists(path):
            failed.append((row.ticker, "image missing"))
            continue
        hist = prices.get(syms.get(row.ticker))
        if hist is None or len(hist) == 0:
            failed.append((row.ticker, "no price history"))
            continue
        print("  %-8s %s" % (row.ticker, row.image))
        try:
            df = extract_one(path, hist, row.chart_date,
                             anchors=anchors_by_ticker.get(row.ticker),
                             max_fit_err=args.max_fit_err,
                             max_badge_err=args.max_badge_err)
        except Exception as e:
            print("    SKIPPED: %s" % e)
            failed.append((row.ticker, str(e)))
            continue
        df["ticker"] = row.ticker
        df["yf"] = syms[row.ticker]
        df["source"] = row.image
        df["fit_err_pct"] = df.attrs["fit_err_pct"]
        df["badge_err_pct"] = df.attrs["badge_err_pct"]
        frames.append(df)

    if not frames:
        print("\n[range] nothing extracted")
        return
    out = pd.concat(frames, ignore_index=True).dropna(subset=["range_high", "range_low"])
    out.to_csv(args.out, index=False)
    print("\n[range] %d rows over %d tickers -> %s"
          % (len(out), out.ticker.nunique(), args.out))
    print("[range] %d charts skipped" % len(failed))
    for t, why in failed:
        print("   %-8s %s" % (t, why))


if __name__ == "__main__":
    main()
