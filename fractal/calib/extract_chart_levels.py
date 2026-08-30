"""Read RANGE levels straight off published Similar Set chart images.

Why this exists: the newsletter prints the TRADE and TREND values as axis badges,
so those become labels by reading two numbers per chart. The RANGE has no badge
-- it is only drawn as the red (high) and green (low) lines -- and a single quoted
value ("SLV top end $64.30") is not enough to fit a range model. Tracing the two
lines turns one chart into ~40 daily RANGE observations.

How it works:

1. Axis calibration. The right-hand price scale carries filled badges whose values
   are already known and verified in labels.csv (the last price, SS:TRADE,
   SS:TREND). Their pixel centres plus their known prices give a least-squares
   linear map price(y). Two anchors are enough; three let us report a residual,
   which is the check that the axis really is linear.

2. Line tracing. The range lines are the only colour components that span most of
   the plot width -- the dotted TRADE/TREND series break into many small blobs and
   the annotation arrows are short. Filtering connected components by x-extent
   separates them cleanly without needing to model the dots at all.

3. Dates. Candle columns are detected from the white candle pixels, giving the bar
   pitch. The rightmost bar is the chart date; earlier bars step back through the
   trading calendar of the price history.

Everything it emits is checkable: `--verify` compares the traced value on the last
bar against a known quoted level (SLV 2026-08-28 range top = 64.30).
"""
from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

# TradingView palette used by the Similar Set charts.
COL_RED = (255, 82, 82)          # RANGE high, and bearish dots
COL_TEAL_LINE = (8, 153, 129)    # RANGE low
COL_TEAL_MARK = (0, 150, 136)    # badges, dots, annotation arrows
TOL = 24
TOL_TEAL = 4        # line teal (8,153,129) vs marker teal (0,150,136) differ by 8


def _mask(img, colour, tol=TOL):
    return np.all(np.abs(img - np.array(colour)) <= tol, axis=2)


def _white(img):
    mx = img.max(axis=2)
    mn = img.min(axis=2)
    return (mn > 195) & ((mx - mn) < 30)


# ------------------------------------------------------------- axis geometry
PANEL_FRAC = 0.84        # price-scale badges live right of this fraction of the width


def find_axis_x(img):
    """First column treated as price scale rather than plot.

    The custom SS badges overhang the scale panel into the plot, so this is a
    fixed fraction rather than "right of the last plotted pixel" -- the drawn
    series all end well before it (the charts leave blank space for future
    sessions).
    """
    return int(img.shape[1] * PANEL_FRAC)


def find_badges(img, axis_x):
    """Filled badge rectangles on the price scale -> (y_centre, kind), top-down."""
    panel = img[:, axis_x:]
    out = []
    for kind, m in (("price", _white(panel)),
                    ("ss", _mask(panel, COL_TEAL_MARK) | _mask(panel, COL_RED))):
        lab, n = ndimage.label(m)
        for sl in ndimage.find_objects(lab):
            if sl is None:
                continue
            ys, xs = sl
            h = ys.stop - ys.start
            w = xs.stop - xs.start
            fill = m[sl].mean()
            # a badge is a wide, solid, single-line-tall block; digits are small
            # and gridline text is thin, so require both size and fill
            if 18 <= h <= 60 and w >= 60 and fill > 0.55:
                out.append(((ys.start + ys.stop) / 2.0, kind, h, w))
    out.sort()

    # A badge renders as two blocks (the "SS:TREND" caption and the value); they
    # share a baseline, so collapse anything within a few pixels of the same row.
    merged = []
    for y, kind, h, w in out:
        if merged and abs(y - merged[-1][0]) <= 10:
            py, pk, ph, pw = merged[-1]
            merged[-1] = ((py + y) / 2.0, pk, max(ph, h), pw + w)
        else:
            merged.append((y, kind, h, w))
    return merged


def describe_badges(img, axis_x=None):
    axis_x = axis_x if axis_x is not None else find_axis_x(img)
    return find_badges(img, axis_x)


def calibrate(img, anchors, axis_x=None):
    """anchors: list of prices, HIGH TO LOW, matching badges top-down.

    Returns (slope, intercept, resid_pct, badge_ys) for price = slope*y + intercept.
    """
    axis_x = axis_x if axis_x is not None else find_axis_x(img)
    badges = find_badges(img, axis_x)
    ys = [b[0] for b in badges]
    if len(ys) < 2 or len(anchors) < 2:
        return None
    if len(ys) != len(anchors):
        return None                     # ambiguous; caller reports and skips
    ys = np.asarray(ys, dtype=float)
    px = np.asarray(anchors, dtype=float)
    slope, intercept = np.polyfit(ys, px, 1)
    fit = slope * ys + intercept
    resid = float(np.max(np.abs(fit / px - 1.0)) * 100)
    return slope, intercept, resid, ys


# --------------------------------------------------------------- line traces
def column_runs(mask, x, min_gap=2):
    """Centres of contiguous vertical runs of `mask` in column x."""
    ys = np.nonzero(mask[:, x])[0]
    if len(ys) == 0:
        return []
    breaks = np.nonzero(np.diff(ys) > min_gap)[0]
    return [float(g.mean()) for g in np.split(ys, breaks + 1)]


def trace_series(mask, x_lo, x_hi, max_jump=25.0, max_gap=15):
    """Follow one continuous curve through a colour mask, column by column.

    Connected-component labelling does not work here: the white candles are drawn
    over the lines and cut them into dozens of fragments. What does work is that
    the RANGE line moves only a few pixels per column while the dotted TRADE and
    TREND series sit far away, so nearest-neighbour tracking stays on the line.

    Seeding matters more than tracking. The seed is the longest stretch of columns
    that contain exactly one run -- in such a stretch the only thing present is the
    line, because the dots are sparse in x. Tracking then extends right and left
    from there, carrying the last position across occlusions instead of jumping.
    """
    runs = {x: column_runs(mask, x) for x in range(x_lo, x_hi)}

    # longest stretch of unambiguous columns
    best = (0, None)
    cur_start = None
    for x in range(x_lo, x_hi):
        if len(runs[x]) == 1:
            if cur_start is None:
                cur_start = x
            if x - cur_start + 1 > best[0]:
                best = (x - cur_start + 1, cur_start)
        else:
            cur_start = None
    if best[1] is None:
        return None
    seed_x = best[1] + best[0] // 2

    out = np.full(mask.shape[1], np.nan)
    for direction in (1, -1):
        y = runs[seed_x][0]
        x = seed_x
        gap = 0
        while x_lo <= x < x_hi:
            cands = runs[x]
            hit = False
            if cands:
                nearest = min(cands, key=lambda c: abs(c - y))
                if abs(nearest - y) <= max_jump:
                    y = nearest
                    out[x] = y
                    hit = True
            # A real line is continuous. A long blank stretch means the line has
            # ended, and anything beyond it is an annotation (arrow, ticker box)
            # that tracking must not jump onto.
            gap = 0 if hit else gap + 1
            if gap > max_gap:
                break
            x += direction
    return out


def trace_line(img, colour, axis_x, x_hi=None, tol=TOL, min_cover=0.4):
    """Traced y-per-column for one RANGE line, or None if it cannot be followed."""
    m = _mask(img, colour, tol)
    m[:, axis_x:] = False
    x_lo = 0
    x_hi = int(x_hi) if x_hi is not None else axis_x
    ys = trace_series(m, x_lo, x_hi)
    if ys is None:
        return None
    covered = np.isfinite(ys[x_lo:x_hi]).mean()
    return ys if covered >= min_cover else None


def candle_columns(img, axis_x):
    """x centres of the candles, used to convert pixel x into a bar index."""
    w = _white(img)
    w[:, axis_x:] = False
    w[: int(img.shape[0] * 0.06), :] = False        # header text
    lab, n = ndimage.label(w, structure=np.ones((3, 3)))
    xs = []
    for sl in ndimage.find_objects(lab):
        if sl is None:
            continue
        h = sl[0].stop - sl[0].start
        ww = sl[1].stop - sl[1].start
        if ww <= 30 and h >= 6:                     # thin+tall = candle, not a label box
            xs.append((sl[1].start + sl[1].stop) / 2.0)
    xs = np.array(sorted(xs))
    if len(xs) < 8:
        return xs
    # merge body/wick fragments that belong to the same bar
    merged, cur = [], [xs[0]]
    gaps = np.diff(xs)
    pitch = float(np.median(gaps[gaps > 2])) if (gaps > 2).any() else 10.0
    for a, b in zip(xs[:-1], xs[1:]):
        if b - a < pitch * 0.5:
            cur.append(b)
        else:
            merged.append(np.mean(cur))
            cur = [b]
    merged.append(np.mean(cur))
    return np.array(merged)


# ------------------------------------------------------------------- driver
def extract(path, anchors, dates=None, verbose=True):
    img = np.array(Image.open(path).convert("RGB")).astype(int)
    axis_x = find_axis_x(img)
    cal = calibrate(img, anchors, axis_x)
    if cal is None:
        badges = find_badges(img, axis_x)
        raise ValueError("axis calibration failed: found %d badges, given %d anchors"
                         % (len(badges), len(anchors)))
    slope, intercept, resid, badge_ys = cal

    hi_y = trace_line(img, COL_RED, axis_x)
    lo_y = trace_line(img, COL_TEAL_LINE, axis_x, tol=TOL_TEAL)
    if hi_y is None or lo_y is None:
        raise ValueError("could not trace both range lines")

    # The lines are drawn once per bar, so their right-hand end marks the last
    # session. Anything the candle detector finds beyond it is an annotation.
    drawn = np.isfinite(hi_y) & np.isfinite(lo_y)
    x_hi = int(np.nonzero(drawn)[0].max()) + 2

    cols = candle_columns(img, axis_x)
    cols = cols[cols <= x_hi]
    if len(cols) < 8:
        raise ValueError("too few candles detected (%d)" % len(cols))

    rows = []
    for k, cx in enumerate(cols):
        x = int(round(cx))
        lo_x = max(0, x - 1)
        hi_x = min(x_hi, x + 2)
        h = np.nanmean(hi_y[lo_x:hi_x])
        l = np.nanmean(lo_y[lo_x:hi_x])
        rows.append({
            "bar_from_right": len(cols) - 1 - k,
            "x": cx,
            "range_high": slope * h + intercept if np.isfinite(h) else np.nan,
            "range_low": slope * l + intercept if np.isfinite(l) else np.nan,
        })
    df = pd.DataFrame(rows)
    if dates is not None:
        n = min(len(df), len(dates))
        df["date"] = pd.NaT
        df.loc[df.index[-n:], "date"] = list(dates[-n:])
    df.attrs["axis_resid_pct"] = resid
    df.attrs["n_candles"] = len(cols)
    if verbose:
        print("  axis fit residual %.3f%% over %d badges | %d candles"
              % (resid, len(badge_ys), len(cols)))
    return df


def main():
    ap = argparse.ArgumentParser(description="Trace RANGE lines from a chart image.")
    ap.add_argument("image")
    ap.add_argument("--anchors", required=True,
                    help="comma-separated badge prices, HIGHEST first")
    ap.add_argument("--verify", type=float, default=None,
                    help="known range-high on the last bar; prints the error")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    anchors = [float(x) for x in args.anchors.split(",")]
    df = extract(args.image, anchors)
    print(df.tail(10).to_string(index=False))
    if args.verify is not None:
        got = float(df["range_high"].iloc[-1])
        print("\n  verify: traced %.3f vs known %.3f -> %+.2f%%"
              % (got, args.verify, 100 * (got / args.verify - 1)))
    if args.out:
        df.to_csv(args.out, index=False)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
