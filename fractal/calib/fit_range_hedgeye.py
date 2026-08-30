"""Fit a RANGE profile to Hedgeye's *own* published ranges (not Similar Set's).

The Similar Set fit (calib/fit_range.py) reproduces Similar Set, which the
Hedgeye validation showed is narrower than Hedgeye on quiet instruments. Similar
Set and Hedgeye are simply different-width products, so the fix is not to change
the Similar Set fit but to add a second profile aimed at Hedgeye.

Model: high = A*exp(+max(m_up*sigma, floor)),  low = A*exp(-max(m_dn*sigma, floor))
       A = EMA(close, anchor_span),  sigma = close-to-close EWMA(lambda)

The floor is the new degree of freedom. The hypothesis from the validation is
that Hedgeye holds a minimum width on calm names, which a pure multiplier cannot
express (it would have to widen every name, blowing out the volatile ones).

Ground truth: the two reference CSVs harvested from email (35 ETFs @ 8/21 +
17 macro @ 8/27). Small and only two dates, so this is a coarse fit — enough to
set sensible defaults and to show the floor helps, not a precision calibration.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices, repo_path, save_params
from ..model.range_ewma import ewma_sigma

LAMBDAS = [0.94, 0.95, 0.96, 0.97, 0.98, 0.985]
SPANS = [1, 3, 5, 8]
FLOORS = [0.0, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.025]


DEFAULT_REFS = ("hedgeye_ranges.csv", "hedgeye_early_look_all.csv",
                "hedgeye_ranges_0830.csv")


def load_refs(files=None):
    """Every published range we hold, tagged by whether it is an ETF or macro.

    The two populations are kept distinguishable because a single multiplier fitted
    across both hides a real difference: the ETF weeklies imply a much deeper
    downside reach than the macro Early Looks do, and pooling them splits the
    difference in a way that fits neither.
    """
    frames = []
    for f in (files or DEFAULT_REFS):
        d = pd.read_csv(repo_path("reference", f))
        if "yf" not in d:
            d["yf"] = d["ticker"]
        d = d.copy()
        d["kind"] = "macro" if "early_look" in f else "etf"
        frames.append(d[["ticker", "yf", "prior_close_date", "rr_low", "rr_high", "kind"]])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["ticker", "prior_close_date"])


class Obs:
    __slots__ = ("anchor", "sig", "lo", "hi", "ticker")

    def __init__(self, anchor, sig, lo, hi, ticker):
        self.anchor = anchor
        self.sig = sig
        self.lo = lo
        self.hi = hi
        self.ticker = ticker


def build(refs, lam, span, params):
    prices = load_prices(sorted(refs["yf"].unique()), params=params, verbose=False)
    out = []
    for r in refs.itertuples():
        px = prices.get(r.yf)
        if px is None:
            continue
        c = px["Close"].dropna()
        pos = c.index.searchsorted(pd.Timestamp(r.prior_close_date), side="right") - 1
        if pos < 40:
            continue
        sig = ewma_sigma(c, lam=lam).values[pos]
        anchor = (c.values[pos] if span <= 1 else
                  c.ewm(span=span, adjust=False).mean().values[pos])
        if not (np.isfinite(sig) and sig > 0 and np.isfinite(anchor)):
            continue
        out.append(Obs(anchor, sig, float(r.rr_low), float(r.rr_high), r.ticker))
    return out


def fit_mult(obs, floor, side):
    """Closed-form-ish multiplier for one edge given a floor.

    half-width = max(m*sigma, floor). For a target log half-width h_i, on points
    where the floor does not bind we want m*sigma_i ~= h_i. Fit m by least squares
    over the whole set, then the floor lifts the binding points. Search m on a fine
    grid — cheap and avoids the hinge non-linearity.
    """
    tgt = np.array([np.log(o.hi / o.anchor) if side == "up" else np.log(o.anchor / o.lo)
                    for o in obs])
    sig = np.array([o.sig for o in obs])
    best = (np.inf, 1.9)
    for m in np.arange(1.2, 3.6, 0.02):
        hw = np.maximum(m * sig, floor)
        sse = float(np.sum((hw - tgt) ** 2))
        if sse < best[0]:
            best = (sse, m)
    return best[1]


def score(obs, m_up, m_dn, floor):
    sse = 0.0
    for o in obs:
        hu = max(m_up * o.sig, floor)
        hd = max(m_dn * o.sig, floor)
        sse += (o.anchor * np.exp(hu) / o.hi - 1.0) ** 2
        sse += (o.anchor * np.exp(-hd) / o.lo - 1.0) ** 2
    return sse, 2 * len(obs)


def main():
    ap = argparse.ArgumentParser(description="Fit a Hedgeye-targeted RANGE profile.")
    ap.add_argument("--csv", action="append", default=None,
                    help="reference CSV under reference/ (repeatable; default: all three)")
    ap.add_argument("--kind", choices=["all", "etf", "macro"], default="all",
                    help="fit on ETF rows, macro rows, or both")
    ap.add_argument("--write", action="store_true",
                    help="save as the 'hedgeye_anchor' range profile in params.yaml")
    args = ap.parse_args()

    params = load_params()
    refs = load_refs(args.csv)
    if args.kind != "all":
        refs = refs[refs["kind"] == args.kind]

    results = []
    for lam in LAMBDAS:
        for span in SPANS:
            obs = build(refs, lam, span, params)
            if len(obs) < 20:
                continue
            for floor in FLOORS:
                m_up = fit_mult(obs, floor, "up")
                m_dn = fit_mult(obs, floor, "dn")
                sse, n = score(obs, m_up, m_dn, floor)
                results.append({"lam": lam, "anchor_span": span, "floor": floor,
                                "m_up": m_up, "m_dn": m_dn, "sse": sse, "n": n,
                                "rmse_pct": 100 * np.sqrt(sse / n)})
    results.sort(key=lambda r: r["sse"])

    print("=== fit RANGE to Hedgeye's published edges ===")
    print("  %d observations, %d dates, %d ETF / %d macro"
          % (len(refs), refs["prior_close_date"].nunique(),
             int((refs["kind"] == "etf").sum()),
             int((refs["kind"] == "macro").sum())))
    print("  best:")
    for r in results[:8]:
        print("    lambda=%.3f span=%d floor=%.4f  m_up=%.2f m_dn=%.2f  rmse=%.2f%%"
              % (r["lam"], r["anchor_span"], r["floor"], r["m_up"], r["m_dn"], r["rmse_pct"]))

    no_floor = min((r for r in results if r["floor"] == 0), key=lambda r: r["sse"])
    best = results[0]
    print("\n  best WITHOUT floor: lambda=%.3f span=%d  m_up=%.2f m_dn=%.2f  rmse=%.2f%%"
          % (no_floor["lam"], no_floor["anchor_span"], no_floor["m_up"],
             no_floor["m_dn"], no_floor["rmse_pct"]))
    print("  best WITH    floor: lambda=%.3f span=%d floor=%.4f  m_up=%.2f m_dn=%.2f  rmse=%.2f%%"
          % (best["lam"], best["anchor_span"], best["floor"], best["m_up"],
             best["m_dn"], best["rmse_pct"]))
    gain = 100 * (1 - best["rmse_pct"] / no_floor["rmse_pct"])
    print("  floor improves rmse by %.1f%%" % gain)

    if args.write:
        params["range"]["hedgeye_anchor"] = {
            "lam": float(best["lam"]), "anchor_span": int(best["anchor_span"]),
            "m_up": round(float(best["m_up"]), 3), "m_dn": round(float(best["m_dn"]), 3),
            "floor": round(float(best["floor"]), 4), "winsor_z": None,
            "fit_rmse_pct": round(float(best["rmse_pct"]), 3),
            "note": ("fitted to %d Hedgeye ranges across %d dates (%s rows). ETF and macro "
                     "are separate populations - ETFs want a wider, downside-skewed band "
                     "(~2.3/2.7) and macro a narrower symmetric one (~1.7/1.7) - so this "
                     "profile is fitted to the population the report actually covers."
                     % (len(refs), refs["prior_close_date"].nunique(), args.kind)),
        }
        save_params(params)
        print("\n[hedgeye-fit] wrote 'hedgeye_anchor' profile to config/params.yaml")
        print("  activate with: range.active = hedgeye_anchor  (add floor to compute dispatch)")


if __name__ == "__main__":
    main()
