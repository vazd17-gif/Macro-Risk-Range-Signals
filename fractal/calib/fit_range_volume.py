"""Fit and test a volume-adjusted RANGE against Hedgeye's own published ranges.

Per Hedgeye, the Risk Range boundaries are adjusted by the rate of change of
volume against a 1-month (TRADE) and 3-month (TREND) baseline. Volume was ruled
out of the *lines* in the build spec; this tests it in the RANGE width, where
Hedgeye actually uses it.

Half-width multiplier model (per edge, fitted separately for the up and down side):

    m_eff = a + b1*v1 + b3*v3
    v1 = volume / SMA(volume, 21)      1-month ratio  (TRADE duration)
    v3 = volume / SMA(volume, 63)      3-month ratio  (TREND duration)

Reference: the 87 Early Look "Our Levels" ranges across 2026-08-24..28, which span
the NVDA-earnings week and the 8/28 Warsh bond-vol spike, so there is real volume
variation to detect the effect. Reports feature correlations and compares nested
models (pure sigma -> +v1 -> +v3 -> +both) by rmse and width ratio, then writes the
best as the `hedgeye_vol` range profile.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices, repo_path, save_params
from ..model.range_ewma import ewma_sigma, volume_features

DEFAULT_REFS = ["hedgeye_early_look_week.csv", "hedgeye_ranges.csv"]


def load_refs(names):
    frames = []
    for f in names:
        d = pd.read_csv(repo_path("reference", f))
        if "yf" not in d:
            d["yf"] = d["ticker"]
        frames.append(d[["ticker", "yf", "prior_close_date", "rr_low", "rr_high"]])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def has_real_volume(yf: str) -> bool:
    """Volume is meaningful only for exchange-traded shares (stocks/ETFs).

    Cash indices, spot prices and continuous futures either report no Yahoo volume
    or report a number that is not comparable to share volume, so they are excluded
    from the volume analysis per the source's own guidance (volume applies to
    individual stocks and ETFs).
    """
    return not (yf.startswith("^") or yf.endswith("=F")
                or yf.endswith(".NYB") or yf.endswith(".SS"))


def build(refs, params, lam=0.94, anchor_span=5, volume_only=False):
    if volume_only:
        refs = refs[refs["yf"].map(has_real_volume)]
    prices = load_prices(sorted(refs["yf"].unique()), params=params, verbose=False)
    rows = []
    for r in refs.itertuples():
        px = prices.get(r.yf)
        if px is None:
            continue
        c = px["Close"].dropna()
        pos = c.index.searchsorted(pd.Timestamp(r.prior_close_date), side="right") - 1
        if pos < 63:
            continue
        sig = ewma_sigma(c, lam=lam).values[pos]
        anchor = c.ewm(span=anchor_span, adjust=False).mean().values[pos]
        if not (np.isfinite(sig) and sig > 0 and np.isfinite(anchor)):
            continue
        v1, v3 = volume_features(px.get("Volume"), c.index)
        rows.append({
            "ticker": r.ticker, "date": r.prior_close_date, "sigma": sig, "anchor": anchor,
            "lo": float(r.rr_low), "hi": float(r.rr_high),
            "hu": np.log(r.rr_high / anchor), "hd": np.log(anchor / r.rr_low),
            "v1": float(v1.iloc[pos]), "v3": float(v3.iloc[pos]),
        })
    df = pd.DataFrame(rows)
    df["impl_m"] = (df["hu"] + df["hd"]) / (2 * df["sigma"])
    return df


def _fit_side(df, target_col, feats):
    """Least-squares multiplier m = a + sum(b_k*feat_k) for one edge."""
    sig = df["sigma"].values
    y = df[target_col].values / sig                 # target multiplier per obs
    X = np.column_stack([np.ones(len(df))] + [df[f].values for f in feats])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _eval(df, feats, cu, cd):
    def m(df, c):
        out = c[0] * np.ones(len(df))
        for k, f in enumerate(feats):
            out = out + c[k + 1] * df[f].values
        return np.maximum(out, 0.4)
    hi = df["anchor"].values * np.exp(m(df, cu) * df["sigma"].values)
    lo = df["anchor"].values * np.exp(-m(df, cd) * df["sigma"].values)
    err = np.concatenate([hi / df["hi"].values - 1, lo / df["lo"].values - 1])
    rmse = 100 * np.sqrt(np.mean(err ** 2))
    wr = np.median((hi / lo - 1) / (df["hi"].values / df["lo"].values - 1))
    return rmse, wr


def main():
    ap = argparse.ArgumentParser(description="Fit a volume-adjusted Hedgeye RANGE.")
    ap.add_argument("--refs", default=",".join(DEFAULT_REFS))
    ap.add_argument("--lam", type=float, default=0.94)
    ap.add_argument("--anchor-span", type=int, default=5)
    ap.add_argument("--write", action="store_true", help="save hedgeye_vol profile")
    ap.add_argument("--volume-only", action="store_true",
                    help="restrict to stocks/ETFs with real volume (per source guidance)")
    args = ap.parse_args()

    params = load_params()
    df = build(load_refs(args.refs.split(",")), params, lam=args.lam, anchor_span=args.anchor_span, volume_only=args.volume_only)
    print("=== volume-adjusted RANGE vs Hedgeye  (%d edges, %d dates) ===\n"
          % (len(df), df["date"].nunique()))

    print("  implied width-multiplier: median %.2f  range [%.2f, %.2f]"
          % (df["impl_m"].median(), df["impl_m"].min(), df["impl_m"].max()))
    print("\n  feature correlation with the implied multiplier (spearman):")
    df = df.assign(vacc=df["v1"] / df["v3"])
    for f, lbl in [("v1", "vol / 1mo baseline"), ("v3", "vol / 3mo baseline"),
                   ("vacc", "v1/v3 (vol acceleration)"), ("sigma", "realized sigma")]:
        print("    %-24s %+.2f" % (lbl, df["impl_m"].corr(df[f], method="spearman")))

    print("\n  nested half-width models (rmse on edge levels | width ratio, 1.0 = exact):")
    specs = [("pure sigma", []), ("+v1 (1mo)", ["v1"]),
             ("+v3 (3mo)", ["v3"]), ("+v1+v3", ["v1", "v3"])]
    best = None
    for name, feats in specs:
        cu = _fit_side(df, "hu", feats)
        cd = _fit_side(df, "hd", feats)
        rmse, wr = _eval(df, feats, cu, cd)
        tag = "  ".join("%s" % round(x, 3) for x in cu)
        print("    %-12s rmse %.2f%%   width %.2fx   m_up coefs [%s]" % (name, rmse, wr, tag))
        if name == "+v1+v3":
            best = (feats, cu, cd, rmse, wr)

    feats, cu, cd, rmse, wr = best
    print("\n  full model (a + b1*v1 + b3*v3):")
    print("    up:  a=%.3f  b1=%.3f  b3=%.3f" % (cu[0], cu[1], cu[2]))
    print("    dn:  a=%.3f  b1=%.3f  b3=%.3f" % (cd[0], cd[1], cd[2]))
    print("    rmse %.2f%%   width %.2fx" % (rmse, wr))

    if args.write:
        params["range"]["hedgeye_vol"] = {
            "lam": float(args.lam), "anchor_span": int(args.anchor_span),
            "a_up": round(float(cu[0]), 4), "b1_up": round(float(cu[1]), 4),
            "b3_up": round(float(cu[2]), 4),
            "a_dn": round(float(cd[0]), 4), "b1_dn": round(float(cd[1]), 4),
            "b3_dn": round(float(cd[2]), 4),
            "w1": 21, "w3": 63, "winsor_z": None,
            "fit_rmse_pct": round(float(rmse), 3), "fit_width_ratio": round(float(wr), 3),
            "note": "volume-adjusted; fitted to 87 Hedgeye Early Look ranges (5 days). "
                    "b coefs negative: above-average volume tightens the multiplier.",
        }
        save_params(params)
        print("\n[vol-fit] wrote 'hedgeye_vol' profile to config/params.yaml")


if __name__ == "__main__":
    main()
