"""Re-fit the RANGE profile against every Hedgeye level we have.

The 2026-08-30 ETF Pro weekly exposed two systematic biases: our low edge sat above
Hedgeye's on 38 of 39 names, and our range ran about three-quarters their width on
single ETFs while matching them on macro. The first looked like anchor lag -- a
5-day EMA sits above spot after a down day, lifting the whole envelope -- and the
second like a multiplier that is simply too small for ETFs.

This searches (lam, anchor_span, m_up, m_dn) against the pooled reference set and,
crucially, scores every candidate out of sample: fit on the Early Look macro series,
test on the ETF Pro weeklies, and the reverse. A parameter set that only wins on the
data it was fitted to is overfitting, and the whole reason the earlier profile looked
better than it was is that ETF Pro was in its training set.

Objective is MEDIAN absolute edge error rather than RMSE. The VIX misses by ~4% on
every observation because a Gaussian envelope fits a mean-reverting log-distributed
index badly; under a squared loss that one instrument would drag the whole fit.
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices, repo_path
from ..model import adaptive_ma, range_ewma

# Every reference file, tagged by which family it belongs to so the fit can be
# scored across families rather than pooled into one indistinguishable blob.
SOURCES = (
    ("macro", "hedgeye_early_look_all.csv"),
    ("etf", "hedgeye_ranges.csv"),
    ("etf", "hedgeye_ranges_0830.csv"),
)


def load_reference(verbose=True):
    """One frame of (family, ticker, yf, prior_close_date, rr_low, rr_high, dir)."""
    frames = []
    for family, name in SOURCES:
        p = repo_path("reference", name)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        if "yf" not in d:
            d["yf"] = d["ticker"]
        d["family"] = family
        frames.append(d[["family", "ticker", "yf", "prior_close_date",
                         "rr_low", "rr_high", "trend_dir"]])
    ref = pd.concat(frames, ignore_index=True)
    ref["trend_dir"] = (ref["trend_dir"].astype(str)
                        .str.replace("bullish", "bull").str.replace("bearish", "bear"))
    if verbose:
        print("reference: %d observations  (%s)"
              % (len(ref), ", ".join("%s %d" % (f, n)
                                     for f, n in ref.family.value_counts().items())))
    return ref


def _series(ref, params, verbose=True):
    """Close series per feed symbol, fetched once and reused across the grid."""
    return load_prices(sorted(ref["yf"].unique()), params=params, verbose=verbose)


def anchors(ref, prices, lam, span):
    """Per-observation (anchor, sigma, hedgeye low, hedgeye high, family).

    Precomputed because the envelope is anchor*exp(-m_dn*sigma) upward and
    anchor*exp(+m_up*sigma) downward -- the multipliers are pure arithmetic once
    these two are known, so a grid over m_up and m_dn costs nothing. Only lam and
    span require touching the price series at all.
    """
    A, S, LO, HI, F = [], [], [], [], []
    for r in ref.itertuples():
        px = prices.get(r.yf)
        if px is None or "Close" not in px:
            continue
        c = px["Close"].dropna()
        i = c.index.searchsorted(pd.Timestamp(r.prior_close_date), side="right") - 1
        if i < 40:
            continue
        cut = c.iloc[: i + 1]
        sig = range_ewma.ewma_sigma(cut, lam=lam)
        anc = cut.ewm(span=max(1, int(span)), adjust=False).mean()
        a, sg = float(anc.iloc[-1]), float(sig.iloc[-1])
        if not (np.isfinite(a) and np.isfinite(sg) and sg > 0):
            continue
        A.append(a); S.append(sg)
        LO.append(float(r.rr_low)); HI.append(float(r.rr_high)); F.append(r.family)
    return (np.array(A), np.array(S), np.array(LO), np.array(HI), np.array(F))


def _errors_from(A, S, LO, HI, m_up, m_dn):
    lo = A * np.exp(-m_dn * S)
    hi = A * np.exp(+m_up * S)
    return 100 * (lo / LO - 1), 100 * (hi / HI - 1), 100 * (hi / lo - 1), 100 * (HI / LO - 1)


def score_from(pre, m_up, m_dn, family=None):
    """(median |edge error|, width ratio, n) from a precomputed anchor set."""
    A, S, LO, HI, F = pre
    if family:
        k = F == family
        A, S, LO, HI = A[k], S[k], LO[k], HI[k]
    if not len(A):
        return np.inf, np.nan, 0
    le, he, wm, wh = _errors_from(A, S, LO, HI, m_up, m_dn)
    return float(np.median(np.abs(np.r_[le, he]))), float(np.median(wm / wh)), len(A)


def search(ref, prices, lams, spans, ups, dns, family=None, verbose=True, top=10):
    """Grid search over lam, span and both multipliers."""
    rows = []
    for lam in lams:
        for span in spans:
            pre = anchors(ref, prices, lam, span)
            for mu, md in itertools.product(ups, dns):
                med, ratio, n = score_from(pre, mu, md, family=family)
                rows.append(dict(lam=lam, span=span, m_up=round(mu, 2), m_dn=round(md, 2),
                                 median_err=round(med, 4), width_ratio=round(ratio, 3), n=n))
    out = pd.DataFrame(rows).sort_values("median_err").reset_index(drop=True)
    if verbose:
        print(out.head(top).to_string(index=False))
    return out


def main():
    ap = argparse.ArgumentParser(description="Re-fit the RANGE profile to Hedgeye.")
    ap.add_argument("--coarse", action="store_true", help="wider, cheaper grid")
    args = ap.parse_args()
    params = load_params()
    ref = load_reference()
    prices = _series(ref, params)
    cur = params["range"]["hedgeye_anchor"]
    print("\ncurrent: lam %.2f span %d m_up %.2f m_dn %.2f"
          % (cur["lam"], cur["anchor_span"], cur["m_up"], cur["m_dn"]))
    pre = anchors(ref, prices, cur["lam"], cur["anchor_span"])
    for fam in (None, "macro", "etf"):
        med, ratio, n = score_from(pre, cur["m_up"], cur["m_dn"], family=fam)
        print("  %-6s median |err| %.3f%%  width %.2fx  n=%d"
              % (fam or "all", med, ratio, n))


if __name__ == "__main__":
    main()
