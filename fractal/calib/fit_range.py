"""Fit the RISK RANGE against traced published ranges.

Input is `data/range_labels.csv` from calib/extract_ranges.py: one row per
(ticker, session) with the published range high and low.

The model family is

    RANGE_high[t+1] = A[t] * exp(+m_up * sigma[t])
    RANGE_low [t+1] = A[t] * exp(-m_dn * sigma[t])

with sigma the close-to-close EWMA volatility and A an EMA of close of length
`anchor_span`. Setting anchor_span = 1 makes A the last close, which recovers the
build spec's locked `spot_ewma` form exactly, so the two are nested and directly
comparable rather than being separate stories.

For any (lambda, anchor_span) the two widths have a closed form: in log space the
edges are linear in sigma through the anchor, so m_up and m_dn are least-squares
slopes through the origin. Only the two-dimensional grid needs searching.

The diagnostic that matters most is printed first: whether the published range is
in fact centred on spot. The build spec records it as symmetric and locked, on the
strength of one matching upper edge.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..data.loader import load_params, load_prices, repo_path, save_params
from ..model.range_ewma import ewma_sigma

LAMBDAS = [0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98]
SPANS = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60]


def load_range_labels(path=None):
    path = path or repo_path("data", "range_labels.csv")
    df = pd.read_csv(path, parse_dates=["date"])
    return df.dropna(subset=["range_high", "range_low"])


class Book:
    """One ticker: price history plus the published range rows aligned to it."""

    def __init__(self, sym, close, rows):
        self.sym = sym
        self.close = close
        self.rows = rows          # (prior_index, range_high, range_low, date, ticker)
        self._sig = {}

    def sigma(self, lam, winsor_z=None):
        key = (lam, winsor_z)
        if key not in self._sig:
            self._sig[key] = ewma_sigma(self.close, lam=lam, winsor_z=winsor_z).values
        return self._sig[key]


def build_books(labels, params=None, verbose=True):
    prices = load_prices(sorted(labels["yf"].unique()), params=params, verbose=verbose)
    books = []
    for sym, grp in labels.groupby("yf"):
        px = prices.get(sym)
        if px is None:
            continue
        close = px["Close"].dropna()
        idx = close.index
        rows = []
        for r in grp.itertuples():
            # the range for session `date` is built from data through the prior bar
            prior = idx.searchsorted(r.date) - 1
            if prior < 30:
                continue
            rows.append((prior, float(r.range_high), float(r.range_low),
                         r.date, r.ticker))
        if rows:
            books.append(Book(sym, close, rows))
    return books


def symmetry_report(books):
    """How the published range sits relative to the last close."""
    up, dn = [], []
    per = {}
    for b in books:
        c = b.close.values
        u, d = [], []
        for prior, hi, lo, _, tk in b.rows:
            base = c[prior]
            u.append(np.log(hi / base))
            d.append(np.log(base / lo))
        up.extend(u)
        dn.extend(d)
        per[b.rows[0][4]] = (100 * np.median(u), 100 * np.median(d), len(u))
    up, dn = np.asarray(up), np.asarray(dn)
    return up, dn, per


def fit(books, lambdas=LAMBDAS, spans=SPANS, winsor_z=None):
    results = []
    for lam in lambdas:
        for span in spans:
            num_u = den = num_d = 0.0
            n = 0
            for b in books:
                sig = b.sigma(lam, winsor_z)
                anchor = (b.close.values if span <= 1 else
                          b.close.ewm(span=span, adjust=False).mean().values)
                for prior, hi, lo, _, _ in b.rows:
                    s = sig[prior]
                    a = anchor[prior]
                    if not (np.isfinite(s) and s > 0 and np.isfinite(a) and a > 0):
                        continue
                    num_u += s * np.log(hi / a)
                    num_d += s * np.log(a / lo)
                    den += s * s
                    n += 1
            if den <= 0 or n < 20:
                continue
            m_up, m_dn = num_u / den, num_d / den

            sse = 0.0
            for b in books:
                sig = b.sigma(lam, winsor_z)
                anchor = (b.close.values if span <= 1 else
                          b.close.ewm(span=span, adjust=False).mean().values)
                for prior, hi, lo, _, _ in b.rows:
                    s, a = sig[prior], anchor[prior]
                    if not (np.isfinite(s) and s > 0 and np.isfinite(a) and a > 0):
                        continue
                    sse += (a * np.exp(m_up * s) / hi - 1.0) ** 2
                    sse += (a * np.exp(-m_dn * s) / lo - 1.0) ** 2
            results.append({"lam": lam, "anchor_span": span, "m_up": m_up, "m_dn": m_dn,
                            "sse": sse, "n": 2 * n,
                            "rmse_pct": 100 * np.sqrt(sse / (2 * n))})
    return sorted(results, key=lambda r: r["sse"])


def residuals(books, spec):
    lam, span = spec["lam"], spec["anchor_span"]
    rows = []
    for b in books:
        sig = b.sigma(lam)
        anchor = (b.close.values if span <= 1 else
                  b.close.ewm(span=span, adjust=False).mean().values)
        for prior, hi, lo, date, tk in b.rows:
            s, a = sig[prior], anchor[prior]
            if not (np.isfinite(s) and s > 0):
                continue
            rows.append({"ticker": tk, "date": date.date().isoformat(),
                         "hi_err_pct": 100 * (a * np.exp(spec["m_up"] * s) / hi - 1),
                         "lo_err_pct": 100 * (a * np.exp(-spec["m_dn"] * s) / lo - 1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Fit the RISK RANGE to traced published ranges.")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--winsor", type=float, default=None,
                    help="clip returns at this many prior sigmas before the vol update")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    params = load_params()
    labels = load_range_labels(args.labels)
    books = build_books(labels, params=params)
    n_obs = sum(len(b.rows) for b in books)
    print("[range-fit] %d published ranges over %d tickers\n" % (n_obs, len(books)))

    up, dn, per = symmetry_report(books)
    print("=== is the range centred on the last close? ===")
    print("  median distance to the upper edge: %+.2f%%" % (100 * np.median(up)))
    print("  median distance to the lower edge: %+.2f%%" % (100 * np.median(dn)))
    print("  lower/upper width ratio: %.2fx" % (np.median(dn) / np.median(up)))
    print("  per ticker (up%% / down%% / n):")
    for tk in sorted(per):
        u, d, k = per[tk]
        print("    %-8s %5.2f / %5.2f  (%.2fx, n=%d)" % (tk, u, d, d / u if u else np.nan, k))

    print("\n=== model fit ===")
    res = fit(books, winsor_z=args.winsor)
    if not res:
        print("  no usable fit")
        return
    best = res[0]
    print("  WINNER  lambda=%.2f anchor_span=%d  m_up=%.3f m_dn=%.3f  rmse=%.2f%%  (n=%d)"
          % (best["lam"], best["anchor_span"], best["m_up"], best["m_dn"],
             best["rmse_pct"], best["n"]))
    print("  runners-up:")
    for r in res[1:5]:
        print("    lambda=%.2f span=%-3d m_up=%.3f m_dn=%.3f  rmse=%.2f%%"
              % (r["lam"], r["anchor_span"], r["m_up"], r["m_dn"], r["rmse_pct"]))

    spot = [r for r in res if r["anchor_span"] == 1]
    if spot:
        s = spot[0]
        print("\n  build-spec form (spot-centred, anchor_span=1): "
              "lambda=%.2f m_up=%.3f m_dn=%.3f rmse=%.2f%%"
              % (s["lam"], s["m_up"], s["m_dn"], s["rmse_pct"]))
        sym = min((r for r in res if r["anchor_span"] == 1),
                  key=lambda r: abs(r["m_up"] - r["m_dn"]))
        print("  spot-centred and symmetric would need m_up == m_dn; "
              "best spot fit has m_dn/m_up = %.2fx" % (s["m_dn"] / s["m_up"]))

    res_df = residuals(books, best)
    by_tk = res_df.groupby("ticker")[["hi_err_pct", "lo_err_pct"]].apply(
        lambda g: pd.Series({"hi_med": g.hi_err_pct.median(),
                             "lo_med": g.lo_err_pct.median(),
                             "n": len(g)}))
    print("\n  per-ticker residuals (median %%):")
    print(by_tk.to_string(float_format=lambda v: "%.2f" % v))

    if args.write:
        params["range"]["active"] = "anchor_ewma"
        params["range"]["anchor_ewma"].update({
            "lam": float(best["lam"]), "anchor_span": int(best["anchor_span"]),
            "m_up": round(float(best["m_up"]), 4), "m_dn": round(float(best["m_dn"]), 4),
            "winsor_z": args.winsor,
            "fit_rmse_pct": round(float(best["rmse_pct"]), 3),
        })
        params["meta"]["notes"] = "range fitted by calib/fit_range.py on traced charts"
        save_params(params)
        print("\n[range-fit] wrote config/params.yaml")


if __name__ == "__main__":
    main()
