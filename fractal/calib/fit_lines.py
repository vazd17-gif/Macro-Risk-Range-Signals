"""Fit TRADE / TREND / TAIL against the published-level labels.

Objective (build spec section 5): minimise sum((model/label - 1)^2) over label
rows, separately per line.

The search covers three axes rather than one, because the build spec's premise
(the lines are adaptive, KAMA-family) is a hypothesis this harness is meant to
test, not assume:

  family  sma | ema | wma | kama      -- "upgraded SMA" could be any of these
  price   close | hlc3                -- typical price vs close
  lag     0..N bars                   -- the published dots are visibly a step
                                         function that holds a level for several
                                         sessions, so the label may reflect a
                                         value computed some bars before the
                                         last completed bar

Reported per-asset residuals are what identify systematic outliers (XLF wanting
a slower line, SLV TRADE likewise) versus plain noise.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from ..data.loader import load_prices, load_params, save_params
from ..model.adaptive_ma import efficiency_ratio
from . import labels as labels_mod

KAMA_N = [8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60, 75, 90]
KAMA_FAST = [2, 3, 4]
KAMA_SLOW = [20, 24, 28, 32, 36, 40, 45, 50, 55, 60, 66, 72, 80, 88, 96]


# --------------------------------------------------------------- MA families
def sma(px, window):
    return pd.Series(px).rolling(int(window)).mean().values


def ema(px, span):
    return pd.Series(px).ewm(span=int(span), adjust=False).mean().values


def wma(px, window):
    w = np.arange(1, int(window) + 1, dtype=float)
    w /= w.sum()
    return (pd.Series(px).rolling(int(window))
            .apply(lambda x: float(np.dot(x, w)), raw=True).values)


def aema_full(px, er, span_fast, span_slow):
    """EMA whose span is interpolated by the efficiency ratio.

    span(t) = span_slow + ER(t) * (span_fast - span_slow)

    A cleaner test of the adaptivity hypothesis than KAMA: the smoothing constant
    moves linearly in ER instead of being squared, so an efficient (trending) path
    shortens the window smoothly rather than collapsing it.
    """
    out = np.full(len(px), np.nan)
    n0 = int(max(span_slow, 2))
    if len(px) <= n0:
        return out
    v = float(np.mean(px[: n0 + 1]))
    out[n0] = v
    for t in range(n0 + 1, len(px)):
        e = er[t]
        e = 0.0 if not np.isfinite(e) else e
        span = span_slow + e * (span_fast - span_slow)
        a = 2.0 / (span + 1.0)
        v += a * (px[t] - v)
        out[t] = v
    return out


def kama_full(px, er, n, fast, slow):
    fs, ss = 2.0 / (fast + 1.0), 2.0 / (slow + 1.0)
    diff = fs - ss
    out = np.full(len(px), np.nan)
    if len(px) <= n:
        return out
    v = float(np.mean(px[: n + 1]))
    out[n] = v
    for t in range(n + 1, len(px)):
        e = er[t]
        s = (e * diff + ss) ** 2 if np.isfinite(e) else ss * ss
        v += s * (px[t] - v)
        out[t] = v
    return out


class TickerBook:
    def __init__(self, sym, ohlc, rows):
        self.sym = sym
        self.index = ohlc.index
        self.close = ohlc["Close"].values.astype(float)
        self.hlc3 = ((ohlc["High"] + ohlc["Low"] + ohlc["Close"]) / 3.0).values.astype(float)
        self.rows = rows                      # (position, label_value, display_ticker)
        self._er = {}

    def price(self, basis):
        return self.close if basis == "close" else self.hlc3

    def er(self, basis, n):
        key = (basis, n)
        if key not in self._er:
            self._er[key] = efficiency_ratio(pd.Series(self.price(basis)), n).values
        return self._er[key]


def build_books(resolved, line, params=None, verbose=True):
    sel = labels_mod.fit_set(resolved, line)
    prices = load_prices(sorted(sel["yf"].unique()), params=params, verbose=verbose)
    books, skipped = [], []
    for sym, grp in sel.groupby("yf"):
        px = prices.get(sym)
        if px is None:
            skipped.append(sym)
            continue
        ohlc = px.dropna(subset=["Close"])
        pos = {d: i for i, d in enumerate(ohlc.index)}
        rows = [(pos[r.close_date], float(getattr(r, "ss_" + line)), r.ticker)
                for r in grp.itertuples() if r.close_date in pos]
        if rows:
            books.append(TickerBook(sym, ohlc, rows))
        else:
            skipped.append(sym)
    if verbose and skipped:
        print("[fit] skipped (no usable history): " + ", ".join(skipped))
    return books


def _series(book, family, basis, cfg):
    px = book.price(basis)
    if family == "sma":
        return sma(px, cfg["window"])
    if family == "ema":
        return ema(px, cfg["span"])
    if family == "wma":
        return wma(px, cfg["window"])
    if family == "kama":
        return kama_full(px, book.er(basis, cfg["n"]), cfg["n"], cfg["fast"], cfg["slow"])
    if family == "aema":
        return aema_full(px, book.er(basis, cfg["n"]), cfg["span_fast"], cfg["span_slow"])
    raise ValueError(family)


def _score(books, family, basis, cfg, lag):
    err = []
    for b in books:
        s = _series(b, family, basis, cfg)
        for i, lab, _ in b.rows:
            j = i - lag
            if j < 0:
                continue
            v = s[j]
            if np.isfinite(v):
                err.append(v / lab - 1.0)
    if not err:
        return np.inf, 0
    e = np.asarray(err)
    return float(np.sum(e ** 2)), len(e)


def _grid(family, wide):
    if family in ("sma", "wma"):
        rng = range(3, 401) if wide else range(3, 201)
        key = "window"
        return [{key: w} for w in rng]
    if family == "ema":
        rng = range(3, 401) if wide else range(3, 201)
        return [{"span": s} for s in rng]
    if family == "kama":
        return [{"n": n, "fast": f, "slow": s}
                for n in KAMA_N for f in KAMA_FAST for s in KAMA_SLOW if s > f + 4]
    if family == "aema":
        fasts = [3, 5, 8, 12, 16, 20, 25, 30, 40, 50, 65, 80]
        slows = [20, 30, 40, 50, 65, 80, 100, 125, 150, 200, 260]
        return [{"n": n, "span_fast": f, "span_slow": s}
                for n in KAMA_N for f in fasts for s in slows if s > f]
    raise ValueError(family)


def search(books, families, bases, lags, wide=False, verbose=True):
    """Full sweep over family x price-basis x parameter x lag."""
    results = []
    for family in families:
        grid = _grid(family, wide)
        for basis in bases:
            for lag in lags:
                best = (np.inf, None, 0)
                for cfg in grid:
                    sse, k = _score(books, family, basis, cfg, lag)
                    if sse < best[0]:
                        best = (sse, cfg, k)
                if best[1] is None:
                    continue
                results.append({
                    "family": family, "price": basis, "lag": lag,
                    "params": best[1], "sse": best[0], "n_obs": best[2],
                    "rmse_pct": 100 * np.sqrt(best[0] / max(best[2], 1)),
                })
        if verbose:
            top = min((r for r in results if r["family"] == family),
                      key=lambda r: r["sse"], default=None)
            if top:
                print("  %-5s best: %s price=%s lag=%d  rmse=%.2f%%"
                      % (family, top["params"], top["price"], top["lag"], top["rmse_pct"]))
    return sorted(results, key=lambda r: r["sse"])


def residuals(books, spec):
    rows = []
    for b in books:
        s = _series(b, spec["family"], spec["price"], spec["params"])
        for i, lab, tk in b.rows:
            j = i - spec["lag"]
            v = s[j] if j >= 0 else np.nan
            rows.append({"ticker": tk, "date": b.index[i].date().isoformat(),
                         "label": lab, "model": v,
                         "err_pct": 100 * (v / lab - 1.0) if np.isfinite(v) else np.nan})
    out = pd.DataFrame(rows)
    return out.reindex(out.err_pct.abs().sort_values(ascending=False).index)


def main():
    ap = argparse.ArgumentParser(description="Fit TRADE/TREND lines to published labels.")
    ap.add_argument("--lines", default="trade,trend")
    ap.add_argument("--families", default="sma,ema,wma,kama")
    ap.add_argument("--price", default="close,hlc3")
    ap.add_argument("--max-lag", type=int, default=0,
                    help="sweep model lag 0..N bars behind the last completed bar")
    ap.add_argument("--wide", action="store_true", help="extend window search to 400 bars")
    ap.add_argument("--write", action="store_true", help="save winners to config/params.yaml")
    args = ap.parse_args()

    params = load_params()
    resolved = labels_mod.resolve(params=params, verbose=True)
    print("[fit] %d/%d label rows verified\n" % (int(resolved["ok"].sum()), len(resolved)))

    families = [s.strip() for s in args.families.split(",") if s.strip()]
    bases = [s.strip() for s in args.price.split(",") if s.strip()]
    lags = list(range(0, args.max_lag + 1))

    fitted = {}
    for line in [s.strip() for s in args.lines.split(",") if s.strip()]:
        books = build_books(resolved, line, params=params, verbose=True)
        nobs = sum(len(b.rows) for b in books)
        print("=== %s  (%d tickers, %d labels) ===" % (line.upper(), len(books), nobs))
        if nobs < 3:
            print("  too few labels to fit\n")
            continue

        results = search(books, families, bases, lags, wide=args.wide)
        best = results[0]
        print("\n  WINNER: %s %s price=%s lag=%d  rmse=%.2f%%"
              % (best["family"], best["params"], best["price"], best["lag"], best["rmse_pct"]))
        print("  runners-up:")
        for r in results[1:5]:
            print("    %-5s %-34s price=%-5s lag=%d  rmse=%.2f%%"
                  % (r["family"], r["params"], r["price"], r["lag"], r["rmse_pct"]))

        res = residuals(books, best)
        print("\n  per-label residuals:")
        print(res.to_string(index=False, float_format=lambda v: "%.3f" % v))
        print("  median |err| = %.2f%%   max |err| = %.2f%%\n"
              % (res.err_pct.abs().median(), res.err_pct.abs().max()))
        fitted[line] = best

    if args.write and fitted:
        for line, spec in fitted.items():
            params["lines"][line].update(spec["params"])
            params["lines"][line]["family"] = spec["family"]
            params["lines"][line]["price"] = spec["price"]
            params["lines"][line]["lag"] = spec["lag"]
            params["lines"][line]["fit_rmse_pct"] = round(spec["rmse_pct"], 3)
        params["meta"]["fitted_on"] = pd.Timestamp.today().date().isoformat()
        params["meta"]["n_labels"] = int(resolved["ok"].sum())
        params["meta"]["notes"] = "fitted by calib/fit_lines.py"
        save_params(params)
        print("[fit] wrote config/params.yaml")


if __name__ == "__main__":
    main()
