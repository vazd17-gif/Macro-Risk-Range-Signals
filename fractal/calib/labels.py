"""Load and validate the reference-level ground truth (build spec section 5).

Each row is one published Similar Set / Hedgeye level for one ticker on one
chart date. `prior_close` is the last completed daily bar feeding that chart —
levels are quoted intraday off the prior close, so the model must be evaluated
at that index, not at the chart date.

Rows carry `chart_last` and `chart_chg` (the price and daily change printed on
the source chart). Those exist purely to *verify the ticker mapping*: the implied
prior close (chart_last - chart_chg) must agree with the price history, which
catches a mislabelled symbol before it poisons a fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.loader import load_prices, repo_path
from ..data.universe import yf_symbol

LABELS_CSV = repo_path("data", "labels.csv")


def load(path: str = LABELS_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["chart_date"] = pd.to_datetime(df["chart_date"])
    if "fit" not in df:
        df["fit"] = 1
    df["fit"] = df["fit"].fillna(1).astype(int)
    return df


def resolve(df: pd.DataFrame | None = None, params: dict | None = None,
            tol_pct: float = 0.75, verbose: bool = True) -> pd.DataFrame:
    """Attach the price history index each label should be evaluated at.

    Adds:
      yf          resolved Yahoo symbol
      close_date  the completed bar the levels were computed from
      close       that bar's close
      implied     chart_last - chart_chg, when both are present
      check_pct   100 * (implied/close - 1); large => wrong symbol or wrong date
      ok          check passed (or nothing to check against)
    """
    df = (df if df is not None else load()).copy()
    syms = sorted({yf_symbol(t) for t in df["ticker"]})
    prices = load_prices(syms, params=params, verbose=verbose)

    rows = []
    for _, r in df.iterrows():
        sym = yf_symbol(r["ticker"])
        px = prices.get(sym)
        rec = dict(r)
        rec["yf"] = sym
        rec["close_date"] = pd.NaT
        rec["close"] = np.nan
        rec["implied"] = np.nan
        rec["check_pct"] = np.nan
        rec["ok"] = False

        if px is None or len(px) == 0:
            rec["note"] = "no price history"
            rows.append(rec)
            continue

        # Last completed bar strictly before the chart date. If prior_close was
        # supplied in the sheet, prefer the bar that matches it.
        hist = px[px.index < r["chart_date"]]
        if len(hist) == 0:
            rec["note"] = "chart date precedes history"
            rows.append(rec)
            continue
        idx = hist.index[-1]
        if pd.notna(r.get("prior_close")):
            diffs = (hist["Close"] - float(r["prior_close"])).abs()
            if diffs.min() <= 0.02 * abs(float(r["prior_close"])):
                idx = diffs.idxmin()

        rec["close_date"] = idx
        rec["close"] = float(px.loc[idx, "Close"])

        if pd.notna(r.get("chart_last")) and pd.notna(r.get("chart_chg")):
            rec["implied"] = float(r["chart_last"]) - float(r["chart_chg"])
            rec["check_pct"] = 100.0 * (rec["implied"] / rec["close"] - 1.0)
            rec["ok"] = abs(rec["check_pct"]) <= tol_pct
        elif pd.notna(r.get("prior_close")):
            rec["check_pct"] = 100.0 * (float(r["prior_close"]) / rec["close"] - 1.0)
            rec["ok"] = abs(rec["check_pct"]) <= tol_pct
        else:
            rec["ok"] = True          # nothing to verify against
            rec["note"] = "unverified"
        rows.append(rec)

    return pd.DataFrame(rows)


def fit_set(resolved: pd.DataFrame, line: str) -> pd.DataFrame:
    """Rows usable for fitting one line: flagged for fit, verified, and labelled."""
    col = f"ss_{line}"
    sel = resolved[(resolved["fit"] == 1) & resolved["ok"] &
                   resolved[col].notna() & resolved["close"].notna()]
    return sel.copy()


if __name__ == "__main__":
    r = resolve()
    cols = ["ticker", "yf", "chart_date", "close_date", "close", "implied",
            "check_pct", "ok", "ss_trade", "ss_trend"]
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(r[cols].to_string(index=False))
    bad = r[~r["ok"]]
    print(f"\n{len(r) - len(bad)}/{len(r)} rows verified.")
    if len(bad):
        print("FAILED:", ", ".join(f"{b.ticker}({b.check_pct:.2f}%)" if pd.notna(b.check_pct)
                                   else f"{b.ticker}(no data)" for b in bad.itertuples()))
