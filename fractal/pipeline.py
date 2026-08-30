"""Daily pipeline: prices -> RANGE + TRADE/TREND/TAIL -> state -> scan table."""
from __future__ import annotations

import pandas as pd

from .data.loader import load_params, load_prices
from .data.universe import all_tickers, group_of
from .model import adaptive_ma, range_ewma, state as state_mod
from .model.persistence import texture


def compute_one(ticker: str, ohlc: pd.DataFrame, params: dict,
                spot: float | None = None, with_texture: bool = False) -> dict:
    close = ohlc["Close"].dropna()
    lines = adaptive_ma.compute(close, params)
    rng = range_ewma.compute(close, params, volume=ohlc.get("Volume"))
    row = state_mod.snapshot(ticker, ohlc, lines, rng, spot=spot)
    row["group"] = group_of(ticker)
    if with_texture and len(close) > 260:
        row.update({k: (None if pd.isna(v) else round(float(v), 3))
                    for k, v in texture(close).items()})
    return row


def run(tickers=None, params: dict | None = None, source: str | None = None,
        with_texture: bool = True, verbose: bool = True) -> pd.DataFrame:
    params = params or load_params()
    tickers = tickers or all_tickers()
    prices = load_prices(tickers, params=params, source=source, verbose=verbose)

    rows, missing = [], []
    for t in tickers:
        df = prices.get(t)
        if df is None or len(df) < 80:
            missing.append(t)
            continue
        rows.append(compute_one(t, df, params, with_texture=with_texture))

    if verbose and missing:
        print(f"[pipeline] no usable history for: {', '.join(missing)}")

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["group", "ticker"]).reset_index(drop=True)
    return out


def series_for(ticker: str, params: dict | None = None, source: str | None = None) -> pd.DataFrame:
    """Full per-bar history of levels for one ticker — used by calib and the chart view."""
    params = params or load_params()
    df = load_prices([ticker], params=params, source=source, verbose=False).get(ticker)
    if df is None:
        raise ValueError(f"no data for {ticker}")
    close = df["Close"].dropna()
    lines = adaptive_ma.compute(close, params)
    rng = range_ewma.compute(close, params, volume=df.get("Volume"))
    out = df.join(lines).join(rng[["sigma", "center", "range_low", "range_high"]])
    st = state_mod.state_series(close, lines)
    out = out.join(st)
    out["phase"] = state_mod.fractal_phase(st)
    return out
