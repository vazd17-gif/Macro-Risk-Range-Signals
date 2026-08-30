"""Yahoo daily OHLCV with an on-disk parquet/csv cache.

Fallback source (build spec section 2). Fine for prototyping and for the RANGE.
Occasionally rate-limited, hence the cache and the batched download.
"""
from __future__ import annotations

import os
import time
import datetime as dt

import pandas as pd

COLS = ["Open", "High", "Low", "Close", "Volume"]


def _cache_path(cache_dir: str, ticker: str) -> str:
    return os.path.join(cache_dir, f"{ticker.replace('/', '_')}.csv")


def _read_cache(cache_dir: str, ticker: str):
    p = _cache_path(cache_dir, ticker)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df if len(df) else None


def _write_cache(cache_dir: str, ticker: str, df: pd.DataFrame) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    df.to_csv(_cache_path(cache_dir, ticker))


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise a yfinance frame to Open/High/Low/Close/Volume, drop unfinished bars."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns=str.title)
    keep = [c for c in COLS if c in df.columns]
    df = df[keep]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # A bar with no Close is the in-progress session; the model works off
    # completed daily bars only.
    df = df.dropna(subset=["Close"])
    return df


def fetch(tickers, years: int = 5, cache_dir: str = "cache",
          use_cache: bool = True, max_age_hours: float = 12.0) -> dict:
    """Return {ticker: DataFrame(Open,High,Low,Close,Volume)} indexed by date."""
    import yfinance as yf

    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(dict.fromkeys(tickers))

    out, need = {}, []
    now = time.time()
    for t in tickers:
        if use_cache:
            p = _cache_path(cache_dir, t)
            if os.path.exists(p) and (now - os.path.getmtime(p)) < max_age_hours * 3600:
                df = _read_cache(cache_dir, t)
                if df is not None:
                    out[t] = df
                    continue
        need.append(t)

    if need:
        start = (dt.date.today() - dt.timedelta(days=int(365.25 * years) + 10)).isoformat()
        raw = yf.download(need, start=start, interval="1d", auto_adjust=False,
                          progress=False, group_by="ticker", threads=True)
        for t in need:
            try:
                sub = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = _tidy(sub)
                if len(df) == 0:
                    raise ValueError("empty")
                out[t] = df
                _write_cache(cache_dir, t, df)
            except Exception:
                cached = _read_cache(cache_dir, t)          # stale cache beats nothing
                if cached is not None:
                    out[t] = cached
    return out
