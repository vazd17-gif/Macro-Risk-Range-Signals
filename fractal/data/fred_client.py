"""FRED (St. Louis Fed) daily series, for rates Yahoo does not quote honestly.

Added for the 2-year. Yahoo has no cash 2Y index: ^FVX is the FIVE-year, and the
CBOT micro yield future 2YY=F -- which replaced it -- turned out to be barely
quoted. Over 60 sessions it printed zero change on 30 of them, sat pinned at
4.1700 for nine consecutive days, and its daily changes correlated 0.110 with the
5-year. On 2026-09-02 it read 4.2000 against a true 4.3609, understating the
2s10s spread by 19bp.

DGS2 is the constant-maturity series behind the published curve: 0.892 correlation
with the 5-year on daily changes, 6 zero-change days in 60, and history to 1976.

The series carries one observation per day, so Open/High/Low are set to the close.
Nothing downstream reads them for these names -- the range is built from closes
and the hedgeye_anchor profile does not use volume.
"""
from __future__ import annotations

import io
import os
import urllib.request

import pandas as pd

PREFIX = "FRED:"
_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=%s"
_TIMEOUT = 25


def is_fred(symbol: str) -> bool:
    return str(symbol).upper().startswith(PREFIX)


def series_id(symbol: str) -> str:
    return str(symbol)[len(PREFIX):]


def _read(series: str) -> pd.Series:
    raw = urllib.request.urlopen(_URL % series, timeout=_TIMEOUT).read().decode()
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c)
    val_col = next(c for c in df.columns if c != date_col)
    out = df.set_index(pd.to_datetime(df[date_col]))[val_col]
    # FRED writes "." for a non-publication day (holidays); coerce and drop.
    return pd.to_numeric(out, errors="coerce").dropna()


# H.15 publishes about a business day behind the tape, so the raw series is a day
# short of the equity close the rest of the book is priced on. Left alone, UST2Y
# trips the "one report, one date" staleness guard EVERY day and the 2s10s spread
# gets computed across mismatched dates. The last published rate is carried forward
# to the current session -- what any desk would quote intraday -- capped at
# MAX_CARRY business days so a genuinely broken feed still surfaces as stale rather
# than quietly repeating itself, which is the failure 2YY=F just cost us.
#
# The carried bar adds one zero-return day to a 1,249-bar series, so its effect on
# the EWMA sigma is immaterial and unwinds as soon as the real print lands.
MAX_CARRY = 3


def _carry_forward(s: pd.Series, verbose: bool, sid: str) -> pd.Series:
    today = pd.Timestamp.today().normalize()
    gap = len(pd.bdate_range(s.index[-1], today)) - 1
    if gap <= 0 or gap > MAX_CARRY:
        if gap > MAX_CARRY and verbose:
            print("[fred] %s last print %s is %d business days old -- NOT carried "
                  "forward; it will show as stale" % (sid, s.index[-1].date(), gap))
        return s
    idx = pd.bdate_range(s.index[-1] + pd.offsets.BDay(1), today)
    if verbose:
        print("[fred] %s carried %s forward %d business day(s) to %s"
              % (sid, s.index[-1].date(), len(idx), idx[-1].date()))
    return pd.concat([s, pd.Series(float(s.iloc[-1]), index=idx)])


def fetch(symbols, years: int = 5, cache_dir: str | None = None,
          verbose: bool = True) -> dict:
    """{symbol: OHLCV frame} for FRED: symbols. Falls back to cache on failure."""
    out = {}
    for sym in symbols:
        sid = series_id(sym)
        cache = os.path.join(cache_dir, "FRED_%s.csv" % sid) if cache_dir else None
        s = None
        try:
            s = _read(sid)
            if cache:
                os.makedirs(cache_dir, exist_ok=True)
                s.to_csv(cache, header=["value"])
        except Exception as exc:
            if verbose:
                print("[fred] %s fetch failed (%s); trying cache"
                      % (sid, type(exc).__name__))
            if cache and os.path.exists(cache):
                c = pd.read_csv(cache, index_col=0, parse_dates=True)
                s = pd.to_numeric(c.iloc[:, 0], errors="coerce").dropna()
        if s is None or not len(s):
            if verbose:
                print("[fred] %s unavailable and no cache -- skipped" % sid)
            continue
        if years:
            s = s[s.index >= s.index[-1] - pd.DateOffset(years=years)]
        s = _carry_forward(s, verbose, sid)
        out[sym] = pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s,
                                 "Volume": 0.0}, index=s.index)
    return out
