"""Single entry point for price history: IB when available, Yahoo otherwise."""
from __future__ import annotations

import os
import yaml

from . import fred_client
from . import yahoo_client
from .ib_client import IBUnavailable

_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "config", "params.yaml")


def load_params(path: str = _CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_params(params: dict, path: str = _CONFIG) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(params, fh, sort_keys=False, default_flow_style=False)


def repo_path(*parts) -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, *parts)


def load_prices(tickers, params: dict | None = None, source: str | None = None,
                verbose: bool = True) -> dict:
    """{ticker: OHLCV DataFrame}. Falls back to Yahoo if IB is unreachable."""
    params = params or load_params()
    cfg = params.get("data", {})
    source = source or cfg.get("source", "yahoo")
    years = int(cfg.get("history_years", 5))
    cache_dir = repo_path(cfg.get("cache_dir", "cache"))

    if source == "ib":
        try:
            from . import ib_client
            return ib_client.fetch(tickers, years=years)
        except IBUnavailable as e:
            if verbose:
                print(f"[loader] IB unavailable ({e}); falling back to Yahoo.")

    # FRED-sourced names are split out and merged back. Yahoo has no honest cash
    # 2-year, so UST2Y is served from FRED DGS2 -- see fred_client for why.
    fred = [t for t in tickers if fred_client.is_fred(t)]
    rest = [t for t in tickers if not fred_client.is_fred(t)]
    out = yahoo_client.fetch(rest, years=years, cache_dir=cache_dir) if rest else {}
    if fred:
        out.update(fred_client.fetch(fred, years=years, cache_dir=cache_dir,
                                     verbose=verbose))
    return out

def next_session(asof):
    """The trading day that levels computed from the `asof` close apply to.

    A close fixes the levels and the levels govern the session that follows, so the
    report is dated by that session rather than by the bar it came from: a Monday
    newsletter carries Monday's levels, computed off Friday's close.

    Weekends are skipped; exchange holidays are not modelled, so a holiday Monday
    would still be dated to that Monday rather than rolled to Tuesday.
    """
    import datetime as _dt
    import pandas as _pd
    try:
        d = _pd.Timestamp(asof).date()
    except Exception:
        return None
    d += _dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += _dt.timedelta(days=1)
    return d
