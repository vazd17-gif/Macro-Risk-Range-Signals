"""Single entry point for price history: IB when available, Yahoo otherwise."""
from __future__ import annotations

import os
import yaml

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

    return yahoo_client.fetch(tickers, years=years, cache_dir=cache_dir)
