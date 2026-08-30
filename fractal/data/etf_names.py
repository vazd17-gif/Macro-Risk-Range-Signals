"""Fund names for the ETF watchlist, cached to disk.

Names come from the data provider once and are cached in `data/etf_names.csv`,
because looking them up is far slower than pulling prices and they almost never
change. `refresh()` re-fetches; everything else reads the cache.

Provider names are long and repetitive ("Invesco S&P 500 Equal Weight ETF",
"iShares MSCI Japan Value ETF"). `shorten()` trims the boilerplate that adds no
information beside a ticker -- the issuer prefix, the ETF/Fund/Trust suffix, and
registered-mark symbols -- so the name fits on one line next to the symbol.
"""
from __future__ import annotations

import os
import re

import pandas as pd

from .loader import repo_path
from .etf_universe import all_etfs

CACHE = "etf_names.csv"

# Each trailing word must be preceded by whitespace. Without that requirement the
# "Shares" branch matches inside CoinShares, CurrencyShares and AdvisorShares and
# silently truncates the issuer name.
_SUFFIXES = re.compile(
    r"(?:\s+(?:ETF|ETV|Fund|Trust|Shares|Portfolio|Inc\.?|plc)\.?)+\s*$", re.I)

# Tails like ", LP" and "- ETF Class Shares" that survive the suffix pass.
_TAIL_JUNK = re.compile(
    r"(?:\s*[-,]\s*(?:LP|New|(?:ETF\s+)?Class(?:\s+Shares)?)\.?)+\s*$", re.I)

_PREFIXES = re.compile(
    r"^(?:iShares|State Street SPDR|State Street|SPDR|Invesco|Vanguard|Global X|"
    r"VanEck|First Trust|Direxion|ProShares|Roundhill|Amplify|abrdn Physical|abrdn|"
    r"Franklin|Schwab|WisdomTree|Xtrackers|JPMorgan|Goldman Sachs|Fidelity|"
    r"KraneShares|Simplify|Defiance|Tema|United States|Teucrium|Sprott|"
    r"AdvisorShares|CoinShares|Alerian|Hilton|Fundstrat)\s+", re.I)

_MARKS = re.compile("[®™]")
_ELLIPSIS = "…"


def shorten(name, ticker="", keep_issuer=False, limit=40):
    """Trim provider boilerplate so the name reads well beside a ticker.

    Returns "" when nothing informative survives -- a name that reduces to the
    ticker itself (Invesco QQQ Trust -> QQQ) adds nothing next to the symbol.
    """
    if not name:
        return ""
    s = _MARKS.sub("", str(name)).strip()
    s = _TAIL_JUNK.sub("", s)
    if not keep_issuer:
        s = _PREFIXES.sub("", s)
    s = _SUFFIXES.sub("", s)
    s = _TAIL_JUNK.sub("", s)
    s = _SUFFIXES.sub("", s)
    s = re.sub(r"\s{2,}", " ", s).strip(" -,")
    if ticker and s.upper() == ticker.upper():
        return ""
    if len(s) > limit:                    # trim on a word boundary, never mid-word
        cut = s[:limit].rsplit(" ", 1)[0].rstrip(" -,&")
        s = (cut or s[:limit].rstrip()) + _ELLIPSIS
    return s


def path():
    return repo_path("data", CACHE)


def load():
    """{ticker: raw provider name}. Empty dict if the cache does not exist yet."""
    p = path()
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["ticker"], df["name"].fillna("")))


def short_names():
    """{ticker: display name}, ready to render."""
    return {t: shorten(n, t) for t, n in load().items()}


def refresh(tickers=None, verbose=True):
    """Fetch names for `tickers` (default: the whole watchlist) and update the cache."""
    import warnings
    warnings.filterwarnings("ignore")
    import yfinance as yf

    tickers = tickers or all_etfs()
    have = load()
    out = dict(have)
    missed = []
    for t in tickers:
        if have.get(t):
            continue
        name = ""
        try:
            info = yf.Ticker(t).get_info()
            name = info.get("longName") or info.get("shortName") or ""
        except Exception:
            pass
        if name:
            out[t] = name
        else:
            missed.append(t)
    pd.DataFrame(sorted(out.items()), columns=["ticker", "name"]).to_csv(path(), index=False)
    if verbose:
        print("names cached: %d/%d" % (len(out), len(tickers)))
        if missed:
            print("no name for: " + ", ".join(missed))
    return out


if __name__ == "__main__":
    names = refresh()
    for t in sorted(names):
        print("  %-6s %s" % (t, shorten(names[t], t)))
