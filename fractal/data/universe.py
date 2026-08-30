"""Hedgeye-native watchlist (build spec section 6) plus helpers."""

UNIVERSE = {
    "index":    ["SPY", "QQQ", "IWM"],
    "sector":   ["XLK", "XLF", "XLV", "XLE", "XLB", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC"],
    "intl":     ["EEM", "EWJ", "FXI"],
    "fx":       ["UUP", "FXE", "FXB", "FXY"],
    "commod":   ["GLD", "SLV", "USO", "UNG", "CORN", "SOYB", "CPER"],
    "rates":    ["TLT", "IEF", "LQD", "EMB", "HYG"],
    "megacap":  ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],
}

# IBKR contract ids confirmed in the build spec. Anything not listed is resolved
# at runtime by exact-symbol match on the US primary listing.
IB_CONIDS = {
    "NVDA": 4815747,
    "SLV": 39039301,
    "TLT": 15547841,
    "XLF": 4215220,
    "XLV": 4215205,
}


def all_tickers():
    """Flat, de-duplicated universe in declaration order."""
    seen, out = set(), []
    for group in UNIVERSE.values():
        for t in group:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def group_of(ticker):
    for name, members in UNIVERSE.items():
        if ticker in members:
            return name
    return "other"


# Yahoo uses different symbols for indices/futures/crypto than the label sheet.
YF_SYMBOL = {
    "VIX": "^VIX",
    "MOVE": "^MOVE",
    "WTI": "CL=F",
    "BTC": "BTC-USD",
}


def yf_symbol(ticker: str) -> str:
    return YF_SYMBOL.get(ticker, ticker)
