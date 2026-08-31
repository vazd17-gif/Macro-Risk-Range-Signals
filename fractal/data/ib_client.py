"""Interactive Brokers daily OHLCV via ib_insync (build spec section 2, primary source).

Requires TWS or IB Gateway running and reachable, with the relevant market-data
subscriptions enabled. `ib_insync` is an optional dependency: if it is missing or
the gateway is down, `fetch` raises IBUnavailable and callers fall back to Yahoo.
"""
from __future__ import annotations

import pandas as pd

from .universe import IB_CONIDS

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7497          # 7497 paper TWS / 7496 live TWS / 4002 paper GW / 4001 live GW
DEFAULT_CLIENT_ID = 17


class IBUnavailable(RuntimeError):
    """Gateway not reachable, ib_insync not installed, or contract unresolvable."""


def _connect(host, port, client_id, timeout=8):
    try:
        from ib_insync import IB
    except ImportError as e:
        raise IBUnavailable("ib_insync is not installed (pip install ib_insync)") from e
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id, timeout=timeout, readonly=True)
    except Exception as e:
        raise IBUnavailable(f"could not connect to IB at {host}:{port} — {e}") from e
    return ib


def _resolve(ib, ticker: str):
    """Resolve the US primary listing by exact symbol match."""
    from ib_insync import Stock, Contract

    if ticker in IB_CONIDS:
        c = Contract(conId=IB_CONIDS[ticker], exchange="SMART")
        details = ib.reqContractDetails(c)
        if details:
            return details[0].contract

    for exchange in ("SMART", "ARCA", "NASDAQ", "NYSE"):
        c = Stock(ticker, exchange, "USD")
        details = ib.reqContractDetails(c)
        # exact symbol match only — avoids picking up a foreign or derivative listing
        exact = [d for d in details if d.contract.symbol == ticker]
        if exact:
            return exact[0].contract
    raise IBUnavailable(f"could not resolve {ticker} to a US primary listing")


def fetch(tickers, years: int = 5, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
          client_id: int = DEFAULT_CLIENT_ID, what: str = "TRADES") -> dict:
    """Return {ticker: DataFrame(Open,High,Low,Close,Volume)} indexed by date."""
    if isinstance(tickers, str):
        tickers = [tickers]

    ib = _connect(host, port, client_id)
    out = {}
    try:
        duration = f"{max(1, int(years))} Y"
        for t in tickers:
            try:
                contract = _resolve(ib, t)
            except IBUnavailable:
                continue
            bars = ib.reqHistoricalData(
                contract, endDateTime="", durationStr=duration,
                barSizeSetting="1 day", whatToShow=what,
                useRTH=True, formatDate=1,
            )
            if not bars:
                continue
            df = pd.DataFrame(
                [{"Date": b.date, "Open": b.open, "High": b.high,
                  "Low": b.low, "Close": b.close, "Volume": b.volume} for b in bars]
            )
            df["Date"] = pd.to_datetime(df["Date"])
            out[t] = df.set_index("Date").sort_index()
    finally:
        ib.disconnect()

    if not out:
        raise IBUnavailable("IB returned no data for any requested ticker")
    return out
