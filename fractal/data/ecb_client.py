"""Daily yield series from the ECB Data Portal, for instruments Yahoo does not carry.

European government yields are not on the Yahoo feed at all -- every candidate symbol
404s, and what does resolve are bond ETF *prices*, which move inversely to yield and
would read backwards beside UST10Y.

The ECB Data Portal is public, keyless and documented, which makes it a better
dependency than scraping a rendered quote page. It returns a yield rather than a
price, so the series means the same thing as our Treasury rows: bullish TREND is
yields rising.

One honest caveat, carried in the display name. The daily series is the euro area
AAA-rated curve, which is dominated by German bunds but is not the Bund itself. It
has run 5-11bp above the German 10-year over the last twelve months, a stable offset
averaging 8.5bp. ECB do publish a German-specific long-term rate, but only monthly,
which is useless for a daily range. The level is therefore a few basis points high;
the dynamics, which is what the range is built from, are the same.
"""
from __future__ import annotations

import csv
import io
import urllib.request

import pandas as pd

BASE = "https://data-api.ecb.europa.eu/service/data"

# display ticker -> (ECB series key, human name)
SERIES = {
    "EU10Y": ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y",
              "Euro Area AAA 10-Year Yield"),
    "EU2Y":  ("YC/B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y",
              "Euro Area AAA 2-Year Yield"),
}


def _fetch_one(key, n=2000, timeout=30):
    url = "%s/%s?format=csvdata&lastNObservations=%d" % (BASE, key, n)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    txt = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(txt)))
    if not rows:
        return None
    idx = pd.to_datetime([r["TIME_PERIOD"] for r in rows])
    val = pd.to_numeric([r["OBS_VALUE"] for r in rows], errors="coerce")
    s = pd.Series(val, index=idx).dropna().sort_index()
    if s.empty:
        return None
    # A yield has no open, high, low or volume. The model reads Close everywhere and
    # falls back to Close when Open is absent, so giving the same number for the OHLC
    # columns is honest rather than invented: there is one observation per day.
    return pd.DataFrame({"Open": s, "High": s, "Low": s, "Close": s, "Volume": 0.0})


def fetch(tickers, verbose=False):
    """{ticker: frame} for whichever of `tickers` this source knows. Never raises --
    a source that is down should cost one row on the page, not the whole build."""
    out = {}
    for t in tickers:
        spec = SERIES.get(t)
        if not spec:
            continue
        try:
            df = _fetch_one(spec[0])
        except Exception as e:
            if verbose:
                print("[ecb] %s unavailable: %s" % (t, e))
            continue
        if df is not None and len(df) > 200:
            out[t] = df
    return out
