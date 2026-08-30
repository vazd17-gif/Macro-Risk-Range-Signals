"""The watchlist for the Risk Range dashboard and newsletter.

Mostly ETFs, plus the highest dollar-volume S&P 500 single names. They are kept in
separate lists because they are maintained separately, but the report treats them
identically -- the model does not care whether a symbol is a fund or a stock.

Kept separate from `universe.py` (the Hedgeye-native macro list) because this is a
user-supplied working list that changes independently.

`ML PX` in the source list is a typo for MLPX (Global X MLP & Energy
Infrastructure). It is corrected here rather than silently dropped.

GXG in the source list is carried as COLO: Global X MSCI Colombia stopped printing
under GXG on 2026-07-17 and now trades as COLO, which is also the symbol Hedgeye
and Similar Set use for it.
"""
from __future__ import annotations

RAW = """
AAAU, AQWA, ARKG, ARKK, ARKQ, BBN, BDRY, BNO, BUXX, BWET, CANE, CIBR, CLOX,
CLOZ, COPX, CORN, CPER, DBC, DESK, DRAM, DUST, EDEN, EEM, EIS, EPU, EPHE, EWA,
EWC, EWG, EWH, EWI, EWJ, EWJV, EWN, EWO, EWQ, EWS, EWY, EWZ, EWW, EZA, EFNL,
EMXC, ENZL, FCG, FUTY, FXB, FXC, FXE, FXI, FXY, GDX, GDXJ, GII, GLD, GLIN,
GREK, GRNY, COLO, HBDC, HYG, IAK, IBIT, IEF, IGV, IHF, IIGD, INDA, INDY,
ITA, ITB, IWC, IWO, IWM, IVOL, IVES, IDX, JETS, JNK, JPXN, KEMX, KRE, KSA,
KWEB, KWT, LIT, LQD, MAGS, MTBA, NIKL, NLR, OIH, PALL, PBD, PINK, PJP,
PPLT, PSP, PSCU, PSCC, PSCD, PSCH, QQQ, QTUM, RSP, SCJ, SHY, SIL, SILJ,
SKYY, SLV, SLX, SMH, SMIN, SPMO, SOYB, SPLV, SPY, TBIL, TAN, TLT, TUR, UAE,
UGA, UNG, UUP, URA, USO, WEAT, WGMI, WOOD, XHE, XLF, XLI, XLK, XLP, XLU,
XLRE, XLV, XLY, XLG, XOP, XRT, XTL, YCS
"""

# Single names: the highest daily dollar-volume S&P 500 constituents, plus any
# individually requested symbol. SBSW (Sibanye Stillwater) is one of the latter --
# a NYSE-listed South African PGM miner, not an index member. Berkshire is carried
# as BRK-B: the data feed does not recognise the BRK.B form.
STOCKS = """
NVDA, AAPL, MSFT, AMZN, GOOGL, GOOG, AVGO, META, TSLA, BRK-B, MU, LLY, JPM, WMT,
AMD, V, JNJ, XOM, MA, INTC, ABBV, PLTR, BAC, ORCL, CSCO, COST, CVX, KO, LRCX, CAT,
AMAT, SNDK, SBSW
"""


# Volatility indices. Not tradeable, so they never raise a buy/sell signal -- they
# are carried for market context. Bearish TRADE and TREND on both is falling
# volatility, which is supportive for risk assets.
INDICES = """
VIX, MOVE
"""

# Display ticker -> data-feed symbol, where they differ.
YF_MAP = {"VIX": "^VIX", "MOVE": "^MOVE"}


def yf_symbol(ticker):
    return YF_MAP.get(ticker, ticker)


def is_index(ticker):
    return ticker in _parse(INDICES)


# Grouping for the dashboard and newsletter. Anything unlisted falls to "other".
GROUPS = {
    "us_equity":    ["SPY", "QQQ", "RSP", "IWM", "IWC", "IWO", "MAGS", "SPLV", "SPMO",
                     "XLG", "GRNY"],
    "us_sector":    ["XLF", "XLI", "XLK", "XLP", "XLU", "XLRE", "XLV", "XLY", "XRT",
                     "XTL", "XHE", "XOP", "ITA", "ITB", "IAK", "IHF", "PJP", "KRE",
                     "JETS", "OIH", "PSP", "DESK", "FUTY"],
    "us_smallcap":  ["PSCU", "PSCC", "PSCD", "PSCH"],
    "thematic":     ["ARKG", "ARKK", "ARKQ", "CIBR", "SKYY", "QTUM", "DRAM",
                     "LIT", "TAN", "URA", "NLR", "NIKL", "IVES", "WGMI", "SMH", "IGV",
                     "AQWA", "PBD", "PINK", "WOOD", "SLX", "GII"],
    "intl":         ["EEM", "EMXC", "KEMX", "EWA", "EWC", "EWG", "EWH", "EWI", "EWJ", "EWJV",
                     "EWN", "EWO", "EWQ", "EWS", "EWY", "EWZ", "EWW", "EZA", "EFNL",
                     "EDEN", "EIS", "EPU", "EPHE", "ENZL", "GREK",
                     "COLO", "GLIN", "INDA", "INDY", "SMIN", "IDX", "JPXN", "SCJ",
                     "KSA", "KWT", "TUR", "UAE", "FXI", "KWEB"],
    "commodity":    ["GLD", "AAAU", "SLV", "SIL", "SILJ", "GDX", "GDXJ", "DUST",
                     "PALL", "PPLT", "CPER", "COPX", "DBC", "USO", "BNO", "UGA",
                     "UNG", "FCG", "CORN", "WEAT", "SOYB", "CANE", "BDRY", "BWET"],
    "fixed_income": ["TLT", "IEF", "SHY", "LQD", "HYG", "JNK", "BBN", "BUXX", "CLOX",
                     "CLOZ", "IIGD", "MTBA", "TBIL", "IVOL", "HBDC"],
    "fx_crypto":    ["UUP", "FXB", "FXC", "FXE", "FXY", "YCS", "IBIT"],
    "volatility":   ["VIX", "MOVE"],
    "stock":        ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "AVGO", "META",
                     "TSLA", "BRK-B", "MU", "LLY", "JPM", "WMT", "AMD", "V", "JNJ",
                     "XOM", "MA", "INTC", "ABBV", "PLTR", "BAC", "ORCL", "CSCO",
                     "COST", "CVX", "KO", "LRCX", "CAT", "AMAT", "SNDK", "SBSW"],
}


# Symbols dropped from the source list because they no longer price. Kept here as a
# record so the removal is auditable rather than invisible:
#   CHIR  Global X MSCI China Real Estate  - liquidated
#   EGPT  VanEck Egypt                     - liquidated
#   MOON  Direxion Moonshot Innovators     - liquidated
#   WNDY  Global X Wind Energy             - liquidated (FAN is the surviving wind ETF)
#   RAYC  one bar only, last 2025-11-28
#   EPG   unrecognised symbol - removed
#   PP    unrecognised symbol - removed
UNRESOLVED = {}

def _parse(block):
    return [x.strip().upper() for x in block.replace("\n", " ").split(",")]


def etf_tickers():
    """Just the funds."""
    return [t for t in dict.fromkeys(_parse(RAW)) if t and t not in UNRESOLVED]


def stock_tickers():
    """Just the single names."""
    return [t for t in dict.fromkeys(_parse(STOCKS)) if t]


def index_tickers():
    """Just the volatility indices."""
    return [t for t in dict.fromkeys(_parse(INDICES)) if t]


def all_etfs(include_unresolved: bool = False):
    """Flat, de-duplicated watchlist in declaration order: funds, then single names."""
    seen, out = set(), []
    for t in _parse(INDICES) + _parse(RAW) + _parse(STOCKS):
        if not t or t in seen:
            continue
        if not include_unresolved and t in UNRESOLVED:
            continue
        seen.add(t)
        out.append(t)
    return out


def group_of(ticker: str) -> str:
    for name, members in GROUPS.items():
        if ticker in members:
            return name
    return "other"
