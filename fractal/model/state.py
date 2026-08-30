"""State machine + fractal-cycle overlay (build spec sections 3.3 and 5).

State is read off the computed lines. Ordering of TRADE vs TREND is never
assumed — SLV printed TREND above TRADE during the August 2026 recovery — it is
read from the numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRENDING_LONG = "trending_long"
TRENDING_SHORT = "trending_short"
COUNTER_TREND = "counter_trend"

# Similar Set fractal cycle (handbook, "The Fractal Cycle"). Phase is derived from
# the *sequence* of TRADE breaks inside one TREND regime, not from a single bar.
PHASES = {
    1: "breakout",         # TRADE + TREND aligned, regime just began
    2: "pullback",         # 1st break of TRADE, TREND intact
    3: "resume",           # TRADE reclaimed
    4: "late_trend",       # 2nd break of TRADE, TREND intact
    5: "trend_transition", # TREND breaks; immediately becomes phase 1 the other way
}


def bull(price: pd.Series, level: pd.Series) -> pd.Series:
    return (price > level)


def classify(trade_bull: bool, trend_bull: bool) -> str:
    if trade_bull and trend_bull:
        return TRENDING_LONG
    if (not trade_bull) and (not trend_bull):
        return TRENDING_SHORT
    return COUNTER_TREND


def state_series(close: pd.Series, lines: pd.DataFrame) -> pd.DataFrame:
    """Per-bar bull/bear per duration plus the combined state."""
    c = pd.Series(close).astype(float)
    df = pd.DataFrame(index=lines.index)
    for name in ("trade", "trend", "tail"):
        if name in lines:
            # object dtype so an undefined line stays NaN rather than silently
            # becoming False, which would read as "bearish"
            col = (c.reindex(lines.index) > lines[name]).astype(object)
            col[lines[name].isna()] = np.nan
            df[f"{name}_bull"] = col

    def _row(r):
        if pd.isna(r.get("trade_bull")) or pd.isna(r.get("trend_bull")):
            return None
        return classify(bool(r["trade_bull"]), bool(r["trend_bull"]))

    df["state"] = df.apply(_row, axis=1)
    return df


def fractal_phase(states: pd.DataFrame) -> pd.Series:
    """Walk the TRADE-break sequence within each TREND regime to label phases 1-5.

    Phase 5 (TREND flip) is the last bar of a regime; the next bar restarts at
    phase 1 on the other side, which is what makes the cycle a loop.
    """
    trade = states.get("trade_bull")
    trend = states.get("trend_bull")
    if trade is None or trend is None:
        return pd.Series(index=states.index, dtype="object")

    out = pd.Series(index=states.index, dtype="object")
    prev_trend = None
    prev_trade = None
    breaks = 0          # completed TRADE breaks inside the current TREND regime
    in_break = False

    for i, ts in enumerate(states.index):
        tr, td = trend.iloc[i], trade.iloc[i]
        if pd.isna(tr) or pd.isna(td):
            continue
        tr, td = bool(tr), bool(td)

        if prev_trend is None:
            out.iloc[i] = PHASES[1]
        elif tr != prev_trend:
            out.iloc[i] = PHASES[5]          # TREND flipped: transition
            breaks, in_break = 0, False
        else:
            aligned = (td == tr)             # TRADE agrees with the TREND regime
            if not aligned:
                if not in_break:
                    breaks += 1
                    in_break = True
                out.iloc[i] = PHASES[2] if breaks == 1 else PHASES[4]
            else:
                in_break = False
                out.iloc[i] = PHASES[1] if breaks == 0 else PHASES[3]

        prev_trend, prev_trade = tr, td
    return out


def snapshot(ticker: str, ohlc: pd.DataFrame, lines: pd.DataFrame,
             rng: pd.DataFrame, spot: float | None = None) -> dict:
    """One row of the daily dashboard for one ticker."""
    close = ohlc["Close"].dropna()
    if len(close) == 0 or lines.dropna(how="all").empty:
        return {"ticker": ticker, "error": "insufficient data"}

    asof = close.index[-1]
    px = float(spot if spot is not None else close.iloc[-1])

    lv = lines.loc[asof] if asof in lines.index else lines.iloc[-1]
    rv = rng.loc[asof] if asof in rng.index else rng.iloc[-1]

    trade, trend, tail = (float(lv.get(k)) if pd.notna(lv.get(k)) else None
                          for k in ("trade", "trend", "tail"))
    lo, hi = float(rv["range_low"]), float(rv["range_high"])

    states = state_series(close, lines)
    phase = fractal_phase(states)
    st = states.iloc[-1]

    def pct(a, b):
        return None if (a is None or b in (None, 0)) else (a / b - 1.0) * 100.0

    row = {
        "ticker": ticker,
        "asof": asof.date().isoformat(),
        "spot": px,
        "range_low": lo,
        "range_high": hi,
        "range_width_pct": (hi / lo - 1.0) * 100.0,
        "pct_to_high": pct(hi, px),
        "pct_to_low": pct(lo, px),
        "pos_in_range": (px - lo) / (hi - lo) if hi > lo else None,
        "trade": trade,
        "trend": trend,
        "tail": tail,
        "trade_bull": None if pd.isna(st.get("trade_bull")) else bool(st["trade_bull"]),
        "trend_bull": None if pd.isna(st.get("trend_bull")) else bool(st["trend_bull"]),
        "tail_bull": None if pd.isna(st.get("tail_bull")) else bool(st["tail_bull"]),
        "state": st.get("state"),
        "phase": phase.iloc[-1] if len(phase) else None,
        "pct_to_trade": pct(trade, px),
        "pct_to_trend": pct(trend, px),
        "tail_confident": bool(lines.attrs.get("tail_confident", False)),
        "bars": int(len(close)),
    }
    return row
