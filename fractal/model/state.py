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
