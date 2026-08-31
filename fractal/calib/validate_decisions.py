"""Score the model's *decisions* against Hedgeye's ETF Pro change notes.

Validating levels answers "do we draw the same lines". This answers the question
that actually matters: on the day Hedgeye told subscribers to buy or sell a name,
did our ladder say the same thing?

The change notes are the only place Hedgeye publishes a dated, unambiguous action
per ETF. They go out mid-morning, so the decision is made off the previous close --
which is exactly the bar our daily run uses. For each note we recompute the model as
of that close and compare.

The ranges quoted in a change note are NOT fresh: a removal on 2026-08-28 quotes the
same numbers as the 2026-08-21 weekly report, so they are re-quotes of the last
weekly and are ignored here. Only the action is used.

Hedgeye's four actions map onto our headings:

    ADD_LONG      -> BUY          (at the low end, or a volume breakout)
    REMOVE_LONG   -> SELL         (TREND broke) or SELL SOME (TRADE broke, trim)
    ADD_SHORT     -> SELL SHORT   (at the high end, or a volume breakdown)
    REMOVE_SHORT  -> COVER SHORT  (TREND reclaimed) or BUY SOME (TRADE reclaimed)

A trim counts as agreement with a removal. Hedgeye's list is binary -- a name is on
it or it is not -- so a partial exit is the closest thing we have to "came off the
long side", and scoring it as a miss would penalise the distinction the ladder was
built to make.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..app import signals as S
from ..data.loader import load_params, load_prices, repo_path

# Hedgeye action -> the headings that count as agreeing with it.
EXPECTED = {
    "ADD_LONG":     {"BUY"},
    "REMOVE_LONG":  {"SELL", "SELL SOME"},
    "ADD_SHORT":    {"SELL SHORT"},
    "REMOVE_SHORT": {"COVER SHORT", "BUY SOME"},
}

# Directional agreement: did we at least lean the same way, even if the exact
# heading differed? A removal from the long side and an addition to the short side
# are both "get out of / against this name".
SIDE = {"ADD_LONG": "bullish", "REMOVE_SHORT": "bullish",
        "REMOVE_LONG": "bearish", "ADD_SHORT": "bearish"}
OUR_SIDE = {"BUY": "bullish", "COVER SHORT": "bullish", "BUY SOME": "bullish",
            "SELL": "bearish", "SELL SOME": "bearish", "SELL SHORT": "bearish"}


def _prior_close_pos(close, report_date):
    """Index of the last bar strictly before the report went out."""
    pos = close.index.searchsorted(pd.Timestamp(report_date), side="left") - 1
    return pos if pos >= 120 else None


def run(csv_path=None, profile="hedgeye_anchor", verbose=True):
    csv_path = csv_path or repo_path("reference", "hedgeye_etfpro_changes.csv")
    ref = pd.read_csv(csv_path)
    params = load_params()
    params["range"] = dict(params["range"])
    params["range"]["active"] = profile

    prices = load_prices(sorted(ref["ticker"].unique()), params=params, verbose=False)

    rows = []
    for r in ref.itertuples():
        df = prices.get(r.ticker)
        if df is None or "Close" not in df:
            rows.append({"date": r.report_date, "ticker": r.ticker, "hedgeye": r.action,
                         "model": None, "note": "no price history"})
            continue
        close = df["Close"].dropna()
        pos = _prior_close_pos(close, r.report_date)
        if pos is None:
            rows.append({"date": r.report_date, "ticker": r.ticker, "hedgeye": r.action,
                         "model": None, "note": "too little history"})
            continue
        # Evaluate exactly as the daily run would have, on the bar before the note.
        cut = df.iloc[: pos + 1]
        out = S.evaluate(r.ticker, cut, params)
        if out is None:
            rows.append({"date": r.report_date, "ticker": r.ticker, "hedgeye": r.action,
                         "model": None, "note": "not evaluable"})
            continue
        rows.append({"date": r.report_date, "ticker": r.ticker, "hedgeye": r.action,
                     "model": S.label(out["signal"]) if out["signal"] else None,
                     "why": out["why"], "asof": out["asof"],
                     "trade_bull": out["trade_bull"], "trend_bull": out["trend_bull"],
                     "pos_in_range": round(out["pos_in_range"], 3), "note": ""})

    res = pd.DataFrame(rows)
    res["exact"] = [m in EXPECTED.get(h, set()) if m else False
                    for h, m in zip(res.hedgeye, res.model)]
    res["same_side"] = [OUR_SIDE.get(m) == SIDE.get(h) if m else False
                        for h, m in zip(res.hedgeye, res.model)]
    if verbose:
        report(res)
    return res


def report(res):
    ok = res[res.note == ""]
    print("=" * 78)
    print("ETF Pro change notes  --  did the model call the same action?")
    print("=" * 78)
    cols = ["date", "ticker", "hedgeye", "model", "exact", "same_side", "pos_in_range"]
    print(ok[cols].to_string(index=False))
    if (res.note != "").any():
        print("\nnot scored: " + ", ".join(
            "%s (%s)" % (t, n) for t, n in zip(res[res.note != ""].ticker,
                                               res[res.note != ""].note)))
    n = len(ok)
    if not n:
        return
    print("\n  exact heading match : %d/%d (%.0f%%)" % (ok.exact.sum(), n, 100 * ok.exact.mean()))
    print("  same direction      : %d/%d (%.0f%%)" % (ok.same_side.sum(), n, 100 * ok.same_side.mean()))
    print("  model silent        : %d/%d" % (ok.model.isna().sum(), n))
    print("\n  by Hedgeye action:")
    for act, sub in ok.groupby("hedgeye"):
        print("    %-13s n=%2d  exact %d  same direction %d  silent %d"
              % (act, len(sub), sub.exact.sum(), sub.same_side.sum(), sub.model.isna().sum()))


def main():
    ap = argparse.ArgumentParser(description="Score model decisions vs Hedgeye ETF Pro changes.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--profile", default="hedgeye_anchor")
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
