"""Score our TREND read against the positions Keith names on The Macro Show.

The show's summary email opens with a TL;DR listing every name mentioned, split
BULLISH / BEARISH. That is a published book with a date on it -- the closest thing
available to a daily answer key, and unlike the ETF Pro change notes it covers what
he HOLDS rather than only what changed.

What this can and cannot test. It tests direction: does our TREND agree with the
side he is on. It does NOT test timing -- his entries are discretionary and sit
mid-range, which we established against his Real-Time Alerts ledger, where our
durations agreed 7 of 9 times and we would have taken none of the trades. So a high
score here means we read the same market, not that we would trade it the same way.

Names he mentions as watching rather than holding ("buy menu / not yet bought") are
kept, because a direction is still being asserted.

The reference file is Hedgeye research and stays out of git.
"""
from __future__ import annotations

import argparse

import pandas as pd

from ..app import signals as S
from ..data.loader import load_params, repo_path


def run(csv_path=None, asof=None, verbose=True):
    csv_path = csv_path or repo_path("reference", "macroshow_positions.csv")
    ref = pd.read_csv(csv_path)
    ref["report_date"] = ref["report_date"].astype(str)
    if asof:
        ref = ref[ref["report_date"] == str(asof)]
    if ref.empty:
        raise SystemExit("no positions for %s in %s" % (asof, csv_path))

    df = S.run(sorted(ref["ticker"].unique()), params=load_params(), verbose=False)
    read = {r.ticker: r for r in df.itertuples()}

    rows = []
    for r in ref.itertuples():
        m = read.get(r.ticker)
        if m is None:
            rows.append(dict(date=r.report_date, ticker=r.ticker, keith=r.side,
                             ours=None, agree=None, note="not tracked"))
            continue
        ours = ("neutral" if m.trend_neutral
                else "bull" if m.trend_bull is True else "bear")
        rows.append(dict(date=r.report_date, ticker=r.ticker, keith=r.side, ours=ours,
                         # Neutral is neither agreement nor disagreement: the model is
                         # explicitly declining to take a side, so it is scored apart
                         # rather than counted as a miss.
                         agree=(None if ours == "neutral" else ours == r.side),
                         pos=round(float(m.pos_in_range), 2),
                         signal=m.signal or "", note=r.note))
    out = pd.DataFrame(rows)
    if verbose:
        report(out)
    return out


def report(out):
    pd.set_option("display.width", 200)
    print("=" * 84)
    print("Our TREND vs Keith's Macro Show positions")
    print("=" * 84)
    for d, g in out.groupby("date"):
        scored = g[g.agree.notna()]
        n_ok = int(scored.agree.sum())
        print("\n%s  --  %d of %d agree, %d neutral, %d untracked"
              % (d, n_ok, len(scored), int((g.ours == "neutral").sum()),
                 int(g.ours.isna().sum())))
        for r in g.itertuples():
            mark = ("ok " if r.agree else "DIFFER") if r.agree is not None else (
                "neutral" if r.ours else "-")
            print("   %-7s keith %-5s  ours %-8s %-8s %s"
                  % (r.ticker, r.keith, r.ours or "-", mark, r.note))
    scored = out[out.agree.notna()]
    if len(scored):
        print("\n  overall: %d of %d (%.0f%%) on names where we take a side"
              % (int(scored.agree.sum()), len(scored), 100 * scored.agree.mean()))
        print("  neutral (no side taken): %d   untracked: %d"
              % (int((out.ours == "neutral").sum()), int(out.ours.isna().sum())))


def main():
    ap = argparse.ArgumentParser(description="Score the model against Macro Show positions.")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--asof", default=None, help="score a single report date")
    a = ap.parse_args()
    run(a.csv, a.asof)


if __name__ == "__main__":
    main()
