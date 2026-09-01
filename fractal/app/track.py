"""Equal-weighted P&L track for the book, from inception.

One row per session. Each open position contributes its return for that session and
they are averaged equally -- the book carries no size, so equal weight is what it
actually holds. A position opened during the session earns fill-to-close; one held
from before earns prior-close-to-close; one closed during the session earns
prior-close-to-exit and then drops out.

Marked on the FULL price path deliberately. An earlier backtest marked every
position close-over-open and silently discarded overnight gaps, which removed 16.5
points from a year of SPY. Positions are held overnight and earn what happens
overnight.

The track is append-only and keyed by session, so re-running a day replaces that
day rather than double-counting it.
"""
from __future__ import annotations

import datetime as dt
import io
import os

import numpy as np
import pandas as pd

from . import portfolio as P
from ..data.loader import load_params, load_prices, repo_path
from ..data.etf_universe import yf_symbol

INCEPTION = "2026-09-01"


def track_path(custom=None):
    return custom or repo_path("data", "pnl_track.csv")


def load_track(custom=None):
    p = track_path(custom)
    if not os.path.exists(p):
        return pd.DataFrame(columns=["date", "n_open", "day_pct", "cum_pct", "detail"])
    return pd.read_csv(p)


def _bars(tickers, params):
    px = load_prices(sorted({yf_symbol(t) for t in tickers}), params=params, verbose=False)
    return {t: px.get(yf_symbol(t)) for t in tickers}


def session_return(session, book_csv=None, params=None, verbose=True):
    """Equal-weighted return of the book for one session, and the per-name detail."""
    params = params or load_params()
    day = pd.Timestamp(session)
    pos = P.load(book_csv)
    if pos.empty:
        return 0.0, []

    # Anything that was open at some point during this session.
    live = pos[(pos["entry_date"].astype(str) <= str(session))
               & ((pos["status"] == P.OPEN)
                  | (pos["exit_date"].astype(str) >= str(session)))]
    if live.empty:
        return 0.0, []

    bars = _bars(sorted(live["ticker"].unique()), params)
    rows = []
    for r in live.itertuples():
        d = bars.get(r.ticker)
        if d is None or day not in d.index:
            continue
        close = float(d.loc[day]["Close"])
        sign = 1.0 if r.side == P.LONG else -1.0
        opened_today = str(r.entry_date) == str(session)
        closed_today = (r.status == P.CLOSED) and str(r.exit_date) == str(session)

        # Where the position is measured FROM this session.
        if opened_today:
            frm = float(r.entry_price)
        else:
            prior = d.loc[:day]["Close"]
            if len(prior) < 2:
                continue
            frm = float(prior.iloc[-2])
        # ...and TO.
        to = float(r.exit_price) if closed_today else close
        if not frm:
            continue
        rows.append((r.ticker, sign * (to / frm - 1.0) * 100.0,
                     "opened" if opened_today else ("closed" if closed_today else "held")))
    if not rows:
        return 0.0, []
    return float(np.mean([x[1] for x in rows])), rows


def update(session=None, book_csv=None, custom=None, params=None, verbose=True):
    """Add or replace one session in the track. Returns the whole track."""
    session = str(session or dt.date.today())
    day_pct, detail = session_return(session, book_csv=book_csv, params=params)
    t = load_track(custom)
    t = t[t["date"].astype(str) != session]
    row = {"date": session, "n_open": len(detail), "day_pct": round(day_pct, 4),
           "cum_pct": 0.0,
           "detail": "; ".join("%s %+.2f%% (%s)" % (a, b, c) for a, b, c in detail)}
    t = pd.concat([t, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    # Compound, so the cumulative is a real return rather than a sum of percentages.
    t["cum_pct"] = ((1 + t["day_pct"] / 100.0).cumprod() - 1.0) * 100.0
    t.to_csv(track_path(custom), index=False)
    if verbose:
        print("[track] %s: %d position(s), %+.2f%% on the session, %+.2f%% since %s"
              % (session, len(detail), day_pct, t["cum_pct"].iloc[-1], INCEPTION))
        for a, b, c in detail:
            print("    %-6s %+.2f%%  (%s)" % (a, b, c))
    return t


# ------------------------------------------------------------------- reporting

def render(track, book_csv=None):
    """A small, plain report. This one goes to the owner only, so it says what the
    number is and what is behind it rather than explaining the model."""
    import html as _h
    if track.empty:
        return "<p>No sessions tracked yet.</p>"
    last = track.iloc[-1]
    cum = float(last["cum_pct"])
    col = "#0b8f6e" if cum >= 0 else "#d33"
    rows = "".join(
        '<tr><td style="padding:6px 0;border-bottom:1px solid #e6e8ec">%s</td>'
        '<td align="right" style="padding:6px 0;border-bottom:1px solid #e6e8ec;'
        'font-variant-numeric:tabular-nums">%d</td>'
        '<td align="right" style="padding:6px 0 6px 16px;border-bottom:1px solid #e6e8ec;'
        'font-variant-numeric:tabular-nums;color:%s">%+.2f%%</td>'
        '<td align="right" style="padding:6px 0 6px 16px;border-bottom:1px solid #e6e8ec;'
        'font-variant-numeric:tabular-nums;font-weight:700;color:%s">%+.2f%%</td></tr>'
        % (r["date"], int(r["n_open"]),
           "#0b8f6e" if r["day_pct"] >= 0 else "#d33", r["day_pct"],
           "#0b8f6e" if r["cum_pct"] >= 0 else "#d33", r["cum_pct"])
        for _, r in track.iterrows())
    detail = _h.escape(str(last.get("detail", "") or ""))
    book = P.open_positions(book_csv)
    holds = "".join(
        '<tr><td style="padding:4px 0;color:#5a6270">%s</td>'
        '<td align="right" style="padding:4px 0;color:#5a6270;'
        'font-variant-numeric:tabular-nums">%s from %s</td></tr>'
        % (r.ticker, r.side, _h.escape(str(r.entry_date))) for r in book.itertuples())
    return """<div style="font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
max-width:640px;margin:0 auto;padding:22px;color:#111">
<div style="font-size:12px;letter-spacing:.07em;color:#8b94a5;font-weight:700">EQUAL-WEIGHTED P&amp;L TRACK</div>
<div style="font-size:30px;font-weight:700;margin:6px 0 2px;color:%s">%+.2f%%</div>
<div style="color:#5a6270;font-size:13px;margin-bottom:18px">since inception %s
&middot; %d session(s) &middot; %d open now</div>
<table width="100%%" cellpadding="0" cellspacing="0" style="font-size:13px">
<tr><td style="padding-bottom:6px;color:#8b94a5;font-size:11px;letter-spacing:.05em">SESSION</td>
<td align="right" style="padding-bottom:6px;color:#8b94a5;font-size:11px">POSITIONS</td>
<td align="right" style="padding:0 0 6px 16px;color:#8b94a5;font-size:11px">DAY</td>
<td align="right" style="padding:0 0 6px 16px;color:#8b94a5;font-size:11px">CUMULATIVE</td></tr>
%s</table>
<div style="margin-top:18px;color:#8b94a5;font-size:11px;letter-spacing:.05em">LATEST SESSION</div>
<div style="color:#5a6270;font-size:12.5px;margin-top:4px;line-height:1.6">%s</div>
<div style="margin-top:16px;color:#8b94a5;font-size:11px;letter-spacing:.05em">OPEN NOW</div>
<table width="100%%" cellpadding="0" cellspacing="0" style="font-size:12.5px;margin-top:4px">%s</table>
<div style="margin-top:20px;color:#8b94a5;font-size:11.5px;line-height:1.6">
Equal weight across whatever is open, with no position cap &mdash; a one-name book is
fully invested in one name. Marked on the full price path, so overnight moves count.
Gross of costs.</div></div>""" % (col, cum, INCEPTION, len(track),
                                  len(book), rows, detail or "&mdash;", holds)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Update and optionally send the P&L track.")
    ap.add_argument("--session", default=None)
    ap.add_argument("--send-to", default=None, help="email the report to this address")
    a = ap.parse_args()
    t = update(a.session)
    out = repo_path("out", "pnl_track.html")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(render(t))
    print("[track] wrote %s" % out)
    if a.send_to:
        from . import publish
        msg = publish.build_message([a.send_to], subject="P&L track - %+.2f%% since %s"
                                    % (float(t["cum_pct"].iloc[-1]), INCEPTION),
                                    html_path=out)
        publish.deliver(msg, [a.send_to])
        # Say so. The first run printed nothing at all, which is indistinguishable
        # from having quietly failed.
        print("[track] sent to %s" % a.send_to)


if __name__ == "__main__":
    main()
