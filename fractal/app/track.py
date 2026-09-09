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
    """Unit-weighted return of the book for one session, and the per-name detail.

    Each lot contributes its move scaled by its size: units x UNIT_PCT of capital.
    The book is not fully invested -- the residual is cash and earns nothing -- so
    this is the return on TOTAL capital, which is the honest sized-book number.
    Before sizing existed the book was equal-weighted; a lot with no units recorded
    counts as the starter unit, so old sessions still compute.
    """
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
        w = P.units_of(r) * P.UNIT_PCT / 100.0        # fraction of capital in this lot
        rows.append((r.ticker, sign * (to / frm - 1.0) * 100.0,
                     "opened" if opened_today else ("closed" if closed_today else "held"),
                     w))
    if not rows:
        return 0.0, []
    # return on total capital = sum(weight_i * ret_i); cash (the rest) earns 0
    day_ret = float(sum(w * (x / 100.0) for _, x, _, w in rows) * 100.0)
    deploy = float(sum(w for _, _, _, w in rows) * 100.0)   # % of capital at work
    detail = [(tk, x, tag) for tk, x, tag, _ in rows]
    return day_ret, detail, deploy


def bars_ready(session, book_csv=None, params=None):
    """Do we actually have the session's close for the names we hold?

    The track skips any position whose session bar is missing, so a run before the
    daily bars publish silently produces "0 positions, +0.00%" -- which is not a
    flat day, it is no data, and on 3 Sep 2026 that went out by email as though it
    were a real result. The settle fires 21:20 local, twenty minutes after the US
    close, and Yahoo had not posted the 09-03 bars yet.
    """
    params = params or load_params()
    pos = P.load(book_csv)
    live = pos[pos["status"] == P.OPEN]
    if live.empty:
        return True, 0, 0            # nothing held is a legitimate flat day
    tickers = sorted(live["ticker"].unique())
    # Force a live fetch. The Yahoo cache holds for 12 hours, so the noon run's
    # file is still "fresh" at 21:20 and this check would sit re-reading a copy
    # that ends at the PREVIOUS close -- reporting "not published" long after the
    # bars were up. That is what happened on 3 Sep 2026.
    bars = {t: d for t, d in zip(
        tickers, [load_prices([yf_symbol(t)], params=params, verbose=False,
                              max_age_hours=0).get(yf_symbol(t)) for t in tickers])}
    day = pd.Timestamp(session)
    have = sum(1 for t in tickers
               if bars.get(t) is not None and day in bars[t].index)
    return have > 0, have, len(tickers)


def update(session=None, book_csv=None, custom=None, params=None, verbose=True):
    """Add or replace one session in the track. Returns the whole track."""
    session = str(session or dt.date.today())
    day_pct, detail, deploy = session_return(session, book_csv=book_csv, params=params)
    t = load_track(custom)
    t = t[t["date"].astype(str) != session]
    row = {"date": session, "n_open": len(detail), "day_pct": round(day_pct, 4),
           "cum_pct": 0.0, "deploy_pct": round(deploy, 2),
           "detail": "; ".join("%s %+.2f%% (%s)" % (a, b, c) for a, b, c in detail)}
    t = pd.concat([t, pd.DataFrame([row])], ignore_index=True).sort_values("date")
    # Compound, so the cumulative is a real return rather than a sum of percentages.
    t["cum_pct"] = ((1 + t["day_pct"] / 100.0).cumprod() - 1.0) * 100.0
    t.to_csv(track_path(custom), index=False)
    if verbose:
        spy = spy_since_inception(t, params)
        spy_txt = ("  |  SPY %+.2f%%  (%+.2f%% vs SPY)"
                   % (spy, t["cum_pct"].iloc[-1] - spy)) if spy is not None else ""
        print("[track] %s: %d position(s), %.0f%% deployed, %+.2f%% on the "
              "session, %+.2f%% since %s%s" % (session, len(detail), deploy, day_pct,
              t["cum_pct"].iloc[-1], INCEPTION, spy_txt))
        for a, b, c in detail:
            print("    %-6s %+.2f%%  (%s)" % (a, b, c))
    return t


# ------------------------------------------------------------------- reporting

def spy_since_inception(track, params=None):
    """SPY total return over exactly the sessions the track covers.

    Compounded across the same dates as the book so the comparison is like for
    like: each tracked session contributes SPY's move that day (prior close to
    close). Returns None if SPY history is unavailable.
    """
    if track is None or track.empty:
        return None
    params = params or load_params()
    d = load_prices(["SPY"], params=params, verbose=False).get("SPY")
    if d is None or "Close" not in d:
        return None
    r = d["Close"].dropna().pct_change()
    dates = pd.to_datetime(track["date"].astype(str))
    sel = r.reindex(dates).dropna()
    if sel.empty:
        return None
    return float(((1 + sel).prod() - 1) * 100)


def render(track, book_csv=None):
    """A small, plain report. This one goes to the owner only, so it says what the
    number is and what is behind it rather than explaining the model."""
    import html as _h
    if track.empty:
        return "<p>No sessions tracked yet.</p>"
    last = track.iloc[-1]
    cum = float(last["cum_pct"])
    col = "#0b8f6e" if cum >= 0 else "#d33"
    spy = spy_since_inception(track, book_csv=None) if False else spy_since_inception(track)
    if spy is not None:
        vs = cum - spy
        spy_line = ('<div style="color:#5a6270;font-size:13px;margin:-12px 0 18px">'
                    'SPY over the same sessions <b>%+.2f%%</b> &middot; '
                    '<span style="color:%s;font-weight:700">%+.2f%% vs SPY</span></div>'
                    % (spy, "#0b8f6e" if vs >= 0 else "#d33", vs))
    else:
        spy_line = ""
    rows = "".join(
        '<tr><td style="padding:6px 0;border-bottom:1px solid #e6e8ec">%s</td>'
        '<td align="right" style="padding:6px 0;border-bottom:1px solid #e6e8ec;'
        'font-variant-numeric:tabular-nums">%d &middot; %.0f%%</td>'
        '<td align="right" style="padding:6px 0 6px 16px;border-bottom:1px solid #e6e8ec;'
        'font-variant-numeric:tabular-nums;color:%s">%+.2f%%</td>'
        '<td align="right" style="padding:6px 0 6px 16px;border-bottom:1px solid #e6e8ec;'
        'font-variant-numeric:tabular-nums;font-weight:700;color:%s">%+.2f%%</td></tr>'
        % (r["date"], int(r["n_open"]), float(r.get("deploy_pct", 0) or 0),
           "#0b8f6e" if r["day_pct"] >= 0 else "#d33", r["day_pct"],
           "#0b8f6e" if r["cum_pct"] >= 0 else "#d33", r["cum_pct"])
        for _, r in track.iterrows())
    detail = _h.escape(str(last.get("detail", "") or ""))
    book = P.open_positions(book_csv)
    holds = "".join(
        '<tr><td style="padding:4px 0;color:#5a6270">%s</td>'
        '<td align="right" style="padding:4px 0;color:#5a6270;'
        'font-variant-numeric:tabular-nums">%s &middot; %du (%.0f%%) from %s</td></tr>'
        % (r.ticker, r.side, P.units_of(r), P.units_of(r) * P.UNIT_PCT,
           _h.escape(str(r.entry_date))) for r in book.itertuples())
    deploy_now = sum(P.units_of(r) for r in book.itertuples()) * P.UNIT_PCT
    return """<div style="font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
max-width:640px;margin:0 auto;padding:22px;color:#111">
<div style="font-size:12px;letter-spacing:.07em;color:#8b94a5;font-weight:700">SIZED P&amp;L TRACK</div>
<div style="font-size:30px;font-weight:700;margin:6px 0 2px;color:%s">%+.2f%%</div>
<div style="color:#5a6270;font-size:13px;margin-bottom:18px">on total capital since
inception %s &middot; %d session(s) &middot; %d open now &middot; %.0f%% deployed</div>
%s
<table width="100%%" cellpadding="0" cellspacing="0" style="font-size:13px">
<tr><td style="padding-bottom:6px;color:#8b94a5;font-size:11px;letter-spacing:.05em">SESSION</td>
<td align="right" style="padding-bottom:6px;color:#8b94a5;font-size:11px">POSITIONS &middot; DEPLOYED</td>
<td align="right" style="padding:0 0 6px 16px;color:#8b94a5;font-size:11px">DAY</td>
<td align="right" style="padding:0 0 6px 16px;color:#8b94a5;font-size:11px">CUMULATIVE</td></tr>
%s</table>
<div style="margin-top:18px;color:#8b94a5;font-size:11px;letter-spacing:.05em">LATEST SESSION</div>
<div style="color:#5a6270;font-size:12.5px;margin-top:4px;line-height:1.6">%s</div>
<div style="margin-top:16px;color:#8b94a5;font-size:11px;letter-spacing:.05em">OPEN NOW</div>
<table width="100%%" cellpadding="0" cellspacing="0" style="font-size:12.5px;margin-top:4px">%s</table>
<div style="margin-top:20px;color:#8b94a5;font-size:11.5px;line-height:1.6">
<b>Sizing.</b> 1 unit = 3%% of capital. A position scales in one unit at a time as the
signal confirms and scales out one unit on a trim, closing only at the floor. Caps
are per asset class &mdash; equities 9%%, commodities 6%%, fixed income 9%%, FX 12%% &mdash;
and shorts run smaller than longs (max 3%% vs 9%%). The book is not fully invested; the
rest is cash, so this is the return on <b>total</b> capital. Sessions before 9 Sep 2026
are <b>restated</b> at starter sizing (the book was traded binary before then, so no
ladder history exists). Marked on the full price path; gross of costs.</div></div>""" % (
    col, cum, INCEPTION, len(track), len(book), deploy_now, spy_line, rows,
    detail or "&mdash;", holds)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Update and optionally send the P&L track.")
    ap.add_argument("--session", default=None)
    ap.add_argument("--send-to", default=None, help="email the report to this address")
    ap.add_argument("--force", action="store_true",
                    help="send even when the session's bars have not published")
    a = ap.parse_args()
    session = str(a.session or dt.date.today())
    ready, have, need = bars_ready(session)
    if not ready and not a.force:
        print("[track] ABORTED - no %s close for any of the %d open position(s). "
              "This is missing data, not a flat session; sending it would report "
              "+0.00%% as though it were real. Re-run once the bars publish, or "
              "pass --force." % (session, need))
        return 1
    if have < need:
        print("[track] WARNING - only %d of %d position(s) have a %s bar; the "
              "session return is computed on those." % (have, need, session))
    t = update(a.session)
    out = repo_path("out", "pnl_track.html")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(render(t))
    print("[track] wrote %s" % out)
    # Do not re-email a session already sent. publish.py has had this guard from
    # the start; the track did not, so every run that reached this point sent
    # again. On 3 Sep 2026 overlapping background retries put five copies of the
    # same P&L in the owner's inbox. --force overrides, for a genuine resend.
    stamp = repo_path("out", ".track_sent")
    already = ""
    if os.path.exists(stamp):
        already = io.open(stamp, encoding="utf-8").read().strip()
    if a.send_to and already == session and not a.force:
        print("[track] already sent the %s track; not sending again (--force to "
              "override)" % session)
        return 0
    if a.send_to:
        from . import publish
        msg = publish.build_message([a.send_to], subject="P&L track - %+.2f%% since %s"
                                    % (float(t["cum_pct"].iloc[-1]), INCEPTION),
                                    html_path=out)
        publish.deliver(msg, [a.send_to])
        # Say so. The first run printed nothing at all, which is indistinguishable
        # from having quietly failed.
        with io.open(stamp, "w", encoding="utf-8") as fh:
            fh.write(session)
        print("[track] sent to %s" % a.send_to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
