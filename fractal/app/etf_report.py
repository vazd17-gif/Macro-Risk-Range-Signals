"""Macro Risk Range report: interactive dashboard + email newsletter.

Two renderers over the same signal set (app/signals.py):

  dashboard   a self-contained interactive page - sortable, filterable, with a
              visual range bar per name. For working through the whole list.
  newsletter  a narrow, inline-styled HTML email that leads with the actionable
              names and keeps the full table as an appendix. Email clients strip
              <style> blocks and ignore flexbox, so the newsletter is built with
              tables and inline styles only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import os

import numpy as np
import pandas as pd

from ..data.loader import load_params, next_session, repo_path
from . import signals as S
from . import portfolio as P
from ..data.etf_names import short_names

# ---------------------------------------------------------------- shared bits
# Keyed by the heading shown rather than by the signal, because several signals
# report under one order: price at the low end and a short being closed are both
# buys. The per-row `why` keeps the reason they differ.
SIG_STYLE = {
    "BUY":        ("#0ea37f", "Buy - at the low end of the RANGE with TRADE and "
                              "TREND still bullish"),
    "SELL SOME":  ("#d9a441", "Sell some - TRADE has broken while TREND still holds; "
                              "trim, do not exit"),
    "SELL":       ("#ef5350", "Sell - TREND has broken; the long comes off"),
    "SELL SHORT": ("#c0392b", "Sell short - a bearish TREND rallying into the high "
                              "end of the RANGE"),
    "BUY SOME":   ("#5c9ded", "Buy some - TRADE reclaimed while TREND is still bearish; "
                              "buy back part of the short"),
    "COVER SHORT":("#5c9ded", "Cover - TREND reclaimed; close the short out"),
    "WATCHLIST":  ("#8b94a5", "Watchlist - nothing to act on yet"),
}
BULL, BEAR, FLAT = "#0ea37f", "#ef5350", "#8b94a5"


def _new_badge(r):
    """A mark on instructions that changed since the last session.

    Only the new ones are marked. Everything in the list is still true, so a name
    carrying the same instruction as last time needs no label -- the absence of one
    says it. Marking both halves put a grey word on 21 of 52 rows and made the thing
    worth spotting harder to spot.
    """
    if not getattr(r, "signal", None):
        return ""                      # nothing was instructed, so nothing is new
    if not getattr(r, "is_new", True):
        return ""                      # still standing, and silence says that
    return ('<span style="color:#0b8f6e;font-weight:700;font-size:10.5px;'
            'letter-spacing:.04em"> NEW</span>')


def _closed_block(closed, dark=True):
    """Positions that came off this session, with what they made.

    A reduction is the moment P&L stops being paper, so it is reported rather than
    the name simply dropping out of the open list. Without this a "sell some" would
    look like the position had never been held.
    """
    if closed is None or not len(closed):
        return ""
    line, dim = ("var(--line)", "var(--dim)") if dark else ("#e6e8ec", "#8b94a5")
    rows = []
    for r in closed.itertuples():
        col = ("#0ea37f" if r.pnl_pct >= 0 else "#ef5350") if dark else (
              "#0b8f6e" if r.pnl_pct >= 0 else "#d33")
        rows.append(
            '<tr><td style="padding:7px 0;border-bottom:1px solid %s">'
            '<span style="font-weight:700">%s</span>'
            '<span style="color:%s;font-size:12px"> &nbsp;%s closed</span>'
            '<span style="float:right;color:%s;font-weight:700">%+.2f%%</span>'
            '<div style="color:%s;font-size:12.5px;margin-top:2px">'
            '%s &rarr; %s</div></td></tr>'
            % (line, r.ticker, dim, r.side, col, r.pnl_pct, dim,
               _f(r.entry_price), _f(r.exit_price)))
    return "".join(rows)


# A volatility index is coloured by what it means for everything else, not by which
# side of its own TREND it sits on -- falling vol is the green case.
STANCE_COL = {"supportive": "#0ea37f", "risk_off": "#ef5350", "mixed": "#d9a441"}


def _provisional(r, live):
    """True when a signal is a mid-session STATE rather than something that happened.

    Intraday the book only opens on a duration line actually cleared. A name sitting
    at the low end of its range at 11am has not done anything -- that is just where
    price is standing, and it says nothing about where it closes. Both readings wore
    the same green pill, so the dashboard could show twenty buys while the book took
    two, with nothing on the page explaining the gap.

    Off the close there is no distinction to draw: the levels are fixed and the
    reading is the decision.
    """
    if not live or not getattr(r, "signal", None):
        return False
    x = getattr(r, "intraday", "")
    return not (x and x == x and str(x).strip())


def _trend_col(trend_bull):
    """Ticker colour: green above TREND, red below, grey neutral or unknown.

    Tested identity-first rather than by truthiness. A NaN -- which is what a
    neutral TREND becomes if the frame has been through a CSV -- is truthy, so the
    obvious `BULL if trend_bull` would paint an undecided name bright green.
    """
    if trend_bull is True:
        return BULL
    if trend_bull is False:
        return BEAR
    return FLAT


# Long side first, then short, trims beside the full action they reduce.
SIGNAL_ORDER = S.SECTIONS

GROUP_LABEL = {
    "us_equity": "US Equity", "us_sector": "US Sectors", "us_smallcap": "US Small Cap",
    "thematic": "Thematic", "intl": "International", "commodity": "Commodities",
    "fixed_income": "Fixed Income / Credit", "fx_crypto": "FX & Crypto",
    "stock": "S&P 500 Stocks", "other": "Other",
}


def _universe_label(df):
    """e.g. "143 ETFs, 32 stocks, 2 vol indices" -- the list is no longer funds only."""
    n_stock = int((df["group"] == "stock").sum()) if "group" in df else 0
    n_idx = int(df["is_index"].sum()) if "is_index" in df else 0
    n_mac = int(df["is_macro"].sum()) if "is_macro" in df else 0
    n_fund = len(df) - n_stock - n_idx - n_mac
    parts = []
    for n, one, many in ((n_fund, "ETF", "ETFs"), (n_stock, "stock", "stocks"),
                         (n_idx, "vol index", "vol indices"),
                         (n_mac, "macro ref", "macro refs")):
        if n:
            parts.append("%d %s" % (n, one if n == 1 else many))
    return ", ".join(parts) or "%d names" % len(df)


def _band_note(band):
    """The range-edge band in force, which the VIX level sets."""
    if not band:
        return ""
    buy, sell = band
    return (" &middot; buy zone the outer <b>%.0f%%</b>, sell zone the outer <b>%.0f%%</b>"
            % (100 * buy, 100 * sell))


def _vix_meter(idx, email=False, band=None):
    """The three VIX buckets as a lit meter, with the live one filled.

    Rendered as a meter rather than a sentence because the bucket is a severity
    scale and severity reads faster as position than as words -- you want to see
    how far along the scale you are, not only which name you landed on. The inactive
    buckets stay visible for the same reason: "INVESTABLE" means little without the
    two worse states beside it.

    Email clients cannot be trusted with flexbox, so the same thing is laid out as
    a table when `email` is set.
    """
    if idx.empty:
        return ""
    vix = idx[idx["ticker"] == "VIX"]
    if not len(vix):
        return ""
    lvl = float(vix.iloc[0]["spot"])
    live, _band, _note, _stance = S.vix_bucket(lvl)
    if not live:
        return ""

    # `rng` is the bucket's own "9-19" label. It is deliberately not called `band`:
    # that name belongs to the range-edge fraction this function also reports, and
    # the two were quietly colliding.
    cells = []
    for _ceiling, name, rng, note, stance in S.VIX_BUCKETS:
        col = STANCE_COL.get(stance, "#8b94a5")
        on = name == live
        if on:
            style = ("background:%s;border:1px solid %s;color:#0d0f13" % (col, col))
            sub, subcol = note, "#0d0f13"
        else:
            style = ("background:transparent;border:1px solid %s44;color:%s99"
                     % (col, col))
            sub, subcol = rng, "%s88" % col
        cells.append((style, name, rng, sub, subcol, on))

    if email:
        tds = "".join(
            '<td width="33%%" style="%s;border-radius:6px;padding:7px 9px;'
            'font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif">'
            '<div style="font-size:11px;font-weight:700;letter-spacing:.05em">%s'
            '<span style="font-weight:400;opacity:.75"> &nbsp;%s</span></div>'
            '<div style="font-size:11px;margin-top:2px;color:%s">%s</div></td>'
            '<td width="6"></td>' % (style, name, rng, subcol, sub if on else "&nbsp;")
            for style, name, rng, sub, subcol, on in cells)
        return ('<div style="font-size:12.5px;color:#5a6270;margin-bottom:6px">'
                'VIX <b style="color:#111;font-size:15px">%.2f</b>%s</div>'
                '<table width="100%%" cellpadding="0" cellspacing="0"><tr>%s</tr></table>'
                % (lvl, _band_note(band), tds))

    segs = "".join(
        '<div style="flex:1 1 0;%s;border-radius:7px;padding:7px 10px">'
        '<div style="font-size:11px;font-weight:700;letter-spacing:.05em">%s'
        '<span style="font-weight:400;opacity:.7"> &nbsp;%s</span></div>'
        '<div style="font-size:11px;margin-top:2px;color:%s">%s</div></div>'
        % (style, name, rng, subcol, sub if on else "&nbsp;")
        for style, name, rng, sub, subcol, on in cells)
    return ('<div style="color:var(--dim);font-size:12.5px;margin-bottom:7px">'
            'VIX <b style="color:var(--fg);font-size:15px">%.2f</b>%s</div>'
            '<div style="display:flex;gap:7px;margin-bottom:11px">%s</div>'
            % (lvl, _band_note(band), segs))


def _vol_regime(idx, band=None):
    """One-line read of the volatility complex.

    Falling volatility is supportive for risk assets, so bearish TRADE and TREND on
    the VIX and MOVE is a tailwind rather than a warning. Both bullish is the
    opposite. The point of carrying them is that this read frames every other
    signal on the page.
    """
    if idx.empty:
        return "", "#8b94a5"

    bear = int(((idx["trade_bull"] == False) & (idx["trend_bull"] == False)).sum())
    bull = int(((idx["trade_bull"] == True) | (idx["trend_bull"] == True)).sum())
    n = len(idx)
    if bear == n:
        tail, col = ("Volatility is bearish on both durations across the complex "
                     "&mdash; falling vol, supportive for risk assets."), "#0ea37f"
    elif bear == 0:
        tail, col = ("Volatility is bullish &mdash; rising vol, a headwind for risk "
                     "assets. Treat long signals below with more caution."), "#ef5350"
    else:
        tail, col = ("Volatility is mixed across the complex &mdash; no clear "
                     "tailwind either way."), "#d9a441"
    return tail, col


def _index_rows(idx):
    """(ticker, name, spot, trade, trend, read, colour) per index."""
    out = []
    for r in idx.itertuples():
        both_bear = (r.trade_bull is False) and (r.trend_bull is False)
        both_bull = bool(r.trade_bull) and bool(r.trend_bull)
        if both_bear:
            read, col = "bearish TRADE and TREND &middot; supportive", "#0ea37f"
        elif both_bull:
            read, col = "bullish TRADE and TREND &middot; risk-off", "#ef5350"
        else:
            read, col = "mixed", "#d9a441"
        out.append((r.ticker, r.spot, r.trade, r.trend, read, col))
    return out


def _macro_block(df, dark=True):
    """Hedgeye's "Our Levels" in our own numbers: the indices, yields, currencies
    and commodities that frame everything else.

    Levels and a direction, never an instruction -- a yield has no shares, and the
    point of carrying them is context. Direction is the three-state read, so a name
    resting on its TREND line says so rather than being forced to a side.
    """
    if "is_macro" not in df:
        return ""
    m = df[df["is_macro"] == True]
    if not len(m):
        return ""
    from ..data.etf_universe import MACRO_NAMES, MACRO, _parse
    order = {t: i for i, t in enumerate(_parse(MACRO))}
    m = m.sort_values("ticker", key=lambda c: c.map(lambda t: order.get(t, 999)))
    line, dim, ink = (("var(--line)", "var(--dim)", "var(--ink)") if dark
                      else ("#e6e8ec", "#8b94a5", "#111"))
    rows = []
    for r in m.itertuples():
        if r.trend_neutral:
            d, dc = "neutral", "#8b94a5"
        elif r.trend_bull is True:
            d, dc = "bullish", "#0ea37f"
        else:
            d, dc = "bearish", "#ef5350"
        # The ticker itself carries the TREND colour, the same way it does on the
        # scan and on the alert chips. One glance down the column reads the macro
        # setup without anyone parsing a word at the end of each row.
        rows.append(
            '<tr><td style="padding:6px 0;border-bottom:1px solid %s">'
            '<span style="font-weight:700;color:%s">%s</span>'
            '<span style="color:%s;font-size:12px"> &nbsp;%s</span></td>'
            '<td align="right" style="padding:6px 0 6px 14px;border-bottom:1px solid %s;'
            'font-variant-numeric:tabular-nums;font-weight:650;color:%s">%s</td>'
            '<td align="right" style="padding:6px 0 6px 14px;border-bottom:1px solid %s;'
            'font-variant-numeric:tabular-nums;color:%s">%s &ndash; %s</td>'
            '<td align="right" style="padding:6px 0 6px 14px;border-bottom:1px solid %s;'
            'color:%s;font-weight:650;font-size:12.5px">%s</td></tr>'
            % (line, dc, r.ticker, dim, html.escape(MACRO_NAMES.get(r.ticker, "")),
               line, ink, _f(r.spot),
               line, dim, _f(r.range_low), _f(r.range_high), line, dc, d))
    return "".join(rows)


def _session_label(asof):
    """The trading day these levels are for, spelled out.

    The report is dated by the session it governs rather than by the bar it was
    built from, and rather than by when the job happened to run -- a Sunday
    regeneration still carries Monday's levels.
    """
    d = next_session(asof)
    return d.strftime("%A, %d %B %Y").replace(" 0", " ") if d else str(asof)


REFRESH_SECONDS = 300      # the live page reloads itself every 5 minutes

# Installability. With these, "Add to Home Screen" gives a real app icon and a
# standalone window with no browser chrome. The service worker deliberately caches
# nothing -- an offline copy of the dashboard would show stale levels, which is
# worse than showing nothing -- but registering one is what makes Android offer the
# install prompt at all.
PWA_HEAD = """<link rel="manifest" href="manifest.json">
<meta name="theme-color" content="#0d0f13">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Risk Range">
<link rel="apple-touch-icon" href="icons/apple-touch-icon.png">
<script>if("serviceWorker" in navigator){addEventListener("load",function(){
navigator.serviceWorker.register("sw.js").catch(function(){});});}</script>"""

VOL_COLOUR = {"surge": "#d9a441", "dry": "#5c9ded"}

# Columns dropped on a narrow screen -- reference detail rather than decisions.
OPTIONAL_COLS = {"Range low", "Range high", "% to low", "% to high",
                 "TRADE", "TREND", "Volume", "z vs 1m", "z vs 3m", "Why"}


def _vol(v):
    """Compact share volume: 25020400 -> 25.0M."""
    if v is None or not np.isfinite(v):
        return "&ndash;"
    for cut, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= cut:
            return "%.1f%s" % (v / cut, suf)
    return "%.0f" % v


def _z_cell(z, thresh=2.0):
    """Volume z-score, tinted once it clears the outlier threshold."""
    if z is None or not np.isfinite(z):
        return "&ndash;"
    txt = "%+.1f" % z
    if z >= thresh:
        return '<span style="color:%s;font-weight:650">%s</span>' % (VOL_COLOUR["surge"], txt)
    if z <= -thresh:
        return '<span style="color:%s;font-weight:650">%s</span>' % (VOL_COLOUR["dry"], txt)
    return txt


def _f(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "&ndash;"
    return ("%%.%df" % nd) % v


def _bull_cell(level, bull):
    if level is None or not np.isfinite(level):
        return "&ndash;"
    colour = ("#0ea37f" if bull is True else
              "#ef5350" if bull is False else "#8b94a5")     # neutral / unknown
    return '<span style="color:%s">%s</span>' % (colour, _f(level))


# ------------------------------------------------------------------ dashboard
CSS = """
:root{--bg:#0d0f13;--panel:#151920;--line:#242a34;--fg:#e6e9ef;--dim:#8b94a5;
--bull:#0ea37f;--bear:#ef5350;--warn:#d9a441;--info:#5c9ded}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1600px;margin:0 auto;padding:26px 20px 64px}
h1{font-size:21px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:10px 15px;min-width:132px;border-left:3px solid var(--line)}
.card .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:20px;font-weight:650;margin-top:2px}
.filters{display:flex;flex-direction:column;gap:9px;margin-bottom:16px}
.frow{display:grid;grid-template-columns:66px 1fr;gap:9px;align-items:start}
.flab{color:var(--dim);font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;
padding-top:7px}
.fbtns{display:flex;flex-wrap:wrap;gap:8px}
@media(max-width:640px){.frow{grid-template-columns:1fr;gap:4px}.flab{padding-top:0}}
button{background:var(--panel);color:var(--fg);border:1px solid var(--line);
border-radius:999px;padding:6px 13px;font-size:12.5px;cursor:pointer}
button:hover{border-color:#39414f}
button[aria-pressed="true"]{background:#1d2530;border-color:#3d4a5c;color:#fff}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;min-width:1180px}
th,td{padding:7px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:#11151b;color:var(--dim);font-weight:600;font-size:11px;
text-transform:uppercase;letter-spacing:.05em;cursor:pointer;z-index:1}
th:first-child,td:first-child,td.l{text-align:left}
tbody tr:hover{background:#1a1f27}
.tk{font-weight:650}.grp{color:var(--dim);font-size:11px}
.bar{position:relative;width:132px;height:8px;border-radius:4px;background:#1e242e;display:inline-block;vertical-align:middle}
.bar u{position:absolute;inset:0;border-radius:4px;
background:linear-gradient(90deg,rgba(14,163,127,.30),rgba(90,100,120,.12),rgba(239,83,80,.30))}
.bar i{position:absolute;top:-3px;width:3px;height:14px;border-radius:2px;background:#fff}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11px;font-weight:600;
letter-spacing:.02em;white-space:nowrap}
.why{color:var(--dim);font-size:11.5px}
footer{color:var(--dim);font-size:12px;margin-top:20px;line-height:1.7}
code{background:#11151b;border:1px solid var(--line);border-radius:4px;padding:1px 5px}

/* On a phone the table cannot carry thirteen columns, so the reference detail
   (range edges, distances, line levels, volume) is dropped and only what you act
   on survives: the name, spot, where it sits in the range, and the signal. */
@media (max-width: 820px){
  .wrap{padding:18px 12px 48px}
  h1{font-size:19px}
  table{min-width:0}
  th.opt,td.opt{display:none}
  th,td{padding:7px 8px}
  .bar{width:76px}
  .card{min-width:0;flex:1 1 42%;padding:9px 12px}
  .card .v{font-size:18px}
  .grp{display:none}
  input[type=search]{flex:1 1 100%}
}
"""

JS = """
// Scoped to the scan table by id. Unscoped, the portfolio is a table too -- and in
// a .tablewrap of its own, so selecting by class picks the wrong one -- and its
// rows were swept up by every filter: switching to BUY hid your open positions.
const scan=document.getElementById('scan');
const rows=[...scan.querySelectorAll('tbody tr')];let dir={};
scan.querySelectorAll('th').forEach((th,i)=>{th.onclick=()=>{dir[i]=!dir[i];
const tb=scan.querySelector('tbody');
rows.slice().sort((a,b)=>{const x=a.children[i].dataset.v??a.children[i].textContent,
y=b.children[i].dataset.v??b.children[i].textContent;const nx=parseFloat(x),ny=parseFloat(y);
const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:String(x).localeCompare(String(y));return dir[i]?-c:c;})
.forEach(r=>tb.appendChild(r));};});
function apply(){const f=document.querySelector('button[data-sig][aria-pressed="true"]');
const g=document.querySelector('button[data-grp][aria-pressed="true"]');
const v=document.querySelector('button[data-vol][aria-pressed="true"]');
rows.forEach(r=>{const okS=!f||r.dataset.sig===f.dataset.sig;
const okG=!g||r.dataset.grp===g.dataset.grp;
const okV=!v||r.dataset.vol===v.dataset.vol;
r.style.display=(okS&&okG&&okV)?'':'none';});}
document.querySelectorAll('button[data-sig],button[data-grp],button[data-vol]').forEach(b=>{b.onclick=()=>{
const on=b.getAttribute('aria-pressed')==='true';
const attr=b.dataset.sig!==undefined?'data-sig':(b.dataset.grp!==undefined?'data-grp':'data-vol');
document.querySelectorAll('button['+attr+']').forEach(o=>o.setAttribute('aria-pressed','false'));
b.setAttribute('aria-pressed',on?'false':'true');apply();};});
"""


def render_dashboard(df, params, generated=None, book=None, closed=None):
    generated = generated or dt.datetime.now()
    asof = df["asof"].max() if len(df) else "-"
    counts = df["signal"].map(S.LABEL).value_counts().to_dict()

    # Whether this build is a live re-price decides whether a signal is something
    # that happened or just where price is standing.
    live_at = df.attrs.get("live_at")
    body = []
    for r in df[~df.get("is_index", False)].itertuples() if "is_index" in df else df.itertuples():
        pos = float(np.clip(r.pos_in_range, 0, 1))
        # The pill and the filter both carry the heading; the reason survives in
        # the "why" column, so a BUY row still says whether it is an edge, a
        # breakout or a short being closed.
        sig = S.label(r.signal) if r.signal else ""
        colour = SIG_STYLE.get(sig, ("#8b94a5", ""))[0]
        prov = _provisional(r, bool(live_at))
        if not sig:
            pill = ""
        elif prov:
            # "WATCHLIST BUY" rather than "BUY": the same call, not yet earned. Kept
            # dashed as well as renamed, so it reads at a glance and on the label.
            pill = ('<span class="pill" style="color:%s;background:transparent;'
                    'border:1px dashed %s66;opacity:.8">'
                    '<span style="font-weight:400">WATCHLIST</span> %s</span>'
                    % (colour, colour, html.escape(sig)))
        else:
            pill = ('<span class="pill" style="color:%s;background:%s22;border:1px solid %s55">%s</span>'
                    % (colour, colour, colour, html.escape(sig)))
        body.append(
            '<tr id="%s" data-sig="%s" data-grp="%s" data-vol="%s">'
            '<td class="l"><span class="tk" style="color:%s">%s</span> <span class="grp">%s</span></td>'
            '<td data-v="%s">%s</td>'
            '<td class="opt" data-v="%s">%s</td><td class="opt" data-v="%s">%s</td>'
            '<td data-v="%.4f"><div class="bar"><u></u><i style="left:%.1f%%"></i></div></td>'
            '<td class="opt" data-v="%s">%s</td><td class="opt" data-v="%s">%s</td>'
            '<td class="opt" data-v="%s">%s</td><td class="opt" data-v="%s">%s</td>'
            '<td class="opt" data-v="%s">%s</td><td class="opt" data-v="%s">%s</td>'
            '<td class="opt" data-v="%s">%s</td>'
            '<td class="l">%s</td><td class="l opt why">%s</td></tr>'
            % (r.ticker, html.escape(sig), r.group, r.vol_flag or "",
               _trend_col(r.trend_bull),
               r.ticker, GROUP_LABEL.get(r.group, r.group),
               r.spot, _f(r.spot), r.range_low, _f(r.range_low), r.range_high, _f(r.range_high),
               pos, pos * 100,
               r.pct_to_low, _f(r.pct_to_low), r.pct_to_high, _f(r.pct_to_high),
               r.trade, _bull_cell(r.trade, r.trade_bull),
               r.trend, _bull_cell(r.trend, r.trend_bull),
               r.volume, _vol(r.volume),
               r.vol_z_1m, _z_cell(r.vol_z_1m),
               r.vol_z_3m, _z_cell(r.vol_z_3m),
               pill, html.escape(r.why or "")))

    cards = [("Names", len(df), "var(--line)")]
    for name in SIGNAL_ORDER:
        cards.append((name, counts.get(name, 0), SIG_STYLE[name][0]))
    # Neutral earns a card of its own: it is the reason a chunk of the list is
    # silent, and without it the silence looks like nothing happening.
    if "trend_neutral" in df and int(df["trend_neutral"].sum()):
        cards.append(("NEUTRAL TREND", int(df["trend_neutral"].sum()), "#8b94a5"))

    # The alert strip carries the two things that are only true right now: a line
    # crossed during this session, and price sitting at an edge of today's range.
    # Both are transient -- by tonight's close the crossing is just history and the
    # edge has usually been left behind -- which is what makes them worth the top of
    # the screen rather than a column in the table.
    alerts = []
    if "intraday" in df:
        for h in df[df["intraday"].astype(bool)].itertuples():
            if getattr(h, "is_index", False):
                label, stance = S.vol_read(cross=h.intraday)
                col = STANCE_COL.get(stance, FLAT)
                alerts.append((0, h.ticker, col, label, _f(h.spot), col))
                continue
            lost = h.intraday.startswith("lost")
            alerts.append((0, h.ticker, "#ef5350" if lost else "#5c9ded",
                           h.intraday, _f(h.spot), _trend_col(h.trend_bull)))
    seen = {a[1] for a in alerts}
    for h in df.itertuples():
        if h.ticker in seen or getattr(h, "cash_like", False):
            continue
        if getattr(h, "is_index", False):
            label, stance = S.vol_read(at_low=getattr(h, "at_low", False),
                                       at_high=getattr(h, "at_high", False))
            if label:
                col = STANCE_COL.get(stance, FLAT)
                alerts.append((1, h.ticker, col, label, _f(h.spot), col))
            continue
        if getattr(h, "at_low", False):
            alerts.append((1, h.ticker, BULL, "at the low end", _f(h.spot),
                           _trend_col(h.trend_bull)))
        elif getattr(h, "at_high", False):
            alerts.append((1, h.ticker, "#d9a441", "at the high end", _f(h.spot),
                           _trend_col(h.trend_bull)))
    alerts_html = ""
    if alerts:
        alerts.sort(key=lambda a: (a[0], a[1]))
        chips = "".join(
            '<a href="#%s" style="text-decoration:none;display:inline-flex;'
            'align-items:center;gap:7px;background:%s18;border:1px solid %s55;'
            'border-radius:8px;padding:6px 11px;color:var(--fg)">'
            '<b style="color:%s">%s</b><span style="color:%s;font-size:12px">%s</span>'
            '<span style="color:var(--dim);font-size:12px">%s</span></a>'
            % (tk, col, col, tkcol, tk, col, html.escape(label), spot)
            for _, tk, col, label, spot, tkcol in alerts)
        n_cross = sum(1 for a in alerts if a[0] == 0)
        n_edge = len(alerts) - n_cross
        bits = []
        if n_cross:
            bits.append("%d crossed a line" % n_cross)
        if n_edge:
            bits.append("%d at a range edge" % n_edge)
        alerts_html = (
            '<div style="border:1px solid #3d4a5c;background:#141a22;border-radius:10px;'
            'padding:13px 15px;margin-bottom:20px">'
            '<div style="font-size:11px;text-transform:uppercase;letter-spacing:.06em;'
            'color:var(--dim);margin-bottom:9px">Alert &middot; %s</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:8px">%s</div></div>'
            % (" &middot; ".join(bits), chips))

    idx = df[df["is_index"]] if "is_index" in df else df.iloc[0:0]
    vol_html = ""
    if len(idx):
        note, ncol = _vol_regime(idx)
        cells = "".join(
            '<div style="flex:1 1 220px;background:#11151b;border:1px solid var(--line);'
            'border-radius:8px;padding:10px 13px">'
            '<div><span class="tk" style="color:%s">%s</span>'
            '<span style="color:var(--dim);font-size:12px"> &nbsp;%s</span></div>'
            '<div style="color:var(--dim);font-size:11.5px;margin-top:3px">'
            'TRADE %s &middot; TREND %s</div>'
            '<div style="color:%s;font-size:11.5px;margin-top:2px">%s</div></div>'
            % (col, tk, _f(spot), _f(trade), _f(trend), col, read)
            for tk, spot, trade, trend, read, col in _index_rows(idx))
        vol_html = (
            '<h2 style="font-size:15px;margin:6px 0 9px;letter-spacing:-.01em">'
            'Market volatility</h2>'
            + _vix_meter(idx, band=(df.attrs.get("edge_buy"), df.attrs.get("edge_sell"))) +
            '<div style="color:%s;font-size:12.5px;margin-bottom:10px">%s</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:24px">%s</div>'
            % (ncol, note, cells))
    mac = _macro_block(df, dark=True)
    if mac:
        vol_html += (
            '<h2 style="font-size:15px;margin:6px 0 9px;letter-spacing:-.01em">Macro '
            '<span style="color:var(--dim);font-weight:400;font-size:13px">'
            '&middot; indices, yields, FX and commodities &middot; levels only, never a '
            'position</span></h2>'
            '<div class="tablewrap" style="margin-bottom:24px">'
            '<table style="min-width:460px"><tbody>%s</tbody></table></div>' % mac)

    pf_html = ""
    if book is not None and not book.empty:
        prow = []
        for b in book.itertuples():
            ac = P.ACTION_COLOUR.get(b.action, "#8b94a5")
            pc = "#0ea37f" if b.pnl_pct >= 0 else "#ef5350"
            tc = "#0ea37f" if (b.day_pct or 0) >= 0 else "#ef5350"
            sidec = "#0ea37f" if b.side == "long" else "#ef5350"
            days = ("%d" % b.days_held) if np.isfinite(b.days_held) else "&ndash;"
            prow.append(
                '<tr><td class="l"><span class="tk">%s</span></td>'
                '<td class="l" style="color:%s;text-transform:uppercase;font-size:11.5px;font-weight:650">%s</td>'
                '<td class="l" style="color:var(--dim)">%s</td>'
                '<td data-v="%s">%s</td>'
                '<td data-v="%s" style="font-weight:650">%s</td>'
                '<td data-v="%.4f" style="color:%s;font-weight:650">%+.2f%%</td>'
                '<td data-v="%.4f" style="color:%s">%+.2f%%</td>'
                '<td>%s</td>'
                '<td class="l"><span class="pill" style="color:%s;background:%s22;border:1px solid %s55">%s</span></td>'
                '<td class="l why">%s</td></tr>'
                % (b.ticker, sidec, b.side, b.entry_date,
                   b.entry_price, _f(b.entry_price),
                   b.spot, _f(b.spot),
                   b.pnl_pct, pc, b.pnl_pct,
                   b.day_pct, tc, b.day_pct,
                   days,
                   ac, ac, ac, html.escape(b.action),
                   html.escape(b.action_why or "")))
        pf_html = (
            '<h2 style="font-size:15px;margin:6px 0 10px;letter-spacing:-.01em">Portfolio '
            '<span style="color:var(--dim);font-weight:400;font-size:13px">'
            '&middot; %d open &middot; since entry %+.2f%% &middot; today %+.2f%%</span></h2>'
            '<div class="tablewrap" style="margin-bottom:24px"><table style="min-width:1000px">'
            '<thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>'
            % (len(book), book["pnl_pct"].mean(), book["day_pct"].mean(),
               "".join("<th>%s</th>" % h for h in
                       ["Position", "Side", "Added", "Entry",
                        "Spot", "P&L", "Today", "Days", "Action", "Why"]),
               "".join(prow)))

    if closed is not None and len(closed):
        pf_html += (
            '<h2 style="font-size:15px;margin:6px 0 10px;letter-spacing:-.01em">Closed today '
            '<span style="color:var(--dim);font-weight:400;font-size:13px">'
            '&middot; %d &middot; realised %+.2f%%</span></h2>'
            '<table style="margin-bottom:24px;width:100%%;max-width:640px">'
            '<tbody>%s</tbody></table>'
            % (len(closed), closed["pnl_pct"].mean(), _closed_block(closed, dark=True)))

    heads = ["ETF", "Spot", "Range low", "Range high", "In range",
             "% to low", "% to high", "TRADE", "TREND",
             "Volume", "z vs 1m", "z vs 3m", "Signal", "Why"]
    groups = [g for g in GROUP_LABEL if g in set(df["group"])]
    n_live = df.attrs.get("n_live", 0)
    if live_at:
        # Say how many of the signals the book would actually act on. Without this
        # the strip reads "20 buys" while the book takes two, and nothing on the page
        # accounts for the difference.
        n_sig = int(df["signal"].notna().sum())
        n_prov = sum(1 for r in df.itertuples() if _provisional(r, True))
        act = n_sig - n_prov
        stamp = ('<span style="color:var(--bull)">&#9679; live</span> &middot; '
                 "%d quotes at %s" % (n_live, live_at.strftime("%H:%M")))
        if n_sig:
            stamp += ('<br><span style="color:var(--dim);font-size:12px">'
                      '%d of %d signals have cleared a line and are actionable now. '
                      'The other %d show as WATCHLIST &mdash; price is at the edge of its '
                      'range this minute, which is not the same as something having '
                      'happened. They settle at the close.</span>' % (act, n_sig, n_prov))
        refresh = '<meta http-equiv="refresh" content="%d">' % REFRESH_SECONDS
    else:
        stamp = "generated %s" % generated.strftime("%Y-%m-%d %H:%M")
        refresh = ""

    return """<title>Macro Risk Range Signals</title>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
""" + PWA_HEAD + """
%s
<style>%s</style>
<div class="wrap">
<h1>Macro Risk Range Signals</h1>
<div class="sub">%s &middot; %s &middot; levels for this session, computed from the %s close
&middot; %s</div>
<div class="cards">%s</div>
%s
%s
%s
<div class="filters">
<div class="frow"><span class="flab">Action</span><div class="fbtns">%s</div></div>
<div class="frow"><span class="flab">Volume</span><div class="fbtns">
<button data-vol="surge" aria-pressed="false">Surge</button>
<button data-vol="dry" aria-pressed="false">Dry</button></div></div>
<div class="frow"><span class="flab">Category</span><div class="fbtns">%s</div></div>
</div>
<div class="tablewrap"><table id="scan"><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>
<footer>
Green level = price above it (bullish for that duration), red = below.
A ticker is coloured by TREND, so a red ticker inside a green &ldquo;at the low end&rdquo;
chip is a name at the bottom of its range that is below TREND &mdash; the one case the
handbook says not to buy. VIX and MOVE are the exception: they carry no position, so
they are coloured by what they mean for everything else &mdash; green when volatility
is falling, red when it is rising.
&ldquo;In range&rdquo; shows where spot sits between the low and high edge; the outer fifth at each end counts as &ldquo;at the end&rdquo;.
Volume is shown as a z-score of log volume against the fund's own 1-month and
3-month distributions; amber marks an unusually heavy session (z &ge; +2) and blue
an unusually light one (z &le; &minus;2).
</footer></div>
<script>%s</script>
""" % (refresh, CSS, _session_label(asof), _universe_label(df), asof, stamp,
       "".join('<div class="card" style="border-left-color:%s"><div class="k">%s</div>'
               '<div class="v">%s</div></div>' % (c, k, v) for k, v, c in cards),
       alerts_html,
       pf_html,
       vol_html,
       "".join('<button data-sig="%s" aria-pressed="false">%s</button>' % (s, s)
               for s in SIGNAL_ORDER),
       "".join('<button data-grp="%s" aria-pressed="false">%s</button>'
               % (g, GROUP_LABEL[g]) for g in groups),
       "".join('<th%s>%s</th>' % (' class="opt"' if h in OPTIONAL_COLS else "", h)
               for h in heads),
       "".join(body), JS)


# ----------------------------------------------------------------- newsletter
def _nl_section(title, colour, blurb, rows, kind, NAMES=None):
    NAMES = NAMES or {}
    if rows.empty:
        return ""
    items = []
    for r in rows.itertuples():
        if kind == "long":
            detail = ("buy zone <b>%s</b> &ndash; %s &middot; only %.1f%% above the low end "
                      "&middot; TRADE %s, TREND %s"
                      % (_f(r.range_low), _f(r.range_high), abs(r.pct_to_low),
                         _f(r.trade), _f(r.trend)))
        elif kind == "short":
            detail = ("range <b>%s</b> &ndash; <b>%s</b> &middot; only %.1f%% below the high end "
                      "&middot; TRADE %s, TREND %s"
                      % (_f(r.range_low), _f(r.range_high), abs(r.pct_to_high),
                         _f(r.trade), _f(r.trend)))
        else:
            detail = ("%s &middot; TRADE %s (%+.1f%%), TREND %s (%+.1f%%)"
                      % (html.escape(r.why or ""), _f(r.trade), r.pct_to_trade,
                         _f(r.trend), r.pct_to_trend))
        nm = NAMES.get(r.ticker, "")
        if np.isfinite(r.vol_z_3m):
            vcol = VOL_COLOUR.get(r.vol_flag, "#8b94a5")
            detail += ('<br><span style="color:%s">volume %s &middot; z %+.1f vs 1m '
                       '&middot; z %+.1f vs 3m%s</span>'
                       % (vcol, _vol(r.volume), r.vol_z_1m, r.vol_z_3m,
                          (" &middot; " + r.vol_flag.upper()) if r.vol_flag else ""))
        meta = " &nbsp;&middot;&nbsp; ".join(
            x for x in (html.escape(nm) if nm else "", _f(r.spot)) if x)
        items.append(
            '<tr><td style="padding:9px 0;border-bottom:1px solid #e6e8ec">'
            '<div><span style="font-weight:700;font-size:15px;color:#111">%s</span>%s'
            '<span style="color:#8b94a5;font-size:12px"> &nbsp;%s</span></div>'
            '<div style="color:#5a6270;font-size:12.5px;margin-top:2px">%s</div>'
            '</td></tr>' % (r.ticker, _new_badge(r), meta, detail))
    return (
        '<tr><td style="padding:22px 0 6px">'
        '<span style="display:inline-block;background:%s;color:#fff;font-size:12px;font-weight:700;'
        'letter-spacing:.06em;padding:4px 10px;border-radius:4px">%s &nbsp;(%d)</span>'
        '<div style="color:#5a6270;font-size:12.5px;margin-top:7px">%s</div></td></tr>'
        '<tr><td><table width="100%%" cellpadding="0" cellspacing="0">%s</table></td></tr>'
        % (colour, title, len(rows), blurb, "".join(items)))


def _explainer():
    """The "how to read this" block that opens every issue.

    Set small and tight: it sits above the signals, so it has to be skimmable by
    someone who already knows it and complete for someone who does not.
    """
    terms = [
        ("RISK RANGE",
         "The high and low price is likely to trade between today, from recent "
         "volatility. It is an envelope, not a direction call &mdash; it says how far, "
         "not which way. The edges are where decisions get made: at the low end you "
         "are being offered the best price of the session, at the high end the worst. "
         "Levels are fixed by yesterday&rsquo;s close and hold all day, so price moves "
         "through them rather than with them."),
        ("TRADE",
         "The short-duration line. Price above it is bullish TRADE, below is bearish. "
         "It is the tactical read &mdash; where to add inside a position you already "
         "hold, and the first thing to break when momentum turns."),
        ("TREND",
         "The cycle line, and the one that matters. Price above it is bullish TREND. A "
         "break of TREND is a regime change, not a wobble; a break of TRADE while TREND "
         "holds is usually noise inside an intact move."),
        ("NEUTRAL",
         "A third state, and the most common one. When price sits within 1.5% of its "
         "own TREND line it is too close to call, so the name raises no signal at all "
         "&mdash; a crossing there is noise rather than a change of regime. Neutral "
         "names are greyed out and simply do not appear in the sections below."),
        ("PUTTING THEM TOGETHER",
         "Both bullish is trending long; both bearish is trending short; one of each is "
         "counter-trend. The rule that saves the most money: do not buy the low end of "
         "the RANGE while TRADE is broken &mdash; wait for TREND to hold. Those names "
         "are held back rather than listed as a BUY."),
        ("NEW",
         "Marks an instruction that changed since the last issue. Anything unmarked is "
         "one you have already been given and that still stands &mdash; the same call, "
         "not a fresh one."),
        ("VOLUME",
         "How unusual today&rsquo;s volume is for that name, measured in standard "
         "deviations rather than percent &mdash; a +60% day is routine for a thin fund "
         "and remarkable for a large one. Beyond &plusmn;2 is flagged. It does not "
         "generate a signal on its own; it tells you whether a move happened on "
         "conviction or on nobody trading."),
    ]
    rows = "".join(
        '<div style="margin-top:9px">'
        '<span style="font-weight:700;color:#3d4552;font-size:11px;letter-spacing:.05em">'
        '%s</span><br>'
        '<span style="color:#6b7280;font-size:11.5px;line-height:1.5">%s</span></div>'
        % (term, body) for term, body in terms)
    return (
        '<tr><td style="padding:14px 0 4px">'
        '<div style="background:#fafbfc;border:1px solid #e6e8ec;border-radius:8px;'
        'padding:12px 14px 14px">'
        '<div style="font-weight:700;font-size:10.5px;letter-spacing:.07em;color:#8b94a5">'
        'HOW TO READ THIS</div>%s</div></td></tr>' % rows)


def render_newsletter(df, params, generated=None, book=None, closed=None):
    generated = generated or dt.datetime.now()
    asof = df["asof"].max() if len(df) else "-"
    b = S.buckets(df)

    names = short_names()
    pf = ""
    if book is not None and not book.empty:
        items = []
        for pos in book.itertuples():
            ac = P.ACTION_COLOUR.get(pos.action, "#8b94a5")
            pc = "#0b8f6e" if pos.pnl_pct >= 0 else "#d33"
            items.append(
                '<tr><td style="padding:9px 0;border-bottom:1px solid #e6e8ec">'
                '<div><span style="font-weight:700;font-size:15px;color:#111">%s</span>'
                '<span style="color:#8b94a5;font-size:12px"> &nbsp;%s%s since %s</span>'
                '<span style="float:right;font-weight:700;color:%s">%+.2f%%</span></div>'
                '<div style="color:#5a6270;font-size:12.5px;margin-top:2px">'
                'entry <b>%s</b> &rarr; spot <b>%s</b>'
                '<span style="color:#8b94a5"> &middot; %s day%s held &middot; '
                'today %+.2f%%</span></div>'
                '<div style="margin-top:4px"><span style="background:%s;color:#fff;font-size:11px;'
                'font-weight:700;padding:2px 7px;border-radius:3px">%s</span>'
                '<span style="color:#5a6270;font-size:12.5px"> &nbsp;%s</span></div></td></tr>'
                % (pos.ticker,
                   (html.escape(names.get(pos.ticker, "")) + " &nbsp;&middot;&nbsp; ")
                   if names.get(pos.ticker) else "",
                   pos.side, pos.entry_date, pc, pos.pnl_pct,
                   _f(pos.entry_price), _f(pos.spot),
                   ("%d" % pos.days_held) if np.isfinite(pos.days_held) else "&ndash;",
                   "" if pos.days_held == 1 else "s",
                   pos.day_pct,
                   ac, html.escape(pos.action), html.escape(pos.action_why or "holding")))
        pf = ('<tr><td style="padding:20px 0 6px">'
              '<span style="display:inline-block;background:#111;color:#fff;font-size:12px;'
              'font-weight:700;letter-spacing:.06em;padding:4px 10px;border-radius:4px">'
              'PORTFOLIO &nbsp;(%d open)</span>'
              '</td></tr>'
              '<tr><td><table width="100%%" cellpadding="0" cellspacing="0">%s</table></td></tr>'
              % (len(book), "".join(items)))

    cl = ""
    if closed is not None and len(closed):
        realised = closed["pnl_pct"].mean()
        cl = ('<tr><td style="padding:20px 0 6px">'
              '<span style="display:inline-block;background:#111;color:#fff;font-size:12px;'
              'font-weight:700;letter-spacing:.06em;padding:4px 10px;border-radius:4px">'
              'CLOSED TODAY &nbsp;(%d &middot; %+.2f%% average)</span></td></tr>'
              '<tr><td><table width="100%%" cellpadding="0" cellspacing="0">%s</table></td></tr>'
              % (len(closed), realised, _closed_block(closed, dark=False)))
    pf = pf + cl

    # The neutral state is defined in the explainer and counted in the chip row, so
    # it does not also need a paragraph of its own above the sections.
    n_neutral = int(df["trend_neutral"].sum()) if "trend_neutral" in df else 0

    blurbs = {
        "BUY": ("Buy. Price at or near the LOW end of the Risk Range with TRADE and "
                "TREND still bullish."),
        "SELL SOME": ("Trim. TRADE has broken while TREND still holds, so the position "
                      "comes down - TREND is what decides whether you hold at all. The "
                      "book carries no size, so it books the reduction as the whole lot "
                      "and reports what it made under CLOSED TODAY."),
        "SELL": ("Sell. TREND has broken, which is a regime change rather than a "
                 "wobble, so the long comes off entirely."),
        "SELL SHORT": ("Open a short, or avoid if long-only. A bearish TREND has rallied "
                       "into the HIGH end of its Risk Range. TREND decides the side; "
                       "TRADE only times it, so a reclaimed TRADE does not veto the short."),
        "BUY SOME": ("Buy back part of the short. TRADE has been reclaimed while TREND "
                     "is still bearish, so the short comes down but the bearish call "
                     "stands. The book carries no size, so it buys the whole short back "
                     "and reports what it made under CLOSED TODAY."),
        "COVER SHORT": ("TREND has been reclaimed. Close the short out."),
        "WATCHLIST": ("Nothing to act on yet. At an extreme of the Risk Range but the "
                      "signal has not confirmed, or a break that failed to hold."),
    }
    kinds = {"BUY": "long", "SELL SHORT": "short"}
    sections = "".join(
        _nl_section(name, SIG_STYLE[name][0], blurbs[name], b[name],
                    kinds.get(name, "event"), names)
        for name in S.SECTIONS)
    if not sections:
        sections = ('<tr><td style="padding:26px 0;color:#5a6270">No ETF triggered a signal '
                    'today. Everything on the list is mid-range with no fresh line breaks.</td></tr>')

    # volume outliers, independent of the price signals
    outl = df[df["vol_flag"].astype(bool)] if "vol_flag" in df else df.iloc[0:0]
    if len(outl):
        rowsv = []
        for r in outl.sort_values("vol_z", ascending=False).itertuples():
            col = VOL_COLOUR.get(r.vol_flag, "#8b94a5")
            rowsv.append(
                '<tr><td style="padding:9px 0;border-bottom:1px solid #e6e8ec">'
                '<div><span style="font-weight:700;color:#111">%s</span>'
                '<span style="color:#8b94a5;font-size:12px"> &nbsp;%s</span>'
                '<span style="float:right;font-weight:700;color:%s">%s</span></div>'
                '<div style="color:#5a6270;font-size:12.5px;margin-top:3px">'
                'traded <b>%s</b> shares'
                '<br>1-month average <b>%s</b> &nbsp;&rarr;&nbsp; %+.0f%% '
                '<span style="color:%s">(z %+.1f)</span>'
                '<br>3-month average <b>%s</b> &nbsp;&rarr;&nbsp; %+.0f%% '
                '<span style="color:%s">(z %+.1f)</span>'
                '</div></td></tr>'
                % (r.ticker, html.escape(names.get(r.ticker, "")), col, r.vol_flag.upper(),
                   _vol(r.volume),
                   _vol(r.vol_1m_avg), r.vol_vs_1m_pct, col, r.vol_z_1m,
                   _vol(r.vol_3m_avg), r.vol_vs_3m_pct, col, r.vol_z_3m))
        sections += (
            '<tr><td style="padding:22px 0 6px">'
            '<span style="display:inline-block;background:#6b4fa8;color:#fff;font-size:12px;'
            'font-weight:700;letter-spacing:.06em;padding:4px 10px;border-radius:4px">'
            'VOLUME OUTLIERS &nbsp;(%d)</span>'
            '<div style="color:#5a6270;font-size:12.5px;margin-top:7px">'
            'Sessions where volume was unusually heavy or light for that fund. Shown '
            'against both the 1-month and the 3-month average, in shares and as a '
            'z-score of log volume against that window.</div></td></tr>'
            '<tr><td><table width="100%%" cellpadding="0" cellspacing="0">%s</table></td></tr>'
            % (len(outl), "".join(rowsv)))

    # The volatility complex reads before the signals rather than after them: which
    # VIX bucket you are in decides how much any of the signals below is worth
    # acting on, so it belongs above them, not in an appendix.
    idx = df[df["is_index"]] if "is_index" in df else df.iloc[0:0]
    if len(idx):
        note, ncol = _vol_regime(idx)
        items = "".join(
            '<tr><td style="padding:8px 0;border-bottom:1px solid #e6e8ec">'
            '<div><span style="font-weight:700;color:#111">%s</span>'
            '<span style="color:#8b94a5;font-size:12px"> &nbsp;%s</span>'
            '<span style="float:right;color:%s;font-size:12.5px;font-weight:600">%s</span></div>'
            '<div style="color:#5a6270;font-size:12.5px;margin-top:2px">'
            'TRADE %s &middot; TREND %s</div></td></tr>'
            % (tk, _f(spot), col, read, _f(trade), _f(trend))
            for tk, spot, trade, trend, read, col in _index_rows(idx))
        sections = (
            '<tr><td style="padding:22px 0 6px">'
            '<span style="display:inline-block;background:#334155;color:#fff;font-size:12px;'
            'font-weight:700;letter-spacing:.06em;padding:4px 10px;border-radius:4px">'
            'MARKET VOLATILITY</span>'
            '<div style="margin-top:9px">' + _vix_meter(idx, email=True, band=(df.attrs.get("edge_buy"), df.attrs.get("edge_sell"))) + '</div>'
            '<div style="color:%s;font-size:12.5px;margin-top:7px">%s</div></td></tr>'
            '<tr><td><table width="100%%" cellpadding="0" cellspacing="0">%s</table></td></tr>'
            % (ncol, note, items)) + sections

    # Macro sits directly under volatility, because both frame the list rather than
    # instructing anything in it -- Hedgeye run their "Our Levels" the same way.
    mac = _macro_block(df, dark=False)
    if mac:
        sections = (
            '<tr><td style="padding:22px 0 6px">'
            '<span style="display:inline-block;background:#334155;color:#fff;font-size:12px;'
            'font-weight:700;letter-spacing:.06em;padding:4px 10px;border-radius:4px">'
            'MACRO</span></td></tr>'
            '<tr><td><table width="100%%" cellpadding="0" cellspacing="0">%s</table></td></tr>'
            % mac) + sections

    # appendix: every name, compact
    app = []
    tradeable = df
    for col in ("is_index", "is_macro"):
        if col in tradeable:
            tradeable = tradeable[~tradeable[col].astype(bool)]
    for g, grp in tradeable.groupby("group", sort=False):
        app.append('<tr><td colspan="8" style="padding:14px 0 4px;font-weight:700;font-size:12px;'
                   'color:#8b94a5;letter-spacing:.05em;text-transform:uppercase">%s</td></tr>'
                   % GROUP_LABEL.get(g, g))
        for r in grp.sort_values("ticker").itertuples():
            tc = "#0ea37f" if r.trade_bull else "#ef5350"
            nc = ("#0ea37f" if r.trend_bull is True else
                  "#ef5350" if r.trend_bull is False else "#8b94a5")
            app.append(
                '<tr>'
                '<td style="padding:3px 6px 3px 0;color:%s">'
                '<span style="font-weight:700">%s</span>'
                '<span style="color:#9aa1ad;font-weight:400;font-size:11px"> %s</span></td>'
                '<td align="right" style="padding:3px 6px">%s</td>'
                '<td align="right" style="padding:3px 6px;color:#5a6270">%s</td>'
                '<td align="right" style="padding:3px 6px;color:#5a6270">%s</td>'
                '<td align="right" style="padding:3px 6px;color:%s">%s</td>'
                '<td align="right" style="padding:3px 0 3px 6px;color:%s">%s</td>'
                '<td align="right" style="padding:3px 0 3px 10px">%s</td>'
                '<td align="right" style="padding:3px 0 3px 6px">%s</td>'
                '</tr>' % (nc, r.ticker, html.escape(names.get(r.ticker, "")),
                           _f(r.spot), _f(r.range_low), _f(r.range_high),
                           tc, _f(r.trade), nc, _f(r.trend),
                           _z_cell(r.vol_z_1m), _z_cell(r.vol_z_3m)))

    counts = df["signal"].map(S.LABEL).value_counts().to_dict()
    chips = "".join(
        '<span style="display:inline-block;background:%s22;color:%s;border:1px solid %s55;'
        'border-radius:999px;padding:3px 10px;font-size:12px;font-weight:600;margin:0 6px 6px 0">'
        '%s %d</span>' % (SIG_STYLE[s][0], SIG_STYLE[s][0], SIG_STYLE[s][0], s,
                          counts.get(s, 0))
        # A chip for a section that is not in the letter is a promise the letter does
        # not keep -- WATCHLIST has been reading "0" against no section for weeks.
        for s in SIGNAL_ORDER if counts.get(s, 0))
    # Neutral is a state, not the absence of one, so it is counted where the reader
    # can see it. These names raise no directional signal at all.
    if n_neutral:
        chips += ('<span style="display:inline-block;background:#8b94a522;color:#5a6270;'
                  'border:1px solid #8b94a555;border-radius:999px;padding:3px 10px;'
                  'font-size:12px;font-weight:600;margin:0 6px 6px 0">'
                  'NEUTRAL TREND %d</span>' % n_neutral)

    return """<div style="background:#f4f5f7;padding:22px 0;font-family:-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif">
<table width="100%%" cellpadding="0" cellspacing="0"><tr><td align="center">
<table width="640" cellpadding="0" cellspacing="0" style="width:640px;max-width:100%%;background:#fff;
border:1px solid #e0e3e8;border-radius:10px;padding:26px 30px">

<tr><td style="padding-bottom:4px">
<div style="font-size:21px;font-weight:750;color:#111;letter-spacing:-.2px">Macro Risk Range Signals</div>
<div style="color:#8b94a5;font-size:13px;margin-top:3px">%s &middot; %s &middot;
levels for this session, off the %s close</div>
</td></tr>
<tr><td style="padding:12px 0 2px">%s</td></tr>
%s
%s
%s

<tr><td style="padding:26px 0 6px;border-top:1px solid #e6e8ec">
<div style="font-weight:700;font-size:12px;color:#8b94a5;letter-spacing:.06em">FULL LIST</div>
<div style="color:#8b94a5;font-size:11.5px;margin-top:3px">
spot &middot; range low &middot; range high &middot; <span style="color:#0ea37f">TRADE</span> &middot;
<span style="color:#0ea37f">TREND</span> (green = price above the line) &middot; then the
volume z-score vs the 1-month and vs the 3-month distribution</div>
</td></tr>
<tr><td><table width="100%%" cellpadding="0" cellspacing="0" style="font-size:12.5px">%s</table></td></tr>
</table></td></tr></table></div>
""" % (_session_label(asof), _universe_label(df), asof, chips,
       _explainer(), pf, sections,
       "".join(app))


# --------------------------------------------------------------------- driver
def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the Macro Risk Range dashboard and newsletter.")
    ap.add_argument("--tickers", default=None, help="override the watchlist")
    ap.add_argument("--profile", default="hedgeye_anchor",
                    help="range profile: hedgeye_anchor | anchor_ewma | hedgeye_vol")
    ap.add_argument("--edge", type=float, default=None,
                    help='pin the range-edge band on both sides (default: scaled by the VIX)')
    ap.add_argument("--fresh-days", type=int, default=S.FRESH_DAYS)
    ap.add_argument("--outdir", default=repo_path("out"))
    ap.add_argument("--portfolio", default=None, help="portfolio CSV (default data/portfolio.csv)")
    ap.add_argument("--live", action="store_true",
                    help="re-price against live quotes; levels stay from the last close")
    ap.add_argument("--sync", action="store_true",
                    help="open portfolio positions for today's range breaks")
    ap.add_argument("--push", action="store_true",
                    help="push new alerts to the phone (see fractal.app.alerts --setup)")
    args = ap.parse_args(argv)

    params = load_params()
    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    df = S.run(tickers, params=params, profile=args.profile,
               edge=args.edge, fresh_days=args.fresh_days)
    if df.empty:
        print("no data")
        return 1
    if args.live:
        from . import live as LIVE
        df = LIVE.reprice(df, edge=args.edge)
        at = df.attrs.get("live_at")
        if at is None:
            print("[live] no quote newer than the %s close - report left as the daily build"
                  % df["asof"].max())
        else:
            print("[live] re-priced %d of %d names at %s"
                  % (df.attrs.get("n_live", 0), len(df), at.strftime("%H:%M")))

    # What changed since the session we last published. The dashboard still shows
    # every name -- that is the pull surface -- but the newsletter and the phone
    # only carry what is new, which is Hedgeye's own "signal, not noise" rule.
    df = S.mark_new(df)

    eff = dict(params)
    eff["range"] = dict(params["range"])
    eff["range"]["active"] = args.profile

    os.makedirs(args.outdir, exist_ok=True)
    stamp = dt.date.today().isoformat()
    paths = {
        "dashboard": os.path.join(args.outdir, "etf_dashboard.html"),
        "newsletter": os.path.join(args.outdir, "etf_newsletter.html"),
        "csv": os.path.join(args.outdir, "etf_signals_%s.csv" % stamp),
    }
    # On a daily run the whole ladder drives the book. On a live run only a clean
    # intraday break does -- and only when the quotes are actually fresh, since a
    # stale-quote pass re-prices nothing and would act on the close twice.
    if args.sync:
        if not args.live:
            P.sync(df, custom=args.portfolio)
        elif df.attrs.get("live_at") is not None:
            P.sync(df, custom=args.portfolio, only_intraday=True)
    book = P.reconcile(df, custom=args.portfolio, live=args.live)
    # A reduction is only real once the lot is off, so both surfaces report it. The
    # exit is stamped with the close it was decided on, which for an intraday run is
    # still the prior close -- hence both dates.
    _a = str(df.attrs.get("asof", stamp))
    _n = next_session(_a)
    stamps = {_a} | ({_n.strftime("%Y-%m-%d")} if _n is not None else set())
    closed = pd.concat([P.closed_on(d, custom=args.portfolio) for d in sorted(stamps)],
                       ignore_index=True)
    with open(paths["dashboard"], "w", encoding="utf-8") as fh:
        fh.write(render_dashboard(df, eff, book=book, closed=closed))
    with open(paths["newsletter"], "w", encoding="utf-8") as fh:
        fh.write(render_newsletter(df, eff, book=book, closed=closed))
    df.to_csv(paths["csv"], index=False)

    if not book.empty:
        print("portfolio: %d open | since entry %+.2f%% | since the baseline close %+.2f%% | %s"
              % (len(book), book["pnl_pct"].mean(), book["day_pct"].mean(),
                 "  ".join("%s=%d" % kv for kv in book["action"].value_counts().items())))
    counts = df["signal"].map(S.LABEL).value_counts().to_dict()
    print("%s | %s" % (_universe_label(df), "  ".join(
        "%s=%d" % (k, counts.get(k, 0))
        for k in SIGNAL_ORDER)))
    n_new = int(df["is_new"].sum()) if "is_new" in df else 0
    n_std = int((df["signal"].notna() & ~df["is_new"]).sum()) if "is_new" in df else 0
    print("changes: %d new, %d already standing" % (n_new, n_std))
    # Only a completed daily build advances the record of what the reader has been
    # told. A live re-price during the session must not, or the first intraday run
    # would mark the whole day's set as seen and the rest would look like silence.
    if not args.live:
        S.save_state(str(df.attrs.get("asof", stamp)), df)

    for k, v in paths.items():
        print("wrote %s" % v)

    # Pushed last, and only from the written state, so a failure to render never
    # results in a phone alert for a report that does not exist.
    if args.push:
        from . import alerts as A
        A.notify(df, asof=str(df.attrs.get("asof", stamp)), seed=not args.live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
