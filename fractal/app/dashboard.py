"""Render the daily scan as a self-contained HTML dashboard.

No external assets: the file opens straight from disk. Each row shows where spot
sits inside the RANGE as a bar, so "at the top / at the bottom" reads at a glance
rather than needing the percentages to be compared by eye.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html

import numpy as np
import pandas as pd

from ..data.loader import load_params, repo_path
from . import scan as scan_mod

STATE_LABEL = {
    "trending_long": "Trending long",
    "trending_short": "Trending short",
    "counter_trend": "Counter-trend",
}

CSS = """
:root{--bg:#0d0f13;--panel:#151920;--line:#242a34;--fg:#e6e9ef;--dim:#8b94a5;
--bull:#0ea37f;--bear:#ef5350;--warn:#d9a441}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1500px;margin:0 auto;padding:28px 20px 60px}
h1{font-size:20px;margin:0 0 4px;letter-spacing:-.01em}
.sub{color:var(--dim);font-size:13px;margin-bottom:22px}
.cards{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:10px 14px;min-width:120px}
.card .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.06em}
.card .v{font-size:19px;font-weight:600;margin-top:2px}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
button{background:var(--panel);color:var(--fg);border:1px solid var(--line);
border-radius:999px;padding:6px 13px;font-size:12.5px;cursor:pointer}
button:hover{border-color:#39414f}
button[aria-pressed="true"]{background:#1d2530;border-color:#3d4a5c;color:#fff}
.tablewrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;min-width:1080px}
th,td{padding:8px 10px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th{position:sticky;top:0;background:#11151b;color:var(--dim);font-weight:600;
font-size:11px;text-transform:uppercase;letter-spacing:.05em;cursor:pointer;z-index:1}
th:first-child,td:first-child{text-align:left}
tbody tr:hover{background:#1a1f27}
.tk{font-weight:600}
.grp{color:var(--dim);font-size:11.5px}
.bull{color:var(--bull)}.bear{color:var(--bear)}
.pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11.5px;
border:1px solid var(--line)}
.pill.long{color:var(--bull);border-color:rgba(14,163,127,.4);background:rgba(14,163,127,.10)}
.pill.short{color:var(--bear);border-color:rgba(239,83,80,.4);background:rgba(239,83,80,.10)}
.pill.counter{color:var(--warn);border-color:rgba(217,164,65,.4);background:rgba(217,164,65,.10)}
.bar{position:relative;width:150px;height:8px;border-radius:4px;background:#1e242e;
display:inline-block;vertical-align:middle}
.bar i{position:absolute;top:-3px;width:3px;height:14px;border-radius:2px;background:var(--fg)}
.bar u{position:absolute;top:0;height:8px;border-radius:4px;
background:linear-gradient(90deg,rgba(239,83,80,.28),rgba(14,163,127,.28));left:0;right:0}
.flags{color:var(--dim);font-size:11.5px}
.flag{color:var(--warn)}
footer{color:var(--dim);font-size:12px;margin-top:22px;line-height:1.7}
code{background:#11151b;border:1px solid var(--line);border-radius:4px;padding:1px 5px}
"""

JS = """
const rows=[...document.querySelectorAll('tbody tr')];
let dir={};
document.querySelectorAll('th').forEach((th,i)=>{
  th.onclick=()=>{
    dir[i]=!dir[i];
    const tb=document.querySelector('tbody');
    rows.slice().sort((a,b)=>{
      const x=a.children[i].dataset.v??a.children[i].textContent;
      const y=b.children[i].dataset.v??b.children[i].textContent;
      const nx=parseFloat(x),ny=parseFloat(y);
      const c=(!isNaN(nx)&&!isNaN(ny))?nx-ny:String(x).localeCompare(String(y));
      return dir[i]?-c:c;
    }).forEach(r=>tb.appendChild(r));
  };
});
document.querySelectorAll('button[data-filter]').forEach(b=>{
  b.onclick=()=>{
    const on=b.getAttribute('aria-pressed')==='true';
    document.querySelectorAll('button[data-filter]').forEach(o=>o.setAttribute('aria-pressed','false'));
    b.setAttribute('aria-pressed', on?'false':'true');
    const f=on?null:b.dataset.filter;
    rows.forEach(r=>{r.style.display=(!f||r.dataset[f]==='1')?'':'none';});
  };
});
"""


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "&ndash;"
    return ("%%.%df" % nd) % v


def _cell(v, nd=2, cls=""):
    txt = _fmt(v, nd)
    dv = "" if txt == "&ndash;" else ' data-v="%s"' % v
    c = ' class="%s"' % cls if cls else ""
    return "<td%s%s>%s</td>" % (dv, c, txt)


def _level_cell(level, bull, spot):
    if level is None or not np.isfinite(level):
        return '<td data-v="">&ndash;</td>'
    cls = "bull" if bull else ("bear" if bull is not None else "")
    return '<td data-v="%s" class="%s">%s</td>' % (level, cls, _fmt(level))


def render(df, params, generated=None):
    generated = generated or dt.datetime.now()
    asof = df["asof"].max() if len(df) else "-"
    counts = df["state"].value_counts().to_dict() if len(df) else {}

    body = []
    for r in df.itertuples():
        pos = getattr(r, "pos_in_range", None)
        pos = None if pos is None or not np.isfinite(pos) else float(np.clip(pos, 0, 1))
        bar = ('<div class="bar"><u></u><i style="left:%.1f%%"></i></div>'
               % (pos * 100 if pos is not None else 50)) if pos is not None else "&ndash;"

        state = getattr(r, "state", None)
        pill_cls = {"trending_long": "long", "trending_short": "short",
                    "counter_trend": "counter"}.get(state, "")
        pill = '<span class="pill %s">%s</span>' % (pill_cls, STATE_LABEL.get(state, "&ndash;"))

        flags = [n for n in ("at_top", "at_bottom", "first_break_trade", "cross", "trend_flip")
                 if getattr(r, n, False)]
        flag_html = " ".join('<span class="flag">%s</span>' % f.replace("_", " ")
                             for f in flags) or "&ndash;"

        data = " ".join('data-%s="%d"' % (n, 1 if getattr(r, n, False) else 0)
                        for n in ("at_top", "at_bottom", "first_break_trade", "cross", "trend_flip"))
        body.append(
            "<tr %s>"
            '<td><span class="tk">%s</span> <span class="grp">%s</span></td>'
            "%s%s%s%s%s"
            '<td data-v="%s">%s</td>'
            "%s%s%s"
            "<td>%s</td>"
            '<td class="flags">%s</td>'
            "</tr>" % (
                data, html.escape(str(r.ticker)), html.escape(str(getattr(r, "group", ""))),
                _cell(r.spot, 2),
                _cell(getattr(r, "range_low", None), 2),
                _cell(getattr(r, "range_high", None), 2),
                _cell(getattr(r, "pct_to_low", None), 2),
                _cell(getattr(r, "pct_to_high", None), 2),
                "" if pos is None else pos, bar,
                _level_cell(getattr(r, "trade", None), getattr(r, "trade_bull", None), r.spot),
                _level_cell(getattr(r, "trend", None), getattr(r, "trend_bull", None), r.spot),
                _level_cell(getattr(r, "tail", None), getattr(r, "tail_bull", None), r.spot),
                pill,
                flag_html,
            ))

    heads = ["Ticker", "Spot", "Range low", "Range high", "% to low", "% to high",
             "In range", "TRADE", "TREND", "TAIL", "State", "Signals"]
    cards = [
        ("Names", len(df)),
        ("Trending long", counts.get("trending_long", 0)),
        ("Trending short", counts.get("trending_short", 0)),
        ("Counter-trend", counts.get("counter_trend", 0)),
        ("At range edge", int(df.get("at_top", pd.Series(dtype=bool)).sum()
                              + df.get("at_bottom", pd.Series(dtype=bool)).sum())),
    ]
    lines = params["lines"]
    rng = params["range"][params["range"]["active"]]
    recipe = ("TRADE %s(%s) &middot; TREND %s(%s) &middot; RANGE EMA%s anchor, "
              "&lambda;=%s, +%.2f&sigma;/-%.2f&sigma;" % (
                  lines["trade"].get("family", "ema").upper(),
                  lines["trade"].get("window", lines["trade"].get("span")),
                  lines["trend"].get("family", "ema").upper(),
                  lines["trend"].get("window", lines["trend"].get("span")),
                  rng.get("anchor_span", 1), rng["lam"],
                  rng.get("m_up", rng.get("m", 1)), rng.get("m_dn", rng.get("m", 1))))

    return """<title>Fractal Trend &amp; Range</title>
<style>%s</style>
<div class="wrap">
<h1>Fractal Trend &amp; Range</h1>
<div class="sub">Levels for the next session, computed from the bar closing %s &middot;
generated %s</div>
<div class="cards">%s</div>
<div class="controls">
<button data-filter="atTop" aria-pressed="false">At top of range</button>
<button data-filter="atBottom" aria-pressed="false">At bottom of range</button>
<button data-filter="firstBreakTrade" aria-pressed="false">Break of TRADE</button>
<button data-filter="cross" aria-pressed="false">TRADE/TREND cross</button>
<button data-filter="trendFlip" aria-pressed="false">TREND flip</button>
</div>
<div class="tablewrap"><table><thead><tr>%s</tr></thead><tbody>%s</tbody></table></div>
<footer>
Green level = price above it (bullish for that duration), red = below.
&ldquo;In range&rdquo; marks where spot sits between the low and high edge.<br>
Model: %s<br>
This is a reconstruction fitted to published Hedgeye Risk Range&trade; and Similar Set
levels, not their proprietary math. See <code>README.md</code> for residuals.
</footer>
</div>
<script>%s</script>
""" % (CSS, asof, generated.strftime("%Y-%m-%d %H:%M"),
       "".join('<div class="card"><div class="k">%s</div><div class="v">%s</div></div>'
               % (k, v) for k, v in cards),
       "".join("<th>%s</th>" % h for h in heads),
       "".join(body), recipe, JS)


def main():
    ap = argparse.ArgumentParser(description="Render the daily scan to HTML.")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--out", default=repo_path("out", "dashboard.html"))
    ap.add_argument("--source", default=None, choices=["ib", "yahoo"])
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    params = load_params()
    df = scan_mod.run(tickers, params=params, source=args.source)
    if df.empty:
        print("nothing to render")
        return
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(render(df, params))
    print("wrote %s  (%d names)" % (args.out, len(df)))


if __name__ == "__main__":
    main()
