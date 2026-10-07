"""TRADE / TREND / RANGE charts for the IBKR book, mailed at the open and the close.

One chart per underlying held in the brokerage account: price over the last six
months, the two duration lines drawn across it, and the Risk Range band shaded on
the right where price is actually trading against it. The read a Similar Set mail
gives you in a picture, for the names that are actually at risk rather than for the
whole 255-name universe.

Why the names come from a file rather than from IBKR directly: the brokerage
connector is an MCP tool available to Claude inside a session, not to a scheduled
python process. `fractal/data/ibkr_names.txt` is therefore the source of truth and
is refreshed on request from a live account read. A stale list shows stale NAMES --
it cannot show stale LEVELS, which are recomputed on every run.

Sent to the owner only. Positions are private in a way the newsletter is not.
"""
import argparse
import datetime as dt
import io
import os
from email.message import EmailMessage
from email.utils import make_msgid

import matplotlib
matplotlib.use("Agg")                     # no display on a scheduled job
import matplotlib.pyplot as plt
import numpy as np

from . import signals as S
from . import publish
from ..data.loader import load_params, load_prices, repo_path

OWNER = "vazd17@gmail.com"
NAMES_FILE = ("data", "ibkr_names.txt")
LOOKBACK = 130                            # sessions drawn, about six months

INK = "#11151b"
DIM = "#8b94a5"
LINE = "#dfe3e8"
GREEN = "#0ea37f"
RED = "#ef5350"
AMBER = "#d9a441"
BLUE = "#5c9ded"


def held_names(path=None):
    """The underlyings to chart. Blank lines and '#' comments ignored."""
    p = path or repo_path(*NAMES_FILE)
    out = []
    with io.open(p, encoding="utf-8") as fh:
        for raw in fh:
            t = raw.split("#")[0].strip()
            if t:
                out.append(t.upper())
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _read(o):
    """(label, colour) for the two durations together -- the headline of the chart."""
    tb, trn = o.get("trade_bull"), o.get("trend_bull")
    if trn is True and tb:
        return "bullish TRADE and TREND", GREEN
    if trn is False and not tb:
        return "bearish TRADE and TREND", RED
    if trn is None:
        return "TREND on the line - no conviction", DIM
    if trn is True and not tb:
        return "bullish TREND, TRADE broken", AMBER
    return "bearish TREND, TRADE reclaimed", AMBER


def chart(ticker, ohlc, params, vix):
    """One PNG: price, the two lines, and the live RANGE band. Returns (bytes, read)."""
    eb, es = S.edge_for_gauge(vix, S.gauge_for(ticker)) or (S.EDGE_BUY, S.EDGE_SELL)
    o = S.evaluate(ticker, ohlc, params, edge_buy=eb, edge_sell=es)
    if not o:
        return None, None
    close = ohlc["Close"].dropna()
    from ..model import adaptive_ma
    lines = adaptive_ma.compute(close, params)
    c = close.iloc[-LOOKBACK:]
    trade = lines["trade"].reindex(close.index).iloc[-LOOKBACK:]
    trend = lines["trend"].reindex(close.index).iloc[-LOOKBACK:]
    x = np.arange(len(c))

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # The RANGE is a statement about where price trades NEXT, so it is drawn as a
    # band on the forward edge rather than smeared back over history it never
    # governed. Tinted by what the model would DO there: buy the low, short the high.
    lo, hi = o["range_low"], o["range_high"]
    span = max(len(c) // 6, 8)
    ax.axhspan(lo, hi, xmin=1 - span / max(len(c), 1), xmax=1.0,
               color="#5c9ded", alpha=0.10, zorder=0)
    buy_top = lo + (hi - lo) * eb
    sell_bot = hi - (hi - lo) * es
    ax.axhspan(lo, buy_top, xmin=1 - span / max(len(c), 1), xmax=1.0,
               color=GREEN, alpha=0.16, zorder=0)
    ax.axhspan(sell_bot, hi, xmin=1 - span / max(len(c), 1), xmax=1.0,
               color=RED, alpha=0.13, zorder=0)
    for y in (lo, hi):
        ax.hlines(y, len(c) - span, len(c) - 1, color="#5c9ded", lw=0.9, alpha=0.8)

    ax.plot(x, trade.values, color=BLUE, lw=1.25, label="TRADE", zorder=2)
    ax.plot(x, trend.values, color=AMBER, lw=1.45, label="TREND", zorder=2)
    ax.plot(x, c.values, color=INK, lw=1.6, label="price", zorder=3)
    ax.scatter([len(c) - 1], [o["spot"]], s=26, color=INK, zorder=4)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(colors=DIM, labelsize=7.5, length=3)
    ax.grid(axis="y", color=LINE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    step = max(len(c) // 6, 1)
    ax.set_xticks(list(x[::step]))
    ax.set_xticklabels([d.strftime("%d %b") for d in c.index[::step]])
    ax.set_xlim(-1, len(c) + span * 0.15)

    # Three stacked rows above the axes: name, then the read and the levels on one
    # line. The title pad has to clear BOTH or they draw on top of each other.
    read, col = _read(o)
    ax.set_title("%s   %.2f" % (ticker, o["spot"]), loc="left", fontsize=12,
                 fontweight="bold", color=INK, pad=26)
    ax.text(0, 1.035, read, transform=ax.transAxes, fontsize=8.5, color=col,
            fontweight="bold", ha="left", va="bottom")
    ax.text(1, 1.035, "RANGE %.2f - %.2f   TRADE %.2f   TREND %.2f"
            % (lo, hi, o["trade"], o["trend"]),
            transform=ax.transAxes, fontsize=8, color=DIM, ha="right", va="bottom")
    # Below the axes, not inside: an "upper left" legend lands on the price line
    # whenever a name is falling, which is exactly when you are reading the chart.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3,
              frameon=False, fontsize=7.5, labelcolor=DIM)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue(), o


def build(when_label, names=None, params=None):
    """Render every held name. Returns (html, [(cid, png)], rows)."""
    params = params or load_params()
    params["range"] = dict(params["range"])
    params["range"].setdefault("active", "hedgeye_anchor")
    names = names or held_names()
    # max_age_hours=0 so an open/close run never draws a cached intraday frame.
    data = load_prices(list(names) + ["^VIX"], params=params, verbose=False,
                       max_age_hours=0)
    try:
        vix = float(data["^VIX"]["Close"].dropna().iloc[-1])
    except Exception:
        vix = 15.0

    imgs, rows, missing = [], [], []
    for t in names:
        x = data.get(t)
        if x is None or x["Close"].dropna().empty:
            missing.append(t)
            continue
        try:
            png, o = chart(t, x, params, vix)
        except Exception as exc:                 # one bad name must not kill the mail
            missing.append("%s (%s)" % (t, type(exc).__name__))
            continue
        if png is None:
            missing.append("%s (too little history)" % t)
            continue
        cid = make_msgid()[1:-1]
        imgs.append((cid, png))
        rows.append((t, o, cid))

    parts = ['<div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;'
             'max-width:800px;margin:0 auto;color:%s">' % INK,
             '<h2 style="font-size:17px;margin:0 0 2px">IBKR book &middot; '
             'TRADE / TREND / RANGE</h2>',
             '<div style="color:%s;font-size:12.5px;margin-bottom:16px">%s &middot; '
             '%d name(s) &middot; VIX %.2f</div>' % (DIM, when_label, len(rows), vix)]
    for t, o, cid in rows:
        read, col = _read(o)
        pos = 100 * o["pos_in_range"]
        sig = o.get("signal") or "no signal &ndash; hold"
        parts.append(
            '<div style="margin:0 0 26px">'
            '<img src="cid:%s" style="width:100%%;max-width:770px;display:block">'
            '<div style="font-size:12px;color:%s;margin-top:4px">'
            '<b style="color:%s">%s</b> &middot; %.0f%% into the RANGE &middot; '
            'signal: <b>%s</b></div></div>' % (cid, DIM, col, read, pos, sig))
    if missing:
        parts.append('<div style="color:%s;font-size:12px;border-top:1px solid %s;'
                     'padding-top:10px">no chart: %s</div>'
                     % (DIM, LINE, ", ".join(missing)))
    parts.append('<div style="color:%s;font-size:11.5px;margin-top:14px">'
                 'Names from fractal/data/ibkr_names.txt (refreshed from the account '
                 'on request); levels recomputed fresh on every run.</div>' % DIM)
    parts.append("</div>")
    return "".join(parts), imgs, rows


def send(html, imgs, when_label, to_addr=OWNER):
    msg = EmailMessage()
    frm = os.environ.get("FRACTAL_SMTP_USER", "")
    msg["Subject"] = "IBKR levels - %s" % when_label
    msg["From"] = frm
    msg["To"] = to_addr
    msg.set_content("This report is HTML. Open in an HTML-capable client.")
    msg.add_alternative(html, subtype="html")
    payload = msg.get_payload()[-1]
    for cid, png in imgs:
        payload.add_related(png, maintype="image", subtype="png", cid="<%s>" % cid)
    publish.deliver(msg, [to_addr])
    return msg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=None,
                    help='e.g. "market open" / "market close"')
    ap.add_argument("--to", default=OWNER)
    ap.add_argument("--no-send", action="store_true",
                    help="render and write out/ibkr_charts.html, do not mail it")
    a = ap.parse_args(argv)
    label = a.label or dt.datetime.now().strftime("%A %d %B %Y, %H:%M")
    html, imgs, rows = build(label)
    out = repo_path("out", "ibkr_charts.html")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print("[charts] %d name(s) rendered -> %s" % (len(rows), out))
    if a.no_send:
        return 0
    if not rows:
        print("[charts] nothing to send")
        return 1
    send(html, imgs, label, a.to)
    print("[charts] sent to %s" % a.to)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
