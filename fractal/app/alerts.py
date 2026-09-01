"""Push the alerts to a phone.

The dashboard already knows what is worth looking at right now: a name that crossed
TRADE or TREND during this session, and a name sitting at an edge of its range. What
it cannot do is reach you when the phone is in your pocket, because GitHub Pages
serves a static file and a static file cannot start a conversation.

So the push is sent from here, by the same scheduled job that re-prices the page. It
goes to a notification service that already has an app on the phone, which is a far
shorter path than Web Push: no VAPID keys, no subscription store, no server of our
own, and it works on iOS without the site being installed first.

Two consequences of the phone being an iPhone shape everything below. iOS cannot
hand an https link to an installed home-screen web app, so tapping an alert lands in
Safari rather than the dashboard -- which means the notification has to carry the
whole decision itself, not a pointer to it. And iOS gives a sender no control over
notification colour, so the signal colours from the dashboard are carried as emoji,
which is the only colour that reaches the banner.

The hard part is not sending, it is *not* sending. The live job runs every ten
minutes, so a naive version would re-announce the same broken TREND thirty times
before lunch. Every alert therefore carries a key, keys already pushed are kept in a
state file, and the file resets when the session date rolls over -- so each distinct
event fires exactly once per day.
"""
from __future__ import annotations

import collections
import json
import os
import urllib.parse
import urllib.request

from .signals import (ADD_LONG, ADD_SHORT, BREAKDOWN, BREAKOUT, COVER_SHORT, REMOVE_LONG,
                      TRIM_LONG, TRIM_SHORT, vol_read)
from .signals import label as sig_label

STATE = ("out", ".alerted")
TIMEOUT = 15

# More than this many new alerts in one pass and they go as a single digest.
# Twenty separate buzzes is not an alert, it is a reason to turn alerts off.
DIGEST_AT = 4

# These mirror SIG_STYLE in etf_report exactly, so a glance at the phone reads the
# same as a glance at the dashboard: green add long, red remove long or a line lost,
# amber add short, blue cover short or a line reclaimed.
TAGS = {
    ADD_LONG:     "green_circle",
    BREAKOUT:     "rocket",
    BREAKDOWN:    "bangbang",
    TRIM_LONG:    "orange_circle",
    REMOVE_LONG:  "red_circle",
    ADD_SHORT:    "orange_circle",
    TRIM_SHORT:   "orange_circle",
    COVER_SHORT:  "large_blue_circle",
    "lost":       "red_circle",
    "reclaimed":  "large_blue_circle",
    "digest":     "bar_chart",
    # A volatility index is tagged by what it means for everything else, not by
    # which way it moved: falling vol is the green case even though the same event
    # on an ETF would be red.
    "supportive": "green_circle",
    "risk_off":   "red_circle",
    "mixed":      "orange_circle",
}

Event = collections.namedtuple("Event", "key title body short urgent tag index")


# --------------------------------------------------------------------- backends

def _post(url, data, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status


def _ntfy(title, body, urgent, tag):
    """ntfy.sh: no account, no key -- the topic name is the address.

    Which is also the catch. Anyone who knows the topic can read it, so it has to be
    unguessable; `--setup` generates a random one.

    `Title` is an HTTP header and so has to stay ASCII; the body is sent as UTF-8
    and may hold anything.
    """
    topic = os.environ.get("FRACTAL_NTFY_TOPIC", "").strip()
    if not topic:
        return False
    server = os.environ.get("FRACTAL_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    headers = {
        "Title": title.encode("ascii", "replace").decode(),
        "Priority": "high" if urgent else "default",
        "Tags": tag or "",
        "Click": os.environ.get("FRACTAL_SITE_URL", ""),
    }
    _post("%s/%s" % (server, urllib.parse.quote(topic)),
          body.encode("utf-8"), {k: v for k, v in headers.items() if v})
    return True


def _pushover(title, body, urgent, tag):
    """Pushover has no tag vocabulary, so the colour cue is dropped here."""
    token = os.environ.get("FRACTAL_PUSHOVER_TOKEN", "").strip()
    user = os.environ.get("FRACTAL_PUSHOVER_USER", "").strip()
    if not (token and user):
        return False
    payload = {"token": token, "user": user, "title": title, "message": body,
               "priority": 1 if urgent else 0}
    url = os.environ.get("FRACTAL_SITE_URL", "")
    if url:
        payload["url"] = url
        payload["url_title"] = "Open dashboard"
    _post("https://api.pushover.net/1/messages.json",
          urllib.parse.urlencode(payload).encode("utf-8"),
          {"Content-Type": "application/x-www-form-urlencoded"})
    return True


BACKENDS = (_ntfy, _pushover)


def configured():
    """Names of the backends that have what they need in the environment."""
    out = []
    if os.environ.get("FRACTAL_NTFY_TOPIC"):
        out.append("ntfy")
    if os.environ.get("FRACTAL_PUSHOVER_TOKEN") and os.environ.get("FRACTAL_PUSHOVER_USER"):
        out.append("pushover")
    return out


def send(title, body, urgent=False, tag=None):
    """Deliver to every configured backend. Returns how many accepted it."""
    n = 0
    for backend in BACKENDS:
        try:
            if backend(title, body, urgent, tag):
                n += 1
        except Exception as e:
            print("  %s failed: %s" % (backend.__name__.strip("_"), e))
    return n


# ----------------------------------------------------------------------- events

def _fmt(x):
    if x is None or x != x:
        return "-"
    return ("%.2f" if abs(x) < 1000 else "%.0f") % x


def _num(x):
    """None for anything that is not a real number, so callers test once."""
    if x is None or x != x:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _bull(x):
    """True / False / None for a duration flag, from a frame or a re-read CSV.

    Read back from CSV the flag arrives as the string "True", and bool("False") is
    True -- which would label every bearish name bullish. Strings are compared, not
    coerced.
    """
    if x is None or x != x:
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        return True if s == "true" else False if s == "false" else None
    return bool(x)


def _side(bull):
    return "" if bull is None else ("bullish" if bull else "bearish")


def _detail(r):
    """The lines that make a notification worth reading without opening anything.

    Three or four lines is what an expanded iOS banner shows without truncating,
    which is the budget: spot, the range with where price sits inside it, both
    duration lines, and for a signal the reason.
    """
    lines = []

    head = "Spot " + _fmt(_num(getattr(r, "spot", None)))
    chg = _num(getattr(r, "chg_pct", None))
    if chg is not None and abs(chg) >= 0.05:   # below this it rounds to "-0.0%"
        head += "  %+.1f%% today" % chg
    lines.append(head)

    lo, hi = _num(getattr(r, "range_low", None)), _num(getattr(r, "range_high", None))
    if lo is not None and hi is not None:
        rng = "Range %s - %s" % (_fmt(lo), _fmt(hi))
        pos = _num(getattr(r, "pos_in_range", None))
        if pos is not None:
            rng += "  %d%% in" % round(100 * min(max(pos, 0.0), 1.0))
        lines.append(rng)

    # Each duration carries which side spot is on. The bare numbers left the reader
    # comparing them against spot in their head, which is the one thing the alert
    # exists to save them doing.
    parts = []
    for label, key in (("TRADE", "trade"), ("TREND", "trend")):
        level = _num(getattr(r, key, None))
        if level is None:
            continue
        word = _side(_bull(getattr(r, key + "_bull", None)))
        parts.append("%s %s%s" % (label, _fmt(level), "  " + word if word else ""))
    if parts:
        lines.append("   ".join(parts))
    return lines


def events(df):
    """Everything currently worth a push, as Event records.

    `short` is the one-line form used when several fire at once and go as a digest;
    `body` is the full read used when an alert travels on its own.

    A crossing outranks a signal: crossing a line is a change of state, whereas a
    signal can restate a condition that has held for days.
    """
    nl = chr(10)
    out = []
    for r in df.itertuples():
        if getattr(r, "cash_like", False):
            continue
        tk = r.ticker
        spot = _fmt(_num(getattr(r, "spot", None)))

        cross = getattr(r, "intraday", "")
        cross = "" if (cross is None or cross != cross) else str(cross).strip()

        # A macro reference is never a position either, but the S&P losing TREND or
        # the 10-year reclaiming it is worth knowing the moment it happens -- it
        # reframes the whole list. Alerted like the volatility indices: only on an
        # actual crossing, never on where price merely sits, and never as an
        # instruction. Nothing here asks anyone to buy or sell.
        if getattr(r, "is_macro", False):
            if not cross:
                flips = []
                for name, key in (("TREND", "trend"), ("TRADE", "trade")):
                    if _num(getattr(r, key + "_flip_days", None)) == 0:
                        bull = _bull(getattr(r, key + "_bull", None))
                        if bull is not None:
                            flips.append(("reclaimed " if bull else "lost ") + name)
                cross = " and ".join(flips)
            if cross:
                out.append(Event(
                    key="%s|%s" % (tk, cross),
                    title="%s %s" % (tk, cross),
                    body=nl.join(_detail(r) + ["Macro reference - context, not a position."]),
                    short="%s %s at %s" % (tk, cross, spot),
                    urgent=False,
                    tag=TAGS.get(cross.split(" ")[0], ""),
                    index=True))
            continue

        # A volatility index never carries a position, so it never produces a
        # signal -- but a regime shift in it frames every other name on the page,
        # which is the whole reason it is worth waking a phone for.
        if getattr(r, "is_index", False):
            # A volatility index never carries a position, so it never produces a
            # signal -- but it does cross lines, and a vol regime change frames
            # every other name on the page. Intraday the crossing comes from the
            # re-pricer; on a daily run it has to be read off the flip age, and only
            # a flip on THIS bar counts. The break flags stay true for three
            # sessions, which would otherwise announce the same break three times.
            if not cross:
                flips = []
                for name, key in (("TREND", "trend"), ("TRADE", "trade")):
                    if _num(getattr(r, key + "_flip_days", None)) == 0:
                        bull = _bull(getattr(r, key + "_bull", None))
                        if bull is not None:
                            flips.append(("reclaimed " if bull else "lost ") + name)
                cross = " and ".join(flips)
            label, stance = vol_read(cross=cross,
                                     at_low=bool(getattr(r, "at_low", False)),
                                     at_high=bool(getattr(r, "at_high", False)))
            if label:
                note = {"supportive": "Falling volatility - supportive for risk assets.",
                        "risk_off": "Rising volatility - a headwind for risk assets.",
                        "mixed": "Volatility mixed across durations."}.get(stance, "")
                out.append(Event(
                    key="%s|%s" % (tk, label),
                    title="%s %s" % (tk, label),
                    body=nl.join(_detail(r) + ([note] if note else [])),
                    short="%s %s at %s" % (tk, label, spot),
                    urgent=bool(cross),
                    tag=TAGS.get(stance, ""),
                    index=True))
            continue
        if cross:
            out.append(Event(
                key="%s|%s" % (tk, cross),
                title="%s %s" % (tk, cross),
                body=nl.join(_detail(r)),
                short="%s %s at %s" % (tk, cross, spot),
                urgent=True,
                tag=TAGS.get(cross.split(" ")[0], ""),
                index=False))
            continue

        sig = getattr(r, "signal", None)
        if sig in TAGS and sig not in ("digest", "supportive", "risk_off", "mixed"):
            why = getattr(r, "why", "")
            why = "" if (why is None or why != why) else str(why).strip()
            out.append(Event(
                key="%s|%s" % (tk, sig),
                title="%s %s" % (tk, sig_label(sig)),
                body=nl.join(_detail(r) + ([why] if why else [])),
                short="%s %s at %s%s" % (tk, sig_label(sig), spot, " - " + why if why else ""),
                urgent=sig in (REMOVE_LONG, TRIM_LONG, COVER_SHORT),
                tag=TAGS.get(sig, ""),
                index=False))
    return out


# ------------------------------------------------------------------------ state

def _state_path():
    from ..data.loader import repo_path
    return repo_path(*STATE)


def _load(asof):
    """Keys already pushed for `asof`. A new session starts from empty."""
    try:
        with open(_state_path()) as f:
            st = json.load(f)
        if st.get("asof") == asof:
            return set(st.get("sent", []))
    except Exception:
        pass
    return set()


def _save(asof, sent):
    p = _state_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"asof": asof, "sent": sorted(sent)}, f, indent=1)


# ------------------------------------------------------------------------- main

def notify(df, asof="", verbose=True, dry=False, seed=False):
    """Push anything in `df` that has not already been pushed this session.

    `seed` records the current events as known without sending any of them. The
    daily job seeds, because the whole daily signal set is what the noon newsletter
    is for; pushing all forty of them to the phone at the same moment would be a
    second copy of the email, not an alert. What the phone is for is what happens
    *after* that -- a line crossed mid-session, a name that reaches an edge while
    the market is open.
    """
    nl = chr(10)
    if seed:
        ev = events(df)
        asof = asof or str(df.attrs.get("asof", "")) or "?"
        already = _load(asof)

        # Volatility indices are the exception to seeding. The rest of the daily set
        # is what the noon newsletter is for, but a vol regime change is not a
        # position to read about later -- it reframes every signal underneath it,
        # there are only three of them so they cannot flood, and waiting until the
        # live job starts at 14:30 would hold it back for two and a half hours.
        vol = [e for e in ev if e.index and e.key not in already]
        for e in vol:
            if not dry and configured():
                send(e.title, e.body, e.urgent, e.tag)
            if verbose:
                print("  push: %s [%s]" % (e.title, e.tag))
        if not dry:
            _save(asof, {e.key for e in ev} | already)
        if verbose:
            print("alerts: seeded %d daily signals, pushed %d volatility event(s)"
                  % (len(ev) - len(vol), len(vol)))
        return len(vol)

    if not configured():
        if verbose:
            print("alerts: no push backend configured "
                  "(run: python -m fractal.app.alerts --setup)")
        return 0

    asof = asof or str(df.attrs.get("asof", "")) or "?"
    already = _load(asof)
    # A name repeating yesterday's instruction is not an alert. It is still on the
    # dashboard and still in the scan; it just does not earn a buzz for a second
    # day. Vol indices are exempt -- a regime read is context, not a position.
    if "is_new" in getattr(df, "columns", []):
        keep = set(df.loc[df["is_new"].fillna(True).astype(bool), "ticker"])
        idx = set()
        for col in ("is_index", "is_macro"):
            if col in df:
                idx |= set(df.loc[df[col].astype(bool), "ticker"])
        src = df[df["ticker"].isin(keep | idx)]
    else:
        src = df
    new = [e for e in events(src) if e.key not in already]
    if not new:
        if verbose:
            print("alerts: nothing new (%d already pushed today)" % len(already))
        return 0

    # A backend with nothing configured returns False rather than raising, so a run
    # with no topic set looked exactly like a run that delivered: the digest printed
    # either way and the count was never reported. Say what was actually accepted.
    backends = configured()
    if not dry and not backends and verbose:
        print("  !! NOT PUSHED - no alert backend configured in this process. "
              "Set FRACTAL_NTFY_TOPIC (see: python -m fractal.app.alerts --setup)")

    accepted = 0
    if len(new) <= DIGEST_AT:
        for e in new:
            if not dry:
                accepted += send(e.title, e.body, e.urgent, e.tag)
            if verbose:
                print("  push: %s [%s]%s         %s"
                      % (e.title, e.tag, nl, e.body.replace(nl, nl + "         ")))
    else:
        body = nl.join(e.short for e in new)
        if not dry:
            accepted += send("Risk Range: %d alerts" % len(new), body,
                             any(e.urgent for e in new), TAGS["digest"])
        if verbose:
            print("  push digest (%d):%s    %s"
                  % (len(new), nl, body.replace(nl, nl + "    ")))
    if not dry and verbose:
        print("  delivered to %d backend(s): %s"
              % (accepted, ", ".join(backends) if backends else "none"))

    # Only record events as pushed if something actually took them. Marking them
    # sent after a silent failure loses them for the rest of the session.
    if not dry and accepted:
        _save(asof, already | {e.key for e in new})
    return len(new)


def setup():
    """Print the exact steps to turn alerts on, with a private topic generated."""
    import base64
    import secrets
    topic = "riskrange-" + base64.b32encode(
        secrets.token_bytes(10)).decode().lower().rstrip("=")
    print("""
Phone alerts via ntfy -- free, no account, works on iPhone and Android.

1. Install the "ntfy" app (App Store / Play Store).

2. In the app tap + and subscribe to exactly this topic:

       %s

   Anyone who knows a topic name can read it, so this one is random. Keep it to
   yourself -- it is the only thing protecting the channel.

3. Run these once in PowerShell, then open a NEW PowerShell window:

   setx FRACTAL_NTFY_TOPIC "%s"
   setx FRACTAL_SITE_URL "https://vazd17-gif.github.io/Macro-Risk-Range-Signals/"

4. Test:   python -m fractal.app.alerts --test
""" % (topic, topic))


# One of each colour, so the whole tag vocabulary can be checked on the phone in a
# single run rather than waiting for the market to produce each kind.
#
# The ticker is deliberately not a real one and every sample says TEST. A styling
# demo that used real symbols and plausible levels once arrived on the phone looking
# exactly like a live alert, and contradicted the model while doing it -- a test
# notification has to be unmistakable at a glance or it is worse than no test.
def _samples():
    nl = chr(10)
    def body(trade_side, trend_side, why=""):
        return ("TEST - not a real signal" + nl +
                "Spot 100.00  +0.0% today" + nl +
                "Range 98.00 - 102.00  50% in" + nl +
                "TRADE 99.00  %s   TREND 97.00  %s" % (trade_side, trend_side) +
                (nl + why if why else ""))
    return [
        ("TEST ADD LONG", body("bullish", "bullish",
                               "low end of RANGE, bullish TRADE and TREND"),
         False, TAGS[ADD_LONG]),
        ("TEST lost TREND", body("bullish", "bearish"),
         True, TAGS["lost"]),
        ("TEST ADD SHORT", body("bearish", "bearish",
                                "high end of RANGE, bearish TRADE and TREND"),
         False, TAGS[ADD_SHORT]),
        ("TEST reclaimed TRADE", body("bullish", "bearish"),
         True, TAGS["reclaimed"]),
    ]


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        setup()
    elif "--test" in sys.argv:
        total = 0
        for title, body, urgent, tag in _samples():
            total += send(title, body, urgent, tag)
            print("  %-18s %s" % (title, tag))
        print("sent %d via: %s" % (total, ", ".join(configured()) or "none"))
    else:
        print("configured backends:", ", ".join(configured()) or "none")
