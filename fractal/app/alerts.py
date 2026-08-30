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

from .signals import ADD_LONG, ADD_SHORT, REMOVE_LONG, COVER_SHORT

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
    REMOVE_LONG:  "red_circle",
    ADD_SHORT:    "orange_circle",
    COVER_SHORT:  "large_blue_circle",
    "lost":       "red_circle",
    "reclaimed":  "large_blue_circle",
    "digest":     "bar_chart",
}

Event = collections.namedtuple("Event", "key title body short urgent tag")


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

    trade, trend = _num(getattr(r, "trade", None)), _num(getattr(r, "trend", None))
    if trade is not None or trend is not None:
        lines.append("TRADE %s  TREND %s" % (_fmt(trade), _fmt(trend)))
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
        if getattr(r, "is_index", False) or getattr(r, "cash_like", False):
            continue
        tk = r.ticker
        spot = _fmt(_num(getattr(r, "spot", None)))

        cross = getattr(r, "intraday", "")
        cross = "" if (cross is None or cross != cross) else str(cross).strip()
        if cross:
            out.append(Event(
                key="%s|%s" % (tk, cross),
                title="%s %s" % (tk, cross),
                body=nl.join(_detail(r)),
                short="%s %s at %s" % (tk, cross, spot),
                urgent=True,
                tag=TAGS.get(cross.split(" ")[0], "")))
            continue

        sig = getattr(r, "signal", None)
        if sig in (ADD_LONG, ADD_SHORT, REMOVE_LONG, COVER_SHORT):
            why = getattr(r, "why", "")
            why = "" if (why is None or why != why) else str(why).strip()
            out.append(Event(
                key="%s|%s" % (tk, sig),
                title="%s %s" % (tk, sig),
                body=nl.join(_detail(r) + ([why] if why else [])),
                short="%s %s at %s%s" % (tk, sig, spot, " - " + why if why else ""),
                urgent=sig in (REMOVE_LONG, COVER_SHORT),
                tag=TAGS.get(sig, "")))
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
        keys = {e.key for e in events(df)}
        asof = asof or str(df.attrs.get("asof", "")) or "?"
        if not dry:
            _save(asof, keys)
        if verbose:
            print("alerts: seeded %d daily signals (intraday changes will push)" % len(keys))
        return 0

    if not configured():
        if verbose:
            print("alerts: no push backend configured "
                  "(run: python -m fractal.app.alerts --setup)")
        return 0

    asof = asof or str(df.attrs.get("asof", "")) or "?"
    already = _load(asof)
    new = [e for e in events(df) if e.key not in already]
    if not new:
        if verbose:
            print("alerts: nothing new (%d already pushed today)" % len(already))
        return 0

    if len(new) <= DIGEST_AT:
        for e in new:
            if not dry:
                send(e.title, e.body, e.urgent, e.tag)
            if verbose:
                print("  push: %s [%s]%s         %s"
                      % (e.title, e.tag, nl, e.body.replace(nl, nl + "         ")))
    else:
        body = nl.join(e.short for e in new)
        if not dry:
            send("Risk Range: %d alerts" % len(new), body,
                 any(e.urgent for e in new), TAGS["digest"])
        if verbose:
            print("  push digest (%d):%s    %s"
                  % (len(new), nl, body.replace(nl, nl + "    ")))

    if not dry:
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
    body = ("TEST - not a real signal" + nl +
            "Spot 100.00  +0.0% today" + nl +
            "Range 98.00 - 102.00  50% in" + nl +
            "TRADE 99.00  TREND 97.00")
    return [
        ("TEST ADD LONG",     body, False, TAGS[ADD_LONG]),
        ("TEST REMOVE LONG",  body, True,  TAGS[REMOVE_LONG]),
        ("TEST ADD SHORT",    body, False, TAGS[ADD_SHORT]),
        ("TEST COVER SHORT",  body, True,  TAGS[COVER_SHORT]),
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
