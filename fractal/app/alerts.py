"""Push the alerts to a phone.

The dashboard already knows what is worth looking at right now: a name that crossed
TRADE or TREND during this session, and a name sitting at an edge of its range. What
it cannot do is reach you when the phone is in your pocket, because GitHub Pages
serves a static file and a static file cannot start a conversation.

So the push is sent from here, by the same scheduled job that re-prices the page. It
goes to a notification service that already has an app on the phone, which is a far
shorter path than Web Push: no VAPID keys, no subscription store, no server of our
own, and it works on iOS without the site being installed first.

The hard part is not sending, it is *not* sending. The live job runs every ten
minutes, so a naive version would re-announce the same broken TREND thirty times
before lunch. Every alert therefore carries a key, keys already pushed are kept in a
state file, and the file resets when the session date rolls over -- so each distinct
event fires exactly once per day.
"""
from __future__ import annotations

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


# --------------------------------------------------------------------- backends

def _post(url, data, headers=None):
    req = urllib.request.Request(url, data=data, headers=headers or {})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.status


def _ntfy(title, body, urgent):
    """ntfy.sh: no account, no key -- the topic name is the address.

    Which is also the catch. Anyone who knows the topic can read it, so it has to be
    unguessable; `--setup` generates a random one.
    """
    topic = os.environ.get("FRACTAL_NTFY_TOPIC", "").strip()
    if not topic:
        return False
    server = os.environ.get("FRACTAL_NTFY_SERVER", "https://ntfy.sh").rstrip("/")
    headers = {
        "Title": title.encode("ascii", "replace").decode(),
        "Priority": "high" if urgent else "default",
        "Tags": "chart_with_upwards_trend",
        "Click": os.environ.get("FRACTAL_SITE_URL", ""),
    }
    _post("%s/%s" % (server, urllib.parse.quote(topic)),
          body.encode("utf-8"), {k: v for k, v in headers.items() if v})
    return True


def _pushover(title, body, urgent):
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


def send(title, body, urgent=False):
    """Deliver to every configured backend. Returns how many accepted it."""
    n = 0
    for backend in BACKENDS:
        try:
            if backend(title, body, urgent):
                n += 1
        except Exception as e:
            print("  %s failed: %s" % (backend.__name__.strip("_"), e))
    return n


# ----------------------------------------------------------------------- events

def _fmt(x):
    if x is None or x != x:
        return "-"
    return ("%.2f" if abs(x) < 1000 else "%.0f") % x


def events(df):
    """(key, line, urgent) for everything currently worth a push.

    A crossing outranks an edge: crossing a line is a change of state, sitting at an
    edge is a condition that can persist for days.
    """
    out = []
    for r in df.itertuples():
        if getattr(r, "is_index", False) or getattr(r, "cash_like", False):
            continue
        tk = r.ticker
        cross = getattr(r, "intraday", "")
        cross = "" if (cross is None or cross != cross) else str(cross).strip()
        if cross:
            out.append(("%s|%s" % (tk, cross),
                        "%s %s at %s" % (tk, cross, _fmt(r.spot)), True))
            continue
        sig = getattr(r, "signal", None)
        if sig in (ADD_LONG, ADD_SHORT, REMOVE_LONG, COVER_SHORT):
            out.append(("%s|%s" % (tk, sig),
                        "%s %s at %s - %s" % (tk, sig, _fmt(r.spot), getattr(r, "why", "")),
                        sig in (REMOVE_LONG, COVER_SHORT)))
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
    if seed:
        keys = {e[0] for e in events(df)}
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
    new = [e for e in events(df) if e[0] not in already]
    if not new:
        if verbose:
            print("alerts: nothing new (%d already pushed today)" % len(already))
        return 0

    if len(new) <= DIGEST_AT:
        for _, line, urgent in new:
            if not dry:
                send("Risk Range alert", line, urgent)
            if verbose:
                print("  push: " + line)
    else:
        body = "\n".join(e[1] for e in new)
        if not dry:
            send("Risk Range: %d alerts" % len(new), body, any(e[2] for e in new))
        if verbose:
            print("  push digest (%d):\n    %s" % (len(new), body.replace("\n", "\n    ")))

    if not dry:
        _save(asof, already | {e[0] for e in new})
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


if __name__ == "__main__":
    import sys
    if "--setup" in sys.argv:
        setup()
    elif "--test" in sys.argv:
        n = send("Risk Range alert", "Test push -- alerts are working.", False)
        print("sent via %d backend(s): %s" % (n, ", ".join(configured()) or "none"))
    else:
        print("configured backends:", ", ".join(configured()) or "none")
