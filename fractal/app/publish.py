"""Daily driver for the live report: refresh, then email, without duplicate sends.

`run_daily.py` recomputes and writes the files. This adds the two things a
scheduled job needs on top of that:

  state       The report is keyed to the last *completed* session. A scheduler
              firing on a holiday, or twice in one evening, would otherwise send
              the same newsletter again. The as-of date of the last successful
              send is kept in `out/.last_sent`, and a send is skipped unless the
              data has actually moved on.

  delivery    Sending is deliberately separate from generating, and defaults to
              creating a draft rather than sending. Credentials are never stored
              here: SMTP details come from the environment
              (FRACTAL_SMTP_USER / FRACTAL_SMTP_PASS / FRACTAL_MAIL_TO), so the
              password lives in the OS keystore or the task definition, not in the
              repository.

Gmail requires an app password for SMTP, not the account password. Create one at
https://myaccount.google.com/apppasswords and set it in the environment yourself --
this module never asks for it and never writes it anywhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import smtplib
import ssl
from email.message import EmailMessage

from ..data.loader import next_session, repo_path

STATE = "out/.last_sent"
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465


def _state_path():
    return repo_path(*STATE.split("/"))


def last_sent():
    p = _state_path()
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as fh:
        return fh.read().strip() or None


def mark_sent(asof):
    p = _state_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(str(asof))


def report_asof():
    """As-of date of the newsletter currently on disk."""
    import pandas as pd
    import glob
    files = sorted(glob.glob(repo_path("out", "etf_signals_*.csv")))
    if not files:
        return None
    d = pd.read_csv(files[-1])
    return str(d["asof"].max()) if len(d) else None


def _pretty_date(iso):
    """2026-08-28 -> 28 August 2026. Falls back to the raw string if unparseable."""
    try:
        return dt.date.fromisoformat(str(iso)).strftime("%d %B %Y").lstrip("0")
    except Exception:
        return str(iso)


def build_message(to_addrs, subject=None, html_path=None, sender=None, as_draft=False):
    html_path = html_path or repo_path("out", "etf_newsletter.html")
    with open(html_path, "r", encoding="utf-8") as fh:
        html = fh.read()
    asof = report_asof() or dt.date.today().isoformat()

    msg = EmailMessage()
    # Dated by the session the levels are for, not by the close they came from: a
    # Monday newsletter carries Monday's levels, and an inbox line reading last
    # Friday would look like yesterday's post.
    session = next_session(asof) or asof
    msg["Subject"] = subject or ("Macro Risk Range Signals - %s"
                                 % _pretty_date(getattr(session, "isoformat", lambda: session)()))
    # Recipients go in the envelope, not the headers, so nobody on the list sees
    # who else is on it. The message therefore carries no To: of its own -- the
    # sender is addressed instead, which is what a Bcc-only mail looks like and
    # what keeps clients from rendering "undisclosed recipients" as a broken field.
    # `send` passes the real list to SMTP separately; anything that builds a message
    # without doing that will mail nobody, which is the safe direction to fail.
    frm = sender or os.environ.get("FRACTAL_SMTP_USER", "")
    msg["From"] = frm
    msg["To"] = frm
    if as_draft:
        # A saved .eml is a draft someone may send by hand, and a draft with no
        # recipients mails nobody. Bcc belongs here and only here -- on the wire the
        # list travels in the envelope instead.
        msg["Bcc"] = ", ".join(to_addrs)
    msg.set_content(
        "This report is HTML. Levels for the %s session, computed off the %s close.\n"
        "Open in an HTML-capable client to see the tables." % (session, asof))
    msg.add_alternative(html, subtype="html")
    return msg


def deliver(msg, to_addrs):
    """Put a built message on the wire. Lifted out of send() so other reports can
    use the same credentials and the same error handling without restating either."""
    user = (os.environ.get("FRACTAL_SMTP_USER") or "").strip()
    pwd = "".join((os.environ.get("FRACTAL_SMTP_PASS") or "").split())
    if not user or not pwd:
        raise RuntimeError(
            "set FRACTAL_SMTP_USER and FRACTAL_SMTP_PASS in the environment "
            "(use a Gmail app password, not the account password)")
    if not msg.get("From"):
        msg["From"] = user
    ctx = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as smtp:
            smtp.login(user, pwd)
            smtp.send_message(msg, from_addr=user, to_addrs=list(to_addrs))
    except smtplib.SMTPAuthenticationError as e:
        raise RuntimeError(
            "Gmail rejected the login for %s. Use a 16-character app password from "
            "myaccount.google.com/apppasswords, not the account password, and make "
            "sure 2-Step Verification is on for that account. (%s)"
            % (user, getattr(e, "smtp_code", "auth error"))) from None
    return True


def send(to_addrs, subject=None, html_path=None, force=False):
    """Send via SMTP. Skips unless the data has moved on, unless `force`."""
    asof = report_asof()
    if not force and asof and last_sent() == asof:
        print("[publish] already sent the %s report; nothing to do" % asof)
        return False

    user = (os.environ.get("FRACTAL_SMTP_USER") or "").strip()
    # Google shows app passwords as "xxxx xxxx xxxx xxxx"; the spaces are display
    # formatting and SMTP auth fails if they are sent, so strip all whitespace.
    pwd = "".join((os.environ.get("FRACTAL_SMTP_PASS") or "").split())
    if not user or not pwd:
        raise RuntimeError(
            "set FRACTAL_SMTP_USER and FRACTAL_SMTP_PASS in the environment "
            "(use a Gmail app password, not the account password)")

    msg = build_message(to_addrs, subject, html_path, sender=user)
    deliver(msg, to_addrs)
    if asof:
        mark_sent(asof)
    print("[publish] sent the %s report to %d bcc recipient(s): %s"
          % (asof, len(to_addrs), ", ".join(to_addrs)))
    return True


def save_eml(to_addrs, out_path=None, subject=None, html_path=None):
    """Write the message as a .eml file - open it in any mail client to review or send.

    The draft carries the recipients as Bcc so it is actually sendable by hand.
    """
    out_path = out_path or repo_path("out", "newsletter.eml")
    msg = build_message(to_addrs, subject, html_path, as_draft=True)
    with open(out_path, "wb") as fh:
        fh.write(bytes(msg))
    print("[publish] wrote %s" % out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Email the Macro Risk Range newsletter.")
    ap.add_argument("--to", default=os.environ.get("FRACTAL_MAIL_TO", ""),
                    help="comma-separated recipients (or set FRACTAL_MAIL_TO)")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--html", default=None)
    ap.add_argument("--send", action="store_true",
                    help="actually send. Without this a .eml file is written instead.")
    ap.add_argument("--force", action="store_true",
                    help="send even if this as-of date was already sent")
    args = ap.parse_args()

    to = [a.strip() for a in args.to.split(",") if a.strip()]
    if not to:
        print("no recipients. pass --to a@b.com or set FRACTAL_MAIL_TO")
        return 1

    if args.send:
        send(to, args.subject, args.html, force=args.force)
    else:
        save_eml(to, subject=args.subject, html_path=args.html)
        print("      review it, then re-run with --send to deliver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
