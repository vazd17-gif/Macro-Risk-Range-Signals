"""Block until the session's daily bars have actually published.

The settle is scheduled 21:20 local, twenty minutes after the 16:00 ET close, on
the assumption stated in settle_update.bat that "by 21:15 the day's bar is in the
feed". On 2026-09-03 it was not: every one of the 12 held names still ended at
09-02, so the settle rebuilt the dashboard off the PREVIOUS close and the P&L
track emailed "0 position(s), +0.00%" as though the session had been flat.

Guessing a later clock time trades one assumption for another -- it depends on
when the machine is awake and when the vendor happens to publish. Waiting for the
data itself does not.

Exits 0 once a session bar exists for at least one held name, 1 on timeout. The
settle treats a timeout as "leave the intraday dashboard up", which is what would
have happened anyway.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from .track import bars_ready


def wait(session=None, minutes=90, every=300, verbose=True):
    session = str(session or dt.date.today())
    deadline = time.time() + minutes * 60
    first = True
    while True:
        try:
            ready, have, need = bars_ready(session)
        except Exception as exc:                      # a feed wobble is not fatal
            ready, have, need = False, 0, 0
            if verbose:
                print("[await] check failed (%s); retrying" % type(exc).__name__)
        if ready:
            if verbose:
                print("[await] %s bars are in (%d of %d held names)"
                      % (session, have, need))
            return 0
        if time.time() >= deadline:
            if verbose:
                print("[await] TIMEOUT after %d min - no %s bar for any of %d held "
                      "name(s). Skipping the settle rather than settling off the "
                      "previous close." % (minutes, session, need))
            return 1
        if first and verbose:
            print("[await] %s bars not published yet (0 of %d); waiting up to %d min"
                  % (session, need, minutes))
            first = False
        time.sleep(every)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=None)
    ap.add_argument("--minutes", type=int, default=90)
    ap.add_argument("--every", type=int, default=300)
    a = ap.parse_args(argv)
    return wait(a.session, a.minutes, a.every)


if __name__ == "__main__":
    sys.exit(main())
