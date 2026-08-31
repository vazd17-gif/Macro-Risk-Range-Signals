#!/usr/bin/env python
"""Daily refresh: macro dashboard plus the ETF risk-range dashboard and newsletter."""
import argparse
import datetime as dt
import os

from fractal.data.loader import load_params, repo_path
from fractal.app import scan as scan_mod
from fractal.app.dashboard import render


def main():
    ap = argparse.ArgumentParser(description="Refresh the fractal trend/range dashboard.")
    ap.add_argument("--tickers", default=None)
    ap.add_argument("--source", default=None, choices=["ib", "yahoo"])
    ap.add_argument("--outdir", default=repo_path("out"))
    ap.add_argument("--skip-etf", action="store_true", help="macro dashboard only")
    ap.add_argument("--profile", default="hedgeye_anchor",
                    help="RANGE profile for the ETF report")
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")] if args.tickers else None
    params = load_params()
    df = scan_mod.run(tickers, params=params, source=args.source)
    if df.empty:
        print("no data")
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    stamp = dt.date.today().isoformat()
    html_path = os.path.join(args.outdir, "dashboard.html")
    csv_path = os.path.join(args.outdir, "scan_%s.csv" % stamp)

    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render(df, params))
    df.to_csv(csv_path, index=False)

    edges = int(df.get("at_top").sum() + df.get("at_bottom").sum())
    print("macro: %d names | %d at a range edge | %s"
          % (len(df), edges, ", ".join("%s=%d" % kv for kv in df["state"].value_counts().items())))
    print("wrote %s\nwrote %s" % (html_path, csv_path))

    if not args.skip_etf:
        # Delegate rather than reimplement. This block used to be its own copy of
        # the ETF build and had quietly drifted from it -- it rendered without the
        # closed-position report and printed raw signal names. One code path means
        # the daily and intraday jobs can no longer disagree about what a build is.
        from fractal.app import etf_report
        rc = etf_report.main(["--sync", "--push",
                              "--profile", args.profile, "--outdir", args.outdir])
        if rc:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
