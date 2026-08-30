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
        from fractal.app import etf_report, signals as S
        from fractal.app import portfolio as PF
        etf = S.run(params=params, profile=args.profile)
        if not etf.empty:
            # Open any range breaks before reconciling, so the book the report
            # renders already contains today's new positions.
            PF.sync(etf)
            book = PF.reconcile(etf)
            eff = dict(params)
            eff["range"] = dict(params["range"])
            eff["range"]["active"] = args.profile
            dash = os.path.join(args.outdir, "etf_dashboard.html")
            news = os.path.join(args.outdir, "etf_newsletter.html")
            with open(dash, "w", encoding="utf-8") as fh:
                fh.write(etf_report.render_dashboard(etf, eff, book=book))
            with open(news, "w", encoding="utf-8") as fh:
                fh.write(etf_report.render_newsletter(etf, eff, book=book))
            etf.to_csv(os.path.join(args.outdir, "etf_signals_%s.csv" % stamp), index=False)
            if not book.empty:
                print("book:  %d open | since entry %+.2f%% | since the baseline close %+.2f%%"
                      % (len(book), book["pnl_pct"].mean(), book["since_close_pct"].mean()))
            c = etf["signal"].value_counts().to_dict()
            print("macro: %d names | %s" % (len(etf), "  ".join(
                "%s=%d" % (k, c.get(k, 0)) for k in
                S.SIGNALS)))
            print("wrote %s\nwrote %s" % (dash, news))
            from fractal.app import alerts as A
            A.notify(etf, asof=str(etf.attrs.get("asof", stamp)), seed=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
