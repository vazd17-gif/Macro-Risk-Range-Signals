"""Pull the chart images out of the source newsletter PDFs.

Keeps the pipeline reproducible from the original documents rather than from a
committed pile of PNGs. Charts are identified by size and aspect: the newsletter
renders every TradingView capture at 2400x1546.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pypdfium2 as pdfium

MIN_W, MIN_H, MIN_ASPECT = 1200, 700, 1.2


def extract(pdf_path, out_dir, tag=None):
    tag = tag or re.sub(r"[^A-Za-z0-9]+", "", os.path.basename(pdf_path))[:14]
    os.makedirs(out_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    written = []
    for pi in range(len(pdf)):
        for oi, obj in enumerate(pdf[pi].get_objects()):
            if obj.type != 3:
                continue
            try:
                im = obj.get_bitmap(render=False).to_pil()
            except Exception:
                continue
            if im.width < MIN_W or im.height < MIN_H:
                continue
            if im.width / im.height < MIN_ASPECT:
                continue
            name = "%s_p%02d_%d.png" % (tag, pi + 1, oi)
            im.save(os.path.join(out_dir, name))
            written.append(name)
    return written


def main():
    ap = argparse.ArgumentParser(description="Extract chart images from newsletter PDFs.")
    ap.add_argument("pdfs", nargs="+", help="PDF paths or globs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = []
    for p in args.pdfs:
        paths.extend(sorted(glob.glob(p)) or [p])
    total = 0
    for p in paths:
        got = extract(p, args.out)
        print("%-50s %d charts" % (os.path.basename(p), len(got)))
        total += len(got)
    print("total %d -> %s" % (total, args.out))


if __name__ == "__main__":
    main()
