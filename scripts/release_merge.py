#!/usr/bin/env python3
"""Merge all grimoire_*.pdf (sorted, excluding the output itself) into
grimoire_complete.pdf.

Usage: release_merge.py [directory]
"""

import sys
from pathlib import Path

from pypdf import PdfWriter

OUTPUT = "grimoire_complete.pdf"


def main() -> None:
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    pdfs = sorted(
        p for p in directory.glob("grimoire_*.pdf") if p.name != OUTPUT
    )
    if not pdfs:
        sys.exit("No grimoire_*.pdf files found.")

    writer = PdfWriter()
    for pdf in pdfs:
        print(f"Appending {pdf}")
        writer.append(str(pdf))

    writer.add_metadata({"/Title": "Grimoire (Complete)", "/Author": "John Law"})
    out = directory / OUTPUT
    with open(out, "wb") as f:
        writer.write(f)
    print(f"Wrote {out} ({len(pdfs)} volumes)")


if __name__ == "__main__":
    main()
