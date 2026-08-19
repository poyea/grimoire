#!/usr/bin/env python3
"""Export every volume to HTML under docs/__compiled/.

Typst's HTML export is still gated behind `--features html` and upstream
labels it incomplete, so this is deliberately best-effort: a volume that
fails to export is reported and skipped rather than taking the whole
Pages deploy down with it. Pass --strict to turn any failure into a
nonzero exit.

The output is generated at deploy time and never committed; docs/__compiled/
is gitignored.

Usage:
  build_html.py [--out docs/__compiled] [--typst typst] [--strict]
"""
from __future__ import annotations

import argparse
import html
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Warnings that are inherent to rendering a paged document as HTML and say
# nothing about the health of this build.
BENIGN = (
    "pagebreak was ignored",
    "page set rule was ignored",
    "align was ignored",
    "html export is under active development",
    "its behaviour may change",
    "do not rely on this feature",
    "see https://github.com/typst/typst/issues",
)


def volumes() -> list[Path]:
    return sorted(p for p in ROOT.glob("*.typ") if p.stem != "template")


def interesting(stderr: str) -> list[str]:
    out = []
    for line in stderr.splitlines():
        s = line.strip()
        if not s.startswith(("warning:", "error:")):
            continue
        if any(b in s for b in BENIGN):
            continue
        out.append(s)
    return out


def write_index(out_dir: Path, built: list[tuple[str, int]]) -> None:
    rows = "\n".join(
        f'      <li><a href="{html.escape(slug)}.html">{html.escape(slug)}</a>'
        f' <span class="kb">{kb:,} KB</span></li>'
        for slug, kb in built
    )
    (out_dir / "index.html").write_text(
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "<title>Grimoire — compiled volumes</title>\n"
        "<style>\n"
        " body{font:16px/1.6 system-ui,sans-serif;max-width:44rem;"
        "margin:3rem auto;padding:0 1rem}\n"
        " li{margin:.35rem 0} .kb{color:#777;font-size:.85em}\n"
        " .note{color:#666;font-size:.9em;border-left:3px solid #ddd;"
        "padding-left:.8rem}\n"
        "</style></head><body>\n"
        "  <h1>Compiled volumes</h1>\n"
        '  <p class="note">Generated from the Typst sources by\n'
        "  <code>scripts/build_html.py</code> at deploy time. Typst HTML\n"
        "  export is experimental; the PDFs on the releases page remain\n"
        "  the reference rendering.</p>\n"
        f"  <ul>\n{rows}\n  </ul>\n"
        "</body></html>\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/__compiled")
    ap.add_argument("--typst", default="typst")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if any volume fails to export")
    args = ap.parse_args()

    if shutil.which(args.typst) is None:
        print(f"error: typst not found on PATH as {args.typst!r}",
              file=sys.stderr)
        return 2

    out_dir = ROOT / args.out
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    built: list[tuple[str, int]] = []
    failed: list[str] = []
    for vol in volumes():
        dest = out_dir / f"{vol.stem}.html"
        proc = subprocess.run(
            [args.typst, "compile", "--features", "html",
             "--format", "html", str(vol), str(dest)],
            capture_output=True, text=True,
            # Typst emits UTF-8; without this the default Windows codepage
            # raises UnicodeDecodeError on the first non-ASCII diagnostic.
            encoding="utf-8", errors="replace",
        )
        notes = interesting(proc.stderr)
        if proc.returncode != 0 or not dest.exists():
            failed.append(vol.stem)
            print(f"FAIL  {vol.stem}")
            for n in notes[:5]:
                print(f"        {n}")
            continue
        kb = dest.stat().st_size // 1024
        built.append((vol.stem, kb))
        flag = f"  ({len(notes)} note(s))" if notes else ""
        print(f"ok    {vol.stem:<32} {kb:>6,} KB{flag}")
        for n in notes[:3]:
            print(f"        {n}")

    write_index(out_dir, built)
    total = sum(kb for _, kb in built)
    print(f"\n{len(built)} volume(s) -> {args.out} ({total:,} KB total); "
          f"{len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
