#!/usr/bin/env python3
"""Export every volume to HTML under docs/compile/.

Typst's HTML export is still gated behind `--features html` and upstream
labels it incomplete, so this is deliberately best-effort: a volume that
fails to export is reported and skipped rather than taking the whole
Pages deploy down with it. Pass --strict to turn any failure into a
nonzero exit.

The output is generated at deploy time and never committed; docs/compile/
is gitignored.

Usage:
  build_html.py [--out docs/compile] [--typst typst] [--strict]
"""
from __future__ import annotations

import argparse
import html
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = "https://poyea.github.io/grimoire/"
INCLUDE_RE = re.compile(r'#include\s+"([^"]+\.typ)"')

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


PROJECT_RE = re.compile(r'#project\(\s*"([^"]+)"')


def volumes() -> list[Path]:
    return sorted(p for p in ROOT.glob("*.typ") if p.stem != "template")


def display_title(vol: Path) -> str:
    """The volume's real title, e.g. "Web and Browser Internals".

    Falls back to the slug so a volume that has not been given a
    #project(...) title still gets listed rather than dropped.
    """
    m = PROJECT_RE.search(vol.read_text(encoding="utf-8"))
    return m.group(1).strip() if m else vol.stem


def chapter_count(vol: Path) -> int:
    """Chapters reachable from a volume, following nested includes.

    coding/cpp-and-java/*.typ is included from a chapter rather than from
    coding.typ, so a single-level scan undercounts by nine.
    """
    seen: set[Path] = set()
    stack, n = [vol], 0
    while stack:
        cur = stack.pop()
        try:
            text = cur.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in INCLUDE_RE.findall(text):
            target = (cur.parent / raw).resolve()
            if target in seen or not target.exists():
                continue
            seen.add(target)
            n += 1
            stack.append(target)
    return n


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


def add_canonical(dest: Path, slug: str) -> None:
    """Point search engines at the curated volume page, not this export.

    Both pages carry the same title and cover the same material, so without
    this they compete for the same query. Typst generates <head> itself and
    rejects a second one emitted from the document body, so the tag has to
    be spliced in afterwards.
    """
    text = dest.read_text(encoding="utf-8")
    if 'rel="canonical"' in text or "</head>" not in text:
        return
    tag = '<link rel="canonical" href="' + SITE + "volumes/" + slug + '.html">'
    dest.write_text(text.replace("</head>", tag + "</head>", 1),
                    encoding="utf-8")


INDEX_CSS = """
  :root { color-scheme: light dark; --ink: #17150f; --paper: #faf8f3;
          --soft: #8883; }
  @media (prefers-color-scheme: dark) {
    :root { --ink: #ece8e0; --paper: #14130f; }
  }
  body { background: var(--paper); color: var(--ink);
         font: 400 17px/1.6 Newsreader, Georgia, serif;
         max-width: 42rem; margin: 0 auto; padding: 2rem 1.2rem 5rem; }
  a { color: inherit; text-underline-offset: 3px; }
  .back { font: 500 .78rem/1 Inter, system-ui, sans-serif;
          letter-spacing: .04em; text-transform: uppercase;
          text-decoration: none; opacity: .7; }
  .back:hover { opacity: 1; }
  h1 { font: 600 2rem/1.2 Newsreader, Georgia, serif; margin: 2rem 0 .5rem; }
  .note { font-size: .95rem; opacity: .8; border-left: 2px solid var(--soft);
          padding-left: .9rem; margin: 1.2rem 0 2rem; }
  ul { list-style: none; padding: 0; margin: 0; }
  li { display: flex; justify-content: space-between; align-items: baseline;
       gap: 1rem; padding: .5rem 0; border-bottom: 1px solid var(--soft); }
  .kb { font: 400 .78rem/1 Inter, system-ui, sans-serif; opacity: .55;
        white-space: nowrap; }
  footer { margin-top: 3rem; font: 400 .8rem/1.5 Inter, system-ui, sans-serif;
           opacity: .65; }
"""

FONTS = ("https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;"
         "0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600&display=swap")


def write_index(out_dir: Path,
                built: list[tuple[str, str, int, int]]) -> None:
    """Listing page for the compiled volumes.

    Mirrors the typography of the curated site (scripts/gen_homepage.py) so
    this does not read as a different property, while staying self-contained
    rather than importing that module's stylesheet wholesale.
    """
    rows = "\n".join(
        '    <li><a href="{s}.html">{t}</a>'
        '<span class="kb">{c} ch · {k:,} KB</span></li>'.format(
            s=html.escape(slug), t=html.escape(title), c=chapters, k=kb)
        for slug, title, chapters, kb in built
    )
    page = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Compiled volumes — Grimoire</title>\n"
        '<meta name="description" content="Full text of every Grimoire volume, '
        'rendered to HTML from the Typst sources.">\n'
        '<link rel="canonical" href="' + SITE + 'compile/index.html">\n'
        '<link href="' + FONTS + '" rel="stylesheet">\n'
        "<style>" + INDEX_CSS + "</style>\n"
        "</head>\n<body>\n"
        '  <a class="back" href="../index.html">← Grimoire</a>\n'
        "  <h1>Compiled volumes</h1>\n"
        '  <p class="note">The full text of every volume, rendered to HTML '
        "from the current sources on every deploy. Typst’s HTML export is "
        'experimental; the <a href="https://github.com/poyea/grimoire/'
        'releases">released PDFs</a> are better typeset, but are cut per '
        "release and may lag these pages.</p>\n"
        "  <ul>\n" + rows + "\n  </ul>\n"
        "  <footer>\n"
        '    by <a href="https://github.com/poyea">@poyea</a> ·\n'
        '    <a href="https://github.com/poyea/grimoire">source ↗</a>\n'
        "  </footer>\n</body>\n</html>\n"
    )
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/compile")
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

    built: list[tuple[str, str, int, int]] = []
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
        add_canonical(dest, vol.stem)
        kb = dest.stat().st_size // 1024
        built.append((vol.stem, display_title(vol),
                      chapter_count(vol), kb))
        flag = f"  ({len(notes)} note(s))" if notes else ""
        print(f"ok    {vol.stem:<32} {kb:>6,} KB{flag}")
        for n in notes[:3]:
            print(f"        {n}")

    write_index(out_dir, built)
    total = sum(kb for *_, kb in built)
    print(f"\n{len(built)} volume(s) -> {args.out} ({total:,} KB total); "
          f"{len(failed)} failed")
    if failed:
        print("failed: " + ", ".join(failed))
    return 1 if (failed and args.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
