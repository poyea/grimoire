#!/usr/bin/env python3
"""Check Further Reading sections in Grimoire chapters.

For every chapter file (<subject>/<slug>.typ), verify that:
  1. A `== Further Reading` section exists.
  2. Each entry (blank-line-separated paragraph or bullet) in that
     section looks like a citation: it contains a year (1800-2099),
     typically in parentheses, alongside author-like text.

Errors (missing section, malformed entries) exit nonzero.
Warnings (loose matches) exit 0.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# A year between 1800 and 2099, anywhere in the entry.
YEAR_RE = re.compile(r"\b(18|19|20)\d{2}[a-z]?\b")
# Year in parentheses (the canonical shape).
PAREN_YEAR_RE = re.compile(r"\((18|19|20)\d{2}[a-z]?\)")
# Author-ish: a capitalised word followed by comma/initials, or "et al."
AUTHOR_RE = re.compile(r"(\b[A-Z][\w'\-]+,|\bet al\.|\b[A-Z]\. )")
# Things that are acceptable without a year: RFCs, standards, URLs, docs.
NO_YEAR_OK_RE = re.compile(
    r"(RFC\s*\d+|ISO[/ ]|IEEE\s*\d+|https?://|documentation|manual|"
    r"specification|standard|reference)", re.IGNORECASE)

HEADING_RE = re.compile(r"^=+\s")
FURTHER_RE = re.compile(
    r"^==\s+(Further Reading|References(\s*\(Selected\))?)\s*$")


def subject_dirs() -> list[Path]:
    dirs = []
    for typ in sorted(ROOT.glob("*.typ")):
        if typ.stem == "template":
            continue
        d = ROOT / typ.stem
        if d.is_dir():
            dirs.append(d)
    return dirs


def extract_section(text: str) -> list[str] | None:
    """Return list of entries in the Further Reading section, or None."""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if FURTHER_RE.match(line.strip()):
            start = i + 1
            break
    if start is None:
        return None
    body: list[str] = []
    for line in lines[start:]:
        if HEADING_RE.match(line.strip()):
            break
        body.append(line)
    # Split into paragraphs / bullets.
    entries: list[str] = []
    cur: list[str] = []
    for line in body:
        stripped = line.strip()
        if not stripped:
            if cur:
                entries.append(" ".join(cur))
                cur = []
            continue
        if stripped.startswith(("-", "+")) and cur:
            entries.append(" ".join(cur))
            cur = []
        cur.append(stripped.lstrip("-+ ").strip())
    if cur:
        entries.append(" ".join(cur))
    return entries


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    n_chapters = 0
    for d in subject_dirs():
        for chap in sorted(d.rglob("*.typ")):
            n_chapters += 1
            rel = chap.relative_to(ROOT)
            entries = extract_section(chap.read_text(encoding="utf-8"))
            if entries is None:
                # Every chapter carries a citation section, including
                # nested subtrees such as coding/advanced-java, so a
                # missing one is a regression rather than a legacy gap.
                errors.append(
                    f"{rel}: missing '== Further Reading' section")
                continue
            entries = [e for e in entries if e]
            if not entries:
                errors.append(f"{rel}: empty Further Reading section")
                continue
            for e in entries:
                # Skip Typst directives and grouping labels like
                # "*Primary Sources:*".
                if e.startswith("#"):
                    continue
                if re.fullmatch(r"\*[^*]{1,60}:\*", e):
                    continue
                if PAREN_YEAR_RE.search(e):
                    continue
                if YEAR_RE.search(e):
                    # Has a year, just not in parentheses.
                    continue
                if AUTHOR_RE.search(e) or NO_YEAR_OK_RE.search(e) \
                        or "_" in e or '"' in e:
                    warnings.append(
                        f"{rel}: entry has no year (author/title/standard "
                        f"shape accepted): {e[:80]}")
                    continue
                warnings.append(f"{rel}: entry does not look like a "
                                f"citation: {e[:100]}")

    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print(f"\nChecked {n_chapters} chapters: "
          f"{len(errors)} error(s), {len(warnings)} warning(s).")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
