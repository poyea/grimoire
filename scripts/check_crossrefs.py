#!/usr/bin/env python3
"""Validate `*See also:*` cross-references in Grimoire chapters.

For every chapter file, the `*See also:*` line is parsed and each
_italicised_ name is resolved against:
  (a) chapter `= Title` headings in the same volume,
  (b) chapter titles in every other volume,
  (c) volume display names (from `#project("...")` in <slug>.typ),
  (d) chapter file slugs (e.g. `programming-languages/lexing`).

Matching is fuzzy: case-insensitive; `-`, `_` and `/` treated as
spaces; parenthetical suffixes and `: subtitle` parts of titles are
ignored; a name also resolves if it appears as a whole-word phrase
inside a title (e.g. "Scheduler" -> "The Scheduler") or equals the
title's acronym (e.g. "RAG").

Unresolved references are reported and exit nonzero.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ITALIC_RE = re.compile(r"_([^_]+)_")
TITLE_RE = re.compile(r"^=\s+(.+?)\s*$", re.M)
PROJECT_RE = re.compile(r'#project\("([^"]+)"\)')
PAREN_RE = re.compile(r"\s*\([^()]*\)\s*$")


def canon(name: str) -> str:
    name = re.sub(r"[-_/&]", " ", name.lower())
    return re.sub(r"\s+", " ", name).strip()


def strip_parens(name: str) -> str:
    prev = None
    while prev != name:
        prev = name
        name = PAREN_RE.sub("", name).strip()
    return name


def title_aliases(title: str) -> set[str]:
    """Canonical alias strings for one chapter title."""
    aliases = {canon(title)}
    aliases.add(canon(strip_parens(title)))          # drop "(...)" suffix
    aliases.add(canon(title.split(":", 1)[0]))       # drop ": subtitle"
    aliases.add(canon(re.sub(r"\([^()]*\)", " ", title)))  # drop inner parens
    # acronym of significant words ("Retrieval-Augmented Generation" -> rag)
    words = [w for w in re.split(r"[\s\-]+", strip_parens(title))
             if w and w.lower() not in ("and", "of", "the", "a", "an",
                                        "for", "in", "on", "vs")]
    if len(words) >= 2:
        aliases.add("".join(w[0].lower() for w in words))
    aliases.discard("")
    return aliases


class Volume:
    def __init__(self, slug: str, display: str):
        self.slug = slug
        self.display = display
        self.aliases: set[str] = set()      # all chapter-title aliases
        self.titles: list[str] = []         # canon full titles (containment)
        self.stems: set[str] = set()        # chapter file stems


def load() -> dict[str, Volume]:
    vols: dict[str, Volume] = {}
    for vfile in sorted(ROOT.glob("*.typ")):
        slug = vfile.stem
        if slug == "template" or not (ROOT / slug).is_dir():
            continue
        m = PROJECT_RE.search(vfile.read_text(encoding="utf-8"))
        vol = Volume(slug, m.group(1) if m else slug)
        for chap in sorted((ROOT / slug).glob("*.typ")):
            vol.stems.add(chap.stem)
            t = TITLE_RE.search(chap.read_text(encoding="utf-8"))
            if t:
                vol.aliases |= title_aliases(t.group(1))
                vol.titles.append(canon(t.group(1)))
        vols[slug] = vol
    return vols


def volume_by_name(name: str, vols: dict[str, Volume]) -> Volume | None:
    c = canon(name)
    for vol in vols.values():
        if c in (canon(vol.display), canon(vol.slug)):
            return vol
    return None


def in_volume(name: str, vol: Volume) -> bool:
    c = canon(strip_parens(name))
    if not c:
        return False
    if c in vol.aliases or c in vol.stems or name.strip() in vol.stems:
        return True
    # whole-phrase containment in a full title, for names of >=4 chars
    if len(c) >= 4:
        pat = re.compile(r"(?:^|\s)" + re.escape(c) + r"(?:\s|$)")
        if any(pat.search(t) for t in vol.titles):
            return True
    return False


def resolve(name: str, here: str, vols: dict[str, Volume]) -> bool:
    name = name.strip().strip(",.;:")
    if not name:
        return True
    # explicit cross-volume pointer: "Title (Some Volume volume)"
    m = re.search(r"\(([^()]*?)\s+volume\)\s*$", name, re.I)
    if m:
        target = volume_by_name(m.group(1), vols)
        base = re.sub(r"\([^()]*\)\s*$", "", name)
        if target and in_volume(base, target):
            return True
    # bare volume reference: "Networking volume" or a volume display name
    bare = re.sub(r"\s+volume$", "", strip_parens(name), flags=re.I)
    if volume_by_name(bare, vols):
        return True
    # slug-style "vol/chapter"
    if "/" in name:
        vslug, _, stem = name.strip().partition("/")
        v = vols.get(vslug.strip())
        if v and stem.strip() in v.stems:
            return True
    if in_volume(name, vols[here]):
        return True
    return any(in_volume(name, v) for s, v in vols.items() if s != here)


def main() -> int:
    vols = load()
    warnings = 0
    for slug in sorted(vols):
        for chap in sorted((ROOT / slug).glob("*.typ")):
            for line in chap.read_text(encoding="utf-8").splitlines():
                if "*See also:*" not in line:
                    continue
                for name in ITALIC_RE.findall(line):
                    if not resolve(name, slug, vols):
                        rel = chap.relative_to(ROOT)
                        print(f"WARNING: {rel}: unresolved reference "
                              f"'{name.strip()}'")
                        warnings += 1
    if warnings:
        print(f"\n{warnings} unresolved cross-reference(s).")
        return 1
    print("All cross-references resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
