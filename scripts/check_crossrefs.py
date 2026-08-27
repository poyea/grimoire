#!/usr/bin/env python3
"""Validate `*See also:*` cross-references in Grimoire chapters.

Structured `#xref("subject", "slug")` calls are resolved exactly:
the target file must exist on disk. Remaining prose references on a
`*See also:*` line are parsed and each _italicised_ name is resolved
fuzzily against:
  (a) chapter `= Title` headings in the same volume,
  (b) chapter titles in every other volume,
  (c) volume display names (from `#project("...")` in <slug>.typ),
  (d) chapter file slugs (e.g. `programming-languages/lexing`).

Matching is fuzzy: case-insensitive; `-`, `_` and `/` treated as
spaces; parenthetical suffixes and `: subtitle` parts of titles are
ignored; a name also resolves if it appears as a whole-word phrase
inside a title (e.g. "Scheduler" -> "The Scheduler") or equals the
title's acronym (e.g. "RAG").

Every #xref target is checked three ways, one per rendering: the volume
exists (so its release PDF does), the chapter is reachable through the
includes (so it ships and gets a site anchor), and its heading carries the
<chapter-slug> label (so same-volume PDF links can jump to it).

Unresolved references are reported and exit nonzero.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

ITALIC_RE = re.compile(r"_([^_]+)_")
XREF_RE = re.compile(r'#xref\("([^"]+)",\s*"([^"]+)"')
TITLE_RE = re.compile(r"^=\s+(.+?)(?:\s*<[a-z0-9-]+>)?\s*$", re.M)
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
        for chap in sorted((ROOT / slug).rglob("*.typ")):
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


INCLUDE_RE = re.compile(r'#include\s+"([^"]+\.typ)"')
LABEL_RE = re.compile(r"^=\s+.*?<([a-z0-9-]+)>\s*$", re.M)


def reachable(volume: str) -> dict[str, Path]:
    """Chapter stem -> path, for chapters actually included by the volume.

    Follows nested includes, since coding/cpp-and-java/*.typ is pulled in
    from a chapter rather than from coding.typ. A file that exists but is
    not reachable ships in no PDF and gets no anchor on the site, so it is
    not a usable #xref target.
    """
    root_typ = ROOT / f"{volume}.typ"
    out: dict[str, Path] = {}
    stack, seen = [root_typ], set()
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
            out[target.stem] = target
            stack.append(target)
    return out


def main() -> int:
    vols = load()
    errors = 0
    reach = {v: reachable(v) for v in vols}
    for slug in sorted(vols):
        for chap in sorted((ROOT / slug).rglob("*.typ")):
            text = chap.read_text(encoding="utf-8")
            rel = chap.relative_to(ROOT)
            # An #xref must survive in all three renderings: an internal
            # jump inside the volume's PDF, a link to that volume's release
            # PDF from another volume, and a site anchor in HTML. Each has a
            # separate precondition, so each is checked.
            for subj, stem in XREF_RE.findall(text):
                if subj not in vols:
                    print(f"ERROR: {rel}: #xref names no such volume "
                          f"'{subj}' (so grimoire_{subj.replace('-', '_')}"
                          f".pdf will not exist)")
                    errors += 1
                    continue
                target = reach[subj].get(stem)
                if target is None:
                    where = "not reachable from" if (ROOT / subj).glob(
                        f"**/{stem}.typ") else "does not exist under"
                    print(f"ERROR: {rel}: #xref target '{subj}/{stem}' "
                          f"is {where} {subj}.typ")
                    errors += 1
                    continue
                labels = LABEL_RE.findall(target.read_text(encoding="utf-8"))
                if stem not in labels:
                    print(f"ERROR: {rel}: #xref target '{subj}/{stem}' has no "
                          f"<{stem}> label on its heading, so same-volume "
                          f"links cannot jump to it")
                    errors += 1
            for line in text.splitlines():
                if "*See also:*" not in line:
                    continue
                for name in ITALIC_RE.findall(line):
                    if not resolve(name, slug, vols):
                        print(f"ERROR: {rel}: unresolved reference "
                              f"'{name.strip()}'")
                        errors += 1
    if errors:
        print(f"\n{errors} broken cross-reference(s).")
        return 1
    print("All cross-references resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
