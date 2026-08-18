#!/usr/bin/env python3
"""Lint Grimoire chapters and the chapters.yml manifest.

Checks performed:
  1. Every `#include "<subject>/<slug>.typ"` directive in each top-level
     `<subject>.typ` resolves to a file that exists.
  2. Every chapter file on disk is referenced by its parent subject file
     (no orphans).
  3. Chapter word counts: warns on files >5000 (suggest splitting) or
     <900 (suggest merging). Configurable via env vars.
  4. chapters.yml (if present) is in sync with the filesystem: every
     chapter entry exists; every chapter file has an entry.

Exit codes:
  0 — clean
  1 — one or more errors (broken includes, orphans, manifest drift)
  2 — usage error

Warnings (size thresholds) do not affect the exit code unless --strict
is passed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

SUBJECTS = [
    "cloud-and-infrastructure",
    "coding",
    "compilers",
    "cpu-architecture",
    "cryptography-and-security",
    "data-engineering",
    "database",
    "distributed-systems",
    "embedded-and-realtime",
    "formal-methods",
    "gpu-architecture",
    "graphics-and-rendering",
    "linux-kernel",
    "llm",
    "machine-learning-foundations",
    "networking",
    "observability-and-sre",
    "operating-systems",
    "programming-languages",
    "quantum-computing",
    "computer-vision",
    "software-architecture",
    "performance-engineering",
    "numerical-computing",
    "web-and-browsers",
    "search-and-ir",
]

INCLUDE_RE = re.compile(r'#include\s+"([^"]+\.typ)"')

# Chapter size is measured in words, not lines. Chapters wrap each
# paragraph onto a single long line, so a line count reflects paragraph
# style rather than substance: search-and-ir/vector-search.typ is 71
# lines but 1350 words, more prose than a 244-line chapter in coding/.
MAX_WORDS = int(os.environ.get("GRIMOIRE_MAX_WORDS", "5000"))
MIN_WORDS = int(os.environ.get("GRIMOIRE_MIN_WORDS", "900"))


def count_words(p: Path) -> int:
    return len(p.read_text(encoding="utf-8",
                           errors="replace").split())


def parse_includes(subject_typ: Path, root: Path
                   ) -> tuple[list[str], list[tuple[str, str]]]:
    """Every chapter reachable from a subject file, following nesting.

    Includes nest and resolve relative to the *including* file, not the
    repo root: coding.typ includes coding/distributed-algorithms.typ,
    which itself includes advanced-java/*.typ. A single-level scan misses
    those nine chapters entirely even though they ship in the PDF.

    Returns (repo-relative paths that exist, [(source, raw include)] that
    do not).
    """
    seen: set[str] = set()
    found: list[str] = []
    broken: list[tuple[str, str]] = []
    stack = [subject_typ]
    while stack:
        cur = stack.pop()
        try:
            text = cur.read_text(encoding="utf-8")
        except OSError:
            continue
        src = cur.relative_to(root).as_posix()
        for raw in INCLUDE_RE.findall(text):
            target = (cur.parent / raw).resolve()
            try:
                rel = target.relative_to(root).as_posix()
            except ValueError:
                broken.append((src, raw))
                continue
            if rel in seen:
                continue
            seen.add(rel)
            if not target.exists():
                broken.append((src, raw))
                continue
            found.append(rel)
            stack.append(target)
    return found, broken


def collect_chapter_files(root: Path, subject: str) -> set[str]:
    d = root / subject
    if not d.is_dir():
        return set()
    return {p.relative_to(root).as_posix()
            for p in d.rglob("*.typ") if p.is_file()}


def load_manifest(root: Path) -> dict | None:
    path = root / "chapters.yml"
    if not path.exists():
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        print("note: PyYAML not installed; skipping manifest sync check", file=sys.stderr)
        return None
    with path.open() as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="Repo root (default: cwd)")
    ap.add_argument("--strict", action="store_true",
                    help="Treat size warnings as errors")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    errors: list[str] = []
    warnings: list[str] = []

    # 1 & 2: include resolution and orphan detection.
    for subject in SUBJECTS:
        subject_typ = root / f"{subject}.typ"
        if not subject_typ.exists():
            errors.append(f"missing subject file: {subject}.typ")
            continue
        included, broken = parse_includes(subject_typ, root)
        included_set = set(included)

        # 1. broken includes
        for src, raw in broken:
            errors.append(f"{src}: broken include {raw!r}")

        # 2. orphans
        on_disk = collect_chapter_files(root, subject)
        orphans = on_disk - included_set
        for o in sorted(orphans):
            errors.append(
                f"orphan chapter (not reachable from {subject}.typ): {o}")

    # 3. size warnings (over all chapter files on disk).
    for subject in SUBJECTS:
        for rel in sorted(collect_chapter_files(root, subject)):
            p = root / rel
            n = count_words(p)
            if n > MAX_WORDS:
                warnings.append(f"{rel}: {n} words (>{MAX_WORDS}, consider splitting)")
            elif n < MIN_WORDS:
                warnings.append(f"{rel}: {n} words (<{MIN_WORDS}, consider merging)")

    # 4. manifest sync.
    manifest = load_manifest(root)
    if manifest is not None:
        entries = manifest.get("chapters", [])
        manifest_paths = set()
        for e in entries:
            try:
                # Use `path`: nested chapters such as
                # coding/advanced-java/core-java-oop.typ cannot be
                # rebuilt from subject + slug alone.
                manifest_paths.add(
                    e.get("path") or f"{e['subject']}/{e['slug']}.typ")
            except (KeyError, TypeError):
                errors.append(f"chapters.yml: malformed entry: {e!r}")
        disk_paths = set()
        for subject in SUBJECTS:
            disk_paths |= collect_chapter_files(root, subject)
        only_disk = disk_paths - manifest_paths
        only_manifest = manifest_paths - disk_paths
        for p in sorted(only_disk):
            errors.append(f"chapters.yml missing entry for {p}")
        for p in sorted(only_manifest):
            errors.append(f"chapters.yml lists nonexistent {p}")

    # Report.
    for w in warnings:
        print(f"warning: {w}")
    for e in errors:
        print(f"error: {e}", file=sys.stderr)

    if errors:
        return 1
    if args.strict and warnings:
        return 1
    print(f"ok — {sum(len(collect_chapter_files(root, s)) for s in SUBJECTS)} chapters checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
