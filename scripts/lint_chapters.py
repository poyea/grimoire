#!/usr/bin/env python3
"""Lint Grimoire chapters and the chapters.yml manifest.

Checks performed:
  1. Every `#include "<subject>/<slug>.typ"` directive in each top-level
     `<subject>.typ` resolves to a file that exists.
  2. Every chapter file on disk is referenced by its parent subject file
     (no orphans).
  3. Chapter line counts: warns on files >800 (suggest splitting) or
     <120 (suggest merging). Configurable via env vars.
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
    "coding",
    "cpu-architecture",
    "database",
    "gpu-architecture",
    "linux-kernel",
    "llm",
    "networking",
    "programming-languages",
]

INCLUDE_RE = re.compile(r'#include\s+"([^"]+\.typ)"')

MAX_LINES = int(os.environ.get("GRIMOIRE_MAX_LINES", "800"))
MIN_LINES = int(os.environ.get("GRIMOIRE_MIN_LINES", "120"))


def count_lines(p: Path) -> int:
    with p.open("rb") as f:
        return sum(1 for _ in f)


def parse_includes(subject_typ: Path) -> list[str]:
    text = subject_typ.read_text(encoding="utf-8")
    return INCLUDE_RE.findall(text)


def collect_chapter_files(root: Path, subject: str) -> set[str]:
    d = root / subject
    if not d.is_dir():
        return set()
    out = set()
    for p in d.iterdir():
        if p.is_file() and p.suffix == ".typ":
            out.add(f"{subject}/{p.name}")
    return out


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
        included = parse_includes(subject_typ)
        included_set = set(included)

        # 1. broken includes
        for rel in included:
            if not (root / rel).exists():
                errors.append(f"{subject}.typ: broken include {rel!r}")

        # 2. orphans
        on_disk = collect_chapter_files(root, subject)
        orphans = on_disk - included_set
        for o in sorted(orphans):
            errors.append(f"orphan chapter (not #included by {subject}.typ): {o}")

    # 3. size warnings (over all chapter files on disk).
    for subject in SUBJECTS:
        for rel in sorted(collect_chapter_files(root, subject)):
            p = root / rel
            n = count_lines(p)
            if n > MAX_LINES:
                warnings.append(f"{rel}: {n} lines (>{MAX_LINES}, consider splitting)")
            elif n < MIN_LINES:
                warnings.append(f"{rel}: {n} lines (<{MIN_LINES}, consider merging)")

    # 4. manifest sync.
    manifest = load_manifest(root)
    if manifest is not None:
        entries = manifest.get("chapters", [])
        manifest_paths = set()
        for e in entries:
            try:
                manifest_paths.add(f"{e['subject']}/{e['slug']}.typ")
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
