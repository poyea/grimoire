#!/usr/bin/env python3
"""Generate release notes from chapters.yml diff between a previous tag and HEAD.

Usage: release_notes.py [previous_tag]

If no tag is given, uses `git describe --tags --abbrev=0 HEAD^`.
Emits markdown to stdout: added / removed / changed chapters, grouped by volume.
"""

import subprocess
import sys

import yaml


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def load_chapters(ref: str | None) -> dict[str, dict]:
    """Map (subject, slug) -> chapter dict, keyed as 'subject/slug'."""
    if ref is None:
        with open("chapters.yml") as f:
            text = f.read()
    else:
        text = run("git", "show", f"{ref}:chapters.yml")
    data = yaml.safe_load(text)
    return {f"{c['subject']}/{c['slug']}": c for c in data.get("chapters", [])}


def main() -> None:
    if len(sys.argv) > 1:
        prev_tag = sys.argv[1]
    else:
        prev_tag = run("git", "describe", "--tags", "--abbrev=0", "HEAD^").strip()

    old = load_chapters(prev_tag)
    new = load_chapters(None)

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        k for k in set(old) & set(new)
        if old[k].get("lines") != new[k].get("lines")
    )

    print(f"## Chapter changes since {prev_tag}")
    print()
    if not (added or removed or changed):
        print("No chapter changes.")
        return

    def by_volume(keys: list[str]) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for k in keys:
            subject, slug = k.split("/", 1)
            groups.setdefault(subject, []).append(slug)
        return groups

    def section(title: str, keys: list[str], fmt) -> None:
        if not keys:
            return
        print(f"### {title}")
        print()
        for subject, slugs in sorted(by_volume(keys).items()):
            print(f"- **{subject}**")
            for slug in slugs:
                print(f"  - {fmt(subject, slug)}")
        print()

    section("Added", added, lambda s, g: f"{g} ({new[f'{s}/{g}']['lines']} lines)")
    section("Removed", removed, lambda s, g: g)
    section(
        "Changed",
        changed,
        lambda s, g: (
            f"{g} ({old[f'{s}/{g}']['lines']} → {new[f'{s}/{g}']['lines']} lines)"
        ),
    )


if __name__ == "__main__":
    main()
