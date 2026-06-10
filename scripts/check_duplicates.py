#!/usr/bin/env python3
"""Detect near-duplicate paragraphs across different Grimoire chapters.

Paragraphs longer than 200 characters (after normalisation) are broken
into word 5-shingles; pairs of paragraphs from *different* chapter files
with Jaccard shingle overlap >= 0.7 are reported as warnings.

Always exits 0 — this is an advisory check.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MIN_PARA_LEN = 200
SHINGLE_K = 5
THRESHOLD = 0.7
TOP_N = 30


def subject_dirs() -> list[Path]:
    return [ROOT / t.stem for t in sorted(ROOT.glob("*.typ"))
            if t.stem != "template" and (ROOT / t.stem).is_dir()]


def normalize(text: str) -> str:
    text = re.sub(r"[`*_#\[\]{}()|\"',.;:!?$\\/<>=+~-]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def paragraphs(path: Path):
    text = path.read_text(encoding="utf-8")
    for raw in re.split(r"\n\s*\n", text):
        raw = raw.strip()
        if not raw or raw.startswith(("=", "#import", "#let")):
            continue
        norm = normalize(raw)
        if len(norm) >= MIN_PARA_LEN:
            yield norm


def shingles(text: str) -> frozenset:
    words = text.split()
    if len(words) < SHINGLE_K:
        return frozenset([tuple(words)])
    return frozenset(tuple(words[i:i + SHINGLE_K])
                     for i in range(len(words) - SHINGLE_K + 1))


def main() -> int:
    paras = []  # (file, index, shingle set)
    index = defaultdict(set)  # shingle -> para ids
    for d in subject_dirs():
        for chap in sorted(d.glob("*.typ")):
            rel = str(chap.relative_to(ROOT))
            for i, p in enumerate(paragraphs(chap)):
                pid = len(paras)
                sh = shingles(p)
                paras.append((rel, i, sh))
                for s in sh:
                    index[s].add(pid)

    # Candidate pairs share at least one shingle.
    candidates = set()
    for ids in index.values():
        if 1 < len(ids) <= 50:
            for a, b in combinations(sorted(ids), 2):
                if paras[a][0] != paras[b][0]:
                    candidates.add((a, b))

    results = []
    for a, b in candidates:
        sa, sb = paras[a][2], paras[b][2]
        jac = len(sa & sb) / len(sa | sb)
        if jac >= THRESHOLD:
            results.append((jac, paras[a], paras[b]))

    results.sort(reverse=True, key=lambda r: r[0])
    if not results:
        print("No near-duplicate paragraphs found "
              f"({len(paras)} paragraphs compared).")
        return 0

    print(f"WARNING: {len(results)} near-duplicate paragraph pair(s) "
          f"found (threshold {THRESHOLD}). Top {TOP_N}:\n")
    for jac, (fa, ia, _), (fb, ib, _) in results[:TOP_N]:
        print(f"  {jac:.2f}  {fa} (para {ia})  <->  {fb} (para {ib})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
