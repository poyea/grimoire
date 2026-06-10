#!/usr/bin/env python3
"""Regenerate the static site in docs/ from the chapter files.

Volume order, display titles, and descriptions are editorial and live in
VOLUMES below. Topics are derived from each chapter's `= Heading`, in the
include order of the subject's root .typ file. The volume count and the
section numeral range are updated to match.

Outputs:
  - docs/index.html      (VOLUMES array + counts updated in place)
  - docs/volumes/<slug>.html  (one page per volume)
  - docs/search.json     (client-side search index)
  - docs/sitemap.xml
  - docs/feed.xml        (Atom; entries from git tags)

Usage: python3 scripts/gen_homepage.py
"""

import html as html_mod
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
INDEX = DOCS / "index.html"
VOLDIR = DOCS / "volumes"
SITE = "https://poyea.github.io/grimoire/"
RELEASE_DL = "https://github.com/poyea/grimoire/releases/latest/download/"
RELEASES = "https://github.com/poyea/grimoire/releases"

# (slug, display title, description)
VOLUMES = [
    ("coding", "Coding",
     "Algorithms and data structures: a practitioner's reference for interviews, contests, and engineering."),
    ("cpu-architecture", "CPU Architecture",
     "Processor design, instruction sets, and performance analysis from pipeline to silicon."),
    ("gpu-architecture", "GPU Architecture",
     "Parallel computing, CUDA internals, Tensor Cores, and ML workload optimization."),
    ("networking", "Networking",
     "Network protocols, congestion control, data-centre fabric, and modern application protocols."),
    ("linux-kernel", "Linux Kernel",
     "Internals for systems engineers: syscalls, memory, scheduling, containers, tracing, and networking."),
    ("llm", "Large Language Models",
     "Transformer internals, training, RLHF, inference optimization, agents, and safety."),
    ("programming-languages", "Languages & Compilers",
     "Automata, semantics, type theory, and a compiler from scratch — from Myhill–Nerode to HoTT."),
    ("database", "Databases",
     "Storage engines, concurrency, query optimization, distributed transactions, and modern analytics."),
    ("distributed-systems", "Distributed Systems",
     "Consensus, replication, time, transactions, failure detection, and workflow engines."),
    ("operating-systems", "Operating Systems",
     "Processes, scheduling, memory management, storage, boot, and the system call interface."),
    ("machine-learning-foundations", "Machine Learning",
     "Mathematical foundations, optimization, generalization theory, and modern training techniques."),
    ("cryptography-and-security", "Cryptography & Security",
     "Symmetric primitives, asymmetric cryptography, hashing, and post-quantum schemes."),
    ("cloud-and-infrastructure", "Cloud & Infrastructure",
     "IaaS fundamentals, Kubernetes internals, serverless, IaC, multi-tenancy, and cost engineering."),
    ("data-engineering", "Data Engineering",
     "Batch processing, streaming, ETL/ELT, orchestration, and lakehouse architecture."),
    ("observability-and-sre", "Observability & SRE",
     "Metrics, logs, distributed tracing, continuous profiling, and SRE principles."),
    ("graphics-and-rendering", "Graphics & Rendering",
     "Rasterization, ray tracing, physically-based rendering, global illumination, and real-time engines."),
    ("compilers", "Compilers",
     "Frontend parsing, IR design, dataflow analysis, and optimization passes."),
    ("formal-methods", "Formal Methods",
     "Model checking, TLA+, SAT/SMT, theorem proving, and separation logic."),
    ("embedded-and-realtime", "Embedded & Real-Time",
     "RTOS internals, hardware interfaces, scheduling, and safety-critical systems."),
    ("quantum-computing", "Quantum Computing",
     "Qubits, quantum algorithms, error correction, NISQ devices, and hardware architectures."),
    ("computer-vision", "Computer Vision",
     "Image formation, CNNs, detection, segmentation, vision transformers, and 3D neural fields."),
    ("software-architecture", "Software Architecture",
     "Architectural styles, DDD, microservices, resilience, and evolutionary architecture."),
    ("performance-engineering", "Performance Engineering",
     "Methodology, profiling, benchmarking, queueing theory, and capacity planning."),
    ("numerical-computing", "Numerical Computing",
     "Floating point, error analysis, linear systems, FFT, ODEs, and optimization."),
    ("web-and-browsers", "Web and Browser Internals",
     "Browser architecture, rendering, JavaScript engines, WebAssembly, and web performance."),
    ("search-and-ir", "Search and Information Retrieval",
     "Inverted indexes, ranking, neural retrieval, vector search, and RAG systems."),
]

INCLUDE_RE = re.compile(r'#include\s+"([^"]+\.typ)"')
HEADING_RE = re.compile(r"^=\s+(.+)$", re.MULTILINE)
SUBHEAD_RE = re.compile(r"^==\s+(.+)$", re.MULTILINE)


def roman(n: int) -> str:
    vals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"),
            (90, "XC"), (50, "L"), (40, "XL"), (10, "X"), (9, "IX"),
            (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for v, s in vals:
        while n >= v:
            out += s
            n -= v
    return out


def js_str(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def chapters(slug: str) -> list[tuple[str, list[str]]]:
    """Return [(chapter title, [subheadings])] in include order."""
    root_typ = ROOT / f"{slug}.typ"
    out = []
    for path in INCLUDE_RE.findall(root_typ.read_text()):
        text = (ROOT / path).read_text()
        m = HEADING_RE.search(text)
        if not m:
            sys.exit(f"error: no `= Heading` in {path}")
        out.append((m.group(1).strip(),
                    [s.strip() for s in SUBHEAD_RE.findall(text)]))
    if not out:
        sys.exit(f"error: no includes found in {root_typ}")
    return out


def esc(s: str) -> str:
    return html_mod.escape(s, quote=True)


SHARED_CSS = """\
  :root {
    --paper: #fafaf7; --paper-dim: #efeee8; --ink: #111; --ink-soft: #444;
    --rule: #d4d2cc; --rule-soft: #e8e6e0; --accent: #9b1c1c;
  }
  :root[data-theme="dark"] {
    --paper: #171614; --paper-dim: #211f1c; --ink: #ece9e2; --ink-soft: #b3aea4;
    --rule: #3d3a35; --rule-soft: #2c2a26; --accent: #e06a5c;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --paper: #171614; --paper-dim: #211f1c; --ink: #ece9e2; --ink-soft: #b3aea4;
      --rule: #3d3a35; --rule-soft: #2c2a26; --accent: #e06a5c;
    }
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--paper); color: var(--ink);
    font-family: 'Newsreader', Georgia, serif;
    min-height: 100vh; line-height: 1.5;
    -webkit-font-smoothing: antialiased;
  }
  .wrap { max-width: 880px; margin: 0 auto; padding: 3rem 2rem 4rem; }
  .kicker {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    letter-spacing: 0.25em; color: var(--accent); text-transform: uppercase;
    margin-bottom: 1.2rem;
  }
  .topbar { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
  .back {
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
    color: var(--accent); text-decoration: none; letter-spacing: 0.06em;
  }
  .theme-toggle {
    background: none; border: 1px solid var(--rule); color: var(--ink-soft);
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    padding: 0.25rem 0.6rem; cursor: pointer; letter-spacing: 0.06em;
  }
  .theme-toggle:hover { border-color: var(--ink); color: var(--ink); }
  h1 {
    font-family: 'Newsreader', serif; font-weight: 500;
    font-size: clamp(2.4rem, 7vw, 4rem); letter-spacing: -0.02em;
    line-height: 1.02; margin: 1rem 0 0.8rem;
    border-bottom: 1px solid var(--ink); padding-bottom: 1.2rem;
  }
  .desc { font-size: 1.15rem; color: var(--ink-soft); margin: 1rem 0 1.5rem; max-width: 60ch; }
  .meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
    letter-spacing: 0.06em; color: var(--ink-soft);
    border-top: 1px solid var(--rule); border-bottom: 1px solid var(--rule);
    padding: 0.8rem 0; margin-bottom: 2.5rem;
    display: flex; gap: 2rem; flex-wrap: wrap;
  }
  .meta strong { color: var(--ink); font-weight: 500; }
  .meta a.pdf {
    color: var(--paper); background: var(--ink); text-decoration: none;
    padding: 0.35rem 1rem; transition: background 0.2s;
  }
  .meta a.pdf:hover { background: var(--accent); }
  .toc details {
    border-bottom: 1px solid var(--rule-soft); padding: 0.55rem 0;
  }
  .toc summary {
    cursor: pointer; list-style: none;
    font-family: 'Newsreader', serif; font-size: 1.15rem;
    display: flex; gap: 0.9rem; align-items: baseline;
  }
  .toc summary::-webkit-details-marker { display: none; }
  .toc summary .n {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: var(--accent); letter-spacing: 0.1em; min-width: 2.2em;
  }
  .toc summary:hover { color: var(--accent); }
  .toc ul { margin: 0.5rem 0 0.3rem 3.1em; list-style: none; }
  .toc li {
    font-family: 'JetBrains Mono', monospace; font-size: 0.74rem;
    color: var(--ink-soft); letter-spacing: 0.03em; padding: 0.12rem 0;
  }
  footer {
    margin-top: 3rem; border-top: 1px solid var(--rule); padding-top: 1rem;
    color: var(--ink-soft); font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; display: flex; justify-content: space-between;
    flex-wrap: wrap; gap: 1rem;
  }
  footer a { color: var(--accent); text-decoration: none; }
"""

THEME_HEAD_JS = """\
<script>
  (function () {
    var t = localStorage.getItem('grimoire-theme');
    if (t) document.documentElement.setAttribute('data-theme', t);
  })();
</script>"""

THEME_TOGGLE_JS = """\
<script>
  document.getElementById('themeToggle').addEventListener('click', function () {
    var root = document.documentElement;
    var cur = root.getAttribute('data-theme');
    if (!cur) cur = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    var next = cur === 'dark' ? 'light' : 'dark';
    root.setAttribute('data-theme', next);
    localStorage.setItem('grimoire-theme', next);
  });
  function openHash() {
    if (!location.hash) return;
    var el = document.getElementById(location.hash.slice(1));
    if (el && el.tagName === 'DETAILS') { el.open = true; el.scrollIntoView(); }
  }
  openHash();
  window.addEventListener('hashchange', openHash);
</script>"""


def volume_page(i: int, slug: str, title: str, desc: str,
                chs: list[tuple[str, list[str]]]) -> str:
    pdf = f"{RELEASE_DL}grimoire_{slug.replace('-', '_')}.pdf"
    items = []
    for n, (ctitle, subs) in enumerate(chs, 1):
        sub_html = ""
        if subs:
            sub_html = "\n      <ul>\n" + "\n".join(
                f"        <li>{esc(s)}</li>" for s in subs) + "\n      </ul>"
        items.append(
            f'    <details id="ch-{n}">\n'
            f'      <summary><span class="n">{n:02d}</span>{esc(ctitle)}</summary>'
            f'{sub_html}\n    </details>'
        )
    toc = "\n".join(items)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{esc(title)} — Grimoire</title>
<meta name="description" content="{esc(desc)}">
<link rel="canonical" href="{SITE}volumes/{slug}.html">
{THEME_HEAD_JS}
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
<style>
{SHARED_CSS}</style>
</head>
<body>
<div class="wrap">
  <div class="topbar">
    <a class="back" href="../index.html">← Grimoire</a>
    <button class="theme-toggle" id="themeToggle" aria-label="Toggle dark mode">◐ theme</button>
  </div>
  <p class="kicker" style="margin-top:1.5rem">Volume {roman(i)} · § {i:02d}</p>
  <h1>{esc(title)}</h1>
  <p class="desc">{esc(desc)}</p>
  <div class="meta">
    <span><strong>CHAPTERS</strong> · {len(chs)}</span>
    <span><strong>FORMAT</strong> · PDF</span>
    <a class="pdf" href="{pdf}">Download PDF ↓</a>
  </div>
  <div class="toc">
{toc}
  </div>
  <footer>
    <span>by <a href="https://github.com/poyea">@poyea</a> with love</span>
    <span><a href="{RELEASES}">all releases ↗</a> · <a href="https://github.com/poyea/grimoire">source ↗</a></span>
  </footer>
</div>
{THEME_TOGGLE_JS}
</body>
</html>
"""


def git_tags() -> list[tuple[str, str]]:
    """[(tag, ISO date)] newest first; empty list if git unavailable."""
    try:
        tags = subprocess.run(
            ["git", "tag", "--sort=-creatordate"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout.split()
        out = []
        for tag in tags[:15]:
            date = subprocess.run(
                ["git", "log", "-1", "--format=%cI", tag],
                cwd=ROOT, capture_output=True, text=True, check=True,
            ).stdout.strip()
            out.append((tag, date))
        return out
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []


def write_feed() -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    entries = []
    for tag, date in git_tags():
        url = f"{RELEASES}/tag/{tag}"
        entries.append(
            f"  <entry>\n"
            f"    <title>Grimoire {esc(tag)}</title>\n"
            f'    <link href="{url}"/>\n'
            f"    <id>{url}</id>\n"
            f"    <updated>{esc(date)}</updated>\n"
            f"    <summary>Release {esc(tag)} of the Grimoire volumes.</summary>\n"
            f"  </entry>"
        )
    if not entries:
        entries.append(
            f"  <entry>\n"
            f"    <title>Grimoire releases</title>\n"
            f'    <link href="{RELEASES}"/>\n'
            f"    <id>{RELEASES}</id>\n"
            f"    <updated>{now}</updated>\n"
            f"    <summary>All Grimoire releases.</summary>\n"
            f"  </entry>"
        )
    feed = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<feed xmlns="http://www.w3.org/2005/Atom">\n'
        "  <title>Grimoire</title>\n"
        f'  <link href="{SITE}"/>\n'
        f'  <link rel="self" href="{SITE}feed.xml"/>\n'
        f"  <id>{SITE}</id>\n"
        f"  <updated>{now}</updated>\n"
        '  <author><name>poyea</name></author>\n'
        + "\n".join(entries) + "\n</feed>\n"
    )
    (DOCS / "feed.xml").write_text(feed)


def write_sitemap() -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    urls = [SITE] + [f"{SITE}volumes/{slug}.html" for slug, _, _ in VOLUMES]
    body = "\n".join(
        f"  <url><loc>{u}</loc><lastmod>{today}</lastmod></url>" for u in urls
    )
    (DOCS / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{body}\n</urlset>\n"
    )


def main() -> None:
    data = {slug: chapters(slug) for slug, _, _ in VOLUMES}

    # 1) index.html VOLUMES array
    cards = []
    for i, (slug, title, desc) in enumerate(VOLUMES, 1):
        topics = ",".join(js_str(t) for t, _ in data[slug])
        cards.append(
            f"    {{ id: '{i:02d}', slug: {js_str(slug)}, title: {js_str(title)},\n"
            f"      desc: {js_str(desc)},\n"
            f"      topics: [{topics}]\n"
            f"    }},"
        )
    block = "  const VOLUMES = [\n" + "\n".join(cards) + "\n  ];"

    html = INDEX.read_text()
    new_html, n = re.subn(
        r"  const VOLUMES = \[.*?\n  \];", block, html, count=1, flags=re.DOTALL
    )
    if n != 1:
        sys.exit("error: VOLUMES array not found in index.html")
    new_html = re.sub(
        r"(<strong>VOLUMES</strong> · )\d+", rf"\g<1>{len(VOLUMES)}", new_html
    )
    new_html = re.sub(
        r'(<span class="index">§ I — § )[IVXLCDM]+',
        rf"\g<1>{roman(len(VOLUMES))}", new_html,
    )
    INDEX.write_text(new_html)
    print(f"Wrote {len(VOLUMES)} volumes to {INDEX.relative_to(ROOT)}")

    # 2) per-volume pages
    VOLDIR.mkdir(exist_ok=True)
    for i, (slug, title, desc) in enumerate(VOLUMES, 1):
        (VOLDIR / f"{slug}.html").write_text(
            volume_page(i, slug, title, desc, data[slug]))
    print(f"Wrote {len(VOLUMES)} pages to {VOLDIR.relative_to(ROOT)}/")

    # 3) search index
    search = []
    for slug, title, _ in VOLUMES:
        for n, (ctitle, subs) in enumerate(data[slug], 1):
            search.append({"slug": slug, "volume": title, "chapter": ctitle,
                           "anchor": f"ch-{n}", "headings": subs})
    (DOCS / "search.json").write_text(
        json.dumps(search, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Wrote {len(search)} entries to docs/search.json")

    # 4) sitemap + feed
    write_sitemap()
    write_feed()
    print("Wrote docs/sitemap.xml and docs/feed.xml")


if __name__ == "__main__":
    main()
