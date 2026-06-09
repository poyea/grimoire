#!/usr/bin/env python3
"""Regenerate the VOLUMES array in docs/index.html from the chapter files.

Volume order, display titles, and descriptions are editorial and live in
VOLUMES below. Topics are derived from each chapter's `= Heading`, in the
include order of the subject's root .typ file. The volume count and the
section numeral range are updated to match.

Usage: python3 scripts/gen_homepage.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "docs" / "index.html"

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
]

INCLUDE_RE = re.compile(r'#include\s+"([^"]+\.typ)"')
HEADING_RE = re.compile(r"^=\s+(.+)$", re.MULTILINE)


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


def chapter_titles(slug: str) -> list[str]:
    root_typ = ROOT / f"{slug}.typ"
    titles = []
    for path in INCLUDE_RE.findall(root_typ.read_text()):
        m = HEADING_RE.search((ROOT / path).read_text())
        if not m:
            sys.exit(f"error: no `= Heading` in {path}")
        titles.append(m.group(1).strip())
    if not titles:
        sys.exit(f"error: no includes found in {root_typ}")
    return titles


def main() -> None:
    cards = []
    for i, (slug, title, desc) in enumerate(VOLUMES, 1):
        topics = ",".join(js_str(t) for t in chapter_titles(slug))
        cards.append(
            f"    {{ id: '{i:02d}', title: {js_str(title)},\n"
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


if __name__ == "__main__":
    main()
