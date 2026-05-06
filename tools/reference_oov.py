#!/usr/bin/env python3
"""Reference for stage 5: OOV letter-to-sound rules."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

CASES = [
    "PyTorch",
    "Kubernetes",
    "TensorFlow",
    "OpenAI",
    "PostgreSQL",
    "JavaScript",
    "TypeScript",
    "GraphQL",
    "Microservice",
    "Hyperparameter",
    "Phantom",
    "Photon",
    "Phone",
    "Chocolate",
    "Shuttle",
    "Thesis",
    "Through",
    "Queue",
    "Whale",
    "Knight",
    "Write",
    "Gnome",
    "Singer",
    "Baking",
    "Running",
    "Played",
    "Happiness",
    "Reasonable",
    "Nationalization",
    "Discussion",
    "Creation",
    "Famous",
    "Curable",
    "Portable",
    "Unknown",
    "Rebuild",
    "Preview",
    "Disagree",
    "Unofficial",
    "Overclock",
    "Underflow",
    "Misaligned",
    "Nonprofit",
    "Interstellar",
    "Transcribe",
    "Science",
    "Algorithmic",
    "Kite",
    "Make",
    "Cube",
    "Little",
    "System",
    "Feature",
    "Data-driven",
    "co-operate",
    "The new PyTorch build runs on Kubernetes.",
    "TensorFlow works with the microservice.",
    "We are rebuilding the system.",
    "The non-profit group uses PostgreSQL.",
    "A whale and a knight were written about in a gnome story.",
]


def espeak_ipa(text: str) -> str:
    if shutil.which("espeak") is None:
        raise SystemExit("espeak binary not found")
    proc = subprocess.run(
        ["espeak", "-q", "-v", "en-us", "--ipa=3", text],
        check=True,
        capture_output=True,
        text=True,
    )
    return normalize_ipa(proc.stdout.strip())


def normalize_ipa(text: str) -> str:
    out = text.replace("_", "")
    out = out.replace("ː", "")
    out = out.replace("tʃ", "ʧ").replace("dʒ", "ʤ")
    out = out.replace("ɝ", "ɜɹ").replace("ɚ", "əɹ")
    out = out.replace("iː", "i").replace("uː", "u").replace("oː", "o").replace("ɔː", "ɔ")
    out = out.replace("aɪ", "aɪ").replace("aʊ", "aʊ").replace("oʊ", "oʊ")
    return "".join(ch for ch in out if not ch.isspace())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tmp/reference_oov.jsonl"))
    args = parser.parse_args()
    with args.out.open("w", encoding="utf-8") as fh:
        for case in CASES:
            fh.write(json.dumps({"case": case, "ipa": espeak_ipa(case)}, ensure_ascii=False))
            fh.write("\n")


if __name__ == "__main__":
    main()
