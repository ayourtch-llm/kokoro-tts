#!/usr/bin/env python3
"""Run a small end-to-end G2P -> WAV -> ASR corpus sweep.

The ASR command is external and supplied by the caller. It must accept a WAV
path as its final positional argument and print the transcript on stdout.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def word_tokens(text: str) -> list[str]:
    return [token for token in text.lower().split() if token]


def edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        cur = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = word_tokens(reference)
    hyp = word_tokens(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="tools/end_to_end_corpus.txt")
    parser.add_argument(
        "--asr-cmd",
        required=True,
        help="Shell-style command template. Use {wav} for the synthesized file path.",
    )
    parser.add_argument(
        "--speak-cmd",
        default="cargo run --quiet --release --bin speak --",
        help="Command prefix used to synthesize WAVs.",
    )
    args = parser.parse_args()

    corpus = [line.strip() for line in Path(args.corpus).read_text().splitlines() if line.strip()]
    with tempfile.TemporaryDirectory(prefix="kokoro-g2p-stage6-") as tmp:
        tmpdir = Path(tmp)
        total_wer = 0.0
        for idx, sentence in enumerate(corpus, start=1):
            wav = tmpdir / f"{idx:03d}.wav"
            speak_cmd = shlex.split(args.speak_cmd) + ["--text", sentence, "--out", str(wav)]
            subprocess.run(speak_cmd, check=True)

            asr_cmd = shlex.split(args.asr_cmd.format(wav=str(wav)))
            result = subprocess.run(asr_cmd, check=True, capture_output=True, text=True)
            transcript = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
            sample_wer = wer(sentence, transcript)
            total_wer += sample_wer
            print(f"{idx:03d} wer={sample_wer:.3f} ref={sentence!r} hyp={transcript!r}")

        avg_wer = total_wer / len(corpus) if corpus else 0.0
        agreement = 1.0 - avg_wer
        print(f"sentences={len(corpus)} avg_wer={avg_wer:.3f} word_agreement={agreement:.3f}")
        return 0 if agreement >= 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
