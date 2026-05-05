#!/usr/bin/env python3
"""Reference for stage 1: select a Kokoro voice style row."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open


def write_bin(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.ascontiguousarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        for dim in arr.shape:
            f.write(np.uint32(dim).tobytes())
        f.write(arr.tobytes())
    print(f"wrote {path}: shape={tuple(arr.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="models/voices/af_heart.safetensors")
    parser.add_argument("--phoneme-count", type=int, default=11)
    parser.add_argument("--out", default="tmp/reference_voice.bin")
    args = parser.parse_args()

    with safe_open(args.voice, framework="np") as f:
        ref_s = f.get_tensor("ref_s")

    idx = min(max(args.phoneme_count, 0), ref_s.shape[0] - 1)
    style = ref_s[idx]
    write_bin(Path(args.out), style)


if __name__ == "__main__":
    main()
