#!/usr/bin/env python3
"""Reference for stage 2: map Kokoro IPA phonemes to vocab IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_PHONEMES = "həlˈoʊ wˈɜɹld"


def write_i64_bin(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.int64)
    arr = np.ascontiguousarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        for dim in arr.shape:
            f.write(np.uint32(dim).tobytes())
        f.write(arr.tobytes())
    print(f"wrote {path}: shape={tuple(arr.shape)} ids={arr.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--phonemes", default=DEFAULT_PHONEMES)
    parser.add_argument("--out", default="tmp/reference_phoneme_ids.bin")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    vocab = config["vocab"]

    ids: list[int] = []
    dropped: list[str] = []
    for phoneme in args.phonemes:
        if phoneme in vocab:
            ids.append(int(vocab[phoneme]))
        else:
            dropped.append(phoneme)

    if dropped:
        print(f"dropped unmapped phonemes: {dropped}")

    write_i64_bin(Path(args.out), np.asarray(ids, dtype=np.int64))


if __name__ == "__main__":
    main()
