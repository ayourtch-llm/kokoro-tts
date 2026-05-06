#!/usr/bin/env python3
"""Reference for stage 7: duration-to-alignment matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_i64_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    ndim = int(np.frombuffer(data[:4], dtype=np.uint32)[0])
    offset = 4
    shape = []
    for _ in range(ndim):
        shape.append(int(np.frombuffer(data[offset : offset + 4], dtype=np.uint32)[0]))
        offset += 4
    arr = np.frombuffer(data[offset:], dtype=np.int64).copy()
    return arr.reshape(shape)


def write_f32_bin(path: Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.ascontiguousarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        for dim in arr.shape:
            f.write(np.uint32(dim).tobytes())
        f.write(arr.tobytes())
    print(f"wrote {path}: shape={tuple(arr.shape)}")


def alignment_from_durations(durations: np.ndarray) -> np.ndarray:
    durations = np.asarray(durations, dtype=np.int64).reshape(-1)
    total = int(durations.sum())
    alignment = np.zeros((durations.shape[0], total), dtype=np.float32)
    cursor = 0
    for token_idx, duration in enumerate(durations):
        if duration < 0:
            raise ValueError(f"negative duration at index {token_idx}: {duration}")
        alignment[token_idx, cursor : cursor + int(duration)] = 1.0
        cursor += int(duration)
    return alignment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--durations", default="tmp/reference_predict_duration_i64.bin")
    parser.add_argument("--out", default="tmp/reference_alignment.bin")
    args = parser.parse_args()

    durations = read_i64_bin(Path(args.durations))
    alignment = alignment_from_durations(durations)
    write_f32_bin(Path(args.out), alignment)


if __name__ == "__main__":
    main()
