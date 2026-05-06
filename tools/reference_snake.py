#!/usr/bin/env python3
"""Reference for Generator Snake1D activation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--alpha-key", default="decoder.generator.resblocks.0.alpha1.0")
    parser.add_argument("--length", type=int, default=37)
    parser.add_argument("--input-out", default="tmp/reference_snake_input.bin")
    parser.add_argument("--out", default="tmp/reference_snake.bin")
    args = parser.parse_args()

    with safe_open(args.model, framework="pt") as f:
        alpha = f.get_tensor(args.alpha_key)

    channels = alpha.shape[1]
    t = torch.arange(channels * args.length, dtype=torch.float32).reshape(1, channels, args.length)
    x = 0.25 * torch.sin(t * 0.017) + 0.10 * torch.cos(t * 0.071)
    out = x + (1 / alpha) * (torch.sin(alpha * x) ** 2)

    write_f32_bin(Path(args.input_out), x.numpy())
    write_f32_bin(Path(args.out), out.numpy())
    print(f"snake shape={tuple(out.shape)} min={out.min().item():.6f} max={out.max().item():.6f}")


if __name__ == "__main__":
    main()
