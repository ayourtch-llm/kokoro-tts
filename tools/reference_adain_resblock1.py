#!/usr/bin/env python3
"""Reference for Generator AdaINResBlock1 (the 3-sub-block-with-Snake1D variant).

Uses decoder.generator.resblocks.0 by default: channels=256, kernel=3,
dilations=[1,3,5], style_dim=128. Loads weights from the converted safetensors
and folds .weight_g/.weight_v into a flat weight tensor before running. Mirrors
upstream kokoro/istftnet.py:34-77 verbatim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def fold_weight_norm(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """w = weight_v * (weight_g / ||weight_v||_per_dim0_slice)."""
    dims = tuple(range(1, weight_v.dim()))
    denom = weight_v.pow(2).sum(dim=dims, keepdim=True).sqrt()
    return weight_v * (weight_g / denom)


def load_conv1d(prefix: str, in_ch: int, out_ch: int, k: int, padding: int, dilation: int, st: dict) -> nn.Conv1d:
    conv = nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=dilation)
    folded = fold_weight_norm(st[f"{prefix}.weight_g"], st[f"{prefix}.weight_v"])
    with torch.no_grad():
        conv.weight.copy_(folded)
        conv.bias.copy_(st[f"{prefix}.bias"])
    return conv


class AdaIN1d(nn.Module):
    def __init__(self, style_dim: int, num_features: int, fc_w: torch.Tensor, fc_b: torch.Tensor):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
        with torch.no_grad():
            self.fc.weight.copy_(fc_w)
            self.fc.bias.copy_(fc_b)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s).view(s.shape[0], -1, 1)
        gamma, beta = h.chunk(2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


def snake1d(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return x + (1 / alpha) * (torch.sin(alpha * x) ** 2)


class AdaINResBlock1(nn.Module):
    def __init__(self, prefix: str, channels: int, kernel_size: int, dilations: list[int], style_dim: int, st: dict):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.adain1 = nn.ModuleList()
        self.adain2 = nn.ModuleList()
        self.alpha1 = nn.ParameterList()
        self.alpha2 = nn.ParameterList()
        for j in range(3):
            d = dilations[j]
            pad1 = (kernel_size * d - d) // 2
            pad2 = (kernel_size - 1) // 2
            self.convs1.append(load_conv1d(f"{prefix}.convs1.{j}", channels, channels, kernel_size, pad1, d, st))
            self.convs2.append(load_conv1d(f"{prefix}.convs2.{j}", channels, channels, kernel_size, pad2, 1, st))
            self.adain1.append(AdaIN1d(style_dim, channels, st[f"{prefix}.adain1.{j}.fc.weight"], st[f"{prefix}.adain1.{j}.fc.bias"]))
            self.adain2.append(AdaIN1d(style_dim, channels, st[f"{prefix}.adain2.{j}.fc.weight"], st[f"{prefix}.adain2.{j}.fc.bias"]))
            self.alpha1.append(nn.Parameter(st[f"{prefix}.alpha1.{j}"].clone(), requires_grad=False))
            self.alpha2.append(nn.Parameter(st[f"{prefix}.alpha2.{j}"].clone(), requires_grad=False))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for j in range(3):
            xt = self.adain1[j](x, s)
            xt = snake1d(xt, self.alpha1[j])
            xt = self.convs1[j](xt)
            xt = self.adain2[j](xt, s)
            xt = snake1d(xt, self.alpha2[j])
            xt = self.convs2[j](xt)
            x = xt + x
        return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--prefix", default="decoder.generator.resblocks.0",
                        help="State-dict prefix for the AdaINResBlock1 instance to validate")
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--dilations", type=int, nargs=3, default=[1, 3, 5])
    parser.add_argument("--style-dim", type=int, default=128)
    parser.add_argument("--length", type=int, default=37)
    parser.add_argument("--input-out", default="tmp/reference_adain_resblock1_input.bin")
    parser.add_argument("--style-out", default="tmp/reference_adain_resblock1_style.bin")
    parser.add_argument("--out", default="tmp/reference_adain_resblock1.bin")
    args = parser.parse_args()

    # Load all relevant state-dict tensors
    st = {}
    with safe_open(args.model, framework="pt") as f:
        for key in f.keys():
            if key.startswith(args.prefix + "."):
                st[key] = f.get_tensor(key)

    block = AdaINResBlock1(args.prefix, args.channels, args.kernel, args.dilations, args.style_dim, st)
    block.eval()

    # Deterministic input + style
    torch.manual_seed(2718)
    t = torch.arange(args.channels * args.length, dtype=torch.float32).reshape(1, args.channels, args.length)
    x = 0.20 * torch.sin(t * 0.013) + 0.08 * torch.cos(t * 0.041)
    s = torch.randn(1, args.style_dim) * 0.5

    with torch.no_grad():
        out = block(x, s)

    write_f32_bin(Path(args.input_out), x.numpy())
    write_f32_bin(Path(args.style_out), s.numpy())
    write_f32_bin(Path(args.out), out.numpy())
    print(f"adain_resblock1 shape={tuple(out.shape)} min={out.min().item():.6f} max={out.max().item():.6f}")


if __name__ == "__main__":
    main()
