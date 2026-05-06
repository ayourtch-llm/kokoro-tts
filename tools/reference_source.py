#!/usr/bin/env python3
"""Reference for SineGen + SourceModuleHnNSF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
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


class SineGen(torch.nn.Module):
    def __init__(
        self,
        samp_rate: int,
        upsample_scale: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _f02sine(self, f0_values: torch.Tensor, rand_ini: torch.Tensor) -> torch.Tensor:
        rad_values = (f0_values / self.sampling_rate) % 1
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = F.interpolate(
            rad_values.transpose(1, 2),
            scale_factor=1 / self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * np.pi
        phase = F.interpolate(
            phase.transpose(1, 2) * self.upsample_scale,
            scale_factor=self.upsample_scale,
            mode="linear",
        ).transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0: torch.Tensor, rand_ini: torch.Tensor, standard_noise: torch.Tensor):
        harmonics = torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device)
        fn = torch.multiply(f0, harmonics.view(1, 1, -1))
        sine_waves = self._f02sine(fn, rand_ini) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * standard_noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.l_sin_gen = SineGen(24000, 300, harmonic_num=8, voiced_threshold=10)
        self.l_linear = torch.nn.Linear(9, 1)
        with torch.no_grad():
            self.l_linear.weight.copy_(weight)
            self.l_linear.bias.copy_(bias)

    def forward(self, f0: torch.Tensor, rand_ini: torch.Tensor, standard_noise: torch.Tensor):
        with torch.no_grad():
            sine_wavs, uv, noise = self.l_sin_gen(f0, rand_ini, standard_noise)
            sine_merge = torch.tanh(self.l_linear(sine_wavs))
            return sine_merge, noise, uv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--length", type=int, default=900)
    parser.add_argument("--f0-out", default="tmp/reference_source_f0.bin")
    parser.add_argument("--rand-out", default="tmp/reference_source_rand_ini.bin")
    parser.add_argument("--noise-out", default="tmp/reference_source_noise.bin")
    parser.add_argument("--out", default="tmp/reference_source.bin")
    parser.add_argument("--uv-out", default="tmp/reference_source_uv.bin")
    args = parser.parse_args()

    torch.manual_seed(1234)
    t = torch.arange(args.length, dtype=torch.float32)
    f0 = (
        130.0
        + 40.0 * torch.sin(0.013 * t)
        + 25.0 * torch.sin(0.031 * t + 0.4)
    ).view(1, args.length, 1)
    f0[:, 120:170, :] = 0.0
    f0[:, 500:530, :] = 8.0

    rand_ini = torch.rand(1, 9)
    rand_ini[:, 0] = 0.0
    standard_noise = torch.randn(1, args.length, 9)

    with safe_open(args.model, framework="pt") as f:
        weight = f.get_tensor("decoder.generator.m_source.l_linear.weight")
        bias = f.get_tensor("decoder.generator.m_source.l_linear.bias")

    source = SourceModuleHnNSF(weight, bias)
    out, _, uv = source(f0, rand_ini, standard_noise)

    write_f32_bin(Path(args.f0_out), f0.numpy())
    write_f32_bin(Path(args.rand_out), rand_ini.numpy())
    write_f32_bin(Path(args.noise_out), standard_noise.numpy())
    write_f32_bin(Path(args.out), out.numpy())
    write_f32_bin(Path(args.uv_out), uv.numpy())
    print(
        f"source shape={tuple(out.shape)} min={out.min().item():.6f} max={out.max().item():.6f}"
    )


if __name__ == "__main__":
    main()
