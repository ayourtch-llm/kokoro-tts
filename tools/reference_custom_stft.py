#!/usr/bin/env python3
"""Reference for CustomSTFT synthetic round-trip."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


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


class CustomSTFT(torch.nn.Module):
    def __init__(self, filter_length=20, hop_length=5, win_length=20, center=True, pad_mode="replicate"):
        super().__init__()
        self.n_fft = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center
        self.pad_mode = pad_mode
        self.freq_bins = self.n_fft // 2 + 1

        window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
        if win_length < filter_length:
            window = F.pad(window, (0, filter_length - win_length))
        elif win_length > filter_length:
            window = window[:filter_length]

        n = np.arange(self.n_fft)
        k = np.arange(self.freq_bins)
        angle = 2 * np.pi * np.outer(k, n) / self.n_fft
        forward_real = np.cos(angle) * window.numpy()
        forward_imag = -np.sin(angle) * window.numpy()
        self.weight_forward_real = torch.from_numpy(forward_real).float().unsqueeze(1)
        self.weight_forward_imag = torch.from_numpy(forward_imag).float().unsqueeze(1)

        inv_window = window.numpy() / self.n_fft
        backward_real = np.cos(angle) * inv_window
        backward_imag = np.sin(angle) * inv_window
        self.weight_backward_real = torch.from_numpy(backward_real).float().unsqueeze(1)
        self.weight_backward_imag = torch.from_numpy(backward_imag).float().unsqueeze(1)

    def transform(self, waveform: torch.Tensor):
        if self.center:
            pad = self.n_fft // 2
            waveform = F.pad(waveform, (pad, pad), mode=self.pad_mode)
        x = waveform.unsqueeze(1)
        real = F.conv1d(x, self.weight_forward_real, stride=self.hop_length)
        imag = F.conv1d(x, self.weight_forward_imag, stride=self.hop_length)
        magnitude = torch.sqrt(real**2 + imag**2 + 1e-14)
        phase = torch.atan2(imag, real)
        phase[(imag == 0) & (real < 0)] = torch.pi
        return magnitude, phase

    def inverse(self, magnitude: torch.Tensor, phase: torch.Tensor, length=None):
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        waveform = F.conv_transpose1d(real, self.weight_backward_real, stride=self.hop_length)
        waveform = waveform - F.conv_transpose1d(imag, self.weight_backward_imag, stride=self.hop_length)
        if self.center:
            pad = self.n_fft // 2
            waveform = waveform[..., pad:-pad]
        if length is not None:
            waveform = waveform[..., :length]
        return waveform

    def forward(self, x: torch.Tensor):
        mag, phase = self.transform(x)
        return self.inverse(mag, phase, length=x.shape[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-fft", type=int, default=20)
    parser.add_argument("--hop", type=int, default=5)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--input-out", default="tmp/reference_custom_stft_input.bin")
    parser.add_argument("--out", default="tmp/reference_custom_stft.bin")
    args = parser.parse_args()

    t = torch.arange(args.length, dtype=torch.float32)
    waveform = (
        0.10 * torch.sin(0.013 * t)
        + 0.05 * torch.sin(0.071 * t + 0.3)
        + 0.02 * torch.cos(0.191 * t)
    ).unsqueeze(0)
    out = CustomSTFT(args.n_fft, args.hop, args.n_fft)(waveform)

    write_f32_bin(Path(args.input_out), waveform.numpy())
    write_f32_bin(Path(args.out), out.squeeze(1).numpy())
    out = out.squeeze(1)
    common = min(waveform.shape[-1], out.shape[-1])
    diff = (waveform[..., :common] - out[..., :common]).abs()
    print(f"roundtrip_vs_input max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")


if __name__ == "__main__":
    main()
