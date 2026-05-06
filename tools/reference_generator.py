#!/usr/bin/env python3
"""Reference for the iSTFTNet Generator (decoder.generator.*) from upstream
kokoro/istftnet.py:257-326. Loads weights from the converted safetensors,
folds .weight_g/.weight_v inline, runs the full forward with deterministic
controls, and dumps inputs + output."""

from __future__ import annotations

import argparse
import math
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


def fold_wn(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, weight_v.dim()))
    denom = weight_v.pow(2).sum(dim=dims, keepdim=True).sqrt()
    return weight_v * (weight_g / denom)


def load_conv1d(prefix, in_ch, out_ch, k, padding=0, dilation=1, stride=1, st=None):
    conv = nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=dilation, stride=stride)
    if f"{prefix}.weight_g" in st:
        with torch.no_grad():
            conv.weight.copy_(fold_wn(st[f"{prefix}.weight_g"], st[f"{prefix}.weight_v"]))
    else:
        with torch.no_grad():
            conv.weight.copy_(st[f"{prefix}.weight"])
    with torch.no_grad():
        conv.bias.copy_(st[f"{prefix}.bias"])
    return conv


def load_conv_transpose1d(prefix, in_ch, out_ch, k, stride, padding, st):
    conv = nn.ConvTranspose1d(in_ch, out_ch, k, stride=stride, padding=padding)
    with torch.no_grad():
        conv.weight.copy_(fold_wn(st[f"{prefix}.weight_g"], st[f"{prefix}.weight_v"]))
        conv.bias.copy_(st[f"{prefix}.bias"])
    return conv


# ===== AdaIN / Snake1D / AdaINResBlock1 =====

def snake1d(x, alpha):
    return x + (1 / alpha) * (torch.sin(alpha * x) ** 2)


class AdaIN1d(nn.Module):
    def __init__(self, style_dim, num_features, fc_w, fc_b):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)
        with torch.no_grad():
            self.fc.weight.copy_(fc_w)
            self.fc.bias.copy_(fc_b)

    def forward(self, x, s):
        h = self.fc(s).view(s.shape[0], -1, 1)
        gamma, beta = h.chunk(2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdaINResBlock1(nn.Module):
    def __init__(self, prefix, channels, kernel_size, dilations, style_dim, st):
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
            self.convs1.append(load_conv1d(f"{prefix}.convs1.{j}", channels, channels, kernel_size, padding=pad1, dilation=d, st=st))
            self.convs2.append(load_conv1d(f"{prefix}.convs2.{j}", channels, channels, kernel_size, padding=pad2, dilation=1, st=st))
            self.adain1.append(AdaIN1d(style_dim, channels, st[f"{prefix}.adain1.{j}.fc.weight"], st[f"{prefix}.adain1.{j}.fc.bias"]))
            self.adain2.append(AdaIN1d(style_dim, channels, st[f"{prefix}.adain2.{j}.fc.weight"], st[f"{prefix}.adain2.{j}.fc.bias"]))
            self.alpha1.append(nn.Parameter(st[f"{prefix}.alpha1.{j}"].clone(), requires_grad=False))
            self.alpha2.append(nn.Parameter(st[f"{prefix}.alpha2.{j}"].clone(), requires_grad=False))

    def forward(self, x, s):
        for j in range(3):
            xt = self.adain1[j](x, s)
            xt = snake1d(xt, self.alpha1[j])
            xt = self.convs1[j](xt)
            xt = self.adain2[j](xt, s)
            xt = snake1d(xt, self.alpha2[j])
            xt = self.convs2[j](xt)
            x = xt + x
        return x


# ===== SineGen / SourceModuleHnNSF =====

class SineGen(nn.Module):
    def __init__(self, samp_rate=24000, upsample_scale=300, harmonic_num=8, sine_amp=0.1, noise_std=0.003, voiced_threshold=10):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.upsample_scale = upsample_scale

    def _f02uv(self, f0):
        return (f0 > self.voiced_threshold).type(torch.float32)

    def _f02sine(self, f0_values, rand_ini):
        rad_values = (f0_values / self.sampling_rate) % 1
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale, mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * math.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale, mode="linear").transpose(1, 2)
        return torch.sin(phase)

    def forward(self, f0, rand_ini, standard_noise):
        harmonics = torch.arange(1, self.harmonic_num + 2, dtype=f0.dtype, device=f0.device)
        fn = torch.multiply(f0, harmonics.view(1, 1, -1))
        sine_waves = self._f02sine(fn, rand_ini) * self.sine_amp
        uv = self._f02uv(f0)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        noise = noise_amp * standard_noise
        sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(nn.Module):
    def __init__(self, w, b):
        super().__init__()
        self.l_sin_gen = SineGen()
        self.l_linear = nn.Linear(9, 1)
        with torch.no_grad():
            self.l_linear.weight.copy_(w)
            self.l_linear.bias.copy_(b)

    def forward(self, f0, rand_ini, standard_noise):
        with torch.no_grad():
            sine_wavs, uv, noise = self.l_sin_gen(f0, rand_ini, standard_noise)
            sine_merge = torch.tanh(self.l_linear(sine_wavs))
            return sine_merge, noise, uv


# ===== TorchSTFT mirror of CustomSTFT (Conv1d-based, no complex) =====
# We use upstream's CustomSTFT semantics by computing forward/inverse via
# Conv1d/ConvTranspose1d to match the Rust port bit-equivalent. Reuses the
# pattern from tools/reference_custom_stft.py.

class CustomSTFT(nn.Module):
    def __init__(self, n_fft=20, hop_length=5, center=True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        freq_bins = n_fft // 2 + 1
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
        n = np.arange(n_fft)
        k = np.arange(freq_bins)
        ang = 2 * np.pi * np.outer(k, n) / n_fft  # (freq_bins, n_fft)
        fr = np.cos(ang) * window.numpy()
        fi = -np.sin(ang) * window.numpy()
        inv = 1.0 / n_fft
        br = np.cos(ang) * window.numpy() * inv
        bi = np.sin(ang) * window.numpy() * inv
        self.register_buffer("wfr", torch.from_numpy(fr).float().unsqueeze(1))
        self.register_buffer("wfi", torch.from_numpy(fi).float().unsqueeze(1))
        self.register_buffer("wbr", torch.from_numpy(br).float().unsqueeze(1))
        self.register_buffer("wbi", torch.from_numpy(bi).float().unsqueeze(1))

    def transform(self, waveform):
        if self.center:
            pad = self.n_fft // 2
            waveform = F.pad(waveform, (pad, pad), mode="replicate")
        x = waveform.unsqueeze(1)
        real = F.conv1d(x, self.wfr, stride=self.hop_length)
        imag = F.conv1d(x, self.wfi, stride=self.hop_length)
        # Zero imag at DC (bin 0) and Nyquist (bin n_fft/2) — mathematically
        # zero for real input; otherwise float-precision sign-of-zero diverges
        # between torch's BLAS conv path and candle's path, and atan2 amplifies
        # the divergence to ±π.
        imag = imag.clone()
        imag[:, 0, :] = 0
        imag[:, self.n_fft // 2, :] = 0
        magnitude = (real**2 + imag**2 + 1e-14).sqrt()
        phase = torch.atan2(imag, real)
        return magnitude, phase

    def inverse(self, magnitude, phase, length=None):
        real_part = magnitude * phase.cos()
        imag_part = magnitude * phase.sin()
        rr = F.conv_transpose1d(real_part, self.wbr, stride=self.hop_length)
        ii = F.conv_transpose1d(imag_part, self.wbi, stride=self.hop_length)
        wf = rr - ii
        if self.center:
            pad = self.n_fft // 2
            wf = wf[..., pad : wf.shape[-1] - pad]
        return wf.squeeze(1)


# ===== Generator =====

class Generator(nn.Module):
    def __init__(self, st, *, style_dim=128, upsample_initial_channel=512,
                 upsample_rates=(10, 6), upsample_kernel_sizes=(20, 12),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
                 n_fft=20, hop_size=5):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.upsample_rates = upsample_rates
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.f0_upsample_factor = math.prod(upsample_rates) * hop_size

        self.m_source = SourceModuleHnNSF(
            st["decoder.generator.m_source.l_linear.weight"],
            st["decoder.generator.m_source.l_linear.bias"],
        )
        self.stft = CustomSTFT(n_fft=n_fft, hop_length=hop_size)

        self.ups = nn.ModuleList()
        for i in range(self.num_upsamples):
            in_ch = upsample_initial_channel // (2 ** i)
            out_ch = upsample_initial_channel // (2 ** (i + 1))
            k = upsample_kernel_sizes[i]
            stride = upsample_rates[i]
            padding = (k - stride) // 2
            self.ups.append(load_conv_transpose1d(
                f"decoder.generator.ups.{i}", in_ch, out_ch, k, stride, padding, st))

        self.noise_convs = nn.ModuleList()
        self.noise_res = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j in range(self.num_kernels):
                self.resblocks.append(AdaINResBlock1(
                    f"decoder.generator.resblocks.{i*self.num_kernels+j}",
                    ch, resblock_kernel_sizes[j], resblock_dilation_sizes[j], style_dim, st))
            if i + 1 < self.num_upsamples:
                stride_f0 = math.prod(upsample_rates[i + 1:])
                self.noise_convs.append(load_conv1d(
                    f"decoder.generator.noise_convs.{i}", n_fft + 2, ch,
                    stride_f0 * 2, padding=(stride_f0 + 1) // 2, stride=stride_f0, st=st))
                self.noise_res.append(AdaINResBlock1(
                    f"decoder.generator.noise_res.{i}", ch, 7, [1, 3, 5], style_dim, st))
            else:
                self.noise_convs.append(load_conv1d(
                    f"decoder.generator.noise_convs.{i}", n_fft + 2, ch, 1, st=st))
                self.noise_res.append(AdaINResBlock1(
                    f"decoder.generator.noise_res.{i}", ch, 11, [1, 3, 5], style_dim, st))

        final_ch = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = load_conv1d(
            "decoder.generator.conv_post", final_ch, n_fft + 2, 7, padding=3, st=st)

        self.reflection_pad = nn.ReflectionPad1d((1, 0))

    def forward(self, x, s, f0, rand_ini, standard_noise):
        # f0: [B, T_f0]
        f0_up = F.interpolate(f0.unsqueeze(1), scale_factor=self.f0_upsample_factor, mode="nearest").transpose(1, 2)
        with torch.no_grad():
            har_source, _, _ = self.m_source(f0_up, rand_ini, standard_noise)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, negative_slope=0.1)
            x_source = self.noise_convs[i](har)
            x_source = self.noise_res[i](x_source, s)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            x = x + x_source
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x, s)
                else:
                    xs = xs + self.resblocks[i * self.num_kernels + j](x, s)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        n_freq = self.n_fft // 2 + 1
        spec = torch.exp(x[:, :n_freq, :])
        phase = torch.sin(x[:, n_freq:, :])
        return self.stft.inverse(spec, phase)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--t-dec", type=int, default=8, help="Decoder-side time dim (small for speed)")
    parser.add_argument("--x-out", default="tmp/reference_generator_x.bin")
    parser.add_argument("--s-out", default="tmp/reference_generator_s.bin")
    parser.add_argument("--f0-out", default="tmp/reference_generator_f0.bin")
    parser.add_argument("--rand-out", default="tmp/reference_generator_rand_ini.bin")
    parser.add_argument("--noise-out", default="tmp/reference_generator_noise.bin")
    parser.add_argument("--out", default="tmp/reference_generator.bin")
    args = parser.parse_args()

    # Load all decoder.generator.* keys
    st = {}
    with safe_open(args.model, framework="pt") as f:
        for key in f.keys():
            if key.startswith("decoder.generator."):
                st[key] = f.get_tensor(key)

    gen = Generator(st)
    gen.eval()

    torch.manual_seed(31415)
    t_dec = args.t_dec  # decoder T after the front-end's terminal upsample → use 2*T_dec_pre_upsample if mirroring full path
    # x is post-decode-block features: [B, 512, t_dec]
    x = torch.randn(1, 512, t_dec) * 0.5
    s = torch.randn(1, 128) * 0.5
    # F0_curve from predictor is [B, 2*T_dec_pre_upsample]; here we pass the
    # generator-side time dim directly. Use a smooth contour with a brief unvoiced
    # gap to exercise the UV path.
    t = torch.arange(t_dec, dtype=torch.float32)
    f0 = (130.0 + 30.0 * torch.sin(0.21 * t) + 20.0 * torch.cos(0.07 * t)).view(1, t_dec)
    f0[:, t_dec // 4 : t_dec // 4 + 1] = 0.0  # one unvoiced frame

    f0_audio_len = t_dec * gen.f0_upsample_factor
    rand_ini = torch.rand(1, 9)
    rand_ini[:, 0] = 0.0
    standard_noise = torch.randn(1, f0_audio_len, 9)

    with torch.no_grad():
        out = gen(x, s, f0, rand_ini, standard_noise)

    write_f32_bin(Path(args.x_out), x.numpy())
    write_f32_bin(Path(args.s_out), s.numpy())
    write_f32_bin(Path(args.f0_out), f0.numpy())
    write_f32_bin(Path(args.rand_out), rand_ini.numpy())
    write_f32_bin(Path(args.noise_out), standard_noise.numpy())
    write_f32_bin(Path(args.out), out.numpy())
    print(f"generator out shape={tuple(out.shape)} min={out.min().item():.6f} max={out.max().item():.6f}")


if __name__ == "__main__":
    main()
