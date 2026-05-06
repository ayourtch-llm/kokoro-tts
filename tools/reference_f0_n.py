#!/usr/bin/env python3
"""Reference for stage 8: ProsodyPredictor.F0Ntrain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from reference_alignment import alignment_from_durations, read_i64_bin
from reference_custom_albert import DEFAULT_PHONEMES, custom_albert, phonemes_to_ids, st_load, write_f32_bin
from reference_predict_duration import duration_encoder, load_voice_style


def fold_weight_norm(sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    if f"{prefix}.weight" in sd:
        return sd[f"{prefix}.weight"]
    g = sd[f"{prefix}.weight_g"]
    v = sd[f"{prefix}.weight_v"]
    denom = torch.sqrt(torch.sum(v.square(), dim=(1, 2), keepdim=True))
    return v / denom * g


def linear(x: torch.Tensor, sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return F.linear(x, sd[f"{prefix}.weight"], sd[f"{prefix}.bias"])


def lstm_forward(x: torch.Tensor, sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    lstm = nn.LSTM(x.shape[-1], 256, 1, batch_first=True, bidirectional=True)
    lstm.load_state_dict(
        {
            "weight_ih_l0": sd[f"{prefix}.weight_ih_l0"],
            "weight_hh_l0": sd[f"{prefix}.weight_hh_l0"],
            "bias_ih_l0": sd[f"{prefix}.bias_ih_l0"],
            "bias_hh_l0": sd[f"{prefix}.bias_hh_l0"],
            "weight_ih_l0_reverse": sd[f"{prefix}.weight_ih_l0_reverse"],
            "weight_hh_l0_reverse": sd[f"{prefix}.weight_hh_l0_reverse"],
            "bias_ih_l0_reverse": sd[f"{prefix}.bias_ih_l0_reverse"],
            "bias_hh_l0_reverse": sd[f"{prefix}.bias_hh_l0_reverse"],
        }
    )
    lstm.eval()
    out, _ = lstm(x)
    return out


def adain1d(x: torch.Tensor, style: torch.Tensor, sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    h = linear(style, sd, f"{prefix}.fc").unsqueeze(2)
    gamma, beta = h.chunk(2, dim=1)
    x = F.instance_norm(x, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, eps=1e-5)
    return (1.0 + gamma) * x + beta


def adain_resblk1d(
    x: torch.Tensor,
    style: torch.Tensor,
    sd: dict[str, torch.Tensor],
    prefix: str,
    upsample: bool,
    learned_sc: bool,
) -> torch.Tensor:
    residual = adain1d(x, style, sd, f"{prefix}.norm1")
    residual = F.leaky_relu(residual, negative_slope=0.2)
    if upsample:
        residual = F.conv_transpose1d(
            residual,
            fold_weight_norm(sd, f"{prefix}.pool"),
            sd[f"{prefix}.pool.bias"],
            stride=2,
            padding=1,
            output_padding=1,
            groups=residual.shape[1],
        )
    residual = F.conv1d(residual, fold_weight_norm(sd, f"{prefix}.conv1"), sd[f"{prefix}.conv1.bias"], padding=1)
    residual = adain1d(residual, style, sd, f"{prefix}.norm2")
    residual = F.leaky_relu(residual, negative_slope=0.2)
    residual = F.conv1d(residual, fold_weight_norm(sd, f"{prefix}.conv2"), sd[f"{prefix}.conv2.bias"], padding=1)

    shortcut = x
    if upsample:
        shortcut = F.interpolate(shortcut, scale_factor=2, mode="nearest")
    if learned_sc:
        shortcut = F.conv1d(shortcut, fold_weight_norm(sd, f"{prefix}.conv1x1"), bias=None)

    return (residual + shortcut) / np.sqrt(2.0)


def f0_n_train(en: torch.Tensor, style: torch.Tensor, sd: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    x = en.transpose(1, 2)
    s = style.unsqueeze(1).expand(x.shape[0], x.shape[1], style.shape[1])
    x = lstm_forward(torch.cat([x, s], dim=2), sd, "predictor.shared")

    f0 = x.transpose(1, 2)
    f0 = adain_resblk1d(f0, style, sd, "predictor.F0.0", upsample=False, learned_sc=False)
    f0 = adain_resblk1d(f0, style, sd, "predictor.F0.1", upsample=True, learned_sc=True)
    f0 = adain_resblk1d(f0, style, sd, "predictor.F0.2", upsample=False, learned_sc=False)
    f0 = F.conv1d(f0, sd["predictor.F0_proj.weight"], sd["predictor.F0_proj.bias"]).squeeze(1)

    n = x.transpose(1, 2)
    n = adain_resblk1d(n, style, sd, "predictor.N.0", upsample=False, learned_sc=False)
    n = adain_resblk1d(n, style, sd, "predictor.N.1", upsample=True, learned_sc=True)
    n = adain_resblk1d(n, style, sd, "predictor.N.2", upsample=False, learned_sc=False)
    n = F.conv1d(n, sd["predictor.N_proj.weight"], sd["predictor.N_proj.bias"]).squeeze(1)
    return f0, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--voice", default="models/voices/af_heart.safetensors")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--durations", default="tmp/reference_predict_duration_i64.bin")
    parser.add_argument("--phonemes", default=DEFAULT_PHONEMES)
    parser.add_argument("--style-index", type=int)
    parser.add_argument("--en-out", default="tmp/reference_f0_n_en.bin")
    parser.add_argument("--f0-out", default="tmp/reference_f0.bin")
    parser.add_argument("--n-out", default="tmp/reference_n.bin")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    ids = [0] + phonemes_to_ids(config, args.phonemes) + [0]
    input_ids = torch.tensor([ids], dtype=torch.long)
    style_index = args.style_index if args.style_index is not None else max(len(args.phonemes) - 1, 0)

    sd = st_load(args.model)
    style = load_voice_style(args.voice, style_index)[:, 128:]
    bert_dur = custom_albert(input_ids, sd, config["plbert"])
    d_en = F.linear(bert_dur, sd["bert_encoder.weight"], sd["bert_encoder.bias"]).transpose(1, 2)
    d = duration_encoder(d_en, style, sd, int(config["n_layer"]))
    alignment = torch.from_numpy(alignment_from_durations(read_i64_bin(Path(args.durations))))
    en = torch.matmul(d, alignment)
    f0, n = f0_n_train(en, style, sd)

    write_f32_bin(Path(args.en_out), en)
    write_f32_bin(Path(args.f0_out), f0)
    write_f32_bin(Path(args.n_out), n)


if __name__ == "__main__":
    main()
