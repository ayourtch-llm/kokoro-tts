#!/usr/bin/env python3
"""Reference for stage 6: ProsodyPredictor.predict_duration logits and integer durations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from reference_custom_albert import (
    DEFAULT_PHONEMES,
    custom_albert,
    phonemes_to_ids,
    st_load,
    write_f32_bin,
)


def write_i64_bin(path: Path, t: torch.Tensor) -> None:
    arr = t.detach().cpu().long().contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        for dim in arr.shape:
            f.write(np.uint32(dim).tobytes())
        f.write(arr.tobytes())
    print(f"wrote {path}: shape={tuple(arr.shape)} values={arr.reshape(-1).tolist()}")


def load_voice_style(path: str, style_index: int) -> torch.Tensor:
    with safe_open(path, framework="pt") as f:
        ref_s = f.get_tensor("ref_s")
    idx = min(max(style_index, 0), ref_s.shape[0] - 1)
    return ref_s[idx]


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


def layer_norm_last_dim(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    hidden = x.shape[-1]
    centered = x - (x.sum(dim=-1, keepdim=True) / hidden)
    variance = centered.square().sum(dim=-1, keepdim=True) / hidden
    return centered / torch.sqrt(variance + eps)


def ada_layer_norm(x: torch.Tensor, style: torch.Tensor, sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    h = linear(style, sd, f"{prefix}.fc").unsqueeze(1)
    gamma, beta = h.chunk(2, dim=2)
    return layer_norm_last_dim(x) * (gamma + 1.0) + beta


def duration_encoder(
    d_en: torch.Tensor,
    style: torch.Tensor,
    sd: dict[str, torch.Tensor],
    n_layers: int,
) -> torch.Tensor:
    x = d_en.transpose(1, 2)
    s = style.unsqueeze(1).expand(x.shape[0], x.shape[1], style.shape[1])

    for i in range(n_layers):
        x = torch.cat([x, s], dim=2)
        x = lstm_forward(x, sd, f"predictor.text_encoder.lstms.{i * 2}")
        x = ada_layer_norm(x, style, sd, f"predictor.text_encoder.lstms.{i * 2 + 1}")

    return x.transpose(1, 2)


def predict_duration(
    d_en: torch.Tensor,
    style: torch.Tensor,
    sd: dict[str, torch.Tensor],
    n_layers: int,
) -> torch.Tensor:
    d = duration_encoder(d_en, style, sd, n_layers)
    d = d.transpose(1, 2)
    s = style.unsqueeze(1).expand(d.shape[0], d.shape[1], style.shape[1])
    d = torch.cat([d, s], dim=2)
    x = lstm_forward(d, sd, "predictor.lstm")
    return linear(x, sd, "predictor.duration_proj.linear_layer")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--voice", default="models/voices/af_heart.safetensors")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--phonemes", default=DEFAULT_PHONEMES)
    parser.add_argument("--style-index", type=int)
    parser.add_argument("--out", default="tmp/reference_predict_duration.bin")
    parser.add_argument("--durations-out", default="tmp/reference_predict_duration_i64.bin")
    parser.add_argument("--d-en-out", default="tmp/reference_predict_duration_d_en.bin")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    ids = [0] + phonemes_to_ids(config, args.phonemes) + [0]
    input_ids = torch.tensor([ids], dtype=torch.long)
    style_index = args.style_index if args.style_index is not None else max(len(args.phonemes) - 1, 0)

    sd = st_load(args.model)
    bert_dur = custom_albert(input_ids, sd, config["plbert"])
    d_en = F.linear(bert_dur, sd["bert_encoder.weight"], sd["bert_encoder.bias"]).transpose(1, 2)

    # Upstream model.py uses ref_s[:, 128:] for predictor.text_encoder and F0Ntrain.
    style = load_voice_style(args.voice, style_index)[:, 128:]
    logits = predict_duration(d_en, style, sd, int(config["n_layer"]))
    durations = torch.round(torch.sigmoid(logits).sum(dim=-1)).clamp(min=1).long().squeeze(0)

    write_f32_bin(Path(args.d_en_out), d_en)
    write_f32_bin(Path(args.out), logits)
    write_i64_bin(Path(args.durations_out), durations)


if __name__ == "__main__":
    main()
