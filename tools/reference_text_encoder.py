#!/usr/bin/env python3
"""Reference for stage 5: TextEncoder embed -> CNNx3 -> BiLSTM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from reference_custom_albert import DEFAULT_PHONEMES, phonemes_to_ids, write_f32_bin, write_i64_bin


def st_load(path: str) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            out[key] = f.get_tensor(key)
    return out


def fold_weight_norm(sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    if f"{prefix}.weight" in sd:
        return sd[f"{prefix}.weight"]
    g = sd[f"{prefix}.weight_g"]
    v = sd[f"{prefix}.weight_v"]
    denom = torch.sqrt(torch.sum(v.square(), dim=(1, 2), keepdim=True))
    return v / denom * g


def channel_layer_norm(x: torch.Tensor, sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (x.shape[-1],), sd[f"{prefix}.gamma"], sd[f"{prefix}.beta"], eps=1e-5)
    return x.transpose(1, -1)


def text_encoder(input_ids: torch.Tensor, sd: dict[str, torch.Tensor]) -> torch.Tensor:
    x = F.embedding(input_ids, sd["text_encoder.embedding.weight"])
    x = x.transpose(1, 2)

    for idx in range(3):
        conv_prefix = f"text_encoder.cnn.{idx}.0"
        norm_prefix = f"text_encoder.cnn.{idx}.1"
        w = fold_weight_norm(sd, conv_prefix)
        b = sd[f"{conv_prefix}.bias"]
        x = F.conv1d(x, w, b, padding=2)
        x = channel_layer_norm(x, sd, norm_prefix)
        x = F.leaky_relu(x, negative_slope=0.2)

    x = x.transpose(1, 2)
    lstm = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True)
    lstm.load_state_dict(
        {
            "weight_ih_l0": sd["text_encoder.lstm.weight_ih_l0"],
            "weight_hh_l0": sd["text_encoder.lstm.weight_hh_l0"],
            "bias_ih_l0": sd["text_encoder.lstm.bias_ih_l0"],
            "bias_hh_l0": sd["text_encoder.lstm.bias_hh_l0"],
            "weight_ih_l0_reverse": sd["text_encoder.lstm.weight_ih_l0_reverse"],
            "weight_hh_l0_reverse": sd["text_encoder.lstm.weight_hh_l0_reverse"],
            "bias_ih_l0_reverse": sd["text_encoder.lstm.bias_ih_l0_reverse"],
            "bias_hh_l0_reverse": sd["text_encoder.lstm.bias_hh_l0_reverse"],
        }
    )
    lstm.eval()
    x, _ = lstm(x)
    return x.transpose(1, 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--phonemes", default=DEFAULT_PHONEMES)
    parser.add_argument("--out", default="tmp/reference_text_encoder.bin")
    parser.add_argument("--input-out", default="tmp/reference_text_encoder_input_ids.bin")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    ids = [0] + phonemes_to_ids(config, args.phonemes) + [0]
    input_ids = torch.tensor([ids], dtype=torch.long)
    sd = st_load(args.model)
    out = text_encoder(input_ids, sd)

    write_i64_bin(Path(args.input_out), input_ids)
    write_f32_bin(Path(args.out), out)


if __name__ == "__main__":
    main()
