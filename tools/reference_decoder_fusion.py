#!/usr/bin/env python3
"""Reference for stage 9: Decoder pre-vocoder fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from reference_alignment import alignment_from_durations, read_i64_bin
from reference_custom_albert import DEFAULT_PHONEMES, phonemes_to_ids, st_load, write_f32_bin
from reference_f0_n import adain_resblk1d, fold_weight_norm
from reference_predict_duration import load_voice_style
from reference_text_encoder import text_encoder


def decoder_fusion(
    asr: torch.Tensor,
    f0_curve: torch.Tensor,
    n_curve: torch.Tensor,
    style: torch.Tensor,
    sd: dict[str, torch.Tensor],
) -> torch.Tensor:
    f0 = F.conv1d(f0_curve.unsqueeze(1), fold_weight_norm(sd, "decoder.F0_conv"), sd["decoder.F0_conv.bias"], stride=2, padding=1)
    n = F.conv1d(n_curve.unsqueeze(1), fold_weight_norm(sd, "decoder.N_conv"), sd["decoder.N_conv.bias"], stride=2, padding=1)
    x = torch.cat([asr, f0, n], dim=1)
    x = adain_resblk1d(x, style, sd, "decoder.encode", upsample=False, learned_sc=True)
    asr_res = F.conv1d(asr, fold_weight_norm(sd, "decoder.asr_res.0"), sd["decoder.asr_res.0.bias"])

    res = True
    for idx in range(4):
        if res:
            x = torch.cat([x, asr_res, f0, n], dim=1)
        x = adain_resblk1d(
            x,
            style,
            sd,
            f"decoder.decode.{idx}",
            upsample=(idx == 3),
            learned_sc=True,
        )
        if idx == 3:
            res = False
    return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--voice", default="models/voices/af_heart.safetensors")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--durations", default="tmp/reference_predict_duration_i64.bin")
    parser.add_argument("--f0", default="tmp/reference_f0.bin")
    parser.add_argument("--n", default="tmp/reference_n.bin")
    parser.add_argument("--phonemes", default=DEFAULT_PHONEMES)
    parser.add_argument("--style-index", type=int)
    parser.add_argument("--asr-out", default="tmp/reference_decoder_asr.bin")
    parser.add_argument("--out", default="tmp/reference_decoder_fusion.bin")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)
    ids = [0] + phonemes_to_ids(config, args.phonemes) + [0]
    input_ids = torch.tensor([ids], dtype=torch.long)
    style_index = args.style_index if args.style_index is not None else max(len(args.phonemes) - 1, 0)

    sd = st_load(args.model)
    t_en = text_encoder(input_ids, sd)
    alignment = torch.from_numpy(alignment_from_durations(read_i64_bin(Path(args.durations))))
    asr = torch.matmul(t_en, alignment)

    f0 = torch.from_numpy(read_f32_bin(Path(args.f0)))
    n = torch.from_numpy(read_f32_bin(Path(args.n)))
    style = load_voice_style(args.voice, style_index)[:, :128]
    out = decoder_fusion(asr, f0, n, style, sd)

    write_f32_bin(Path(args.asr_out), asr)
    write_f32_bin(Path(args.out), out)


def read_f32_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    ndim = int(np.frombuffer(data[:4], dtype=np.uint32)[0])
    offset = 4
    shape = []
    for _ in range(ndim):
        shape.append(int(np.frombuffer(data[offset : offset + 4], dtype=np.uint32)[0]))
        offset += 4
    arr = np.frombuffer(data[offset:], dtype=np.float32).copy()
    return arr.reshape(shape)


if __name__ == "__main__":
    main()
