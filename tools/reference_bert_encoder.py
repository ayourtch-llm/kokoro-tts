#!/usr/bin/env python3
"""Reference for stage 4: CustomAlbert followed by Kokoro bert_encoder linear."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch.nn.functional as F

from reference_custom_albert import (
    DEFAULT_PHONEMES,
    custom_albert,
    phonemes_to_ids,
    st_load,
    write_f32_bin,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--phonemes", default=DEFAULT_PHONEMES)
    parser.add_argument("--out", default="tmp/reference_bert_encoder.bin")
    parser.add_argument("--bert-out", default="tmp/reference_bert_dur.bin")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    import torch

    ids = [0] + phonemes_to_ids(config, args.phonemes) + [0]
    input_ids = torch.tensor([ids], dtype=torch.long)
    sd = st_load(args.model)
    bert_dur = custom_albert(input_ids, sd, config["plbert"])
    out = F.linear(bert_dur, sd["bert_encoder.weight"], sd["bert_encoder.bias"])
    write_f32_bin(Path(args.bert_out), bert_dur)
    write_f32_bin(Path(args.out), out)


if __name__ == "__main__":
    main()
