#!/usr/bin/env python3
"""Reference for stage 3: Kokoro PL-BERT / CustomAlbert."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

DEFAULT_PHONEMES = "həlˈoʊ wˈɜɹld"


def st_load(path: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def write_f32_bin(path: Path, t: torch.Tensor) -> None:
    arr = t.detach().cpu().float().contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        for dim in arr.shape:
            f.write(np.uint32(dim).tobytes())
        f.write(arr.tobytes())
    print(f"wrote {path}: shape={tuple(arr.shape)}")


def write_i64_bin(path: Path, t: torch.Tensor) -> None:
    arr = t.detach().cpu().long().contiguous().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(np.uint32(arr.ndim).tobytes())
        for dim in arr.shape:
            f.write(np.uint32(dim).tobytes())
        f.write(arr.tobytes())
    print(f"wrote {path}: shape={tuple(arr.shape)} ids={arr.reshape(-1).tolist()}")


def layer_norm(x: torch.Tensor, sd: dict[str, torch.Tensor], prefix: str, eps: float) -> torch.Tensor:
    hidden = x.shape[-1]
    centered = x - (x.sum(dim=-1, keepdim=True) / hidden)
    variance = centered.square().sum(dim=-1, keepdim=True) / hidden
    x = centered / torch.sqrt(variance + eps)
    return x * sd[f"{prefix}.weight"] + sd[f"{prefix}.bias"]


def linear(x: torch.Tensor, sd: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
    return F.linear(x, sd[f"{prefix}.weight"], sd[f"{prefix}.bias"])


def embeddings(input_ids: torch.Tensor, sd: dict[str, torch.Tensor], eps: float) -> torch.Tensor:
    batch, seq_len = input_ids.shape
    device = input_ids.device
    token_type_ids = torch.zeros((batch, seq_len), dtype=torch.long, device=device)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)

    x = F.embedding(input_ids, sd["bert.embeddings.word_embeddings.weight"])
    x = x + F.embedding(token_type_ids, sd["bert.embeddings.token_type_embeddings.weight"])
    x = x + F.embedding(position_ids, sd["bert.embeddings.position_embeddings.weight"]).unsqueeze(0)
    return layer_norm(x, sd, "bert.embeddings.LayerNorm", eps)


def albert_layer(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    sd: dict[str, torch.Tensor],
    cfg: dict,
) -> torch.Tensor:
    prefix = "bert.encoder.albert_layer_groups.0.albert_layers.0"
    attn_prefix = f"{prefix}.attention"
    batch, seq_len, hidden_size = hidden.shape
    num_heads = int(cfg["num_attention_heads"])
    head_dim = hidden_size // num_heads

    def split_heads(x: torch.Tensor) -> torch.Tensor:
        return x.view(batch, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()

    query = split_heads(linear(hidden, sd, f"{attn_prefix}.query"))
    key = split_heads(linear(hidden, sd, f"{attn_prefix}.key"))
    value = split_heads(linear(hidden, sd, f"{attn_prefix}.value"))

    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(head_dim)
    scores = scores + attention_mask
    probs = torch.softmax(scores, dim=-1)

    context = torch.matmul(probs, value)
    context = context.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
    attention_output = linear(context, sd, f"{attn_prefix}.dense")
    attention_output = layer_norm(hidden + attention_output, sd, f"{attn_prefix}.LayerNorm", 1e-12)

    ffn_output = linear(attention_output, sd, f"{prefix}.ffn")
    ffn_output = 0.5 * ffn_output * (1.0 + torch.erf(ffn_output / math.sqrt(2.0)))
    ffn_output = linear(ffn_output, sd, f"{prefix}.ffn_output")
    return layer_norm(ffn_output + attention_output, sd, f"{prefix}.full_layer_layer_norm", 1e-12)


def custom_albert(input_ids: torch.Tensor, sd: dict[str, torch.Tensor], cfg: dict) -> torch.Tensor:
    hidden = embeddings(input_ids, sd, eps=1e-12)
    hidden = linear(hidden, sd, "bert.encoder.embedding_hidden_mapping_in")

    attention = torch.ones_like(input_ids, dtype=torch.float32)
    extended_attention = (1.0 - attention)[:, None, None, :] * torch.finfo(torch.float32).min

    for _ in range(int(cfg["num_hidden_layers"])):
        hidden = albert_layer(hidden, extended_attention, sd, cfg)
    return hidden


def phonemes_to_ids(config: dict, phonemes: str) -> list[int]:
    vocab = config["vocab"]
    return [int(vocab[p]) for p in phonemes if p in vocab]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.safetensors")
    parser.add_argument("--config", default="models/config.json")
    parser.add_argument("--phonemes", default=DEFAULT_PHONEMES)
    parser.add_argument("--out", default="tmp/reference_custom_albert.bin")
    parser.add_argument("--input-out", default="tmp/reference_custom_albert_input_ids.bin")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    ids = [0] + phonemes_to_ids(config, args.phonemes) + [0]
    input_ids = torch.tensor([ids], dtype=torch.long)
    sd = st_load(args.model)
    out = custom_albert(input_ids, sd, config["plbert"])

    write_i64_bin(Path(args.input_out), input_ids)
    write_f32_bin(Path(args.out), out)


if __name__ == "__main__":
    main()
