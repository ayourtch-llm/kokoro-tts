"""
Convert Kokoro .pth weights to safetensors for Candle.

Usage:
    pip install torch safetensors huggingface_hub
    python convert_weights.py [--repo hexgrad/Kokoro-82M] [--output ./models]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download


def download_file(repo_id: str, filename: str, cache_dir: str | None = None) -> str:
    """Download a file from HuggingFace Hub."""
    return hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)


def convert_pth_to_safetensors(pth_path: str, output_path: str) -> None:
    """Convert PyTorch .pth to safetensors format."""
    print(f"Loading {pth_path}...")
    state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)

    # The .pth file contains nested dicts: {module_name: {param_name: tensor}}
    # We need to flatten to {module_name.param_name: tensor}
    flat = {}
    for module_name, params in state_dict.items():
        if isinstance(params, dict):
            for param_name, tensor in params.items():
                key = f"{module_name}.{param_name}"
                if isinstance(tensor, torch.Tensor):
                    flat[key] = tensor.cpu().contiguous()
                else:
                    print(f"  WARNING: skipping non-tensor {key}")
        else:
            flat[module_name] = params.cpu().contiguous()

    print(f"Saving {len(flat)} tensors to {output_path}...")
    save_file(flat, output_path)
    print("Done.")


def convert_voice_pt(pt_path: str, output_path: str) -> None:
    """Convert a voice .pt tensor to safetensors."""
    print(f"Loading voice {pt_path}...")
    tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
    save_file({"ref_s": tensor.cpu().contiguous()}, output_path)
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert Kokoro weights to safetensors")
    parser.add_argument("--repo", default="hexgrad/Kokoro-82M", help="HuggingFace repo ID")
    parser.add_argument("--output", default="./models", help="Output directory")
    parser.add_argument("--voices", nargs="*", default=None, help="Voice files to convert (None=all)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download and convert main model
    pth_path = download_file(args.repo, "kokoro-v1_0.pth")
    safetensors_path = out_dir / "model.safetensors"
    if not safetensors_path.exists():
        convert_pth_to_safetensors(pth_path, str(safetensors_path))
    else:
        print(f"Skipping {safetensors_path} (already exists)")

    # Download config.json
    config_path = download_file(args.repo, "config.json")
    config_out = out_dir / "config.json"
    if not config_out.exists():
        import shutil
        shutil.copy2(config_path, str(config_out))
        print(f"Copied config to {config_out}")
    else:
        print(f"Skipping {config_out} (already exists)")

    # Download voice files
    voices_dir = out_dir / "voices"
    voices_dir.mkdir(exist_ok=True)

    # List voice files from HF
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(args.repo, repo_type="model")
    voice_files = [f for f in all_files if f.startswith("voices/") and f.endswith(".pt")]

    if args.voices:
        voice_files = [f"voices/{v}" for v in args.voices if not v.startswith("voices/")]
        voice_files += [v for v in args.voices if v.startswith("voices/")]

    print(f"\nConverting {len(voice_files)} voice files...")
    for vf in voice_files:
        name = Path(vf).stem
        pt_path = download_file(args.repo, vf)
        sf_path = voices_dir / f"{name}.safetensors"
        if not sf_path.exists():
            convert_voice_pt(pt_path, str(sf_path))
        else:
            print(f"  Skipping {sf_path}")

    print(f"\nAll files saved to {out_dir}/")
    print("Ready for Candle inference.")


if __name__ == "__main__":
    main()
