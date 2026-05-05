"""
Convert Kokoro .pth weights to safetensors for Candle.

Usage:
    pip install torch safetensors
    python convert_weights.py [--input ./models] [--output ./models]
"""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


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
    parser.add_argument("--input", default="./models", help="Directory containing local .pth/.pt files")
    parser.add_argument("--output", default="./models", help="Output directory")
    parser.add_argument("--voices", nargs="*", default=["af_heart.pt"], help="Voice .pt files under input/voices")
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert main model from local file.
    pth_path = input_dir / "kokoro-v1_0.pth"
    if not pth_path.exists():
        raise FileNotFoundError(f"Missing local model file: {pth_path}")
    safetensors_path = out_dir / "model.safetensors"
    if not safetensors_path.exists():
        convert_pth_to_safetensors(str(pth_path), str(safetensors_path))
    else:
        print(f"Skipping {safetensors_path} (already exists)")

    # Copy config.json from local file if needed.
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing local config file: {config_path}")
    config_out = out_dir / "config.json"
    if config_path.resolve() != config_out.resolve():
        import shutil
        shutil.copy2(config_path, str(config_out))
        print(f"Copied config to {config_out}")
    else:
        print(f"Skipping {config_out} (already exists)")

    # Download voice files
    voices_dir = out_dir / "voices"
    voices_dir.mkdir(exist_ok=True)
    voice_files = args.voices

    print(f"\nConverting {len(voice_files)} voice files...")
    for vf in voice_files:
        pt_path = Path(vf)
        if not pt_path.is_absolute():
            pt_path = input_dir / ("voices" if pt_path.parent == Path(".") else "") / pt_path
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing local voice file: {pt_path}")
        name = pt_path.stem
        sf_path = voices_dir / f"{name}.safetensors"
        if not sf_path.exists():
            convert_voice_pt(pt_path, str(sf_path))
        else:
            print(f"  Skipping {sf_path}")

    print(f"\nAll files saved to {out_dir}/")
    print("Ready for Candle inference.")


if __name__ == "__main__":
    main()
