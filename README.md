# kokoro-tts

A native Rust + [candle](https://github.com/huggingface/candle) port of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M), a fast text-to-speech model. Self-contained: no Python, no subprocesses, no network at inference. Just a Rust binary and ~330 MB of model weights.

**Status:** Both milestones achieved.

- **M1 — Model port** ([spec](docs/specs/kokoro-rust-port.md)): full Kokoro architecture (PL-BERT → text encoder → prosody predictor → alignment → F0/N → ISTFTNet generator) ported, all 12 numerical validation stages green vs PyTorch reference (max abs deltas 1e-8 to 5e-3 depending on stage).
- **M2 — Native G2P** ([spec](docs/specs/g2p-rust-port.md)): text-to-IPA pipeline with misaki-gold lexicon (13k common words, bit-exact) + CMUdict fallback (134k words, ARPAbet→IPA) + text normalization (numbers, dates, abbreviations, acronyms, money, time, units) + homograph disambiguation + OOV letter-to-sound rules.

End-to-end: `text → IPA → 24 kHz mono WAV`, runs at ~1.1× realtime on M1 CPU.

## Quick start

Prerequisites:

- Rust toolchain (`cargo`)
- Python 3 with `torch` and `safetensors` (only used once, for weight conversion)

```sh
git clone <this repo> kokoro-tts
cd kokoro-tts

# 1. Download model weights (~330 MB total)
cargo run --release --bin download-model

# 2. Convert .pth → safetensors (one-time setup)
pip install torch safetensors
python3 scripts/convert_weights.py --input ./models --output ./models

# 3. Speak!
cargo run --release --bin speak -- --text "Hello, world!"
# → produces ./hello.wav (24 kHz mono, 16-bit PCM)
open hello.wav  # macOS; or play / aplay / etc.
```

### If `download-model` fails (TLS / network issues)

The Rust hf-hub crate has had macOS TLS handshake issues in some sandboxed environments. If `download-model` errors out, fetch the three files directly with `curl`:

```sh
mkdir -p models/voices
curl -sSL -o models/config.json     "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json"
curl -sSL -o models/kokoro-v1_0.pth "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.pth"
curl -sSL -o models/voices/af_heart.pt "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt"

# Then continue with step 2 (python scripts/convert_weights.py ...)
```

## Usage

The `speak` binary takes either text (run through the full G2P pipeline) or raw IPA phonemes (bypass G2P).

```sh
# Default text path: full G2P pipeline
cargo run --release --bin speak -- --text "She read the book at 3:45 PM on May 6th, 2026."

# Raw phonemes path (advanced; bypass G2P)
cargo run --release --bin speak -- --phonemes "həlˈO wˈɜɹld"

# Custom output filename and voice
cargo run --release --bin speak -- \
    --text "Custom output" \
    --voice models/voices/af_heart.safetensors \
    --out custom.wav

# Slow it down (or speed it up)
cargo run --release --bin speak -- --text "Slowly..." --speed 0.7
```

Available flags:

| Flag | Default | Notes |
|---|---|---|
| `--text "..."` | (none) | Input text. G2P normalizes numbers/dates/etc. and looks up phonemes. |
| `--phonemes "..."` | (none) | Raw IPA phoneme string. Bypasses G2P entirely. Mutually exclusive with `--text`. |
| `--model-dir PATH` | `models` | Where `model.safetensors` and `config.json` live. |
| `--voice PATH` | `models/voices/af_heart.safetensors` | The reference voice tensor. |
| `--out FILE` | `hello.wav` | Output WAV path. |
| `--speed F` | `1.0` | Playback speed multiplier (0.5–2.0 is sensible; outside that gets weird). |

## Voices

This repo ships with the `af_heart` American-female voice (provided by upstream Kokoro). To use a different voice:

1. Download another voice file (e.g. `voices/af_bella.pt`) from the [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices) repo to `models/voices/`.
2. Re-run `python3 scripts/convert_weights.py` to convert it.
3. Pass `--voice models/voices/af_bella.safetensors` to `speak`.

## Architecture

The model itself is a faithful port of Kokoro-82M's StyleTTS2-flavored architecture:

```
text → G2P → IPA phonemes
              ↓
              vocab map → input_ids
              ↓
              PL-BERT (custom ALBERT) ──────► d_en
              ↓                                ↓
              TextEncoder (CNN×3 + BiLSTM) ──► t_en
              ↓                                ↓
voice ref_s → ProsodyPredictor → durations → alignment matrix
                              ↘ F0/N curves
              ↓
              Decoder front-end (asr+f0+n fusion through 4 AdainResBlk1d)
              ↓
              ISTFTNet generator (NSF + multi-scale upsample + MRF + CustomSTFT inverse)
              ↓
              waveform [B, N_samples] @ 24 kHz
```

For the full architectural breakdown, validation methodology, and numerical receipts, see [docs/specs/kokoro-rust-port.md](docs/specs/kokoro-rust-port.md).

The G2P pipeline is described in [docs/specs/g2p-rust-port.md](docs/specs/g2p-rust-port.md).

## Performance

On an Apple M1 (CPU only, no Metal/CUDA acceleration):

- Model load: ~3.5 s (one-time cost)
- Inference: ~1.1× realtime (1.5 s of audio in ~1.4 s)
- WAV output: 24 kHz mono, 16-bit PCM, ~75 KB per second of audio

Metal and CUDA backends are theoretically supported via candle's feature flags but haven't been exercised; the validation pipeline uses CPU.

## Known limitations

- **Homograph disambiguation** uses a hand-written rule table over the 30 most common English homographs, with a previous-word POS heuristic. Validation in M2 uses a rule-mirror reference (since `nltk`/`spacy` aren't required at runtime); a real POS-tagger oracle for receipts is deferred to M3. Audio quality is unaffected — listening tests on the curated corpus all sound right.
- **OOV letter-to-sound rules** cover ~70 hand-written patterns and reach ~84% character agreement with espeak-ng on technical terms ("PyTorch", "Kubernetes", etc.). They sound right on common patterns but can mispronounce rare/foreign words. Espeak-ng has 3000+ rules; we have an order of magnitude fewer.
- **English only.** Non-English text is out of scope.
- **Single-speaker per call.** Multi-voice mixing isn't wired.
- **No streaming.** Synthesis is one-shot per `speak` invocation; latency is acceptable for short utterances but not interactive.
- **CPU-only validated.** Other devices may work but haven't been numerically verified.

## Validation

Every layer of the model port and every G2P stage has a Python reference (in `tools/`) and a Rust check binary (in `src/bin/*_check.rs`) that diffs against it. Receipts are stamped in the M1 spec (§9) and the M2 spec (§10).

You can re-run any of them, e.g.:

```sh
# Verify the lexicon path
python3 tools/reference_phonemize_lexicon.py --out tmp/reference_lexicon.tsv
cargo run --release --bin lexicon_check -- --ref tmp/reference_lexicon.tsv
# → OK (42 cases)

# Or the full end-to-end mixed corpus
python3 tools/end_to_end_roundtrip.py    # see harness for ASR command
```

## Acknowledgments

- [Kokoro-82M](https://github.com/hexgrad/kokoro) — the upstream model and weights, by hexgrad.
- [misaki](https://github.com/hexgrad/misaki) — Kokoro's official G2P, whose `us_gold.json` lexicon (Apache 2.0) we vendor as the primary lookup tier.
- [CMUdict](https://github.com/cmusphinx/cmudict) — the Carnegie Mellon Pronouncing Dictionary, public domain, used as the fallback lexicon.
- [candle](https://github.com/huggingface/candle) — the Rust ML framework this is built on.
- [nemotron-speech-rs](../nemotron-speech) — the sibling streaming ASR used for round-trip validation.
- The original [StyleTTS2](https://github.com/yl4579/StyleTTS2) by Yinghao Aaron Li et al, on which Kokoro's architecture is based.

## License

The code in this repo is MIT licensed. Vendored data files retain their original licenses (CMUdict: public domain; misaki gold lexicon: Apache 2.0).
