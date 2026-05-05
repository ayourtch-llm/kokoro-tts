# Kokoro-82M Native Rust Port — Implementation Brief

**Audience:** the implementing instance (codex / pty-10).
**Reviewers:** pty-1 (voice-agent integrator) and the Claude instance in this repo.
**Status at handoff:** scaffold exists, does not compile, no end-to-end path. Zero commits.

## 1. Goal

**Milestone 1 (everything else is downstream of this):** running the binary on a single English phrase produces a valid 24 kHz mono WAV file that, when played, sounds like Kokoro saying the phrase. Quality matching the reference Python implementation comes from the per-stage validation discipline in §7, not from listening tests.

Not in scope for milestone 1: the axum server, streaming, the voice-agent UDP integration, multi-voice mixing, non-English. Those land in later milestones once the offline path is proven correct.

**Hard constraint:** native Rust + candle. No ONNX runtime. No Python at inference time (Python is allowed and required for *validation*, see §7). Don't propose ONNX as a fallback if the port gets hard — the port getting hard is the work.

## 2. Architecture (Kokoro = StyleTTS2-flavored)

```
text
 └─► G2P ──► IPA phoneme string
              └─► vocab map ──► input_ids [B, T+2]   (BOS/EOS = id 0)
                                ├─► CustomAlbert (PL-BERT) ──► bert_dur [B, T, 768]
                                │                                └─► bert_encoder Linear ──► d_en [B, T, 512]
                                │                                                            └─► transpose ──► [B, 512, T]
                                │
                                └─► TextEncoder (embed → CNN×3 → BiLSTM) ──► t_en [B, 512, T]

reference voice tensor ref_s [1, 256]   (loaded per-utterance, indexed by phoneme count)
 ├─► s_pred = ref_s[:, 128:]    (passed to predictor.text_encoder AND predictor.F0Ntrain)
 └─► s_dec  = ref_s[:, :128]    (passed to decoder)
   (verified against hexgrad/kokoro/kokoro/model.py:104,115,118 — naming is by consumer, not "duration vs prosody")

(d_en, s_pred, lengths, text_mask) ──► predictor.text_encoder ──► d [B, 512, T]
                                                                  ├─► .lstm ──► x [B, T, 512] ──► duration_proj ──► [B, T, max_dur=50]
                                                                  │                                                  └─► sigmoid → sum(axis=-1) / speed → [B, T] continuous
                                                                  │                                                                                        └─► round, clamp(min=1), long → pred_dur [T] integer
                                                                  │                                                                                            └─► alignment matrix pred_aln_trg [T, sum(pred_dur)]   (one-hot scatter via repeat_interleave)
                                                                  │
                                                                  └─► transpose(-1,-2) @ pred_aln_trg ──► en [B, sum(pred_dur), 512]
                                                                                                          └─► (en, s_pred) ──► predictor.F0Ntrain ──► (F0_pred, N_pred) each [B, sum(pred_dur)]

input_ids ──► standalone TextEncoder (embed → CNN×3 → BiLSTM) ──► t_en [B, 512, T]
                                                                   └─► @ pred_aln_trg ──► asr [B, 512, sum(pred_dur)]

(asr, F0_pred, N_pred, s_dec) ──► Decoder ──► waveform [B, 1, sum(pred_dur)*300]   @ 24 kHz
```

The decoder's final upsampling factor is 300 (gen_istft_hop_size × upsample_rates product). 1 frame of duration ≈ 300 audio samples ≈ 12.5 ms.

## 3. What works (don't redo)

- `Cargo.toml` (`Cargo.toml`) — deps are reasonable (candle 0.8, candle-transformers, axum, hound, hf-hub). Add what you need; don't churn what's there.
- `src/main.rs` — axum server skeleton, queue, `/speak` `/stop` `/status` routes. The processing path is a stub (`main.rs:130-143`) but the harness around it is fine.
- `scripts/convert_weights.py` — flattens `kokoro-v1_0.pth` (nested `{module: {param: tensor}}`) into a flat safetensors. Also handles voice `.pt → .safetensors`. Looks correct; verify after first end-to-end run.
- `src/bin/download-model.rs` — fetches `hexgrad/Kokoro-82M` config + weights + `voices/af_heart.pt`. **Has a real bug**: `download-model.rs:72` calls `local_dir.create_dir_all()` — no such method on `Path`; replace with `std::fs::create_dir_all(local_dir)?`. Also re-check that `repo.get(filename)` is actually awaited / matches the hf-hub 0.3 API (it's used in an async context but called synchronously).
- `src/model/config.rs` — Kokoro `config.json` schema. Field shapes are fine; **two bugs**:
  - `config.rs:2` imports `use candle_transformers::albert::AlbertConfig;` — **this module does not exist in candle-transformers 0.8** (verified against 0.8.4 and 0.9.2; candle-transformers ships bert/distilbert/modernbert/jina_bert/debertav2/xlm_roberta only). The previous implementer hallucinated it. The local `to_albert_config()` method's return type must change to a struct defined alongside our own ALBERT impl in `src/model/bert.rs` (see §4 bert.rs subsection). Move or rename `to_albert_config` accordingly.
  - `config.rs:75` references `candle_core::safetensors::MergedTensors` which is not a real type. Either delete `load_model_weights` (it's unused; `mod.rs::load` builds a `VarBuilder` directly) or fix the return type to `HashMap<String, Tensor>`.

## 4. Concrete bug map (from a careful read)

The port author understood the architecture but the code has API drift against candle 0.8 and several logic bugs. None of these compile today.

### `src/model/bert.rs` — depends on a hallucinated module; needs from-scratch CustomAlbert

This was misread on the first audit pass as "a thin wrapper around `candle_transformers::albert::AlbertModel`." That module **does not exist** in candle-transformers (verified against 0.8.4 and 0.9.2: only bert/distilbert/modernbert/jina_bert/debertav2/xlm_roberta ship). The current `bert.rs` is a 19-line stub that imports a fake path. It does not work and cannot be made to work by fixing imports — the underlying ALBERT implementation has to be written from scratch in this repo.

**What to implement.** PL-BERT is BERT-shaped with two structural twists vs canonical BERT, both of which Kokoro's converted state dict expects:

1. **Factorized embedding.** Vocab IDs go through an `Embedding(vocab_size, embedding_size)` (small, e.g. 128) and then a `Linear(embedding_size, hidden_size)` projection up to the transformer hidden width. This contrasts with BERT, which embeds straight into `hidden_size`. The HuggingFace-equivalent state-dict keys are `embeddings.word_embeddings.weight`, `embeddings.position_embeddings.weight`, `embeddings.token_type_embeddings.weight`, `embeddings.LayerNorm.{weight,bias}`, then a separate `encoder.embedding_hidden_mapping_in.{weight,bias}` for the project-up. Kokoro's `config.plbert` has `hidden_size` but `embedding_size` may be implicit (often hidden_size for ALBERT-style — verify by dumping safetensors keys early).
2. **Cross-layer parameter sharing.** A single transformer block is reused `num_hidden_layers` times rather than instantiating `num_hidden_layers` distinct blocks (this is the entire point of ALBERT — parameter count stays small). State-dict-wise: keys live under `encoder.albert_layer_groups.0.albert_layers.0.<...>` (single group, single inner layer). Implementation-wise: load one block, call it `num_hidden_layers` times in the forward pass.

The transformer block itself is canonical: pre/post LayerNorm + multi-head self-attention + feed-forward (typically GELU). Token type embeddings are a no-op for Kokoro (`type_vocab_size = 1` per `config.rs:62`) but must still be wired since the converted weights include them.

**Reference implementations:**
- Candle's `candle-transformers/src/models/bert.rs` — the closest existing template; copy structure, then add factorized embedding + the layer-sharing twist.
- HuggingFace `transformers/src/transformers/models/albert/modeling_albert.py` — authoritative for the key layout and forward pass shape.
- The original ALBERT paper (Lan et al. 2019) for the factorization rationale; not needed for porting but useful context.

**Local types to define** (suggested layout): in a single new `bert.rs`, define a local `AlbertConfig` struct (replacing the hallucinated import), `CustomAlbert` model, and a `forward(input_ids, attention_mask) -> Tensor [B, T, hidden_size]` returning the last hidden state directly (matching what `mod.rs:126-127` expects). The existing `mod.rs` integration point (`bert.forward(...)` then `bert_encoder.forward(...)`) does not need to change.

**Validation note:** stage 3 in §7 ("CustomAlbert (PL-BERT)") gates this work — it must diff against the upstream `kokoro` Python package's `self.bert(input_ids, attention_mask=...)` output to ≤1e-5 max abs before moving on. Use `transformers.AlbertModel` loaded with the same converted weights as a *secondary* cross-check; if our impl matches both, the port is solid.

**Compile-pass workaround for §8 step 1:** to keep `cargo check` green before CustomAlbert is written, replace the `candle_transformers::albert` imports in `bert.rs` and `config.rs` with local placeholder types (an empty struct + a stub `forward` that returns an error or `unimplemented!()`). The compile pass should not block on the real ALBERT implementation — that's its own dedicated step (§8 step 3).

### `src/model/text_encoder.rs` — won't compile

- `text_encoder.rs:1-10` — imports use submodule paths that don't exist:
  - `candle_nn::lstm::{Lstm, LstmBlock}` → real path is `candle_nn::Lstm`
  - `candle_nn::embedding::Embedding` → `candle_nn::Embedding`
  - `candle_nn::linear::Linear` → `candle_nn::Linear`
  - `candle_nn::conv1d::Conv1d` → `candle_nn::Conv1d`
  - `candle_nn::layer_norm::LayerNorm` → `candle_nn::LayerNorm`
  - `candle_nn::var_builder::VarBuilder` → `candle_nn::VarBuilder`
  - `candle_nn::functional::dropout_none` → does not exist; use `candle_nn::ops::dropout` with p=0 or just drop the call (we're inference-only, dropout is a no-op).
- `text_encoder.rs:21-31` and `:61-72` call `conv1d(...)`, `embedding(...)`, `layer_norm(...)`, `lstm(...)`, `Conv1dConfig` without prefixes — needs `candle_nn::` prefix or appropriate `use`.
- LSTM hx/cx are sized `(2, 1, channels)` (`text_encoder.rs:91-94`) — should be `(1, batch, hidden)` per direction; for bidirectional you build forward + backward separately or use `candle_nn::lstm` with `direction: Bidirectional`. Verify against candle's actual bidirectional LSTM API.

### `src/model/predictor.rs` — won't compile, plus logic bugs

- `predictor.rs:1-7` — `candle_nn::seq::{Lstm, LstmBlock}` and `candle_nn::functional::{dropout_none, leaky_relu, log_softmax, sigmoid}` are wrong paths. Real homes: `candle_nn::Lstm`, `candle_nn::ops::{dropout, leaky_relu, log_softmax, sigmoid}` (verify against the 0.8 source — some live as free functions on `Tensor`).
- `predictor.rs:170` — `actv: leaky_relu` stored as `fn(&Tensor) -> Result<Tensor>`, but `leaky_relu` takes `(&Tensor, f64)`. Either curry, store `f64` separately, or just inline the call site (only one user).
- `predictor.rs:82-100` — `DurationEncoder::forward` does `let x = ...` then `x = ...` later (immutable rebinding). Use `let mut x` or rename.
- `predictor.rs:104-188` — this file has its own `AdainResBlk1d` distinct from the decoder's. **It has no `pool` field and ignores its `upsample` constructor argument.** That breaks the F0/N branch which needs upsampling at index 1. Either reuse the decoder's version (preferred — single source of truth) or make this one actually upsample.
- `predictor.rs:74` — `vb.pp(format!("lstms.{i}.1"))` — confirm the `.0`/`.1` indexing matches the converted state-dict key shape after `convert_weights.py` flattening. The Python original is `nn.ModuleList([nn.ModuleList([LSTM, AdaLayerNorm]), ...])`, which the flattener should turn into `predictor.text_encoder.lstms.{i}.0.weight_ih_l0` etc. Worth dumping the safetensors keys early and grepping.
- `predictor.rs::predict_duration` returns the raw `[B, T, max_dur]` tensor; `mod.rs:134-137` then sigmoids and sums along dim 1. **Verified bug** against upstream `kokoro/model.py:108`: `duration = torch.sigmoid(duration).sum(axis=-1) / speed`. The sum should be over the last axis (`D::Minus1` or dim 2 in the `[B, T, max_dur]` layout) to collapse `max_dur` into per-token continuous durations of shape `[B, T]`. The current `sum(1)` collapses the T axis instead and produces nonsense. Fix.
- `mod.rs:134-143` is also missing the integer-duration step. Upstream `model.py:109` does `pred_dur = torch.round(duration).clamp(min=1).long().squeeze()` — round to nearest, clamp minimum to 1 (so every phoneme gets at least one frame), cast to int64, squeeze the batch dim. The Rust code stops at `/speed` and bails. The integer `pred_dur` is what feeds the alignment matrix in §2.

### `src/model/decoder.rs` — ~30% complete

- Imports look closer to correct than the other files but need a compile pass — `candle_nn::Module` import path, `InstanceNorm1d` and `instance_norm1d` confirmation against 0.8 (it does exist; verify the config struct name).
- `decoder.rs:171-192` — `Decoder::forward` fuses asr/f0/n features through 4 residual blocks. **The iSTFTNet generator is entirely absent** — the comment at `decoder.rs:190` admits this. Without the generator there is no audio output, only intermediate features. See §5 for the missing components.
- `decoder.rs:185` — `if block.upsample` reads a private field; pub it or expose a method.
- `decoder.rs:148-151` — channel arithmetic `1024 + 2 + 64` assumes f0 / n / asr_res concat sizes; verify against the StyleTTS2 reference once before touching numbers.

### `src/model/mod.rs` — bails after duration

- `mod.rs:131` — verified against upstream `kokoro/model.py:104,115,118`. **The current code is correct.** Upstream does `s = ref_s[:, 128:]` and feeds that to `predictor.text_encoder` and `predictor.F0Ntrain`; `ref_s[:, :128]` goes to the decoder. The Rust extraction at this line matches the predictor side. Earlier draft of this brief flagged it as a swap based on StyleTTS2-convention memory; the convention does not apply here. Don't touch this line. **For the decoder integration (not yet wired):** pass `ref_s[:, :128]` (first half) to the decoder, NOT the same `s` used for the predictor.
- `mod.rs:139-143` — pipeline ends after duration. Missing: alignment matrix construction (length regulation; see §2 for the recipe and `model.py:110-113` for the upstream — `indices = repeat_interleave(arange(T), pred_dur)`, scatter ones into `[T, sum(pred_dur)]`, unsqueeze batch), F0/N forward, text encoder forward, decoder forward. **F0Ntrain takes `(predictor.text_encoder(d_en, ...) @ pred_aln_trg, s_pred)`, NOT `(d_en @ pred_aln_trg, s_pred)`** — the input to F0/N is the predictor's intermediate text features, not raw bert output. See `model.py:114-115`.
- `mod.rs:122-123` — `text_mask = Tensor::zeros((1, input_len), DType::U8, ...)` is "all valid" — fine for batch=1, no padding. **Convention note** from `model.py:100-102`: text_mask is `True for padding positions` (computed via `(arange+1) > input_lengths`), and the bert call passes `(~text_mask).int()` as `attention_mask` — i.e. attention_mask is `1 for valid, 0 for padding`. The current Rust code passes the mask straight through; verify candle's `AlbertModel::forward_all` agrees on the inversion convention before adding any padding logic. Document the assumption; it'll bite if batching is ever added.
- `mod.rs:7` is `pub use config::Config;` and `mod.rs:15` re-imports `use self::config::Config;` — minor duplicate, harmless.
- The reference voice file `voices/af_heart.safetensors` is loaded **nowhere**. Add a `Voice::load(path)` that opens the safetensors, reads the `ref_s` tensor, and exposes `style_for(phoneme_count: usize) -> Tensor`. The Kokoro reference voice format is `[N_max, 1, 256]` indexed by phoneme count (each row is the style for utterances of that exact phoneme length). On out-of-range, clamp to the last row.

## 5. The iSTFTNet generator — what's missing in the decoder

This is the heaviest single piece. It's a vocoder: takes the asr+f0+n+style conditioning produced by the existing decoder front-end and outputs a waveform.

Components required, roughly in flow order:

1. **NSF (Neural Source Filter) harmonic source.** Given F0 contour and noise, generate a set of sine harmonics + noise excitation as the time-domain "source" signal. Kokoro/StyleTTS2 reuses the iSTFTNet variant from MB-iSTFT-VITS / RingFormer; expect a `SourceModuleHnNSF` or `SineGen` analog. Output channels feed into the decoder upsample stack.
2. **Multi-scale upsampler** (`ups` ModuleList in the reference). Sequence of `ConvTranspose1d` blocks with `upsample_rates` from config (likely `[10, 6]` or similar — read it). Each upsample stage is followed by a stack of `ResBlock` (with `resblock_kernel_sizes` × `resblock_dilation_sizes` from config) — multi-receptive-field fusion (MRF).
3. **Source downsampler / fuser** (`noise_convs` in the reference). The harmonic source from (1) is downsampled to match each upsample stage's temporal resolution and added in.
4. **Weight normalization on every Conv1d / ConvTranspose1d.** Reference uses `torch.nn.utils.weight_norm`. Candle has no built-in `weight_norm`; either (a) fold `weight_g * (weight_v / ||weight_v||)` at load time so inference uses plain Conv1d (preferred — matches the convention candle uses for batchnorm folding), or (b) implement a thin wrapper. **Critical**: `convert_weights.py` currently flattens `weight_g` and `weight_v` as separate tensors. You either need to fold during conversion (modify the script) or fold at model-load time in Rust. Pick one and document.
5. **Final iSTFT.** Last layer of the network produces a complex spectrogram (`gen_istft_n_fft / 2 + 1` magnitude + phase channels split). Inverse STFT with `gen_istft_hop_size` and `gen_istft_n_fft` from config gives the waveform. Candle does not ship an iSTFT — write it. Reference impl: overlap-add with synthesis window (Hann), normalized by the squared-window OLA correction. Test against `torch.istft` to ≤1e-4 abs.
6. **Output activation.** Typically `tanh` then a fixed gain.

Read the upstream reference (`hexgrad/kokoro` repo, `kokoro/istftnet.py` is the file) before writing — the structure has variations from canonical iSTFTNet.

## 6. G2P strategy — pick one, justify in the PR

There is no Rust port of misaki (Kokoro's official G2P frontend). You must choose. None of the options is obviously right; the choice depends on accent target and how much non-Rust runtime dependency is acceptable.

**Hard constraint from the model:** Kokoro's vocab is IPA single-codepoint tokens (e.g. `ə`, `ɪ`, `tʃ` as two chars). Whichever G2P you pick, its output must be IPA in the exact symbol set Kokoro was trained on. The vocab is in `config.json` under `vocab` (mapped at `mod.rs:83-88` via `chars().filter_map(vocab.get)`). Phonemes the vocab doesn't know are silently dropped — this will mask G2P bugs. **Add a warning when any input phoneme is unmapped.**

Options:

- **(a) espeak-ng FFI.** The Rust crate `espeakng-sys` (or build espeak-ng from source and call its IPA mode). Kokoro reference uses espeak-ng as misaki's fallback for OOV words, so the symbol set already overlaps. Quality below misaki for English (worse homograph disambiguation, no learned prosody hints) but fully native at runtime. Pulls in a C dependency.
- **(b) Port misaki to Rust.** Misaki is ~few thousand lines of Python: a CMUdict-style lexicon, a small homograph disambiguator, English text normalization, espeak-ng fallback. Possible but several days of work and easy to get subtly wrong.
- **(c) Python sidecar.** Ship `misaki` as a separate process; Rust talks to it over stdin/stdout or a unix socket. Best quality, smallest porting risk, but reintroduces Python at runtime — violates the "native Rust" goal in spirit.

**Recommended starting point** (justify your actual pick): (a) for milestone 1 with a `#[cfg(feature = "espeak")]` gate, behind a `Phonemizer` trait. The trait makes (b) or (c) a swappable later upgrade without churning the model code. Document in your PR what you chose and why, and what it costs us in quality vs misaki.

For milestone 1 only, a hand-coded `phonemize_for_test()` that returns a hardcoded IPA string for one canned phrase is acceptable as a stub to unblock end-to-end work. **Do not ship it.** Replace before milestone 2.

## 7. Validation pattern (this is non-negotiable)

The pattern that worked on `~/rust/nemotron-speech` is the same pattern here. Read `~/rust/nemotron-speech/state.md` and `~/rust/nemotron-speech/tools/` before starting. Summary:

**For every model stage, write two artifacts:**

1. `tools/reference_<stage>.py` — a from-scratch PyTorch reimplementation that loads the *converted safetensors* (not the original `.pth`) and dumps the stage's output as `tmp/reference_<stage>.bin` + `.npz`. Loading from converted safetensors validates the conversion pipeline simultaneously with the port.
2. `src/bin/<stage>_check.rs` — a Rust binary that runs the candle implementation on identical inputs (read from the same `.npz`), compares to the `.bin`, prints `max_abs` and `mean_abs` deltas. Exit non-zero if `max_abs > target`.

**Stages to validate, in order:**

| # | Stage | Input | Output shape | Target max_abs |
|---|-------|-------|--------------|----------------|
| 1 | Voice tensor load | `voices/af_heart.safetensors`, phoneme_count | `[1, 256]` | exact (bit-identical) |
| 2 | `phonemes_to_ids` | IPA string | `Vec<i64>` | exact |
| 3 | CustomAlbert (PL-BERT) | input_ids, attention_mask | `[1, T, 768]` | 1e-5 |
| 4 | `bert_encoder` Linear | bert_dur | `[1, T, 512]` | 1e-5 |
| 5 | TextEncoder (embed→CNN×3→BiLSTM) | input_ids | `[1, 512, T]` | 1e-4 |
| 6 | ProsodyPredictor.predict_duration | (d_en, s_dur, mask) | `[1, T, 50]` then sum→`[1, T]` | 1e-4 logits, integer durations exact |
| 7 | Alignment matrix | durations | `[T, sum(dur)]` | exact |
| 8 | F0/N prediction | (d_en@aln, s_pred) | each `[1, sum(dur)]` | 1e-3 |
| 9 | Decoder pre-vocoder (asr+f0+n fusion) | all of the above | `[1, C, sum(dur)]` | 1e-4 |
| 10 | iSTFTNet upsampler stages | decoder features | per-stage shapes | 1e-3 |
| 11 | Inverse STFT | complex spectrogram | `[1, sum(dur)*300]` | 1e-4 |
| 12 | Full forward | text | waveform `[1, N_samples]` | 1e-3 |

These targets are starting estimates from the nemotron-speech experience; tighten them as receipts come in. Do not relax them silently — if a stage refuses to converge, find the bug, don't bump the threshold.

**Why this order matters:** every stage builds on the previous one's output. If stage 8 fails, you need to know whether the bug is in stage 8 or upstream. Per-stage diffs localize the bug to the layer that introduced it. Skipping this step is the single most common way ports of generative models silently produce garbage that "kind of sounds right."

The Python reference can either (a) be a from-scratch reimpl that loads the converted safetensors (highest signal — also tests conversion), or (b) the official `hexgrad/kokoro` Python package with hooks added to dump intermediates. Option (a) is harder but nemotron-speech proved it's worth it. **Pick (a) unless there's a specific reason not to.**

## 8. Step ordering (suggested)

**Commit cadence:** commit at the end of each numbered step at minimum; commit per-stage inside step 5 (one commit per validated stage). Reviewers should be able to step through commits and see the receipts table fill in row by row. A "compile pass green" commit from step 1 is the first commit on this repo.

1. **Compile pass.** Fix the API-drift bugs in §4 with no logic changes. Goal: `cargo check` is clean. For `bert.rs` and `config.rs`, replace the hallucinated `candle_transformers::albert` imports with local placeholder types (empty struct + stub forward returning `unimplemented!()` is fine for this step) — do not write the real ALBERT here, that's step 3. Do not advance past this step until clean. **Commit.**
2. **G2P stub + decision.** Implement the `Phonemizer` trait, hardcode one phrase for the test path, document your G2P choice for the real path. **Commit.**
3. **Implement CustomAlbert (PL-BERT).** Replace the `bert.rs` placeholder from step 1 with a real ALBERT implementation per §4 bert.rs subsection: factorized embedding + cross-layer parameter sharing + canonical transformer block. Define local `AlbertConfig` struct here. Wire `config.rs::to_albert_config()` to return the local type. Forward signature matches what `mod.rs:126-127` already calls (`bert.forward(input_ids, attention_mask) -> Tensor [B, T, hidden_size]`). Validation of this step lands as stage 3 in step 4. **Commit** when the impl compiles and at least loads weights from the converted safetensors without key-mismatch errors.
4. **Validation infra.** Set up `tools/reference_*.py` skeleton and one `<stage>_check` Rust binary. Get stages 1–4 (voice load → BERT → bert_encoder Linear) green before moving on — **stage 3 (CustomAlbert) is the gating diff that confirms step 3's implementation is correct.** **Commit per stage** (4 commits expected here).
5. **Per-layer port forward.** Stages 5–9 in order. Each stage: write the reference, get the Rust diff under threshold, **commit with the receipt in the message**, move on.
6. **iSTFTNet generator.** Stages 10–11. The biggest single chunk; budget accordingly. Validate the inverse STFT against `torch.istft` in isolation first (tiny synthetic input) before plugging in the real spectrogram. **Commit per sub-stage** — at minimum: weight-norm folding decision landed, NSF source matches, each upsampler stage matches, iSTFT against synthetic input matches, iSTFT against real spectrogram matches.
7. **End-to-end glue in `mod.rs::forward`.** Wire stages together: input_ids → bert+text_encoder → predict duration → build alignment → F0/N → decoder → audio. This should be ~30 lines once the pieces work. **Commit.**
8. **Voice tensor loader + WAV writer.** `Voice::load` reading the safetensors and a `hound`-based mono 24 kHz WAV writer wired into a CLI binary `src/bin/speak.rs` that takes `--text` and `--out file.wav`. **Commit.**
9. **Stage 12: end-to-end validation.** Reference Python produces a WAV; Rust produces a WAV; diff samples to ≤1e-3 max abs. **This is the milestone 1 acceptance test.** Listening is a sanity check on top, not a substitute. **Commit with the final receipt.**
10. **Only after stage 12 is green:** wire into the axum `process_queue` (replacing the `main.rs:130-143` stub). The server is already there; this is plumbing. **Commit.**
11. **Voice-agent integration.** UDP wire format with pty-1. Defer until 10 is done; format depends on whether we stream chunks or whole utterances.

Do not parallelize 5 with 6 — debugging an end-to-end audible failure when the upstream stages aren't validated is the trap to avoid.

## 9. Validation receipts

Fill this in as you go. Same format as `~/rust/nemotron-speech/state.md`. Numbers are max_abs vs PyTorch reference unless noted; mean_abs in parens.

| Stage | Max abs | Mean abs | Notes |
|-------|---------|----------|-------|
| 1. Voice tensor load | — | — | |
| 2. phonemes_to_ids | — | — | |
| 3. CustomAlbert (PL-BERT) | — | — | |
| 4. bert_encoder Linear | — | — | |
| 5. TextEncoder | — | — | |
| 6. predict_duration logits | — | — | |
| 6b. Integer durations | — | — | exact match required |
| 7. Alignment matrix | — | — | exact match required |
| 8. F0 prediction | — | — | |
| 8b. N prediction | — | — | |
| 9. Decoder fusion | — | — | |
| 10. Upsampler stage 0 | — | — | |
| 10b. Upsampler stage 1 | — | — | |
| 11. Inverse STFT (synthetic) | — | — | |
| 11b. Inverse STFT (real spectrogram) | — | — | |
| 12. End-to-end waveform | — | — | **milestone 1 gate** |

## 10. Don't do

- **Don't ONNX-fallback.** If candle is missing a primitive, write it (iSTFT is the obvious one). If a layer is hard, that's the work, not a reason to bail.
- **Don't add Python at inference time.** Python is for `tools/reference_*.py` only.
- **Don't extend `convert_weights.py` without an immediate Rust counterpart.** New conversion logic with no Rust consumer becomes dead code that drifts.
- **Don't skip the per-layer diff step.** It feels like overhead until stage 12 produces noise and you have no idea which of 11 layers introduced it. Every shortcut here costs more time than it saves.
- **Don't widen the `max_abs` thresholds in §7 to make a test pass.** The thresholds are calibrated; a stage that needs 1e-1 to pass has a bug, not a tolerance problem.
- **Don't refactor `main.rs` or the axum harness while doing model work.** It's fine. Touch it only when wiring step 9.
- **Don't commit `models/` or anything from `tmp/`.** Add to `.gitignore` if not already.
- **Don't silently drop unmapped phonemes.** The current `phonemes_to_ids` filters them out — instrument with a warning so G2P bugs surface.
- **Don't ship the hardcoded G2P stub from §6.** It exists to unblock end-to-end during step 4–6; it must be replaced before milestone 2.

## 11. Reference material

- Upstream model repo: `hexgrad/Kokoro-82M` on HuggingFace. `kokoro-v1_0.pth`, `config.json`, `voices/*.pt`.
- Upstream Python inference code: `hexgrad/kokoro` on GitHub. `kokoro/model.py`, `kokoro/istftnet.py` are the two files to read closely.
- StyleTTS2 reference (Kokoro is a derivative): `yl4579/StyleTTS2`. Useful for the prosody predictor and decoder front-end.
- Misaki (G2P): `hexgrad/misaki`.
- Validation pattern exemplar: `~/rust/nemotron-speech/state.md` + `~/rust/nemotron-speech/tools/`.
- Candle 0.8 source for API confirmation: `https://github.com/huggingface/candle/tree/0.8.0`.

## 12. Coordination

- This repo's main Claude instance (the one who wrote this brief) is available for review and can pair on tricky stages. Ping via pty-1.
- pty-1 owns the voice-agent side and the eventual UDP wire format. Don't design the wire format in this repo; surface a clean Rust API (`fn synthesize(text: &str) -> Result<Vec<f32>>` plus a streaming variant) and let pty-1 wrap it.
- Commit early and often. The first commit of this repo should be "compile pass green" from step 1, not a giant lump.
