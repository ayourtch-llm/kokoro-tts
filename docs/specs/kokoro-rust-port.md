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
                                                                                                          └─► (en, s_pred) ──► predictor.F0Ntrain ──► (F0_pred, N_pred) each [B, 2 * sum(pred_dur)]
                                                                                                                                                       (F0[1]/N[1] AdainResBlk1d upsamples by 2;
                                                                                                                                                        decoder.f0_conv / n_conv stride=2 collapses
                                                                                                                                                        back to sum(pred_dur). Verified at stage 8:
                                                                                                                                                        for sum(pred_dur)=63, F0_pred shape=[1, 126].)

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

The transformer block itself is canonical: pre/post LayerNorm + multi-head self-attention + feed-forward (typically GELU). Token type embeddings are wired through but used only for the BOS/EOS slots in our single-batch path. **Important correction from codex's stage-3 work:** the converted state-dict has `bert.embeddings.token_type_embeddings.weight` of shape `[2, embedding_size]`, **not** `[1, embedding_size]`. Local `AlbertConfig.type_vocab_size` must be `2` to load this correctly. The earlier draft of this brief said `1`; that was wrong (Kokoro's HF `config.json` has no explicit `type_vocab_size` field, so the actual value comes from inspecting the converted weights — codex did and committed the fix).

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

### `src/model/decoder.rs` — front-end correct, generator missing, weight-loading bugs

The encode + 4-decode-block pipeline structure matches upstream Decoder (istftnet.py:384-421). The `AdainResBlk1d` here (lines 39-153 post-compile-pass) is **structurally correct** — it has the `pool` ConvTranspose1d for upsample, the `conv1x1` learned shortcut, the `_residual` and `_shortcut` paths, and the final `* rsqrt(2)` scale. **This is the source of truth for `AdainResBlk1d`** — when fixing the broken duplicate in `predictor.rs` (review queue item #5), copy from here.

- `decoder.rs::Decoder::forward` (post-compile-pass) ends with `Ok(x)` where `x` is the post-decode features. **The Generator (iSTFTNet vocoder) is entirely absent** — see §5 for the full breakdown. Without it there is no audio, only conditioned features.
- `decoder.rs::asr_res` is loaded as `candle_nn::conv1d(512, 64, 1, ..., vb.pp("asr_res"))`, expecting key `asr_res.weight`. **Upstream (istftnet.py:402)** wraps it in `nn.Sequential(weight_norm(nn.Conv1d(512, 64, 1)))` — state-dict keys are `asr_res.0.weight` (Sequential indexing) and Conv1d is weight-normed (so `asr_res.0.parametrizations.weight.original0/1` unless folded). Two layers of mismatch.
- `decoder.rs::f0_conv`, `decoder.rs::n_conv` are plain Conv1d in codex's code; **upstream (istftnet.py:400-401)** wraps both in `weight_norm`. Same parametrization key mismatch unless folded.
- All Conv1d in `AdainResBlk1d` (`conv1`, `conv2`, `conv1x1`) and the `pool` ConvTranspose1d are weight-normed upstream (istftnet.py:355,356,360,352). Same folding decision applies pervasively.
- **`decoder.rs::AdaIN1d` affine — RETRACTED, codex's original no-affine code is correct.** I previously flagged this as a "silent param skip" bug (the original review queue item #6) on the grounds that `istftnet.py:24` has `nn.InstanceNorm1d(num_features, affine=True)`. After actually inspecting the trained checkpoint: the `kokoro-v1_0.pth` has **zero** `*.norm.weight`/`*.norm.bias` keys for predictor or decoder (verified by direct `torch.load` of the .pth and grep of the converted safetensors header — both empty). Upstream's `affine=True` is dead code at inference: PyTorch initializes those params to ones/zeros by default, upstream's load uses `strict=False` fallback so missing keys leave defaults in place, and `IN(x)*1 + 0` is mathematically `IN(x)`. The actual scale/shift comes entirely from `(1 + style_gamma) * IN(x) + style_beta` where gamma/beta are produced by the AdaIN1d's `fc` Linear (which IS in the state-dict). **Action**: revert commit `6a6be6c` (which added `vb.get(num_features, "norm.weight")?` calls to `decoder.rs::AdaIN1d::load` and `predictor.rs::Adain1d::load`); those `vb.get(...)?` calls will fail strict-load the moment the full Kokoro::load actually instantiates F0/N or decoder AdaIN1d (stage 8 onwards). Codex's pre-6a6be6c implementation was right.
- `decoder.rs::Decoder::load` channel arithmetic `1024 + 2 + 64` matches upstream (istftnet.py:396-399). No fix needed there.
- `decoder.rs::Decoder::forward` fusion logic verified against upstream (istftnet.py:407-421) line-for-line: `f0_conv(f0.unsqueeze(1))` → `n_conv(n.unsqueeze(1))` → `cat([asr, f0, n], dim=1)` → `encode(x, s)` → loop with `res` flag controlling whether to re-cat `[x, asr_res, f0, n]` before each `block(x, s)` → flip `res = false` after the upsample block. Identical structure. No fix needed in the forward body — the only outstanding work here is appending the Generator call after the decode loop (see §5).

### `src/model/mod.rs` — bails after duration

- `mod.rs:131` — verified against upstream `kokoro/model.py:104,115,118`. **The current code is correct.** Upstream does `s = ref_s[:, 128:]` and feeds that to `predictor.text_encoder` and `predictor.F0Ntrain`; `ref_s[:, :128]` goes to the decoder. The Rust extraction at this line matches the predictor side. Earlier draft of this brief flagged it as a swap based on StyleTTS2-convention memory; the convention does not apply here. Don't touch this line. **For the decoder integration (not yet wired):** pass `ref_s[:, :128]` (first half) to the decoder, NOT the same `s` used for the predictor.
- `mod.rs:139-143` — pipeline ends after duration. Missing: alignment matrix construction (length regulation; see §2 for the recipe and `model.py:110-113` for the upstream — `indices = repeat_interleave(arange(T), pred_dur)`, scatter ones into `[T, sum(pred_dur)]`, unsqueeze batch), F0/N forward, text encoder forward, decoder forward. **F0Ntrain takes `(predictor.text_encoder(d_en, ...) @ pred_aln_trg, s_pred)`, NOT `(d_en @ pred_aln_trg, s_pred)`** — the input to F0/N is the predictor's intermediate text features, not raw bert output. See `model.py:114-115`.
- **F0/N return shape contract** (verified at stage 8): `predictor.F0Ntrain` returns `(F0_pred, N_pred)` each of shape `[B, 2 * sum(pred_dur)]` — both are 2D (already `.squeeze(1)`'d, matching upstream `modules.py:134`). Don't re-squeeze, don't `.unsqueeze(1)` before passing to the decoder. The 2× length comes from the F0[1] / N[1] AdainResBlk1d having `upsample=true`; the decoder's `f0_conv` / `n_conv` (stride=2) collapses it back to `sum(pred_dur)` internally before the asr+f0+n cat. So the *wire format* between predictor and decoder is `[B, 2*T_dec]`, but the *post-fusion* feature length the encode block sees is `[B, _, T_dec]`.
- `mod.rs:122-123` — `text_mask = Tensor::zeros((1, input_len), DType::U8, ...)` is "all valid" — fine for batch=1, no padding. **Convention note** from `model.py:100-102`: text_mask is `True for padding positions` (computed via `(arange+1) > input_lengths`), and the bert call passes `(~text_mask).int()` as `attention_mask` — i.e. attention_mask is `1 for valid, 0 for padding`. The current Rust code passes the mask straight through; verify candle's `AlbertModel::forward_all` agrees on the inversion convention before adding any padding logic. Document the assumption; it'll bite if batching is ever added.
- `mod.rs:7` is `pub use config::Config;` and `mod.rs:15` re-imports `use self::config::Config;` — minor duplicate, harmless.
- The reference voice file `voices/af_heart.safetensors` is loaded **nowhere**. Add a `Voice::load(path)` that opens the safetensors, reads the `ref_s` tensor, and exposes `style_for(phoneme_count: usize) -> Tensor`. The Kokoro reference voice format is `[N_max, 1, 256]` indexed by phoneme count (each row is the style for utterances of that exact phoneme length). On out-of-range, clamp to the last row.

## 5. The iSTFTNet generator — what's missing in the decoder

The heaviest single piece. The decoder front-end (asr + f0 + n + style fusion through 4 AdainResBlk1d) produces conditioned features; the **Generator** (`istftnet.py:257-326`) turns those into a waveform via an NSF source + multi-stage upsampling + iSTFT. Currently entirely absent in `decoder.rs`.

Read `kokoro/istftnet.py` (421 lines) end-to-end before writing. The key insights from that read are below — they're load-bearing and several are not obvious from "iSTFTNet" alone.

### 5.0. Concrete constants for Kokoro-82M

Pulled directly from `hexgrad/Kokoro-82M/config.json` so codex doesn't have to wait for the model download to plan shapes:

| Constant | Value | Where it shows up |
|----------|-------|-------------------|
| `upsample_rates` | `[10, 6]` | `Generator.ups` strides; `num_upsamples = 2` |
| `upsample_kernel_sizes` | `[20, 12]` | `Generator.ups` kernel sizes |
| `resblock_kernel_sizes` | `[3, 7, 11]` | per-stage MRF kernels; `num_kernels = 3` |
| `resblock_dilation_sizes` | `[[1,3,5], [1,3,5], [1,3,5]]` | per-kernel dilation triples inside each `AdaINResBlock1` |
| `upsample_initial_channel` | `512` | channel at start of generator; halved each upsample |
| `gen_istft_n_fft` | `20` | iSTFT filter length / window |
| `gen_istft_hop_size` | `5` | iSTFT hop |
| `style_dim` | `128` | each AdaIN1d style input |
| `hidden_dim` | `512` | predictor / decoder feature width |
| `n_layer` | `3` | predictor.text_encoder LSTM-stack depth |
| `max_dur` | `50` | duration_proj output channels |
| `n_token` | `178` | vocab size for embeddings |

Derived:
- Total upsample factor: `prod(upsample_rates) * gen_istft_hop_size = 10 * 6 * 5 = 300` audio samples per duration frame (matches the §2 architecture note: 1 frame ≈ 12.5 ms @ 24 kHz).
- Generator channel progression: `512 → 256 → 128` (`upsample_initial_channel // 2^i`). `ch` after stage 0 = 256, after stage 1 = 128.
- Total `Generator.resblocks` instances: `num_upsamples * num_kernels = 2 * 3 = 6` (state-dict keys `resblocks.{0..5}.*`).
- For `noise_convs[0]` (i=0, not last stage): `stride_f0 = prod(upsample_rates[1:]) = 6`, kernel = `12`, padding = `(6+1)//2 = 3`, in_ch = `gen_istft_n_fft + 2 = 22`, out_ch = `256`.
- For `noise_convs[1]` (i=1, last stage): kernel = `1`, in_ch = `22`, out_ch = `128`.
- `noise_res[0]` = `AdaINResBlock1(256, kernel=7, dilations=[1,3,5], style_dim=128)`.
- `noise_res[1]` = `AdaINResBlock1(128, kernel=11, dilations=[1,3,5], style_dim=128)`.
- `conv_post`: `Conv1d(128, gen_istft_n_fft + 2 = 22, kernel=7, padding=3)` (weight-normed).
- Output split: `spec = exp(x[:, :11, :])`, `phase = sin(x[:, 11:, :])` (each `[B, 11, T_audio]` since `n_fft//2 + 1 = 11`).
- NSF source: `harmonic_num = 8`, `voiced_threshold = 10` (hard-coded in upstream `Generator.__init__` at istftnet.py:265, not config-driven), `sampling_rate = 24000`, `upsample_scale = 300`.

### 5.1. Two distinct AdaIN block classes — don't conflate

Upstream defines **two different** AdaIN-conditioned residual blocks. They have different signatures, different forward semantics, and different parameter layouts. Names differ by one letter.

- **`AdainResBlk1d`** (istftnet.py:340-381) — the one used by `Decoder.encode`/`decode` and by `ProsodyPredictor.F0`/`N`. 2 convs (conv1, conv2), 2 AdaIN1d (norm1, norm2), optional `pool` ConvTranspose1d for upsample, optional `conv1x1` for dim mismatch, leaky_relu activation, final `* rsqrt(2)` scale on the residual+shortcut sum. Codex's `decoder.rs` has this correctly; `predictor.rs` has a broken duplicate (review queue item #5).
- **`AdaINResBlock1`** (istftnet.py:34-77) — the one used by `Generator.resblocks` and `Generator.noise_res`. Three parallel dilated paths: `convs1` (3 dilated Conv1d, dilations from `resblock_dilation_sizes[i]`) + `convs2` (3 plain Conv1d) + 6 AdaIN1d (3 adain1, 3 adain2) + 6 learnable per-channel `alpha` parameters (alpha1, alpha2). Activation is **Snake1D**, not leaky_relu (see §5.2). Each of the 3 sub-blocks does `xt = norm(x) → snake(xt) → conv1 → norm → snake → conv2`, then `x = xt + x` (residual at *every* sub-block, not at the end). Returns `x` after all 3 sub-blocks accumulate. **This block does not exist in any form in codex's port.**

State-dict keys for `AdaINResBlock1`: `convs1.{0,1,2}.*`, `convs2.{0,1,2}.*`, `adain1.{0,1,2}.*`, `adain2.{0,1,2}.*`, `alpha1.{0,1,2}` (each is a `[1, channels, 1]` parameter), `alpha2.{0,1,2}`.

### 5.2. Snake1D activation — custom, learnable

Used inside `AdaINResBlock1` (istftnet.py:71,74):
```python
xt = xt + (1 / a) * (torch.sin(a * xt) ** 2)
```
where `a` is a learnable `[1, channels, 1]` parameter (initialized to ones, see istftnet.py:65-66). This is the BigVGAN/Snake activation — **not** leaky_relu, not GELU, not in candle. Implement as `x + (1/alpha) * sin(alpha * x).square()`. Easy in candle, just don't substitute leaky_relu by accident — the resulting audio will be wrong in a hard-to-localize way.

Numerical caveat: when `alpha` is near zero, `1/alpha` blows up. Upstream relies on alpha being trained away from zero; do not add a clamp/eps unless validation receipts force it (and document if you do).

**Rust callsite pre-flag** (codex hasn't started this yet — read before implementing AdaINResBlock1):

- Snake1D fires **6 times per `AdaINResBlock1.forward()` call** — twice per sub-block (after norm1 and after norm2), three sub-blocks per forward. Each call uses a *different* learnable alpha tensor (`alpha1.{0,1,2}` for the post-norm1 calls, `alpha2.{0,1,2}` for the post-norm2 calls). Don't share alpha across sub-blocks.
- Order inside each sub-block (istftnet.py:69-76): `xt = norm1(x, s) → snake(xt, alpha1[i]) → conv1(xt) → norm2(xt, s) → snake(xt, alpha2[i]) → conv2(xt) → x = xt + x`. Snake is *between* norm and conv, not before norm or after conv.
- State-dict load: `alpha1.{0,1,2}` and `alpha2.{0,1,2}` are stored *directly* as parameters (no `.weight` suffix — they're `nn.Parameter`, not `nn.Linear`). In candle this is `vb.pp("alpha1.0").get((1, channels, 1), "")?` — the empty string is the param name within the prefix. Don't try `vb.pp("alpha1").get(..., "0")?` — that won't match the flattened state-dict layout the converter produces.
- Suggested Rust shape: a `Snake1D { alpha: Tensor }` struct with `forward(&self, x: &Tensor) -> Result<Tensor>`. Or a free function `snake1d(x: &Tensor, alpha: &Tensor) -> Result<Tensor>` if you'd rather not box it. Either works; struct is cleaner if you need to validate alpha shape on load.
- Validation pattern: dump `(x, alpha) → snake1d(x, alpha)` from upstream on random input first (one isolated check before plugging into AdaINResBlock1). Stage-10 receipts will hide a snake bug behind 5 other things; the standalone check localizes it.

### 5.3. NSF source path (istftnet.py:108-254 + Generator forward)

Flow:
1. F0 curve from the predictor, shape `[B, T]`. Upsample to audio rate via `nn.Upsample(scale_factor=prod(upsample_rates) * gen_istft_hop_size, mode='nearest')` — for Kokoro that's typically 300× (matches one frame of duration → one 12.5ms audio chunk @ 24kHz).
2. `SineGen` (istftnet.py:108-209): produces sine waveforms at fundamental + `harmonic_num` overtones (Kokoro: `harmonic_num=8`, so 9 channels). The phase generation involves cumulative-sum of (f0 / sample_rate) and an interpolation trick (lines 153-157) for sub-rate phase computation; **port carefully** — naive cumsum at audio rate produces accumulated float-precision drift. UV (voiced/unvoiced) mask is `(f0 > voiced_threshold)` with Kokoro's threshold = 10.
3. `SourceModuleHnNSF` (istftnet.py:212-254): `Linear(harmonic_num+1, 1)` merges the harmonics into a single excitation, then `tanh`. Produces `[B, T_audio, 1]`.
4. **STFT of the harmonic source** (Generator forward line 304): `har_spec, har_phase = self.stft.transform(har_source)` → `[B, freq_bins, T_stft]` magnitude and phase. These are concatenated on the channel axis: `har = cat([har_spec, har_phase], dim=1)` → `[B, 2*freq_bins, T_stft]` = `[B, gen_istft_n_fft + 2, T_stft]`.

The `har` tensor is what `noise_convs[i]` consume — it is **STFT'd source features, not raw waveform** entering the upsample stack.

### 5.4. Generator forward (istftnet.py:299-325)

Shapes annotated assuming Kokoro constants (§5.0). `T_dec = sum(pred_dur)` is the time dim of decoder front-end output. The generator upsample stack produces `T_pre_istft = T_dec * 60` frames (= `T_dec * prod(upsample_rates)`); the iSTFT does the final 5× via `hop_size`, yielding `T_audio = T_dec * 300` audio samples. `T_stft = T_pre_istft = T_dec * 60` for the source-path STFT (same hop). Exact frame counts at stage boundaries depend on padding details and need empirical verification once codex runs the upstream reference; the annotations below are correct to within ±1 frame at boundaries.

```
# Inputs to Generator.forward(x, s, F0_curve):
#   x        : [B, 512, T_dec]   — output of Decoder front-end (post 4-decode-block loop)
#   s        : [B, 128]          — s_dec = ref_s[:, :128]
#   F0_curve : [B, T_dec]        — F0 from predictor.F0Ntrain

# === NSF source path (computed inside Generator.forward, with no_grad) ===
f0    = f0_upsamp(F0_curve.unsqueeze(1)).transpose(1, 2)
                                       # [B, T_audio, 1]   T_audio = T_dec * 300
har_source, _, _ = m_source(f0)        # [B, T_audio, 1]
har_source = har_source.transpose(1,2).squeeze(1)
                                       # [B, T_audio]
har_spec, har_phase = stft.transform(har_source)
                                       # each [B, 11, T_stft]   freq_bins = 20//2+1 = 11; T_stft ≈ T_dec*60
har = cat([har_spec, har_phase], dim=1)
                                       # [B, 22, T_stft]

# === Per-upsample-stage loop, num_upsamples = 2 ===
# Stage i = 0  (ch_in = 512, ch_out = 256, stride = 10, kernel = 20)
# Stage i = 1  (ch_in = 256, ch_out = 128, stride = 6,  kernel = 12)  — last stage
for i in 0..2:
    x = leaky_relu(x, 0.1)
    x_source = noise_convs[i](har)        # i=0: [B, 256, ~T_dec*10]; i=1: [B, 128, T_stft] = [B, 128, ~T_dec*60]
    x_source = noise_res[i](x_source, s)  # AdaINResBlock1; same shape out
    x = ups[i](x)                         # ConvTranspose1d upsample by upsample_rates[i]
                                          # i=0: [B, 256, ~T_dec*10]; i=1: [B, 128, ~T_dec*60]
    if i == 1:                            # last stage only
        x = reflection_pad(x)             # ReflectionPad1d((1, 0)): +1 frame on left, 0 on right
                                          # shape becomes [B, 128, ~T_dec*60 + 1]; the +1 reconciles
                                          # an off-by-one between ups output and noise_conv output —
                                          # verify empirically; this is the kind of detail that bites
    x = x + x_source                      # shapes must match — see reflection_pad note above
    xs = sum(resblocks[i*3 + j](x, s) for j in 0..3) / 3
                                          # MRF: average AdaINResBlock1 outputs across kernels [3,7,11]
    x = xs                                # same shape as input

# === Post-processing → spectrogram → iSTFT ===
x = leaky_relu(x)
x = conv_post(x)                          # Conv1d(128, 22, kernel=7, padding=3, weight_norm)
                                          # [B, 22, T_pre_istft]   T_pre_istft ≈ T_dec*60
spec  = exp(x[:, :11, :])                 # [B, 11, T_pre_istft]
phase = sin(x[:, 11:, :])                 # [B, 11, T_pre_istft]
return stft.inverse(spec, phase)          # [B, T_audio]   T_audio = T_pre_istft * hop_size = T_dec*300
```

Items easy to miss:
- **Output activation is NOT tanh.** It's `exp` for the magnitude half and `sin` for the phase half. The earlier draft of this spec said "tanh + fixed gain" — that was wrong. (Magnitude via exp is so the network can output negative log-magnitudes with full dynamic range.)
- **ReflectionPad1d((1, 0)) only on the last upsample iteration**, before adding the source. One frame of left-padding, zero of right.
- **MRF is mean, not sum.** `xs / num_kernels` at the end (line 320). With `num_kernels = len(resblock_kernel_sizes)` (typically 3).
- **resblocks indexing is flat**: `i*num_kernels + j` selects from a flat ModuleList of `num_upsamples * num_kernels` AdaINResBlock1 instances (state-dict keys `resblocks.{0..N-1}.*`).
- **noise_convs structure changes per stage**: for `i < num_upsamples - 1`, `noise_convs[i]` is `Conv1d(n_fft+2, ch, kernel_size=stride_f0*2, stride=stride_f0, padding=(stride_f0+1)//2)` where `stride_f0 = prod(upsample_rates[i+1:])` (downsample to current resolution). On the last stage, `Conv1d(n_fft+2, ch, kernel_size=1)` (no downsample). `noise_res[i]` is `AdaINResBlock1(ch, 7, [1,3,5], style_dim)` for non-last stages, `AdaINResBlock1(ch, 11, [1,3,5], style_dim)` for last.

### 5.5. iSTFT — pick a strategy

Two upstream options, both implemented in istftnet.py:
- **`TorchSTFT`** (istftnet.py:80-105): uses `torch.stft` / `torch.istft` with complex tensors. Closer to numerical reference but candle has no complex tensors as of 0.8 — would require either a custom complex implementation or workarounds.
- **`CustomSTFT`** (`kokoro/custom_stft.py`, 197 lines): implements STFT/iSTFT as `Conv1d` (forward) and `ConvTranspose1d` (inverse) with precomputed real/imag kernels of shape `[freq_bins, 1, n_fft]`. No complex tensors needed. Used upstream when `disable_complex=True` (for ONNX export). **This is the candle-friendly path** — port `CustomSTFT` rather than reimplement `torch.istft`. Caveat in upstream comments (custom_stft.py:79): "approximate reconstruction with Hann + typical overlap" — it doesn't do the DC/Nyquist non-doubling correction that a textbook real iSTFT does. Validate stage 11 against `CustomSTFT`'s output (not `torch.istft`'s) so reference and Rust agree on the same approximation. If the milestone-1 acceptance test (stage 12) shows this approximation is too lossy, switch reference to `TorchSTFT` and implement complex iSTFT in Rust later — but start with CustomSTFT.

Constants from Kokoro's config: `gen_istft_n_fft = 20`, `gen_istft_hop_size = 5` (verify by reading `models/config.json` after download). Window is Hann, periodic, dtype f32.

### 5.6. Weight normalization — pervasive

Upstream applies weight-norm to every `Conv1d` and `ConvTranspose1d` in the decoder + generator + TextEncoder.cnn. **Convention check (verified against the actual `kokoro-v1_0.pth`):** the trained checkpoint uses the **deprecated `torch.nn.utils.weight_norm`** API — state-dict keys are `<conv>.weight_g` (magnitude scalar, shape `[out_ch, 1, 1]`) and `<conv>.weight_v` (direction, shape `[out_ch, in_ch, kernel]`). 178 such pairs in the safetensors. The newer `torch.nn.utils.parametrizations.weight_norm` API (which would produce `parametrizations.weight.original0/1`) is *not* what was used — even though current upstream `istftnet.py` imports the new API. (The trained .pth predates that source change; the source code now uses the new API but the saved checkpoint has the old key layout.)

Folded weight: `weight = weight_v * (weight_g / ||weight_v||_per_output_channel)`, where the norm is computed across all dims except dim 0.

Two strategies (pick one, document):
- **(A) Fold during conversion.** Modify `convert_weights.py` to detect `.weight_g`/`.weight_v` pairs, compute the folded `weight`, and store as flat `weight`. Rust load uses plain `candle_nn::conv1d` etc. **Recommended.** Matches the candle convention of folding train-time gymnastics at conversion time.
- **(B) Fold at load time.** Define a `weight_normed_conv1d` Rust helper that reads `<prefix>.weight_g` / `<prefix>.weight_v`, computes the weight in Rust, constructs a `Conv1d` with the folded value, then loads bias separately. More code but conversion script stays simpler.

Either way: this affects **TextEncoder.cnn convs, every Conv1d in AdainResBlk1d (conv1, conv2, conv1x1, pool), Decoder.f0_conv/n_conv/asr_res, Generator.ups, Generator.conv_post, AdaINResBlock1.convs1/convs2** — basically every conv except the AdaIN1d FC linears and the standalone ProsodyPredictor.{F0,N}_proj convs (the latter is a tiny `Conv1d(d_hid/2, 1, 1)` with no weight_norm). Codex landed strategy choice at stage 5 (CnnBlock fix in 679ed97, validated to max_abs=2.161e-6 at stage 5); reuse that machinery across the rest of the model.

### 5.7. Implementation order suggestion

1. Port `CustomSTFT` standalone, validate forward+inverse round-trip vs upstream `CustomSTFT` (not `torch.istft`) to ≤1e-5 on a synthetic input. This is small, contained, and gates everything else.
2. Port `SineGen` + `SourceModuleHnNSF`, validate `har_source` output against upstream on a fixed F0 contour. The cumsum precision matters — diff should be ≤1e-3 abs.
3. Port `Snake1D` activation as a free function or a small struct holding `alpha`. Validate vs upstream on random input.
4. Port `AdaINResBlock1` (the 3-block-with-Snake one). Validate one block in isolation with random style + features.
5. Port `Generator.forward` end-to-end, validate against upstream Generator on saved (x, s, f0) inputs. This is the integration test for steps 1-4.
6. Wire into `Decoder.forward`: after the existing 4-decode-block loop, call `Generator(x, s, F0_curve)` and return the waveform.

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
| 1. Voice tensor load | 0.000e0 | 0.000e0 | exact match, shape `[1, 256]`, phoneme_count=11 |
| 2. phonemes_to_ids | exact | exact | ids `[50, 83, 54, 156, 57, 135, 16, 65, 156, 87, 123, 54, 46]` |
| 3. CustomAlbert (PL-BERT) | 7.361e-6 | 1.355e-6 | release checker, shape `[1, 15, 768]`; converted token_type embeddings are `[2, 128]` |
| 4. bert_encoder Linear | 5.007e-6 | 2.953e-7 | isolated linear check using reference `bert_dur`, shape `[1, 15, 512]` |
| 5. TextEncoder | 2.161e-6 | 9.375e-8 | release checker, shape `[1, 512, 15]`; validates CNN weight-norm fold + channel LayerNorm + BiLSTM |
| 6. predict_duration logits | 1.144e-5 | 2.248e-6 | release checker, shape `[1, 15, 50]`; uses upstream-verified predictor style half `ref_s[:, 128:]` |
| 6b. Integer durations | exact | exact | `[17, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 4, 3, 12, 8]` |
| 7. Alignment matrix | exact | exact | shape `[15, 63]`, zero mismatches |
| 8. F0 prediction | 1.793e-4 | 1.602e-5 | release checker, shape `[1, 126]`; F0/N branch upsamples by 2 |
| 8b. N prediction | 2.861e-6 | 1.061e-6 | release checker, shape `[1, 126]` |
| 9. Decoder fusion | 5.531e-5 | 2.929e-6 | release checker, shape `[1, 512, 126]`; decoder shortcut upsample is nearest-neighbor, residual path uses ConvTranspose pool |
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
