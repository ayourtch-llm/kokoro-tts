# Metal-acceleration modernization

**Audience:** the implementing instance (opencode / pty-3), with claude (pty-1) reviewing.
**Status at handoff:** Kokoro Rust port is shipped and parity-validated against the upstream PyTorch checkpoint via per-layer harnesses in `src/bin/*_check.rs`. Metal builds work (`cargo build --release --features metal`) but several hot paths force tensors back to CPU mid-pipeline. This milestone removes those round-trips, tidies the device-selection ergonomics, and bumps the candle stack.

## 0. Validation contract (read before touching code)

Three rules apply to every phase:

1. **Per-layer diff against a captured reference, not round-trip audio.** Round-trip equality (`inverse(forward(x)) ≈ x`) does not validate intermediate values that downstream code consumes as numbers — STFT phase is fed to `noise_convs` at `src/model/generator.rs:349`, so changing how phase is computed must be diffed at the `har` tensor, not at the audio output. The existing `src/bin/custom_stft_check.rs`, `src/bin/source_check.rs`, `src/bin/generator_check.rs`, and `src/bin/predict_duration_check.rs` are the canonical harnesses — extend them; don't invent new ones.
2. **The trained checkpoint is the highest-authority source.** If a "more efficient" formula produces different numbers than the existing CPU path, the conv weights downstream were trained for the existing numbers — match them exactly, don't replace them.
3. **Diff both CPU and Metal against the reference.** Every change must keep `--features metal` and the default CPU build numerically equivalent (within a documented tolerance, e.g. `< 1e-4` abs for f32 ops, looser for accumulated paths). Add a CI-style invocation comment to each `*_check.rs` you touch.

If a phase's validation can't be made to pass within tolerance, **stop and surface it** — do not merge.

## 1. Findings being addressed

Joint evaluation by claude (pty-1) and codex (pty-2). Ranked by impact:

| # | Location | Issue |
|---|----------|-------|
| 1 | `src/model/source.rs:52-151` (`SineGen::forward_with_controls`) | Entire NSF sine generation runs on CPU every Kokoro forward — `f0.to_vec1()`, Rust scalar loops for voiced mask + harmonics + cumulative phase + linear interp, then `Tensor::from_vec(...)` back to device. |
| 2 | `src/model/stft.rs:116-124` (`atan2_tensor`) | `imag` and `real` are pulled to host on every iSTFTNet forward to compute `f32::atan2`, then phase is re-uploaded. Called from `generator.rs:348` and `generator.rs:406`. |
| 3 | `src/model/mod.rs:185-196` | `duration.to_vec1()` + Rust scatter to build alignment matrix, then `Tensor::from_vec(...)` back. Smaller volume than #1/#2 but still a hard sync before decoder. |
| 4 | `src/model/mod.rs:65-70` | `VarBuilder::from_mmaped_safetensors(..., DType::F32, ...)` hard-codes F32 — no F16/BF16 path. |
| 5 | `src/model/predictor.rs:92, 102, 168`; `src/model/decoder.rs:87`; `src/model/text_encoder.rs:50` | Repeated `Tensor::new(eps, x.device())` and `Tensor::ones(...)` allocations inside hot normalization paths. Each is a Metal allocation + command. |
| 6 | `src/lib.rs:9-17`; `src/bin/speak.rs:100`; `src/bin/speak-server.rs:81, 395` | `Device::new_metal(0).expect("...")` hard-panics if Metal init fails; no `--device` runtime override; no logging of which device was selected. |
| 7 | `Cargo.toml:14-16` | candle is pinned at `0.8`. Newer Metal kernel coverage (`candle-metal-kernels 0.10.x`) likely fixes some of the ops we're working around. |

## 2. Phases

Order is picked so low-risk wins land first and each subsequent phase builds on a validated baseline. **Each phase is a separate commit.** Run `cargo test` and the relevant `*_check` binary before committing.

### Phase A — Scalar/constant allocation cleanup (item #5)

**Effort:** ~30-60 min. **Risk:** low.

Targets:
- `src/model/predictor.rs:88-95` (`instance_norm1d`)
- `src/model/predictor.rs:97-105` (`layer_norm_last_dim`)
- `src/model/predictor.rs:162-170` (`AdaLayerNorm::forward` — the `Tensor::ones(...)` at line 168)
- `src/model/decoder.rs:87` (`AdaIN1d` instance-norm — note `decoder.rs` already uses `gamma.ones_like()` at line 302, so the pattern exists)
- `src/model/text_encoder.rs:50`

Replace `&var.broadcast_add(&Tensor::new(eps as f32, x.device())?)?` with the scalar-affine equivalent. Candle supports scalar add via `tensor.affine(1.0, eps)` (multiplies by `1.0`, adds `eps`). Replace `Tensor::ones(gamma.shape(), DType::F32, x.device())?` with `gamma.ones_like()?` (already the pattern used in `decoder.rs`).

**Validation:** `cargo run --bin predict_duration_check` and `cargo run --bin decoder_fusion_check` produce identical numeric output to before. Both with and without `--features metal`.

**Not in scope:** fusing norm + affine into a single op; redesigning `AdaIN1d`/`AdaLayerNorm` interfaces.

### Phase B — Device ergonomics (item #6)

**Effort:** ~1 hr. **Risk:** low.

1. Change `src/lib.rs::default_device()` to **try Metal, fall back to CPU with a warning** instead of panicking:
   ```rust
   #[cfg(feature = "metal")]
   {
       match Device::new_metal(0) {
           Ok(d) => return d,
           Err(e) => tracing::warn!("Metal unavailable, falling back to CPU: {e}"),
       }
   }
   Device::Cpu
   ```
2. Add a `--device` flag to both `src/bin/speak.rs` and `src/bin/speak-server.rs` (values: `auto`, `cpu`, `metal`). `auto` calls `default_device()`. `cpu` forces `Device::Cpu`. `metal` returns an error if the binary wasn't built with the feature.
3. In both binaries, **log the resolved device** once at startup via `tracing::info!("device: {:?}", device)`.
4. In `src/bin/speak-server.rs:395`, the second `default_device()` call should use the device that was chosen at startup — thread it through rather than re-resolving.

**Validation:**
- `cargo build` (no features) — `--device metal` exits with a clear error message.
- `cargo build --features metal` — `--device cpu` runs on CPU even on a Mac.
- `cargo run --bin speak -- --text "test"` logs `device: Metal(...)` or `device: Cpu`.
- On a Mac where Metal init fails (simulate by temporarily breaking the feature), the binary now warns and continues on CPU rather than panicking.

**Not in scope:** picking a specific Metal device index (always 0); per-tensor device placement.

### Phase C — candle upgrade (item #7)

**Effort:** ~2-4 hr depending on API churn. **Risk:** medium — API breakage possible.

1. Bump `candle-core`, `candle-nn`, `candle-transformers` from `0.8` to the latest `0.9.x` first (not `0.10`); validate; then attempt `0.10.x`. Two-step lets us bisect breakage.
2. Run **every** `*_check.rs` binary in `src/bin/` with both `--features metal` and without — they're the parity gates. Compare outputs against the pre-upgrade run (capture them first into `/tmp/pre_upgrade_*.bin` via the existing `dump_tensor` machinery where available).
3. If any check exceeds its tolerance, roll back the version bump and document which op diverged in this spec under "Notes after upgrade".
4. Specifically check whether `Tensor::atan2` (or equivalent) became available — if so, note it for Phase D.

**Validation:** all `*_check.rs` pass under both feature configurations. `cargo test` clean. End-to-end `speak --text "hello world"` produces audio that subjectively matches the pre-upgrade output (and numerically matches to within tolerance — diff the WAV samples).

**Not in scope:** API surface changes inside our code beyond what the compiler forces.

### Phase D — STFT phase on-device (item #2)

**Effort:** ~half a day. **Risk:** medium — must produce numerically identical phase, since the trained `noise_convs` weights expect canonical `atan2` output in `[-π, π]`.

Approach (pick the first that passes validation):

1. **If candle (post-upgrade) exposes an on-device `atan2`:** replace `atan2_tensor` body with the candle op. No semantic change.
2. **If not:** implement `atan2` using existing ops via the standard piecewise formula:
   - Compute `atan(y/x)` (or `atan(x/y)` and flip) using a Taylor/Padé approximation in tensor form, or via `(y/x).atan()` if `atan` exists.
   - Apply quadrant corrections using sign tensors and broadcast-add.
   - Handle `x == 0` and `y == 0` boundaries to match `f32::atan2` (test: `atan2(0, 0) == 0`, `atan2(0, -0) == π`, `atan2(±0, +x) == ±0`, `atan2(±y, -0) == ±π/2`).
   - Confirm output is in `[-π, π]` element-wise.

The existing `zero_dc_and_nyquist_imag` workaround at `stft.rs:104-114` stays — it's correct and necessary.

**Validation:**
- Extend `src/bin/custom_stft_check.rs` to dump the **phase tensor** (not just the round-trip waveform), then diff against the PyTorch reference at element-level. Tolerance: max abs diff `< 1e-5` per element.
- Extend `src/bin/generator_check.rs` to dump the `har` tensor (input to `noise_convs`) before and after this change — they must match the pre-change Rust output to within `< 1e-6` (we're not changing the math, just the device).
- Both with and without `--features metal`.

**Not in scope:** replacing `atan2` with a different representation (cos/sin or normalized real/imag) — that would change conv inputs and require re-evaluation of the trained weights.

### Phase E — NSF source on-device (item #1)

**Effort:** ~1-2 days. **Risk:** high — most invasive change. **Hardest validation.**

The CPU reference in `src/model/source.rs` is verified working. Goal is a tensor-only implementation of `SineGen::forward_with_controls` that produces bit-identical output on CPU and numerically-equivalent output on Metal, eliminating the `to_vec1`/`Tensor::from_vec` round-trip.

Tensor mapping of the existing scalar loops:

- **Voiced mask** (`source.rs:90-95`): `uv = f0.gt(self.voiced_threshold)?` then cast to f32. Shape `[B, T, 1]`.
- **Harmonic indices** (`source.rs:96-103`): build a constant tensor `harmonic_scales = Tensor::from_vec((1..=dim).map(|h| h as f32 / sampling_rate).collect(), (1, 1, dim), device)`. Then `rad = f0.broadcast_mul(&harmonic_scales)?.fract()?`. Add `rand_ini` (broadcast over time) at `t == 0` only — easiest is to add the full `rand_ini` then zero everything except the first time step via mask, or add it inside the cumsum-prefix.
- **First linear interp** (`source.rs:107-113`, `interpolate_linear_btd` with scale `1/upsample_scale`): downsamples by `upsample_scale`. Can be done as `narrow` at strided indices + a final fractional blend, or as an explicit gather. **Verify the half-pixel convention** (`(t_out + 0.5)/scale - 0.5` per the existing code) — must match.
- **Cumulative phase** (`source.rs:117-128`): this is the load-bearing op. Use `tensor.cumsum(D::Minus2)?` (the sequence dim) — candle has `cumsum`. Then multiply by `2π * upsample_scale`.
- **Second linear interp** (`source.rs:129-130`, scale `upsample_scale`): upsample by `upsample_scale`. Note `upsample_nearest1d` in `src/model/upsample.rs` is nearest, not linear — you need a linear variant. Implement as `nearest_upsample → conv1d` with a small averaging kernel, or as explicit indexing + frac blend.
- **Sine + noise blending** (`source.rs:132-145`): `sine = phase.sin()?.broadcast_mul(&Tensor::new(self.sine_amp, device)?)?.broadcast_mul(&uv_expanded)?`. `noise_amp = uv * noise_std + (1-uv) * sine_amp/3`. `out = sine + noise_amp * noise_input`.

Implementation note: keep `forward_with_controls` accepting `rand_ini` and `noise` as `Option<&Tensor>` exactly as today — the validation harness depends on injecting deterministic values.

**Validation (this is the rigorous part):**
1. Capture reference: run `cargo run --bin source_check` on the current CPU implementation with fixed `rand_ini` and `noise`, dump the three output tensors (`sine_merge`, `sine_noise`, `uv`) to `/tmp/source_ref/*.bin`.
2. After the rewrite, re-run with identical inputs on CPU. Element-wise diff against `/tmp/source_ref` — tolerance: max abs `< 1e-5`, mean abs `< 1e-7`. **No exceptions.** If `cumsum` accumulates differently in candle than the existing `f64` Rust loop, you may need to keep that one accumulation in `f64` even on-device (via `cast → cumsum → cast`).
3. Re-run with `--features metal` and diff against the same reference.
4. Full end-to-end `speak --text "..."` audio diff: CPU vs CPU-pre-change must match within `< 1e-4` per sample.

**Not in scope:** changing the NSF algorithm itself (e.g. removing the half-pixel offset, swapping linear-interp for cubic). This is a pure CPU→GPU port.

### Phase F — F16 dtype path (item #4)

**Effort:** ~1 day including benchmarking. **Risk:** medium-high — numerical sensitivity in normalization/STFT.

1. Add `--dtype f32|f16|bf16` flag to both binaries. Plumb through `Kokoro::load` and `Kokoro::load_voice`. Default stays `f32`.
2. Wire the dtype into `VarBuilder::from_mmaped_safetensors` at `src/model/mod.rs:65-70`.
3. Identify ops that must stay F32 even when weights are F16 — at minimum: STFT (`src/model/stft.rs`), instance/layer normalization, the `cumsum` in the NSF source. Cast inputs up at module boundaries, cast back down for matmul/conv-heavy layers.
4. Benchmark: `time cargo run --release --bin speak -- --text "$(cat docs/specs/bench_text.txt)" --device metal --dtype {f32,f16}` (create a representative 30-second `bench_text.txt`). Record real-time factor before/after.
5. Numerical: full `*_check.rs` sweep must still pass at F16 with **relaxed** tolerance (`< 5e-3` mean abs per intermediate, `< 1e-2` per audio sample). Audio-level subjective check: no audible artifacts on at least three sentences with varied prosody.

**Validation:** F32 path remains unchanged numerically. F16 path is fast and audibly indistinguishable.

**Not in scope (this milestone):** BF16 (add the flag value; mark as experimental); per-layer dtype selection beyond the boundary casts above; quantization.

### Phase G — Deferred: alignment scatter on-device (item #3)

Volume is small (one tensor per forward, sized by phoneme count). Implement only if Phase E+F profile shows it as the next-largest sync point. Stub a TODO; do not block on it.

## 3. Cross-cutting expectations

- **One commit per phase.** Commit message format: `metal: phase X — <short>`. Body explains validation result (e.g. `max abs diff vs reference: 3.2e-6`).
- **No new abstractions** beyond what each phase needs. Don't introduce a `MetalContext` struct, a `DeviceManager`, or a `DType` newtype. Match Andrew's preference for surgical changes (per the existing port style in `src/model/`).
- **No comments explaining what the code does** — name things well. Only comment when there's a non-obvious *why* (e.g. "f64 accumulation matches Rust loop's precision" near the cumsum).
- **Do not modify `*_check.rs` semantics** — only *extend* them with dumps. The existing reference-comparison logic is the trusted oracle.
- After each phase, paste the validation output (max-abs / mean-abs diffs, benchmark numbers) into the conversation before moving on.

## 4. Stop conditions

Pause and ask Andrew (via this conversation) if:

- Any `*_check.rs` exceeds its tolerance after a change and root-causing it takes more than ~30 minutes.
- The candle upgrade in Phase C breaks more than 3 call sites in our code.
- The on-device `atan2` (Phase D) cannot match `f32::atan2` to `< 1e-5` element-wise.
- Phase E's NSF rewrite cannot match the CPU reference within tolerance and the diff is not isolated to the `cumsum` precision issue noted above.

## 5. Out of scope (this milestone)

- ONNX / CoreML backends.
- Per-sentence streaming synthesis (handled in m3-realtime).
- Multi-GPU / multi-device.
- Compile-time / runtime quantization (int8, int4).
- Custom Metal kernels written in MSL — we use candle's kernels only. If candle is missing an op, document it in this file under "Notes after upgrade" rather than writing MSL.

---

## Notes after upgrade

*(Implementer fills this in as work progresses — candle version landed at, ops that diverged, ops still missing on Metal, benchmark numbers per phase.)*
