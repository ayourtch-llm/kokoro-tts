# M3.5 — Acoustic Echo Cancellation (AEC)

**Audience:** codex (kokoro-tts side) + a Claude in nemotron-speech (ASR side).
**Status at handoff:** M3 phase 1+2+3 shipped (voice agent talks; sentence-streaming TTS through speakers). Textual echo dedup at the agent layer (`step 7: textual echo dedup at ASR-receive boundary`) handles full-sentence echoes but **fails on fragmented echoes** — ASR transcribes speaker output in 1-3 token chunks ("Eleph", "ants are capable of recogn", "izing themselves") and token-set Jaccard scores are below threshold for short fragments. Tested empirically; the loop became a 12-minute, 179-WAV elephant-fact regression.

The fix lives at the audio layer, not the text layer.

## 1. Goal

Cancel the speaker's audio from the microphone signal **before** ASR transcription. With AEC working, the user can:
- Speak naturally without lowering the volume.
- Interrupt the LLM mid-sentence and be heard correctly.
- The agent's own voice never reaches the LLM as a "user" turn.

This is the architecture used by Alexa/Echo/Siri: the TTS audio is shared as a "reference signal" to the ASR side, which subtracts it from the mic input.

## 2. Architecture

```
                      kokoro-tts speak-server                          nemotron-speech transcribe_live
                      ─────────────────────                            ─────────────────────────────
                       synthesize audio (24 kHz f32 mono)
                              │
                       ┌──────┴───────┬───────────────┐
                       ▼              ▼               ▼
                  cpal stream   save WAV (opt)   resample 24→16 kHz
                      │                                  │
                      ▼                                  ▼
                  speakers                    UDP datagrams (16k f32-LE mono)
                      │                                  │
                      ▼                                  ▼
                ── air & echo ──>           ┌───────────────────┐
                      │                     │   AEC kernel      │ ◄── (mic UDP from udp_mic_send)
                      ▼                     │ (option b first)  │
                  microphone                │                   │
                  (already                  └─────────┬─────────┘
                   streaming                          │
                   16k f32 LE                         ▼
                   to box:9999) ──────────►   cleaned 16 kHz audio
                                                      │
                                                      ▼
                                       existing transcribe pipeline
                                          (mel → encoder → RNN-T)
```

Wire format (one side change at a time, additive):
- **New: speak-server `--reference-out <host:port>`** publishes its audio as raw 16 kHz f32-LE mono PCM datagrams (same wire format as `udp_mic_send`). On the box, this can be a Tailscale or LAN address so the box-side ASR can receive it.
- **New: transcribe_live `--reference-listen <addr>`** binds another UDP socket for the reference stream.
- Existing `--udp-listen` for mic stays unchanged.

This means: same UDP/PCM convention as the existing mic pipeline, just a second input.

## 3. Two implementation phases

### Phase A — Plumbing + (b) basic AEC (this milestone)

1. **kokoro-tts speak-server (codex):**
   - Add `--reference-out <host:port>` flag.
   - On each datagram synthesized, after the existing playback queue enqueue: resample the 24 kHz f32 buffer to 16 kHz, chunk into ~20ms (320-sample) UDP datagrams of f32-LE, send to target.
   - Resampling: linear interpolation is sufficient (we're sending it for echo cancellation, not pristine playback). The phase-3 audio module already has resampling for cpal output — pull the same helper into `synthesis.rs` or `audio.rs` for reuse.
   - The reference stream is **timed against the playback clock** — i.e. when speakers play sample N at time T, the reference packet for sample N is sent ~within the same time window. The simplest implementation: send to the network on the same thread that pushes to the playback queue. Some buffering is fine — the AEC side will align via cross-correlation.
   - Wire format: raw f32-LE bytes, 320 samples per UDP datagram (matches `udp_mic_send`'s 20ms chunking; check the latter for exact byte layout).
   - When `--reference-out` is unset: no extra work, behavior unchanged.

2. **nemotron-speech AEC integration (pty-9 after /clear):**
   - In `transcribe_live`, add `--reference-listen <addr>` flag (defaults to off).
   - When set, bind a second UDP socket; spawn a task that consumes reference samples into a ring buffer (~3 seconds of history is plenty).
   - Add an `AecKernel` trait + initial impl `SpectralSubtractionAec` (option b):
     - On each ~32ms (512-sample) mic frame: compute cross-correlation with reference history to find the propagation-delay offset. Update a rolling estimate (smoothed exponentially).
     - At the estimated delay, take the matching reference window. Compute a scalar gain: `gain = <mic, ref_aligned> / <ref_aligned, ref_aligned>` (least-squares-y).
     - Output: `cleaned = mic - gain * ref_aligned` (in time domain). Optionally compute in frequency domain (windowed FFT, magnitude subtraction with floor) — this is more robust to phase drift but more code. Time-domain version is fine for v1.
     - On reference-silent intervals: pass through mic unchanged.
   - Cleaned samples → existing mel/encoder/RNN-T pipeline.
   - Keep mic source unchanged for backward compat (no `--reference-listen` = current behavior).

3. **Validation strategy:**
   - **Synthetic test**: in nemotron-speech, write a tool that takes two WAV files (mic-with-echo, reference). Run AEC kernel offline. Output cleaned WAV. Inspect by ear + by passing cleaned WAV through the existing `transcribe` binary — should not produce the echo's text.
   - **Live test (Andrew)**: same elephant-fact prompt as before. With AEC on, the loop should not cascade. Andrew should be able to interrupt without lowering volume.
   - **Regression**: with `--reference-listen` unset, transcribe_live behaves identically to before.

### Phase B — Adaptive AEC (option c, separate session)

Same plumbing as Phase A. Swap `SpectralSubtractionAec` for `NlmsAec` (or `RlsAec`), 256-tap adaptive FIR filter modeling the room impulse response. Adds proper double-talk detection. ~200-400 LoC additional. Multi-day care: step size, filter length, divergence detection. Out of scope for this milestone — separate spec when ready.

## 4. Don't do

- **Don't run the AEC kernel inline on the mic-source thread** — keep it in its own task with a small bounded queue. Audio thread starvation is a real failure mode.
- **Don't share state between AEC instances** — one `AecKernel` per `transcribe_live` invocation.
- **Don't try to send the reference at 24 kHz** — must match mic rate (16 kHz), otherwise time-alignment math gets weird.
- **Don't pipeline (b) and (c) yet** — ship (b) as the kernel, validate, only then consider stacking.
- **Don't break the existing transcribe_live behavior** — `--reference-listen` is opt-in.

## 5. Coordination

- Codex (pty-10) ships kokoro-tts side: `--reference-out` flag, 24→16 kHz resample, UDP send. Self-contained, ~50 LoC.
- Pty-9 (after `/clear` — they have 580k stale tokens) ships nemotron-speech side: `--reference-listen`, AEC kernel + integration. The kernel is the meat (~150-200 LoC).
- pty-1 (me) coordinates and runs the live elephant-fact retest after both sides land.
- Same friendly-collaboration mode + format-only-commit pattern as M2/M3.

## 6. Receipts

| Stage | Description | Acceptance |
|---|---|---|
| A.1 | speak-server `--reference-out` ships | Datagrams visible via `tcpdump`/`nc -ul` on the target port, 16 kHz f32 PCM verified by writing to WAV |
| A.2 | transcribe_live `--reference-listen` binds + buffers | Stream consumed without blocking mic path |
| A.3 | AEC kernel (option b) | Synthetic test: cleaned WAV transcribes to silence (or near-silence) when input is "speaker-only"; transcribes to user words when input is "user + speaker" |
| A.4 | Live retest | "30 elephant facts" prompt: agent should not feedback-loop, Andrew can interrupt without lowering volume |
| B | NLMS adaptive kernel | Future spec |
