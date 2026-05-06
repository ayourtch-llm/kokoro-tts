# M3 — Realtime voice loop

**Audience:** the implementing instance (codex / pty-10), with pty-9 reviewing.
**Status at handoff:** M1 (Kokoro model port) and M2 (native Rust G2P) are shipped. End-to-end pipeline `text → 24 kHz mono WAV` works. This milestone makes it feel like a real voice loop: speak through speakers, integrate with the voice agent, stream incrementally, and add voice variety.

## 1. Goal

Close the voice loop on the user's laptop:

> mic → ASR (nemotron-speech) → LLM (llama-server / OpenAI-compat) → text → kokoro-tts → speakers

Each phase below is independently shippable. Order is `3 → 1 → 2 → 4` per Andrew's call, picked so the most rewarding/concrete pieces land first and the engineering-heavy streaming work is bracketed by working baselines on either side.

## 2. Phases

### Phase 3 — Speaker playback (in this milestone, ship first)

**Goal:** `cargo run --release --bin speak -- --text "..." --play` plays the synthesized audio through the default output device, blocking until done.

Concrete deliverable:

- New module `src/audio.rs` (or `src/playback.rs`) exposing `pub fn play_samples(samples: &[f32], sample_rate: u32) -> anyhow::Result<()>`. Built on the [`cpal`](https://docs.rs/cpal) crate (cross-platform, lightweight, no decoder deps we don't need). Function blocks until the queued samples have been consumed by the output device, then returns. Handles default-device discovery and stream-format negotiation.
- `--play` flag in `speak.rs`: after synthesis, calls `play_samples(...)`. `--play` and `--out` are independent — both can be set (save AND play); either or neither is fine.
- Add `cpal` to `Cargo.toml`. Don't pull in `rodio` — its decoder/mixer abstractions are irrelevant for us and add transitive bloat.
- The playback function should be **reusable** (callable from a future streaming daemon, not baked into a CLI flag). Expose via `src/lib.rs` so other binaries / future daemon modes can use it.

Validation:
- `cargo run --bin speak -- --text "Hello, world!" --play` produces audible "Hello world" through laptop speakers.
- `cargo run --bin speak -- --text "Hello!" --play --out /tmp/test.wav` plays AND saves the WAV.
- No regression on existing `speak --text` behavior.
- `cargo test` clean. Format-only commit if rustfmt finds anything.

**Not in scope:** input device selection, volume control, gapless streaming, latency tuning.

### Phase 1 — Voice agent integration (after phase 3)

**Goal:** the nabu-agent-rust LLM agent's output gets spoken through the speakers in roughly real time per assistant turn.

Concrete deliverable:

- The voice agent (~/rust/nabu-agent-rust) currently emits LLM text deltas to stdout. Add an output mode that, on each completed assistant turn (or each completed sentence — pick the more natural unit), invokes kokoro-tts to synthesize and play.
- Two architecture options (pty-9 + codex pick):
  - (a) **Subprocess**: agent spawns `speak --text "..." --play` per turn, blocks until exit. Simple, no IPC. Latency = synthesis time per turn.
  - (b) **Long-lived TTS daemon**: kokoro-tts runs as a daemon listening on UDP/TCP, agent forwards text chunks to it. Lower per-turn overhead (no model-load per turn), enables streaming in phase 2.
- Recommend (b) — it sets up phase 2 cleanly. Phase 3's `play_samples` becomes the daemon's playback primitive.
- Wire format: same UDP plain-text-newline format the voice agent already uses for ASR ingest. Agent listens for ASR (port A), forwards to LLM, then sends LLM output text to TTS (port B). TTS daemon synthesizes + plays.
- Daemon binary: `src/bin/speak-server.rs` — listens on `--listen <host:port>`, on each datagram synthesizes + plays. Reuses the existing model-loading + chunked-synthesis infrastructure.

Validation:
- mic → ASR → agent → kokoro-tts → speakers loop works end-to-end. User asks a question, hears the answer.
- The full voice agent integration test from the original voice-agent build (overnight) replays cleanly with TTS audio out.

**Not in scope:** mid-turn interruption (already exists at the agent level via cancellation token; just don't break it).

### Phase 2 — Streaming TTS (after phase 1)

**Goal:** assistant audio starts playing before the LLM has finished generating the response, reducing perceived latency from "model finishes → audio starts" to "first sentence finishes → audio starts".

Concrete deliverable:

- The TTS daemon from phase 1 currently synthesizes full text per datagram. In phase 2, it receives partial text deltas (or full text but begins playback per sentence as soon as the sentence boundary is detected).
- Two granularity options:
  - **Sentence-level streaming**: parse incoming text for sentence boundaries (`.!?\n`), synthesize each as it completes, append to playback queue. Natural; simple.
  - **Chunk-level streaming**: synthesize fixed-size chunks of ~20 phonemes regardless of sentence boundaries. Lower latency but worse prosody (phrase-level intonation lost). Probably not worth it.
- Recommend sentence-level. Latency-to-first-audio = (LLM generates first sentence) + (synthesize first sentence). Typical: ~1-2 seconds total.
- The blocking `play_samples` from phase 3 becomes a long-lived audio output stream that accepts samples from a `mpsc::Receiver<Vec<f32>>`. Synthesizer thread feeds, playback thread consumes.

Validation:
- Round-trip latency: from "user finishes speaking" to "TTS starts playing" should be under ~3 seconds for typical short LLM answers (1-2 sentences).
- Audio doesn't gap or click between sentences (the 80 ms inter-sentence padding from `speak.rs` chunked-synthesis applies here too).
- Fast LLM turns don't overrun the audio queue (back-pressure handled).

**Not in scope:** mid-stream voice/speed change, dynamic interruption mid-audio, sub-sentence streaming.

### Phase 4 — Voice variety (after phase 2)

**Goal:** more voices than just `af_heart`. Pick a voice per invocation.

Concrete deliverable:

- Upstream Kokoro ships ~50 voices in `voices/*.pt`. Currently we vendor one (`af_heart`).
- Extend `download-model` to fetch a curated subset (5-10 voices: a few American female, American male, British, etc.).
- Run them through `convert_weights.py` to produce `models/voices/<name>.safetensors` for each.
- `speak --voice <name>` already works (takes a path). Just need shorthand: accept `--voice af_bella` and resolve to `models/voices/af_bella.safetensors`.
- TTS daemon (phase 1) accepts an optional voice tag in the wire format (or via separate CLI flag), letting the agent pick voice per turn (or globally).

Validation:
- `cargo run --bin speak -- --text "..." --voice af_bella --play` plays in the new voice.
- Voice files are small (~500 KB each); 5-10 voices add ~5 MB to repo size — fine.

**Not in scope:** voice cloning from arbitrary audio, multi-speaker per utterance, voice morphing.

## 3. Validation pattern (cross-phase)

Same pattern as M1 and M2:
- Each phase commits independently. Format-only commit if rustfmt finds churn (per Andrew's pattern).
- For phases that produce audio, round-trip ASR through nemotron-speech as a sanity check (pty-1 owns this cross-repo step).
- For phase 3, listening through speakers is the validation (no automated test possible without a microphone-capture loop, which is out of scope).

## 4. Don't do

- Don't pull in `rodio` for phase 3 — `cpal` direct is the lightweight choice.
- Don't break `--out` while adding `--play` — both must work independently.
- Don't bake `play_samples` into the CLI binary; expose as lib for phase 1/2 reuse.
- Don't pre-train a streaming-aware model — phase 2 is just chunk-level coordination, not a model rework.
- Don't ship phase 2 before phase 1 — phase 2 needs the daemon architecture phase 1 establishes.
- Don't skip the per-turn round-trip test in phase 1 — first integration is where wire-format mismatches surface.

## 5. Coordination

- Codex implements (per phase). pty-9 reviews. pty-1 owns cross-repo round-trip ASR + voice-agent integration coordination.
- Same friendly-team-collaboration mode as M1/M2. Substantive commit + format-only commit pattern. Receipts as you go.
- After phase 4, M3 stamp commit closes the milestone.
