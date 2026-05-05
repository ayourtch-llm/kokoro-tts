# G2P Decision

Milestone 1 uses a `Phonemizer` trait with a deliberately narrow `StubPhonemizer` for one canned phrase: `hello world`.
The stub returns IPA directly so model-stage validation can proceed without committing to frontend quality too early.
It must be removed before milestone 2.

The real native path should be espeak-ng behind the existing `espeak` Cargo feature, exposed through the same trait.
This keeps runtime native Rust plus a C library dependency and avoids a Python sidecar.
It will not match misaki's English quality: homographs, text normalization, and learned Kokoro-specific frontend behavior will be weaker.
The tradeoff is acceptable for milestone 1 because espeak-ng's IPA output overlaps Kokoro's training frontend fallback and the trait keeps a future misaki port swappable.

Do not silently drop unknown IPA symbols.
`Kokoro::phonemes_to_ids` now logs each unmapped phoneme before filtering it out; validation stage 2 should turn this into an exact receipt against `config.json`.
