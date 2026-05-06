# Native Rust G2P (text → IPA) — Implementation Brief

**Audience:** the implementing instance (codex / pty-10), with pty-9 reviewing.
**Reviewers:** pty-1 (drafted this), pty-9 (will refine + commit corrections inline as bugs surface, same pattern as kokoro-rust-port.md).
**Status at handoff:** scaffold has `Phonemizer` trait + `StubPhonemizer` (hardcoded for "hello world") + `--phonemes` pass-through in `speak.rs`. `--features espeak` flag exists but does nothing. No real text → IPA path.

## 1. Goal

**Milestone target:** `cargo run --release --bin speak -- --text "..."` produces an intelligible 24 kHz WAV from any reasonable English input, using only Rust + embedded data — no subprocess, no Python at runtime, no neural model beyond Kokoro itself.

Quality bar: **comparable to misaki + espeak-ng en-us** on a curated 100-sentence corpus. Not bit-identical (misaki and espeak-ng themselves disagree on edges) — target ≥95% phoneme-character agreement and 100% intelligibility via ASR round-trip.

Not in scope: non-English, multi-speaker voice selection, SSML, pitch/rate hints. Those are M3+.

**Hard constraint:** native Rust + embedded data. No subprocess (no `espeak` shell-out). No HF Hub / network at inference. No Python. The validation pipeline uses Python references; runtime does not.

## 2. Architecture

```
text input
 │
 ▼
text normalizer            # numbers, dates, abbreviations, units → words
 │
 ▼
tokenizer + sentence split # split on .?! preserving boundary tokens
 │
 ▼  per token:
lexicon lookup ──► hit?  ──yes──► ARPAbet → IPA (with stress mapping)
                    │
                    no
                    ▼
                  homograph disambiguator   # for words with multiple entries
                    │
                    ▼
                  OOV LTS rules             # letter-to-sound fallback
 │
 ▼
IPA assembly + punctuation passthrough
 │
 ▼
output: IPA string compatible with Kokoro vocab
```

## 3. What's already there (don't redo)

- `src/phonemizer.rs` — `Phonemizer` trait + `StubPhonemizer` + `MILESTONE_TEST_PHONEMES` constant. Trait is minimal: `fn phonemize(&self, text: &str) -> Result<String>`. Keep this surface; add new impls.
- `src/bin/speak.rs` — accepts `--text` (uses configured phonemizer) or `--phonemes` (raw passthrough). Default phonemizer should switch from `StubPhonemizer` to the new full one once stage 1 lands.
- `Cargo.toml` — has the `espeak` feature flag declared but unwired. Either repurpose it (make it gate the new impl) or drop it.
- `models/config.json::vocab` — 114-entry IPA-only character vocab. **The output of our G2P must use only these characters** (filtering happens silently in `Kokoro::phonemes_to_ids` — there's already a warning when chars are dropped, so OOV-vocab will be visible at synthesis time).

Vocab characters that matter:
- Letters: standard a-z subset + extended IPA (`æ ç ð ø ŋ œ ɐ ɑ ɒ ɔ ɕ ɖ ə ɚ ɛ ɜ ɟ ɡ ɣ ɤ ɥ ɨ ɪ ɯ ɰ ɲ ɳ ɴ ɸ ɹ ɻ ɽ ɾ ʁ ʂ ʃ ʈ ʊ ʋ ʌ ʎ ʒ ʔ ʝ ʣ ʤ ʥ ʦ ʧ ʨ`)
- Stress: `ˈ` (primary), `ˌ` (secondary)
- Length: `ː`
- Diacritics: `̃` (nasalization), `ʰ` (aspiration), `ʲ` (palatalization)
- Punctuation: ` ,.!?;:()"'` and the curly quotes `“”`, em-dash `—`, ellipsis `…`, arrows
- Some less-common: `β θ χ ᵊ ᵝ ᵻ ꭧ`

Dump the vocab keys at startup in the new phonemizer impl and assert the ARPAbet→IPA table only emits chars that exist.

## 4. Stage breakdown

Each stage is its own commit. Each ships value on its own — don't conflate. Validation receipts go in §10 as they land.

### Stage 1 — Misaki gold lexicon + CMUdict fallback (architecture revised after empirical findings)

**Goal:** lookup-only G2P with two tiers. Most common English words → bit-exact misaki IPA; long-tail words → CMUdict-derived IPA via ARPAbet. OOV → stage 5 fallback.

**Architecture revision (from codex's stage-1 sanity dump against actual misaki output):**

The original draft of this stage was "CMUdict + ARPAbet→IPA mapping." Codex ran a misaki sanity dump and found two things that reshape stage 1:

1. **Misaki's `us_gold.json` lexicon (~13k words) is already in IPA** in *exactly* the form Kokoro was trained on. Going `CMUdict → ARPAbet → IPA` is lossy at the conversion step (multiple ARPAbet symbols map to the same or near-IPA, conventions differ); going `misaki-gold → IPA` is identity. Top-13k English words cover roughly 93–95% of running text.
2. **Misaki strips length marks (`ː`) and uses ligature affricates (`ʧ`, `ʤ`) in US English**, contrary to the espeak en-us convention the original spec table was based on. Bit-matching misaki is what the trained model expects.

**New stage-1 lookup order:**

```
1. Strip casing + ASCII-normalize → key
2. Try misaki gold lexicon (data/misaki_us_gold.json, ~13k entries, IPA-already)
   ├─ hit:  return IPA verbatim
   └─ miss: ↓
3. Try CMUdict (data/cmudict-0.7b, ~134k entries, ARPAbet)
   ├─ hit:  ARPAbet → IPA via the table below, return
   └─ miss: ↓
4. Stage-5 OOV fallback (literal spellout in 5a; LTS rules in 5b)
```

Files:
- `data/misaki_us_gold.json` — copy from misaki upstream (Apache 2.0, vendor freely). Embed via `include_str!`. Parse once at first phonemize call into `HashMap<String, String>`.
- `data/cmudict-0.7b` (~3.6 MB). Embed via `include_str!`. Strip comments + alternative-pronunciation lines (those ending with `(2)`, `(3)`) — keep first per word as default; defer routing to stage 4.
- `src/phonemizer/lexicon.rs` — both lexicons + the lookup-order glue.
- `src/phonemizer/arpabet.rs` — ARPAbet → IPA table for the CMUdict fallback path. ~40 entries.

**ARPAbet → IPA mapping (US English, MATCHING MISAKI — verified empirically by codex stage-1 sanity dump):**

Table updated from the earlier draft per misaki's actual emission pattern. **No length marks. Affricates as ligatures.**

| ARPA | IPA (unstressed / stressed) | ARPA | IPA |
|---|---|---|---|
| **Long vowels (NO length mark `ː` in US misaki — stress mark only):** | | | |
| AA | ɑ / ˈɑ / ˌɑ | IY | i / ˈi / ˌi |
| AO | ɔ / ˈɔ / ˌɔ | UW | u / ˈu / ˌu |
| **AH (vowel-quality split per stress):** | | | |
| AH | ə (unstressed) / ˈʌ / ˌʌ | | |
| **ER (vowel-quality split per stress, see note):** | | | |
| ER | əɹ (unstressed) / ˈɜɹ / ˌɜɹ | | |
| **Short vowels (stress mark only):** | | | |
| AE | æ | IH | ɪ |
| EH | ɛ | UH | ʊ |
| **Diphthongs (misaki uses SINGLE CAPITAL LETTERS, not two-char IPA):** | | | |
| AW | W (U+0057) | OW | O (U+004F) |
| AY | I (U+0049) | OY | Y (U+0059) |
| EY | A (U+0041) | | |
| **Consonants (stress-invariant; affricates as LIGATURES):** | | | |
| B | b | N | n |
| CH | **ʧ** (ligature, U+02A7) | NG | ŋ |
| D | d | P | p |
| DH | ð | R | ɹ |
| F | f | S | s |
| G | ɡ | SH | ʃ |
| HH | h | T | t |
| JH | **ʤ** (ligature, U+02A4) | TH | θ |
| K | k | V | v |
| L | l | W | w |
| M | m | Y | j |
| (silence) | (space) | Z | z |
| | | ZH | ʒ |

**Notes (verified against misaki US output by codex's stage-1 sanity dump):**

- **No length marks `ː` in US English mode.** Misaki emits "speed" = `spˈid` (not `spˈiːd`), "father" = `fˈɑðəɹ`, "thought" = `θˈɔt`. The earlier draft of this spec said long vowels carry `ː` when stressed — that's the espeak `--ipa=3` convention but NOT misaki's US convention. Drop the length mark for US. (Length mark is in Kokoro's vocab so synthesis won't break either way, but matching the training distribution is safer.)
- **Affricates as ligatures (`ʧ` U+02A7, `ʤ` U+02A4), not two-char.** Misaki emits "church" = `ʧˈɜɹʧ` and "judge" = `ʤˈʌʤ`. Both forms are in Kokoro's vocab; ligatures match what the model saw at training. The earlier draft recommended two-char — wrong; corrected.
- **Diphthongs as single capital letters, not two-char IPA** (verified by direct read of misaki's us_gold.json, 38k+ entries): misaki uses `O` for OW (12,619 entries — "hello"=`həlˈO`, "go"=`ɡˌO`, "no"=`nˈO`), `I` for AY (11,617 — "sky"=`skˈI`, "high"=`hˈI`), `A` for EY (11,558 — "face"=`fˈAs`, "day"=`dˈA`), `W` for AW (2,051 — "how"=`hˌW`, "now"=`nˈW`), `Y` for OY (971 — "boy"=`bˈY`, "toy"=`tˈY`). All five capital letters are in Kokoro's vocab (ids 24/25/31/39/41). The earlier draft of this spec used two-char IPA (`oʊ`/`aɪ`/`eɪ`/`aʊ`/`ɔɪ`) — wrong; corrected. The model accepts both forms (milestone-1 ASR succeeded with `oʊ` for "hello"), but bit-match with misaki uses the single capitals.
- **AH stays special** — vowel quality changes with stress (`ʌ` ↔ `ə`), not length. AH0 → ə, AH1 → ˈʌ, AH2 → ˌʌ.
- **ER renders as a vowel + r split, EMPIRICALLY CONFIRMED across many samples.** Misaki:
  - Stressed (ER1/ER2): `ɜɹ` — "world"=`wˈɜɹld`, "church"=`ʧˈɜɹʧ`, "bird"=`bˈɜɹd`, "work"=`wˈɜɹk`, "first"=`fˈɜɹst` — never `ɜː`, never `ɚ`.
  - Unstressed (ER0): `əɹ` — "father"=`fˈɑðəɹ`, "teacher"=`tˈiʧəɹ`, "doctor"=`dˈɑktəɹ`, "mother"=`mˈʌðəɹ`, "butter"=`bˈʌɾəɹ` — never `ɚ` in any sampled word.
  - The `ɚ` ligature is in Kokoro's vocab but misaki US doesn't appear to use it. Drop it from the fallback table.
  - Bonus quirk: "butter" emits flap-T `ɾ` instead of `t`. Misaki's normalizer is doing American flap-T phonology; that's a stage-3-or-beyond concern, not stage 1's lexicon.
- **Stress mark placement**: `ˈ`/`ˌ` immediately precedes the IPA vowel (or the first vowel of a diphthong), not the consonant cluster. CMUdict marks the stressed *vowel* (AA1, IY2, etc.); when emitting, prepend the stress char to the vowel's IPA.
- **The misaki gold path bypasses this table entirely.** ARPAbet→IPA is only exercised when CMUdict is hit but misaki isn't — i.e., the long tail. So the table's correctness matters for less-common words, but the *common* case (top-13k) is bit-exact regardless of table edge cases.

Punctuation: pass `, . ! ? ; :` through unchanged. Drop other punctuation (except curly quotes / em-dashes / ellipses which are in vocab — pass through, let the model handle). Insert space between successive words.

**Validation:**
- `tools/reference_phonemize_lexicon.py` — generates expected IPA from misaki directly (since it's our gold standard) plus phonemizer/espeak as a secondary reference for OOV-from-misaki words. Curated test set of ~50 common in-misaki words + ~30 misaki-OOV-but-CMUdict-hit words.
- `src/bin/lexicon_check.rs` — runs Rust phonemizer on the test set; for misaki-hit words target is **100% bit-exact match**, for CMUdict-fallback words target is ≥95% character-level match (some minor ARPAbet-conversion divergence is acceptable since misaki itself is the gold).
- `cargo test` smoke test: phonemize "hello world" → must match `MILESTONE_TEST_PHONEMES` exactly. **The existing constant `"həlˈoʊ wˈɜɹld"` differs from misaki's actual emission — verified.** Misaki gold has "hello"=`həlˈO` (capital O, not `oʊ`) and "world"=`wˈɜɹld` (matches the constant). Combined misaki-correct value: **`"həlˈO wˈɜɹld"`**. The model is robust enough to accept both forms (milestone-1 ASR succeeded with `oʊ`), but for bit-match with misaki the constant should be updated. Stage 1 should: (a) update `MILESTONE_TEST_PHONEMES` to the misaki-correct form, (b) re-run kokoro-tts speak + ASR round-trip with the new constant to confirm "Hello world." is still transcribed, (c) commit.

**Commit message:** `g2p stage 1: misaki gold + CMUdict fallback + ARPAbet→IPA`

### Stage 2 — Punctuation, sentence boundaries, prosody

**Goal:** correct prosodic phrasing. Long input gets split into sentences; punctuation lands in the right place for Kokoro's prosody predictor.

- Recognize sentence boundaries on `. ! ? \n\n` (with abbreviation guards: `Mr.` `Dr.` `etc.` are not sentence-ends).
- For each sentence: phonemize independently, accumulate to one IPA string. Synthesis can either feed the whole concatenated IPA in one shot (if under `max_position_embeddings = 512`) or call `Kokoro::forward` per sentence and concat WAVs (cleaner audio prosody, small click risk between segments — favor whole-input feed unless we hit length limits).
- Keep `, ; :` for intra-sentence pauses. Drop quotes (vocab has them but they don't affect pronunciation).
- Curly quotes (`“ ”`), em-dashes (`—`), ellipses (`…`) are in vocab — pass through and let the model handle prosody.

**Validation:**
- Test set of ~10 multi-sentence inputs. Diff Rust output against Python phonemizer's per-sentence output reassembled.
- Listen test: synthesized output of "Hello. How are you? I'm fine, thanks!" should have audible pauses at the right places.

**Commit message:** `g2p stage 2: punctuation + sentence boundaries`

### Stage 3 — Text normalization

**Goal:** numerals, dates, abbreviations, units, etc. → spoken words before lexicon lookup.

Sub-features (each as its own sub-commit if they grow):

1. **Cardinal numbers**: `82` → "eighty two", `1234` → "one thousand two hundred thirty four". Range of 0 to ~10⁹. Negatives handled.
2. **Decimals**: `3.14` → "three point one four"; `0.5` → "zero point five".
3. **Ordinals**: `1st` `2nd` `3rd` `4th` → "first" "second" "third" "fourth".
4. **Years**: `2026` → "twenty twenty six"; `1999` → "nineteen ninety nine"; `2008` → "two thousand eight".
5. **Money**: `$5` → "five dollars", `€5` → "five euros", `$5.50` → "five dollars fifty cents".
6. **Time**: `3:45` → "three forty five"; `3:00 PM` → "three P M". (Don't try to be too clever here — most listeners will accept "three forty five P M" for 3:45 PM.)
7. **Dates**: `2026-05-06` → "May sixth twenty twenty six". `5/6/2026` is ambiguous (US vs EU); default to US (May 6th); add a config knob for EU later.
8. **Common abbreviations**: `Mr.`/`Mrs.`/`Ms.`/`Dr.` → titles spoken; `St.` → "Saint" or "Street" (context-dependent — punt: always "Saint" for now); `e.g.` → "for example"; `i.e.` → "that is"; `etc.` → "et cetera"; `vs.` → "versus".
9. **Acronyms**: heuristic — if all-caps and no vowels (or a 2-3 char all-caps with consonants), spell letter-by-letter (`FBI` → "F B I"). If pronounceable (`NASA`, `RADAR`), pronounce as a normal word.
10. **Units**: `kg` → "kilograms", `km` → "kilometers", `mph` → "miles per hour", `°C` → "degrees Celsius". Pluralize based on preceding number if any.

This stage is the longest tail — there are always more cases. Aim for 95% coverage of normal English text and call it shipped. Stage 5 will catch the rest as OOV.

**Validation:**
- `tools/reference_normalize.py` — uses NumberToWords from `num2words` Python lib for numbers, plus a fixed abbreviation dict. Generate ~100-200 test pairs spanning all sub-features.
- `src/bin/normalize_check.rs` — runs Rust normalizer, diffs textually. Target: 100% match on the curated set; document any deliberate divergences.

**Commit cadence (firmer than earlier draft):** **split per sub-feature, one commit each.** Don't land all 10 sub-features in one commit — the diffs become unreviewable and a regression in #6 (time) will be hard to bisect from a fix in #9 (acronyms). Suggested order — start with the highest-coverage cases: 1 (cardinal numbers) → 8 (abbreviations) → 4 (years) → 5 (money) → 9 (acronyms) → rest. The first three cover ~80% of normal English text where normalization actually fires.

**Commit message:** `g2p stage 3.<N>: <feature>` per sub-feature.

### Stage 4 — Homograph disambiguation

**Goal:** words with multiple pronunciations get the right one based on context.

Common English homographs that must be handled:
- Tense: `read` (past=red, present=reed), `lead` (verb=leed, noun=led).
- Part-of-speech: `live` (verb=liv, adj=lyve), `wind` (noun=wind, verb=wynd), `bow` (noun-tie=boh, verb-bend=baw), `tear` (rip=tair, drop=teer), `wound` (injury=woond, past=waund), `bass` (fish=bass, music=bayss), `close` (verb=kloze, adj=klohs), `present` (noun=PREZ-ənt, verb=prez-ENT), `record` (noun=REK-ord, verb=re-KORD).

Approach (pragmatic, not perfect):
- Hardcode ~30-50 most common homograph entries with a tiny rule each based on **previous word part-of-speech**.
- POS tagging: use a minimal ruleset (regex-based — last 100 most common verbs/nouns/adjectives; otherwise use suffix heuristics: `-ing` → verb, `-ed` → past verb, `-ly` → adverb, `-tion` → noun).
- This won't be perfect; document known limitations. Target: ≥80% accuracy on a curated homograph test set.

**Validation:**
- `tools/reference_homograph.py` — uses a pretrained spaCy or NLTK POS tagger for ground truth. Test set of ~50 sentences using each tracked homograph in both senses.
- `src/bin/homograph_check.rs` — diffs Rust output. Target: ≥80% agreement (allowing for genuine ambiguity in 1-2 sentences).

**Commit message:** `g2p stage 4: homograph disambiguation`

### Stage 5 — OOV letter-to-sound rules

**Goal:** for words not in CMUdict, emit reasonable IPA based on letter patterns. The "long pole."

Approach options (pick one — pty-9 / codex to advise):

**Option A: hand-written LTS rules** (Festvox-style). ~100-200 rules covering: silent-e, ph→f, ch→tʃ, ck→k, qu→kw, common prefixes (un-, re-, pre-), common suffixes (-tion, -ing, -ed, -ly), common letter combinations, simple syllable splitting + stress-on-first-syllable default. Bounded effort (~6-10h codex), "good enough" for most novel English words. Easiest to debug.

**Option B: port espeak-ng's English LTS rules**. The `*_dict` data files compile to a binary trie. Need to either port the compiler or pre-compile and ship the binary table + an interpreter. **License caution:** espeak-ng is GPLv3. If we link or vendor its rule data we inherit GPL. Not a fit unless we change kokoro-tts's license.

**Option C: small neural G2P model**. Train a tiny seq2seq transformer on CMUdict to predict ARPAbet from graphemes. Could be ~1MB of weights, run on candle. Best quality for OOV but real training cost; risk of weird hallucinations.

**Recommendation: A, but ship in two stages.**

- **Stage 5a (must ship first, ~30 min):** OOV fallback = literal letter spellout. Each unknown letter → its IPA "name pronunciation" (`a` → `eɪ`, `b` → `biː`, `f` → `ɛf`, etc.). Crude but produces *something* for every input — meaning stage 6 integration can run and ASR round-trip can validate the lexicon path on real text without being blocked by LTS rule quality. This is the analog of pty-9's StubPhonemizer for the OOV path.
- **Stage 5b (the real work, 6-10h):** hand-written Festvox-style rules per option A. Replace the spellout fallback with rule-based pronunciation. Validate against the ≥70% target.

This split means stages 6 + end-to-end ASR round-trip can land *before* stage 5b is done, gated on stage 5a only. Useful because the round-trip on a 100-sentence corpus is what tells us whether the *pipeline* works, separate from how good the LTS rules are. If 5a + lexicon get us to 90% intelligibility (likely — most English text words are in CMUdict), 5b becomes polish rather than a blocker.

**On options B and C** (kept here for the record):

- **B (vendor espeak-ng's LTS) is a real legal hazard, not a gray area.** espeak-ng is GPLv3. Linking *or* statically embedding its compiled rule data makes kokoro-tts GPLv3. There is no "we just used the data not the code" exception under GPL — derivative works of GPL data are GPL. Skip.
- **C (small neural G2P)** would be ~1MB of weights trained on CMUdict, runnable on candle. Real training cost (~half a day on a GPU + dataset prep) and risk of hallucinated pronunciations on rare patterns. Worth revisiting in M3 if rule-based stage 5b plateaus below acceptable, but not for M2.
- **No mature pure-Rust alternative exists** that I know of. (Searched: no crate on crates.io provides English G2P at production quality. `ttssrust` and similar are tiny demos.)

**Validation:**
- `tools/reference_oov.py` — phonemizer (espeak backend) on a curated ~100-word OOV test set: technical terms (`PyTorch`, `Kubernetes`), proper nouns, made-up words, rare English words.
- `src/bin/oov_check.rs` — Rust OOV-only path (force-skip lexicon to test the rules). Target: ≥70% character-level agreement vs espeak. Imperfect is acceptable; we just need not-broken for arbitrary text.

**Commit message:** `g2p stage 5: OOV letter-to-sound rules`

### Stage 6 — Integration + default-phonemizer wiring

After stages 1–5 land:
- Replace `StubPhonemizer` as the default in `speak.rs` with the full pipeline.
- Drop the `--features espeak` flag (or repurpose to allow optional espeak shell-out for diagnostics).
- Add `cargo run --release --bin speak -- --text "any English"` as a working command.
- End-to-end ASR round-trip on a 100-sentence corpus to confirm intelligibility.

**Commit message:** `g2p: wire full pipeline as default phonemizer`

## 5. Validation infrastructure (cross-stage)

Same pattern as the model port:

- Each stage has `tools/reference_<stage>.py` producing golden output.
- Each stage has `src/bin/<stage>_check.rs` running Rust impl + comparing.
- Per-stage thresholds in §10 receipts table.
- Final integration: full text → IPA → WAV → ASR → text round-trip on a curated corpus. Target ≥90% word-level agreement of round-trip text vs original text.

Python reference dependencies (validation only, not runtime):
- `phonemizer` (uses espeak under the hood) — primary IPA reference
- `num2words` — number normalization reference
- `nltk` or `spacy` — POS tagging for homograph reference
- Optionally: `misaki` itself for high-fidelity comparison

These can be installed in a venv; runtime Rust binary stays fully native.

## 6. Receipts table (fill in as stages land)

| Stage | What | Target | Result | Notes |
|---|---|---|---|---|
| 1 | CMUdict + ARPAbet→IPA | 100% match on in-vocab test set | — | |
| 2 | Sentence + punctuation | 100% match on multi-sentence set | — | |
| 3.1 | Numbers (cardinal, decimal) | 100% on 50-pair set | — | |
| 3.2 | Ordinals + years | 100% on 30-pair set | — | |
| 3.3 | Money + time | 100% on 30-pair set | — | |
| 3.4 | Dates | 100% on 20-pair set (US default) | — | |
| 3.5 | Abbreviations + acronyms | 100% on 50-pair set | — | |
| 3.6 | Units | 100% on 20-pair set | — | |
| 4 | Homograph disambiguation | ≥80% on 50-sentence set | — | |
| 5a | OOV literal-spellout fallback | every input produces SOME IPA (no panics, no empty) | — | unblocks stage 6 round-trip |
| 5b | OOV LTS rules | ≥70% char-agreement on 100-word OOV set | — | replaces 5a; ship as polish after 6 |
| 6 | End-to-end intelligibility | ≥90% word agreement on 100-sentence ASR round-trip | — | **gate for M2 ship** (run with 5a; re-run after 5b lands) |

## 7. Don't do

- **Don't subprocess espeak / espeak-ng** at runtime. The whole point is native.
- **Don't load a giant Python NLP toolkit at runtime**. Validation only.
- **Don't depend on network at runtime** (no HF Hub, no online lexicon).
- **Don't widen test thresholds to make tests pass.** If stage 5 is at 50% agreement, fix the rules — don't change the target.
- **Don't link GPL data** (espeak-ng's rule files) unless we relicense kokoro-tts. Mention any borrowed rules + their license in the commit + spec.
- **Don't try to be perfect on stage 5.** It's the long tail; ship "good enough" and note the known weaknesses. Misaki itself isn't perfect either.
- **Don't refactor the existing `Phonemizer` trait** — it's correct. Extend with new impls.
- **Don't reach for `unicode-segmentation` or other heavy unicode crates.** English text + IPA passthrough = ASCII-aware tokenization is sufficient. Avoid the dependency weight; if you need word boundaries, `text.split_whitespace()` is the answer.
- **Don't naively split sentences by `.`/`!`/`?` regex.** Abbreviations (`Mr.` `Dr.` `e.g.` `etc.` `Inc.` `vs.` decimals like `3.14`) will break it. Maintain an explicit abbreviation guard list in the sentence splitter; the test corpus must include "Mr. Smith arrived. He waited." and "She's 3.14 meters tall? Yes." cases.
- **Don't load CMUdict eagerly at process start** if startup latency matters — use `OnceLock` for lazy initialization, parse on first phonemize call. CMUdict is ~3.6 MB so embedding via `include_str!` adds that to the binary; verify the parse-once cost is <100 ms before considering alternatives.
- **Don't drop or transliterate non-ASCII letters silently** in input text. If user text contains "café" or "naïve", either pass the accented form through to OOV (the LTS rules will only see ASCII anyway, fine) OR strip diacritics deterministically — but never silently drop the whole word. The kokoro-tts `phonemes_to_ids` already warns on unmapped chars; the same surfacing discipline applies here.
- **Don't ship the lexicon module without confirming CMUdict's license is compatible.** It's typically distributed under a BSD-ish "use freely" notice (verify the exact file you embed says so; the cmudict.dict at github.com/cmusphinx/cmudict is public-domain dedicated). Mention the license in a `LICENSE-3RD-PARTY` file or the data file's header comment.
- **Don't conflate stage 5a (literal spellout) with stage 5b (LTS rules) in the same commit.** 5a unblocks the pipeline; 5b is the quality work. Different acceptance criteria, different debug surfaces.

## 8. References

- CMUdict: https://github.com/cmusphinx/cmudict (file `cmudict.dict`, public domain).
- ARPAbet → IPA mappings: https://en.wikipedia.org/wiki/ARPABET (verify against Kokoro vocab).
- Misaki (the upstream Kokoro G2P): https://github.com/hexgrad/misaki — reference for tone/expectation, not for code.
- Phonemizer Python library: https://github.com/bootphon/phonemizer — espeak wrapper, useful as Python validation reference.
- Festvox CART trees / LTS rule format: http://festvox.org/docs/manual-1.4.3/festvox_13.html (background reading for stage 5 option A).
- Kokoro vocab: `models/config.json::vocab` (114 entries, IPA-only).
- Existing scaffold: `src/phonemizer.rs`, `docs/g2p.md`.

## 9. Coordination

- Codex (pty-10) owns implementation; pty-9 reviews + writes/refines spec inline as discoveries surface (same pattern as kokoro-rust-port.md).
- pty-1 (me) coordinates between them and runs final round-trip validation against nemotron-speech ASR.
- Optional: Andrew may spin up Opencode (Qwen3.6-27B local) as a third worker. Best fit: stage 1 (mechanical, low-risk lexicon work) or stage 3 (tedious normalization rules). NOT stage 5 (LTS rules need care).
- Each stage's commit lands independently. Pty-9 may also commit fixes to scaffold bugs surfaced during implementation.
- After stage 6 lands, a final `g2p: M2 spec stamp` commit on the spec stamps the receipts table and milestone status.

## 10. Note from pty-1 (drafting reviewer)

This spec is intentionally aggressive: stages 1–5 in 1–2 days at codex pace, with parallelizable middle stages. The scope mirrors what misaki itself is, minus its neural homograph classifier (stage 4 here is rules-based) and minus its ML-training overhead. We're building a **lexicon-first, rules-augmented** G2P, which is the durable architecture before neural augmentation if we ever want it.

The validation discipline is what carried the model port through 12 numerical stages; the same pattern keeps G2P honest. Pty-9 — feel free to refine targets, push back on stage decomposition, or surface things I missed (text normalization is full of edge cases I might have glossed over). Andrew's coffee-break window is the right time to surface concerns; my draft is a starting point, not a contract.
