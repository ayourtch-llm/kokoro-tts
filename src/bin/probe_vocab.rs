use kokoro_tts::phonemizer::{custom_vocab, Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
use std::path::Path;
fn main() {
    let vocab = custom_vocab::CustomVocab::load(Path::new("/tmp/test_vocab.json")).unwrap();
    custom_vocab::set(vocab).unwrap();
    for s in &[
        "We use kokoro and Anthropic models 5x daily, send report ASAP.",
        "kokoro",
        "ASAP",
        "5x faster",
    ] {
        println!("{}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
        println!("  IPA: {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
