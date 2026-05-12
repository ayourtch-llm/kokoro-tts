use kokoro_tts::phonemizer::{custom_vocab, Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
use std::path::Path;
fn main() {
    let vocab = custom_vocab::CustomVocab::load(Path::new("tmp/loanword_vocab.json")).unwrap();
    custom_vocab::set(vocab).unwrap();
    for s in &[
        "She studied at l'École polytechnique and the Universität zu Köln.",
        "Ångström, Müller, and Hernández-García co-authored the paper.",
        "Bach's St. Matthew Passion premiered in 1727.",
        "Lao Tzu's 道德经 is the foundational Taoist text.",
    ] {
        println!("{}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
        println!("  IPA: {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
