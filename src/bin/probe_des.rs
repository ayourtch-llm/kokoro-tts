use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
fn main() {
    for s in &["design, 37–62", "37–62", "299–320"] {
        println!("{:?}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
        println!("  IPA: {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
