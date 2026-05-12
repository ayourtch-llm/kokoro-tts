use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
fn main() {
    for s in &["$5.99", "$PATH", "$FOO_BAR", "$100", "export PATH=$PATH:/usr/local/bin"] {
        println!("{:?}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
    }
}
