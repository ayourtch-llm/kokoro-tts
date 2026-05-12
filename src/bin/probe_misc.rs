use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    for s in &["process (a, b)", "f(x, y)", "a == b", "a != b", "x >= y"] {
        println!("{:?} -> {:?}", s, TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
