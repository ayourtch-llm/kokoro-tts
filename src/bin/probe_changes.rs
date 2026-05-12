use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    for s in &["brain plasticity changes", "those negative changes", "changes occur", "she changes", "changes."] {
        println!("{} -> {}", s, p.phonemize(s).unwrap());
    }
}
