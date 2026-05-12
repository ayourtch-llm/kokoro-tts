use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    for s in &["What, me worried?", "What me worried", "Hello, world.", "Are you sure?"] {
        println!("{:?} -> {:?}", s, p.phonemize(s).unwrap());
    }
}
