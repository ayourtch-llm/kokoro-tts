use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    for s in &["LOCKDOWN IN EFFECT.", "Oh, Chen.", "Shu stopped, aghast."] {
        println!("{:?} -> {:?}", s, TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
