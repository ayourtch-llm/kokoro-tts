use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    for s in &[
        "She studied at l'École polytechnique and the Universität zu Köln.",
        "Lao Tzu's 道德经 is the foundational Taoist text.",
        "Ångström, Müller, and Hernández-García co-authored the paper.",
        "Bach's St. Matthew Passion (BWV 244) premiered in 1727.",
    ] {
        println!("{}", s);
        println!("  -> {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
