use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    for s in &[
        "[B]efore publishing Pilgrim",
        "Goh et al.",
        "Oettingen et al.",
        "Sardine, can",
        "Yimou Lee",
        ", 3.",
    ] {
        println!("{:?}", s);
        println!("  -> {:?}", p.phonemize(s).unwrap());
    }
}
