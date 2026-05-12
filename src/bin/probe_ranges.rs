use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    for s in &[
        "299–320",
        "pages 299–320",
        "This is a –EV pot odds call.",
        "1939–1945",
    ] {
        println!("{:?} -> {:?}", s, p.phonemize(s).unwrap());
    }
}
