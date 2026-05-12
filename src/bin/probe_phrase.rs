use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    let phrases = [
        "COME TO PARIS ON IMPORTANT BUSINESS.",
        "do what you think best—BUT FIND LIVINGSTONE!",
        "BUT FIND LIVINGSTONE",
    ];
    for s in &phrases {
        match p.phonemize(s) {
            Ok(out) => println!("{}\n  -> {}\n", s, out),
            Err(e) => println!("{}\n  -> ERR: {}\n", s, e),
        }
    }
}
