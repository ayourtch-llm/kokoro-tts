use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    for s in &["This is a –EV pot odds call.", "11.5bb Pot + 11.5bb Bet = 23bb"] {
        println!("{}", s);
        println!("  -> {}", p.phonemize(s).unwrap());
    }
}
