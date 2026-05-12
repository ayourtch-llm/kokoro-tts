use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    for s in &[
        "We have A♣ T♣ and the flop is 5♣ K♣ 8♥.",
        "Trip Jacks Draw : 2 Outs (J♥, J♠)",
        "AA vs KK pre-flop.",
    ] {
        println!("{}", s);
        println!("  -> {}", p.phonemize(s).unwrap());
    }
}
