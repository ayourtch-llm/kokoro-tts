use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    let p = TwoTierPhonemizer;
    for s in &[
        "Flush Draw : 9 Outs",
        "Pot Odds Ratio = 23bb:11.5bb = 2:1 Pot Odds",
        "All-in equity",
        "Rule of 2 & 4: Flop Not All-in Equity: 4 x 2 = 8% Equity",
    ] {
        println!("{}", s);
        println!("  -> {}", p.phonemize(s).unwrap());
    }
}
