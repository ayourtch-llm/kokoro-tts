use kokoro_tts::phonemizer::pre_phonemize_for_test;
fn main() {
    for s in &[
        "23bb:11.5bb",
        "Pot Odds Ratio = 23bb:11.5bb = 2:1 Pot Odds",
        "23 bb : 11.5 bb",
        "11.5",
    ] {
        println!("{:?}", s);
        println!("  -> {:?}", pre_phonemize_for_test(s));
    }
}
