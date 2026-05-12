use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
fn main() {
    for s in &[
        "Convert 3/4 cup to milliliters.",
        "0xDEADBEEF",
        "0x7FFFFFFF8000",
        "He's 6'2\" and weighs 185 lbs.",
        "Edit ~/.bashrc",
        "~/.bashrc",
        "NaCl",
    ] {
        println!("{}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
        println!("  IPA: {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
