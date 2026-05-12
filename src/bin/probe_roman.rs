use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
fn main() {
    for s in &[
        "Pope John Paul II canonized over 480 saints.",
        "Read Chapter VII, verses 1–17.",
        "Star Wars: Episode IV",
        "King George V reigned from 1910 to 1936.",
        "page ix in the preface",
        "Henry VIII had six wives.",
        "I will mix the IV drip.",
        "Type II diabetes",
        "The MIX of CDs and DVDs",
    ] {
        println!("{}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
        println!("  IPA: {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
