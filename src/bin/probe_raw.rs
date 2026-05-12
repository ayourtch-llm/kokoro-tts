use kokoro_tts::phonemizer::{lexicon_for_test};
fn main() {
    for w in &["changes", "matches", "wishes", "dishes"] {
        let phones = lexicon_for_test(w);
        println!("{:10} -> {:?}", w, phones);
    }
}
