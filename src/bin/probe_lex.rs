use kokoro_tts::phonemizer::lexicon_for_test;
fn main() {
    for w in ["babies", "BABIES", "Babies", "weirdies"] {
        println!("{:10} -> {:?}", w, lexicon_for_test(w));
    }
}
