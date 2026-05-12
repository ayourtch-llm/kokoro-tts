use kokoro_tts::phonemizer::{misaki_gold, lexicon};
fn main() {
    let g = misaki_gold::lexicon();
    let l = lexicon::lexicon();
    for w in ["introduction", "iNTRODUCTION", "Introduction", "INTRODUCTION", "i", "I", "NTRODUCTION", "ntroduction"] {
        let gl = g.lookup(w);
        let ll = l.lookup(w);
        println!("{:20} gold={:?} cmu={:?}", w, gl, ll);
    }
}
