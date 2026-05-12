use kokoro_tts::phonemizer::{misaki_gold, lexicon};
fn main() {
    let g = misaki_gold::lexicon();
    let l = lexicon::lexicon();
    for w in ["babies", "baby", "babie", "babi"] {
        let gl = g.lookup(w);
        let ll = l.lookup(w).map(|p| p.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" "));
        println!("{:8}  gold={:?}  cmu={:?}", w, gl, ll);
    }
}
