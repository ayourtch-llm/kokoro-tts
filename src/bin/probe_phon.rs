use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};

fn main() {
    let p = TwoTierPhonemizer;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let words: Vec<&str> = if args.is_empty() {
        vec![
            "krone", "kronen", "Krone", "Kronen", "Helfferich",
            "Deutschösterreich", "Frau", "Eisenmenger", "Horthy",
            "bourse", "Reichsbank", "centime", "centimes",
            "Versailles", "Wilson", "Bela", "Kun", "Bolshevik",
            "Budapest", "Austrian", "ao-kronen",
        ]
    } else {
        args.iter().map(|s| s.as_str()).collect()
    };
    for w in words {
        match p.phonemize(w) {
            Ok(s) => println!("{:30} -> {}", w, s),
            Err(e) => println!("{:30} -> ERR: {}", w, e),
        }
    }
}
