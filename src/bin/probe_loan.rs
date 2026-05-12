use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer};
fn main() {
    for s in &[
        "crème fraîche",
        "krem fresh",
        "pâté de foie gras",
        "pa-TAY duh fwah grah",
        "Universität zu Köln",
        "oo-nee-ver-zee-TATE tsoo Curln",
        "Schleichhändler",
        "SHLYKH-hend-lur",
        "Ceridwen",
        "KEH-rid-wen",
        "Władysław Jagiełło",
        "Vlah-DEE-swaff yah-GYEH-woh",
    ] {
        println!("{:32} -> {}", s, TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
