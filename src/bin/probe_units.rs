use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
fn main() {
    for s in &[
        "Cache: L1 32KB, L2 256KB, L3 8MB, RAM 32GB, SSD 1TB.",
        "File sizes: 1.5 KiB, 2.3 MiB, 4.7 GiB, 1 TiB.",
        "Bandwidth: 1 Mbps, 100 Mbps, 1 Gbps, 10 Gbps.",
        "2.4 GHz at -40 dBm",
    ] {
        println!("{}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
    }
}
