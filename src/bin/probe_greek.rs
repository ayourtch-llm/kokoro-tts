use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
fn main() {
    for s in &[
        "The Greek letters α, β, γ, δ, ε, ζ, η, θ, ι, κ are common in physics.",
        "Pythagoras: a² + b² = c².",
        "λ-calculus uses (λx.x+1) for the increment function.",
        "H₂O is water; CO₂ is carbon dioxide.",
        "She memorized π to 20 digits: 3.14159265358979323846.",
    ] {
        println!("{}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
        println!("  IPA: {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
