use kokoro_tts::phonemizer::pre_phonemize_for_test;
fn main() {
    for s in &["variable va : std_logic_vector(a'length-1 downto 0) := a", "x == y", "match { Some(x) => x, None => 0 }"] {
        println!("{:?} -> {:?}", s, pre_phonemize_for_test(s));
    }
}
