use kokoro_tts::phonemizer::{
    normalize_urls_for_test, pre_phonemize_for_test, Phonemizer, TwoTierPhonemizer,
};
fn main() {
    let p = TwoTierPhonemizer;
    let urls = [
        "example.com/path",
        "go to example.com today",
        "www.soft-wired.com/ref/ch01",
        "https://example.com/path",
    ];
    for u in &urls {
        println!("INPUT:  {}", u);
        println!("URLS:   {:?}", normalize_urls_for_test(u));
        println!("PRENRM: {:?}", pre_phonemize_for_test(u));
        println!("IPA:    {}", p.phonemize(u).unwrap());
        println!();
    }
}
