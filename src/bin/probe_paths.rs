use kokoro_tts::phonemizer::{Phonemizer, TwoTierPhonemizer, pre_phonemize_for_test};
fn main() {
    for s in &[
        "/usr/src",
        "Most distributions install kernel source files in /usr/src.",
        "/boot/vmlinuz",
        "arch/i386/boot/bzImage",
        "path_to_kernel_src/Documentation/Configure.help",
    ] {
        println!("{:?}", s);
        println!("  PRE: {:?}", pre_phonemize_for_test(s));
        println!("  IPA: {}", TwoTierPhonemizer.phonemize(s).unwrap());
    }
}
