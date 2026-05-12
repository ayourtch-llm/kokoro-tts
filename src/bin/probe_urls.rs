use kokoro_tts::phonemizer::{normalize_urls_for_test, Phonemizer, TwoTierPhonemizer};

fn main() {
    let p = TwoTierPhonemizer;
    let urls = [
        // The one Andrew flagged
        "www.soft-wired.com/ref/ch01",
        // Variations on www. prefix
        "www.example.com",
        "www.example.com/",
        "www.example.com/page",
        // Scheme prefixes
        "http://example.com",
        "https://example.com",
        "ftp://files.example.org",
        // Paths with hyphens and digits
        "https://my-site.com/path/sub-page/v2",
        "example.com/a/b/c/d/e",
        // Query strings & fragments
        "example.com/search?q=hello&n=10",
        "example.com/page#section-2",
        // Common in books: github-style
        "github.com/user/repo",
        "github.com/user/repo/blob/main/README.md",
        // Wikipedia-style
        "en.wikipedia.org/wiki/Plasticity_(neuroscience)",
        // Trailing punctuation (sentence)
        "Visit https://example.com.",
        "See www.example.com! It's cool.",
        // In running prose
        "We posted the data at www.soft-wired.com/ref/ch01 last week.",
        // All-caps domain
        "WWW.EXAMPLE.COM/PATH",
        // Numbers in domains
        "example123.com/v2/api",
        // IP-ish
        "192.168.1.1",
        "127.0.0.1:8080",
        // No-scheme but TLD-ish (false positive risk)
        "I have a B.A. in art.",
        "the U.S. economy",
    ];
    for u in &urls {
        println!("INPUT:  {}", u);
        println!("URLS:   {}", normalize_urls_for_test(u));
        println!("IPA:    {}", p.phonemize(u).unwrap());
        println!();
    }
}
