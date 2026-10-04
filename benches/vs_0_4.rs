//! This release against 0.4.0 on the same input: parse a 4,000-token Chat
//! Completions response and flag low-confidence spans.
//!
//! Run: `cargo bench --bench vs_0_4`

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn big_response() -> String {
    let sample: serde_json::Value =
        serde_json::from_str(include_str!("../examples/logprobs/eiffel.json")).unwrap();
    let content = sample["choices"][0]["logprobs"]["content"].as_array().unwrap().clone();
    let mut all = Vec::new();
    while all.len() < 4_000 {
        all.extend(content.iter().cloned());
    }
    serde_json::json!({"choices": [{"logprobs": {"content": all}}]}).to_string()
}

fn bench(c: &mut Criterion) {
    let json = big_response();
    let mut g = c.benchmark_group("parse_and_detect_4k_tokens");
    g.bench_function("0.5.0", |b| {
        b.iter(|| {
            let t = llm_token_visualizer::detect::parse_logprobs(black_box(&json)).unwrap();
            llm_token_visualizer::detect::detect(&t, 0.6).spans.len()
        })
    });
    g.bench_function("0.4.0", |b| {
        b.iter(|| {
            let t = llm_token_visualizer_prev::detect::parse_logprobs(black_box(&json)).unwrap();
            llm_token_visualizer_prev::detect::detect(&t, 0.6).spans.len()
        })
    });
    g.finish();
    // Same answer from both versions on English text.
    let a = llm_token_visualizer::detect::detect(&llm_token_visualizer::detect::parse_logprobs(&json).unwrap(), 0.6);
    let b = llm_token_visualizer_prev::detect::detect(&llm_token_visualizer_prev::detect::parse_logprobs(&json).unwrap(), 0.6);
    assert_eq!(a.spans.len(), b.spans.len());
}

criterion_group!(benches, bench);
criterion_main!(benches);
