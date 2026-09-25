# Contributing

Thanks for helping. Small, focused PRs are easiest to review.

## Setup

You need a stable Rust toolchain.

```bash
git clone https://github.com/Mattbusel/LLM-Hallucination-Detection-Script
cd LLM-Hallucination-Detection-Script
cargo test
cargo run -- --logprobs-file examples/logprobs/cuyp.json --threshold 0.6
```

## Before you open a PR

CI runs these; run them locally first:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo clippy --all-targets --no-default-features -- -D warnings
cargo test
cargo run --example detect
```

## Good places to start

- Detection logic lives in `src/detect.rs`, with unit tests at the bottom of the file. New heuristics should come with a test and, ideally, a real sample response in `examples/logprobs/`.
- Sample responses must be real API output (say which model and provider in the PR). Do not hand-edit logprobs.
- `rust_mvps/` is design sketches that are not built. Turning one into a working crate is welcome, but open an issue first.

Issues labelled `good first issue` are sized for a first PR.
