# Contributing to LLM-Hallucination-Detection-Script

Thanks for your interest. Contributions that improve detection accuracy, add new methods, extend provider integrations, or add test coverage are all welcome.

## What we want

- **New detection methods** -- additional heuristics or ML-based approaches
- **Provider integrations** -- examples using Mistral, Gemini, local models
- **Rust MVP improvements** -- better streaming detection, new dashboard features
- **Bug fixes** -- correctness issues, edge cases, false positive reduction
- **Tests** -- more labeled examples, accuracy benchmarks

## How to contribute

1. Fork and clone
2. Python: `python -m pytest tests/` must pass
3. Rust: `cargo test --all` must pass
4. Open a PR with a description of the change and why

## Questions

Open a [Discussion](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/discussions) for design questions or ideas before building.
