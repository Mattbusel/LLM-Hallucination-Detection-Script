# LLM Hallucination Detector: token confidence visualizer

[![CI](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/actions/workflows/ci.yml)

A Rust CLI and library that shows an LLM response token by token, colored by the model's confidence, with labeled spans for facts, uncertain claims and likely hallucinations. Output goes to the terminal, a standalone HTML report, or Markdown.

Hallucinations tend to hide in fluent text. When you already have per-token confidence (for example from an API's logprobs) the fastest way to spot a shaky claim is to look at where confidence drops. This tool turns that JSON into something you can read at a glance and paste into a review, a bug report or a PR.

## What it does

- Renders every token with a five-band confidence color scale (very low < 0.3, low < 0.5, medium < 0.7, high < 0.9, very high).
- Highlights flagged spans (`fact`, `uncertain`, `hallucination`, or any label you choose) with a description.
- Three renderers behind one `Renderer` trait: `TerminalRenderer` (ANSI colors), `HtmlRenderer` (self-contained page), `MarkdownRenderer` (emoji legend, works in GitHub comments).
- Summary metrics: token count, min/max/average confidence, low-confidence and flagged token counts.
- Simple issue detection: very low confidence tokens and sudden confidence dips between neighbours.
- Built-in demo so you can see the output without any data.

It does not call a model and does not produce confidence scores itself. You bring the scores; it visualizes and summarizes them.

## Quick start

Requires a Rust toolchain.

```bash
git clone https://github.com/Mattbusel/LLM-Hallucination-Detection-Script
cd LLM-Hallucination-Detection-Script

# Built-in demo in the terminal
cargo run -- --demo

# Your own data: a text file plus a confidence JSON file (samples included)
cargo run -- --text-file sample_text.txt --confidence-file demo_data.json

# HTML report
cargo run -- --text-file sample_text.txt --confidence-file demo_data.json --format html --output report.html

# Markdown, with stats
cargo run -- --text-file sample_text.txt --confidence-file demo_data.json --format markdown --verbose
```

CLI flags: `--text` / `--text-file`, `--confidence` / `--confidence-file`, `--format terminal|html|markdown`, `--output <path>`, `--verbose`, `--demo`.

### Input format

```json
{
  "tokens": [
    {"text": "The", "confidence": 0.95},
    {"text": " tower", "confidence": 0.88},
    {"text": " purple", "confidence": 0.15}
  ],
  "flags": [
    {"start": 2, "end": 3, "flag": "hallucination", "description": "Never painted purple"}
  ]
}
```

`start` is inclusive and `end` is exclusive, both as token indices. `demo_data.json` is a full example that pairs with `sample_text.txt`.

### Library use

The crate is named `llm-token-visualizer`.

```rust
use llm_token_visualizer::{visualize_tokens, analyze_with_issues, quick_analyze, TokenAnalysis};

let analysis: TokenAnalysis = serde_json::from_str(&json)?;
let html = visualize_tokens(&text, &analysis, "html", None)?;

let (metrics, issues) = analyze_with_issues(&analysis);
println!("avg confidence {:.2}, {} issues", metrics.avg_confidence, issues.len());

// Quick look with deterministic placeholder scores (no real confidence data)
let md = quick_analyze("Some text to try", "markdown")?;
```

## Repository layout

```
src/
  main.rs       CLI (clap)
  lib.rs        public API: visualize_tokens, quick_analyze, analyze_with_issues
  data.rs       TokenAnalysis, TokenInfo, TokenFlag, ConfidenceLevel
  renderer.rs   Terminal, HTML and Markdown renderers
  utils.rs      tokenizer for demos, metrics, issue detection
demo_data.json, sample_text.txt   example input
rust_mvps/                        design sketches, see below
real-time fact-checking DAG engine.cpp   standalone C++ sketch, see below
```

## Status and limitations

The Rust visualizer in `src/` is the working part of this repo: it builds, has unit tests, and CI runs fmt, clippy and tests on every push.

The rest is exploratory and should be read as design notes, not shipped features:

- `rust_mvps/` holds source sketches for a BERT-based detector (candle), multi-language phrase patterns, a streaming detector with a WebSocket server, and a web dashboard. They have no Cargo manifests and are not wired into the build. The neural detector expects model weights that are not published.
- `real-time fact-checking DAG engine.cpp` is a single-file C++ sketch of a Boost Graph based fact graph. It is not part of any build here and needs Boost to compile.
- Earlier versions of this README described a Python `hallucination_detector.py` module. That file is not in the repository, so its documentation has been removed.
