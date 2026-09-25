# LLM Hallucination Detector

[![CI](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/llm-token-visualizer.svg)](https://crates.io/crates/llm-token-visualizer)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A Rust CLI and library that reads an LLM answer's token log probabilities and flags the words the model was unsure about, with the alternatives it was weighing at that point. It can also render per-token confidence as colored terminal output, a standalone HTML report, or Markdown.

Hallucinations often sit where the model's confidence drops: a name, a date, a city it half-remembers. If your API returns `logprobs` (OpenAI and many OpenAI-compatible servers do), this tool shows you those spots in one command.

```text
$ llm-token-visualizer --logprobs-file examples/logprobs/cuyp.json --threshold 0.6
...
Aelbert Cuyp died in 1691 in Dordrecht, Netherlands.

  [uncertain] 'Aelbert' (tokens 0-2): p=0.49 at "A"; model also considered "The" (0.43), "D" (0.08)
  [uncertain] ' Dordrecht' (tokens 11-13): p=0.57 at "ord"; model also considered "üsseldorf" (0.39), "elf" (0.03)

2 low-confidence span(s) at threshold 0.60 (17 tokens, mean p=0.88).
Check these claims before trusting the answer.
```

That is a real response from Llama 3.1 8B Instruct. The answer happens to be right, but the model gave Düsseldorf a 39% chance, which is exactly the kind of claim to double-check. The "Aelbert" flag is the model choosing between starting with the name or with "The": low probability can be about phrasing, not facts.

## Install

Prebuilt binaries for Linux, macOS (Intel and Apple Silicon) and Windows are attached to each [GitHub Release](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/releases/latest). Download the archive for your platform, unpack it, and run `llm-token-visualizer` (the archive includes the sample responses under `samples/`).

Or install from crates.io with a Rust toolchain:

```bash
cargo install llm-token-visualizer
```

## Quick start

```bash
git clone https://github.com/Mattbusel/LLM-Hallucination-Detection-Script
cd LLM-Hallucination-Detection-Script

# Offline: analyze a bundled real response, no API key needed
cargo run -- --logprobs-file examples/logprobs/cuyp.json --threshold 0.6

# All four bundled samples, through the library API
cargo run --example detect

# Machine-readable report
cargo run -- --logprobs-file examples/logprobs/cuyp.json --format json

# Live: ask a model and analyze its answer (any OpenAI-compatible API)
export OPENAI_API_KEY=sk-...
cargo run -- --live "Who was the second person to walk on the Moon?" --save answer.json
```

### How detection works

1. Each token's probability is `exp(logprob)`.
2. Tokens are grouped into words (a subword like `ord` in `Dordrecht` belongs to its word).
3. A word is flagged when any of its tokens with letters or digits has probability below `--threshold` (default `0.5`, meaning the model put more weight on other options than on the one it picked). Pure punctuation and whitespace never trigger a flag.
4. Neighbouring flagged words merge into one span. Each span reports its weakest token and the `top_logprobs` alternatives at that token.

Raise the threshold to catch more (and noisier) spans, lower it to see only the shakiest ones.

### Input

`--logprobs-file` accepts a full Chat Completions response saved as JSON, just its `logprobs` object (`{"content": [...]}`), or a bare array of `{"token", "logprob", "top_logprobs"}` entries. Use `-` to read stdin. Request completions with `"logprobs": true` and, for alternatives, `"top_logprobs": 3` or more. Special tokens such as `<|eot_id|>` are ignored.

### Live mode

`--live "<prompt>"` sends the prompt with temperature 0, `logprobs: true` and `top_logprobs: 3`, then analyzes the answer.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | required | Bearer token for the API |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any server with Chat Completions and logprobs, e.g. `https://router.huggingface.co/v1` |

Flags: `--model` (default `gpt-4o-mini`), `--max-tokens` (default 200), and `--save <path>` to keep the raw response so you can re-run it offline with `--logprobs-file`. Anthropic's API does not return logprobs, so Claude models cannot be analyzed this way.

The bundled samples were fetched through the Hugging Face router (`meta-llama/Llama-3.1-8B-Instruct:novita`). Build with `--no-default-features` for an offline-only binary without an HTTP client.

### Use it in CI or scripts

`--fail-on-flag` exits with status 2 when any span is flagged, so you can gate a pipeline on it or route flagged answers to review:

```bash
llm-token-visualizer --logprobs-file answer.json --fail-on-flag --format json -o report.json
```

## What it cannot tell you

Low token probability is a useful signal, not a fact checker. The bundled `moonwalk.json` sample shows the failure mode: the model answers "Pete Conrad was the second person to walk on the Moon" (it was Buzz Aldrin) with at least 72% probability on every token of the name, so the wrong name is not flagged; only phrasing words like "which" are. Models can be confidently wrong. Use this to decide where to look first, not to certify an answer.

## Visualizing your own confidence scores

If you already have per-token scores from another source, the visualizer mode renders them with a five-band color scale (very low < 0.3, low < 0.5, medium < 0.7, high < 0.9, very high) and labeled spans (`fact`, `uncertain`, `hallucination`, or any label):

```bash
# Built-in demo (hand-written scores)
cargo run -- --demo

# A text file plus a confidence JSON file (samples included)
cargo run -- --text-file sample_text.txt --confidence-file demo_data.json

# HTML report
cargo run -- --text-file sample_text.txt --confidence-file demo_data.json --format html --output report.html

# Markdown, with stats
cargo run -- --text-file sample_text.txt --confidence-file demo_data.json --format markdown --verbose
```

Confidence file format:

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

`start` is inclusive and `end` is exclusive, both as token indices.

### Library use

The crate is named `llm-token-visualizer`.

```rust
use llm_token_visualizer::detect::{detect, parse_logprobs, to_token_analysis};
use llm_token_visualizer::visualize_tokens;

let tokens = parse_logprobs(&response_json)?;
let report = detect(&tokens, 0.5);
for span in &report.spans {
    println!("{}: {}", span.text, span.describe());
}

// Render the same result as HTML
let analysis = to_token_analysis(&tokens, &report);
let html = visualize_tokens(&report.text, &analysis, "html", None)?;
```

`analyze_with_issues` and `quick_analyze` are still available.

## Repository layout

```
src/
  main.rs       CLI (clap)
  lib.rs        public API: visualize_tokens, quick_analyze, analyze_with_issues
  detect.rs     logprob parsing and low-confidence span detection
  live.rs       OpenAI-compatible API client for --live (feature "live")
  data.rs       TokenAnalysis, TokenInfo, TokenFlag, ConfidenceLevel
  renderer.rs   Terminal, HTML and Markdown renderers
  utils.rs      tokenizer for demos, metrics, issue detection
examples/logprobs/                real Llama 3.1 8B responses with logprobs
examples/detect.rs                library example over those samples
demo_data.json, sample_text.txt   example input for the visualizer
rust_mvps/                        design sketches, see below
real-time fact-checking DAG engine.cpp   standalone C++ sketch, see below
```

## Status and limitations

The Rust crate in `src/` is the working part of this repo: it builds, has unit tests, and CI runs fmt, clippy, tests and a detector smoke test on every push.

The rest is exploratory and should be read as design notes, not shipped features:

- `rust_mvps/` holds source sketches for a BERT-based detector (candle), multi-language phrase patterns, a streaming detector with a WebSocket server, and a web dashboard. They have no Cargo manifests and are not wired into the build. The neural detector expects model weights that are not published.
- `real-time fact-checking DAG engine.cpp` is a single-file C++ sketch of a Boost Graph based fact graph. It is not part of any build here and needs Boost to compile.
- Earlier versions of this README described a Python `hallucination_detector.py` module. That file is not in the repository, so its documentation has been removed.

## License

MIT, see [LICENSE](LICENSE).
