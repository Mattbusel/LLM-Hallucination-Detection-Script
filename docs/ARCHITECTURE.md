# How it works, and what is in this repo

## How detection works

See the [animated diagram](img/how-it-works.svg) for the same steps on a real answer.

1. Each token's probability is `exp(logprob)`.
2. Tokens are grouped into words (a subword like `ord` in `Dordrecht` belongs to its word). A token starts a new word when it begins with whitespace or punctuation, or when the previous token ended with one.
3. A word is flagged when any of its tokens with letters or digits has probability below `--threshold` (default `0.5`, meaning the model put more weight on other options than on the one it picked). Pure punctuation and whitespace never trigger a flag.
4. Neighbouring flagged words (adjacent, or separated only by whitespace tokens) merge into one span. Each span reports its weakest token and the `top_logprobs` alternatives at that token.

Raise the threshold to catch more (and noisier) spans, lower it to see only the shakiest ones.

The colors in the terminal, HTML and Markdown reports come from `report::heat`, which interpolates between four stops: red at p 0.25 and below, orange at 0.55, yellow at 0.80, and a neutral grey at 0.97 and above.

The code is in [`src/detect.rs`](../src/detect.rs) (parsing and span detection) and [`src/report.rs`](../src/report.rs) (colors and reports).

## Repository layout

```
src/
  main.rs       CLI (clap)
  lib.rs        public API: visualize_tokens, quick_analyze, analyze_with_issues
  detect.rs     logprob parsing and low-confidence span detection
  report.rs     detect-mode reports: terminal heatmap, HTML page, Markdown
  live.rs       OpenAI-compatible API client for --live (feature "live")
  data.rs       TokenAnalysis, TokenInfo, TokenFlag, ConfidenceLevel
  renderer.rs   Terminal, HTML and Markdown renderers for the visualizer mode
  utils.rs      tokenizer for demos, metrics, issue detection
examples/logprobs/                real Llama 3.1 8B responses with logprobs
examples/detect.rs                library example over those samples
docs/                             project site (GitHub Pages), README images and these docs
demo_data.json, sample_text.txt   example input for the visualizer
rust_mvps/                        design sketches, see below
real-time fact-checking DAG engine.cpp   standalone C++ sketch, see below
```

## Status and limitations

The Rust crate in `src/` is the working part of this repo: it builds, has unit tests, and CI runs fmt, clippy, tests and detector smoke tests (JSON, HTML and Markdown) on every push.

The rest is exploratory and should be read as design notes, not shipped features:

- `rust_mvps/` holds source sketches for a BERT-based detector (candle), multi-language phrase patterns, a streaming detector with a WebSocket server, and a web dashboard. They have no Cargo manifests and are not wired into the build. The neural detector expects model weights that are not published.
- `real-time fact-checking DAG engine.cpp` is a single-file C++ sketch of a Boost Graph based fact graph. It is not part of any build here and needs Boost to compile.
- Earlier versions of this README described a Python `hallucination_detector.py` module. That file is not in the repository, so its documentation has been removed.
