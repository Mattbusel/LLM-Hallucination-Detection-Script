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
  interop.rs    adapters for other crates (feature "async-openai")
examples/logprobs/                real Llama 3.1 8B responses with logprobs
examples/                         library examples over those samples
tests/properties.rs               property tests (proptest) for the parser and detector
benches/vs_0_4.rs                 criterion benchmark against the 0.4.0 release
docs/                             project site, README images and these docs
demo_data.json, sample_text.txt   example input for the visualizer
```

## Status and limitations

The Rust crate in `src/` is the whole project: it builds, has unit and property tests, and GitLab CI runs the tests, clippy and rustdoc on every push to the default branch.

Earlier versions of the repository also held design sketches (a BERT-based detector, a WebSocket streaming detector, a C++ fact graph) that were never built or wired in; 0.5.0 removed them. They are still in the git history.
