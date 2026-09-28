# Reference

Everything the command-line tool and the library can do. For the short version, see the [README](../README.md).

## Command-line options

Run `llm-token-visualizer --help` for the same list.

| Flag | Default | What it does |
|---|---|---|
| `--logprobs-file <PATH>` | | Detect: a Chat Completions response (or its logprobs) saved as JSON; `-` reads stdin |
| `--live <PROMPT>` | | Detect: send this prompt to an OpenAI-compatible API (needs `OPENAI_API_KEY`) and analyze the answer |
| `--model <MODEL>` | `gpt-4o-mini` | Model for `--live` |
| `--max-tokens <N>` | `200` | Max tokens for `--live` |
| `--save <PATH>` | | Save the raw `--live` response so it can be re-analyzed offline with `--logprobs-file` |
| `--threshold <0-1>` | `0.5` | Flag words containing a token with probability below this |
| `--fail-on-flag` | | Exit with status 2 if any span is flagged (for CI and scripts) |
| `-f, --format` | `terminal` | `terminal`, `html`, `markdown`, or `json` (json: detect mode only) |
| `-o, --output <PATH>` | | Output file for html, markdown and json |
| `-t, --text`, `--text-file` | | Visualizer mode: the response text |
| `-c, --confidence`, `--confidence-file` | | Visualizer mode: per-token confidence JSON |
| `--verbose` | | Show detailed token information |
| `--demo` | | Built-in demo data (hand-written scores, visualizer only) |

Running with no arguments prints a short getting-started guide and exits with status 2.

## Output formats

| `--format` | What you get |
|---|---|
| `terminal` (default) | The answer as a heatmap, each flagged span with bars for the alternatives. Honors `NO_COLOR`. |
| `html` | One self-contained page (no scripts, no external assets, light and dark). Hover or focus any word for its probability and alternatives. |
| `markdown` | A compact report for a pull request, an issue, or `$GITHUB_STEP_SUMMARY`. |
| `json` | The spans, probabilities and alternatives, for scripts. |

```bash
llm-token-visualizer --logprobs-file answer.json --format html -o report.html
```

<img alt="HTML report for the eiffel sample in dark mode: the answer as a heatmap with four wavy-underlined spans, a per-token probability strip, and bars for the alternatives at each span." src="https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/docs/assets/report-html.png" width="520">

## Use it in CI or scripts

`--fail-on-flag` exits with status 2 when any span is flagged, so you can gate a pipeline on it or route flagged answers to review:

```bash
llm-token-visualizer --logprobs-file answer.json --fail-on-flag --format json -o report.json
llm-token-visualizer --logprobs-file answer.json --format markdown >> "$GITHUB_STEP_SUMMARY"
```

## Input formats

`--logprobs-file` accepts a full Chat Completions response saved as JSON, just its `logprobs` object (`{"content": [...]}`), or a bare array of `{"token", "logprob", "top_logprobs"}` entries. Use `-` to read stdin. Request completions with `"logprobs": true` and, for alternatives, `"top_logprobs": 3` or more. Special tokens such as `<|eot_id|>` are ignored.

## Live mode

`--live "<prompt>"` sends the prompt with temperature 0, `logprobs: true` and `top_logprobs: 3`, then analyzes the answer.

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | required | Bearer token for the API |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | Any server with Chat Completions and logprobs, e.g. `https://router.huggingface.co/v1` |

```bash
export OPENAI_API_KEY=sk-...
llm-token-visualizer --live "Who was the second person to walk on the Moon?" --save answer.json
```

Flags: `--model` (default `gpt-4o-mini`), `--max-tokens` (default 200), and `--save <path>` to keep the raw response so you can re-run it offline with `--logprobs-file`. Anthropic's API does not return logprobs, so Claude models cannot be analyzed this way.

The bundled samples were fetched through the Hugging Face router (`meta-llama/Llama-3.1-8B-Instruct:novita`). Build with `--no-default-features` for an offline-only binary without an HTTP client.

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

## Library use

```toml
[dependencies]
llm-token-visualizer = "0.3"
```

```rust
use llm_token_visualizer::detect::{detect, parse_logprobs};
use llm_token_visualizer::report::{self, Meta};

let tokens = parse_logprobs(&response_json)?;
let report = detect(&tokens, 0.5);
for span in &report.spans {
    println!("{}: {}", span.text, span.describe());
}

// The same result as a standalone HTML page, Markdown, or colored terminal text
let meta = Meta::from_response("answer.json", &response_json);
let html = report::html(&tokens, &report, &meta);
let md = report::markdown(&tokens, &report, &meta);
```

`detect::to_token_analysis` plus `visualize_tokens` still render detections with the visualizer's renderers, and `analyze_with_issues` and `quick_analyze` are still available. Full API docs: [docs.rs/llm-token-visualizer](https://docs.rs/llm-token-visualizer).

`cargo run --example detect` runs the detector over every bundled sample (output in the README).
