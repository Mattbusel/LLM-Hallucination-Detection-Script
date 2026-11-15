# Reference

Everything the command-line tool and the library can do. For the short version, see the [README](../README.md).

## Command-line options

Run `llm-token-visualizer --help` for the same list.

| Flag | Default | What it does |
|---|---|---|
| `--logprobs-file <PATH>` | | Detect: a response with token logprobs saved as JSON (any shape listed below), or a recorded SSE / NDJSON stream; `-` reads stdin |
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

<img alt="HTML report for the eiffel sample in dark mode: the answer as a heatmap with four wavy-underlined spans, a per-token probability strip, and bars for the alternatives at each span." src="https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/raw/main/docs/assets/report-html.png" width="520">

## Use it in CI or scripts

`--fail-on-flag` exits with status 2 when any span is flagged, so you can gate a pipeline on it or route flagged answers to review:

```bash
llm-token-visualizer --logprobs-file answer.json --fail-on-flag --format json -o report.json
llm-token-visualizer --logprobs-file answer.json --format markdown >> "$GITHUB_STEP_SUMMARY"
```

## Input formats

`--logprobs-file` (and `parse_any` in the library) reads the response as saved, whichever API it came from; the shape is detected from its keys. Use `-` to read stdin. Special tokens such as `<|eot_id|>` are ignored.

| API | Ask for logprobs with | Where they are |
|---|---|---|
| OpenAI Chat Completions, and vLLM, SGLang, Fireworks, Groq, DeepSeek, OpenRouter, Together, Ollama `/v1`, llama.cpp `/v1`, TGI `/v1` | `"logprobs": true, "top_logprobs": 3` | `choices[i].logprobs.content` |
| OpenAI legacy completions, vLLM `/v1/completions`, Together | `"logprobs": 3` | `choices[i].logprobs.tokens` / `token_logprobs` / `top_logprobs` |
| OpenAI Responses API | `"include": ["message.output_text.logprobs"], "top_logprobs": 3` | `output[].content[].logprobs` |
| Ollama `/api/chat`, `/api/generate` | `"logprobs": true, "top_logprobs": 3` | top-level `logprobs` |
| llama.cpp server `/completion` | `"n_probs": 3` (optionally `"post_sampling_probs": true`) | `completion_probabilities` |
| Google Gemini / Vertex AI `generateContent` | `"generationConfig": {"responseLogprobs": true, "logprobs": 3}` | `candidates[i].logprobsResult` |
| Hugging Face TGI `/generate` | `"parameters": {"details": true, "top_n_tokens": 3}` | `details.tokens`, `details.top_tokens` |

A bare `logprobs` object (`{"content": [...]}`) or array of `{"token", "logprob", "top_logprobs"}` entries works too. A recorded stream (the output of `curl -N` against any of these APIs with `"stream": true`: SSE `data:` lines, or Ollama's one JSON object per line) is read as well. JSON written by Python with `-Infinity` logprobs (vLLM) is accepted; `-Infinity` counts as probability 0.

## Live mode

`--live "<prompt>"` sends the prompt with temperature 0, `logprobs: true` and `top_logprobs: 3` (Together: `logprobs: 3`), then analyzes the answer.

`--provider <name>` picks a preset (default `openai`); `--base-url <URL>` overrides its address (the part before `/chat/completions`).

| `--provider` | Base URL | API key variable | Default model |
|---|---|---|---|
| `openai` | `https://api.openai.com/v1`, or `OPENAI_BASE_URL` if set | `OPENAI_API_KEY` (required) | `gpt-4o-mini` |
| `openrouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` (required) | `openai/gpt-4o-mini` |
| `together` | `https://api.together.ai/v1` | `TOGETHER_API_KEY` (required) | none |
| `vllm` | `http://localhost:8000/v1` | `VLLM_API_KEY` (optional) | none |
| `ollama` | `http://localhost:11434/v1` | none | none |

`OPENAI_BASE_URL` still works for the `openai` preset, e.g. `https://router.huggingface.co/v1`. OpenRouter requests also set `provider.require_parameters: true` so they are only routed to upstream providers that accept logprobs. If the response has no token logprobs, the command fails with a message naming the provider and model (and `--save` still writes the raw response).

```bash
export OPENAI_API_KEY=sk-...
llm-token-visualizer --live "Who was the second person to walk on the Moon?" --save answer.json
```

Flags: `--model` (default depends on the provider, see above), `--max-tokens` (default 200), and `--save <path>` to keep the raw response so you can re-run it offline with `--logprobs-file`. Anthropic's API does not return logprobs, so Claude models cannot be analyzed this way.

The bundled samples were fetched through the Hugging Face router (`meta-llama/Llama-3.1-8B-Instruct:novita`). `--live` needs the `live` feature (`cargo install llm-token-visualizer --features live`; the release binaries have it). Without it the binary is offline-only and has no HTTP client.

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
llm-token-visualizer = { version = "0.7", default-features = false }
```

```rust
use llm_token_visualizer::{detect, parse_any};
use llm_token_visualizer::report::{self, Meta};

let tokens = parse_any(&response_json)?;
let report = detect(&tokens, 0.5);
for span in &report.spans {
    println!("{}: {}", span.text, span.describe());
}

// The same result as a standalone HTML page, Markdown, or colored terminal text
let meta = Meta::from_response("answer.json", &response_json);
let html = report::html(&tokens, &report, &meta);
let md = report::markdown(&tokens, &report, &meta);
```

### Streaming

`StreamDetector` flags spans while the answer streams in. Feed it the response body as it arrives (SSE or NDJSON bytes, cut anywhere) and each call returns the spans that just became final, usually a few tokens after the model wrote them. Over the whole stream they are exactly the spans `detect` finds in the complete answer.

```rust
use llm_token_visualizer::StreamDetector;

let mut det = StreamDetector::new(0.5);
for chunk in response_body_chunks {
    for span in det.push_sse(&chunk)? {
        println!("unsure: {}", span.describe());
    }
}
for span in det.finish()? {
    println!("unsure: {}", span.describe());
}
let report = det.report(); // the same Report `detect` returns
```

`push_event` takes one chunk's JSON instead, and `push_tokens` takes tokens you already parsed. `missing_logprobs()` says whether the chunks had text but no logprobs (the request did not ask for them).

`detect::to_token_analysis` plus `visualize_tokens` still render detections with the visualizer's renderers, and `analyze_with_issues` and `quick_analyze` are still available. Full API docs: [docs.rs/llm-token-visualizer](https://docs.rs/llm-token-visualizer).

`cargo run --example detect` runs the detector over every bundled sample (output in the README).

## Python

`pip install llm-token-visualizer` gives the same detector in Python, built from this crate (no Python dependencies). `llm_token_visualizer.score(resp)` takes an OpenAI `ChatCompletion` or `Completion` made with `logprobs=True`, the same response as a dict or JSON string (vLLM and other OpenAI-compatible servers), or a vLLM `RequestOutput`, and returns the spans with character offsets, an HTML report and a summary; `gate(resp, threshold)` is the yes/no check for CI. See [python.md](python.md).
