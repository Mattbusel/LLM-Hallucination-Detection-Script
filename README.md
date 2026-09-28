# LLM Hallucination Detector

**Highlights the words an AI chatbot was unsure about, so you know which parts of its answer to double-check.**

For anyone who ships or checks LLM answers: developers, evaluators, and CI pipelines. Works with OpenAI and any OpenAI-compatible API that returns token logprobs. Rust CLI and library (`llm-token-visualizer`).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/docs/assets/hero-dark.png">
  <img alt="LLM hallucination detection on a real Llama 3.1 answer, 'Aelbert Cuyp died in 1691 in Dordrecht, Netherlands.', shown as a token confidence heatmap with Aelbert and Dordrecht flagged, and the alternatives at 'ord': ord 0.57, üsseldorf 0.39, elf 0.03." src="https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/docs/assets/hero-light.png" width="100%">
</picture>

<p align="center">
  <a href="https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://crates.io/crates/llm-token-visualizer"><img alt="crates.io" src="https://img.shields.io/crates/v/llm-token-visualizer.svg"></a>
  &nbsp;<a href="https://mattbusel.github.io/LLM-Hallucination-Detection-Script/"><b>Try it in the browser</b></a>
</p>

## Download

### [Download for Windows (.exe)](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/releases/latest/download/llm-token-visualizer-x86_64-pc-windows-msvc.exe)

It is a command-line tool: open PowerShell in your Downloads folder and run `.\llm-token-visualizer-x86_64-pc-windows-msvc.exe`. With no arguments it prints how to get started.

| Platform | Download |
|---|---|
| macOS (Apple Silicon) | [llm-token-visualizer-aarch64-apple-darwin.tar.gz](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/releases/latest/download/llm-token-visualizer-aarch64-apple-darwin.tar.gz) |
| macOS (Intel) | [llm-token-visualizer-x86_64-apple-darwin.tar.gz](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/releases/latest/download/llm-token-visualizer-x86_64-apple-darwin.tar.gz) |
| Linux (x86_64) | [llm-token-visualizer-x86_64-unknown-linux-gnu.tar.gz](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/releases/latest/download/llm-token-visualizer-x86_64-unknown-linux-gnu.tar.gz) |
| Any OS with Rust | `cargo install llm-token-visualizer` |
| Rust library | `cargo add llm-token-visualizer` |

The archives include the sample answers under `samples/`. Every release: [releases page](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/releases/latest).

## How it works

<img alt="Animated diagram of how the detector works on a real answer. Step 1: every token comes with a logprob, p equals e to the logprob, and each token is colored from red (unsure) to grey (sure); 'A' has p 0.49 and 'ord' has p 0.57. Step 2: tokens join into words and a word is flagged if any of its tokens is below the 0.6 threshold, so Aelbert and Dordrecht are flagged. Step 3: each flagged span shows the alternatives the model weighed: The 0.43 instead of A, üsseldorf 0.39 instead of ord." src="https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/docs/img/how-it-works.svg" width="100%">

When a model writes an answer, it picks each token (a word or piece of a word) from a list of candidates, each with a probability. APIs expose these as `logprobs`. Hallucinations often sit where that probability drops: a name, a date, a city the model half-remembers. This tool turns the numbers into a heatmap and flags the shaky words, with the alternatives the model almost said. No model of its own, no API calls unless you ask for `--live`.

## Examples

Real answers from Llama 3.1 8B Instruct, bundled in `examples/logprobs/`. Output of `cargo run --example detect` (threshold 0.6):

```text
examples/logprobs/cuyp.json
  answer: Aelbert Cuyp died in 1691 in Dordrecht, Netherlands.
  flagged "Aelbert": p=0.49 at "A"; model also considered "The" (0.43), "D" (0.08)
  flagged "Dordrecht": p=0.57 at "ord"; model also considered "üsseldorf" (0.39), "elf" (0.03)

examples/logprobs/tour-de-france.json
  answer: Stephen Roche won the 1987 Tour de France, riding for the Carrera Jeans-Vagabond team.
  flagged "Stephen": p=0.49 at "Stephen"; model also considered "The" (0.43), "Steven" (0.03)
  flagged "-Vagabond": p=0.57 at "-V"; model also considered "–" (0.21), " -" (0.14)
```

The Dordrecht answer is right, but the model gave Düsseldorf a 39% chance: exactly the kind of claim to double-check. The "Aelbert" flag is the model choosing between starting with the name or with "The": low probability can be about phrasing, not facts.

The terminal report for the first one (`--logprobs-file examples/logprobs/cuyp.json --threshold 0.6`):

<img alt="Terminal output of llm-token-visualizer on examples/logprobs/cuyp.json: the answer with Aelbert and Dordrecht underlined, then bars for the candidates at each weak token: A 0.49, The 0.43, D 0.08, and ord 0.57, üsseldorf 0.39, elf 0.03." src="https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/docs/assets/terminal-cuyp.png" width="720">

### What it cannot tell you

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/docs/assets/confidently-wrong-dark.png">
  <img alt="The moonwalk sample: 'Pete Conrad was the second person to walk on the Moon'. Pete Conrad is outlined as wrong but not flagged: P 0.72, ete 0.76, Conrad 1.00. Flagged instead: was 0.60, which 0.14, during 0.55." src="https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/docs/assets/confidently-wrong-light.png" width="100%">
</picture>

Models can be confidently wrong. In `moonwalk.json` the model says "Pete Conrad was the second person to walk on the Moon" (it was Buzz Aldrin) with at least 72% probability on every token of the name, so the wrong name is not flagged. Use this to decide where to look first, not to certify an answer.

## Use it in 3 steps

1. **Get a response with logprobs.** Ask your API for `"logprobs": true, "top_logprobs": 3` and save the JSON, or let the tool do it: set `OPENAI_API_KEY` and run `llm-token-visualizer --live "your question" --save answer.json`.
2. **Run it:** `llm-token-visualizer --logprobs-file answer.json --threshold 0.6`
3. **Share or gate it:** `--format html -o report.html` for a page you can send, `--format markdown` for a PR comment, `--fail-on-flag` to fail a CI job when anything is flagged.

No API key handy? Grab a sample first: `curl -LO https://raw.githubusercontent.com/Mattbusel/LLM-Hallucination-Detection-Script/main/examples/logprobs/cuyp.json`

## Documentation

| | |
|---|---|
| [Reference](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/blob/main/docs/REFERENCE.md) | Every flag, output formats (terminal, HTML, Markdown, JSON), CI use, input formats, live mode, your own confidence scores, library API |
| [How it works and repo layout](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/blob/main/docs/ARCHITECTURE.md) | The detection rules and color scale in detail, source layout, what is a sketch and what ships |
| [API docs on docs.rs](https://docs.rs/llm-token-visualizer) | The Rust library |
| [Project site](https://mattbusel.github.io/LLM-Hallucination-Detection-Script/) | Try the samples with a threshold slider |
| [Changelog](https://github.com/Mattbusel/LLM-Hallucination-Detection-Script/blob/main/CHANGELOG.md) | What changed in each release |

## License

MIT, see [LICENSE](LICENSE).
