# LLM Hallucination Detector: the model already told you which words it guessed

English | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

[![crates.io](https://img.shields.io/crates/v/llm-token-visualizer.svg)](https://crates.io/crates/llm-token-visualizer)

The usual way to catch a hallucination is to pay for more model calls: RAGAS faithfulness asks an LLM to check each claim ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)), and SelfCheckGPT samples the answer again and compares ([paper](https://arxiv.org/abs/2303.08896)). The model already reported how sure it was about every token, and most apps never ask: in OpenAI's Chat Completions API, `logprobs` is off unless you turn it on ([API reference](https://platform.openai.com/docs/api-reference/chat/create)). This tool reads those logprobs and shows which words of an AI answer to double-check, with no second LLM call: 9.0 ms for a 4,000-token answer.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/hero-dark.png">
  <img alt="A real Llama 3.1 answer as a confidence heatmap, with Aelbert and Dordrecht flagged." src="docs/assets/hero-light.png" width="100%">
</picture>

- Reads the model's own token logprobs.
- Highlights the words it was unsure about.
- Shows what it almost said instead.
- Outputs terminal, HTML, Markdown or JSON.
- Fails CI with `--fail-on-flag`.

## Install

```sh
cargo install --locked llm-token-visualizer
```

No Rust? [Download for Windows (.exe)](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases/permalink/latest/downloads/llm-token-visualizer-windows-x86_64.exe).

## Use

```sh
llm-token-visualizer --logprobs-file answer.json --threshold 0.6
```

[Reference](docs/REFERENCE.md) · [docs.rs](https://docs.rs/llm-token-visualizer) · [Site](https://hallucination-highlighter.vercel.app/) · [Releases](https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/releases)

License: MIT, see [LICENSE](LICENSE).

More: [docs/DETAILS.md](docs/DETAILS.md)
