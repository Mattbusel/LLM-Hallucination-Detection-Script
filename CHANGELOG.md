# Changelog

## 0.2.0 (2026-09-25)

- New: hallucination-risk detection from token logprobs. `--logprobs-file` reads a Chat Completions response (or its `logprobs` block, or a bare token array) and flags words below `--threshold` (default 0.5), with the alternatives the model considered.
- New: `--live "<prompt>"` calls any OpenAI-compatible API (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `--model`, `--save`). Behind the default `live` feature.
- New: `--format json` report and `--fail-on-flag` (exit status 2) for scripts and CI.
- New: four real Llama 3.1 8B sample responses in `examples/logprobs/` and `cargo run --example detect`.
- Fix: `TerminalRenderer::render` no longer prints to stdout itself, so library callers of `visualize_tokens(.., "terminal", ..)` control output. The CLI prints the returned string.
- Release workflow builds Linux, macOS and Windows binaries on tag push.
- Repo: build output (`target/`) is no longer tracked; sketch directories renamed.

## 0.1.0

- Token confidence visualizer: terminal, HTML and Markdown renderers.
