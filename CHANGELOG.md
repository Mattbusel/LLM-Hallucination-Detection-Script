# Changelog

## Unreleased

- New: `--format html` in detect mode writes a self-contained report page (no scripts, no external assets, light and dark): the answer as a confidence heatmap, a per-token probability strip, and each flagged span with bars for the alternatives. Hover or focus any word for its probability and alternatives.
- New: `--format markdown` in detect mode writes a compact report for pull requests, issues and `$GITHUB_STEP_SUMMARY`.
- Terminal output in detect mode is now a heatmap with aligned alternative bars per span. Confident tokens keep the terminal's own color so it reads on light and dark themes; `NO_COLOR` is honored.
- `--help` lists examples. Emoji removed from the visualizer headers.
- New `report` module (`report::terminal`, `report::html`, `report::markdown`, `report::heat`).
- Project site at https://mattbusel.github.io/LLM-Hallucination-Detection-Script/.

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
