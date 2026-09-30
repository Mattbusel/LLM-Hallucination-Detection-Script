# Changelog

## 0.4.0 (2026-09-30)

- New: `--provider openai|openrouter|together|vllm|ollama` for `--live`, each with its default base URL, API key variable and (for openai and openrouter) a default model; `--base-url` overrides the address. Without `--provider`, `--live` behaves as before (OpenAI, `OPENAI_API_KEY`, `OPENAI_BASE_URL`).
- Together requests use Together's integer `logprobs` form; OpenRouter requests ask to be routed only to upstream providers that support the parameters.
- If a provider answers without token logprobs, `--live` now fails with a clear message (and the answer) instead of a generic parse error.
- Input: the `tokens` / `token_logprobs` / `top_logprobs` logprobs shape is read too.
- Library: `live::Provider`, `live::Target` (`resolve`, `body`, `url`), `live::fetch_target`, `live::check_has_logprobs`. `live::fetch` and `live::request_body` are unchanged.
- Ollama preset checked against a real Ollama 0.34.4 server (qwen2.5-coder:14b): logprobs and alternatives come back.
- New browser page `docs/try/` (served at /try/): the four sample answers with a threshold slider, detected with the same rules as the CLI, and an optional "ask your own question" form for OpenAI, OpenRouter, Together or a custom OpenAI-compatible URL with your own key, sent only from the browser to that provider.

## 0.3.1 (2026-09-28)

- Running with no arguments (for example by double-clicking the Windows .exe) prints a short getting-started guide instead of an error about missing flags.
- A missing `--logprobs-file` now says to check the path and where the samples are.
- docs.rs front page: one-paragraph overview and a compiling example.
- Releases also ship a plain `llm-token-visualizer-x86_64-pc-windows-msvc.exe` and versionless archives, so `releases/latest/download/...` links always work.
- README shortened; reference material moved to `docs/REFERENCE.md` and `docs/ARCHITECTURE.md`.

## 0.3.0 (2026-09-25)

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
