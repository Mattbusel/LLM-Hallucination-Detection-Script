# Changelog

## 0.6.1 (unreleased)

A Python package, and character offsets for spans. No breaking changes.

- **Python package `llm-token-visualizer`** (`import llm_token_visualizer`), built from this crate with pyo3 and maturin as one abi3 wheel per platform (CPython 3.9 and newer). No runtime Python dependencies. Docs: `docs/python.md`.
  - `score(resp, threshold=0.5, *, choice=0)` takes an OpenAI SDK `ChatCompletion` or `Completion` made with `logprobs=True`, a single `Choice` or its `logprobs`, the same response as a dict, JSON string or bytes (vLLM's OpenAI-compatible server and the other providers `parse_logprobs` reads), or a vLLM `RequestOutput` / `CompletionOutput` from offline generation. It returns a `Score` with `.spans()` (character offsets into `.text`, text, min and mean logprob, the weakest token and its alternatives), `.to_html()`, `.to_markdown()`, `.summary()`, `.to_json()`, `.flagged`, `.mean_prob` and `.perplexity`.
  - `gate(resp, threshold)` returns `True` when nothing is below the threshold, for CI and eval scripts.
  - Missing or `None` logprobs raise `NoLogprobsError` with a message saying to request `logprobs=True` and naming the choice that was scored (`choices[1]` for `choice=1`); API error bodies, invalid JSON, bytes that are not UTF-8 and malformed logprobs raise `LogprobsError`. Both are `ValueError`s. A threshold outside 0 to 1 is a `ValueError`.
  - SDK objects inside a list or dict are dumped too, so the token list `resp.choices[0].logprobs.content` (a list of `ChatCompletionTokenLogprob`) scores the same as the whole response.
  - `Span` compares equal by value (`score(r).spans() == score(r).spans()`), and is not hashable.
  - `-inf` logprobs (vLLM) are read as probability 0 instead of failing as invalid JSON.
  - The bindings are in `python/` (a separate, unpublished crate, `llm-token-visualizer-python`); `pyproject.toml` is at the repository root. Tests: `python/tests` (pytest) against hand-written responses in the OpenAI Chat Completions, Completions and vLLM shapes, plus OpenAI SDK objects when `openai` is installed.
  - `.github/workflows/python.yml` builds wheels for Linux (manylinux and musllinux, x86_64 and aarch64), macOS (x86_64 and arm64) and Windows x86_64 plus an sdist on tags named `py-v*`, tests them on Linux, macOS and Windows, and publishes to PyPI with trusted publishing.
- Packaging: the crate's `include` list anchors `README.md`, `LICENSE` and the other top-level files to the crate root. The bare patterns matched files of that name at any depth, so packaging from a checkout with a local Python virtualenv picked up its `LICENSE` and `README.md` files (`cargo package --list` now lists only the crate's own 25 files).
- **New: `Span::char_range(&tokens)`**, where a span sits in `Report::text` in characters (Unicode scalar values), the unit Python indexes strings by. `Span::start` / `end` stay token indices.

## 0.6.0 (2026-11-14)

A lean library core, a typed parse error, and a fix for `null` logprobs. This is a breaking release; migration notes are at the end of this entry.

- **Lean core.** With `default-features = false` the crate depends only on serde, serde_json, unicode-segmentation and thiserror (`cargo tree -e normal --no-default-features`), and the library builds for `wasm32-unknown-unknown`. Before, clap (with its derive macros), colored and anyhow were always pulled in, plus ureq by default.
- **New features:** `cli` (the binary: clap, anyhow), `terminal` (`report::terminal` and `renderer::TerminalRenderer`: colored) and `legacy` (the deprecated `data`, `renderer` and `utils` modules). `default = ["cli", "terminal", "legacy"]`. The `[[bin]]` requires `cli`.
- **`live` is no longer on by default.** `cargo install llm-token-visualizer --features live` keeps `--live`; the GitLab release binaries are built with it. Without it, `--live` says how to install it.
- **Typed error:** `parse_logprobs` and `interop::async_openai::tokens_from_response` return `ParseError` (`InvalidJson`, `ApiError`, `NoLogprobs`, `NullLogprobs`, `Empty`, `Malformed`) instead of `anyhow::Error`, so a caller can retry with `"logprobs": true` on `NoLogprobs` without matching on message text. The messages are the same as before, apart from small wording changes in the async-openai adapter. `ParseError` is `#[non_exhaustive]`.
- **Fixed: a `null` logprob counted as probability 0.** Legacy completions with `echo: true` return `null` for the first token; that token was always flagged with p=0.00 and made the perplexity infinite. A `null` logprob (in the legacy shape and now also in the `content` array shape, where it used to be a parse error) is kept as `NaN`: the text stays complete, and `detect` never flags the token and leaves it out of `mean_prob` and `perplexity`. New `LogprobToken::has_logprob`.
- **Hardened numerics:** `Candidate.prob` is clamped to 0..=1 (it was a bare `exp(logprob)`), alternatives with an unknown logprob are dropped, and a `NaN` token is never chosen as a span's weakest token. `detect` documents that it never panics: a threshold outside 0..=1 is clamped and a `NaN` threshold flags nothing (`Report::threshold` holds the value used). New `try_detect` returns `InvalidThreshold` for such a threshold instead.
- **`Report`, `Span` and `Candidate` are `#[non_exhaustive]` and implement `Deserialize`,** so new metrics can ship in minor releases and a stored `--format json` report can be read back. An infinite perplexity (written as `null` by serde_json) reads back as infinity. New `Candidate::new`.
- **The detect API is at the crate root:** `llm_token_visualizer::{parse_logprobs, detect, try_detect, Report, Span, LogprobToken, Alternative, Candidate, ParseError, InvalidThreshold, top_k_entropy_bits, DEFAULT_THRESHOLD}`. The `detect` module paths still work.
- **Deprecated:** the `data`, `renderer` and `utils` modules, their root re-exports, `visualize_tokens`, `analyze_with_issues` and `detect::to_token_analysis` (the 0.1 visualizer model, which the detect API and `report` replace). They are behind the `legacy` feature, on by default for this release only. The CLI's `--text` / `--confidence` / `--demo` mode still works.
- Crate docs: a "which API to use" section and a feature table. Keywords and categories now describe a library (`llm`, `logprobs`, `hallucination`, `uncertainty`, `openai`; `science`, `api-bindings`).
- Property tests for unknown logprobs: `NaN` and `null` are never flagged and never make the mean or perplexity `NaN`; perplexity equals the one over the known tokens; any threshold is safe and `try_detect` accepts exactly 0..=1; reports round-trip through JSON.
- GitLab CI checks the library with no default features, fails if clap, colored, anyhow or ureq reach the lean core, and checks it for `wasm32-unknown-unknown`.
- Dependencies: removed the unused `thiserror` 1; added `thiserror` 2 for the error types. Minimum Rust version is still 1.88 (checked with `cargo +1.88.0 check --all-targets --all-features`).

### Migrating from 0.5

- Library users who only call `parse_logprobs` / `detect`: depend with `default-features = false`.
- `parse_logprobs(..)?` inside a function returning `anyhow::Result` keeps working (`ParseError` implements `std::error::Error`). Code that named the return type as `anyhow::Result<Vec<LogprobToken>>` changes to `Result<Vec<LogprobToken>, ParseError>`.
- Building `Report`, `Span` or `Candidate` with a struct literal outside the crate no longer compiles; get them from `detect`, deserialize them, or use `Candidate::new`.
- `report::terminal` needs the `terminal` feature and `live` needs `features = ["live"]`.
- A token with a `null` logprob now has `logprob.is_nan()`; check `has_logprob()` before using its probability.

## 0.5.1 (2026-10-21)

- No code changes. Republished so the crates.io page shows the rewritten README.

## 0.5.0 (2026-10-04)

- **Fixed: answers without spaces were flagged as one span.** A word was "everything up to the next space or ASCII punctuation", so in a Chinese or Japanese answer one unsure character flagged the whole sentence. Words now come from Unicode word segmentation (UAX #29, `unicode-segmentation`). English results on the bundled samples are unchanged.
- **New metrics:** each report has the answer's perplexity (`exp(-mean logprob)`, shown in the terminal and Markdown headers and in JSON), and each flagged span the entropy in bits of the alternatives at its weakest token (`entropy_bits`, a lower bound since only the top-k are returned).
- **Google Gemini responses** (`candidates[0].logprobsResult`, from `responseLogprobs: true`) are read by `--logprobs-file` and `parse_logprobs`.
- **`async-openai` feature:** `interop::async_openai::tokens_from_response`, tested to give the same tokens as the JSON parser on a bundled sample.
- **Fixed: `--live` could hang forever** on a stalled server (ureq has no overall timeout by default); requests now time out after 120 s.
- **Faster:** parsing no longer copies the logprobs JSON before deserializing. A 4,000-token response is parsed and checked in 9.0 ms, against 14.6 ms with 0.4.0 (`benches/vs_0_4.rs`, criterion, i7-13700KF).
- `TokenAnalysis::get_confidence_stats` returned `(inf, -inf, NaN)` for an empty analysis; it returns zeros.
- `quick_analyze` and `utils::create_mock_analysis` are deprecated: their "confidence" is a hash of each word, not anything a model produced. The docs now say so.
- Property tests (proptest): the parser never panics on arbitrary text, spans are ordered, in bounds and below the threshold, and token lists round-trip through JSON. The README's Rust example runs as a doctest.
- Every public item is documented (`deny(missing_docs)`); docs.rs builds all features.
- Examples: `ci_gate` (a directory of answers, exit 2 when any is flagged) and `html_report`, next to `detect`.
- `cargo binstall llm-token-visualizer` downloads the GitLab release binary (checked with a dry run against 0.4.0).
- Removed from the repository: the `rust_mvps/` sketches, the C++ fact-graph sketch, `templates/`, `partners.md`, `sponsors.md` and the dead GitHub workflows (none were built or used). The packaged crate lists what it ships; minimum Rust version 1.88 (checked).
- README: the "try it in your browser" link pointed at a page that is not deployed (`/try/` returns 404); it now points at the page source.
- GitLab CI runs tests with all and with no default features, clippy and rustdoc with `-D warnings`, and an MSRV check.

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
