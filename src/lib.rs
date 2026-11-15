#![cfg_attr(docsrs, feature(doc_cfg))]
//! Flag the words an LLM answer was unsure about, from the token log
//! probabilities (logprobs) that OpenAI-compatible APIs return.
//!
//! Each token's probability is `exp(logprob)`. A word is flagged when any of
//! its tokens with letters or digits falls below the threshold, neighbouring
//! flagged words merge into one span, and each span keeps the alternatives the
//! model was weighing at its weakest token.
//!
//! ```
//! use llm_token_visualizer::{detect, parse_logprobs};
//!
//! // Three real tokens from a Llama 3.1 8B answer ("... in Dordrecht").
//! let json = r#"[
//!   {"token": " D", "logprob": -0.0153},
//!   {"token": "ord", "logprob": -0.5620, "top_logprobs": [
//!     {"token": "ord", "logprob": -0.5620},
//!     {"token": "üsseldorf", "logprob": -0.9370},
//!     {"token": "elf", "logprob": -3.5620}]},
//!   {"token": "recht", "logprob": -0.0004}
//! ]"#;
//!
//! let tokens = parse_logprobs(json)?;
//! let report = detect(&tokens, 0.6);
//!
//! assert_eq!(report.spans.len(), 1);
//! assert_eq!(report.spans[0].text, " Dordrecht");
//! assert_eq!(
//!     report.spans[0].describe(),
//!     r#"p=0.57 at "ord"; model also considered "üsseldorf" (0.39), "elf" (0.03)"#
//! );
//! # Ok::<(), llm_token_visualizer::ParseError>(())
//! ```
//!
//! # Which API to use
//!
//! - [`parse_any`] (also reachable as [`parse_logprobs`]) reads a provider
//!   response into [`LogprobToken`]s, detecting its shape: OpenAI Chat
//!   Completions and the servers that copy it (vLLM, Together, Fireworks,
//!   OpenRouter, Groq, Ollama and llama.cpp `/v1`), legacy completions, the
//!   OpenAI Responses API, Ollama's native `/api/chat`, the llama.cpp
//!   server's `/completion` with `n_probs`, Google Gemini, Hugging Face TGI,
//!   or a bare token array. [`formats`] lists the shapes ([`Format`]) and
//!   has [`parse_choice`] for `n` > 1. Failures are a typed [`ParseError`],
//!   so you can tell "the request did not ask for logprobs"
//!   ([`ParseError::NoLogprobs`]) from bad JSON or a provider error.
//! - [`detect()`] turns the tokens into a [`Report`]: flagged [`Span`]s,
//!   mean probability and perplexity. It never panics; [`try_detect`] also
//!   rejects a `NaN` or out-of-range threshold.
//! - [`StreamDetector`] flags spans while a response streams in: feed it SSE
//!   or NDJSON bytes as they arrive (any of the shapes above, chunked), and
//!   each call returns the spans that just became final. Together they are
//!   exactly the spans `detect` gives for the whole answer.
//!   [`parse_stream`] reads a recorded stream into tokens.
//! - [`report`] renders a report as a self-contained HTML page or Markdown,
//!   and as a colored terminal heatmap with the `terminal` feature.
//! - [`interop`] takes tokens straight from other crates' response types.
//!
//! The `data`, `renderer` and `utils` modules (the 0.1 visualizer model) are
//! deprecated since 0.6.0 and sit behind the `legacy` feature.
//!
//! # Features
//!
//! | Feature | Default | Adds |
//! |---|---|---|
//! | `cli` | yes | the `llm-token-visualizer` binary (clap, anyhow) |
//! | `terminal` | yes | `report::terminal` and the colored terminal renderer (colored) |
//! | `legacy` | yes, for 0.6 only | the deprecated `data`, `renderer` and `utils` modules |
//! | `live` | no | `live`: call an OpenAI-compatible API (ureq) |
//! | `async-openai` | no | [`interop`] adapter for async-openai response types |
//!
//! A library that only needs detection can depend on it with
//! `default-features = false`; the only dependencies left are serde,
//! serde_json, unicode-segmentation and thiserror, and the crate builds for
//! `wasm32-unknown-unknown`:
//!
//! ```toml
//! [dependencies]
//! llm-token-visualizer = { version = "0.7", default-features = false }
//! ```
//!
//! The command-line tool is `llm-token-visualizer`
//! (`cargo install llm-token-visualizer`, plus `--features live` for
//! `--live`).
//!
//! # Python
//!
//! The same detector ships on PyPI as `llm-token-visualizer`
//! (`import llm_token_visualizer`): `score(resp)` takes an OpenAI
//! `ChatCompletion` made with `logprobs=True` (or a vLLM output) and returns
//! the spans with character offsets ([`Span::char_range`]), an HTML report and
//! a summary. The bindings are in the repository's `python/` directory.

#![deny(missing_docs)]

/// The README's Rust examples, compiled and run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

pub mod detect;
pub mod formats;
pub mod interop;
#[cfg(feature = "live")]
#[cfg_attr(docsrs, doc(cfg(feature = "live")))]
pub mod live;
pub mod report;
pub mod stream;

pub use detect::{
    detect, parse_logprobs, top_k_entropy_bits, try_detect, Alternative, Candidate,
    InvalidThreshold, LogprobToken, ParseError, Report, Span, DEFAULT_THRESHOLD,
};
pub use formats::{parse_any, parse_choice, Format};
pub use stream::{parse_stream, StreamDetector};

/// Input types for the visualizer renderers.
#[cfg(feature = "legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy")))]
// Not deprecated in this crate's own test build: the test harness names
// every test function and would warn about each one.
#[cfg_attr(
    not(test),
    deprecated(
        since = "0.6.0",
        note = "the 0.1 visualizer model; use parse_logprobs, detect and the report module"
    )
)]
#[allow(deprecated)]
pub mod data;
/// Terminal, HTML and Markdown renderers for [`TokenAnalysis`] input (visualizer mode).
#[cfg(feature = "legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy")))]
// Not deprecated in this crate's own test build: the test harness names
// every test function and would warn about each one.
#[cfg_attr(
    not(test),
    deprecated(
        since = "0.6.0",
        note = "the 0.1 visualizer model; render a Report with the report module"
    )
)]
#[allow(deprecated)]
pub mod renderer;
/// Helpers for the visualizer: a simple tokenizer, metrics and issue detection.
#[cfg(feature = "legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy")))]
// Not deprecated in this crate's own test build: the test harness names
// every test function and would warn about each one.
#[cfg_attr(
    not(test),
    deprecated(
        since = "0.6.0",
        note = "the 0.1 visualizer model; use parse_logprobs and detect"
    )
)]
#[allow(deprecated)]
pub mod utils;

#[cfg(feature = "legacy")]
#[allow(deprecated)]
pub use data::{
    ConfidenceLevel, FlagType, TokenAnalysis, TokenFlag, TokenInfo, VisualizationConfig,
};
#[cfg(all(feature = "legacy", feature = "terminal"))]
#[allow(deprecated)]
pub use renderer::TerminalRenderer;
#[cfg(feature = "legacy")]
#[allow(deprecated)]
pub use renderer::{HtmlRenderer, MarkdownRenderer, Renderer};
#[cfg(feature = "legacy")]
#[allow(deprecated)]
pub use utils::create_mock_analysis;
#[cfg(feature = "legacy")]
#[allow(deprecated)]
pub use utils::{detect_issues, simple_tokenize, AnalysisMetrics};

/// Render a [`TokenAnalysis`] as `"terminal"`, `"html"` or `"markdown"`.
///
/// # Errors
///
/// For an unknown format, for `"terminal"` without the `terminal` feature,
/// and when the analysis is inconsistent (a flag outside the token list).
#[cfg(feature = "legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy")))]
#[deprecated(
    since = "0.6.0",
    note = "the 0.1 visualizer model; use report::html, report::markdown or report::terminal on a Report"
)]
#[allow(deprecated)]
pub fn visualize_tokens(
    text: &str,
    analysis: &TokenAnalysis,
    format: &str,
    config: Option<VisualizationConfig>,
) -> anyhow::Result<String> {
    let config = config.unwrap_or_default();

    match format {
        #[cfg(feature = "terminal")]
        "terminal" => {
            let renderer = TerminalRenderer::new();
            renderer.render(text, analysis, &config)
        }
        #[cfg(not(feature = "terminal"))]
        "terminal" => Err(anyhow::anyhow!(
            "terminal output needs the \"terminal\" feature"
        )),
        "html" => {
            let renderer = HtmlRenderer::new();
            renderer.render(text, analysis, &config)
        }
        "markdown" => {
            let renderer = MarkdownRenderer::new();
            renderer.render(text, analysis, &config)
        }
        _ => Err(anyhow::anyhow!("Unsupported format: {}", format)),
    }
}

/// Renders made-up scores (see [`utils::create_mock_analysis`]); the
/// confidence values are a hash of each word, not anything a model said.
#[cfg(feature = "legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy")))]
#[deprecated(
    since = "0.5.0",
    note = "renders made-up scores; use detect::parse_logprobs and detect::detect on a real response"
)]
#[allow(deprecated)]
pub fn quick_analyze(text: &str, format: &str) -> anyhow::Result<String> {
    let analysis = utils::create_mock_analysis(text);
    let config = VisualizationConfig::default();
    visualize_tokens(text, &analysis, format, Some(config))
}

/// Metrics and plain-text issue descriptions for a [`TokenAnalysis`].
#[cfg(feature = "legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy")))]
#[deprecated(
    since = "0.6.0",
    note = "the 0.1 visualizer model; use detect, whose Report carries typed spans"
)]
#[allow(deprecated)]
pub fn analyze_with_issues(analysis: &TokenAnalysis) -> (utils::AnalysisMetrics, Vec<String>) {
    let metrics = utils::AnalysisMetrics::from_analysis(analysis);
    let issues = utils::detect_issues(analysis);
    (metrics, issues)
}

#[cfg(all(test, feature = "legacy"))]
#[allow(deprecated)]
mod tests {
    use super::*;

    #[test]
    fn test_visualize_tokens() {
        let analysis = TokenAnalysis {
            tokens: vec![
                TokenInfo {
                    text: "Hello".to_string(),
                    confidence: 0.9,
                },
                TokenInfo {
                    text: " world".to_string(),
                    confidence: 0.8,
                },
            ],
            flags: vec![],
        };

        let result = visualize_tokens("Hello world", &analysis, "markdown", None);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Hello"));
    }

    #[test]
    #[allow(deprecated)]
    fn test_quick_analyze() {
        let result = quick_analyze("This is a test", "markdown");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Token Analysis"));
    }

    #[test]
    fn test_analyze_with_issues() {
        let analysis = TokenAnalysis {
            tokens: vec![
                TokenInfo {
                    text: "good".to_string(),
                    confidence: 0.9,
                },
                TokenInfo {
                    text: "bad".to_string(),
                    confidence: 0.1,
                },
            ],
            flags: vec![],
        };

        let (metrics, issues) = analyze_with_issues(&analysis);
        assert_eq!(metrics.total_tokens, 2);
        assert!(!issues.is_empty());
    }
}
