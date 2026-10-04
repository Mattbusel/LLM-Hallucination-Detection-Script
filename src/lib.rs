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
//! use llm_token_visualizer::detect::{detect, parse_logprobs};
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
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! [`report`] renders the same result as a terminal heatmap, a self-contained
//! HTML page or Markdown. The command-line tool is `llm-token-visualizer`
//! (`cargo install llm-token-visualizer`).

#![deny(missing_docs)]

/// The README's Rust examples, compiled and run as doctests.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
pub struct ReadmeDoctests;

/// Input types for the visualizer renderers.
pub mod data;
pub mod detect;
pub mod interop;
#[cfg(feature = "live")]
pub mod live;
/// Terminal, HTML and Markdown renderers for [`TokenAnalysis`] input (visualizer mode).
pub mod renderer;
pub mod report;
/// Helpers for the visualizer: a simple tokenizer, metrics and issue detection.
pub mod utils;

pub use data::{
    ConfidenceLevel, FlagType, TokenAnalysis, TokenFlag, TokenInfo, VisualizationConfig,
};
pub use renderer::{HtmlRenderer, MarkdownRenderer, Renderer, TerminalRenderer};
#[allow(deprecated)]
pub use utils::create_mock_analysis;
pub use utils::{detect_issues, simple_tokenize, AnalysisMetrics};

use anyhow::Result;

/// Main visualization function that can be used by other applications
pub fn visualize_tokens(
    text: &str,
    analysis: &TokenAnalysis,
    format: &str,
    config: Option<VisualizationConfig>,
) -> Result<String> {
    let config = config.unwrap_or_default();

    match format {
        "terminal" => {
            let renderer = TerminalRenderer::new();
            renderer.render(text, analysis, &config)
        }
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
#[deprecated(
    since = "0.5.0",
    note = "renders made-up scores; use detect::parse_logprobs and detect::detect on a real response"
)]
#[allow(deprecated)]
pub fn quick_analyze(text: &str, format: &str) -> Result<String> {
    let analysis = utils::create_mock_analysis(text);
    let config = VisualizationConfig::default();
    visualize_tokens(text, &analysis, format, Some(config))
}

/// Comprehensive analysis with issue detection
pub fn analyze_with_issues(analysis: &TokenAnalysis) -> (utils::AnalysisMetrics, Vec<String>) {
    let metrics = utils::AnalysisMetrics::from_analysis(analysis);
    let issues = utils::detect_issues(analysis);
    (metrics, issues)
}

#[cfg(test)]
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
