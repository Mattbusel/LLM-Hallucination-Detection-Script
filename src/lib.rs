pub mod data;
pub mod renderer;
pub mod utils;

pub use data::{TokenAnalysis, TokenInfo, TokenFlag, VisualizationConfig, ConfidenceLevel, FlagType};
pub use renderer::{Renderer, TerminalRenderer, HtmlRenderer, MarkdownRenderer};
pub use utils::{simple_tokenize, create_mock_analysis, detect_issues, AnalysisMetrics};

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

/// Quick analysis function for testing - creates mock data and visualizes
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
                TokenInfo { text: "Hello".to_string(), confidence: 0.9 },
                TokenInfo { text: " world".to_string(), confidence: 0.8 },
            ],
            flags: vec![],
        };
        
        let result = visualize_tokens("Hello world", &analysis, "markdown", None);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Hello"));
    }

    #[test]
    fn test_quick_analyze() {
        let result = quick_analyze("This is a test", "markdown");
        assert!(result.is_ok());
        assert!(result.unwrap().contains("Token Analysis"));
    }

    #[test]
    fn test_analyze_with_issues() {
        let analysis = TokenAnalysis {
            tokens: vec![
                TokenInfo { text: "good".to_string(), confidence: 0.9 },
                TokenInfo { text: "bad".to_string(), confidence: 0.1 },
            ],
            flags: vec![],
        };
        
        let (metrics, issues) = analyze_with_issues(&analysis);
        assert_eq!(metrics.total_tokens, 2);
        assert!(!issues.is_empty());
    }
}
