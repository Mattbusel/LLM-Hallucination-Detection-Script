use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
/// One token for the visualizer: its text and a confidence from 0 to 1.
pub struct TokenInfo {
    /// Token text, including any leading space.
    pub text: String,
    /// Confidence from 0 (unsure) to 1 (sure); for detect mode this is the token probability.
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A labelled range of tokens, `start..end` (end exclusive).
pub struct TokenFlag {
    /// First token index (inclusive).
    pub start: usize,
    /// Last token index (exclusive).
    pub end: usize,
    /// Label, e.g. `uncertain` or `fact`.
    pub flag: String,
    /// Optional explanation shown with the flag.
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// Input to the visualizer renderers: tokens with confidences and flags.
pub struct TokenAnalysis {
    /// The tokens, in order.
    pub tokens: Vec<TokenInfo>,
    /// Flagged ranges over `tokens`.
    pub flags: Vec<TokenFlag>,
}

#[derive(Debug, Clone)]
/// Rendering options for the visualizer renderers.
pub struct VisualizationConfig {
    /// Show per-token details.
    pub verbose: bool,
    /// Print confidence scores next to tokens.
    pub show_confidence_scores: bool,
    /// Print the flags section.
    pub show_flags: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            verbose: false,
            show_confidence_scores: true,
            show_flags: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Confidence bucket used for colors.
pub enum ConfidenceLevel {
    /// Below 0.3.
    VeryLow,  // 0.0 - 0.3
    /// 0.3 to 0.5.
    Low,      // 0.3 - 0.5
    /// 0.5 to 0.7.
    Medium,   // 0.5 - 0.7
    /// 0.7 to 0.9.
    High,     // 0.7 - 0.9
    /// 0.9 and above.
    VeryHigh, // 0.9 - 1.0
}

impl From<f64> for ConfidenceLevel {
    fn from(confidence: f64) -> Self {
        match confidence {
            c if c < 0.3 => ConfidenceLevel::VeryLow,
            c if c < 0.5 => ConfidenceLevel::Low,
            c if c < 0.7 => ConfidenceLevel::Medium,
            c if c < 0.9 => ConfidenceLevel::High,
            _ => ConfidenceLevel::VeryHigh,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// Kind of a [`TokenFlag`], parsed from its label.
pub enum FlagType {
    /// `fact`
    Fact,
    /// `uncertain`
    Uncertain,
    /// `overconfident`
    Overconfident,
    /// `hallucination`
    Hallucination,
    /// Any other label.
    Other(String),
}

impl From<&str> for FlagType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "fact" => FlagType::Fact,
            "uncertain" => FlagType::Uncertain,
            "overconfident" => FlagType::Overconfident,
            "hallucination" => FlagType::Hallucination,
            _ => FlagType::Other(s.to_string()),
        }
    }
}

impl TokenAnalysis {
    /// Check that there are tokens, every confidence is within 0..=1, and every flag range is inside the token list.
    pub fn validate(&self) -> Result<(), String> {
        if self.tokens.is_empty() {
            return Err("No tokens provided".to_string());
        }

        for (i, token) in self.tokens.iter().enumerate() {
            if token.confidence < 0.0 || token.confidence > 1.0 {
                return Err(format!(
                    "Invalid confidence score for token {}: {}",
                    i, token.confidence
                ));
            }
        }

        for flag in &self.flags {
            if flag.start >= self.tokens.len() || flag.end > self.tokens.len() {
                return Err(format!(
                    "Flag span out of bounds: {} to {}",
                    flag.start, flag.end
                ));
            }
            if flag.start >= flag.end {
                return Err(format!("Invalid flag span: {} to {}", flag.start, flag.end));
            }
        }

        Ok(())
    }

    /// Flags whose range covers `token_index`.
    pub fn get_flags_for_token(&self, token_index: usize) -> Vec<&TokenFlag> {
        self.flags
            .iter()
            .filter(|flag| token_index >= flag.start && token_index < flag.end)
            .collect()
    }

    /// `(min, max, mean)` confidence; `(0.0, 0.0, 0.0)` when there are no tokens.
    pub fn get_confidence_stats(&self) -> (f64, f64, f64) {
        let confidences: Vec<f64> = self.tokens.iter().map(|t| t.confidence).collect();
        if confidences.is_empty() {
            // Before 0.5.0 this returned (inf, -inf, NaN).
            return (0.0, 0.0, 0.0);
        }
        let sum: f64 = confidences.iter().sum();
        let avg = sum / confidences.len() as f64;
        let min = confidences.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = confidences.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min, max, avg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_of_empty_analysis_are_zero_not_nan() {
        let a = TokenAnalysis { tokens: vec![], flags: vec![] };
        assert_eq!(a.get_confidence_stats(), (0.0, 0.0, 0.0));
    }
}
