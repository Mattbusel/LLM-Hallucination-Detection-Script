use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub text: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenFlag {
    pub start: usize,
    pub end: usize,
    pub flag: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAnalysis {
    pub tokens: Vec<TokenInfo>,
    pub flags: Vec<TokenFlag>,
}

#[derive(Debug, Clone)]
pub struct VisualizationConfig {
    pub verbose: bool,
    pub show_confidence_scores: bool,
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
pub enum ConfidenceLevel {
    VeryLow,  // 0.0 - 0.3
    Low,      // 0.3 - 0.5
    Medium,   // 0.5 - 0.7
    High,     // 0.7 - 0.9
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
pub enum FlagType {
    Fact,
    Uncertain,
    Overconfident,
    Hallucination,
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
    pub fn validate(&self) -> Result<(), String> {
        if self.tokens.is_empty() {
            return Err("No tokens provided".to_string());
        }
        
        for (i, token) in self.tokens.iter().enumerate() {
            if token.confidence < 0.0 || token.confidence > 1.0 {
                return Err(format!("Invalid confidence score for token {}: {}", i, token.confidence));
            }
        }
        
        for flag in &self.flags {
            if flag.start >= self.tokens.len() || flag.end > self.tokens.len() {
                return Err(format!("Flag span out of bounds: {} to {}", flag.start, flag.end));
            }
            if flag.start >= flag.end {
                return Err(format!("Invalid flag span: {} to {}", flag.start, flag.end));
            }
        }
        
        Ok(())
    }
    
    pub fn get_flags_for_token(&self, token_index: usize) -> Vec<&TokenFlag> {
        self.flags
            .iter()
            .filter(|flag| token_index >= flag.start && token_index < flag.end)
            .collect()
    }
    
    pub fn get_confidence_stats(&self) -> (f64, f64, f64) {
        let confidences: Vec<f64> = self.tokens.iter().map(|t| t.confidence).collect();
        let sum: f64 = confidences.iter().sum();
        let avg = sum / confidences.len() as f64;
        let min = confidences.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = confidences.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        (min, max, avg)
    }
}
