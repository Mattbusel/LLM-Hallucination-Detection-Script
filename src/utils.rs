use crate::data::{TokenInfo, TokenAnalysis, TokenFlag};

/// Simple tokenizer for demo purposes - splits on whitespace and punctuation
pub fn simple_tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current_token = String::new();
    
    for ch in text.chars() {
        if ch.is_whitespace() {
            if !current_token.is_empty() {
                tokens.push(current_token.clone());
                current_token.clear();
            }
        } else if ch.is_ascii_punctuation() {
            if !current_token.is_empty() {
                tokens.push(current_token.clone());
                current_token.clear();
            }
            tokens.push(ch.to_string());
        } else {
            current_token.push(ch);
        }
    }
    
    if !current_token.is_empty() {
        tokens.push(current_token);
    }
    
    tokens
}

/// Create a basic token analysis from text with random confidence scores
/// This is useful for testing when you don't have actual LLM confidence data
pub fn create_mock_analysis(text: &str) -> TokenAnalysis {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let tokens = simple_tokenize(text);
    let mut token_infos = Vec::new();
    
    for token in tokens {
        // Use a simple hash-based "random" confidence for consistent results
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        let hash = hasher.finish();
        let confidence = 0.5 + (hash % 100) as f64 / 200.0; // Range: 0.5 to 1.0
        
        token_infos.push(TokenInfo {
            text: token,
            confidence,
        });
    }
    
    // Add some mock flags
    let mut flags = Vec::new();
    if token_infos.len() > 5 {
        flags.push(TokenFlag {
            start: 0,
            end: 2,
            flag: "fact".to_string(),
            description: Some("Opening statement".to_string()),
        });
        
        if token_infos.len() > 10 {
            flags.push(TokenFlag {
                start: token_infos.len() - 3,
                end: token_infos.len(),
                flag: "uncertain".to_string(),
                description: Some("Concluding remarks".to_string()),
            });
        }
    }
    
    TokenAnalysis {
        tokens: token_infos,
        flags,
    }
}

/// Calculate various metrics from token analysis
pub struct AnalysisMetrics {
    pub total_tokens: usize,
    pub avg_confidence: f64,
    pub min_confidence: f64,
    pub max_confidence: f64,
    pub low_confidence_tokens: usize,
    pub flagged_tokens: usize,
}

impl AnalysisMetrics {
    pub fn from_analysis(analysis: &TokenAnalysis) -> Self {
        let confidences: Vec<f64> = analysis.tokens.iter().map(|t| t.confidence).collect();
        
        let total_tokens = confidences.len();
        let avg_confidence = confidences.iter().sum::<f64>() / total_tokens as f64;
        let min_confidence = confidences.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_confidence = confidences.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low_confidence_tokens = confidences.iter().filter(|&&c| c < 0.5).count();
        
        let mut flagged_tokens = std::collections::HashSet::new();
        for flag in &analysis.flags {
            for i in flag.start..flag.end {
                flagged_tokens.insert(i);
            }
        }
        
        Self {
            total_tokens,
            avg_confidence,
            min_confidence,
            max_confidence,
            low_confidence_tokens,
            flagged_tokens: flagged_tokens.len(),
        }
    }
}

/// Detect potential issues in token analysis
pub fn detect_issues(analysis: &TokenAnalysis) -> Vec<String> {
    let mut issues = Vec::new();
    
    // Check for very low confidence tokens
    let low_conf_tokens: Vec<(usize, &TokenInfo)> = analysis.tokens
        .iter()
        .enumerate()
        .filter(|(_, token)| token.confidence < 0.3)
        .collect();
    
    if !low_conf_tokens.is_empty() {
        issues.push(format!("Found {} tokens with very low confidence (<0.3)", low_conf_tokens.len()));
    }
    
    // Check for sudden confidence drops
    for window in analysis.tokens.windows(3) {
        let confidences: Vec<f64> = window.iter().map(|t| t.confidence).collect();
        if confidences[1] < confidences[0] - 0.4 && confidences[1] < confidences[2] - 0.4 {
            issues.push("Detected sudden confidence drop (potential hallucination point)".to_string());
            break;
        }
    }
    
    // Check for overconfident sequences
    let mut overconfident_streak = 0;
    for token in &analysis.tokens {
        if token.confidence > 0.95 {
            overconfident_streak += 1;
        } else {
            if overconfident_streak > 5 {
                issues.push("Detected long sequence of overconfident tokens (potential hallucination)".to_string());
            }
            overconfident_streak = 0;
        }
    }
    
    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokenize() {
        let text = "Hello, world! This is a test.";
        let tokens = simple_tokenize(text);
        assert_eq!(tokens, vec!["Hello", ",", "world", "!", "This", "is", "a", "test", "."]);
    }

    #[test]
    fn test_create_mock_analysis() {
        let text = "This is a test sentence.";
        let analysis = create_mock_analysis(text);
        
        assert!(!analysis.tokens.is_empty());
        assert_eq!(analysis.tokens.len(), 6); // "This", "is", "a", "test", "sentence", "."
        
        for token in &analysis.tokens {
            assert!(token.confidence >= 0.5 && token.confidence <= 1.0);
        }
    }

    #[test]
    fn test_analysis_metrics() {
        let analysis = TokenAnalysis {
            tokens: vec![
                TokenInfo { text: "test1".to_string(), confidence: 0.8 },
                TokenInfo { text: "test2".to_string(), confidence: 0.3 },
                TokenInfo { text: "test3".to_string(), confidence: 0.9 },
            ],
            flags: vec![
                TokenFlag {
                    start: 0,
                    end: 2,
                    flag: "test".to_string(),
                    description: None,
                }
            ],
        };
        
        let metrics = AnalysisMetrics::from_analysis(&analysis);
        assert_eq!(metrics.total_tokens, 3);
        assert_eq!(metrics.low_confidence_tokens, 1);
        assert_eq!(metrics.flagged_tokens, 2);
        assert!((metrics.avg_confidence - 0.6667).abs() < 0.001);
    }

    #[test]
    fn test_detect_issues() {
        let analysis = TokenAnalysis {
            tokens: vec![
                TokenInfo { text: "good".to_string(), confidence: 0.8 },
                TokenInfo { text: "bad".to_string(), confidence: 0.2 },
                TokenInfo { text: "good".to_string(), confidence: 0.9 },
            ],
            flags: vec![],
        };
        
        let issues = detect_issues(&analysis);
        assert!(!issues.is_empty());
        assert!(issues.iter().any(|issue| issue.contains("very low confidence")));
    }
}
