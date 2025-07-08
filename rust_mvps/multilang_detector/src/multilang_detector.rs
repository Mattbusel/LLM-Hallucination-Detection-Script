use std::collections::HashMap;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLangPattern {
    pub language: String,
    pub uncertainty_phrases: Vec<String>,
    pub overconfidence_phrases: Vec<String>,
    pub contradiction_patterns: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct MultiLanguageDetector {
    patterns: HashMap<String, MultiLangPattern>,
    language_detector: whatlang::Detector,
}

impl MultiLanguageDetector {
    pub fn new() -> Self {
        let mut patterns = HashMap::new();
        
        // English patterns
        patterns.insert("en".to_string(), MultiLangPattern {
            language: "English".to_string(),
            uncertainty_phrases: vec![
                "I think".to_string(),
                "maybe".to_string(),
                "possibly".to_string(),
                "might be".to_string(),
            ],
            overconfidence_phrases: vec![
                "definitely".to_string(),
                "absolutely".to_string(),
                "without doubt".to_string(),
                "guaranteed".to_string(),
            ],
            contradiction_patterns: vec![
                r"always.*never".to_string(),
                r"all.*none".to_string(),
            ],
        });
        
        // Spanish patterns
        patterns.insert("es".to_string(), MultiLangPattern {
            language: "Spanish".to_string(),
            uncertainty_phrases: vec![
                "creo que".to_string(),
                "tal vez".to_string(),
                "posiblemente".to_string(),
                "puede ser".to_string(),
            ],
            overconfidence_phrases: vec![
                "definitivamente".to_string(),
                "absolutamente".to_string(),
                "sin duda".to_string(),
                "garantizado".to_string(),
            ],
            contradiction_patterns: vec![
                r"siempre.*nunca".to_string(),
                r"todo.*nada".to_string(),
            ],
        });
        
        // French patterns
        patterns.insert("fr".to_string(), MultiLangPattern {
            language: "French".to_string(),
            uncertainty_phrases: vec![
                "je pense".to_string(),
                "peut-être".to_string(),
                "possiblement".to_string(),
                "il se peut".to_string(),
            ],
            overconfidence_phrases: vec![
                "définitivement".to_string(),
                "absolument".to_string(),
                "sans aucun doute".to_string(),
                "garanti".to_string(),
            ],
            contradiction_patterns: vec![
                r"toujours.*jamais".to_string(),
                r"tout.*rien".to_string(),
            ],
        });
        
        Self {
            patterns,
            language_detector: whatlang::Detector::new(),
        }
    }
    
    pub fn detect_language(&self, text: &str) -> Option<String> {
        self.language_detector.detect(text).map(|info| info.lang().code().to_string())
    }
    
    pub fn analyze_multilingual(&self, text: &str) -> Result<MultiLangDetectionResult, Box<dyn std::error::Error>> {
        let detected_lang = self.detect_language(text).unwrap_or("en".to_string());
        
        let pattern = self.patterns.get(&detected_lang)
            .ok_or("Language not supported")?;
        
        let uncertainty_score = self.count_patterns(text, &pattern.uncertainty_phrases);
        let overconfidence_score = self.count_patterns(text, &pattern.overconfidence_phrases);
        let contradiction_score = self.count_regex_patterns(text, &pattern.contradiction_patterns);
        
        let total_score = (uncertainty_score + overconfidence_score + contradiction_score) / 3.0;
        
        Ok(MultiLangDetectionResult {
            language: pattern.language.clone(),
            language_code: detected_lang,
            hallucination_probability: total_score,
            uncertainty_score,
            overconfidence_score,
            contradiction_score,
        })
    }
    
    fn count_patterns(&self, text: &str, patterns: &[String]) -> f32 {
        let text_lower = text.to_lowercase();
        let count = patterns.iter()
            .map(|pattern| text_lower.matches(&pattern.to_lowercase()).count())
            .sum::<usize>();
        
        (count as f32 / text.split_whitespace().count() as f32).min(1.0)
    }
    
    fn count_regex_patterns(&self, text: &str, patterns: &[String]) -> f32 {
        let mut total_matches = 0;
        
        for pattern in patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                total_matches += regex.find_iter(text).count();
            }
        }
        
        (total_matches as f32 / 10.0).min(1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLangDetectionResult {
    pub language: String,
    pub language_code: String,
    pub hallucination_probability: f32,
    pub uncertainty_score: f32,
    pub overconfidence_score: f32,
    pub contradiction_score: f32,
}

