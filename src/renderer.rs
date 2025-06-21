use anyhow::Result;
use colored::{ColoredString, Colorize};

use crate::data::{TokenAnalysis, TokenInfo, VisualizationConfig, ConfidenceLevel, FlagType};

pub trait Renderer {
    fn render(&self, text: &str, analysis: &TokenAnalysis, config: &VisualizationConfig) -> Result<String>;
}

pub struct TerminalRenderer;

impl TerminalRenderer {
    pub fn new() -> Self {
        Self
    }

    fn format_token(&self, token: &TokenInfo, token_index: usize, analysis: &TokenAnalysis) -> ColoredString {
        let confidence_level = ConfidenceLevel::from(token.confidence);
        let flags = analysis.get_flags_for_token(token_index);
        
        let mut formatted = match confidence_level {
            ConfidenceLevel::VeryLow => token.text.red().bold(),
            ConfidenceLevel::Low => token.text.yellow(),
            ConfidenceLevel::Medium => token.text.normal(),
            ConfidenceLevel::High => token.text.cyan(),
            ConfidenceLevel::VeryHigh => token.text.green().bold(),
        };
        
        // Apply flag formatting
        for flag in flags {
            let flag_type = FlagType::from(flag.flag.as_str());
            formatted = match flag_type {
                FlagType::Fact => formatted.on_green(),
                FlagType::Uncertain => formatted.underline(),
                FlagType::Overconfident => formatted.on_yellow(),
                FlagType::Hallucination => formatted.on_red().bold(),
                FlagType::Other(_) => formatted.italic(),
            };
        }
        
        formatted
    }
}

impl Renderer for TerminalRenderer {
    fn render(&self, _text: &str, analysis: &TokenAnalysis, config: &VisualizationConfig) -> Result<String> {
        analysis.validate().map_err(|e| anyhow::anyhow!(e))?;
        
        let mut output = String::new();
        
        // Header
        output.push_str(&format!("🔍 LLM Token Analysis\n"));
        output.push_str(&format!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"));
        
        if config.verbose {
            let (min_conf, max_conf, avg_conf) = analysis.get_confidence_stats();
            output.push_str(&format!("📊 Confidence Stats: Min: {:.2}, Max: {:.2}, Avg: {:.2}\n", min_conf, max_conf, avg_conf));
            output.push_str(&format!("🏷️  Flags: {}\n", analysis.flags.len()));
            output.push_str(&format!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"));
        }
        
        // Legend
        output.push_str("🎨 Color Legend:\n");
        output.push_str(&format!("   {} Very Low Confidence (0.0-0.3)\n", "██".red().bold()));
        output.push_str(&format!("   {} Low Confidence (0.3-0.5)\n", "██".yellow()));
        output.push_str(&format!("   {} Medium Confidence (0.5-0.7)\n", "██".normal()));
        output.push_str(&format!("   {} High Confidence (0.7-0.9)\n", "██".cyan()));
        output.push_str(&format!("   {} Very High Confidence (0.9-1.0)\n", "██".green().bold()));
        output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        
        // Render tokens
        output.push_str("📝 Token Visualization:\n\n");
        for (i, token) in analysis.tokens.iter().enumerate() {
            let formatted_token = self.format_token(token, i, analysis);
            output.push_str(&format!("{}", formatted_token));
        }
        output.push_str("\n\n");
        
        // Show token details if verbose
        if config.verbose && config.show_confidence_scores {
            output.push_str("📋 Token Details:\n");
            output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            for (i, token) in analysis.tokens.iter().enumerate() {
                let flags = analysis.get_flags_for_token(i);
                let flag_str = if flags.is_empty() {
                    "None".to_string()
                } else {
                    flags.iter().map(|f| f.flag.as_str()).collect::<Vec<_>>().join(", ")
                };
                output.push_str(&format!("{:3}: '{}' (conf: {:.3}, flags: {})\n", i, token.text, token.confidence, flag_str));
            }
            output.push_str("\n");
        }
        
        // Show flags if enabled
        if config.show_flags && !analysis.flags.is_empty() {
            output.push_str("🏷️  Flags:\n");
            output.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
            for flag in &analysis.flags {
                let token_text: String = analysis.tokens[flag.start..flag.end]
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect();
                let desc = flag.description.as_deref().unwrap_or("No description");
                output.push_str(&format!("  [{}] '{}' (tokens {}-{}): {}\n", 
                    flag.flag, token_text, flag.start, flag.end - 1, desc));
            }
        }
        
        println!("{}", output);
        Ok(output)
    }
}

pub struct HtmlRenderer;

impl HtmlRenderer {
    pub fn new() -> Self {
        Self
    }
    
    fn confidence_to_color(&self, confidence: f64) -> String {
        let level = ConfidenceLevel::from(confidence);
        match level {
            ConfidenceLevel::VeryLow => "#ff4444".to_string(),
            ConfidenceLevel::Low => "#ffaa44".to_string(),
            ConfidenceLevel::Medium => "#888888".to_string(),
            ConfidenceLevel::High => "#44aaff".to_string(),
            ConfidenceLevel::VeryHigh => "#44ff44".to_string(),
        }
    }
}

impl Renderer for HtmlRenderer {
    fn render(&self, _text: &str, analysis: &TokenAnalysis, config: &VisualizationConfig) -> Result<String> {
        analysis.validate().map_err(|e| anyhow::anyhow!(e))?;
        
        let mut html = String::new();
        
        html.push_str(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Token Analysis</title>
    <style>
        body { font-family: 'Courier New', monospace; margin: 20px; background-color: #1a1a1a; color: #ffffff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { border-bottom: 2px solid #444; padding-bottom: 10px; margin-bottom: 20px; }
        .legend { margin: 20px 0; padding: 15px; background-color: #2a2a2a; border-radius: 5px; }
        .legend-item { display: inline-block; margin-right: 20px; margin-bottom: 5px; }
        .color-box { display: inline-block; width: 20px; height: 20px; margin-right: 5px; vertical-align: middle; }
        .token-container { line-height: 1.8; font-size: 16px; margin: 20px 0; }
        .token { position: relative; padding: 2px 4px; border-radius: 3px; cursor: pointer; }
        .token:hover { opacity: 0.8; }
        .token.fact { box-shadow: 0 -3px 0 0 #44ff44 inset; }
        .token.uncertain { text-decoration: underline; }
        .token.overconfident { box-shadow: 0 -3px 0 0 #ffaa44 inset; }
        .token.hallucination { box-shadow: 0 -3px 0 0 #ff4444 inset; font-weight: bold; }
        .flags { margin-top: 30px; }
        .flag-item { background-color: #2a2a2a; padding: 10px; margin: 5px 0; border-radius: 5px; }
        .stats { margin: 20px 0; padding: 15px; background-color: #2a2a2a; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 LLM Token Analysis</h1>
        </div>
"#);
        
        if config.verbose {
            let (min_conf, max_conf, avg_conf) = analysis.get_confidence_stats();
            html.push_str(&format!(r#"
        <div class="stats">
            <h3>📊 Statistics</h3>
            <p>Confidence Range: {:.3} - {:.3} (Average: {:.3})</p>
            <p>Total Tokens: {}</p>
            <p>Total Flags: {}</p>
        </div>
"#, min_conf, max_conf, avg_conf, analysis.tokens.len(), analysis.flags.len()));
        }
        
        html.push_str(r#"
        <div class="legend">
            <h3>🎨 Confidence Legend</h3>
            <div class="legend-item"><span class="color-box" style="background-color: #ff4444;"></span>Very Low (0.0-0.3)</div>
            <div class="legend-item"><span class="color-box" style="background-color: #ffaa44;"></span>Low (0.3-0.5)</div>
            <div class="legend-item"><span class="color-box" style="background-color: #888888;"></span>Medium (0.5-0.7)</div>
            <div class="legend-item"><span class="color-box" style="background-color: #44aaff;"></span>High (0.7-0.9)</div>
            <div class="legend-item"><span class="color-box" style="background-color: #44ff44;"></span>Very High (0.9-1.0)</div>
        </div>
        
        <div class="token-container">
            <h3>📝 Token Visualization</h3>
"#);
        
        for (i, token) in analysis.tokens.iter().enumerate() {
            let color = self.confidence_to_color(token.confidence);
            let flags = analysis.get_flags_for_token(i);
            let flag_classes: Vec<String> = flags.iter().map(|f| f.flag.clone()).collect();
            let class_str = flag_classes.join(" ");
            let title = format!("Token: '{}' | Confidence: {:.3}", token.text, token.confidence);
            
            html.push_str(&format!(
                r#"<span class="token {}" style="color: {};" title="{}">{}</span>"#,
                class_str, color, title, token.text
            ));
        }
        
        html.push_str("</div>");
        
        if config.show_flags && !analysis.flags.is_empty() {
            html.push_str(r#"
        <div class="flags">
            <h3>🏷️ Flags</h3>
"#);
            
            for flag in &analysis.flags {
                let token_text: String = analysis.tokens[flag.start..flag.end]
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect();
                let desc = flag.description.as_deref().unwrap_or("No description");
                
                html.push_str(&format!(
                    r#"<div class="flag-item">
                <strong>[{}]</strong> "{}" (tokens {}-{})<br>
                <small>{}</small>
            </div>"#,
                    flag.flag, token_text, flag.start, flag.end - 1, desc
                ));
            }
            
            html.push_str("</div>");
        }
        
        html.push_str(r#"
    </div>
</body>
</html>"#);
        
        Ok(html)
    }
}

pub struct MarkdownRenderer;

impl MarkdownRenderer {
    pub fn new() -> Self {
        Self
    }
}

impl Renderer for MarkdownRenderer {
    fn render(&self, _text: &str, analysis: &TokenAnalysis, config: &VisualizationConfig) -> Result<String> {
        analysis.validate().map_err(|e| anyhow::anyhow!(e))?;
        
        let mut md = String::new();
        
        md.push_str("# 🔍 LLM Token Analysis\n\n");
        
        if config.verbose {
            let (min_conf, max_conf, avg_conf) = analysis.get_confidence_stats();
            md.push_str("## 📊 Statistics\n\n");
            md.push_str(&format!("- **Confidence Range**: {:.3} - {:.3} (Average: {:.3})\n", min_conf, max_conf, avg_conf));
            md.push_str(&format!("- **Total Tokens**: {}\n", analysis.tokens.len()));
            md.push_str(&format!("- **Total Flags**: {}\n\n", analysis.flags.len()));
        }
        
        md.push_str("## 🎨 Confidence Legend\n\n");
        md.push_str("- 🔴 Very Low (0.0-0.3)\n");
        md.push_str("- 🟡 Low (0.3-0.5)\n");
        md.push_str("- ⚪ Medium (0.5-0.7)\n");
        md.push_str("- 🔵 High (0.7-0.9)\n");
        md.push_str("- 🟢 Very High (0.9-1.0)\n\n");
        
        md.push_str("## 📝 Token Visualization\n\n");
        
        for (i, token) in analysis.tokens.iter().enumerate() {
            let confidence_level = ConfidenceLevel::from(token.confidence);
            let emoji = match confidence_level {
                ConfidenceLevel::VeryLow => "🔴",
                ConfidenceLevel::Low => "🟡",
                ConfidenceLevel::Medium => "⚪",
                ConfidenceLevel::High => "🔵",
                ConfidenceLevel::VeryHigh => "🟢",
            };
            
            let flags = analysis.get_flags_for_token(i);
            let token_display = if flags.is_empty() {
                format!("{}{}", emoji, token.text)
            } else {
                format!("{}**{}**", emoji, token.text)
            };
            
            md.push_str(&token_display);
        }
        
        md.push_str("\n\n");
        
        if config.show_flags && !analysis.flags.is_empty() {
            md.push_str("## 🏷️ Flags\n\n");
            
            for flag in &analysis.flags {
                let token_text: String = analysis.tokens[flag.start..flag.end]
                    .iter()
                    .map(|t| t.text.as_str())
                    .collect();
                let desc = flag.description.as_deref().unwrap_or("No description");
                
                md.push_str(&format!("- **[{}]** \"{}\" (tokens {}-{}): {}\n", 
                    flag.flag, token_text, flag.start, flag.end - 1, desc));
            }
        }
        
        if config.verbose && config.show_confidence_scores {
            md.push_str("\n## 📋 Detailed Token Information\n\n");
            md.push_str("| Index | Token | Confidence | Flags |\n");
            md.push_str("|-------|-------|------------|-------|\n");
            
            for (i, token) in analysis.tokens.iter().enumerate() {
                let flags = analysis.get_flags_for_token(i);
                let flag_str = if flags.is_empty() {
                    "None".to_string()
                } else {
                    flags.iter().map(|f| f.flag.as_str()).collect::<Vec<_>>().join(", ")
                };
                md.push_str(&format!("| {} | `{}` | {:.3} | {} |\n", i, token.text, token.confidence, flag_str));
            }
        }
        
        Ok(md)
    }
}
