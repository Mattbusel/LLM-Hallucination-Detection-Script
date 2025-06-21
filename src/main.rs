use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

mod data;
mod renderer;
mod utils;

use data::{TokenAnalysis, VisualizationConfig};
use renderer::{TerminalRenderer, HtmlRenderer, MarkdownRenderer};

#[derive(Parser)]
#[command(name = "llm-token-visualizer")]
#[command(about = "Visualize LLM output token-by-token to detect hallucinations and confidence shifts")]
struct Args {
    /// LLM response text
    #[arg(short, long)]
    text: Option<String>,
    
    /// Path to text file containing LLM response
    #[arg(long)]
    text_file: Option<PathBuf>,
    
    /// JSON string with token confidence data
    #[arg(short, long)]
    confidence: Option<String>,
    
    /// Path to JSON file with confidence data
    #[arg(long)]
    confidence_file: Option<PathBuf>,
    
    /// Output format: terminal, html, markdown
    #[arg(short, long, default_value = "terminal")]
    format: String,
    
    /// Output file path (for html/markdown formats)
    #[arg(short, long)]
    output: Option<PathBuf>,
    
    /// Show detailed token information
    #[arg(long)]
    verbose: bool,
    
    /// Use demo data for testing
    #[arg(long)]
    demo: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let (text, token_analysis) = if args.demo {
        load_demo_data()?
    } else {
        let text = load_text(&args)?;
        let token_analysis = load_token_analysis(&args)?;
        (text, token_analysis)
    };
    
    let config = VisualizationConfig {
        verbose: args.verbose,
        show_confidence_scores: true,
        show_flags: true,
    };
    
    match args.format.as_str() {
        "terminal" => {
            let renderer = TerminalRenderer::new();
            renderer.render(&text, &token_analysis, &config)?;
        }
        "html" => {
            let renderer = HtmlRenderer::new();
            let output = renderer.render(&text, &token_analysis, &config)?;
            if let Some(path) = args.output {
                std::fs::write(path, output)?;
            } else {
                println!("{}", output);
            }
        }
        "markdown" => {
            let renderer = MarkdownRenderer::new();
            let output = renderer.render(&text, &token_analysis, &config)?;
            if let Some(path) = args.output {
                std::fs::write(path, output)?;
            } else {
                println!("{}", output);
            }
        }
        _ => anyhow::bail!("Unsupported format: {}", args.format),
    }
    
    Ok(())
}

fn load_text(args: &Args) -> Result<String> {
    if let Some(text) = &args.text {
        Ok(text.clone())
    } else if let Some(path) = &args.text_file {
        Ok(std::fs::read_to_string(path)?)
    } else {
        anyhow::bail!("Either --text or --text-file must be provided")
    }
}

fn load_token_analysis(args: &Args) -> Result<TokenAnalysis> {
    let json_str = if let Some(confidence) = &args.confidence {
        confidence.clone()
    } else if let Some(path) = &args.confidence_file {
        std::fs::read_to_string(path)?
    } else {
        anyhow::bail!("Either --confidence or --confidence-file must be provided")
    };
    
    Ok(serde_json::from_str(&json_str)?)
}

fn load_demo_data() -> Result<(String, TokenAnalysis)> {
    let text = "The Eiffel Tower was built in 1889 and stands 324 meters tall. It's located in Paris, France, and was designed by Gustave Eiffel. The tower has three levels and receives millions of visitors each year.".to_string();
    
    let demo_json = r#"{
        "tokens": [
            {"text": "The", "confidence": 0.95},
            {"text": " Eiffel", "confidence": 0.92},
            {"text": " Tower", "confidence": 0.94},
            {"text": " was", "confidence": 0.88},
            {"text": " built", "confidence": 0.85},
            {"text": " in", "confidence": 0.91},
            {"text": " 1889", "confidence": 0.97},
            {"text": " and", "confidence": 0.89},
            {"text": " stands", "confidence": 0.86},
            {"text": " 324", "confidence": 0.98},
            {"text": " meters", "confidence": 0.96},
            {"text": " tall", "confidence": 0.87},
            {"text": ".", "confidence": 0.99},
            {"text": " It", "confidence": 0.93},
            {"text": "'s", "confidence": 0.91},
            {"text": " located", "confidence": 0.88},
            {"text": " in", "confidence": 0.94},
            {"text": " Paris", "confidence": 0.96},
            {"text": ",", "confidence": 0.99},
            {"text": " France", "confidence": 0.95},
            {"text": ",", "confidence": 0.99},
            {"text": " and", "confidence": 0.90},
            {"text": " was", "confidence": 0.87},
            {"text": " designed", "confidence": 0.89},
            {"text": " by", "confidence": 0.92},
            {"text": " Gustave", "confidence": 0.94},
            {"text": " Eiffel", "confidence": 0.96},
            {"text": ".", "confidence": 0.99},
            {"text": " The", "confidence": 0.91},
            {"text": " tower", "confidence": 0.88},
            {"text": " has", "confidence": 0.85},
            {"text": " three", "confidence": 0.93},
            {"text": " levels", "confidence": 0.89},
            {"text": " and", "confidence": 0.87},
            {"text": " receives", "confidence": 0.84},
            {"text": " millions", "confidence": 0.82},
            {"text": " of", "confidence": 0.90},
            {"text": " visitors", "confidence": 0.86},
            {"text": " each", "confidence": 0.88},
            {"text": " year", "confidence": 0.85},
            {"text": ".", "confidence": 0.99}
        ],
        "flags": [
            {"start": 6, "end": 7, "flag": "fact", "description": "Historical date - high confidence"},
            {"start": 9, "end": 11, "flag": "fact", "description": "Physical measurement - verifiable"},
            {"start": 35, "end": 36, "flag": "uncertain", "description": "Estimate without specific data"},
            {"start": 25, "end": 27, "flag": "fact", "description": "Historical attribution"}
        ]
    }"#;
    
    let token_analysis: TokenAnalysis = serde_json::from_str(demo_json)?;
    Ok((text, token_analysis))
}
