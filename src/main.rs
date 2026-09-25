use anyhow::{Context, Result};
use clap::Parser;
use std::io::Read;
use std::path::PathBuf;

use llm_token_visualizer::detect::{self, Report};
use llm_token_visualizer::{
    HtmlRenderer, MarkdownRenderer, Renderer, TerminalRenderer, TokenAnalysis, VisualizationConfig,
};

#[derive(Parser)]
#[command(name = "llm-token-visualizer", version)]
#[command(
    about = "Flag low-confidence spans in LLM output from token logprobs, and visualize per-token confidence"
)]
struct Args {
    /// Detect: path to a Chat Completions response (or its logprobs) saved as JSON; "-" reads stdin
    #[arg(long, value_name = "PATH")]
    logprobs_file: Option<PathBuf>,

    /// Detect: send this prompt to an OpenAI-compatible API (needs OPENAI_API_KEY) and analyze the answer
    #[arg(long, value_name = "PROMPT")]
    live: Option<String>,

    /// Model for --live
    #[arg(long, default_value = "gpt-4o-mini")]
    model: String,

    /// Max tokens for --live
    #[arg(long, default_value_t = 200)]
    max_tokens: u32,

    /// Save the raw --live response here so it can be re-analyzed offline with --logprobs-file
    #[arg(long, value_name = "PATH")]
    save: Option<PathBuf>,

    /// Detect: flag words containing a token with probability below this (0-1)
    #[arg(long, default_value_t = detect::DEFAULT_THRESHOLD)]
    threshold: f64,

    /// Detect: exit with status 2 if any span is flagged (for CI and scripts)
    #[arg(long)]
    fail_on_flag: bool,

    /// Visualize: LLM response text
    #[arg(short, long)]
    text: Option<String>,

    /// Visualize: path to text file containing LLM response
    #[arg(long)]
    text_file: Option<PathBuf>,

    /// Visualize: JSON string with token confidence data
    #[arg(short, long)]
    confidence: Option<String>,

    /// Visualize: path to JSON file with confidence data
    #[arg(long)]
    confidence_file: Option<PathBuf>,

    /// Output format: terminal, html, markdown, or json (json: detect mode only)
    #[arg(short, long, default_value = "terminal")]
    format: String,

    /// Output file path (for html/markdown/json formats)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Show detailed token information
    #[arg(long)]
    verbose: bool,

    /// Use built-in demo data (hand-written scores, visualizer only)
    #[arg(long)]
    demo: bool,
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {:#}", e);
            std::process::exit(1);
        }
    }
}

fn run() -> Result<i32> {
    let args = Args::parse();

    if !(0.0..=1.0).contains(&args.threshold) {
        anyhow::bail!("--threshold must be between 0 and 1");
    }

    if args.logprobs_file.is_some() || args.live.is_some() {
        return run_detect(&args);
    }

    let (text, token_analysis) = if args.demo {
        load_demo_data()?
    } else {
        let text = load_text(&args)?;
        let token_analysis = load_token_analysis(&args)?;
        (text, token_analysis)
    };

    render(&args, &text, &token_analysis)?;
    Ok(0)
}

fn render(args: &Args, text: &str, analysis: &TokenAnalysis) -> Result<()> {
    let config = VisualizationConfig {
        verbose: args.verbose,
        show_confidence_scores: true,
        show_flags: true,
    };

    let output = match args.format.as_str() {
        "terminal" => TerminalRenderer::new().render(text, analysis, &config)?,
        "html" => HtmlRenderer::new().render(text, analysis, &config)?,
        "markdown" => MarkdownRenderer::new().render(text, analysis, &config)?,
        "json" => anyhow::bail!("--format json needs --logprobs-file or --live"),
        _ => anyhow::bail!("Unsupported format: {}", args.format),
    };
    write_output(args, &output)
}

fn write_output(args: &Args, output: &str) -> Result<()> {
    match (&args.output, args.format.as_str()) {
        (Some(path), f) if f != "terminal" => std::fs::write(path, output)
            .with_context(|| format!("could not write {}", path.display())),
        _ => {
            println!("{}", output);
            Ok(())
        }
    }
}

fn run_detect(args: &Args) -> Result<i32> {
    let json = if let Some(prompt) = &args.live {
        let raw = fetch_live(prompt, args)?;
        if let Some(path) = &args.save {
            std::fs::write(path, &raw)
                .with_context(|| format!("could not write {}", path.display()))?;
        }
        raw
    } else {
        let path = args.logprobs_file.as_ref().expect("checked by caller");
        if path.as_os_str() == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        } else {
            std::fs::read_to_string(path)
                .with_context(|| format!("could not read {}", path.display()))?
        }
    };

    let tokens = detect::parse_logprobs(&json)?;
    let report = detect::detect(&tokens, args.threshold);

    if args.format == "json" {
        write_output(args, &serde_json::to_string_pretty(&report)?)?;
    } else {
        let analysis = detect::to_token_analysis(&tokens, &report);
        render(args, &report.text, &analysis)?;
        if args.format == "terminal" {
            println!("{}", verdict(&report));
        }
    }

    Ok(if args.fail_on_flag && report.flagged() {
        2
    } else {
        0
    })
}

fn verdict(r: &Report) -> String {
    if r.spans.is_empty() {
        format!(
            "No low-confidence spans (threshold {:.2}, {} tokens, mean p={:.2}).\n\
             This does not prove the answer is correct: models can be confidently wrong.",
            r.threshold, r.token_count, r.mean_prob
        )
    } else {
        format!(
            "{} low-confidence span(s) at threshold {:.2} ({} tokens, mean p={:.2}).\n\
             Check these claims before trusting the answer.",
            r.spans.len(),
            r.threshold,
            r.token_count,
            r.mean_prob
        )
    }
}

#[cfg(feature = "live")]
fn fetch_live(prompt: &str, args: &Args) -> Result<String> {
    llm_token_visualizer::live::fetch(prompt, &args.model, args.max_tokens)
}

#[cfg(not(feature = "live"))]
fn fetch_live(_prompt: &str, _args: &Args) -> Result<String> {
    anyhow::bail!("this build has no live mode; rebuild with the default \"live\" feature")
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
