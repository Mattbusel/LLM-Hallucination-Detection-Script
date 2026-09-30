use anyhow::{Context, Result};
use clap::Parser;
use std::io::Read;
use std::path::PathBuf;

use llm_token_visualizer::detect;
use llm_token_visualizer::report;
use llm_token_visualizer::{
    HtmlRenderer, MarkdownRenderer, Renderer, TerminalRenderer, TokenAnalysis, VisualizationConfig,
};

#[derive(Parser)]
#[command(name = "llm-token-visualizer", version)]
#[command(after_help = "Examples:
  llm-token-visualizer --logprobs-file answer.json
  llm-token-visualizer --logprobs-file answer.json --format html -o report.html
  llm-token-visualizer --logprobs-file answer.json --format markdown >> $GITHUB_STEP_SUMMARY
  llm-token-visualizer --live \"Who painted The Night Watch?\" --save answer.json
  llm-token-visualizer --live \"Who painted The Night Watch?\" --provider ollama --model qwen2.5-coder:14b
  llm-token-visualizer --logprobs-file answer.json --fail-on-flag --format json

Sample responses with logprobs are in examples/logprobs/ (samples/ in release archives).")]
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

    /// Provider preset for --live: openai (OPENAI_API_KEY), openrouter (OPENROUTER_API_KEY),
    /// together (TOGETHER_API_KEY), vllm (local, http://localhost:8000/v1), ollama (local, http://localhost:11434/v1)
    #[arg(long, value_name = "NAME", default_value = "openai")]
    provider: String,

    /// Override the provider's API base URL for --live (the part before /chat/completions)
    #[arg(long, value_name = "URL")]
    base_url: Option<String>,

    /// Model for --live [default: gpt-4o-mini for openai, openai/gpt-4o-mini for openrouter; required for the others]
    #[arg(long)]
    model: Option<String>,

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

    /// Output format: terminal, html (a self-contained report page), markdown, or json (json: detect mode only)
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

/// Shown when the program is started without any input, e.g. by double-clicking
/// the Windows .exe, instead of an error about missing flags.
const GETTING_STARTED: &str = "llm-token-visualizer flags the words an LLM answer was unsure about, using the
token log probabilities (logprobs) the API returns.

Try it on a real sample answer (no API key needed):
  curl -LO https://gitlab.com/mattbusel/LLM-Hallucination-Detection-Script/-/raw/main/examples/logprobs/cuyp.json
  llm-token-visualizer --logprobs-file cuyp.json --threshold 0.6

Check a live answer from any OpenAI-compatible API (set OPENAI_API_KEY first):
  llm-token-visualizer --live \"Who painted The Night Watch?\"

Or from a local Ollama, no key needed:
  llm-token-visualizer --live \"Who painted The Night Watch?\" --provider ollama --model qwen2.5-coder:14b

Run llm-token-visualizer --help for every option.";

fn run() -> Result<i32> {
    let args = Args::parse();

    let no_input = args.logprobs_file.is_none()
        && args.live.is_none()
        && args.text.is_none()
        && args.text_file.is_none()
        && args.confidence.is_none()
        && args.confidence_file.is_none()
        && !args.demo;
    if no_input {
        println!("{GETTING_STARTED}");
        return Ok(2);
    }

    if !(0.0..=1.0).contains(&args.threshold) {
        anyhow::bail!("--threshold must be between 0 and 1");
    }

    if args.live.is_none()
        && (args.base_url.is_some() || args.model.is_some() || args.provider != "openai")
    {
        eprintln!("note: --provider, --base-url and --model only apply to --live; ignoring them");
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
        fetch_live(prompt, args)?
    } else {
        let path = args.logprobs_file.as_ref().expect("checked by caller");
        if path.as_os_str() == "-" {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s)?;
            s
        } else {
            std::fs::read_to_string(path).with_context(|| {
                format!(
                    "could not read {} (check the path; samples are in examples/logprobs/)",
                    path.display()
                )
            })?
        }
    };

    let tokens = detect::parse_logprobs(&json)?;
    let report = detect::detect(&tokens, args.threshold);
    let source = match (&args.live, &args.logprobs_file) {
        (Some(_), _) => "live".to_string(),
        (None, Some(p)) if p.as_os_str() == "-" => "stdin".to_string(),
        (None, Some(p)) => p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| p.display().to_string()),
        (None, None) => unreachable!("checked by caller"),
    };
    let meta = report::Meta::from_response(source, &json);

    let output = match args.format.as_str() {
        "json" => serde_json::to_string_pretty(&report)?,
        "html" => report::html(&tokens, &report, &meta),
        "markdown" => report::markdown(&tokens, &report, &meta),
        "terminal" => {
            let mut out = report::terminal(&tokens, &report, &meta);
            if args.verbose {
                out.push_str(
                    "
  token                 p      alternatives
",
                );
                for (i, t) in tokens.iter().enumerate() {
                    let alts: Vec<String> = t
                        .top_logprobs
                        .iter()
                        .filter(|a| a.token != t.token)
                        .map(|a| format!("{:?} {:.2}", a.token, a.logprob.exp()))
                        .collect();
                    out.push_str(&format!(
                        "  {:>3} {:<18} {:.3}  {}
",
                        i,
                        format!("{:?}", t.token),
                        t.prob(),
                        alts.join(", ")
                    ));
                }
            }
            out
        }
        f => anyhow::bail!("unsupported format {f:?}; use terminal, html, markdown or json"),
    };
    write_output(args, &output)?;

    Ok(if args.fail_on_flag && report.flagged() {
        2
    } else {
        0
    })
}

#[cfg(feature = "live")]
fn fetch_live(prompt: &str, args: &Args) -> Result<String> {
    use llm_token_visualizer::live::{self, Provider, Target};
    let provider = Provider::from_name(&args.provider).ok_or_else(|| {
        let names: Vec<&str> = Provider::ALL.iter().map(|p| p.name()).collect();
        anyhow::anyhow!(
            "unknown --provider {:?}; use one of {} (or --base-url for any other OpenAI-compatible server that returns logprobs)",
            args.provider,
            names.join(", ")
        )
    })?;
    let target = Target::resolve(
        provider,
        args.base_url.as_deref(),
        args.model.as_deref(),
        |k| std::env::var(k).ok(),
    )?;
    let raw = live::fetch_target(&target, prompt, args.max_tokens)?;
    if let Some(path) = &args.save {
        std::fs::write(path, &raw)
            .with_context(|| format!("could not write {}", path.display()))?;
    }
    live::check_has_logprobs(&raw, &target)?;
    Ok(raw)
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
