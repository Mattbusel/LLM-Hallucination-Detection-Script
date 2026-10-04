//! A CI gate over a directory of saved LLM answers: prints a Markdown report
//! for each answer that has low-confidence words and exits with status 2 if
//! any did (the same convention as `llm-token-visualizer --fail-on-flag`).
//!
//!     cargo run --example ci_gate -- examples/logprobs 0.5
//!
//! Answers are JSON files: OpenAI-style Chat Completions responses (or just
//! their logprobs), completions-style responses, or Gemini responses with
//! `logprobsResult`.

use llm_token_visualizer::detect::{detect, parse_logprobs};
use llm_token_visualizer::report::{markdown, Meta};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "examples/logprobs".into());
    let threshold: f64 = args.next().map(|s| s.parse()).transpose()?.unwrap_or(0.5);

    let mut paths: Vec<_> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();
    let mut flagged = 0;
    for path in &paths {
        let json = std::fs::read_to_string(path)?;
        let tokens = parse_logprobs(&json)?;
        let report = detect(&tokens, threshold);
        let name = path.display().to_string();
        if report.flagged() {
            flagged += 1;
            println!("{}", markdown(&tokens, &report, &Meta::from_response(name, &json)));
        } else {
            println!("PASS {name} (perplexity {:.2})", report.perplexity);
        }
    }
    println!("{} answer(s), {flagged} flagged at threshold {threshold}", paths.len());
    if flagged > 0 {
        std::process::exit(2);
    }
    Ok(())
}
