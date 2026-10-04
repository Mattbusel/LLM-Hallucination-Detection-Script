//! Write the self-contained HTML report for one saved answer to a file
//! (no scripts, no external assets; open it yourself or attach it to a CI
//! job as an artifact).
//!
//!     cargo run --example html_report -- examples/logprobs/cuyp.json report.html

use llm_token_visualizer::detect::{detect, parse_logprobs};
use llm_token_visualizer::report::{html, Meta};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "examples/logprobs/cuyp.json".into());
    let output = args.next().unwrap_or_else(|| "report.html".into());
    let json = std::fs::read_to_string(&input)?;
    let tokens = parse_logprobs(&json)?;
    let report = detect(&tokens, 0.6);
    std::fs::write(&output, html(&tokens, &report, &Meta::from_response(input.clone(), &json)))?;
    println!(
        "{} flagged span(s) in {input}; wrote {output} ({} tokens, mean p {:.2}, perplexity {:.2})",
        report.spans.len(),
        report.token_count,
        report.mean_prob,
        report.perplexity
    );
    Ok(())
}
