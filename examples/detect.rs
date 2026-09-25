//! Run the detector over every bundled sample response.
//!
//!     cargo run --example detect
//!
//! The samples in examples/logprobs/ are real responses from
//! meta-llama/Llama-3.1-8B-Instruct (temperature 0, top_logprobs 3).

use llm_token_visualizer::detect::{detect, parse_logprobs};

fn main() -> anyhow::Result<()> {
    let threshold = 0.6;
    for name in ["cuyp", "tour-de-france", "eiffel", "moonwalk"] {
        let path = format!("examples/logprobs/{name}.json");
        let tokens = parse_logprobs(&std::fs::read_to_string(&path)?)?;
        let report = detect(&tokens, threshold);
        println!("{path}\n  answer: {}", report.text.trim());
        if report.spans.is_empty() {
            println!("  no spans below p={threshold}");
        }
        for span in &report.spans {
            println!("  flagged {:?}: {}", span.text.trim(), span.describe());
        }
        println!();
    }
    Ok(())
}
