//! Property tests: the parser and detector never panic on arbitrary input,
//! and flagged spans are well-formed.

use llm_token_visualizer::detect::{detect, parse_logprobs, Alternative, LogprobToken};
use proptest::prelude::*;

fn token() -> impl Strategy<Value = LogprobToken> {
    (
        prop_oneof![".{0,6}", "[ a-zA-Z]{1,5}", "\\PC{1,3}", Just("<|eot_id|>".to_string())],
        prop_oneof![-20.0f64..0.0, Just(0.0), Just(f64::NEG_INFINITY)],
        prop::collection::vec(("[a-z ]{1,4}", -10.0f64..0.0), 0..4),
    )
        .prop_map(|(token, logprob, alts)| LogprobToken {
            token,
            logprob,
            top_logprobs: alts
                .into_iter()
                .map(|(token, logprob)| Alternative { token, logprob })
                .collect(),
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn parser_never_panics(s in ".{0,300}") {
        let _ = parse_logprobs(&s);
    }

    #[test]
    fn spans_are_ordered_in_bounds_and_below_threshold(
        tokens in prop::collection::vec(token(), 0..60),
        threshold in 0.0f64..1.0,
    ) {
        let r = detect(&tokens, threshold);
        prop_assert_eq!(r.token_count, tokens.len());
        prop_assert!(r.perplexity.is_nan() || r.perplexity >= 1.0 - 1e-9, "perplexity {}", r.perplexity);
        let mut prev_end = 0;
        for s in &r.spans {
            prop_assert!(s.start < s.end && s.end <= tokens.len());
            prop_assert!(s.start >= prev_end, "spans overlap");
            prev_end = s.end;
            prop_assert!(s.min_prob < threshold);
            if let Some(h) = s.entropy_bits {
                prop_assert!(h >= 0.0 && h.is_finite());
            }
        }
    }

    /// Any token list serialized as an OpenAI logprobs array parses back.
    #[test]
    fn round_trips_through_json(tokens in prop::collection::vec(token(), 1..30)) {
        let finite: Vec<LogprobToken> = tokens.into_iter().filter(|t| t.logprob.is_finite()).collect();
        prop_assume!(!finite.is_empty());
        let json = serde_json::to_string(&finite).unwrap();
        let kept = finite
            .iter()
            .filter(|t| !(t.token.starts_with("<|") && t.token.ends_with("|>")))
            .count();
        match parse_logprobs(&json) {
            Ok(back) => prop_assert_eq!(back.len(), kept),
            Err(e) => prop_assert!(kept == 0 && e.to_string().contains("no tokens"), "{}", e),
        }
    }
}
