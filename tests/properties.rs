//! Property tests: the parser and detector never panic on arbitrary input,
//! flagged spans are well-formed, and unknown (`null` / `NaN`) logprobs are
//! never flagged and never poison the perplexity.

use llm_token_visualizer::{
    detect, parse_any, parse_logprobs, try_detect, Alternative, LogprobToken, ParseError, Report,
    StreamDetector,
};
use proptest::prelude::*;

fn token() -> impl Strategy<Value = LogprobToken> {
    (
        prop_oneof![
            ".{0,6}",
            "[ a-zA-Z]{1,5}",
            "\\PC{1,3}",
            Just("<|eot_id|>".to_string())
        ],
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
            Err(e) => prop_assert!(kept == 0 && matches!(e, ParseError::Empty), "{}", e),
        }
    }
}

/// A token whose logprob is sometimes unknown (`NaN`, what `null` parses to)
/// or `NaN` among its alternatives.
fn token_maybe_unknown() -> impl Strategy<Value = LogprobToken> {
    (
        token(),
        prop::bool::weighted(0.3),
        prop::collection::vec(
            ("[a-z ]{1,4}", prop_oneof![-10.0f64..0.0, Just(f64::NAN)]),
            0..4,
        ),
    )
        .prop_map(|(mut t, unknown, alts)| {
            if unknown {
                t.logprob = f64::NAN;
            }
            if !alts.is_empty() {
                t.top_logprobs = alts
                    .into_iter()
                    .map(|(token, logprob)| Alternative { token, logprob })
                    .collect();
            }
            t
        })
}

/// A legacy completions response with `null` in `token_logprobs` where
/// `logprob` is `None`.
fn legacy_json(tokens: &[(String, Option<f64>)]) -> String {
    let toks: Vec<&str> = tokens.iter().map(|(t, _)| t.as_str()).collect();
    let lps: Vec<Option<f64>> = tokens.iter().map(|(_, lp)| *lp).collect();
    serde_json::json!({"choices": [{"logprobs": {"tokens": toks, "token_logprobs": lps}}]})
        .to_string()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    /// Unknown (`NaN`) logprobs are never flagged, never become a span's
    /// weakest token, and do not make the perplexity or mean `NaN`.
    #[test]
    fn nan_logprobs_are_ignored(
        tokens in prop::collection::vec(token_maybe_unknown(), 0..60),
        threshold in 0.0f64..=1.0,
    ) {
        let r = detect(&tokens, threshold);
        prop_assert!(!r.mean_prob.is_nan());
        prop_assert!((0.0..=1.0).contains(&r.mean_prob), "mean {}", r.mean_prob);
        let any_pos_inf = tokens.iter().any(|t| t.logprob == f64::INFINITY);
        prop_assert!(any_pos_inf || !r.perplexity.is_nan(), "perplexity NaN");
        for s in &r.spans {
            prop_assert!(!s.min_prob.is_nan());
            prop_assert!(s.min_prob < threshold);
            for c in &s.alternatives {
                prop_assert!((0.0..=1.0).contains(&c.prob), "candidate {}", c.prob);
            }
            if let Some(h) = s.entropy_bits {
                prop_assert!(h >= 0.0 && h.is_finite());
            }
        }
        // Dropping the unknown tokens' contribution: the same perplexity as
        // computing it over the known tokens only.
        let known: Vec<f64> = tokens.iter().filter(|t| t.has_logprob()).map(|t| t.logprob).collect();
        if !known.is_empty() && known.iter().all(|lp| lp.is_finite()) {
            let want = (-(known.iter().sum::<f64>() / known.len() as f64)).exp();
            prop_assert!((r.perplexity - want).abs() <= 1e-9 * want.max(1.0));
        }
    }

    /// `null` logprobs in a legacy completions response are kept as tokens
    /// (the text is complete) but never flagged.
    #[test]
    fn null_legacy_logprobs_are_never_flagged(
        tokens in prop::collection::vec(
            ("[a-zA-Z]{1,5}", prop::option::weighted(0.6, -20.0f64..0.0)),
            1..40,
        ),
        threshold in 0.0f64..=1.0,
    ) {
        let tokens: Vec<(String, Option<f64>)> =
            tokens.into_iter().map(|(t, lp)| (format!(" {t}"), lp)).collect();
        let parsed = parse_logprobs(&legacy_json(&tokens)).unwrap();
        prop_assert_eq!(parsed.len(), tokens.len());
        for (p, (_, lp)) in parsed.iter().zip(&tokens) {
            prop_assert_eq!(p.has_logprob(), lp.is_some());
        }
        let r = detect(&parsed, threshold);
        for s in &r.spans {
            for (i, (_, lp)) in tokens.iter().enumerate().take(s.end).skip(s.start) {
                // Every word here is one token, so a span covers only
                // flagged tokens: none of them may be a null.
                prop_assert!(lp.is_some(), "null token {} flagged in {:?}", i, s);
            }
        }
        if tokens.iter().all(|(_, lp)| lp.is_none()) {
            prop_assert!(!r.flagged());
            prop_assert_eq!(r.perplexity, 1.0);
        } else {
            prop_assert!(r.perplexity.is_finite());
        }
    }

    /// `detect` never panics on any threshold, and `try_detect` accepts
    /// exactly the thresholds in 0..=1.
    #[test]
    fn any_threshold_is_safe(
        tokens in prop::collection::vec(token(), 0..20),
        threshold in prop_oneof![
            Just(f64::NAN), Just(f64::INFINITY), Just(f64::NEG_INFINITY), -5.0f64..5.0
        ],
    ) {
        let r = detect(&tokens, threshold);
        prop_assert!((0.0..=1.0).contains(&r.threshold));
        prop_assert_eq!(try_detect(&tokens, threshold).is_ok(), (0.0..=1.0).contains(&threshold));
    }

    /// A report survives a JSON round trip.
    #[test]
    fn report_round_trips_through_json(
        tokens in prop::collection::vec(token(), 0..30),
        threshold in 0.0f64..=1.0,
    ) {
        let finite: Vec<LogprobToken> = tokens.into_iter().filter(|t| t.logprob.is_finite()).collect();
        let r = detect(&finite, threshold);
        let back: Report = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        prop_assert_eq!(back.spans.len(), r.spans.len());
        prop_assert_eq!(&back.text, &r.text);
        prop_assert!((back.perplexity - r.perplexity).abs() <= 1e-9 * r.perplexity.max(1.0));
    }
}

/// Token text for stream tests: words, subwords, numbers, spaces, line
/// breaks, punctuation that can join words ("3.5", "don't"), CJK, emoji,
/// combining marks, special tokens and arbitrary characters.
fn stream_token() -> impl Strategy<Value = LogprobToken> {
    (
        prop_oneof![
            Just(" ".to_string()),
            Just("  ".to_string()),
            Just("\n".to_string()),
            Just("\r\n".to_string()),
            Just(".".to_string()),
            Just(",".to_string()),
            Just("'".to_string()),
            Just("-".to_string()),
            Just("3".to_string()),
            Just(" 16".to_string()),
            Just("e\u{301}".to_string()),
            Just("\u{301}".to_string()),
            Just("\u{200d}".to_string()),
            Just("😀".to_string()),
            Just("🇳🇱".to_string()),
            Just("埃".to_string()),
            Just("黎".to_string()),
            Just("。".to_string()),
            Just("カ".to_string()),
            Just("<|eot_id|>".to_string()),
            "[ a-z]{1,5}",
            "[ a-z0-9.,'\n-]{1,4}",
            r"\PC{1,3}",
        ],
        prop_oneof![
            4 => 0.0f64..1.0,
            1 => Just(1.0),
            1 => Just(f64::NAN),
            1 => Just(0.0),
        ],
        prop::collection::vec(("[a-z ]{1,4}", -10.0f64..0.0), 0..3),
    )
        .prop_map(|(token, p, alts)| LogprobToken {
            token,
            logprob: p.ln(),
            top_logprobs: alts
                .into_iter()
                .map(|(token, logprob)| Alternative { token, logprob })
                .collect(),
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// Feeding tokens to a `StreamDetector` in any batches gives exactly the
    /// spans and report of `detect` on the whole list, each span once.
    #[test]
    fn stream_detector_equals_detect(
        tokens in prop::collection::vec(stream_token(), 0..80),
        cuts in prop::collection::vec(0usize..80, 0..12),
        threshold in prop_oneof![0.0f64..=1.0, Just(0.5), Just(1.0)],
    ) {
        let mut cuts: Vec<usize> = cuts.into_iter().filter(|&c| c <= tokens.len()).collect();
        cuts.push(0);
        cuts.push(tokens.len());
        cuts.sort_unstable();
        let mut det = StreamDetector::new(threshold);
        let mut spans = Vec::new();
        for w in cuts.windows(2) {
            spans.extend(det.push_tokens(tokens[w[0]..w[1]].iter().cloned()));
        }
        spans.extend(det.finish().unwrap());
        let kept: Vec<LogprobToken> = tokens
            .iter()
            .filter(|t| !(t.token.starts_with("<|") && t.token.ends_with("|>")))
            .cloned()
            .collect();
        let want = detect(&kept, threshold);
        prop_assert_eq!(&spans, &want.spans);
        prop_assert_eq!(det.report(), want);
    }

    /// The same tokens sent as an SSE stream, cut at any byte, give the same
    /// spans.
    #[test]
    fn sse_bytes_cut_anywhere(
        tokens in prop::collection::vec(stream_token(), 1..30),
        size in 1usize..40,
        threshold in 0.0f64..=1.0,
    ) {
        let tokens: Vec<LogprobToken> = tokens
            .into_iter()
            .filter(|t| t.logprob.is_finite() && !(t.token.starts_with("<|") && t.token.ends_with("|>")))
            .collect();
        let mut sse = String::new();
        for t in &tokens {
            let chunk = serde_json::json!({"choices": [{"index": 0,
                "delta": {"content": t.token},
                "logprobs": {"content": [t]}}]});
            sse.push_str(&format!("data: {chunk}\n\n"));
        }
        sse.push_str("data: [DONE]\n\n");
        let mut det = StreamDetector::new(threshold);
        let mut spans = Vec::new();
        for piece in sse.as_bytes().chunks(size) {
            spans.extend(det.push_sse(piece).unwrap());
        }
        spans.extend(det.finish().unwrap());
        prop_assert!(det.is_done());
        prop_assert_eq!(det.tokens().len(), tokens.len());
        let want = detect(det.tokens(), threshold);
        prop_assert_eq!(&spans, &want.spans);
    }

    /// The stream reader never panics on arbitrary input.
    #[test]
    fn stream_reader_never_panics(s in r"(data: |\{|\[|\n|[ -~]|\PC){0,200}", size in 1usize..20) {
        let mut det = StreamDetector::new(0.5);
        for piece in s.as_bytes().chunks(size) {
            let _ = det.push_sse(piece);
        }
        let _ = det.finish();
        let _ = parse_any(&s);
    }
}
