//! Hallucination-risk detection from token log probabilities.
//!
//! The input is the `logprobs` block that OpenAI-compatible Chat Completions
//! APIs return when you ask for `"logprobs": true`. Each token's probability
//! is `exp(logprob)`. Words containing a token whose probability is below the
//! threshold are flagged, and neighbouring flagged words are merged into one
//! span. Each span reports the weakest token and the alternatives the model
//! was weighing at that point.
//!
//! Low probability is a signal, not proof: a model can be confidently wrong,
//! and it can be unsure about phrasing while the facts are right.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::data::{TokenAnalysis, TokenFlag, TokenInfo};

/// Default probability below which a token counts as low confidence.
pub const DEFAULT_THRESHOLD: f64 = 0.5;

/// Label used for spans produced by [`detect`].
pub const FLAG_LABEL: &str = "uncertain";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Alternative {
    pub token: String,
    pub logprob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LogprobToken {
    pub token: String,
    pub logprob: f64,
    #[serde(default)]
    pub top_logprobs: Vec<Alternative>,
}

impl LogprobToken {
    pub fn prob(&self) -> f64 {
        self.logprob.exp().clamp(0.0, 1.0)
    }
}

/// A run of low-confidence words.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Span {
    /// First token index (inclusive).
    pub start: usize,
    /// Last token index (exclusive).
    pub end: usize,
    pub text: String,
    /// The least likely token in the span and its probability.
    pub weakest_token: String,
    pub min_prob: f64,
    /// What the model considered at the weakest token, most likely first.
    /// Empty when the input has no `top_logprobs`.
    pub alternatives: Vec<Candidate>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Candidate {
    pub token: String,
    pub prob: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Report {
    pub threshold: f64,
    pub text: String,
    pub token_count: usize,
    pub mean_prob: f64,
    pub spans: Vec<Span>,
}

impl Report {
    pub fn flagged(&self) -> bool {
        !self.spans.is_empty()
    }
}

/// Extract the token list from any of these shapes:
/// a full Chat Completions response (`choices[0].logprobs.content`),
/// a `logprobs` object (`{"content": [...]}`), or a bare token array.
pub fn parse_logprobs(json: &str) -> Result<Vec<LogprobToken>> {
    let value: Value = serde_json::from_str(json).context("input is not valid JSON")?;
    if let Some(err) = value.get("error") {
        bail!("the API returned an error: {}", err);
    }
    let content = if value.is_array() {
        &value
    } else if let Some(c) = value.pointer("/choices/0/logprobs/content") {
        c
    } else if let Some(c) = value.get("content") {
        c
    } else {
        bail!(
            "no token logprobs found; expected choices[0].logprobs.content \
             (request the completion with \"logprobs\": true)"
        );
    };
    if content.is_null() {
        bail!("logprobs are null; request the completion with \"logprobs\": true");
    }
    let mut tokens: Vec<LogprobToken> =
        serde_json::from_value(content.clone()).context("could not read logprobs tokens")?;
    // Drop end-of-turn and other special tokens (e.g. "<|eot_id|>") that some
    // servers include in the logprobs but not in the message text.
    tokens.retain(|t| !is_special_token(&t.token));
    if tokens.is_empty() {
        bail!("logprobs contain no tokens");
    }
    Ok(tokens)
}

fn is_special_token(s: &str) -> bool {
    s.starts_with("<|") && s.ends_with("|>")
}

fn has_word_chars(s: &str) -> bool {
    s.chars().any(|c| c.is_alphanumeric())
}

/// A token starts a new word if it begins with whitespace or punctuation,
/// or if the previous token ended in one (e.g. " " followed by "198").
fn starts_word(tokens: &[LogprobToken], i: usize) -> bool {
    if i == 0 {
        return true;
    }
    let first = tokens[i].token.chars().next();
    let prev_last = tokens[i - 1].token.chars().last();
    let boundary =
        |c: Option<char>| c.is_none_or(|c| c.is_whitespace() || c.is_ascii_punctuation());
    boundary(first) || boundary(prev_last)
}

/// Group token indices into words: returns (start, end) ranges.
fn words(tokens: &[LogprobToken]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::new();
    for i in 0..tokens.len() {
        if starts_word(tokens, i) || out.is_empty() {
            out.push((i, i + 1));
        } else if let Some(last) = out.last_mut() {
            last.1 = i + 1;
        }
    }
    out
}

/// Flag low-confidence spans.
pub fn detect(tokens: &[LogprobToken], threshold: f64) -> Report {
    let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
    let mean_prob = if tokens.is_empty() {
        0.0
    } else {
        tokens.iter().map(|t| t.prob()).sum::<f64>() / tokens.len() as f64
    };

    // A word is flagged when any of its tokens with letters or digits is
    // below the threshold. Pure whitespace/punctuation never triggers a flag.
    let flagged_words: Vec<(usize, usize)> = words(tokens)
        .into_iter()
        .filter(|&(s, e)| {
            tokens[s..e]
                .iter()
                .any(|t| has_word_chars(&t.token) && t.prob() < threshold)
        })
        .collect();

    // Merge words that are adjacent, or separated only by whitespace tokens.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for (s, e) in flagged_words {
        if let Some(last) = ranges.last_mut() {
            if tokens[last.1..s].iter().all(|t| t.token.trim().is_empty()) {
                last.1 = e;
                continue;
            }
        }
        ranges.push((s, e));
    }

    let spans = ranges
        .into_iter()
        .map(|(start, end)| {
            let weakest = (start..end)
                .filter(|&i| has_word_chars(&tokens[i].token))
                .min_by(|&a, &b| tokens[a].prob().total_cmp(&tokens[b].prob()))
                .unwrap_or(start);
            let w = &tokens[weakest];
            let alternatives = w
                .top_logprobs
                .iter()
                .map(|a| Candidate {
                    token: a.token.clone(),
                    prob: a.logprob.exp(),
                })
                .collect();
            Span {
                start,
                end,
                text: tokens[start..end]
                    .iter()
                    .map(|t| t.token.as_str())
                    .collect(),
                weakest_token: w.token.clone(),
                min_prob: w.prob(),
                alternatives,
            }
        })
        .collect();

    Report {
        threshold,
        text,
        token_count: tokens.len(),
        mean_prob,
        spans,
    }
}

impl Span {
    /// One-line human description, e.g.
    /// `p=0.57 at "ord"; model also considered "üsseldorf" (0.39)`.
    pub fn describe(&self) -> String {
        let mut s = format!("p={:.2} at {:?}", self.min_prob, self.weakest_token);
        let others: Vec<String> = self
            .alternatives
            .iter()
            .filter(|c| c.token != self.weakest_token)
            .map(|c| format!("{:?} ({:.2})", c.token, c.prob))
            .collect();
        if !others.is_empty() {
            s.push_str("; model also considered ");
            s.push_str(&others.join(", "));
        }
        s
    }
}

/// Convert tokens plus a report into the visualizer's input format, so the
/// terminal, HTML and Markdown renderers can display the result.
pub fn to_token_analysis(tokens: &[LogprobToken], report: &Report) -> TokenAnalysis {
    TokenAnalysis {
        tokens: tokens
            .iter()
            .map(|t| TokenInfo {
                text: t.token.clone(),
                confidence: t.prob(),
            })
            .collect(),
        flags: report
            .spans
            .iter()
            .map(|s| TokenFlag {
                start: s.start,
                end: s.end,
                flag: FLAG_LABEL.to_string(),
                description: Some(s.describe()),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(t: &str, p: f64) -> LogprobToken {
        LogprobToken {
            token: t.to_string(),
            logprob: p.ln(),
            top_logprobs: vec![],
        }
    }

    #[test]
    fn parses_all_three_shapes() {
        let arr = r#"[{"token":"Hi","logprob":-0.1}]"#;
        let obj = r#"{"content":[{"token":"Hi","logprob":-0.1}]}"#;
        let full = r#"{"choices":[{"logprobs":{"content":[{"token":"Hi","logprob":-0.1,"top_logprobs":[]}]}}]}"#;
        for s in [arr, obj, full] {
            let t = parse_logprobs(s).unwrap();
            assert_eq!(t.len(), 1);
            assert_eq!(t[0].token, "Hi");
        }
    }

    #[test]
    fn special_tokens_are_dropped() {
        let t = parse_logprobs(
            r#"[{"token":"Hi","logprob":-0.1},{"token":"<|eot_id|>","logprob":-3.0}]"#,
        )
        .unwrap();
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn missing_or_null_logprobs_is_a_clear_error() {
        let e = parse_logprobs(r#"{"choices":[{"logprobs":null}]}"#).unwrap_err();
        assert!(e.to_string().contains("logprobs"));
        let e = parse_logprobs(r#"{"error":{"message":"bad key"}}"#).unwrap_err();
        assert!(e.to_string().contains("bad key"));
    }

    #[test]
    fn flags_whole_word_when_a_subword_token_is_low() {
        let t = vec![
            tok("In", 0.9),
            tok(" D", 0.95),
            tok("ord", 0.3),
            tok("recht", 1.0),
        ];
        let r = detect(&t, 0.5);
        assert_eq!(r.spans.len(), 1);
        assert_eq!(r.spans[0].text, " Dordrecht");
        assert_eq!((r.spans[0].start, r.spans[0].end), (1, 4));
        assert_eq!(r.spans[0].weakest_token, "ord");
    }

    #[test]
    fn punctuation_and_whitespace_never_trigger() {
        let t = vec![
            tok("Yes", 0.9),
            tok(",", 0.1),
            tok(" ", 0.1),
            tok("ok", 0.9),
        ];
        assert!(!detect(&t, 0.5).flagged());
    }

    #[test]
    fn adjacent_low_words_merge_into_one_span() {
        let t = vec![
            tok("It", 0.99),
            tok(" was", 0.2),
            tok(" Pete", 0.3),
            tok(".", 0.99),
            tok(" Then", 0.1),
        ];
        let r = detect(&t, 0.5);
        assert_eq!(r.spans.len(), 2);
        assert_eq!(r.spans[0].text, " was Pete");
        assert_eq!(r.spans[1].text, " Then");
    }

    #[test]
    fn threshold_is_strict_less_than() {
        let t = vec![tok("a", 0.5)];
        assert!(!detect(&t, 0.5).flagged());
        assert!(detect(&t, 0.51).flagged());
    }

    #[test]
    fn digits_split_by_space_token_start_a_new_word() {
        let t = vec![
            tok(" in", 0.99),
            tok(" ", 0.99),
            tok("169", 0.2),
            tok("1", 0.99),
        ];
        let r = detect(&t, 0.5);
        assert_eq!(r.spans.len(), 1);
        assert_eq!(r.spans[0].text, "1691");
    }

    #[test]
    fn describe_lists_alternatives_but_not_the_chosen_token() {
        let mut w = tok("ord", 0.57);
        w.top_logprobs = vec![
            Alternative {
                token: "ord".into(),
                logprob: 0.57f64.ln(),
            },
            Alternative {
                token: "üsseldorf".into(),
                logprob: 0.39f64.ln(),
            },
        ];
        let r = detect(&[tok(" D", 0.99), w], 0.6);
        let d = r.spans[0].describe();
        assert!(d.contains("p=0.57"), "{d}");
        assert!(d.contains("üsseldorf"), "{d}");
        assert_eq!(d.matches("\"ord\"").count(), 1, "{d}");
    }

    #[test]
    fn bundled_sample_flags_the_birthplace() {
        let json = include_str!("../examples/logprobs/cuyp.json");
        let tokens = parse_logprobs(json).unwrap();
        let r = detect(&tokens, 0.6);
        assert_eq!(
            r.text,
            "Aelbert Cuyp died in 1691 in Dordrecht, Netherlands."
        );
        assert!(r.spans.iter().any(|s| s.text == " Dordrecht"));
        let a = to_token_analysis(&tokens, &r);
        assert!(a.validate().is_ok());
    }
}
