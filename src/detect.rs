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
use unicode_segmentation::UnicodeSegmentation;

use crate::data::{TokenAnalysis, TokenFlag, TokenInfo};

/// Default probability below which a token counts as low confidence.
pub const DEFAULT_THRESHOLD: f64 = 0.5;

/// Label used for spans produced by [`detect`].
pub const FLAG_LABEL: &str = "uncertain";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One of the candidates the model considered at a position.
pub struct Alternative {
    /// Candidate token text.
    pub token: String,
    /// Natural-log probability.
    pub logprob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One generated token with its log probability and the alternatives returned with it.
pub struct LogprobToken {
    /// Token text, including any leading space.
    pub token: String,
    /// Natural-log probability of this token.
    pub logprob: f64,
    #[serde(default)]
    /// Top alternatives at this position, most likely first.
    pub top_logprobs: Vec<Alternative>,
}

impl LogprobToken {
    /// Probability, `exp(logprob)` clamped to 0..=1.
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
    /// The span text, tokens joined.
    pub text: String,
    /// The least likely token in the span and its probability.
    pub weakest_token: String,
    /// Probability of the weakest token.
    pub min_prob: f64,
    /// What the model considered at the weakest token, most likely first.
    /// Empty when the input has no `top_logprobs`.
    pub alternatives: Vec<Candidate>,
    /// Entropy in bits of the alternatives at the weakest token, after
    /// renormalising them to sum to 1 (only the top-k alternatives are
    /// returned, so this is a lower bound on the model's real entropy).
    /// 0 for one alternative, 1 for two equally likely ones. `None` without
    /// `top_logprobs`.
    pub entropy_bits: Option<f64>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
/// An alternative with its probability.
pub struct Candidate {
    /// Token text.
    pub token: String,
    /// Probability, `exp(logprob)`.
    pub prob: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
/// Result of [`detect`].
pub struct Report {
    /// Probability below which a word is flagged.
    pub threshold: f64,
    /// The whole answer, tokens joined.
    pub text: String,
    /// Number of tokens.
    pub token_count: usize,
    /// Mean token probability.
    pub mean_prob: f64,
    /// Perplexity of the answer, `exp(-mean logprob)`: 1.0 when the model was
    /// certain of every token, higher when it was less sure overall.
    pub perplexity: f64,
    /// Flagged spans, in order.
    pub spans: Vec<Span>,
}

impl Report {
    /// Whether any span was flagged.
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
    // Completions-style shape (`tokens` + `token_logprobs`), which some
    // servers return from Chat Completions too.
    let legacy = value
        .pointer("/choices/0/logprobs")
        .or_else(|| Some(&value).filter(|v| v.get("token_logprobs").is_some()))
        .filter(|l| l.get("tokens").is_some() && l.get("token_logprobs").is_some());
    if let Some(l) = legacy {
        let mut tokens = parse_legacy(l)?;
        tokens.retain(|t| !is_special_token(&t.token));
        if tokens.is_empty() {
            bail!("logprobs contain no tokens");
        }
        return Ok(tokens);
    }
    if let Some(tokens) = parse_gemini(&value)? {
        return Ok(tokens);
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
    // Deserialize from the borrowed value: cloning a large `content` array
    // first cost more than the detection itself.
    let mut tokens: Vec<LogprobToken> =
        Vec::<LogprobToken>::deserialize(content).context("could not read logprobs tokens")?;
    // Drop end-of-turn and other special tokens (e.g. "<|eot_id|>") that some
    // servers include in the logprobs but not in the message text.
    tokens.retain(|t| !is_special_token(&t.token));
    if tokens.is_empty() {
        bail!("logprobs contain no tokens");
    }
    Ok(tokens)
}

/// Read `{"tokens": [..], "token_logprobs": [..], "top_logprobs": [..]}`,
/// where each `top_logprobs` entry is either a `{token: logprob}` map or a
/// list of `{"token", "logprob"}` objects.
fn parse_legacy(l: &Value) -> Result<Vec<LogprobToken>> {
    let toks = l["tokens"]
        .as_array()
        .context("logprobs.tokens is not a list")?;
    let lps = l["token_logprobs"]
        .as_array()
        .context("logprobs.token_logprobs is not a list")?;
    if toks.len() != lps.len() {
        bail!("logprobs.tokens and logprobs.token_logprobs have different lengths");
    }
    let tops = l.get("top_logprobs").and_then(Value::as_array);
    let mut out = Vec::with_capacity(toks.len());
    for (i, (t, lp)) in toks.iter().zip(lps).enumerate() {
        let token = t
            .as_str()
            .context("a logprobs token is not a string")?
            .to_string();
        let logprob = lp.as_f64().unwrap_or(f64::NEG_INFINITY);
        let mut top_logprobs: Vec<Alternative> = match tops.and_then(|a| a.get(i)) {
            Some(Value::Object(m)) => m
                .iter()
                .filter_map(|(k, v)| {
                    v.as_f64().map(|lp| Alternative {
                        token: k.clone(),
                        logprob: lp,
                    })
                })
                .collect(),
            Some(v @ Value::Array(_)) => serde_json::from_value(v.clone()).unwrap_or_default(),
            _ => Vec::new(),
        };
        top_logprobs.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
        out.push(LogprobToken {
            token,
            logprob,
            top_logprobs,
        });
    }
    Ok(out)
}

/// Google Gemini's native `generateContent` response with
/// `responseLogprobs: true`: `candidates[0].logprobsResult` holds
/// `chosenCandidates` (`{token, logProbability}`) and, per position,
/// `topCandidates[i].candidates`. Also accepts a bare `logprobsResult`.
fn parse_gemini(value: &Value) -> Result<Option<Vec<LogprobToken>>> {
    let lr = value
        .pointer("/candidates/0/logprobsResult")
        .or_else(|| value.get("logprobsResult"));
    let Some(lr) = lr else {
        return Ok(None);
    };
    let chosen = lr
        .get("chosenCandidates")
        .and_then(Value::as_array)
        .context("logprobsResult has no chosenCandidates list")?;
    let tops = lr.get("topCandidates").and_then(Value::as_array);
    let cand = |c: &Value| -> Option<Alternative> {
        Some(Alternative {
            token: c.get("token")?.as_str()?.to_string(),
            logprob: c.get("logProbability")?.as_f64()?,
        })
    };
    let mut out = Vec::with_capacity(chosen.len());
    for (i, c) in chosen.iter().enumerate() {
        let Some(a) = cand(c) else {
            bail!("chosenCandidates[{i}] needs a token and a logProbability");
        };
        let mut top: Vec<Alternative> = tops
            .and_then(|t| t.get(i))
            .and_then(|t| t.get("candidates"))
            .and_then(Value::as_array)
            .map(|v| v.iter().filter_map(cand).collect())
            .unwrap_or_default();
        top.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
        out.push(LogprobToken {
            token: a.token,
            logprob: a.logprob,
            top_logprobs: top,
        });
    }
    out.retain(|t| !is_special_token(&t.token));
    if out.is_empty() {
        bail!("logprobs contain no tokens");
    }
    Ok(Some(out))
}

fn is_special_token(s: &str) -> bool {
    s.starts_with("<|") && s.ends_with("|>")
}

fn has_word_chars(s: &str) -> bool {
    s.chars().any(|c| c.is_alphanumeric())
}

/// Group token indices into words: returns (start, end) ranges.
///
/// Word boundaries come from Unicode word segmentation (UAX #29, via the
/// `unicode-segmentation` crate) over the whole answer text: a token starts
/// a new word when its first byte is on a boundary. Before 0.5.0 a word was
/// "until the next whitespace or ASCII punctuation", so in Chinese or
/// Japanese answers, which have no spaces, every token was one word and a
/// single unsure character flagged the whole sentence.
fn words(tokens: &[LogprobToken]) -> Vec<(usize, usize)> {
    let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
    // Boundaries come out in increasing order; walk them alongside the
    // token offsets.
    let mut boundaries = text.split_word_bound_indices().map(|(i, _)| i).peekable();
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut offset = 0usize;
    for (i, t) in tokens.iter().enumerate() {
        while boundaries.next_if(|&b| b < offset).is_some() {}
        let on_boundary = boundaries.peek() == Some(&offset);
        if out.is_empty() || on_boundary {
            out.push((i, i + 1));
        } else if let Some(last) = out.last_mut() {
            last.1 = i + 1;
        }
        offset += t.token.len();
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
    let perplexity = if tokens.is_empty() {
        1.0
    } else {
        let mean_lp = tokens.iter().map(|t| t.logprob).sum::<f64>() / tokens.len() as f64;
        (-mean_lp).exp()
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
            let alternatives: Vec<Candidate> = w
                .top_logprobs
                .iter()
                .map(|a| Candidate {
                    token: a.token.clone(),
                    prob: a.logprob.exp(),
                })
                .collect();
            let entropy_bits = top_k_entropy_bits(&alternatives);
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
                entropy_bits,
            }
        })
        .collect();

    Report {
        threshold,
        text,
        token_count: tokens.len(),
        mean_prob,
        perplexity,
        spans,
    }
}

/// Entropy in bits of `alternatives` renormalised to sum to 1. `None` when
/// there are none (or their probabilities sum to 0).
pub fn top_k_entropy_bits(alternatives: &[Candidate]) -> Option<f64> {
    let total: f64 = alternatives.iter().map(|c| c.prob).filter(|p| p.is_finite() && *p > 0.0).sum();
    if alternatives.is_empty() || total <= 0.0 {
        return None;
    }
    Some(
        alternatives
            .iter()
            .map(|c| c.prob / total)
            .filter(|p| p.is_finite() && *p > 0.0)
            .map(|p| -p * p.log2())
            .sum::<f64>()
            .max(0.0),
    )
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
    fn parses_tokens_and_token_logprobs_shape() {
        let map = r#"{"choices":[{"logprobs":{"tokens":["Hi","!"],"token_logprobs":[-0.1,-2.0],
            "top_logprobs":[{"Hello":-2.5,"Hi":-0.1},{"!":-2.0}]}}]}"#;
        let t = parse_logprobs(map).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].token, "Hi");
        assert_eq!(t[0].top_logprobs[0].token, "Hi");
        assert_eq!(t[0].top_logprobs[1].token, "Hello");
        let list = r#"{"choices":[{"logprobs":{"tokens":["Hi"],"token_logprobs":[-0.1],
            "top_logprobs":[[{"token":"Hi","logprob":-0.1},{"token":"Yo","logprob":-3.0}]]}}]}"#;
        let t = parse_logprobs(list).unwrap();
        assert_eq!(t[0].top_logprobs.len(), 2);
        let bad = r#"{"choices":[{"logprobs":{"tokens":["a","b"],"token_logprobs":[-0.1]}}]}"#;
        assert!(parse_logprobs(bad).is_err());
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
    fn chinese_answer_flags_the_unsure_character_not_the_sentence() {
        // "埃菲尔铁塔在巴黎。" token by token; only 黎 is unsure.
        let t = vec![
            tok("埃", 0.99),
            tok("菲", 0.99),
            tok("尔", 0.99),
            tok("铁", 0.99),
            tok("塔", 0.99),
            tok("在", 0.99),
            tok("巴", 0.95),
            tok("黎", 0.30),
            tok("。", 0.99),
        ];
        let r = detect(&t, 0.5);
        assert_eq!(r.spans.len(), 1);
        // 0.4 flagged the whole sentence: "埃菲尔铁塔在巴黎".
        assert_eq!(r.spans[0].text, "黎");
    }

    #[test]
    fn contractions_and_hyphens_follow_unicode_rules() {
        let t = vec![tok(" don", 0.9), tok("'t", 0.2), tok(" well", 0.9), tok("-", 0.9), tok("known", 0.9)];
        let r = detect(&t, 0.5);
        assert_eq!(r.spans.len(), 1);
        assert_eq!(r.spans[0].text, " don't");
    }

    #[test]
    fn perplexity_and_entropy() {
        let t = vec![tok("a", 1.0), tok("b", 1.0)];
        assert!((detect(&t, 0.5).perplexity - 1.0).abs() < 1e-12);
        let t = vec![tok("a", 0.5), tok("b", 0.5)];
        assert!((detect(&t, 0.5).perplexity - 2.0).abs() < 1e-9);
        let two = [
            Candidate { token: "x".into(), prob: 0.4 },
            Candidate { token: "y".into(), prob: 0.4 },
        ];
        assert!((top_k_entropy_bits(&two).unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(top_k_entropy_bits(&[]), None);
    }

    #[test]
    fn parses_gemini_logprobs_result() {
        let json = r#"{"candidates":[{"content":{"parts":[{"text":"Paris."}]},
            "logprobsResult":{
              "topCandidates":[
                {"candidates":[{"token":"Paris","logProbability":-0.05},{"token":"Lyon","logProbability":-3.2}]},
                {"candidates":[{"token":".","logProbability":-0.01}]}],
              "chosenCandidates":[{"token":"Paris","logProbability":-0.05},{"token":".","logProbability":-0.01}]}}]}"#;
        let t = parse_logprobs(json).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].token, "Paris");
        assert_eq!(t[0].top_logprobs[1].token, "Lyon");
        assert!(parse_logprobs(r#"{"logprobsResult":{"topCandidates":[]}}"#).is_err());
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
