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
//!
//! # Unknown logprobs
//!
//! Some servers return `null` for a token's logprob (legacy completions with
//! `echo: true` do this for the first prompt token). Such a token is kept, so
//! the answer text stays complete, but its `logprob` is `NaN`: [`detect`]
//! never flags it and leaves it out of [`Report::mean_prob`] and
//! [`Report::perplexity`]. See [`LogprobToken::has_logprob`].

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

/// Why [`parse_logprobs`] could not read token logprobs from a response.
///
/// Match on it to react to a specific failure, for example to retry the
/// request with `"logprobs": true` on [`ParseError::NoLogprobs`] or
/// [`ParseError::NullLogprobs`]. New variants may be added in minor releases.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ParseError {
    /// The input is not JSON.
    #[error("input is not valid JSON")]
    InvalidJson(#[source] serde_json::Error),
    /// The provider answered with an `error` object instead of a completion.
    /// Holds that object.
    #[error("the API returned an error: {0}")]
    ApiError(Value),
    /// The response has no logprobs block at all (the request did not ask
    /// for logprobs, or the model does not return them).
    #[error(
        "no token logprobs found; expected choices[0].logprobs.content \
         (request the completion with \"logprobs\": true)"
    )]
    NoLogprobs,
    /// The logprobs block is present but `null`.
    #[error("logprobs are null; request the completion with \"logprobs\": true")]
    NullLogprobs,
    /// The logprobs block holds no tokens (after dropping special tokens
    /// such as `<|eot_id|>`).
    #[error("logprobs contain no tokens")]
    Empty,
    /// The logprobs block is there but does not have the expected shape.
    /// The message says what was wrong.
    #[error("{0}")]
    Malformed(String),
}

/// A threshold that is `NaN` or outside `0..=1`, rejected by [`try_detect`].
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[error("threshold must be a probability between 0 and 1, got {0}")]
pub struct InvalidThreshold(pub f64);

/// A logprob that may be `null` in the JSON; `null` becomes `NaN`, which
/// serde_json writes back as `null`.
fn logprob_or_nan<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

/// Default probability below which a token counts as low confidence.
pub const DEFAULT_THRESHOLD: f64 = 0.5;

/// Label used for spans produced by [`detect`].
pub const FLAG_LABEL: &str = "uncertain";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One of the candidates the model considered at a position.
pub struct Alternative {
    /// Candidate token text.
    pub token: String,
    /// Natural-log probability (`NaN` when the response had `null`).
    #[serde(deserialize_with = "logprob_or_nan")]
    pub logprob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
/// One generated token with its log probability and the alternatives returned with it.
pub struct LogprobToken {
    /// Token text, including any leading space.
    pub token: String,
    /// Natural-log probability of this token, or `NaN` when the response had
    /// `null` (see the [module docs](self#unknown-logprobs)).
    #[serde(deserialize_with = "logprob_or_nan")]
    pub logprob: f64,
    #[serde(default)]
    /// Top alternatives at this position, most likely first.
    pub top_logprobs: Vec<Alternative>,
}

impl LogprobToken {
    /// Probability, `exp(logprob)` clamped to 0..=1. `NaN` when the logprob
    /// is unknown.
    pub fn prob(&self) -> f64 {
        self.logprob.exp().clamp(0.0, 1.0)
    }

    /// Whether the response gave this token a logprob (`false` for `null`,
    /// which is stored as `NaN`). Tokens without one are never flagged and do
    /// not count towards the mean probability or perplexity.
    pub fn has_logprob(&self) -> bool {
        !self.logprob.is_nan()
    }
}

/// A run of low-confidence words.
///
/// `#[non_exhaustive]`: new metrics can be added in minor releases, so build
/// one with [`detect`] (or deserialize a stored one) rather than by hand.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// An alternative with its probability.
pub struct Candidate {
    /// Token text.
    pub token: String,
    /// Probability, `exp(logprob)` clamped to 0..=1.
    pub prob: f64,
}

impl Candidate {
    /// A candidate with `prob` clamped to 0..=1 (`NaN` stays `NaN`).
    pub fn new(token: impl Into<String>, prob: f64) -> Self {
        Candidate {
            token: token.into(),
            prob: prob.clamp(0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
/// Result of [`detect`].
///
/// Serializes to JSON (the CLI's `--format json`) and reads back. A
/// non-finite `perplexity` (an answer with a token of probability 0) is
/// written as `null` and read back as infinity.
pub struct Report {
    /// Probability below which a word is flagged.
    pub threshold: f64,
    /// The whole answer, tokens joined.
    pub text: String,
    /// Number of tokens.
    pub token_count: usize,
    /// Mean token probability, over tokens that have a logprob.
    pub mean_prob: f64,
    /// Perplexity of the answer, `exp(-mean logprob)` over tokens that have a
    /// logprob: 1.0 when the model was certain of every token, higher when it
    /// was less sure overall.
    #[serde(deserialize_with = "perplexity_or_inf")]
    pub perplexity: f64,
    /// Flagged spans, in order.
    pub spans: Vec<Span>,
}

/// serde_json writes an infinite perplexity as `null`.
fn perplexity_or_inf<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::INFINITY))
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
/// Also reads the legacy completions shape (`tokens` + `token_logprobs`)
/// and Gemini's `logprobsResult`.
///
/// Special tokens such as `<|eot_id|>` are dropped. A `null` logprob is kept
/// as `NaN` (see the [module docs](self#unknown-logprobs)).
///
/// # Errors
///
/// A [`ParseError`] saying what was wrong, for example
/// [`ParseError::NoLogprobs`] when the request did not ask for logprobs.
///
/// ```
/// use llm_token_visualizer::{parse_logprobs, ParseError};
///
/// let no_logprobs = r#"{"choices":[{"message":{"content":"Paris"}}]}"#;
/// assert!(matches!(parse_logprobs(no_logprobs), Err(ParseError::NoLogprobs)));
/// ```
pub fn parse_logprobs(json: &str) -> Result<Vec<LogprobToken>, ParseError> {
    let value: Value = serde_json::from_str(json).map_err(ParseError::InvalidJson)?;
    if let Some(err) = value.get("error") {
        return Err(ParseError::ApiError(err.clone()));
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
            return Err(ParseError::Empty);
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
    } else if value
        .pointer("/choices/0/logprobs")
        .is_some_and(Value::is_null)
    {
        return Err(ParseError::NullLogprobs);
    } else {
        return Err(ParseError::NoLogprobs);
    };
    if content.is_null() {
        return Err(ParseError::NullLogprobs);
    }
    // Deserialize from the borrowed value: cloning a large `content` array
    // first cost more than the detection itself.
    let mut tokens: Vec<LogprobToken> = Vec::<LogprobToken>::deserialize(content)
        .map_err(|e| ParseError::Malformed(format!("could not read logprobs tokens: {e}")))?;
    // Drop end-of-turn and other special tokens (e.g. "<|eot_id|>") that some
    // servers include in the logprobs but not in the message text.
    tokens.retain(|t| !is_special_token(&t.token));
    if tokens.is_empty() {
        return Err(ParseError::Empty);
    }
    Ok(tokens)
}

/// Read `{"tokens": [..], "token_logprobs": [..], "top_logprobs": [..]}`,
/// where each `top_logprobs` entry is either a `{token: logprob}` map or a
/// list of `{"token", "logprob"}` objects.
///
/// A `null` entry in `token_logprobs` (the first token of an `echo: true`
/// completion) becomes `NaN`, so [`detect`] skips it. Before 0.6.0 it became
/// `-inf`: probability 0, always flagged, and an infinite perplexity.
fn parse_legacy(l: &Value) -> Result<Vec<LogprobToken>, ParseError> {
    let malformed = |m: &str| ParseError::Malformed(m.to_string());
    let toks = l["tokens"]
        .as_array()
        .ok_or_else(|| malformed("logprobs.tokens is not a list"))?;
    let lps = l["token_logprobs"]
        .as_array()
        .ok_or_else(|| malformed("logprobs.token_logprobs is not a list"))?;
    if toks.len() != lps.len() {
        return Err(malformed(
            "logprobs.tokens and logprobs.token_logprobs have different lengths",
        ));
    }
    let tops = l.get("top_logprobs").and_then(Value::as_array);
    let mut out = Vec::with_capacity(toks.len());
    for (i, (t, lp)) in toks.iter().zip(lps).enumerate() {
        let token = t
            .as_str()
            .ok_or_else(|| malformed("a logprobs token is not a string"))?
            .to_string();
        let logprob = lp.as_f64().unwrap_or(f64::NAN);
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
fn parse_gemini(value: &Value) -> Result<Option<Vec<LogprobToken>>, ParseError> {
    let lr = value
        .pointer("/candidates/0/logprobsResult")
        .or_else(|| value.get("logprobsResult"));
    let Some(lr) = lr else {
        return Ok(None);
    };
    let chosen = lr
        .get("chosenCandidates")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ParseError::Malformed("logprobsResult has no chosenCandidates list".to_string())
        })?;
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
            return Err(ParseError::Malformed(format!(
                "chosenCandidates[{i}] needs a token and a logProbability"
            )));
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
        return Err(ParseError::Empty);
    }
    Ok(Some(out))
}

pub(crate) fn is_special_token(s: &str) -> bool {
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

/// Flag low-confidence spans: words with a token whose probability is
/// strictly below `threshold`.
///
/// Never panics, whatever the input. A `threshold` outside `0..=1` is
/// clamped into it and a `NaN` threshold flags nothing; the threshold that
/// was used is in [`Report::threshold`]. Use [`try_detect`] to get an error
/// for such a threshold instead. Tokens without a logprob (`NaN`) are never
/// flagged and are left out of the mean probability and perplexity.
pub fn detect(tokens: &[LogprobToken], threshold: f64) -> Report {
    let threshold = if threshold.is_nan() {
        0.0
    } else {
        threshold.clamp(0.0, 1.0)
    };
    let text: String = tokens.iter().map(|t| t.token.as_str()).collect();
    let known: Vec<f64> = tokens
        .iter()
        .filter(|t| t.has_logprob())
        .map(|t| t.logprob)
        .collect();
    let mean_prob = if known.is_empty() {
        0.0
    } else {
        known.iter().map(|lp| lp.exp().clamp(0.0, 1.0)).sum::<f64>() / known.len() as f64
    };
    let perplexity = if known.is_empty() {
        1.0
    } else {
        let mean_lp = known.iter().sum::<f64>() / known.len() as f64;
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
                .filter(|&i| has_word_chars(&tokens[i].token) && tokens[i].has_logprob())
                .min_by(|&a, &b| tokens[a].prob().total_cmp(&tokens[b].prob()))
                .unwrap_or(start);
            let w = &tokens[weakest];
            let alternatives: Vec<Candidate> = w
                .top_logprobs
                .iter()
                .filter(|a| !a.logprob.is_nan())
                .map(|a| Candidate::new(a.token.clone(), a.logprob.exp()))
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

/// Like [`detect`], but returns an error for a threshold that is `NaN` or
/// outside `0..=1` instead of clamping it.
///
/// ```
/// use llm_token_visualizer::{try_detect, InvalidThreshold};
///
/// assert_eq!(try_detect(&[], 1.5).unwrap_err(), InvalidThreshold(1.5));
/// assert!(try_detect(&[], 0.5).is_ok());
/// ```
pub fn try_detect(tokens: &[LogprobToken], threshold: f64) -> Result<Report, InvalidThreshold> {
    if (0.0..=1.0).contains(&threshold) {
        Ok(detect(tokens, threshold))
    } else {
        Err(InvalidThreshold(threshold))
    }
}

/// Entropy in bits of `alternatives` renormalised to sum to 1. `None` when
/// there are none (or their probabilities sum to 0).
pub fn top_k_entropy_bits(alternatives: &[Candidate]) -> Option<f64> {
    let total: f64 = alternatives
        .iter()
        .map(|c| c.prob)
        .filter(|p| p.is_finite() && *p > 0.0)
        .sum();
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
/// legacy terminal, HTML and Markdown renderers can display the result.
#[cfg(feature = "legacy")]
#[cfg_attr(docsrs, doc(cfg(feature = "legacy")))]
#[deprecated(
    since = "0.6.0",
    note = "the data/renderer visualizer model is deprecated; render a Report with the report module"
)]
#[allow(deprecated)]
pub fn to_token_analysis(tokens: &[LogprobToken], report: &Report) -> crate::data::TokenAnalysis {
    use crate::data::{TokenAnalysis, TokenFlag, TokenInfo};
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
        assert!(matches!(e, ParseError::NullLogprobs), "{e:?}");
        assert!(e.to_string().contains("logprobs"));
        let e = parse_logprobs(r#"{"choices":[{"message":{"content":"Hi"}}]}"#).unwrap_err();
        assert!(matches!(e, ParseError::NoLogprobs), "{e:?}");
        let e = parse_logprobs(r#"{"error":{"message":"bad key"}}"#).unwrap_err();
        assert!(matches!(e, ParseError::ApiError(ref v) if v["message"] == "bad key"));
        assert!(e.to_string().contains("bad key"));
        assert!(matches!(
            parse_logprobs("{oops"),
            Err(ParseError::InvalidJson(_))
        ));
        assert!(matches!(parse_logprobs("[]"), Err(ParseError::Empty)));
        let e = parse_logprobs(r#"[{"token":1,"logprob":-0.1}]"#).unwrap_err();
        assert!(matches!(e, ParseError::Malformed(_)), "{e:?}");
    }

    #[test]
    fn null_legacy_logprob_is_skipped_not_flagged() {
        // `echo: true` completions return null for the first token.
        let json = r#"{"choices":[{"logprobs":{"tokens":["The"," cat"," sat"],
            "token_logprobs":[null,-0.1,-0.2],"top_logprobs":[null,{" cat":-0.1},{" sat":-0.2}]}}]}"#;
        let t = parse_logprobs(json).unwrap();
        assert_eq!(t.len(), 3);
        assert!(!t[0].has_logprob());
        let r = detect(&t, 0.5);
        assert_eq!(r.text, "The cat sat");
        assert!(!r.flagged(), "{:?}", r.spans);
        let expected = (-(-0.1f64 - 0.2) / 2.0).exp();
        assert!((r.perplexity - expected).abs() < 1e-12, "{}", r.perplexity);
        // The same null in the content-array shape.
        let t = parse_logprobs(r#"[{"token":"Hi","logprob":null},{"token":"!","logprob":-0.1}]"#)
            .unwrap();
        assert!(!t[0].has_logprob());
        assert!(!detect(&t, 0.5).flagged());
    }

    #[test]
    fn report_round_trips_through_json() {
        let json = include_str!("../examples/logprobs/cuyp.json");
        let r = detect(&parse_logprobs(json).unwrap(), 0.6);
        let back: Report = serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        // serde_json's default float parser can be off by one ulp, so
        // compare the numbers approximately.
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(1.0);
        assert_eq!((&back.text, back.token_count), (&r.text, r.token_count));
        assert!(close(back.perplexity, r.perplexity) && close(back.mean_prob, r.mean_prob));
        assert_eq!(back.spans.len(), r.spans.len());
        for (a, b) in back.spans.iter().zip(&r.spans) {
            assert_eq!(
                (a.start, a.end, &a.text, &a.weakest_token),
                (b.start, b.end, &b.text, &b.weakest_token)
            );
            assert!(close(a.min_prob, b.min_prob));
            assert_eq!(a.alternatives.len(), b.alternatives.len());
        }
        let zero = detect(&[tok("a", 0.0)], 0.5);
        assert!(zero.perplexity.is_infinite());
        let back: Report = serde_json::from_str(&serde_json::to_string(&zero).unwrap()).unwrap();
        assert!(back.perplexity.is_infinite());
    }

    #[test]
    fn threshold_is_checked_or_clamped() {
        let t = vec![tok("a", 0.3)];
        assert!(try_detect(&t, f64::NAN).unwrap_err().0.is_nan());
        assert!(try_detect(&t, -0.1).is_err());
        assert!(try_detect(&t, 1.1).is_err());
        assert!(try_detect(&t, 0.5).unwrap().flagged());
        let r = detect(&t, f64::NAN);
        assert!(!r.flagged());
        assert_eq!(r.threshold, 0.0);
        assert_eq!(detect(&t, 7.0).threshold, 1.0);
        assert!(detect(&t, 7.0).flagged());
    }

    #[test]
    fn candidate_prob_is_clamped() {
        let mut w = tok("x", 0.2);
        w.top_logprobs = vec![
            Alternative {
                token: "x".into(),
                logprob: 0.5,
            },
            Alternative {
                token: "y".into(),
                logprob: f64::NAN,
            },
        ];
        let r = detect(&[w], 0.5);
        assert_eq!(r.spans[0].alternatives.len(), 1);
        assert_eq!(r.spans[0].alternatives[0].prob, 1.0);
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
        let t = vec![
            tok(" don", 0.9),
            tok("'t", 0.2),
            tok(" well", 0.9),
            tok("-", 0.9),
            tok("known", 0.9),
        ];
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
            Candidate {
                token: "x".into(),
                prob: 0.4,
            },
            Candidate {
                token: "y".into(),
                prob: 0.4,
            },
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
        #[cfg(feature = "legacy")]
        #[allow(deprecated)]
        {
            let a = to_token_analysis(&tokens, &r);
            assert!(a.validate().is_ok());
        }
    }
}
