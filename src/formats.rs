//! Read token logprobs from any provider's response, detecting its shape.
//!
//! [`parse_any`] takes the JSON body a provider returned and finds the token
//! logprobs in it, whichever of these shapes it has ([`Format`]):
//!
//! | Shape | Who returns it |
//! |---|---|
//! | [`Format::ChatCompletions`]: `choices[i].logprobs.content` | OpenAI Chat Completions, vLLM, SGLang, Fireworks, Groq, DeepSeek, OpenRouter, Together (OpenAI style), Ollama `/v1`, llama.cpp `/v1`, TGI `/v1`, Gemini's OpenAI-compatible endpoint |
//! | [`Format::Completions`]: `tokens` + `token_logprobs` + `top_logprobs` | OpenAI legacy completions, vLLM `/v1/completions`, Together, Fireworks completions |
//! | [`Format::Responses`]: `output[].content[].logprobs` | OpenAI Responses API (`include: ["message.output_text.logprobs"]`) |
//! | [`Format::Ollama`]: top-level `logprobs` list | Ollama `/api/chat` and `/api/generate` (`"logprobs": true`) |
//! | [`Format::LlamaCpp`]: `completion_probabilities` | llama.cpp server `/completion` (`n_probs`), with or without `post_sampling_probs`, and the pre-2025 `probs` form |
//! | [`Format::Gemini`]: `candidates[i].logprobsResult` | Google Gemini and Vertex AI `generateContent` (`responseLogprobs: true`) |
//! | [`Format::Tgi`]: `details.tokens` + `details.top_tokens` | Hugging Face Text Generation Inference `/generate` (`details: true`) |
//! | [`Format::TokenList`]: a bare token list or `{"content": [..]}` | the `logprobs` block on its own |
//!
//! The same functions read one chunk of a streamed response (an SSE `data:`
//! payload or an NDJSON line) and give that chunk's tokens; to follow a whole
//! stream and flag spans as they appear, use [`crate::stream::StreamDetector`].
//!
//! JSON written by Python's `json` module can hold `NaN`, `Infinity` and
//! `-Infinity` (vLLM writes `-inf` logprobs that way), which is not valid
//! JSON. They are read as `null` (an unknown logprob) and as the largest
//! finite numbers, so a `-Infinity` logprob is a probability of 0.

use serde::Deserialize;
use serde_json::Value;

use crate::detect::{is_special_token, Alternative, LogprobToken, ParseError};

/// The response shapes [`parse_any`] reads. See the [module docs](self) for
/// which providers return each one.
///
/// New shapes may be added in minor releases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Format {
    /// `choices[i].logprobs.content`: a list of `{token, logprob,
    /// top_logprobs}` objects (OpenAI Chat Completions and the many servers
    /// that copy it). Also stream chunks whose `logprobs` is a single number
    /// for the chunk's one token (Together's streaming form).
    ChatCompletions,
    /// `tokens`, `token_logprobs` and `top_logprobs` lists (OpenAI legacy
    /// completions, vLLM and Together), under `choices[i].logprobs` or on
    /// their own.
    Completions,
    /// OpenAI Responses API: `output[].content[].logprobs` on `output_text`
    /// parts, and the `response.output_text.delta` / `.done` and
    /// `response.completed` stream events.
    Responses,
    /// Ollama's native API: a top-level `logprobs` list next to `message`
    /// (`/api/chat`) or `response` (`/api/generate`).
    Ollama,
    /// llama.cpp server `/completion`: `completion_probabilities`, as
    /// `{token, logprob, top_logprobs}`, `{token, prob, top_probs}`
    /// (`post_sampling_probs`) or the older `{content, probs: [{tok_str,
    /// prob}]}`.
    LlamaCpp,
    /// Google Gemini / Vertex AI: `candidates[i].logprobsResult` with
    /// `chosenCandidates` and `topCandidates`.
    Gemini,
    /// Hugging Face Text Generation Inference: `details.tokens` and
    /// `details.top_tokens` (`/generate`), or `token` and `top_tokens`
    /// (a `/generate_stream` event).
    Tgi,
    /// A bare list of `{token, logprob, top_logprobs}` objects, or the
    /// `logprobs` object `{"content": [..]}` on its own.
    TokenList,
}

impl Format {
    /// A short name, e.g. `"chat-completions"` or `"ollama"`.
    pub fn name(self) -> &'static str {
        match self {
            Format::ChatCompletions => "chat-completions",
            Format::Completions => "completions",
            Format::Responses => "responses",
            Format::Ollama => "ollama",
            Format::LlamaCpp => "llama.cpp",
            Format::Gemini => "gemini",
            Format::Tgi => "tgi",
            Format::TokenList => "token-list",
        }
    }
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Read token logprobs from any supported response shape (see the
/// [module docs](self)), for the first choice or candidate.
///
/// Special tokens such as `<|eot_id|>` (and TGI tokens marked `special`) are
/// dropped. A `null` logprob is kept as `NaN`, which
/// [`detect`](crate::detect()) never flags.
///
/// # Errors
///
/// A [`ParseError`]: [`ParseError::NoLogprobs`] when the response has no
/// logprobs (the request did not ask for them), [`ParseError::ApiError`] for
/// a provider error body, [`ParseError::InvalidJson`],
/// [`ParseError::Empty`] or [`ParseError::Malformed`].
///
/// ```
/// use llm_token_visualizer::{detect, parse_any};
///
/// // An Ollama /api/chat response made with "logprobs": true.
/// let ollama = r#"{"model":"gemma4:e2b","message":{"role":"assistant","content":"In 1630."},
///   "done":true,"logprobs":[
///     {"token":"In","logprob":-0.02,"top_logprobs":[]},
///     {"token":" 1630","logprob":-2.2,"top_logprobs":[]},
///     {"token":".","logprob":-0.01,"top_logprobs":[]}]}"#;
/// let tokens = parse_any(ollama)?;
/// assert_eq!(detect(&tokens, 0.5).spans[0].text, " 1630");
/// # Ok::<(), llm_token_visualizer::ParseError>(())
/// ```
pub fn parse_any(json: &str) -> Result<Vec<LogprobToken>, ParseError> {
    parse_choice(json, 0)
}

/// Like [`parse_any`], for choice (or Gemini candidate, or TGI list entry)
/// number `choice`, for responses made with `n` > 1.
///
/// The choice is the entry whose `index` field is `choice`, or else entry
/// number `choice` in the list. A choice that is not there is
/// [`ParseError::NoLogprobs`].
///
/// # Errors
///
/// As [`parse_any`].
pub fn parse_choice(json: &str, choice: usize) -> Result<Vec<LogprobToken>, ParseError> {
    let value = to_value(json)?;
    parse_value(&value, choice).map(|(_, tokens)| tokens)
}

/// Read token logprobs from an already parsed response, and say which
/// [`Format`] it had.
///
/// # Errors
///
/// As [`parse_any`]; [`ParseError::NoLogprobs`] when the shape is not
/// recognised at all.
pub fn parse_value(
    value: &Value,
    choice: usize,
) -> Result<(Format, Vec<LogprobToken>), ParseError> {
    check_api_error(value)?;
    let format = detect_format(value).ok_or(ParseError::NoLogprobs)?;
    let tokens = parse_as(value, format, choice)?;
    Ok((format, tokens))
}

/// Which [`Format`] a response has, from its keys alone (the logprobs are
/// not checked). `None` when it matches none of them.
///
/// ```
/// use llm_token_visualizer::formats::{detect_format, Format};
///
/// let v = serde_json::json!({"completion_probabilities": [], "content": ""});
/// assert_eq!(detect_format(&v), Some(Format::LlamaCpp));
/// let v = serde_json::json!({"choices": [{"logprobs": {"content": []}}]});
/// assert_eq!(detect_format(&v), Some(Format::ChatCompletions));
/// ```
pub fn detect_format(value: &Value) -> Option<Format> {
    let obj = match value {
        Value::Array(a) => {
            let tgi = a.first().is_some_and(|e| {
                e.get("generated_text").is_some() || e.get("details").is_some_and(Value::is_object)
            });
            return Some(if tgi { Format::Tgi } else { Format::TokenList });
        }
        Value::Object(o) => o,
        _ => return None,
    };
    let has = |k: &str| obj.contains_key(k);
    if obj
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(|t| t.starts_with("response."))
        || obj.get("object").and_then(Value::as_str) == Some("response")
        || obj
            .get("output")
            .and_then(Value::as_array)
            .is_some_and(|items| items.iter().any(|i| i.get("type").is_some()))
    {
        return Some(Format::Responses);
    }
    if let Some(choices) = obj.get("choices") {
        let first = choices.get(0).and_then(|c| c.get("logprobs"));
        return Some(if first.is_some_and(is_completions_block) {
            Format::Completions
        } else {
            Format::ChatCompletions
        });
    }
    if has("candidates") || has("logprobsResult") || has("chosenCandidates") {
        return Some(Format::Gemini);
    }
    if has("completion_probabilities") {
        return Some(Format::LlamaCpp);
    }
    if obj.get("details").is_some_and(Value::is_object)
        || obj.get("token").is_some_and(Value::is_object)
        || has("generated_text")
    {
        return Some(Format::Tgi);
    }
    if is_completions_block(value) {
        return Some(Format::Completions);
    }
    if let Some(lp) = obj.get("logprobs") {
        let ollama = lp.is_array() || has("done") || has("created_at");
        if ollama {
            return Some(Format::Ollama);
        }
        // A single choice on its own (`resp.choices[0]`).
        return Some(if is_completions_block(lp) {
            Format::Completions
        } else {
            Format::ChatCompletions
        });
    }
    if has("done") && (has("message") || has("response")) {
        return Some(Format::Ollama);
    }
    match obj.get("content") {
        Some(Value::String(_)) => Some(Format::LlamaCpp),
        Some(_) => Some(Format::TokenList),
        None => None,
    }
}

/// Read the tokens of `value` as `format`, for choice `choice`, without
/// guessing the shape. Use it when you know where a response came from.
///
/// # Errors
///
/// As [`parse_any`]. A response of another shape usually gives
/// [`ParseError::NoLogprobs`].
pub fn parse_as(
    value: &Value,
    format: Format,
    choice: usize,
) -> Result<Vec<LogprobToken>, ParseError> {
    parse_as_mode(value, format, choice, false)
}

/// [`parse_as`] for one chunk of a stream, where a chunk can carry another
/// choice than the one followed: choices are matched by `index` only.
pub(crate) fn parse_chunk(
    value: &Value,
    format: Format,
    choice: usize,
) -> Result<Vec<LogprobToken>, ParseError> {
    parse_as_mode(value, format, choice, true)
}

fn parse_as_mode(
    value: &Value,
    format: Format,
    choice: usize,
    stream: bool,
) -> Result<Vec<LogprobToken>, ParseError> {
    check_api_error(value)?;
    let mut tokens = match format {
        Format::ChatCompletions | Format::Completions => parse_choices(value, choice, stream)?,
        Format::Responses => parse_responses(value)?,
        Format::Ollama => match value.get("logprobs") {
            None => return Err(ParseError::NoLogprobs),
            Some(Value::Null) => return Err(ParseError::NullLogprobs),
            Some(l) => token_list(l)?,
        },
        Format::LlamaCpp => parse_llamacpp(value)?,
        Format::Gemini => parse_gemini(value, choice, stream)?,
        Format::Tgi => parse_tgi(value, choice)?,
        Format::TokenList => {
            let content = if value.is_array() {
                value
            } else {
                match value.get("content") {
                    Some(c) => c,
                    None => return Err(ParseError::NoLogprobs),
                }
            };
            token_list(content)?
        }
    };
    tokens.retain(|t| !is_special_token(&t.token));
    if tokens.is_empty() {
        return Err(ParseError::Empty);
    }
    Ok(tokens)
}

/// Parse JSON, accepting the `NaN`, `Infinity` and `-Infinity` literals that
/// Python's `json` module writes.
pub(crate) fn to_value(json: &str) -> Result<Value, ParseError> {
    match serde_json::from_str(json) {
        Ok(v) => Ok(v),
        Err(e) => {
            if json.contains("Infinity") || json.contains("NaN") {
                if let Ok(v) = serde_json::from_str(&replace_non_finite(json)) {
                    return Ok(v);
                }
            }
            Err(ParseError::InvalidJson(e))
        }
    }
}

/// Replace bare `NaN`, `Infinity` and `-Infinity` outside strings with
/// `null` and the largest finite numbers.
fn replace_non_finite(json: &str) -> String {
    const LITERALS: [(&str, &str); 3] = [
        ("-Infinity", "-1.7976931348623157e308"),
        ("Infinity", "1.7976931348623157e308"),
        ("NaN", "null"),
    ];
    let mut out = String::with_capacity(json.len() + 32);
    let mut in_string = false;
    let mut escaped = false;
    let mut i = 0;
    while i < json.len() {
        let rest = &json[i..];
        let c = rest.chars().next().unwrap_or('\0');
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
        } else if c == '"' {
            in_string = true;
        } else if let Some((lit, rep)) = LITERALS.iter().find(|(lit, _)| rest.starts_with(lit)) {
            out.push_str(rep);
            i += lit.len();
            continue;
        }
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// A non-null `error` field is a provider error body.
pub(crate) fn check_api_error(value: &Value) -> Result<(), ParseError> {
    match value.get("error") {
        None | Some(Value::Null) => Ok(()),
        Some(err) => Err(ParseError::ApiError(err.clone())),
    }
}

/// `{tokens: [..], token_logprobs: [..]}`.
fn is_completions_block(v: &Value) -> bool {
    v.get("tokens").is_some() && v.get("token_logprobs").is_some()
}

/// Pick entry `n` of a choices/candidates list: by its `index` field when
/// the entries have one (`default_index` stands in for a missing one, as
/// proto3 JSON leaves out an index of 0), else by position.
pub(crate) fn select(list: &[Value], n: usize, default_index: Option<u64>) -> Option<&Value> {
    let index_of = |c: &Value| c.get("index").and_then(Value::as_u64);
    let any_index = list.iter().any(|c| index_of(c).is_some());
    if any_index {
        return list
            .iter()
            .find(|c| index_of(c).or(default_index) == Some(n as u64));
    }
    list.get(n)
}

/// [`select`] for a whole response: when no entry has index `n`, entry `n`
/// by position (a list already cut down to one choice keeps that choice's
/// `index`). A stream chunk matches by `index` only.
fn pick(list: &[Value], n: usize, default_index: Option<u64>, stream: bool) -> Option<&Value> {
    select(list, n, default_index).or_else(|| if stream { None } else { list.get(n) })
}

/// A list of `{token, logprob, top_logprobs}` objects (OpenAI, Ollama,
/// llama.cpp's OpenAI-compatible endpoints). Extra fields such as `bytes`
/// and `id` are ignored.
pub(crate) fn token_list(content: &Value) -> Result<Vec<LogprobToken>, ParseError> {
    if content.is_null() {
        return Err(ParseError::NullLogprobs);
    }
    // Deserialize from the borrowed value: cloning a large `content` array
    // first cost more than the detection itself.
    Vec::<LogprobToken>::deserialize(content)
        .map_err(|e| ParseError::Malformed(format!("could not read logprobs tokens: {e}")))
}

/// `choices[n].logprobs` in any of its forms, or a single choice object.
fn parse_choices(value: &Value, n: usize, stream: bool) -> Result<Vec<LogprobToken>, ParseError> {
    let choice = match value.get("choices") {
        Some(Value::Array(list)) => pick(list, n, None, stream).ok_or(ParseError::NoLogprobs)?,
        Some(_) => return Err(ParseError::Malformed("choices is not a list".to_string())),
        // A bare completions block, or a single choice.
        None if is_completions_block(value) => return parse_legacy(value),
        None => value,
    };
    let lp = match choice.get("logprobs") {
        None => return Err(ParseError::NoLogprobs),
        Some(Value::Null) => return Err(ParseError::NullLogprobs),
        Some(lp) => lp,
    };
    match lp {
        // Together's streaming chunks: one token per chunk, its logprob as a
        // bare number, the text in `delta.content` (or `text`).
        Value::Number(n) => {
            let token = choice
                .pointer("/delta/content")
                .and_then(Value::as_str)
                .or_else(|| choice.get("text").and_then(Value::as_str))
                .ok_or_else(|| {
                    ParseError::Malformed(
                        "a numeric logprobs value needs the token in delta.content or text"
                            .to_string(),
                    )
                })?;
            Ok(vec![LogprobToken {
                token: token.to_string(),
                logprob: n.as_f64().unwrap_or(f64::NAN),
                top_logprobs: Vec::new(),
            }])
        }
        Value::Array(_) => token_list(lp),
        _ if is_completions_block(lp) => parse_legacy(lp),
        _ => match lp.get("content") {
            Some(c) => token_list(c),
            None => Err(ParseError::NoLogprobs),
        },
    }
}

/// Read `{"tokens": [..], "token_logprobs": [..], "top_logprobs": [..]}`,
/// where each `top_logprobs` entry is either a `{token: logprob}` map or a
/// list of `{"token", "logprob"}` objects.
///
/// A `null` entry in `token_logprobs` (the first token of an `echo: true`
/// completion) becomes `NaN`, so [`detect`](crate::detect()) skips it.
/// Before 0.6.0 it became `-inf`: probability 0, always flagged, and an
/// infinite perplexity.
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
            Some(v @ Value::Array(_)) => Vec::<Alternative>::deserialize(v).unwrap_or_default(),
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

/// OpenAI Responses API: a response object, a `response.completed` (or
/// similar) event holding one, or a `response.output_text.delta` / `.done`
/// event.
fn parse_responses(value: &Value) -> Result<Vec<LogprobToken>, ParseError> {
    if let Some(ty) = value.get("type").and_then(Value::as_str) {
        if ty.starts_with("response.output_text.") {
            return match value.get("logprobs") {
                None => Err(ParseError::NoLogprobs),
                Some(l) => token_list(l),
            };
        }
        if ty == "response.failed" {
            let err = value
                .pointer("/response/error")
                .filter(|e| !e.is_null())
                .unwrap_or(value);
            return Err(ParseError::ApiError(err.clone()));
        }
        if let Some(r) = value.get("response") {
            check_api_error(r)?;
            return parse_responses(r);
        }
        return Err(ParseError::NoLogprobs);
    }
    let output = value
        .get("output")
        .and_then(Value::as_array)
        .ok_or(ParseError::NoLogprobs)?;
    let mut out = Vec::new();
    let mut saw_text = false;
    let parts = output
        .iter()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("message"))
        .filter_map(|item| item.get("content").and_then(Value::as_array))
        .flatten()
        .filter(|p| p.get("type").and_then(Value::as_str) == Some("output_text"));
    for part in parts {
        saw_text = true;
        match part.get("logprobs") {
            None | Some(Value::Null) => {}
            Some(l) => out.extend(token_list(l)?),
        }
    }
    if out.is_empty() {
        // `logprobs: []` on every part: the request did not include
        // "message.output_text.logprobs".
        return Err(if saw_text {
            ParseError::NoLogprobs
        } else {
            ParseError::Empty
        });
    }
    Ok(out)
}

/// llama.cpp server `completion_probabilities`, in its three forms. Entries
/// with empty token text (special tokens such as end-of-turn) are dropped.
fn parse_llamacpp(value: &Value) -> Result<Vec<LogprobToken>, ParseError> {
    let probs = match value.get("completion_probabilities") {
        None => return Err(ParseError::NoLogprobs),
        Some(Value::Null) => return Err(ParseError::NullLogprobs),
        Some(Value::Array(a)) => a,
        Some(_) => {
            return Err(ParseError::Malformed(
                "completion_probabilities is not a list".to_string(),
            ))
        }
    };
    let alt = |a: &Value| -> Option<Alternative> {
        let token = a.get("token").or_else(|| a.get("tok_str"))?.as_str()?;
        let logprob = match a.get("logprob") {
            Some(lp) => lp.as_f64()?,
            None => a.get("prob")?.as_f64()?.ln(),
        };
        Some(Alternative {
            token: token.to_string(),
            logprob,
        })
    };
    let mut out = Vec::with_capacity(probs.len());
    for (i, p) in probs.iter().enumerate() {
        let mut top: Vec<Alternative> = ["top_logprobs", "top_probs", "probs"]
            .iter()
            .find_map(|k| p.get(*k).and_then(Value::as_array))
            .map(|v| v.iter().filter_map(alt).collect())
            .unwrap_or_default();
        top.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
        let token = if let Some(t) = p.get("token").and_then(Value::as_str) {
            let logprob = match (p.get("logprob"), p.get("prob")) {
                (Some(lp), _) => lp.as_f64().unwrap_or(f64::NAN),
                (None, Some(pr)) => pr.as_f64().map_or(f64::NAN, f64::ln),
                (None, None) => f64::NAN,
            };
            LogprobToken {
                token: t.to_string(),
                logprob,
                top_logprobs: top,
            }
        } else if let Some(t) = p.get("content").and_then(Value::as_str) {
            // Before 2025: {content, probs: [{tok_str, prob}]}; the chosen
            // token's probability is its entry in `probs`, if listed.
            let logprob = top
                .iter()
                .find(|a| a.token == t)
                .map_or(f64::NAN, |a| a.logprob);
            LogprobToken {
                token: t.to_string(),
                logprob,
                top_logprobs: top,
            }
        } else {
            return Err(ParseError::Malformed(format!(
                "completion_probabilities[{i}] has no token"
            )));
        };
        if !token.token.is_empty() {
            out.push(token);
        }
    }
    Ok(out)
}

/// Google Gemini's native `generateContent` response with
/// `responseLogprobs: true`: `candidates[n].logprobsResult` holds
/// `chosenCandidates` (`{token, logProbability}`) and, per position,
/// `topCandidates[i].candidates`. Also accepts a bare `logprobsResult`.
///
/// Gemini's JSON leaves out fields with default values, so a missing
/// `logProbability` is 0 (probability 1) and a missing `token` is empty.
fn parse_gemini(value: &Value, n: usize, stream: bool) -> Result<Vec<LogprobToken>, ParseError> {
    let lr = if let Some(c) = value.get("candidates") {
        let list = c
            .as_array()
            .ok_or_else(|| ParseError::Malformed("candidates is not a list".to_string()))?;
        let cand = pick(list, n, Some(0), stream).ok_or(ParseError::NoLogprobs)?;
        cand.get("logprobsResult").ok_or(ParseError::NoLogprobs)?
    } else if let Some(lr) = value.get("logprobsResult") {
        lr
    } else {
        value
    };
    if lr.is_null() {
        return Err(ParseError::NullLogprobs);
    }
    let chosen = lr
        .get("chosenCandidates")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ParseError::Malformed("logprobsResult has no chosenCandidates list".to_string())
        })?;
    let tops = lr.get("topCandidates").and_then(Value::as_array);
    let cand = |c: &Value| -> Option<Alternative> {
        let c = c.as_object()?;
        let token = match c.get("token") {
            Some(t) => t.as_str()?.to_string(),
            None => String::new(),
        };
        let logprob = match c.get("logProbability") {
            Some(lp) => lp.as_f64()?,
            None => 0.0,
        };
        Some(Alternative { token, logprob })
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
    Ok(out)
}

/// One TGI token object, `{id, text, logprob, special}`. `None` for a
/// special token.
fn tgi_token(t: &Value) -> Result<Option<Alternative>, ParseError> {
    if t.get("special").and_then(Value::as_bool) == Some(true) {
        return Ok(None);
    }
    let token = t
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| ParseError::Malformed("a TGI token has no text".to_string()))?;
    let logprob = t.get("logprob").and_then(Value::as_f64).unwrap_or(f64::NAN);
    Ok(Some(Alternative {
        token: token.to_string(),
        logprob,
    }))
}

fn tgi_alternatives(top: Option<&Value>) -> Vec<Alternative> {
    let mut alts: Vec<Alternative> = top
        .and_then(Value::as_array)
        .map(|v| {
            v.iter()
                .filter_map(|t| tgi_token(t).ok().flatten())
                .filter(|a| !a.logprob.is_nan())
                .collect()
        })
        .unwrap_or_default();
    alts.sort_by(|a, b| b.logprob.total_cmp(&a.logprob));
    alts
}

/// Hugging Face TGI: a `/generate` response (or the list the Inference API
/// wraps it in) with `details.tokens`, or one `/generate_stream` event.
fn parse_tgi(value: &Value, n: usize) -> Result<Vec<LogprobToken>, ParseError> {
    let resp = match value {
        Value::Array(list) => list.get(n).ok_or(ParseError::NoLogprobs)?,
        v => v,
    };
    // A stream event: one token.
    if let Some(t) = resp.get("token").filter(|t| t.is_object()) {
        return Ok(tgi_token(t)?
            .map(|a| LogprobToken {
                token: a.token,
                logprob: a.logprob,
                top_logprobs: tgi_alternatives(resp.get("top_tokens")),
            })
            .into_iter()
            .collect());
    }
    let details = match resp.get("details") {
        None => return Err(ParseError::NoLogprobs),
        Some(Value::Null) => return Err(ParseError::NullLogprobs),
        Some(d) => d,
    };
    let tokens = match details.get("tokens") {
        None => return Err(ParseError::NoLogprobs),
        Some(Value::Null) => return Err(ParseError::NullLogprobs),
        Some(t) => t
            .as_array()
            .ok_or_else(|| ParseError::Malformed("details.tokens is not a list".to_string()))?,
    };
    let tops = details.get("top_tokens").and_then(Value::as_array);
    let mut out = Vec::with_capacity(tokens.len());
    for (i, t) in tokens.iter().enumerate() {
        if let Some(a) = tgi_token(t)? {
            out.push(LogprobToken {
                token: a.token,
                logprob: a.logprob,
                top_logprobs: tgi_alternatives(tops.and_then(|t| t.get(i))),
            });
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_literals_outside_strings_are_replaced() {
        let s = r#"{"a":-Infinity,"b":"NaN -Infinity","c":[NaN,Infinity],"d":"q\"NaN"}"#;
        let v = to_value(s).unwrap();
        assert!(v["a"].as_f64().unwrap() < -1e308);
        assert_eq!(v["b"], "NaN -Infinity");
        assert!(v["c"][0].is_null());
        assert!(v["c"][1].as_f64().unwrap() > 1e308);
        assert_eq!(v["d"], "q\"NaN");
        assert!(matches!(to_value("{NaN"), Err(ParseError::InvalidJson(_))));
    }

    #[test]
    fn select_by_index_field_or_position() {
        let l: Vec<Value> =
            serde_json::from_str(r#"[{"index":1,"x":"b"},{"index":0,"x":"a"}]"#).unwrap();
        assert_eq!(select(&l, 0, None).unwrap()["x"], "a");
        assert!(select(&l, 2, None).is_none());
        let l: Vec<Value> = serde_json::from_str(r#"[{"x":"a"},{"x":"b"}]"#).unwrap();
        assert_eq!(select(&l, 1, None).unwrap()["x"], "b");
        // Gemini leaves out index 0.
        let l: Vec<Value> = serde_json::from_str(r#"[{"x":"a"},{"index":1,"x":"b"}]"#).unwrap();
        assert_eq!(select(&l, 0, Some(0)).unwrap()["x"], "a");
        assert_eq!(select(&l, 1, Some(0)).unwrap()["x"], "b");
    }

    #[test]
    fn error_null_is_not_an_api_error() {
        assert!(check_api_error(&serde_json::json!({"error": null})).is_ok());
        assert!(check_api_error(&serde_json::json!({"error": "boom"})).is_err());
    }

    #[test]
    fn llamacpp_legacy_probs_form() {
        let v = serde_json::json!({"content": " Delft", "completion_probabilities": [
            {"content": " Delft", "probs": [{"tok_str": " The", "prob": 0.6}, {"tok_str": " Delft", "prob": 0.3}]},
            {"content": ".", "probs": [{"tok_str": "!", "prob": 0.9}]}
        ]});
        let t = parse_as(&v, Format::LlamaCpp, 0).unwrap();
        assert_eq!(t.len(), 2);
        assert!((t[0].prob() - 0.3).abs() < 1e-12);
        assert_eq!(t[0].top_logprobs[0].token, " The");
        // The chosen token is not among the listed probabilities: unknown.
        assert!(!t[1].has_logprob());
    }

    #[test]
    fn format_names() {
        assert_eq!(Format::LlamaCpp.to_string(), "llama.cpp");
        assert_eq!(Format::Responses.name(), "responses");
    }
}
