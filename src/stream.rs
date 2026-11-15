//! Flag low-confidence spans while a response is still streaming in.
//!
//! [`StreamDetector`] takes a streamed response in whatever pieces it
//! arrives: raw bytes of a server-sent events (SSE) or NDJSON stream
//! ([`StreamDetector::push_sse`]), one chunk's JSON at a time
//! ([`StreamDetector::push_event`]), or tokens you already have
//! ([`StreamDetector::push_tokens`]). Each call returns the spans that became
//! final with that input, so a UI can mark an unsure claim a few tokens
//! after the model wrote it instead of after the whole answer.
//!
//! The chunks can be any shape [`crate::formats`] reads: OpenAI Chat
//! Completions and legacy completions chunks (and the servers that copy them:
//! vLLM, Together, Fireworks, OpenRouter, Ollama and llama.cpp `/v1`), OpenAI
//! Responses API events (`response.output_text.delta`), Ollama's NDJSON,
//! llama.cpp `/completion` with `n_probs`, Gemini `streamGenerateContent`
//! (`alt=sse`) and Hugging Face TGI `/generate_stream`.
//!
//! # What "final" means
//!
//! A span is final when no later token can change it: there is a word after
//! it, with a token that is not whitespace, whose boundaries are settled. The
//! spans returned over the whole stream, with those [`StreamDetector::finish`]
//! returns for the end of the answer, are exactly the spans
//! [`detect`](crate::detect()) gives for the complete token list, in order,
//! and [`StreamDetector::report`] after `finish` equals `detect`'s report.
//!
//! The detector only re-reads the tokens after the last settled word
//! boundary (whitespace, a line break or a CJK ideograph), so the work grows
//! with the length of the answer, not its square: following a 4,000-token
//! answer one token at a time takes about 3 ms (`benches/vs_0_4.rs`).
//!
//! ```
//! use llm_token_visualizer::StreamDetector;
//!
//! // An OpenAI-style stream, cut into network chunks at arbitrary points.
//! let sse = concat!(
//!     "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"In\"},",
//!     "\"logprobs\":{\"content\":[{\"token\":\"In\",\"logprob\":-0.01,\"top_logprobs\":[]}]}}]}\n\n",
//!     "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" 1630\"},",
//!     "\"logprobs\":{\"content\":[{\"token\":\" 1630\",\"logprob\":-2.2,\"top_logprobs\":[]}]}}]}\n\n",
//!     "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" in\"},",
//!     "\"logprobs\":{\"content\":[{\"token\":\" in\",\"logprob\":-0.01,\"top_logprobs\":[]}]}}]}\n\n",
//!     "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" Delft\"},",
//!     "\"logprobs\":{\"content\":[{\"token\":\" Delft\",\"logprob\":-0.02,\"top_logprobs\":[]}]}}]}\n\n",
//!     "data: [DONE]\n\n",
//! );
//! let mut det = StreamDetector::new(0.5);
//! let mut flagged = Vec::new();
//! for piece in sse.as_bytes().chunks(37) {
//!     flagged.extend(det.push_sse(piece)?);
//! }
//! flagged.extend(det.finish()?);
//! assert_eq!(flagged.len(), 1);
//! assert_eq!(flagged[0].text, " 1630");
//! assert!(det.is_done());
//! # Ok::<(), llm_token_visualizer::ParseError>(())
//! ```

use serde_json::Value;

use crate::detect::{
    effective_threshold, find_spans, is_special_token, InvalidThreshold, LogprobToken, ParseError,
    Report, Span,
};
use crate::formats::{self, Format};

/// Detects low-confidence spans incrementally over a streamed response. See
/// the [module docs](self).
#[derive(Debug, Clone)]
pub struct StreamDetector {
    threshold: f64,
    choice: usize,
    tokens: Vec<LogprobToken>,
    text: String,
    /// Tokens before this index are settled and never re-read.
    base: usize,
    /// Spans returned so far.
    spans: Vec<Span>,
    /// End of the last returned span.
    emitted_end: usize,
    /// The non-whitespace token that keeps the last returned span from
    /// merging with later ones.
    blocker: Option<usize>,
    known: usize,
    sum_prob: f64,
    sum_logprob: f64,
    /// Bytes of an incomplete line.
    line: Vec<u8>,
    /// `data:` lines of the current SSE event.
    data: String,
    has_data: bool,
    done: bool,
    missing: bool,
    format: Option<Format>,
}

impl Default for StreamDetector {
    fn default() -> Self {
        StreamDetector::new(crate::DEFAULT_THRESHOLD)
    }
}

impl StreamDetector {
    /// A detector flagging words with a token below `threshold`, treated as
    /// [`detect`](crate::detect()) treats it: clamped to `0..=1`, and a `NaN`
    /// threshold flags nothing.
    pub fn new(threshold: f64) -> Self {
        StreamDetector {
            threshold: effective_threshold(threshold),
            choice: 0,
            tokens: Vec::new(),
            text: String::new(),
            base: 0,
            spans: Vec::new(),
            emitted_end: 0,
            blocker: None,
            known: 0,
            sum_prob: 0.0,
            sum_logprob: 0.0,
            line: Vec::new(),
            data: String::new(),
            has_data: false,
            done: false,
            missing: false,
            format: None,
        }
    }

    /// Like [`StreamDetector::new`], but a threshold that is `NaN` or outside
    /// `0..=1` is an error.
    ///
    /// # Errors
    ///
    /// [`InvalidThreshold`] for such a threshold.
    pub fn try_new(threshold: f64) -> Result<Self, InvalidThreshold> {
        if (0.0..=1.0).contains(&threshold) {
            Ok(StreamDetector::new(threshold))
        } else {
            Err(InvalidThreshold(threshold))
        }
    }

    /// Follow choice (or Gemini candidate) `choice` instead of the first, for
    /// streams made with `n` > 1. Chunks for other choices are skipped.
    #[must_use]
    pub fn with_choice(mut self, choice: usize) -> Self {
        self.choice = choice;
        self
    }

    /// The threshold in use.
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Every token received so far (special tokens dropped).
    pub fn tokens(&self) -> &[LogprobToken] {
        &self.tokens
    }

    /// The answer so far, tokens joined.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The spans returned so far, in order.
    pub fn spans(&self) -> &[Span] {
        &self.spans
    }

    /// The shape of the last chunk read through [`StreamDetector::push_event`]
    /// or [`StreamDetector::push_sse`].
    pub fn format(&self) -> Option<Format> {
        self.format
    }

    /// Whether the stream said it was over: `data: [DONE]`, a
    /// `response.completed` event, Ollama's `"done": true` or llama.cpp's
    /// `"stop": true`.
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Whether a chunk carried answer text but no logprobs, which means the
    /// request did not ask for them (or the server ignores the parameter).
    /// Such chunks are skipped, so check this when a stream gave no tokens.
    pub fn missing_logprobs(&self) -> bool {
        self.missing
    }

    /// Add one token. Returns the spans that became final.
    pub fn push_token(&mut self, token: LogprobToken) -> Vec<Span> {
        self.push_tokens(std::iter::once(token))
    }

    /// Add tokens. Special tokens such as `<|eot_id|>` are dropped, as
    /// [`parse_any`](crate::parse_any) does. Returns the spans that became
    /// final.
    pub fn push_tokens<I: IntoIterator<Item = LogprobToken>>(&mut self, tokens: I) -> Vec<Span> {
        let before = self.tokens.len();
        for t in tokens {
            if is_special_token(&t.token) {
                continue;
            }
            if t.has_logprob() {
                self.known += 1;
                self.sum_prob += t.logprob.exp().clamp(0.0, 1.0);
                self.sum_logprob += t.logprob;
            }
            self.text.push_str(&t.token);
            self.tokens.push(t);
        }
        if self.tokens.len() == before {
            return Vec::new();
        }
        self.advance()
    }

    /// Add one streamed chunk: the JSON of an SSE `data:` payload or an
    /// NDJSON line. Chunks without logprobs (role and usage chunks, other
    /// Responses events, a final `done` line) add nothing. `[DONE]` marks the
    /// stream as done.
    ///
    /// # Errors
    ///
    /// [`ParseError::InvalidJson`] for text that is not JSON,
    /// [`ParseError::ApiError`] for an error event, and
    /// [`ParseError::Malformed`] for logprobs of the wrong shape.
    pub fn push_event(&mut self, json: &str) -> Result<Vec<Span>, ParseError> {
        if json.trim() == "[DONE]" {
            self.done = true;
            return Ok(Vec::new());
        }
        let value = formats::to_value(json)?;
        self.push_event_value(&value)
    }

    /// [`StreamDetector::push_event`] for an already parsed chunk.
    ///
    /// # Errors
    ///
    /// As [`StreamDetector::push_event`].
    pub fn push_event_value(&mut self, value: &Value) -> Result<Vec<Span>, ParseError> {
        let tokens = self.chunk_tokens(value)?;
        Ok(self.push_tokens(tokens))
    }

    /// Add raw bytes of an SSE (`data: ...` lines) or NDJSON (one JSON object
    /// per line, as Ollama streams) response, cut anywhere, even inside a
    /// line or a UTF-8 character. Comments, `event:`, `id:` and `retry:`
    /// lines are skipped. Returns the spans that became final.
    ///
    /// # Errors
    ///
    /// As [`StreamDetector::push_event`], for the first bad event. Spans that
    /// became final earlier in the same call are still in
    /// [`StreamDetector::spans`], and the rest of the input stays buffered
    /// for the next call.
    pub fn push_sse(&mut self, bytes: impl AsRef<[u8]>) -> Result<Vec<Span>, ParseError> {
        self.line.extend_from_slice(bytes.as_ref());
        let mut out = Vec::new();
        while let Some(pos) = self.line.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.line.drain(..=pos).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8_lossy(&line).into_owned();
            out.extend(self.handle_line(&line)?);
        }
        Ok(out)
    }

    /// End of stream: read any input still buffered by
    /// [`StreamDetector::push_sse`] and return the spans that were waiting
    /// for more tokens (a span at the very end of the answer).
    ///
    /// # Errors
    ///
    /// As [`StreamDetector::push_event`], for a bad last event. The pending
    /// spans are still added to [`StreamDetector::spans`].
    pub fn finish(&mut self) -> Result<Vec<Span>, ParseError> {
        let mut read = Ok(Vec::new());
        if !self.line.is_empty() {
            let line = String::from_utf8_lossy(&std::mem::take(&mut self.line)).into_owned();
            let line = line.strip_suffix('\r').unwrap_or(&line).to_string();
            read = self.handle_line(&line);
        }
        let read = read.and_then(|mut out| {
            out.extend(self.dispatch()?);
            Ok(out)
        });
        let tail = self.flush();
        let mut out = read?;
        out.extend(tail);
        Ok(out)
    }

    /// The report over the tokens so far, with every span (returned or
    /// still pending). After [`StreamDetector::finish`] it equals
    /// [`detect`](crate::detect()) on [`StreamDetector::tokens`].
    pub fn report(&self) -> Report {
        let mut spans = self.spans.clone();
        spans.extend(self.window_spans().filter(|s| s.start >= self.emitted_end));
        let (mean_prob, perplexity) = if self.known == 0 {
            (0.0, 1.0)
        } else {
            let k = self.known as f64;
            (self.sum_prob / k, (-(self.sum_logprob / k)).exp())
        };
        Report {
            threshold: self.threshold,
            text: self.text.clone(),
            token_count: self.tokens.len(),
            mean_prob,
            perplexity,
            spans,
        }
    }

    fn handle_line(&mut self, line: &str) -> Result<Vec<Span>, ParseError> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return Ok(Vec::new());
        }
        if let Some(rest) = line.strip_prefix("data:") {
            let rest = rest.strip_prefix(' ').unwrap_or(rest);
            if self.has_data {
                self.data.push('\n');
            }
            self.data.push_str(rest);
            self.has_data = true;
            // Providers send one-line events; read each as soon as it is
            // complete instead of waiting for the blank line.
            if self.data.trim() == "[DONE]" {
                return self.dispatch();
            }
            if let Ok(v) = formats::to_value(&self.data) {
                self.data.clear();
                self.has_data = false;
                return self.push_event_value(&v);
            }
            return Ok(Vec::new());
        }
        let t = line.trim_start();
        if t.starts_with('{') || t.starts_with('[') {
            // NDJSON.
            let mut out = self.dispatch()?;
            out.extend(self.push_event(t)?);
            return Ok(out);
        }
        // `event:`, `id:`, `retry:` and anything else.
        Ok(Vec::new())
    }

    /// Read the buffered `data:` lines as one event.
    fn dispatch(&mut self) -> Result<Vec<Span>, ParseError> {
        if !self.has_data {
            return Ok(Vec::new());
        }
        let data = std::mem::take(&mut self.data);
        self.has_data = false;
        self.push_event(&data)
    }

    fn chunk_tokens(&mut self, v: &Value) -> Result<Vec<LogprobToken>, ParseError> {
        formats::check_api_error(v)?;
        if let Some(ty) = v.get("type").and_then(Value::as_str) {
            match ty {
                "error" => return Err(ParseError::ApiError(v.clone())),
                "response.failed" => {
                    let err = v
                        .pointer("/response/error")
                        .filter(|e| !e.is_null())
                        .unwrap_or(v);
                    return Err(ParseError::ApiError(err.clone()));
                }
                "response.output_text.delta" => {
                    self.format = Some(Format::Responses);
                    let tokens = match v.get("logprobs") {
                        None | Some(Value::Null) => Vec::new(),
                        Some(l) => formats::token_list(l)?,
                    };
                    if tokens.is_empty() && nonempty(v.get("delta")) {
                        self.missing = true;
                    }
                    return Ok(tokens);
                }
                "response.completed" | "response.incomplete" => {
                    self.format = Some(Format::Responses);
                    self.done = true;
                    return Ok(Vec::new());
                }
                // `.done` events and `response.completed` repeat the deltas.
                t if t.starts_with("response.") => {
                    self.format = Some(Format::Responses);
                    return Ok(Vec::new());
                }
                _ => {}
            }
        }
        if v.get("done").and_then(Value::as_bool) == Some(true)
            || v.get("stop").and_then(Value::as_bool) == Some(true)
        {
            self.done = true;
        }
        let Some(format) = formats::detect_format(v) else {
            return Ok(Vec::new());
        };
        self.format = Some(format);
        match formats::parse_chunk(v, format, self.choice) {
            Ok(tokens) => Ok(tokens),
            Err(ParseError::NoLogprobs | ParseError::NullLogprobs | ParseError::Empty) => {
                if chunk_has_text(v, self.choice) {
                    self.missing = true;
                }
                Ok(Vec::new())
            }
            Err(e) => Err(e),
        }
    }

    /// The spans `detect` finds in the unsettled tokens, in stream indices.
    fn window_spans(&self) -> impl Iterator<Item = Span> + '_ {
        let base = self.base;
        find_spans(&self.tokens[base..], self.threshold)
            .into_iter()
            .map(move |mut s| {
                s.start += base;
                s.end += base;
                s
            })
    }

    /// Return the spans that became final and move the window start past
    /// what is settled.
    fn advance(&mut self) -> Vec<Span> {
        let n = self.tokens.len();
        let spans: Vec<Span> = self.window_spans().collect();
        // Word boundaries before the last safe cut cannot change.
        let Some(cut) = (self.base + 1..n)
            .rev()
            .find(|&k| safe_cut(&self.tokens[k - 1].token, &self.tokens[k].token))
        else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let mut pending = None;
        for (i, s) in spans.iter().enumerate() {
            if s.start < self.emitted_end {
                continue;
            }
            let next = spans.get(i + 1).map_or(n, |x| x.start).min(cut);
            // A settled token that is not whitespace keeps later flagged
            // words from merging into this span.
            let blocker = (s.end..next).find(|&j| !self.tokens[j].token.trim().is_empty());
            match blocker {
                Some(j) if s.end < cut => {
                    self.emitted_end = s.end;
                    self.blocker = Some(j);
                    self.spans.push(s.clone());
                    out.push(s.clone());
                }
                _ => {
                    pending = Some(s.start);
                    break;
                }
            }
        }
        let hi = pending.map_or(cut, |p| p.min(cut));
        let lo = self.blocker.map_or(self.base, |j| j.max(self.base));
        if let Some(c) = (lo + 1..=hi)
            .rev()
            .find(|&k| k < n && safe_cut(&self.tokens[k - 1].token, &self.tokens[k].token))
        {
            self.base = c;
        }
        out
    }

    /// Return every span not returned yet.
    fn flush(&mut self) -> Vec<Span> {
        let rest: Vec<Span> = self
            .window_spans()
            .filter(|s| s.start >= self.emitted_end)
            .collect();
        if let Some(last) = rest.last() {
            self.emitted_end = last.end;
        }
        self.spans.extend(rest.iter().cloned());
        rest
    }
}

/// Read a whole recorded stream (SSE or NDJSON, as saved from `curl -N`)
/// into its tokens.
///
/// # Errors
///
/// As [`StreamDetector::push_event`], plus [`ParseError::NoLogprobs`] when
/// the stream has answer text but no logprobs, and [`ParseError::Empty`]
/// when it has no tokens at all.
///
/// ```
/// use llm_token_visualizer::stream::parse_stream;
///
/// let ndjson = concat!(
///     r#"{"message":{"content":"Hi"},"done":false,"logprobs":[{"token":"Hi","logprob":-0.1}]}"#, "\n",
///     r#"{"message":{"content":""},"done":true}"#, "\n",
/// );
/// assert_eq!(parse_stream(ndjson)?[0].token, "Hi");
/// # Ok::<(), llm_token_visualizer::ParseError>(())
/// ```
pub fn parse_stream(text: &str) -> Result<Vec<LogprobToken>, ParseError> {
    parse_stream_choice(text, 0)
}

/// [`parse_stream`] for choice `choice` of a stream made with `n` > 1.
///
/// # Errors
///
/// As [`parse_stream`].
pub fn parse_stream_choice(text: &str, choice: usize) -> Result<Vec<LogprobToken>, ParseError> {
    // Threshold 0 flags nothing: only the tokens are wanted.
    let mut det = StreamDetector::new(0.0).with_choice(choice);
    det.push_sse(text)?;
    det.finish()?;
    if det.tokens.is_empty() {
        return Err(if det.missing {
            ParseError::NoLogprobs
        } else {
            ParseError::Empty
        });
    }
    Ok(det.tokens)
}

/// Whether the word boundary before `cur` is settled whatever comes later,
/// and no word-break rule looks back across it: `cur` starts with whitespace
/// after a token that does not end with whitespace, or follows a line
/// break, or starts with a CJK ideograph.
fn safe_cut(prev: &str, cur: &str) -> bool {
    let (Some(last), Some(first)) = (prev.chars().last(), cur.chars().next()) else {
        return false;
    };
    if last == '\n' || (last == '\r' && first != '\n') {
        return true;
    }
    if first.is_whitespace() {
        return !last.is_whitespace();
    }
    is_ideograph(first) && last != '\u{200d}'
}

fn is_ideograph(c: char) -> bool {
    matches!(c,
        '\u{3400}'..='\u{4dbf}'
        | '\u{4e00}'..='\u{9fff}'
        | '\u{f900}'..='\u{faff}'
        | '\u{20000}'..='\u{2fa1f}')
}

fn nonempty(v: Option<&Value>) -> bool {
    v.and_then(Value::as_str).is_some_and(|s| !s.is_empty())
}

/// Whether a chunk carries answer text (to tell "no logprobs requested"
/// from a role or usage chunk).
fn chunk_has_text(v: &Value, choice: usize) -> bool {
    if let Some(list) = v.get("choices").and_then(Value::as_array) {
        return formats::select(list, choice, None).is_some_and(|c| {
            nonempty(c.pointer("/delta/content"))
                || nonempty(c.get("text"))
                || nonempty(c.pointer("/message/content"))
        });
    }
    if let Some(list) = v.get("candidates").and_then(Value::as_array) {
        return formats::select(list, choice, Some(0))
            .and_then(|c| c.pointer("/content/parts"))
            .and_then(Value::as_array)
            .is_some_and(|parts| parts.iter().any(|p| nonempty(p.get("text"))));
    }
    nonempty(v.pointer("/message/content"))
        || nonempty(v.get("response"))
        || nonempty(v.get("content"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detect;

    fn tok(t: &str, p: f64) -> LogprobToken {
        LogprobToken {
            token: t.to_string(),
            logprob: p.ln(),
            top_logprobs: vec![],
        }
    }

    #[test]
    fn safe_cuts() {
        assert!(safe_cut("cat", " sat"));
        assert!(!safe_cut(" ", " sat"));
        assert!(safe_cut("x\n", "y"));
        assert!(!safe_cut("\r", "\n"));
        assert!(safe_cut("在", "巴"));
        assert!(!safe_cut("don", "'t"));
        assert!(!safe_cut("", " x"));
    }

    #[test]
    fn spans_come_out_once_settled() {
        let mut d = StreamDetector::new(0.5);
        assert!(d.push_token(tok("It", 0.99)).is_empty());
        assert!(d.push_token(tok(" was", 0.2)).is_empty());
        // " Pete" could still merge with " was".
        assert!(d.push_token(tok(" Pete", 0.3)).is_empty());
        assert!(d.push_token(tok(".", 0.99)).is_empty());
        // "." is not settled until a later word starts.
        let out = d.push_token(tok(" Then", 0.1));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, " was Pete");
        let rest = d.finish().unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].text, " Then");
        let tokens = d.tokens().to_vec();
        assert_eq!(d.report(), detect(&tokens, 0.5));
    }

    #[test]
    fn a_number_can_still_grow() {
        // "3" then "." could become "3.5".
        let mut d = StreamDetector::new(0.5);
        d.push_tokens([tok(" in", 0.99), tok(" 3", 0.2), tok(".", 0.99)]);
        let out = d.push_token(tok("5", 0.99));
        assert!(out.is_empty());
        // " m" could still be the start of a flagged word that merges.
        assert!(d.push_token(tok(" m", 0.99)).is_empty());
        let out = d.push_token(tok(" high", 0.99));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, " 3.5");
    }

    #[test]
    fn only_the_unsettled_tail_is_reread() {
        let words = [" the", " painter", " died", " in", " Dordrecht", "."];
        let mut d = StreamDetector::new(0.5);
        let mut flagged = 0;
        for i in 0..6000 {
            let p = if i % 97 == 0 { 0.2 } else { 0.95 };
            flagged += d.push_token(tok(words[i % words.len()], p)).len();
            assert!(
                d.tokens.len() - d.base <= 4,
                "window {} at {i}",
                d.tokens.len() - d.base
            );
        }
        flagged += d.finish().unwrap().len();
        let tokens = d.tokens().to_vec();
        assert_eq!(flagged, detect(&tokens, 0.5).spans.len());
        assert_eq!(d.report(), detect(&tokens, 0.5));
        // Chinese has no spaces; each ideograph settles the text before it.
        let mut d = StreamDetector::new(0.5);
        for (i, c) in "埃菲尔铁塔在巴黎。".repeat(200).chars().enumerate() {
            d.push_token(tok(&c.to_string(), if i % 31 == 0 { 0.1 } else { 0.9 }));
            assert!(d.tokens.len() - d.base <= 3);
        }
    }

    #[test]
    fn sse_split_inside_utf8_and_lines() {
        let s = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\" Zürich\"},\"logprobs\":{\"content\":[{\"token\":\" Zürich\",\"logprob\":-2.0}]}}]}\r\n\r\n";
        let mut d = StreamDetector::new(0.5);
        for b in s.as_bytes() {
            d.push_sse([*b]).unwrap();
        }
        let out = d.finish().unwrap();
        assert_eq!(out[0].text, " Zürich");
        assert_eq!(d.format(), Some(Format::ChatCompletions));
    }

    #[test]
    fn errors_and_missing_logprobs() {
        let mut d = StreamDetector::new(0.5);
        let e = d
            .push_sse("data: {\"error\":{\"message\":\"overloaded\"}}\n\n")
            .unwrap_err();
        assert!(matches!(e, ParseError::ApiError(_)));
        let mut d = StreamDetector::new(0.5);
        d.push_sse("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n")
            .unwrap();
        assert!(d.missing_logprobs());
        assert!(d.tokens().is_empty());
        assert!(matches!(
            parse_stream("data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hi\"}}]}\n\n"),
            Err(ParseError::NoLogprobs)
        ));
        assert!(matches!(parse_stream(""), Err(ParseError::Empty)));
        assert!(matches!(
            StreamDetector::new(0.5).push_event("{oops"),
            Err(ParseError::InvalidJson(_))
        ));
        assert!(StreamDetector::try_new(1.5).is_err());
        assert_eq!(StreamDetector::new(f64::NAN).threshold(), 0.0);
    }

    #[test]
    fn other_choices_are_skipped() {
        let s = concat!(
            "data: {\"choices\":[{\"index\":1,\"delta\":{\"content\":\"B\"},\"logprobs\":{\"content\":[{\"token\":\"B\",\"logprob\":-0.1}]}}]}\n\n",
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"A\"},\"logprobs\":{\"content\":[{\"token\":\"A\",\"logprob\":-0.1}]}}]}\n\n",
        );
        assert_eq!(parse_stream(s).unwrap()[0].token, "A");
        assert_eq!(parse_stream_choice(s, 1).unwrap()[0].token, "B");
    }
}
