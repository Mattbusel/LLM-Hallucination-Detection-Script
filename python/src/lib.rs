//! The native half of the `llm_token_visualizer` Python package.
//!
//! The Python side (`python/llm_token_visualizer/__init__.py`) turns whatever
//! the caller passed (an OpenAI SDK object, a dict, a JSON string, a vLLM
//! output) into JSON and calls [`score_json`]; everything else (parsing,
//! detection, HTML) is the Rust crate.

use llm_token_visualizer::report::{self, Meta};
use llm_token_visualizer::{parse_logprobs, try_detect, LogprobToken, ParseError, Report};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

create_exception!(
    llm_token_visualizer,
    LogprobsError,
    PyValueError,
    "The response could not be read as token logprobs (bad JSON, an API error, an empty or malformed logprobs block)."
);
create_exception!(
    llm_token_visualizer,
    NoLogprobsError,
    LogprobsError,
    "The response has no token logprobs: the request did not ask for them (logprobs=True), or the model does not return them."
);

const HINT: &str = "request the completion with logprobs=True (and top_logprobs=5 to see \
                    the alternatives), e.g. client.chat.completions.create(..., logprobs=True, top_logprobs=5)";

/// `choice` is the index the caller asked for. The Python side has already
/// cut `choices` down to that one entry, so the parser always reads
/// `choices[0]`; the message names the caller's index instead.
fn to_py_err(e: ParseError, choice: usize) -> PyErr {
    match e {
        ParseError::NoLogprobs => NoLogprobsError::new_err(format!(
            "no token logprobs in this response (choices[{choice}].logprobs is missing); {HINT}"
        )),
        ParseError::NullLogprobs => NoLogprobsError::new_err(format!(
            "token logprobs are None in this response (choices[{choice}].logprobs is null); {HINT}"
        )),
        ParseError::InvalidJson(err) => {
            LogprobsError::new_err(format!("input is not valid JSON: {err}"))
        }
        other => LogprobsError::new_err(other.to_string()),
    }
}

/// One run of low-confidence words.
///
/// `start` and `end` are character offsets into `Score.text`, so
/// `score.text[span.start:span.end] == span.text`.
///
/// Spans compare equal when every field is equal (so they are not hashable).
#[pyclass(frozen, eq, module = "llm_token_visualizer", name = "Span")]
#[derive(PartialEq)]
struct PySpan {
    /// Character offset of the first character (inclusive).
    #[pyo3(get)]
    start: usize,
    /// Character offset after the last character (exclusive).
    #[pyo3(get)]
    end: usize,
    /// Index of the first token (inclusive).
    #[pyo3(get)]
    token_start: usize,
    /// Index after the last token (exclusive).
    #[pyo3(get)]
    token_end: usize,
    /// The span text.
    #[pyo3(get)]
    text: String,
    /// Lowest natural-log probability among the span's tokens.
    #[pyo3(get)]
    min_logprob: f64,
    /// Mean natural-log probability of the span's tokens.
    #[pyo3(get)]
    mean_logprob: f64,
    /// Probability of the weakest word token, the one that got the span flagged.
    #[pyo3(get)]
    min_prob: f64,
    /// The weakest word token.
    #[pyo3(get)]
    weakest_token: String,
    /// What the model weighed at the weakest token: (token, probability), most likely first.
    #[pyo3(get)]
    alternatives: Vec<(String, f64)>,
    /// Entropy in bits of those alternatives (a lower bound), or None without top_logprobs.
    #[pyo3(get)]
    entropy_bits: Option<f64>,
    description: String,
}

#[pymethods]
impl PySpan {
    /// One line, e.g. `p=0.41 at " Marco"; model also considered " Jan" (0.38)`.
    fn describe(&self) -> String {
        self.description.clone()
    }

    /// The span as a plain dict (JSON-serializable).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("start", self.start)?;
        d.set_item("end", self.end)?;
        d.set_item("token_start", self.token_start)?;
        d.set_item("token_end", self.token_end)?;
        d.set_item("text", &self.text)?;
        d.set_item("min_logprob", self.min_logprob)?;
        d.set_item("mean_logprob", self.mean_logprob)?;
        d.set_item("min_prob", self.min_prob)?;
        d.set_item("weakest_token", &self.weakest_token)?;
        d.set_item("alternatives", self.alternatives.clone())?;
        d.set_item("entropy_bits", self.entropy_bits)?;
        Ok(d)
    }

    fn __repr__(&self) -> String {
        format!(
            "Span(start={}, end={}, text={:?}, min_logprob={:.4}, mean_logprob={:.4})",
            self.start, self.end, self.text, self.min_logprob, self.mean_logprob
        )
    }
}

/// The result of `score()`: the answer text, its flagged spans and overall
/// confidence numbers.
#[pyclass(frozen, module = "llm_token_visualizer", name = "Score")]
struct PyScore {
    tokens: Vec<LogprobToken>,
    report: Report,
    model: Option<String>,
}

impl PyScore {
    fn build_spans(&self) -> Vec<PySpan> {
        self.report
            .spans
            .iter()
            .map(|s| {
                let chars = s.char_range(&self.tokens);
                let end = s.end.min(self.tokens.len());
                let start = s.start.min(end);
                let known: Vec<f64> = self.tokens[start..end]
                    .iter()
                    .filter(|t| t.has_logprob())
                    .map(|t| t.logprob)
                    .collect();
                let (min_logprob, mean_logprob) = if known.is_empty() {
                    (f64::NAN, f64::NAN)
                } else {
                    (
                        known.iter().copied().fold(f64::INFINITY, f64::min),
                        known.iter().sum::<f64>() / known.len() as f64,
                    )
                };
                PySpan {
                    start: chars.start,
                    end: chars.end,
                    token_start: s.start,
                    token_end: s.end,
                    text: s.text.clone(),
                    min_logprob,
                    mean_logprob,
                    min_prob: s.min_prob,
                    weakest_token: s.weakest_token.clone(),
                    alternatives: s
                        .alternatives
                        .iter()
                        .map(|c| (c.token.clone(), c.prob))
                        .collect(),
                    entropy_bits: s.entropy_bits,
                    description: s.describe(),
                }
            })
            .collect()
    }
}

#[pymethods]
impl PyScore {
    /// The answer, tokens joined. Span offsets index into it.
    #[getter]
    fn text(&self) -> &str {
        &self.report.text
    }

    /// Probability below which a word was flagged.
    #[getter]
    fn threshold(&self) -> f64 {
        self.report.threshold
    }

    /// Number of tokens in the answer.
    #[getter]
    fn token_count(&self) -> usize {
        self.report.token_count
    }

    /// Mean token probability (tokens without a logprob are left out).
    #[getter]
    fn mean_prob(&self) -> f64 {
        self.report.mean_prob
    }

    /// Perplexity, exp(-mean logprob): 1.0 when the model was certain of every token.
    #[getter]
    fn perplexity(&self) -> f64 {
        self.report.perplexity
    }

    /// True when at least one span is below the threshold.
    #[getter]
    fn flagged(&self) -> bool {
        self.report.flagged()
    }

    /// The `model` field of the response, when it had one.
    #[getter]
    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    /// The low-confidence spans, in order.
    fn spans(&self) -> Vec<PySpan> {
        self.build_spans()
    }

    /// A self-contained HTML page (no scripts, no external assets) with every
    /// token shaded by confidence and the flagged spans listed. `title` is
    /// shown in the page header (default: the model name).
    #[pyo3(signature = (title=None))]
    fn to_html(&self, title: Option<String>) -> String {
        report::html(&self.tokens, &self.report, &self.meta(title))
    }

    /// The same report as Markdown, for a pull request or issue comment.
    #[pyo3(signature = (title=None))]
    fn to_markdown(&self, title: Option<String>) -> String {
        report::markdown(&self.tokens, &self.report, &self.meta(title))
    }

    /// A short plain-text summary: one header line, then one line per span.
    fn summary(&self) -> String {
        let r = &self.report;
        let mut out = match r.spans.len() {
            0 => format!("No low-confidence spans below p={:.2}", r.threshold),
            1 => format!("1 low-confidence span below p={:.2}", r.threshold),
            n => format!("{n} low-confidence spans below p={:.2}", r.threshold),
        };
        out.push_str(&format!(
            " in {} tokens (mean p={:.2}, perplexity {:.2})",
            r.token_count, r.mean_prob, r.perplexity
        ));
        out.push_str(if r.spans.is_empty() { "." } else { ":" });
        for s in self.build_spans() {
            out.push_str(&format!(
                "\n  [{}:{}] {:?} {}",
                s.start, s.end, s.text, s.description
            ));
        }
        out
    }

    /// The report as JSON, the same shape as the CLI's `--format json`
    /// (token indices, not character offsets).
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.report).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Score(spans={}, threshold={:.2}, tokens={}, mean_prob={:.3}, perplexity={:.3})",
            self.report.spans.len(),
            self.report.threshold,
            self.report.token_count,
            self.report.mean_prob,
            self.report.perplexity
        )
    }
}

impl PyScore {
    fn meta(&self, title: Option<String>) -> Meta {
        Meta {
            source: title
                .or_else(|| self.model.clone())
                .unwrap_or_else(|| "response".to_string()),
            model: self.model.clone(),
        }
    }
}

/// Parse a response (as JSON text) and flag spans below `threshold`.
/// `model` is the response's model name when the caller already knows it;
/// otherwise it is read from the JSON. `choice` is the choice index the caller
/// picked, used only in error messages.
#[pyfunction]
#[pyo3(signature = (json, threshold, model=None, *, choice=0))]
fn score_json(
    json: &str,
    threshold: f64,
    model: Option<String>,
    choice: usize,
) -> PyResult<PyScore> {
    let tokens = parse_logprobs(json).map_err(|e| to_py_err(e, choice))?;
    let report =
        try_detect(&tokens, threshold).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let model = model.or_else(|| Meta::from_response("", json).model);
    Ok(PyScore {
        tokens,
        report,
        model,
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score_json, m)?)?;
    m.add_class::<PyScore>()?;
    m.add_class::<PySpan>()?;
    m.add("LogprobsError", m.py().get_type::<LogprobsError>())?;
    m.add("NoLogprobsError", m.py().get_type::<NoLogprobsError>())?;
    m.add("DEFAULT_THRESHOLD", llm_token_visualizer::DEFAULT_THRESHOLD)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
