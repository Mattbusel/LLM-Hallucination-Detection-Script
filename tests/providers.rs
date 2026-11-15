//! `parse_any` and the stream reader against one response per provider shape.
//!
//! Fixtures in `tests/fixtures/providers/`:
//!
//! - Captured from real servers (2026-11-15, the same question each time):
//!   `ollama_*` from Ollama 0.40.16 running `gemma4:e2b-it-qat` (native
//!   `/api/chat` and `/api/generate`, plus its OpenAI-compatible `/v1/chat/
//!   completions`, `/v1/completions` and `/v1/responses`, which ignore the
//!   logprobs parameters), and `llamacpp_*` from the llama.cpp server bundled
//!   with that Ollama (`/completion` with `n_probs`, with and without
//!   `post_sampling_probs`, streamed, and `/v1/completions` and a streamed
//!   `/v1/chat/completions`). The local model path is replaced with a file name.
//! - Hand-written in each provider's documented shape, with the token
//!   logprobs of the `ollama_chat.json` capture: `openai_responses*`,
//!   `gemini_*`, `tgi_*`, `together_*`, `fireworks_chat.json` and
//!   `vllm_completions.json` (which holds a `-Infinity` the way Python's
//!   `json` module writes it).

use llm_token_visualizer::formats::{detect_format, parse_value, Format};
use llm_token_visualizer::stream::{parse_stream, parse_stream_choice};
use llm_token_visualizer::{detect, parse_any, parse_logprobs, LogprobToken, ParseError};
use serde_json::Value;

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/providers/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

fn text(tokens: &[LogprobToken]) -> String {
    tokens.iter().map(|t| t.token.as_str()).collect()
}

fn span_texts(tokens: &[LogprobToken], threshold: f64) -> Vec<String> {
    detect(tokens, threshold)
        .spans
        .into_iter()
        .map(|s| s.text)
        .collect()
}

/// Every non-streamed fixture: its shape, the answer text the provider put
/// next to the logprobs (so no token was lost or reordered) and the spans.
#[test]
fn every_provider_shape() {
    let cases: &[(&str, Format, &str, &[&str])] = &[
        (
            "ollama_chat.json",
            Format::Ollama,
            "/message/content",
            &["1630", " The"],
        ),
        (
            "ollama_generate.json",
            Format::Ollama,
            "/response",
            &["1620", " The"],
        ),
        (
            "ollama_v1_chat.json",
            Format::ChatCompletions,
            "/choices/0/message/content",
            &["1620", " The"],
        ),
        (
            "llamacpp_completion.json",
            Format::LlamaCpp,
            "/content",
            &["1660", " Delft"],
        ),
        (
            "llamacpp_completion_post.json",
            Format::LlamaCpp,
            "/content",
            &["1660", " Delft"],
        ),
        (
            "llamacpp_v1_completions.json",
            Format::ChatCompletions,
            "/choices/0/text",
            &["1660", " The"],
        ),
        (
            "openai_responses.json",
            Format::Responses,
            "/output/1/content/0/text",
            &["1630", " The"],
        ),
        (
            "gemini_generate.json",
            Format::Gemini,
            "/candidates/0/content/parts/0/text",
            &["1630", " The"],
        ),
        (
            "tgi_generate.json",
            Format::Tgi,
            "/0/generated_text",
            &["1630", " The"],
        ),
        (
            "together_chat.json",
            Format::Completions,
            "/choices/0/message/content",
            &["1630", " The"],
        ),
        (
            "fireworks_chat.json",
            Format::ChatCompletions,
            "/choices/0/message/content",
            &["1630", " The"],
        ),
    ];
    for (name, format, text_at, spans) in cases {
        let json = fixture(name);
        let value: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(detect_format(&value), Some(*format), "{name}");
        let (got, tokens) = parse_value(&value, 0).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(got, *format, "{name}");
        assert_eq!(
            text(&tokens),
            value.pointer(text_at).and_then(Value::as_str).unwrap(),
            "{name}"
        );
        assert_eq!(span_texts(&tokens, 0.5), *spans, "{name}");
        // parse_logprobs is parse_any.
        let again = parse_logprobs(&json).unwrap();
        assert_eq!(text(&again), text(&tokens), "{name}");
    }
}

/// The hand-written fixtures carry the logprobs of the Ollama capture, so
/// every shape must give back the same tokens.
#[test]
fn same_tokens_from_every_shape() {
    let reference = parse_any(&fixture("ollama_chat.json")).unwrap();
    for name in [
        "openai_responses.json",
        "gemini_generate.json",
        "tgi_generate.json",
        "together_chat.json",
        "fireworks_chat.json",
        "vllm_completions.json",
    ] {
        let tokens = parse_any(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(tokens.len(), reference.len(), "{name}");
        for (a, b) in tokens.iter().zip(&reference) {
            assert_eq!(a.token, b.token, "{name}");
            assert_eq!(a.logprob, b.logprob, "{name} {:?}", a.token);
            let alts = |t: &LogprobToken| {
                let mut v: Vec<(String, u64)> = t
                    .top_logprobs
                    .iter()
                    .filter(|x| x.logprob.is_finite() && x.logprob > -1e300)
                    .map(|x| (x.token.clone(), x.logprob.to_bits()))
                    .collect();
                v.sort();
                v
            };
            assert_eq!(alts(a), alts(b), "{name} {:?}", a.token);
        }
    }
}

#[test]
fn vllm_infinity_is_probability_zero() {
    let json = fixture("vllm_completions.json");
    assert!(json.contains("-Infinity"));
    assert!(serde_json::from_str::<Value>(&json).is_err());
    let tokens = parse_any(&json).unwrap();
    let alt = tokens[13]
        .top_logprobs
        .iter()
        .find(|a| a.token == "<|endoftext|>")
        .unwrap();
    assert_eq!(alt.logprob.exp(), 0.0);
    assert_eq!(alt.logprob, f64::MIN);
}

#[test]
fn gemini_leaves_out_zero_logprobs() {
    // Gemini's JSON drops fields that hold their default value, so a token
    // the model was certain of has no logProbability.
    let json = r#"{"candidates":[{"content":{"parts":[{"text":"Paris."}]},"logprobsResult":{
        "topCandidates":[{"candidates":[{"token":"Paris"},{"token":"Lyon","logProbability":-18.2}]},
                         {"candidates":[{"token":".","logProbability":-0.01}]}],
        "chosenCandidates":[{"token":"Paris"},{"token":".","logProbability":-0.01}]}}]}"#;
    let tokens = parse_any(json).unwrap();
    assert_eq!(tokens.len(), 2);
    assert_eq!(
        (tokens[0].token.as_str(), tokens[0].logprob),
        ("Paris", 0.0)
    );
    assert_eq!(tokens[0].top_logprobs[0].logprob, 0.0);
    assert_eq!(tokens[0].top_logprobs[1].token, "Lyon");
    assert!(!detect(&tokens, 0.5).flagged());
}

#[test]
fn special_tokens_are_dropped_everywhere() {
    // TGI marks the end of sequence as special; llama.cpp writes end-of-turn
    // as an empty token.
    let tgi = parse_any(&fixture("tgi_generate.json")).unwrap();
    assert!(tgi.iter().all(|t| t.token != "<eos>"));
    let raw: Value = serde_json::from_str(&fixture("llamacpp_completion.json")).unwrap();
    let listed = raw["completion_probabilities"].as_array().unwrap().len();
    let llama = parse_any(&fixture("llamacpp_completion.json")).unwrap();
    assert_eq!(llama.len(), listed - 1);
}

#[test]
fn llamacpp_post_sampling_probs_become_logprobs() {
    let tokens = parse_any(&fixture("llamacpp_completion_post.json")).unwrap();
    let delft = tokens.iter().find(|t| t.token == " Delft").unwrap();
    assert!(delft.prob() > 0.0 && delft.prob() < 0.5);
    assert!(delft.top_logprobs.len() > 1);
    assert!(delft.top_logprobs[0].logprob >= delft.top_logprobs[1].logprob);
}

#[test]
fn responses_without_logprobs_say_so() {
    // Ollama's /v1/responses accepts the request but returns `logprobs: []`.
    let e = parse_any(&fixture("ollama_v1_responses.json")).unwrap_err();
    assert!(matches!(e, ParseError::NoLogprobs), "{e:?}");
    for name in ["ollama_chat_no_logprobs.json", "ollama_v1_completions.json"] {
        let e = parse_any(&fixture(name)).unwrap_err();
        assert!(matches!(e, ParseError::NoLogprobs), "{name}: {e:?}");
        assert!(e.to_string().contains("logprobs"));
    }
}

#[test]
fn api_errors_from_each_provider() {
    for body in [
        r#"{"error":{"message":"Incorrect API key","type":"invalid_request_error"}}"#,
        r#"{"error":"model 'nope' not found"}"#,
        r#"{"error":{"code":400,"message":"API key not valid","status":"INVALID_ARGUMENT"}}"#,
        r#"{"type":"response.failed","response":{"object":"response","error":{"code":"server_error","message":"boom"},"output":[]}}"#,
    ] {
        assert!(
            matches!(parse_any(body), Err(ParseError::ApiError(_))),
            "{body}"
        );
    }
    // A Responses object always has an `error` field; null is not an error.
    let ok = parse_any(&fixture("openai_responses.json"));
    assert!(ok.is_ok());
}

#[test]
fn choices_by_index() {
    let json = r#"{"choices":[
        {"index":0,"logprobs":{"content":[{"token":"Paris","logprob":-0.1}]}},
        {"index":1,"logprobs":{"content":[{"token":"Lyon","logprob":-1.1}]}}]}"#;
    assert_eq!(parse_any(json).unwrap()[0].token, "Paris");
    let second = llm_token_visualizer::parse_choice(json, 1).unwrap();
    assert_eq!(second[0].token, "Lyon");
    assert!(matches!(
        llm_token_visualizer::parse_choice(json, 2),
        Err(ParseError::NoLogprobs)
    ));
    // A list cut down to one choice keeps its index (the Python bindings
    // do this for `choice=1`).
    let cut =
        r#"{"choices":[{"index":1,"logprobs":{"content":[{"token":"Lyon","logprob":-1.1}]}}]}"#;
    assert_eq!(parse_any(cut).unwrap()[0].token, "Lyon");
    // Gemini candidates leave out index 0.
    let gemini = r#"{"candidates":[
        {"logprobsResult":{"chosenCandidates":[{"token":"a","logProbability":-0.1}]}},
        {"index":1,"logprobsResult":{"chosenCandidates":[{"token":"b"}]}}]}"#;
    assert_eq!(parse_any(gemini).unwrap()[0].token, "a");
    let b = llm_token_visualizer::parse_choice(gemini, 1).unwrap();
    assert_eq!((b[0].token.as_str(), b[0].logprob), ("b", 0.0));
}

#[test]
fn single_events_and_choices() {
    // One Responses delta event, one Ollama stream line, one TGI stream
    // event and a lone choice object all parse on their own.
    let delta = r#"{"type":"response.output_text.delta","delta":" Delft","logprobs":[{"token":" Delft","logprob":-2.0,"top_logprobs":[]}]}"#;
    assert_eq!(parse_any(delta).unwrap()[0].token, " Delft");
    let line = fixture("ollama_chat_stream.ndjson");
    let first = line.lines().next().unwrap();
    assert_eq!(parse_any(first).unwrap().len(), 1);
    let tgi = r#"{"index":1,"token":{"id":5,"text":" Delft","logprob":-2.0,"special":false},"top_tokens":[],"generated_text":null,"details":null}"#;
    assert_eq!(parse_any(tgi).unwrap()[0].token, " Delft");
    let choice = r#"{"index":0,"message":{"content":"Hi"},"logprobs":{"content":[{"token":"Hi","logprob":-0.1}]}}"#;
    assert_eq!(parse_any(choice).unwrap()[0].token, "Hi");
}

#[test]
fn recorded_streams_match_the_whole_responses() {
    // Same tokens, whether streamed or not, and nothing counted twice
    // (Responses repeats the text in `.done` events and `response.completed`).
    for (stream, whole) in [
        ("openai_responses_stream.sse", "openai_responses.json"),
        ("gemini_stream.sse", "gemini_generate.json"),
        ("tgi_stream.sse", "tgi_generate.json"),
    ] {
        assert_eq!(
            parse_stream(&fixture(stream)).unwrap(),
            parse_any(&fixture(whole)).unwrap(),
            "{stream}"
        );
    }
    // Together's stream has the logprob of each token but no alternatives.
    let together = parse_stream(&fixture("together_stream.sse")).unwrap();
    let whole = parse_any(&fixture("together_chat.json")).unwrap();
    assert_eq!(together.len(), whole.len());
    for (a, b) in together.iter().zip(&whole) {
        assert_eq!((&a.token, a.logprob), (&b.token, b.logprob));
    }
    // The real streams: the text matches the streamed content deltas.
    for (name, expected) in [
        (
            "ollama_chat_stream.ndjson",
            "Aelbert Cuyp died in 1620 in The Hague.",
        ),
        (
            "ollama_v1_chat_stream.sse",
            "Aelbert Cuyp died in 1620 in The Hague.",
        ),
        (
            "llamacpp_completion_stream.sse",
            " Aelbert Cuyp died in 1660 in Delft.",
        ),
        (
            "llamacpp_v1_chat_stream.sse",
            "Aelbert Cuyp died in 1623 in The Hague.",
        ),
    ] {
        let tokens = parse_stream(&fixture(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(text(&tokens), expected, "{name}");
    }
    // llama.cpp's streamed /completion carries the same numbers as the
    // non-streamed call with the same seed.
    assert_eq!(
        parse_stream(&fixture("llamacpp_completion_stream.sse")).unwrap(),
        parse_any(&fixture("llamacpp_completion.json")).unwrap()
    );
    assert!(parse_stream_choice(&fixture("ollama_v1_chat_stream.sse"), 1).is_err());
}

#[test]
fn stream_without_logprobs_is_no_logprobs() {
    let e = parse_stream(&fixture("ollama_v1_responses_stream.sse")).unwrap_err();
    assert!(matches!(e, ParseError::NoLogprobs), "{e:?}");
}
