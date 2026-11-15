//! `StreamDetector` on recorded streams, cut into network chunks of every
//! size: the spans it returns along the way are the spans `detect` finds in
//! the whole answer, and most come out before the stream ends.

use llm_token_visualizer::stream::parse_stream;
use llm_token_visualizer::{detect, Format, ParseError, Span, StreamDetector};

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/providers/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

const STREAMS: [(&str, Format); 8] = [
    ("ollama_chat_stream.ndjson", Format::Ollama),
    ("ollama_v1_chat_stream.sse", Format::ChatCompletions),
    ("llamacpp_completion_stream.sse", Format::LlamaCpp),
    ("llamacpp_v1_chat_stream.sse", Format::ChatCompletions),
    ("openai_responses_stream.sse", Format::Responses),
    ("gemini_stream.sse", Format::Gemini),
    ("tgi_stream.sse", Format::Tgi),
    ("together_stream.sse", Format::ChatCompletions),
];

/// Feed `data` in chunks of `size` bytes; return the spans each call gave,
/// with the number of bytes read when each came out.
fn run(data: &str, size: usize, threshold: f64) -> (StreamDetector, Vec<(usize, Span)>) {
    let mut det = StreamDetector::new(threshold);
    let mut out = Vec::new();
    let mut read = 0;
    for chunk in data.as_bytes().chunks(size) {
        read += chunk.len();
        for s in det.push_sse(chunk).unwrap() {
            out.push((read, s));
        }
    }
    for s in det.finish().unwrap() {
        out.push((usize::MAX, s));
    }
    (det, out)
}

#[test]
fn streamed_spans_equal_detect_on_the_whole_answer() {
    for (name, format) in STREAMS {
        let data = fixture(name);
        let tokens = parse_stream(&data).unwrap();
        for threshold in [0.3, 0.5, 0.7, 0.9, 0.99] {
            let expected = detect(&tokens, threshold);
            for size in [1, 2, 3, 7, 64, 500, data.len()] {
                let (det, out) = run(&data, size, threshold);
                let spans: Vec<Span> = out.into_iter().map(|(_, s)| s).collect();
                assert_eq!(spans, expected.spans, "{name} t={threshold} size={size}");
                assert_eq!(det.report(), expected, "{name} t={threshold} size={size}");
                assert_eq!(det.spans(), &expected.spans[..]);
                assert_eq!(det.format(), Some(format), "{name}");
                assert!(!det.missing_logprobs(), "{name}");
            }
        }
    }
}

#[test]
fn spans_come_out_before_the_stream_ends() {
    for (name, _) in STREAMS {
        let data = fixture(name);
        let (_, out) = run(&data, 16, 0.5);
        // The year is flagged and settled before the answer ends.
        let (at, year) = &out[0];
        assert!(year.text.starts_with("16"), "{name}: {:?}", year.text);
        assert!(*at != usize::MAX, "{name}: year flagged only by finish()");
        // It is final once the event carrying the place name ("in The
        // Hague" / "in Delft") is read: the next word after " in".
        let place = data.find(" Hague").or_else(|| data.find(" Delft")).unwrap();
        let event_end = place + data[place..].find('\n').unwrap();
        assert!(
            *at <= event_end + 16,
            "{name}: year flagged at byte {at}, place event ends at {event_end}"
        );
    }
}

#[test]
fn done_markers() {
    for (name, _) in STREAMS {
        let (det, _) = run(&fixture(name), 100, 0.5);
        // Gemini and TGI streams just end; the others say so.
        let says_done = !matches!(name, "gemini_stream.sse" | "tgi_stream.sse");
        assert_eq!(det.is_done(), says_done, "{name}");
    }
}

#[test]
fn push_event_one_line_at_a_time() {
    let data = fixture("ollama_chat_stream.ndjson");
    let mut det = StreamDetector::new(0.5);
    let mut spans = Vec::new();
    for line in data.lines() {
        spans.extend(det.push_event(line).unwrap());
    }
    spans.extend(det.finish().unwrap());
    let texts: Vec<&str> = spans.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["1620", " The"]);
    assert!(det.is_done());
    assert_eq!(det.text(), "Aelbert Cuyp died in 1620 in The Hague.");
}

#[test]
fn a_stream_without_logprobs_is_reported() {
    let (det, out) = run(&fixture("ollama_v1_responses_stream.sse"), 50, 0.5);
    assert!(out.is_empty());
    assert!(det.tokens().is_empty());
    assert!(det.missing_logprobs());
    assert!(det.is_done());
}

#[test]
fn an_error_event_stops_the_stream() {
    let data = concat!(
        "event: response.created\n",
        "data: {\"type\":\"response.created\",\"response\":{\"object\":\"response\",\"error\":null,\"output\":[]}}\n\n",
        "event: error\n",
        "data: {\"type\":\"error\",\"code\":\"rate_limit_exceeded\",\"message\":\"slow down\"}\n\n",
    );
    let mut det = StreamDetector::new(0.5);
    let e = det.push_sse(data).unwrap_err();
    assert!(
        matches!(e, ParseError::ApiError(ref v) if v["message"] == "slow down"),
        "{e:?}"
    );
}

#[test]
fn multi_line_sse_data_is_joined() {
    // An SSE event may split its data over several `data:` lines.
    let data = concat!(
        "data: {\"choices\":[{\"index\":0,\n",
        "data: \"logprobs\":{\"content\":[{\"token\":\" Delft\",\"logprob\":-2.0}]}}]}\n",
        "\n",
    );
    let mut det = StreamDetector::new(0.5);
    det.push_sse(data).unwrap();
    let out = det.finish().unwrap();
    assert_eq!(out[0].text, " Delft");
}
