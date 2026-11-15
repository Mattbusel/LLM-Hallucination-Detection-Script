# llm-token-visualizer for Python

Find the words an LLM answer was unsure about. Ask the API for token logprobs, pass the response to `score`, and get back the low-confidence spans, an HTML report and a summary. No extra API calls, no second model, no Python dependencies: the parsing and detection are the [llm-token-visualizer](https://crates.io/crates/llm-token-visualizer) Rust crate, compiled into the wheel.

```bash
pip install llm-token-visualizer
```

## Score an OpenAI answer

```python
from openai import OpenAI
import llm_token_visualizer as ltv
resp = OpenAI().chat.completions.create(model="gpt-4.1-mini", logprobs=True, top_logprobs=5,
    messages=[{"role": "user", "content": "Who won the 1998 Tour de France?"}])
result = ltv.score(resp)
print(result.summary())
```

`logprobs=True` is what makes this work. `top_logprobs=5` is optional and adds the alternatives the model was weighing at each weak word. For an answer of "The 1998 Tour de France was won by Marco Pantani." where the model hesitated on the name (the test fixture `python/tests/fixtures/chat_tour.json`), the summary is:

```text
1 low-confidence span below p=0.50 in 14 tokens (mean p=0.96, perplexity 1.07):
  [34:40] " Marco" p=0.41 at " Marco"; model also considered " Jan" (0.38), " Bjarne" (0.12), " Marc" (0.01)
```

`result.to_html()` gives the same answer as a page with every token shaded by confidence: `open("report.html", "w", encoding="utf-8").write(result.to_html())`.

A low probability is a signal, not proof. A model can be confidently wrong, and it can be unsure about wording while the facts are right. Use the spans to decide what to check, retry or show with a warning.

## What `score` returns

`score(resp, threshold=0.5, *, choice=0)` returns a `Score`:

| | |
|---|---|
| `spans()` | The low-confidence spans, in order. |
| `to_html(title=None)` | A self-contained HTML page (no scripts, no external files) with every token shaded by confidence. Hover a token for its probability and the alternatives. |
| `to_markdown(title=None)` | The same report as Markdown, for a pull request or issue comment. |
| `summary()` | One header line, then one line per span. |
| `to_json()` | The report as JSON, the shape the CLI's `--format json` writes. |
| `text` | The answer, tokens joined. Span offsets index into it. |
| `flagged` | `True` when there is at least one span. |
| `mean_prob`, `perplexity`, `token_count`, `threshold`, `model` | Whole-answer numbers. |

Each `Span` has:

| | |
|---|---|
| `start`, `end` | Character offsets into `Score.text`: `result.text[s.start:s.end] == s.text`. |
| `text` | The span text. |
| `min_logprob`, `mean_logprob` | Lowest and mean natural-log probability of the span's tokens. |
| `min_prob`, `weakest_token` | The word token that got the span flagged, and its probability. |
| `alternatives` | `(token, probability)` pairs the model weighed at that token, most likely first. Empty without `top_logprobs`. |
| `entropy_bits` | Entropy of those alternatives in bits (a lower bound), or `None`. |
| `token_start`, `token_end` | The same span as token indices. |
| `describe()`, `to_dict()` | One line of text, or a plain dict. |

Two spans are equal (`==`) when every field is equal, so `score(r).spans() == score(r).spans()`. Spans are not hashable; use `s.to_dict()` or `(s.start, s.end)` as a key.

A word is flagged when any of its tokens with letters or digits has a probability strictly below `threshold`. Neighbouring flagged words merge into one span, and punctuation or whitespace alone never triggers a flag.

`text` is the tokens joined. It matches `message.content` except where the API splits one character (an emoji, some CJK) across tokens.

## Gate a CI or eval run

`gate(resp, threshold)` is `True` when nothing in the answer is below the threshold:

```python
import llm_token_visualizer as ltv

def test_capital_answer_is_confident(recorded_response):
    assert ltv.gate(recorded_response, 0.4), ltv.score(recorded_response, 0.4).summary()
```

## What it reads

- OpenAI SDK objects: `ChatCompletion`, legacy `Completion`, a single `Choice`, its `logprobs`, or the token list `resp.choices[0].logprobs.content`. Lists and dicts that hold SDK objects work too.
- The same responses as a dict, a JSON string or bytes. That covers any OpenAI-compatible server that returns logprobs: vLLM, OpenRouter, Together, Ollama.
- The Completions shape (`tokens` / `token_logprobs` / `top_logprobs`), including the `null` first logprob of an `echo=True` request (kept in the text, never flagged).
- vLLM offline generation: a `RequestOutput` or `CompletionOutput` from `LLM.generate(..., SamplingParams(logprobs=5))`.
- Google Gemini `generateContent` responses made with `responseLogprobs: true`.

With `n > 1`, pick the choice with `score(resp, choice=1)`.

## Errors

- No logprobs in the response, or `logprobs` is `None`: `NoLogprobsError`, saying to request the completion with `logprobs=True` and naming the choice (`choices[1].logprobs is null` for `choice=1`). Catch it to retry with logprobs on.
- An API error body, invalid JSON, bytes that are not UTF-8, or a malformed logprobs block: `LogprobsError`.
- A threshold outside 0 to 1, or a `choice` the response does not have: `ValueError`.
- An input it cannot read (not a response, dict, string or vLLM output): `TypeError`.

`NoLogprobsError` is a subclass of `LogprobsError`, and both are `ValueError`s.

```python
try:
    result = ltv.score(resp)
except ltv.NoLogprobsError:
    resp = client.chat.completions.create(**request, logprobs=True, top_logprobs=5)
    result = ltv.score(resp)
```

## Platforms

One abi3 wheel per platform covers CPython 3.9 and newer: Linux x86_64 and aarch64 (manylinux and musllinux), macOS x86_64 and arm64, Windows x86_64. Elsewhere pip builds from the source distribution, which needs a Rust toolchain.

## Build from source

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest
maturin develop --release
pytest
```

The bindings are in `python/` (`python/src/lib.rs` and `python/llm_token_visualizer/`); `pyproject.toml` is at the repository root. Wheels are built and published to PyPI by `.github/workflows/python.yml` on tags named `py-v<version>`.

If that workflow cannot run, a release can be uploaded by hand with a PyPI API token. This builds only the wheel for the machine it runs on plus the sdist, so other platforms install from the sdist (which needs Rust) until their wheels are uploaded:

```bash
pip install maturin
maturin build --release --out dist
maturin sdist --out dist
MATURIN_PYPI_TOKEN=pypi-... maturin upload dist/*
```

A version uploaded to PyPI cannot be replaced, only yanked, so run `pytest` against the built wheel first.
