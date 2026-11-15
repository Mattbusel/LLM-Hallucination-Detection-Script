"""Tests for the Python package, against hand-written responses in the
OpenAI, Completions and vLLM shapes (python/tests/fixtures)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

import llm_token_visualizer as ltv

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def raw(name: str) -> str:
    return (FIXTURES / f"{name}.json").read_text(encoding="utf-8")


# --- OpenAI Chat Completions -------------------------------------------------


def test_chat_completion_dict_spans():
    r = ltv.score(load("chat_tour"))
    assert r.text == "The 1998 Tour de France was won by Marco Pantani."
    assert r.model == "gpt-4.1-mini-2025-04-14"
    assert r.flagged
    assert r.token_count == 14
    assert r.threshold == 0.5
    (span,) = r.spans()
    assert (span.start, span.end) == (34, 40)
    assert span.text == " Marco"
    assert r.text[span.start : span.end] == span.text
    assert (span.token_start, span.token_end) == (10, 11)
    assert span.min_logprob == pytest.approx(-0.89)
    assert span.mean_logprob == pytest.approx(-0.89)
    assert span.min_prob == pytest.approx(math.exp(-0.89))
    assert span.weakest_token == " Marco"
    assert [t for t, _ in span.alternatives] == [" Marco", " Jan", " Bjarne", " Marc"]
    assert span.alternatives[1][1] == pytest.approx(math.exp(-0.97))
    assert span.entropy_bits is not None and 1.0 < span.entropy_bits < 2.0
    assert span.describe().startswith('p=0.41 at " Marco"; model also considered " Jan" (0.38)')


def test_summary_text():
    r = ltv.score(load("chat_tour"))
    assert r.summary() == (
        "1 low-confidence span below p=0.50 in 14 tokens (mean p=0.96, perplexity 1.07):\n"
        '  [34:40] " Marco" p=0.41 at " Marco"; model also considered " Jan" (0.38), '
        '" Bjarne" (0.12), " Marc" (0.01)'
    )
    clean = ltv.score(load("chat_tour"), 0.3)
    assert clean.summary() == (
        "No low-confidence spans below p=0.30 in 14 tokens (mean p=0.96, perplexity 1.07)."
    )


def test_json_string_and_bytes_give_the_same_result():
    a = ltv.score(load("chat_tour"))
    b = ltv.score(raw("chat_tour"))
    c = ltv.score(raw("chat_tour").encode("utf-8"))
    assert a.to_json() == b.to_json() == c.to_json()
    assert b.model == "gpt-4.1-mini-2025-04-14"


def test_char_offsets_with_non_ascii_text():
    r = ltv.score(load("chat_unicode"))
    assert r.text == "Die Brücke wurde 1932 eröffnet."
    spans = r.spans()
    assert [(s.start, s.end, s.text) for s in spans] == [(3, 10, " Brücke"), (17, 21, "1932")]
    for s in spans:
        assert r.text[s.start : s.end] == s.text
    # Token offsets differ from character offsets.
    assert [(s.token_start, s.token_end) for s in spans] == [(1, 3), (5, 7)]
    assert spans[1].min_logprob == pytest.approx(-1.05)
    assert spans[1].mean_logprob == pytest.approx((-1.05 - 0.2) / 2)


def test_threshold_and_gate():
    resp = load("chat_tour")
    assert ltv.gate(resp, 0.3) is True
    assert ltv.gate(resp, 0.5) is False
    assert ltv.gate(resp) is False  # default threshold 0.5
    assert ltv.DEFAULT_THRESHOLD == 0.5
    assert ltv.score(resp, 0.0).spans() == []


@pytest.mark.parametrize("bad", [1.5, -0.1, float("nan")])
def test_threshold_out_of_range_is_value_error(bad):
    with pytest.raises(ValueError, match="threshold must be a probability"):
        ltv.score(load("chat_tour"), bad)


def test_choice_picks_among_n_choices():
    resp = load("chat_n2")
    assert ltv.score(resp).text == "Canberra."
    assert not ltv.score(resp).flagged
    second = ltv.score(resp, choice=1)
    assert second.text == "Sydney."
    assert [s.text for s in second.spans()] == ["Sydney"]
    assert ltv.gate(resp) and not ltv.gate(resp, choice=1)
    assert ltv.score(raw("chat_n2"), choice=1).text == "Sydney."
    with pytest.raises(ValueError, match="2 choice"):
        ltv.score(resp, choice=2)
    with pytest.raises(ValueError, match="non-negative"):
        ltv.score(resp, choice=-1)


# --- Missing logprobs and other errors --------------------------------------


def test_none_logprobs_is_a_clear_error():
    with pytest.raises(ltv.NoLogprobsError, match=r"logprobs=True") as e:
        ltv.score(load("chat_no_logprobs"))
    assert "None" in str(e.value)
    assert isinstance(e.value, ltv.LogprobsError)
    assert isinstance(e.value, ValueError)


def test_missing_logprobs_key_is_a_clear_error():
    resp = load("chat_tour")
    del resp["choices"][0]["logprobs"]
    with pytest.raises(ltv.NoLogprobsError, match=r"logprobs=True"):
        ltv.score(resp)
    with pytest.raises(ltv.NoLogprobsError):
        ltv.gate(resp)


def test_single_choice_with_none_logprobs():
    choice = load("chat_no_logprobs")["choices"][0]
    with pytest.raises(ltv.NoLogprobsError, match=r"logprobs=True"):
        ltv.score(choice)


def test_none_logprobs_in_a_later_choice_names_that_choice():
    resp = load("chat_n2")
    resp["choices"][1]["logprobs"] = None
    assert ltv.score(resp).text == "Canberra."
    with pytest.raises(ltv.NoLogprobsError, match=r"choices\[1\]\.logprobs is null"):
        ltv.score(resp, choice=1)
    with pytest.raises(ltv.NoLogprobsError, match=r"choices\[1\]\.logprobs is null"):
        ltv.score(json.dumps(resp), choice=1)
    del resp["choices"][1]["logprobs"]
    with pytest.raises(ltv.NoLogprobsError, match=r"choices\[1\]\.logprobs is missing"):
        ltv.score(resp, choice=1)
    # The first choice keeps its index in the message.
    resp = load("chat_no_logprobs")
    with pytest.raises(ltv.NoLogprobsError, match=r"choices\[0\]\.logprobs is null"):
        ltv.score(resp)


def test_api_error_body():
    with pytest.raises(ltv.LogprobsError, match="the API returned an error") as e:
        ltv.score(load("api_error"))
    assert not isinstance(e.value, ltv.NoLogprobsError)


def test_invalid_json():
    with pytest.raises(ltv.LogprobsError, match="not valid JSON"):
        ltv.score("{not json")
    with pytest.raises(ltv.LogprobsError, match="not valid JSON"):
        ltv.score("{not json", choice=1)


def test_bytes_that_are_not_utf8():
    with pytest.raises(ltv.LogprobsError, match="not valid UTF-8") as e:
        ltv.score(b'{"choices": [' + bytes([0xFF]) + b"]}")
    assert not isinstance(e.value, ltv.NoLogprobsError)
    with pytest.raises(ltv.LogprobsError, match="not valid UTF-8"):
        ltv.score(bytearray([0xC3, 0x28]), choice=1)


def test_unreadable_input_is_type_error():
    with pytest.raises(TypeError, match="got int"):
        ltv.score(42)


# --- Other shapes ------------------------------------------------------------


def test_choice_and_logprobs_block_alone():
    resp = load("chat_tour")
    expected = ltv.score(resp).to_json()
    assert ltv.score(resp["choices"][0]).to_json() == expected
    assert ltv.score(resp["choices"][0]["logprobs"]).to_json() == expected
    assert ltv.score(resp["choices"][0]["logprobs"]["content"]).to_json() == expected


def test_completions_shape_with_echo_null():
    r = ltv.score(load("completion_echo"))
    assert r.text == "Paris is the capital of France"
    assert r.token_count == 6
    # The null logprob of the echoed first token is never flagged.
    assert [s.text for s in r.spans()] == [" is"]
    (span,) = r.spans()
    assert (span.start, span.end) == (5, 8)
    assert span.alternatives[0][0] == " was"
    assert math.isfinite(r.perplexity)


def test_vllm_server_response():
    r = ltv.score(load("vllm_chat"))
    assert r.model == "Qwen/Qwen2.5-7B-Instruct"
    assert r.text == "Water boils at 90 degrees on Everest."
    assert [s.text for s in r.spans()] == ["90", " on"]
    first = r.spans()[0]
    assert first.alternatives[0][0] == "1"
    assert first.alternatives[-1] == ("2", 0.0)  # vLLM's -9999.0


@dataclass
class Logprob:
    """The shape of vllm.logprobs.Logprob."""

    logprob: float
    rank: int | None
    decoded_token: str | None


def vllm_output(logprobs=..., token_ids=(11, 22, 33)):
    if logprobs is ...:
        logprobs = [
            {11: Logprob(-0.01, 1, "The"), 12: Logprob(-5.0, 2, "A")},
            {22: Logprob(-1.2, 2, " moon"), 23: Logprob(-0.6, 1, " sun"), 24: Logprob(float("-inf"), 3, " sky")},
            {33: Logprob(-0.02, 1, ".")},
        ]
    return SimpleNamespace(index=0, text="The moon.", token_ids=list(token_ids), logprobs=logprobs, cumulative_logprob=None, finish_reason="stop")


def test_vllm_offline_completion_output():
    r = ltv.score(vllm_output())
    assert r.text == "The moon."
    (span,) = r.spans()
    assert span.text == " moon"
    assert (span.start, span.end) == (3, 8)
    assert span.min_logprob == pytest.approx(-1.2)
    assert [t for t, _ in span.alternatives] == [" sun", " moon", " sky"]
    assert span.alternatives[-1][1] == 0.0  # -inf logprob


def test_vllm_offline_request_output_and_choice():
    confident = vllm_output([{11: Logprob(-0.01, 1, "The")}, {22: Logprob(-0.02, 1, " sun")}, {33: Logprob(0.0, 1, ".")}])
    req = SimpleNamespace(request_id="0", prompt="Q", outputs=[confident, vllm_output()], finished=True)
    assert ltv.gate(req)
    assert not ltv.gate(req, choice=1)
    with pytest.raises(ValueError, match="2 output"):
        ltv.score(req, choice=2)


def test_vllm_offline_without_logprobs():
    with pytest.raises(ltv.NoLogprobsError, match=r"SamplingParams\(logprobs=5\)"):
        ltv.score(vllm_output(logprobs=None))


def test_vllm_offline_without_decoded_tokens():
    out = vllm_output([{11: Logprob(-0.01, 1, None)}], token_ids=[11])
    with pytest.raises(ltv.LogprobsError, match="detokenize"):
        ltv.score(out)


# --- Reports -------------------------------------------------------------------


def test_to_html_is_a_self_contained_page():
    r = ltv.score(load("chat_tour"))
    page = r.to_html()
    assert page.startswith("<!DOCTYPE html>")
    assert "gpt-4.1-mini-2025-04-14" in page
    assert "Pant" in page and "Jan" in page
    assert "<script" not in page
    assert "http://" not in page.split("<footer")[0]
    assert "<title>Token confidence: My eval</title>" in r.to_html(title="My eval")


def test_html_escapes_token_text():
    tokens = [
        {"token": "Use", "logprob": -0.01},
        {"token": " <script>alert(1)</script>", "logprob": -2.0},
    ]
    page = ltv.score(tokens).to_html()
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_to_markdown_and_to_json():
    r = ltv.score(load("chat_tour"))
    md = r.to_markdown()
    assert "Marco" in md
    report = json.loads(r.to_json())
    assert report["threshold"] == 0.5
    assert report["spans"][0]["text"] == " Marco"
    assert report["spans"][0]["start"] == 10  # token index in the JSON report


def test_span_to_dict_is_json_serializable():
    (span,) = ltv.score(load("chat_tour")).spans()
    d = span.to_dict()
    assert d["start"] == 34 and d["end"] == 40 and d["text"] == " Marco"
    json.dumps(d)


def test_spans_compare_by_value():
    a = ltv.score(load("chat_tour")).spans()
    b = ltv.score(raw("chat_tour")).spans()
    assert a == b
    assert a[0] == b[0] and not (a[0] != b[0])
    other = ltv.score(load("chat_unicode")).spans()
    assert a[0] != other[0]
    assert a[0] != a[0].to_dict()
    # Equal for a later choice too, not only the first.
    assert ltv.score(load("chat_n2"), choice=1).spans() == ltv.score(
        load("chat_n2"), choice=1
    ).spans()
    with pytest.raises(TypeError):
        hash(a[0])


def test_reprs():
    r = ltv.score(load("chat_tour"))
    assert repr(r).startswith("Score(spans=1, threshold=0.50, tokens=14")
    assert repr(r.spans()[0]).startswith("Span(start=34, end=40, text=\" Marco\"")


# --- OpenAI SDK objects (only when the openai package is installed) --------


def test_openai_sdk_chat_completion():
    openai_types = pytest.importorskip("openai.types.chat")
    obj = openai_types.ChatCompletion.model_validate(load("chat_tour"))
    r = ltv.score(obj)
    assert r.to_json() == ltv.score(load("chat_tour")).to_json()
    assert r.model == "gpt-4.1-mini-2025-04-14"
    assert ltv.score(obj.choices[0]).text == r.text
    assert ltv.score(obj.choices[0].logprobs).text == r.text
    assert not ltv.gate(obj, 0.5) and ltv.gate(obj, 0.3)


def test_openai_sdk_token_list():
    # resp.choices[0].logprobs.content is a list of ChatCompletionTokenLogprob.
    openai_types = pytest.importorskip("openai.types.chat")
    obj = openai_types.ChatCompletion.model_validate(load("chat_tour"))
    content = obj.choices[0].logprobs.content
    assert type(content[0]).__name__ == "ChatCompletionTokenLogprob"
    expected = ltv.score(load("chat_tour")).to_json()
    assert ltv.score(content).to_json() == expected
    assert ltv.score(tuple(content)).to_json() == expected
    assert ltv.score({"content": content}).to_json() == expected
    (span,) = ltv.score(content).spans()
    assert (span.start, span.end, span.text) == (34, 40, " Marco")
    assert [t for t, _ in span.alternatives] == [" Marco", " Jan", " Bjarne", " Marc"]


def test_openai_sdk_objects_inside_a_dict():
    openai_types = pytest.importorskip("openai.types.chat")
    obj = openai_types.ChatCompletion.model_validate(load("chat_n2"))
    mixed = {"model": obj.model, "choices": obj.choices}
    assert ltv.score(mixed).text == "Canberra."
    second = ltv.score(mixed, choice=1)
    assert second.text == "Sydney."
    assert second.model == obj.model
    assert second.to_json() == ltv.score(load("chat_n2"), choice=1).to_json()


def test_openai_sdk_chat_completion_without_logprobs():
    openai_types = pytest.importorskip("openai.types.chat")
    obj = openai_types.ChatCompletion.model_validate(load("chat_no_logprobs"))
    assert obj.choices[0].logprobs is None
    with pytest.raises(ltv.NoLogprobsError, match=r"logprobs=True"):
        ltv.score(obj)
    with pytest.raises(ltv.NoLogprobsError, match=r"logprobs=True"):
        ltv.score(obj.choices[0])


def test_openai_sdk_legacy_completion():
    openai_types = pytest.importorskip("openai.types")
    data = load("completion_echo")
    # The SDK types token_logprobs as floats, so validate a response without
    # the echoed null first token.
    lp = data["choices"][0]["logprobs"]
    lp["token_logprobs"][0] = -0.03
    lp["top_logprobs"][0] = {"Paris": -0.03}
    obj = openai_types.Completion.model_validate(data)
    r = ltv.score(obj)
    assert r.text == "Paris is the capital of France"
    assert [s.text for s in r.spans()] == [" is"]


# --- Packaging -----------------------------------------------------------------


def test_version_and_no_runtime_dependencies():
    assert ltv.__version__ == metadata.version("llm-token-visualizer")
    requires = metadata.requires("llm-token-visualizer") or []
    assert [r for r in requires if "extra ==" not in r] == []
