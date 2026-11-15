"""Flag the words an LLM answer was unsure about, from its token logprobs.

Ask for logprobs, then score the response::

    from openai import OpenAI
    import llm_token_visualizer as ltv

    resp = OpenAI().chat.completions.create(
        model="gpt-4.1-mini", messages=[...], logprobs=True, top_logprobs=5)
    result = ltv.score(resp)
    print(result.summary())

:func:`score` takes an OpenAI ``ChatCompletion`` or ``Completion`` object,
the same response as a dict or JSON string (OpenAI, vLLM, OpenRouter,
Together, Ollama and other OpenAI-compatible servers), a vLLM
``RequestOutput`` / ``CompletionOutput`` from offline generation, or a Gemini
``generateContent`` response with ``responseLogprobs``. :func:`gate` is the
yes/no version for CI.

A low probability is a signal, not proof: a model can be confidently wrong,
and it can be unsure about wording while the facts are right.

The parsing, detection and HTML rendering are the ``llm-token-visualizer``
Rust crate, compiled into this package. It has no Python dependencies.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from typing import Any

from ._native import (
    DEFAULT_THRESHOLD,
    LogprobsError,
    NoLogprobsError,
    Score,
    Span,
    __version__,
)
from ._native import score_json as _score_json

__all__ = [
    "DEFAULT_THRESHOLD",
    "LogprobsError",
    "NoLogprobsError",
    "Score",
    "Span",
    "__version__",
    "gate",
    "score",
]

_HINT = (
    "request the completion with logprobs=True (and top_logprobs=5 to see the "
    "alternatives)"
)


def score(resp: Any, threshold: float = DEFAULT_THRESHOLD, *, choice: int = 0) -> Score:
    """Score one LLM answer and return its low-confidence spans.

    ``resp`` is the response of a request made with ``logprobs=True``: an
    OpenAI SDK ``ChatCompletion`` or ``Completion``, a dict or JSON string of
    the same (any OpenAI-compatible server, including vLLM), a vLLM
    ``RequestOutput`` or ``CompletionOutput``, a single choice, or its
    ``logprobs`` block.

    A word is flagged when one of its tokens has a probability strictly below
    ``threshold`` (0 to 1, default 0.5); neighbouring flagged words merge into
    one span. ``choice`` picks the choice to score when the request had
    ``n > 1``.

    Raises :class:`NoLogprobsError` when the response has no logprobs (or they
    are ``None``), :class:`LogprobsError` for an API error body, invalid JSON,
    bytes that are not UTF-8 or a malformed logprobs block (both are
    ``ValueError``), ``ValueError`` for
    a threshold outside 0 to 1 or a bad ``choice``, and ``TypeError`` for an
    input it cannot read.
    """
    payload, model = _to_json(resp, choice)
    return _score_json(payload, float(threshold), model, choice=choice)


def gate(resp: Any, threshold: float = DEFAULT_THRESHOLD, *, choice: int = 0) -> bool:
    """``True`` when no word in the answer is below ``threshold``.

    For CI and eval scripts: ``assert gate(resp, 0.4), score(resp, 0.4).summary()``.
    Takes the same inputs and raises the same errors as :func:`score`.
    """
    return not score(resp, threshold, choice=choice).flagged


def _to_json(resp: Any, choice: int) -> tuple[str, str | None]:
    """The response as JSON text for the parser, plus its model name."""
    if isinstance(choice, bool) or not isinstance(choice, int) or choice < 0:
        raise ValueError(f"choice must be a non-negative integer, got {choice!r}")
    if isinstance(resp, (bytes, bytearray, memoryview)):
        try:
            resp = bytes(resp).decode("utf-8")
        except UnicodeDecodeError as e:
            raise LogprobsError(f"input is not valid UTF-8: {e}") from None
    if isinstance(resp, str):
        if choice == 0:
            return resp, None
        try:
            resp = json.loads(resp)
        except ValueError as e:
            raise LogprobsError(f"input is not valid JSON: {e}") from None

    obj, choice = _plain(resp, choice)
    model = None
    if isinstance(obj, dict):
        m = obj.get("model")
        model = m if isinstance(m, str) else None
        choices = obj.get("choices")
        if isinstance(choices, list):
            if choice >= len(choices):
                raise ValueError(
                    f"choice={choice} but the response has {len(choices)} choice(s)"
                )
            if choice:
                obj = {**obj, "choices": [choices[choice]]}
        elif choice:
            raise ValueError("choice only applies to a response with a choices list")
        elif "logprobs" in obj and "token_logprobs" not in obj:
            # A single choice (resp.choices[0]) rather than the whole response.
            if obj["logprobs"] is None:
                raise NoLogprobsError(f"token logprobs are None in this choice; {_HINT}")
            obj = obj["logprobs"]
    elif choice:
        raise ValueError("choice only applies to a response with a choices list")
    return _dumps(obj), model


_NOT_A_MODEL = object()


def _dump_model(x: Any) -> Any:
    """An SDK object as plain data, or ``_NOT_A_MODEL``.

    Pydantic models: the OpenAI SDK's ChatCompletion, Completion, Choice,
    ChoiceLogprobs, ChatCompletionTokenLogprob, and other SDKs built the same
    way; anything else with a ``to_dict()``.
    """
    dump = getattr(x, "model_dump", None)
    if callable(dump):
        try:
            return dump(mode="json")
        except TypeError:  # pydantic v1-style signature
            return dump()
    to_dict = getattr(x, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    return _NOT_A_MODEL


def _plain_tree(x: Any) -> Any:
    """``x`` with every SDK object inside lists and mappings dumped to dicts.

    Values it cannot convert are left as they are; :func:`_dumps` reports them.
    """
    if x is None or isinstance(x, (str, int, float)):
        return x
    if isinstance(x, (list, tuple)):
        return [_plain_tree(v) for v in x]
    if isinstance(x, Mapping):
        return {k: _plain_tree(v) for k, v in x.items()}
    dumped = _dump_model(x)
    return x if dumped is _NOT_A_MODEL else _plain_tree(dumped)


def _plain(resp: Any, choice: int) -> tuple[Any, int]:
    """Turn an SDK object into plain dicts and lists.

    Returns the object and the choice index still to apply (0 once a vLLM
    output has been picked here).
    """
    # A token list (resp.choices[0].logprobs.content) holds SDK objects too.
    if isinstance(resp, (list, tuple, Mapping)):
        return _plain_tree(resp), choice
    dumped = _dump_model(resp)
    if dumped is not _NOT_A_MODEL:
        return _plain_tree(dumped), choice
    # vLLM offline generation: RequestOutput.outputs[i] is a CompletionOutput.
    outputs = getattr(resp, "outputs", None)
    if isinstance(outputs, (list, tuple)) and not hasattr(resp, "token_ids"):
        if choice >= len(outputs):
            raise ValueError(f"choice={choice} but the output has {len(outputs)} output(s)")
        return _from_vllm(outputs[choice]), 0
    if hasattr(resp, "token_ids") and hasattr(resp, "logprobs"):
        if choice:
            raise ValueError("choice only applies to a RequestOutput; pass the CompletionOutput you want")
        return _from_vllm(resp), 0
    raise TypeError(
        "score() takes an OpenAI ChatCompletion or Completion, a dict or JSON "
        "string of an OpenAI-compatible response, or a vLLM RequestOutput / "
        f"CompletionOutput; got {type(resp).__name__}"
    )


def _from_vllm(out: Any) -> list[dict[str, Any]]:
    """A vLLM CompletionOutput as a bare token list.

    ``out.logprobs`` has one entry per generated token: a mapping from token
    id to a ``Logprob`` (``logprob``, ``rank``, ``decoded_token``) that holds
    the sampled token and the top alternatives.
    """
    positions = getattr(out, "logprobs", None)
    if positions is None:
        raise NoLogprobsError(
            "this vLLM output has no logprobs; generate with SamplingParams(logprobs=5)"
        )
    ids = list(getattr(out, "token_ids", None) or [])
    tokens: list[dict[str, Any]] = []
    for i, pos in enumerate(positions):
        if not pos:
            continue
        if i >= len(ids) or ids[i] not in pos:
            raise LogprobsError(
                f"vLLM logprobs position {i} does not include the sampled token id"
            )
        alts = []
        for tid, lp in pos.items():
            text = getattr(lp, "decoded_token", None)
            value = getattr(lp, "logprob", lp)
            if text is None:
                raise LogprobsError(
                    "vLLM logprobs have no decoded tokens; keep SamplingParams(detokenize=True)"
                )
            alts.append((tid, str(text), float(value)))
        chosen = next(a for a in alts if a[0] == ids[i])
        alts.sort(key=lambda a: a[2], reverse=True)
        tokens.append(
            {
                "token": chosen[1],
                "logprob": chosen[2],
                "top_logprobs": [{"token": t, "logprob": v} for _, t, v in alts],
            }
        )
    return tokens


def _dumps(obj: Any) -> str:
    try:
        try:
            return json.dumps(obj, ensure_ascii=False, allow_nan=False)
        except ValueError:
            # -inf logprobs (vLLM can return them) are not valid JSON.
            return json.dumps(_finite(obj), ensure_ascii=False, allow_nan=False)
    except TypeError as e:
        raise TypeError(f"the response holds a value that is not JSON: {e}") from None


def _finite(x: Any) -> Any:
    if isinstance(x, float):
        if math.isnan(x):
            return None
        if math.isinf(x):
            return math.copysign(sys.float_info.max, x)
        return x
    if isinstance(x, Mapping):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return x
