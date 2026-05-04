from __future__ import annotations

import pytest

from eval.src.scoring import JudgeScoringError, _llm_judge_score


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)


class _Response:
    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]


class _Completions:
    def __init__(self, *, content: str | None = None, exc: Exception | None = None) -> None:
        self.content = content
        self.exc = exc
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return _Response(self.content or '{"correct": true, "score": 1.0, "reason": "ok"}')


class _Client:
    def __init__(self, completions: _Completions) -> None:
        self.chat = type("Chat", (), {"completions": completions})()


def test_llm_judge_does_not_send_sglang_extra_body() -> None:
    completions = _Completions()
    ok, score, reason = _llm_judge_score(
        _Client(completions),
        "openai/gpt-4o-mini",
        "question",
        "gold",
        "prediction",
        "d7_qa",
    )

    assert ok is True
    assert score == 1.0
    assert reason == "llm_judge: ok"
    assert completions.kwargs is not None
    assert "extra_body" not in completions.kwargs


def test_llm_judge_failure_is_not_token_f1_fallback() -> None:
    completions = _Completions(exc=RuntimeError("rate limited"))

    with pytest.raises(JudgeScoringError, match="judge request failed"):
        _llm_judge_score(
            _Client(completions),
            "openai/gpt-4o-mini",
            "question",
            "gold",
            "prediction",
            "d7_qa",
        )


def test_llm_judge_bad_json_is_not_token_f1_fallback() -> None:
    completions = _Completions(content="not json")

    with pytest.raises(JudgeScoringError, match="not valid JSON"):
        _llm_judge_score(
            _Client(completions),
            "openai/gpt-4o-mini",
            "question",
            "gold",
            "prediction",
            "d7_qa",
        )
