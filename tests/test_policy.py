"""Task 2: multi-provider LLMPolicy resolution + batch-robustness (cache, retry).
These never hit a real API — _chat is monkeypatched."""

import numpy as np

from src.policy import LLMPolicy


def _cfg(provider="deepseek", **extra):
    pol = {
        "backend": "llm",
        "provider": provider,
        "temperature": 0.7,
        "providers": {
            "openai": {"api": "openai", "model": "gpt-4o-mini", "base_url": None, "api_key_env": "OPENAI_API_KEY"},
            "qwen": {"api": "openai", "model": "qwen2.5-7b-instruct",
                     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "api_key_env": "DASHSCOPE_API_KEY"},
            "deepseek": {"api": "openai", "model": "deepseek-chat",
                         "base_url": "https://api.deepseek.com/v1", "api_key_env": "DEEPSEEK_API_KEY"},
        },
    }
    pol.update(extra)
    return {"policy": pol}


def test_provider_resolution():
    p = LLMPolicy(_cfg("deepseek"))
    assert p.provider == "deepseek"
    assert p.model == "deepseek-chat"
    assert p.base_url.endswith("api.deepseek.com/v1")
    assert p.api_key_env == "DEEPSEEK_API_KEY"

    q = LLMPolicy(_cfg("qwen"))
    assert q.model == "qwen2.5-7b-instruct"
    assert "dashscope" in q.base_url


def test_backcompat_flat_fields():
    # no providers dict -> fall back to flat fields
    p = LLMPolicy({"policy": {"backend": "llm", "model": "m1", "base_url": "http://x/v1", "api_key_env": "K"}})
    assert p.model == "m1" and p.base_url == "http://x/v1" and p.api_key_env == "K"


def test_cache_avoids_second_call(tmp_path, monkeypatch):
    p = LLMPolicy(_cfg("deepseek", cache=True, cache_dir=str(tmp_path)))
    calls = {"n": 0}

    def fake_chat(system, user):
        calls["n"] += 1
        return '{"0": 5.0, "1": 1.0}'

    monkeypatch.setattr(p, "_chat", fake_chat)
    a = p._safe_chat("sys", "usr")
    b = p._safe_chat("sys", "usr")
    assert a == b
    assert calls["n"] == 1  # second served from cache


def test_retry_then_succeed(monkeypatch):
    p = LLMPolicy(_cfg("deepseek"))
    p.max_retries = 5
    monkeypatch.setattr(p, "_sleep_backoff", lambda attempt: None)  # no real sleeping
    seq = {"i": 0}

    def flaky(system, user):
        seq["i"] += 1
        if seq["i"] < 3:
            raise RuntimeError("rate limited")
        return "ok"

    monkeypatch.setattr(p, "_chat", flaky)
    out = p._safe_chat("s", "u")
    assert out == "ok"
    assert seq["i"] == 3


def test_retry_exhausts_raises(monkeypatch):
    import pytest

    p = LLMPolicy(_cfg("deepseek"))
    p.max_retries = 3
    monkeypatch.setattr(p, "_sleep_backoff", lambda attempt: None)
    monkeypatch.setattr(p, "_chat", lambda s, u: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):  # persistent transient failure must surface, not degrade
        p._safe_chat("s", "u")


def test_fatal_error_fails_fast(monkeypatch):
    import pytest

    p = LLMPolicy(_cfg("deepseek"))
    p.max_retries = 5
    calls = {"n": 0}

    def missing_pkg(s, u):
        calls["n"] += 1
        raise ImportError("No module named 'openai'")

    monkeypatch.setattr(p, "_chat", missing_pkg)
    monkeypatch.setattr(p, "_sleep_backoff", lambda attempt: None)
    with pytest.raises(RuntimeError):
        p._safe_chat("s", "u")
    assert calls["n"] == 1  # fatal => no retries


def test_parse_scores_robust():
    # prose around the JSON, plus an out-of-range index, must not crash
    s = LLMPolicy._parse_scores('here: {"0": 8, "1": 2, "9": 3} done', 2)
    assert s[0] == 8.0 and s[1] == 2.0 and len(s) == 2
    assert np.allclose(LLMPolicy._parse_scores("garbage", 3), 0.0)  # fallback uniform
