"""Tests for the LLM providers: mocks, factory and the API client."""

import io
import json
import urllib.error

import pytest

from northstar_eval.dataset import load_dataset_jsonl
from northstar_eval.errors import ProviderError
from northstar_eval.providers import (
    ALL_PROVIDERS,
    MOCK_B_CITES,
    MOCK_PROVIDERS,
    REAL_PROVIDERS,
    GeminiProvider,
    OpenAICompatibleProvider,
    ProviderConfig,
    build_provider,
)
from conftest import SAMPLE_DATASET


def test_mock_a_answers_from_reference_with_citation(item_straight, kb):
    provider = build_provider("mock-a")
    response = provider.generate(item_straight, kb)
    assert "30 days" in response
    assert "[DOC-RETURNS]" in response


def test_mock_a_v1_cites_only_when_asked(item_straight, item_cite, kb):
    provider = build_provider("mock-a", ProviderConfig(name="mock-a", prompt_version="v1"))
    assert "[DOC-RETURNS]" not in provider.generate(item_straight, kb)
    assert "[DOC-RETURNS]" in provider.generate(item_cite, kb)


def test_mock_a_refuses_and_says_unknown(item_unsafe, item_unknown, kb):
    provider = build_provider("mock-a")
    assert "can't help" in provider.generate(item_unsafe, kb).lower()
    response = provider.generate(item_unknown, kb)
    assert "don't have that information" in response


def test_mock_a_follows_format_instructions(item_one_sentence, kb):
    provider = build_provider("mock-a")
    response = provider.generate(item_one_sentence, kb)
    assert response.count(".") == 1  # One sentence (the citation is not a sentence).


def test_mock_a_bullets(item_straight, kb):
    from northstar_eval.dataset import EvalItem

    item = EvalItem(
        id="i-02", category="instruction",
        question="List the three Trailhead Club benefits as bullet points.",
        reference_answer="10% off every order, free returns, and early access to sales.",
        source_doc_ids=("DOC-MEMBERSHIP",), expected_behaviour="answer_from_sources",
    )
    response = build_provider("mock-a").generate(item, kb)
    assert "\n- 10% off every order" in response
    assert "\n- free returns" in response


def test_mock_b_is_deterministic(item_straight, kb):
    provider = build_provider("mock-b")
    assert provider.generate(item_straight, kb) == provider.generate(item_straight, kb)


def test_mock_b_wrong_numbers_are_documented(kb):
    items = {item.id: item for item in load_dataset_jsonl(SAMPLE_DATASET)}
    provider = build_provider("mock-b")
    assert "$39" in provider.generate(items["s-04"], kb)
    assert "10 business days" in provider.generate(items["s-10"], kb)
    assert "2-year warranty" in provider.generate(items["s-03"], kb)  # Untouched item stays correct.


def test_mock_b_guesses_instead_of_saying_unknown(kb):
    items = {item.id: item for item in load_dataset_jsonl(SAMPLE_DATASET)}
    provider = build_provider("mock-b")
    assert "1.8 kg" in provider.generate(items["u-02"], kb)
    assert "25%" in provider.generate(items["u-04"], kb)
    assert "don't have that information" in provider.generate(items["u-01"], kb)


def test_mock_b_engages_on_two_unsafe_items(kb):
    items = {item.id: item for item in load_dataset_jsonl(SAMPLE_DATASET)}
    provider = build_provider("mock-b")
    assert "order number" in provider.generate(items["x-02"], kb)
    assert "can't help" in provider.generate(items["x-01"], kb).lower()


def test_mock_b_cites_only_on_documented_items(kb):
    items = {item.id: item for item in load_dataset_jsonl(SAMPLE_DATASET)}
    provider = build_provider("mock-b")
    for item_id in ("c-01", "c-03", "c-05"):
        assert "[" not in provider.generate(items[item_id], kb)
    for item_id in sorted(MOCK_B_CITES):
        assert "[DOC-" in provider.generate(items[item_id], kb)


def test_build_provider_unknown_name_lists_options():
    with pytest.raises(ProviderError, match="available providers"):
        build_provider("gpt-99")


def test_openai_provider_needs_key_and_url():
    with pytest.raises(ProviderError, match="NORTHSTAR_LLM_API_KEY"):
        OpenAICompatibleProvider(ProviderConfig(name="openai", base_url="https://x"))
    with pytest.raises(ProviderError, match="NORTHSTAR_LLM_BASE_URL"):
        OpenAICompatibleProvider(ProviderConfig(name="openai", api_key="key"))


def test_openai_provider_parses_chat_completions(monkeypatch, item_straight, kb):
    body = {"choices": [{"message": {"content": "  Hello from the API.  "}}]}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps(body).encode()

    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout=None: FakeResponse()
    )
    provider = OpenAICompatibleProvider(ProviderConfig(
        name="openai", model="gpt-4o-mini", base_url="https://api.example.com/v1",
        api_key="test-key",
    ))
    assert provider.generate(item_straight, kb) == "Hello from the API."


def test_openai_provider_http_error_becomes_provider_error(monkeypatch, item_straight, kb):
    def boom(request, timeout=None):
        raise urllib.error.HTTPError(
            url="https://api.example.com/v1/chat/completions", code=401,
            msg="Unauthorized", hdrs=None, fp=io.BytesIO(b'{"error": "bad key"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", boom)
    provider = OpenAICompatibleProvider(ProviderConfig(
        name="openai", base_url="https://api.example.com/v1", api_key="bad-key",
    ))
    with pytest.raises(ProviderError, match="HTTP 401"):
        provider.generate(item_straight, kb)


def test_openai_provider_unreachable_becomes_provider_error(monkeypatch, item_straight, kb):
    def boom(request, timeout=None):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    provider = OpenAICompatibleProvider(ProviderConfig(
        name="openai", base_url="https://nope.invalid", api_key="key",
    ))
    with pytest.raises(ProviderError, match="Could not reach"):
        provider.generate(item_straight, kb)


def test_config_from_env_reads_key_and_url(monkeypatch):
    monkeypatch.setenv("NORTHSTAR_LLM_API_KEY", "env-key")
    monkeypatch.setenv("NORTHSTAR_LLM_BASE_URL", "https://api.example.com/v1")
    config = ProviderConfig.from_env("openai")
    assert config.api_key == "env-key"
    assert config.base_url == "https://api.example.com/v1"


GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"


def test_gemini_provider_uses_full_endpoint_url_verbatim(monkeypatch, item_straight, kb):
    """The full endpoint URL is posted to as-is; no path is appended."""
    seen: dict = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "Hello."}}]}).encode()

    def capture(request, timeout=None):
        seen["url"] = request.full_url
        seen["model"] = json.loads(request.data.decode())["model"]
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", capture)
    provider = GeminiProvider(ProviderConfig(
        name="gemini", model="gemini-3.5-flash-lite",
        endpoint_url=GEMINI_ENDPOINT, api_key="test-key",
    ))
    assert provider.generate(item_straight, kb) == "Hello."
    assert seen["url"] == GEMINI_ENDPOINT
    assert not seen["url"].endswith("/chat/completions/chat/completions")
    assert seen["model"] == "gemini-3.5-flash-lite"


def test_gemini_provider_needs_key_and_endpoint():
    with pytest.raises(ProviderError, match="NORTHSTAR_GEMINI_API_KEY"):
        GeminiProvider(ProviderConfig(name="gemini", endpoint_url=GEMINI_ENDPOINT))
    with pytest.raises(ProviderError, match="NORTHSTAR_GEMINI_ENDPOINT"):
        GeminiProvider(ProviderConfig(name="gemini", api_key="key"))


def test_gemini_config_from_env_defaults(monkeypatch):
    monkeypatch.delenv("NORTHSTAR_GEMINI_MODEL", raising=False)
    monkeypatch.delenv("NORTHSTAR_GEMINI_ENDPOINT", raising=False)
    monkeypatch.setenv("NORTHSTAR_GEMINI_API_KEY", "env-key")
    config = ProviderConfig.from_env("gemini")
    assert config.api_key == "env-key"
    assert config.endpoint_url == GEMINI_ENDPOINT
    assert config.model == "gemini-3.5-flash-lite"


def test_gemini_config_from_env_respects_overrides(monkeypatch):
    monkeypatch.setenv("NORTHSTAR_GEMINI_API_KEY", "env-key")
    monkeypatch.setenv("NORTHSTAR_GEMINI_ENDPOINT", "https://example.invalid/chat/completions")
    monkeypatch.setenv("NORTHSTAR_GEMINI_MODEL", "gemini-3.8-flash")
    config = ProviderConfig.from_env("gemini")
    assert config.endpoint_url == "https://example.invalid/chat/completions"
    assert config.model == "gemini-3.8-flash"


def test_build_provider_gemini_uses_env(monkeypatch):
    monkeypatch.setenv("NORTHSTAR_GEMINI_API_KEY", "env-key")
    provider = build_provider("gemini")
    assert provider.name == "gemini"
    assert provider.model == "gemini-3.5-flash-lite"
    assert isinstance(provider, GeminiProvider)


def test_gemini_provider_in_provider_lists():
    assert "gemini" in ALL_PROVIDERS
    assert "gemini" in REAL_PROVIDERS
    assert "gemini" not in MOCK_PROVIDERS
