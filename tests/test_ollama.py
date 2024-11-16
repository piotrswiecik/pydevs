import logging

from pydevs.services.ollama import OllamaService
from pydevs.types.completion import OllamaTextCompletionConfig


def test_ollama_completion_with_defaults():
    service = OllamaService()
    completion = service.text_completion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        **{"model": "gemma2"},
    )
    assert len(completion) == 1
    assert completion[0]["role"] == "assistant"
    assert completion[0]["content"] is not None
    assert isinstance(completion[0]["content"], str)


def test_ollama_completion_with_temperature():
    service = OllamaService()
    completion = service.text_completion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        **{"model": "gemma2", "temperature": 0.5},
    )
    assert completion[0]["role"] == "assistant"
    assert completion[0]["content"] is not None
    assert isinstance(completion[0]["content"], str)


def test_ollama_completion_with_max_token_window():
    service = OllamaService()
    completion = service.text_completion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        **{"model": "gemma2", "ctx_size": 4096},
    )
    assert completion[0]["role"] == "assistant"
    assert completion[0]["content"] is not None
    assert isinstance(completion[0]["content"], str)
