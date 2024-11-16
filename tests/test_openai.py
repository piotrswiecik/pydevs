import pytest
import logging
from dotenv import load_dotenv

from pydevs.services.openai import OpenAIService
from pydevs.types.completion import (
    EmbeddingResponse,
    TextCompletionConfig as Config,
    TextCompletionPayload as Payload,
    TextCompletionResponse,
)


load_dotenv()


def test_completion_from_object():
    service = OpenAIService()
    response = service.text_completion(
        payload=[Payload(role="system", content="Hey bro, this is a test!")],
        config=Config(model="gpt-4o-mini", max_completion_tokens=10),
    )
    assert isinstance(response, TextCompletionResponse)
    assert isinstance(response.choices, list)
    assert len(response.choices) == 1
    assert isinstance(response.choices[0], str)


def test_completion_from_dict():
    service = OpenAIService()
    response = service.text_completion(
        payload=[{"role": "system", "content": "Hey bro, this is a test!"}],
        config=Config(model="gpt-4o-mini", max_completion_tokens=10),
    )
    assert isinstance(response, TextCompletionResponse)
    assert isinstance(response.choices, list)
    assert len(response.choices) == 1
    assert isinstance(response.choices[0], str)


def test_completion_from_dict_invalid_should_raise():
    service = OpenAIService()

    with pytest.raises(Exception):
        service.text_completion(
            payload=[{"content": "whatever"}],
            config=Config(model="gpt-4o-mini", max_completion_tokens=10),
        )

    with pytest.raises(Exception):
        service.text_completion(
            payload=[{"role": "system"}],
            config=Config(model="gpt-4o-mini", max_completion_tokens=10),
        )


def test_parse_payload():
    service = OpenAIService(api_key="test")
    payload = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Hey bro, this is a test!"},
    ]
    parsed = service._parse_dict(payload)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert isinstance(parsed[0], Payload)
    assert parsed[0].role == "system"
    assert parsed[0].content == "System message"
    assert parsed[1].role == "user"
    assert parsed[1].content == "Hey bro, this is a test!"


def test_completion_with_defaults():
    service = OpenAIService()
    response = service.text_completion(
        payload=[Payload(role="system", content="Hey bro, this is a test!")],
    )
    assert isinstance(response, TextCompletionResponse)
    assert isinstance(response.choices, list)
    assert len(response.choices) == 1
    assert isinstance(response.choices[0], str)


def test_embedding():
    service = OpenAIService()
    response = service.text_embedding("test")
    assert isinstance(response, EmbeddingResponse)
    logging.info(response)