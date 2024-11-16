from dotenv import load_dotenv

from pydevs.services.openai import OpenAIService

load_dotenv()


def test_openai_simple_completion():
    service = OpenAIService()
    response = service.text_completion(
        messages=[{"role": "user", "content": "Hey bro, how are you?"}],
        **{"model": "gpt-4o-mini"},
    )
    assert len(response) == 1
    assert isinstance(response[0], dict)
    assert "role" in response[0]
    assert "content" in response[0]
    assert response[0]["role"] == "assistant"
    assert isinstance(response[0]["content"], str)


def test_openai_completion_with_temperature():
    service = OpenAIService()
    response = service.text_completion(
        messages=[{"role": "user", "content": "Hey bro, how are you?"}],
        **{"model": "gpt-4o-mini", "temperature": 0.5},
    )
    assert len(response) == 1
    assert isinstance(response[0], dict)
    assert "role" in response[0]
    assert "content" in response[0]
    assert response[0]["role"] == "assistant"
    assert isinstance(response[0]["content"], str)
