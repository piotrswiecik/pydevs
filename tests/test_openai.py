import pytest
import logging
from dotenv import load_dotenv

from pydevs.services.openai import OpenAIService
from pydevs.types.completion import TextCompletionConfig as Config, TextCompletionPayload as Payload


load_dotenv()


def test_completion():
    service = OpenAIService()
    response = service.text_completion(
        payload=[Payload(role="system", content="Hey bro, this is a test!")],
        config=Config(model="gpt-4o-mini", max_completion_tokens=10),
    )
    logging.info(response)

