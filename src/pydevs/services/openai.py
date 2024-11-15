import os
from typing import Optional

from openai import OpenAI

from pydevs.services.base import AIServiceBase, AIServiceError
from pydevs.types.completion import TextCompletionResponse


class OpenAIService(AIServiceBase):
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)

    def text_completion(self, payload, config):
        try:
            api_response = self._client.chat.completions.create(
                model=config.model,
                messages=payload,
                max_completion_tokens=config.max_completion_tokens,
                response_format={"type": "json_object" if config.json_mode else "text"},
                stream=config.stream,
                temperature=config.temperature,
            )
            choices: list[str] = [
                choice.message.content for choice in api_response.choices
            ]
            return TextCompletionResponse(choices=choices)  # TODO: add usage
        except Exception as e:
            raise AIServiceError(f"OpenAI API error: {e}")

    def text_embedding(self, payload):
        raise NotImplementedError
