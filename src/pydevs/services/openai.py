import os
from typing import Dict, List, Literal, Optional

from openai import OpenAI

from pydevs.services.base import AIServiceBase, AIServiceError
from pydevs.types.completion import (
    EmbeddingResponse,
    TextCompletionConfig,
    TextCompletionPayload,
    TextCompletionResponse,
)


class OpenAIService(AIServiceBase):
    def __init__(self, api_key: Optional[str] = None, default_model="gpt-4o-mini"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)
        self._default_model = default_model

    def _parse_dict(self, payload: List[Dict]):
        return [TextCompletionPayload(**item) for item in payload]

    def text_completion(self, payload, config=None):
        if isinstance(payload, list) and isinstance(payload[0], dict):
            try:
                payload = self._parse_dict(payload)
            except Exception as e:
                raise AIServiceError(f"Invalid input: {e}")

        if config is None:
            config = TextCompletionConfig()  # use defaults

        try:
            api_response = self._client.chat.completions.create(
                model=config.model or self._default_model,
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

    def text_embedding(
        self,
        payload: str,
        model: Literal[
            "text-embedding-3-small", "text-embedding-3-large"
        ] = "text-embedding-3-small",
    ):
        try:
            api_response = self._client.embeddings.create(model=model, input=payload)
            embedding = api_response.data[0].embedding
            return EmbeddingResponse(embedding=embedding)  # TODO: add usage
        except (AttributeError, IndexError):
            raise AIServiceError("Invalid API response")
        except Exception as e:
            raise AIServiceError(f"OpenAI API error: {e}")
