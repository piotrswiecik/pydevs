import os
from typing import Dict, List, Literal, Optional

from openai import OpenAI

from pydevs.services.base import AIServiceBase, AIServiceError
from pydevs.types.completion import OpenAIMessage


class OpenAIService(AIServiceBase):
    def __init__(self, api_key: Optional[str] = None, default_model="gpt-4o-mini"):
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key)
        self._default_model = default_model

    def text_completion(
        self,
        messages: list,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        stream: bool = False,
    ) -> List[OpenAIMessage]:
        if model is None and self._default_model is None:
            raise ValueError(
                "Model must be provided as kwarg or during client initialization"
            )

        try:
            api_response = self._client.chat.completions.create(
                model=model or self._default_model,
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object" if json_mode else "text"},
                stream=stream,
                temperature=temperature,
            )
            return [
                {"role": choice.message.role, "content": choice.message.content}
                for choice in api_response.choices
            ]
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
