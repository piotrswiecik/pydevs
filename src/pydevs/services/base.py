from abc import ABC, abstractmethod
from typing import List

from pydevs.types.completion import (
    TextCompletionConfig,
    TextCompletionPayload,
    TextCompletionResponse,
)


class AIServiceBase(ABC):
    @abstractmethod
    def text_completion(
        self, payload: List[TextCompletionPayload], config: TextCompletionConfig
    ) -> TextCompletionResponse:  # TODO: add support for streaming chunks
        pass

    @abstractmethod
    def text_embedding(self, payload) -> List[float]:
        pass


class AIServiceError(Exception):
    pass
