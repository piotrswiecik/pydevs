from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from pydevs.types.completion import EmbeddingResponse


class AIServiceBase(ABC):
    @abstractmethod
    def text_completion(
        self, messages, **kwargs
    ):  # TODO: add support for streaming chunks
        pass

    @abstractmethod
    def text_embedding(self, payload) -> EmbeddingResponse:
        pass


class AIServiceError(Exception):
    pass
