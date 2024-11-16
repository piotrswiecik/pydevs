from abc import ABC, abstractmethod
from typing import List

from pydevs.types.completion import Message


class AIServiceBase(ABC):
    @abstractmethod
    def text_completion(
        self, messages, **kwargs
    ) -> List[Message]:  # TODO: add support for streaming chunks
        pass

    @abstractmethod
    def text_embedding(self, payload):
        pass


class AIServiceError(Exception):
    pass
