from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from pydevs.types.completion import (
    TextCompletionConfig,
    TextCompletionPayload,
    TextCompletionResponse,
)


class AIServiceBase(ABC):
    @abstractmethod
    def text_completion(
        self,
        payload: Union[List[TextCompletionPayload], List[Dict]],
        config: Optional[TextCompletionConfig],
    ) -> TextCompletionResponse:  # TODO: add support for streaming chunks
        pass

    @abstractmethod
    def text_embedding(self, payload) -> List[float]:
        pass


class AIServiceError(Exception):
    pass
