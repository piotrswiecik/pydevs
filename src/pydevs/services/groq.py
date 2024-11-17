import os
from typing import Dict, List, Literal, Optional, BinaryIO

from pydevs.services.base import AIServiceBase, AIServiceError


class GroqService(AIServiceBase):
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("GROQ_API_KEY")
        self._api_key = api_key