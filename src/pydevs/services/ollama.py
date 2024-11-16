import json
import os
import re
from typing import List, Optional

import requests

from pydevs.services.base import AIServiceBase, AIServiceError
from pydevs.types.completion import OllamaMessage


class OllamaService(AIServiceBase):
    def __init__(
        self, default_model: Optional[str] = None, host_url: Optional[str] = None
    ):
        self._default_model = default_model

        if host_url is None:
            self._host_url = os.environ.get("OLLAMA_URL") or "http://localhost:11434"
        else:
            regex = re.compile(
                r"^https?:\/\/(([\w.-]+)|(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})):11434$"
            )
            if not regex.match(host_url):
                raise ValueError(
                    "Invalid host URL - must be in the format <proto>://<hostname_or_ip>:<port>"
                )
            self._host_url = host_url

    def _health(self):
        pass

    def text_completion(
        self,
        messages: list,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.8,
        ctx_size: int = 2048,
        format: Optional[str] = None,
    ) -> List[OllamaMessage]:
        if model is None:
            if self._default_model is None:
                raise ValueError(
                    "Model must be provided as kwarg or during client initialization"
                )

        json_payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "format": format,
            "options": {"temperature": temperature, "num_ctx": ctx_size},
        }

        try:
            response = requests.post(
                f"{self._host_url}/api/chat",
                json=json_payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()  # TODO: proper status & error handling
            try:
                return [response.json()["message"]]
            except (json.JSONDecodeError, KeyError):
                raise AIServiceError("Ollama API response format error")
        except Exception as e:
            raise AIServiceError(f"Ollama API error: {e}")

    def text_embedding(self, payload):
        raise NotImplementedError()
