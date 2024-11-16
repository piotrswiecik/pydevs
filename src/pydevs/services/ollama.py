import os
import re
from typing import Optional

import requests

from pydevs.services.base import AIServiceBase, AIServiceError
from pydevs.types.completion import OllamaTextCompletionConfig


class OllamaService(AIServiceBase):
    def __init__(self, default_model: Optional[str], host_url: Optional[str]):
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

    def text_completion(self, payload, config: OllamaTextCompletionConfig = None):
        if config is None:
            if self._default_model is None:
                raise ValueError(
                    "Default model must be provided in the config or during initialization"
                )
            config = OllamaTextCompletionConfig(model=self._default_model)

        json_payload = {
            "model": config.model,
            "messages": payload,
            "stream": config.stream,
            "format": "json",
            "options": {"temperature": config.temperature, "num_ctx": config.ctx_size},
        }

        try:
            response = requests.post(
                f"{self._host_url}/api/chat",
                json=json_payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()  # TODO: proper status & error handling
            return response.json()
        except Exception as e:
            raise AIServiceError(f"Ollama API error: {e}")

    def text_embedding(self, payload):
        raise NotImplementedError()
