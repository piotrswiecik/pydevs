import logging

from pydevs.services.ollama import OllamaService


def test_ollama_completion():
    service = OllamaService(
        default_model="gemma2:2b", host_url="http://localhost:11434"
    )
    completion = service.text_completion(
        [{"role": "user", "content": "Hello, how are you?"}]
    )
    logging.info(completion)
