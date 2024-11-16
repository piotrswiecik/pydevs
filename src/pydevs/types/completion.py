from typing import TypedDict

OllamaMessage = TypedDict("OllamaMessage", {"role": str, "content": str})
OpenAIMessage = TypedDict("OpenAIMessage", {"role": str, "content": str})
