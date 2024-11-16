from typing import TypedDict

Message = TypedDict("Message", {"role": str, "content": str})
OllamaMessage = TypedDict("OllamaMessage", {"role": str, "content": str})
OpenAIMessage = TypedDict("OpenAIMessage", {"role": str, "content": str})
