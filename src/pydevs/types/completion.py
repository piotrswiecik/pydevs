from typing import List, Optional, TypedDict

from pydantic import BaseModel


OllamaMessage = TypedDict("OllamaMessage", {"role": str, "content": str})



class TextCompletionPayload(BaseModel):
    role: str
    content: str


class TextCompletionConfig(BaseModel):
    model: Optional[str] = None
    stream: bool = False
    json_mode: bool = False
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0


class OllamaTextCompletionConfig(BaseModel):
    model: str
    stream: bool = False
    temperature: float = 1.0
    ctx_size: int = 2048


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class TextCompletionResponse(BaseModel):
    choices: List[str]
    usage: Optional[Usage] = None


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    usage: Optional[Usage] = None
