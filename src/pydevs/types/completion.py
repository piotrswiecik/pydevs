from typing import Optional

from pydantic import BaseModel


class TextCompletionPayload(BaseModel):
    role: str
    content: str


class TextCompletionConfig(BaseModel):
    model: Optional[str] = None
    stream: bool = False
    json_mode: bool = False
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 1.0


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class TextCompletionResponse(BaseModel):
    choices: list[str]
    usage: Optional[Usage] = None
