import os

from dotenv import load_dotenv
from typing import Dict, List, Literal, Optional, BinaryIO

from openai import OpenAI

from pydevs.services.base import AIServiceBase, AIServiceError
from pydevs.types.completion import OpenAIMessage


class OpenAIService(AIServiceBase):
    def __init__(self, api_key: Optional[str] = None, default_model="gpt-4o-mini"):
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
        self._api_key = api_key
        self._client = OpenAI()
        self._client.api_key = api_key
        self._default_model = default_model

    def text_completion(
        self,
        messages: list,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        json_mode: bool = False,
        stream: bool = False,
    ) -> List[OpenAIMessage]:
        if model is None and self._default_model is None:
            raise ValueError(
                "Model must be provided as kwarg or during client initialization"
            )

        try:
            api_response = self._client.chat.completions.create(
                model=model or self._default_model,
                messages=messages,
                max_completion_tokens=max_tokens,
                response_format={"type": "json_object" if json_mode else "text"},
                stream=stream,
                temperature=temperature,
            )
            return [
                {"role": choice.message.role, "content": choice.message.content}
                for choice in api_response.choices
            ]
        except Exception as e:
            raise AIServiceError(f"OpenAI API error: {e}")

    def text_embedding(
        self,
        payload: str,
        model: Literal[
            "text-embedding-3-small", "text-embedding-3-large"
        ] = "text-embedding-3-small",
    ):
        try:
            api_response = self._client.embeddings.create(model=model, input=payload)
            return api_response.data[0].embedding
        except (AttributeError, IndexError):
            raise AIServiceError("Invalid API response")
        except Exception as e:
            raise AIServiceError(f"OpenAI API error: {e}")

    async def speak(self, text: str) -> bytes:
        """
        Convert text to speech using OpenAI's TTS API.
        
        Args:
            text: The text to convert to speech
            
        Returns:
            bytes: The audio data in bytes
        """
        try:
            response = await self._client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=text
            )
            return response.content
        except Exception as e:
            raise AIServiceError(f"OpenAI TTS API error: {e}")

    async def transcribe(self, audio_data: BinaryIO, language: str = 'pl') -> str:
        """
        Transcribe audio data to text using OpenAI's Whisper API.
        
        Args:
            audio_data: Binary audio data (file-like object)
            language: Language code for transcription (default: 'pl' for Polish)
            
        Returns:
            str: The transcribed text
        """
        try:
            transcription = await self._client.audio.transcriptions.create(
                file=audio_data,
                language=language,
                model="whisper-1"
            )
            return transcription.text
        except Exception as e:
            raise AIServiceError(f"OpenAI transcription API error: {e}")
