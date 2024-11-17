import os
import requests

from abc import ABC
from dotenv import load_dotenv
from typing import Dict, List, Literal, Optional, BinaryIO
import httpx

from pydevs.services.base import AIServiceBase, AIServiceError


class ElevenLabsService(ABC):
    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            load_dotenv()
            api_key = os.environ.get("ELEVEN_LABS_API_KEY")
        self._api_key = api_key
        self._base_url = "https://api.elevenlabs.io/v1"
        
    def speak(
        self, 
        text: str, 
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_turbo_v2_5"
    ) -> bytes:
        """
        Generate speech using ElevenLabs API.
        
        Args:
            text: The text to convert to speech
            voice_id: The ID of the voice to use (default: "21m00Tcm4TlvDq8ikWAM")
            model_id: The ID of the model to use (default: "eleven_turbo_v2_5")
            
        Returns:
            bytes: The audio data in bytes
            
        Raises:
            AIServiceError: If there's an error in the API call
        """
        try:
            headers = {
                "xi-api-key": self._api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "text": text,
                "model_id": model_id,
            }
            
            response = requests.post(
                f"{self._base_url}/text-to-speech/{voice_id}",
                headers=headers,
                json=payload
            )
                
            if response.status_code != 200:
                raise AIServiceError(f"ElevenLabs API error: {response.text}")
                
            return response.content
                
        except Exception as e:
            raise AIServiceError(f"ElevenLabs speech generation error: {str(e)}")