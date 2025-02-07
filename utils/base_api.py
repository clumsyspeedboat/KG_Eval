# utils/base_api.py
import logging
import requests
import openai
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

class LLMProvider:
    """Factory to create the appropriate LLM implementation."""
    @staticmethod
    def create(provider: str, **kwargs) -> 'BaseAPI':
        if provider == 'openai':
            return OpenAIAPI(api_key=kwargs.get('api_key'))
        elif provider == 'llama':
            return LlamaAPI(
                api_url=kwargs.get('api_url'),
                api_key=kwargs.get('api_key')
            )
        raise ValueError(f"Unknown provider: {provider}")

class BaseAPI(ABC):
    """Base interface for LLM API implementations."""
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        """Generate a response from the LLM."""
        pass

class OpenAIAPI(BaseAPI):
    """OpenAI API implementation."""
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__()
        openai.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return None

class LlamaAPI(BaseAPI):
    """Llama API implementation."""
    def __init__(self, api_url: str, api_key: str, timeout: int = 60):
        super().__init__()
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Optional[str]:
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except Exception as e:
            self.logger.error(f"Llama API error: {e}")
            return None
