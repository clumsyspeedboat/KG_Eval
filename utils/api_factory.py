from typing import Union
from .openai_api import OpenAIChat
from .llama_api import LlamaChat
from .base_chat import BaseChat
import logging
import configparser

logger = logging.getLogger(__name__)

class AIModelFactory:
    VALID_MODELS = {
        'openai': OpenAIChat,
        'llama': LlamaChat
    }

    @staticmethod
    def create_client(
        model_type: str,
        max_retries: int = 3,
        timeout: int = 60,
        **kwargs
    ) -> BaseChat:
        """
        Create an instance of the specified AI model client.

        Args:
            model_type (str): Type of model ('openai' or 'llama')
            max_retries (int): Maximum number of retry attempts
            timeout (int): Request timeout in seconds
            **kwargs: Model-specific configuration parameters

        Returns:
            BaseChat: Instance of the specified client
        """
        model_type = model_type.strip().lower()
        logger.info(f"Creating client for model type: {model_type}")

        try:
            # Get timeouts from config if available
            config = configparser.ConfigParser()
            config.read('config.ini')
            
            if model_type == 'llama':
                timeout = config.getint('llama', 'TIMEOUT', fallback=timeout)
                max_retries = config.getint('llama', 'MAX_RETRIES', fallback=max_retries)
            
            logger.info(f"Using timeout: {timeout}s, max_retries: {max_retries}")

            if model_type == 'openai':
                client = OpenAIChat(
                    api_key=kwargs.get('openai_api_key', kwargs.get('api_key')),
                    model=kwargs.get('model', 'gpt-4')
                )
            else:  # llama
                client = LlamaChat(
                    api_url=kwargs.get('api_url'),
                    api_key=kwargs.get('llama_api_key', kwargs.get('api_key')),
                    model=kwargs.get('model', 'llama2')
                )

            # Configure common settings from config or defaults
            client.max_retries = max_retries
            client.timeout = timeout
            logger.info(f"Successfully created {model_type} client with timeout={timeout}s")
            return client
            
        except Exception as e:
            logger.error(f"Error creating {model_type} client: {e}")
            raise
