import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from time import sleep

class BaseChat(ABC):
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.message_history = []
        self.max_retries = 3
        self.timeout = 60
        self.backoff_factor = 2

    @abstractmethod
    def generate_query(self, system_prompt: str, user_query: str) -> Optional[str]:
        pass

    @abstractmethod
    def summarize_results(self, results: List[Dict]) -> str:
        pass

    def _handle_retry(self, operation: callable, *args, **kwargs) -> Any:
        """
        Generic retry handler with exponential backoff.
        
        Args:
            operation: Function to retry
            args: Positional arguments for the operation
            kwargs: Keyword arguments for the operation
            
        Returns:
            Any: Result of the operation or None on failure
        """
        last_exception = None
        for attempt in range(self.max_retries):
            try:
                self.logger.debug(f"Attempt {attempt + 1} of {self.max_retries}")
                return operation(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    sleep_time = self.backoff_factor ** attempt
                    self.logger.warning(f"Attempt {attempt + 1} failed, retrying in {sleep_time}s: {str(e)}")
                    sleep(sleep_time)
                else:
                    self.logger.error(f"All retry attempts failed: {str(e)}")
        
        raise last_exception

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.message_history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.message_history = []
        self.logger.info("Conversation history cleared")

    def get_message_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.message_history.copy()
