# utils/openai_api.py

from openai import OpenAI
from typing import Optional, List, Dict
from .base_chat import BaseChat
import logging

logger = logging.getLogger(__name__)

class OpenAIChat(BaseChat):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.logger.info(f"Initializing OpenAIChat with model: {model}")

    def _make_request(self, messages: List[Dict]) -> Dict:
        def request_operation():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0 if messages[0]["role"] == "system" else 0.7
            )
            return response.model_dump()

        return self._handle_retry(request_operation)

    def generate_query(self, system_prompt: str, user_query: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        try:
            response = self._make_request(messages)
            content = response["choices"][0]["message"]["content"]
            self.add_to_history("system", system_prompt)
            self.add_to_history("user", user_query)
            self.add_to_history("assistant", content)
            return content
        except Exception as e:
            self.logger.error(f"Error generating query: {e}")
            return None

    def summarize_results(self, results):
        """
        Summarize the query results using OpenAI.

        Args:
            results (list): The results returned from the Neo4j query.

        Returns:
            str: A summary of the results.
        """
        summary_prompt = "Summarize the following query results in a clear and concise manner."
        messages = [
            {"role": "system", "content": summary_prompt},
            {"role": "user", "content": str(results)}
        ]

        # Optional: Log token count for debugging
        token_count = self.count_tokens(messages)
        logger.info(f"Token count for OpenAI summarization request: {token_count}")

        try:
            response = self._make_request(messages)
            summary = response["choices"][0]["message"]["content"]
            return summary
        except Exception as e:
            logger.error(f"OpenAI API error during summarization: {e}")
            return "I'm sorry, I couldn't generate a summary of the results."

    def count_tokens(self, messages):
        """
        Count the number of tokens in the messages.

        Args:
            messages (list): The list of messages sent to the API.

        Returns:
            int: The total number of tokens.
        """
        # For simplicity, we can estimate token count by word count
        num_tokens = sum(len(message["content"].split()) for message in messages)
        return num_tokens
