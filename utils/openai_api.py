# utils/openai_api.py

import openai
import logging

logger = logging.getLogger(__name__)

class OpenAIChat:
    def __init__(self, api_key, model="gpt-4"):
        """
        Initialize the OpenAIChat class with API key and model.

        Args:
            api_key (str): Your OpenAI API key.
            model (str): The model to use for the chat completion.
        """
        openai.api_key = api_key
        self.model = model

    def generate_query(self, system_prompt, user_query):
        """
        Generate a Cypher query based on the user's input using OpenAI.

        Args:
            system_prompt (str): The system prompt providing instructions.
            user_query (str): The user's natural language query.

        Returns:
            str or None: The assistant's response.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

        # Optional: Log token count for debugging
        token_count = self.count_tokens(messages)
        logger.info(f"Token count for OpenAI request: {token_count}")

        try:
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0
            )
            assistant_message = response.choices[0].message.content
            return assistant_message
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            summary = response.choices[0].message.content
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
