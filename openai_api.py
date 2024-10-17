# openai_api.py

import openai
import logging
import time
import re

logger = logging.getLogger(__name__)


class OpenAIChat:
    def __init__(self, api_key, base_url, model="gpt-4"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        openai.api_key = self.api_key
        openai.api_base = self.base_url

    def generate_response(self, conversation, temperature=0.7):
        """
        Generates a response from the assistant based on the conversation history.

        Args:
            conversation (list): List of message dictionaries containing 'role' and 'content'.
            temperature (float): Sampling temperature.

        Returns:
            str: Assistant's response.
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model, messages=conversation, temperature=temperature
            )
            assistant_message = response["choices"][0]["message"]["content"].strip()
            logger.info("Assistant response generated.")
            return assistant_message

        except openai.APIConnectionError as e:
            logger.error(f"APIConnectionError: {e}")
            print("The server could not be reached.")
            logger.debug(e.__cause__)
            return None

        except openai.RateLimitError as e:
            logger.error(f"RateLimitError: {e}")
            print("Rate limit exceeded. Please wait and try again.")
            time.sleep(5)  # Optionally wait before retrying
            return None

        except openai.APIError as e:
            logger.error(f"APIError: {e}")
            print(f"API Error: {e}")
            return None

        except openai.AuthenticationError as e:
            logger.error(f"AuthenticationError: {e}")
            print("Authentication Error: Check your OpenAI API key.")
            return None

        except Exception as e:
            logger.error(f"Unexpected error during OpenAI interaction: {e}")
            return None

    def generate_result_summary(self, conversation, query_results, temperature=0.7):
        """
        Generates a summary or answer based on the query results.

        Args:
            conversation (list): List of message dictionaries containing 'role' and 'content'.
            query_results (list): List of dictionaries representing the query results.
            temperature (float): Sampling temperature.

        Returns:
            str: Assistant's answer summarizing the query results.
        """
        # Convert query results to a string format suitable for the prompt
        result_str = f"Query Results:\n{query_results}\n"

        # Append the results to the conversation
        extended_conversation = conversation + [
            {"role": "assistant", "content": result_str}
        ]

        # Add a prompt to generate the summary
        extended_conversation.append(
            {
                "role": "assistant",
                "content": "Based on the above query results, here is the answer to your question:",
            }
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=extended_conversation,
                temperature=temperature,
            )
            summary = response["choices"][0]["message"]["content"].strip()
            logger.info("Generated summary based on query results.")
            return summary

        except openai.APIConnectionError as e:
            logger.error(f"APIConnectionError during result summary: {e}")
            print("The server could not be reached.")
            logger.debug(e.__cause__)
            return None

        except openai.RateLimitError as e:
            logger.error(f"RateLimitError during result summary: {e}")
            print("Rate limit exceeded. Please wait and try again.")
            time.sleep(5)  # Optionally wait before retrying
            return None

        except openai.APIError as e:
            logger.error(f"APIError during result summary: {e}")
            print(f"API Error: {e}")
            return None

        except openai.InvalidRequestError as e:
            logger.error(f"InvalidRequestError during result summary: {e}")
            print(f"Invalid request: {e}")
            return None

        except openai.AuthenticationError as e:
            logger.error(f"AuthenticationError during result summary: {e}")
            print("Authentication Error: Check your OpenAI API key.")
            return None

        except Exception as e:
            logger.error(f"Unexpected error during result summary generation: {e}")
            return None
