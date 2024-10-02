# openai_api.py

import openai
import logging
import time
import streamlit as st
from openai import OpenAIError, OpenAI
import re

# Configure the logger
logger = logging.getLogger(__name__)

def chat_gpt(api_key: str, base_url: str, model: str, prompt: str, temperature: float = 0.7) -> str:
    """
    Calls the OpenAI Chat API with the given model, prompt, and temperature, and returns the generated Cypher query.
    
    Args:
        api_key (str): The API key for authenticating OpenAI API requests.
        base_url (str): The base URL for the OpenAI API (for custom or local API servers).
        model (str): The model ID to use for generating the completion.
        prompt (str): The natural language prompt to be translated into a Cypher query.
        temperature (float): The level of randomness for OpenAI responses. Defaults to 0.7.
    
    Returns:
        str: The generated Cypher query, or None if there was an error.
    """
    try:
        # Set the OpenAI key and base URL
        openai.api_key = api_key
        openai.api_base = base_url

        # Initialize the OpenAI client with the API key and base URL
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Make the API call using chat completion, including the temperature parameter
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that translates user queries into Cypher queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature  # Add the temperature parameter here
        )

        # Access the content of the assistant's message properly
        cypher_query = response.choices[0].message.content.strip()

        # Use a regular expression to replace capitalized labels with lowercase labels (e.g., "Drug" to "drug")
        cypher_query = re.sub(r'\b([A-Z][a-z]*)\b', lambda match: match.group(0).lower(), cypher_query)

        logger.info(f"Generated Cypher Query: {cypher_query}")
        return cypher_query

    except openai.AuthenticationError:
        logger.error("Invalid OpenAI API key provided.")
        st.error("Authentication Error: Invalid OpenAI API key. Please check your configuration.")
        return None

    except openai.RateLimitError:
        logger.error("OpenAI API rate limit exceeded.")
        st.error("Rate Limit Exceeded: Please wait a moment and try again.")
        time.sleep(5)  # Wait before retrying
        return None

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {e}")
        st.error(f"OpenAI API error: {e}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error during translation: {e}")
        st.error(f"Unexpected error: {e}")
        return None
