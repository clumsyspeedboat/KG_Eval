# agents/inference_protex_agent.py

import logging
import json
import openai
from pathlib import Path
import time

logger = logging.getLogger(__name__)

class InferenceProtexAgent:
    def __init__(self, openai_api_key: str, ontology_summary_path: str, openai_model: str = "gpt-4"):
        """
        Initialize the InferenceProtexAgent.

        Args:
            openai_api_key (str): OpenAI API key.
            ontology_summary_path (str): Path to the ontology_summary.json file.
            openai_model (str): OpenAI model to use.
        """
        self.openai_api_key = openai_api_key
        openai.api_key = self.openai_api_key
        self.openai_model = openai_model
        self.ontology_summary_path = Path(ontology_summary_path)
        self.ontology = {}
        self.summary = ""
        self.load_ontology_summary()

    def load_ontology_summary(self):
        """
        Load the ontology and its summary from the JSON file.
        """
        try:
            with open(self.ontology_summary_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.ontology = data.get('ontology', {})
            self.summary = data.get('summary', "")
            logger.info(f"Loaded ontology and summary from {self.ontology_summary_path}.")
        except FileNotFoundError as e:
            logger.error(f"Ontology summary file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from ontology summary file: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading ontology summary: {e}")
            raise

    def generate_cypher_query(self, user_query: str) -> str:
        """
        Generate a Cypher query based on the user's natural language query.

        Args:
            user_query (str): The user's natural language query.

        Returns:
            str: The generated Cypher query.
        """
        try:
            prompt = self.user_prompt.format(
                ontology_summary=self.summary,
                user_query=user_query
            )
            response = openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500  # Adjust as needed
            )
            cypher_query = response.choices[0].message['content'].strip()
            logger.info("Generated Cypher query using OpenAI.")
            return cypher_query
        except openai.error.RateLimitError as e:
            logger.warning(f"Rate limit exceeded: {e}. Retrying...")
            time.sleep(1)  # Wait before retrying
            return self.generate_cypher_query(user_query)
        except Exception as e:
            logger.error(f"Error generating Cypher query: {e}")
            raise

    def validate_cypher_query(self, cypher_query: str) -> bool:
        """
        Validate the generated Cypher query syntax.

        Args:
            cypher_query (str): The Cypher query to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Simple validation: Check if the query starts with MATCH, CREATE, RETURN, etc.
        valid_start = ['MATCH', 'CREATE', 'RETURN', 'WITH', 'OPTIONAL MATCH']
        return any(cypher_query.upper().startswith(start) for start in valid_start)

    def process_user_query(self, user_query: str) -> str:
        """
        Process the user's query to generate a valid Cypher query.

        Args:
            user_query (str): The user's natural language query.

        Returns:
            str: The valid Cypher query or an error message.
        """
        try:
            cypher_query = self.generate_cypher_query(user_query)
            if self.validate_cypher_query(cypher_query):
                return cypher_query
            else:
                logger.warning("Generated Cypher query is invalid.")
                return "The generated Cypher query is invalid. Please try rephrasing your query."
        except Exception as e:
            logger.error(f"Error processing user query: {e}")
            return "An error occurred while processing your query."

    def generate_summary(self, prompt: str, text: str) -> str:
        """
        Generate a summary of the provided text using OpenAI's API with retry mechanism.

        Args:
            prompt (str): Prompt for the summarization task.
            text (str): Text to be summarized.

        Returns:
            str: Summary of the text.
        """
        try:
            full_prompt = f"{prompt}\n\n{text}"
            retries = 3
            backoff_factor = 0.5
            for attempt in range(retries):
                try:
                    response = openai.chat.completions.create(
                        model=self.openai_model,
                        messages=[
                            {"role": "system", "content": "You are a detailed and thorough summarizer."},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=0.5,
                        max_tokens=500  # Adjust as needed
                    )
                    summary = response.choices[0].message.content
                    return summary
                except openai.error.RateLimitError as e:
                    wait = backoff_factor * (2 ** attempt)
                    logger.warning(f"Rate limit exceeded. Retrying in {wait} seconds...")
                    time.sleep(wait)
            # After retries
            logger.error("Exceeded maximum retry attempts for OpenAI API.")
            raise Exception("Rate limit exceeded. Please try again later.")
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
