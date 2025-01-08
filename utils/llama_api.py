import logging
import requests
from typing import Optional, List, Dict, Any
from time import sleep
from .base_chat import BaseChat
import configparser

logger = logging.getLogger(__name__)

class LlamaAPI:
    def __init__(self, api_url: str, api_key: str):
        self.api_url = f"{api_url}/generate"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate(self, prompt: str, temperature: float = 0.7) -> Optional[str]:
        """
        Generate a response using the Llama API.
        
        Args:
            prompt (str): The input prompt
            temperature (float): Sampling temperature
            
        Returns:
            Optional[str]: The generated response or None if failed
        """
        try:
            payload = {
                "prompt": prompt,
                "model": "llama3.3",
                "temperature": temperature
            }
            
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            response.raise_for_status()
            return response.json().get("response")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Llama API request failed: {e}")
            return None

class LlamaChat(BaseChat):
    def __init__(self, api_url: str, api_key: str, model: str = "llama2"):
        """
        Initialize the LlamaChat class.

        Args:
            api_url (str): The base URL for the Llama API
            api_key (str): The API key for authentication
            model (str): The model name (default: "llama2")
        """
        super().__init__()
        self.api_url = api_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Get timeout settings from config or use defaults
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.timeout = config.getint('llama', 'TIMEOUT', fallback=120)
        self.analysis_timeout = config.getint('llama', 'ANALYSIS_TIMEOUT', fallback=240)
        self.quick_timeout = config.getint('llama', 'QUICK_TIMEOUT', fallback=30)
        self.max_retries = config.getint('llama', 'MAX_RETRIES', fallback=3)
        self.backoff_factor = config.getint('llama', 'BACKOFF_FACTOR', fallback=5)
        
        self.logger.info(f"Initializing LlamaChat with timeouts: quick={self.quick_timeout}s, full={self.timeout}s")
        self.message_history = []

    def _make_request(self, payload: Dict, operation: str = 'default') -> Dict:
        """
        Make HTTP request with operation-specific timeouts.
        """
        def request_operation():
            # Select timeout based on operation type
            if operation == 'analysis':
                timeout = self.analysis_timeout
            elif operation == 'quick':
                timeout = self.quick_timeout
            else:
                timeout = self.timeout

            self.logger.info(f"Making {operation} request with timeout: {timeout}s")
            
            self.logger.debug(f"Making request to {self.api_url}/generate")
            self.logger.debug(f"Request payload: {payload}")
            
            try:
                response = requests.post(
                    f"{self.api_url}/generate",
                    headers=self.headers,
                    json=payload,
                    timeout=timeout  # Use operation-specific timeout
                )
                
                self.logger.debug(f"Response status: {response.status_code}")
                self.logger.debug(f"Response headers: {dict(response.headers)}")
                
                # First try to parse as JSON
                try:
                    response_data = response.json()
                    if isinstance(response_data, dict) and 'response' in response_data:
                        content = response_data['response']
                    else:
                        content = str(response_data)
                except ValueError:
                    # If not JSON, use raw text
                    content = response.text
                    
                self.logger.debug(f"Processed response content: {content[:500]}")
                
                if not content:
                    raise ValueError("Empty response from Llama API")

                # Standardize response format
                return {
                    "choices": [{
                        "message": {
                            "content": content,
                            "role": "assistant"
                        }
                    }]
                }
            except requests.Timeout:
                self.logger.error(f"Request timed out after {timeout} seconds")
                raise
            except requests.RequestException as e:
                self.logger.error(f"Request failed: {str(e)}")
                raise

        # Track attempt number for progressive timeout
        for attempt in range(self.max_retries):
            self.attempt = attempt
            try:
                return request_operation()
            except requests.Timeout:
                wait_time = self.backoff_factor * (2 ** attempt)
                self.logger.warning(f"Timeout on attempt {attempt + 1}, waiting {wait_time}s before retry")
                sleep(wait_time)
                continue
            except Exception as e:
                self.logger.error(f"Error on attempt {attempt + 1}: {e}")
                raise
        
        raise TimeoutError("All retry attempts failed")

    def _generate_fallback_response(self, message: str) -> Dict:
        """Generate a fallback response when the API fails."""
        return {
            "choices": [{
                "message": {
                    "content": message,
                    "role": "assistant"
                }
            }]
        }

    def generate_query(self, system_prompt: str, user_query: str) -> Optional[str]:
        """Generate a response using the Llama API with detailed logging."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        combined_prompt = "\n\n".join([m["content"] for m in messages])
        self.logger.debug(f"Generated combined prompt:\n{combined_prompt}")
        
        payload = {
            "prompt": combined_prompt,
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 500  # Add reasonable limit
        }
        
        try:
            response = self._make_request(payload)
            content = response["choices"][0]["message"]["content"]
            
            if not content or not content.strip():
                self.logger.warning("Received empty content from API")
                return None
                
            return content.strip()
        except Exception as e:
            self.logger.error(f"Error in generate_query: {str(e)}")
            return None

    def summarize_results(self, results: List[Dict]) -> str:
        """
        Summarize query results using Llama with extended timeout.

        Args:
            results (list): Query results to summarize

        Returns:
            str: Summary of the results
        """
        prompt = f"Summarize these query results concisely:\n{str(results)}"
        payload = {
            "prompt": prompt,
            "model": self.model,
            "temperature": 0.7
        }
        
        try:
            # Use longer timeout for analysis
            response = self._make_request(payload, operation='analysis')
            content = response["choices"][0]["message"]["content"]
            self.add_to_history("system", "Summarize results")
            self.add_to_history("assistant", content)
            return content
        except Exception as e:
            self.logger.error(f"Error summarizing results: {e}")
            return "Failed to generate summary."

    def analyze_results(self, table_md: str) -> str:
        """Analyze results with extended timeout."""
        payload = {
            "prompt": f"Analyze this table:\n{table_md}",
            "model": self.model,
            "temperature": 0.7
        }
        
        try:
            response = self._make_request(payload, operation='analysis')
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            self.logger.error(f"Error analyzing results: {e}")
            return "Failed to analyze results."

    def count_tokens(self, messages: List[Dict]) -> int:
        """
        Estimate token count for messages.

        Args:
            messages (List[Dict]): List of message dictionaries

        Returns:
            int: Estimated token count
        """
        # Use more sophisticated token counting if available
        return sum(len(message["content"].split()) * 1.3 for message in messages)

    def get_message_history(self) -> List[Dict]:
        """
        Get the conversation history.

        Returns:
            List[Dict]: List of message dictionaries
        """
        return self.message_history

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.message_history = []
        self.logger.info("Conversation history cleared")
