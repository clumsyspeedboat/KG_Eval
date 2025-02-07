# centroid_protex_agent.py
import logging
import json
from pathlib import Path
import configparser
from typing import Dict, List
import requests

from .neo4j_agent import Neo4jAgent

class CentroidProtexAgent:
    def __init__(self, config_path=None, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config if config else self._load_config(config_path or 'config.ini')
        self.model_type = self.config.get('model_settings', 'default_model', fallback='llama')
        self.neo4j_agent = None
        
        # Initialize API settings
        self._initialize_api_settings()
        self.logger.info("Loading context files...")
        self.context = self._load_context_files()
        self.logger.info(f"Loaded context from {len(self.context)} files")

    def _initialize_api_settings(self):
        """Initialize API configuration based on model type"""
        try:
            # Convert model type to lowercase for consistent comparison
            self.model_type = self.model_type.lower()
            self.logger.debug(f"[DEBUG] Initializing API settings for model type: {self.model_type}")
            
            if self.model_type == 'openai':
                self.api_key = self.config.get('openai', 'openai_api_key')
                self.logger.debug("[DEBUG] Configured for OpenAI API")
            else:
                self.api_url = self.config.get('llama', 'llama_api_url')
                self.api_key = self.config.get('llama', 'llama_api_key')
                self.logger.debug("[DEBUG] Configured for Llama API")
            self.timeout = self.config.getint('llama', 'timeout', fallback=60)
        except configparser.Error as e:
            self.logger.error(f"Configuration error: {e}")
            raise

    def _load_config(self, config_path: str) -> configparser.ConfigParser:
        config = configparser.ConfigParser()
        config.read(config_path)
        return config

    def set_neo4j_agent(self, agent: Neo4jAgent):
        """Set the Neo4j agent instance"""
        self.neo4j_agent = agent

    def _call_llm(self, prompt: str) -> Dict:
        """Make API call to selected LLM provider"""
        self.logger.debug(f"[DEBUG] Calling LLM with model type: {self.model_type}")
        try:
            if self.model_type == 'openai':
                # Only import openai when needed
                from openai import Client
                client = Client(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7
                )
                return {'response': response.choices[0].message.content.strip()}
            else:
                # Otherwise use Llama
                response = requests.post(
                    self.api_url,
                    headers={
                        'Authorization': f'Bearer {self.api_key}',
                        'Content-Type': 'application/json'
                    },
                    json={'prompt': prompt},
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, dict) and 'response' in result:
                        return result
                    elif isinstance(result, dict):
                        return {'response': json.dumps(result)}
                    else:
                        return {'response': str(result)}
                return {'response': ''}
        except Exception as e:
            self.logger.error(f"LLM API error: {e}")
            return {'response': ''}


    def _load_context_files(self) -> Dict:
        """Load all context files"""
        context = {}
        context_dir = Path(__file__).parent.parent / 'resources' / 'context_files'
        
        if not context_dir.exists():
            self.logger.error(f"Context directory not found: {context_dir}")
            return {}

        self.logger.info(f"Loading context from: {context_dir}")
        context_files = list(context_dir.glob('*.json'))
        self.logger.info(f"Found {len(context_files)} context files")

        for file in context_files:
            self.logger.info(f"Loading context file: {file.name}")
            try:
                # Use utf-8-sig encoding to handle UTF-8 BOM
                with open(file, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                    context[file.stem] = data
                    self.logger.info(f"Successfully loaded {file.name}")
            except json.JSONDecodeError as je:
                self.logger.error(f"JSON parsing error in {file}: {str(je)}")
                # Try fallback without BOM handling
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        context[file.stem] = data
                        self.logger.info(f"Successfully loaded {file.name} with fallback encoding")
                except Exception as e2:
                    self.logger.error(f"Fallback loading failed for {file}: {str(e2)}")
            except Exception as e:
                self.logger.error(f"Error loading context file {file}: {str(e)}")

        return context

    def _format_results_as_table(self, results: List[Dict]) -> str:
        """Format results as a markdown table with proper formatting"""
        if not results:
            return "No results found."

        # Get all unique keys from all results
        headers = set()
        for result in results:
            headers.update(result.keys())
        headers = sorted(list(headers))

        # Create table header
        table = "| " + " | ".join(headers) + " |\n"
        table += "|" + "|".join(["---" for _ in headers]) + "|\n"

        # Add rows
        for result in results:
            row = []
            for header in headers:
                value = result.get(header, "")
                # Format lists and complex objects
                if isinstance(value, (list, dict)):
                    value = str(value).replace("\n", " ")
                # Limit cell length and escape pipes
                value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                value = value.replace("|", "\\|")
                row.append(value)
            table += "| " + " | ".join(row) + " |\n"

        summary = f"""### Summary
- Total results: {len(results)}
- Fields: {', '.join(headers)}

### Detailed Results
{table}
"""
        return summary

    def format_results(self, user_query: str, query_matching: Dict, results: List[Dict]) -> Dict:
        """Enhanced formatting with chain of thought and table display"""
        self.logger.debug(f"[DEBUG] ====== Starting format_results ======")
        self.logger.debug(f"[DEBUG] user_query: {user_query}")
        self.logger.debug(f"[DEBUG] query_matching type: {type(query_matching)}")
        self.logger.debug(f"[DEBUG] query_matching content: {query_matching}")
        self.logger.debug(f"[DEBUG] results length: {len(results)}")
        
        # Force query_matching to be a dict and log the conversion
        if not isinstance(query_matching, dict):
            self.logger.debug(f"[DEBUG] Converting query_matching from {type(query_matching)} to dict")
            try:
                if isinstance(query_matching, str):
                    query_matching = json.loads(query_matching)
                    self.logger.debug("[DEBUG] Successfully parsed query_matching string to dict")
                else:
                    query_matching = {}
                    self.logger.debug("[DEBUG] Reset query_matching to empty dict")
            except Exception as e:
                self.logger.debug(f"[DEBUG] Error converting query_matching: {str(e)}")
                query_matching = {}
        
        try:
            if not results:
                self.logger.debug("[DEBUG] No results found, returning empty response")
                return {
                    "formatted_response": "No results found for your query.",
                    "thought_process": {
                        "original_query": user_query,
                        "query_matching": query_matching,
                        "results_summary": "No results found"
                    },
                    "results": []
                }

            # Log matched_query extraction
            matched = query_matching.get('matched_query', {})
            self.logger.debug(f"[DEBUG] Extracted matched_query: {matched}")
            
            if not isinstance(matched, dict):
                self.logger.debug(f"[DEBUG] matched_query is not a dict, type: {type(matched)}")
                matched = {}

            # Format results as table before sending to LLM
            formatted_table = self._format_results_as_table(results)
            
            prompt = f"""
You are a helpful biomedical data analyst. Analyze the following query results and provide a clear summary:

User Query: {user_query}

Query Matching Process:
Best Match Score: {query_matching.get('match_score', 0)}
Selected Cypher: {matched.get('query', 'No query found')}

Results Analysis:
{formatted_table}

Please provide a natural language response including:
1. Answer to the user's query (e.g., total count if asking "how many")
2. Breakdown of key statistics (e.g., distribution by disease type, drug type)
3. Notable patterns or insights (e.g., commonalities, outliers)

Format your response using h3 markdown headers (i.e., "### Header Title") for section headings and use concise bullet points. where appropriate.
"""
            llm_response = self._call_llm(prompt)
            response_text = llm_response.get('response', '') if isinstance(llm_response, dict) else str(llm_response)
            
            # Combine LLM analysis with the table
            final_response = f"""
{formatted_table}

{response_text}
"""
            return {
                "formatted_response": final_response,
                "thought_process": {
                    "original_query": user_query,
                    "query_matching": query_matching,
                    "results_summary": response_text
                },
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting results: {str(e)}")
            fallback_response = self._fallback_format(results)
            return {
                "formatted_response": fallback_response,  # Always include formatted_response
                "thought_process": {
                    "original_query": user_query,
                    "query_matching": {},
                    "results_summary": f"Error processing results: {str(e)}"
                },
                "results": results
            }

    def _fallback_format(self, results: List[Dict]) -> str:
        """Simple fallback formatting when LLM fails"""
        return self._format_results_as_table(results)
