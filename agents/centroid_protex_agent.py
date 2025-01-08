# /agents/centroid_protex_agent.py

import logging
import json
from pathlib import Path
import time
import psutil
import re
import configparser
from typing import Optional  # Add this import

from utils.context_loader import load_context
from utils.neo4j_helper import Neo4jHelper
from utils.api_factory import AIModelFactory
from utils.prompt_loader import load_agent_prompts


class CentroidProtexAgent:
    """
    The CentroidProtexAgent coordinates interactions between the user, the knowledge graph, 
    and the OpenAI API. It can handle internal and external queries, attempt to match predefined 
    queries, or generate fallback queries from provided context if no suitable match is found.
    """

    def __init__(self, config_path='config.ini'):
        """
        Initialize the agent, load configuration, create Neo4j and OpenAI instances, 
        and prepare predefined queries and prompts.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.logger = self.setup_logging()
        self.config = self.load_config(config_path)
        self.neo4j_helper = Neo4jHelper(
            uri=self.config.get('neo4j', 'NEO4J_URI'),
            user=self.config.get('neo4j', 'NEO4J_USER'),
            password=self.config.get('neo4j', 'NEO4J_PASSWORD')
        )
        
        # Get timeout and retry settings from config
        self.max_retries = self.config.getint('llama', 'MAX_RETRIES', fallback=5)
        self.backoff_factor = self.config.getint('llama', 'BACKOFF_FACTOR', fallback=10)
        self.analysis_timeout = self.config.getint('llama', 'ANALYSIS_TIMEOUT', fallback=600)
        
        # Initialize AI model with validation
        try:
            model_type = self.config.get('model_settings', 'DEFAULT_MODEL', fallback='openai')
            self.logger.info(f"Configured model type: {model_type}")
            
            if model_type.lower().strip() == 'openai':
                self.ai_client = AIModelFactory.create_client(
                    'openai',
                    openai_api_key=self.config.get('openai', 'OPENAI_API_KEY'),
                    model='gpt-4'
                )
            elif model_type.lower().strip() == 'llama':
                # Pass timeout settings from config
                timeout = self.config.getint('llama', 'TIMEOUT', fallback=120)
                max_retries = self.config.getint('llama', 'MAX_RETRIES', fallback=3)
                
                self.ai_client = AIModelFactory.create_client(
                    'llama',
                    api_url=self.config.get('llama', 'LLAMA_API_URL'),
                    llama_api_key=self.config.get('llama', 'LLAMA_API_KEY'),
                    model='llama2',
                    timeout=timeout,
                    max_retries=max_retries
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")
        except Exception as e:
            self.logger.error(f"Error initializing AI client: {e}")
            raise

        base_dir = Path(__file__).parent.parent
        self.context_files_dir = base_dir / 'resources' / 'context_files'
        self.queries_dir = base_dir / 'resources' / 'queries'
        self.configs_dir = base_dir / 'configs'

        self.load_context_files()
        self.load_predefined_queries()
        self.load_prompts()

    def setup_logging(self):
        """
        Set up basic logging for the agent.

        Returns:
            logging.Logger: The configured logger instance.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        return logging.getLogger(__name__)

    def load_config(self, config_path):
        """
        Load configuration from the given INI file.

        Args:
            config_path (str): Path to the config file.

        Returns:
            configparser.ConfigParser: Loaded configuration.
        """
        config = configparser.ConfigParser()
        config_file = Path(config_path)
        if not config_file.is_file():
            self.logger.error(f"Configuration file '{config_path}' not found.")
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        config.read(config_file)
        self.logger.info("Configuration loaded successfully.")
        return config

    def load_context_files(self):
        """
        Load context data (edge types, node types, protein connections) from JSON files.
        """
        start_time = time.time()
        process = psutil.Process()
        cpu_before = process.cpu_times()
        mem_before = process.memory_info().rss

        try:
            self.edge_types = load_context(str(self.context_files_dir / 'Edge_Type.json'), as_dict=True)
            self.node_types = load_context(str(self.context_files_dir / 'Node_Type.json'), as_dict=True)
            self.protein_connections = load_context(str(self.context_files_dir / 'Protein_connections_2Hops.json'), as_dict=True)
            self.logger.info("Context files loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading context files: {e}")
            raise

        elapsed = time.time() - start_time
        cpu_after = process.cpu_times()
        mem_after = process.memory_info().rss
        self.logger.info(f"Loaded context files in {elapsed:.2f} seconds.")
        self.logger.info(f"CPU time used: {cpu_after.user - cpu_before.user:.2f} seconds.")
        self.logger.info(f"Memory used: {(mem_after - mem_before) / (1024 * 1024):.2f} MB.")

    def load_predefined_queries(self):
        """
        Load predefined queries from a JSON file (query_set1.json).
        These queries include 'description', 'keywords', and 'query'.
        """
        self.predefined_queries = {}
        try:
            query_set_path = self.queries_dir / 'query_set1.json'
            with open(query_set_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.predefined_queries = data.get("queries", {})
            self.logger.info("Predefined queries loaded successfully.")
        except FileNotFoundError:
            self.logger.error("Predefined queries file not found.")
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in predefined queries: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error loading predefined queries: {e}")

    def load_prompts(self):
        """
        Load agent prompts (agents_prompts.json) containing system and user prompts for the CentroidProtexAgent.
        """
        try:
            prompts_path = self.configs_dir / 'agent_prompts.json'
            self.agent_prompts = load_agent_prompts(str(prompts_path))
            self.system_prompt = self.agent_prompts.get('CentroidProtexAgent', {}).get('system_prompt', '')
            self.user_prompt = self.agent_prompts.get('CentroidProtexAgent', {}).get('user_prompt', '')
            self.logger.info("Agent prompts loaded successfully.")
        except FileNotFoundError:
            self.logger.error("Agent prompts file not found.")
            self.system_prompt = ''
            self.user_prompt = ''
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error in agent prompts: {e}")
            self.system_prompt = ''
            self.user_prompt = ''
        except Exception as e:
            self.logger.error(f"Unexpected error loading agent prompts: {e}")
            self.system_prompt = ''
            self.user_prompt = ''

    def create_context_summary(self):
        """
        Create a textual summary of context data to guide fallback Cypher query generation.

        Returns:
            str: A summary of node types, edge types, and protein connections.
        """
        summary = "Context Summary:\n\n"
        # Add metaproteomics-specific context
        summary += "Data Types:\n"
        summary += "- Proteins (nodes with UniProt IDs)\n"
        summary += "- Peptides (nodes with sequence information)\n"
        summary += "- Samples (nodes with experimental metadata)\n"
        summary += "- Abundance measurements\n"
        summary += "Node Types:\n"
        for item in self.node_types:
            summary += f"- {item}\n"
        summary += "\nEdge Types:\n"
        for item in self.edge_types:
            summary += f"- {item}\n"
        summary += "\nProtein Connections (2 Hops):\n"
        for item in self.protein_connections:
            summary += f"- {item}\n"
        return summary

    def determine_query_type(self, user_query):
        self.logger.debug(f"Determining query type for: {user_query}")
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.user_prompt}\n\n"
            f"User Query: {user_query}\n"
            "Determine if this query is related to the internal knowledge base or external information. Respond with 'internal' or 'external'."
        )

        try:
            response = self.ai_client.generate_query("", prompt)
            return response.strip().lower() if response else "external"
        except Exception as e:
            self.logger.error(f"Error determining query type: {e}")
            return "external"

    def generate_cypher_query(self, user_query):
        self.logger.debug(f"Generating Cypher query for: {user_query}")
        
        # Add basic query templates for common patterns
        basic_templates = {
            "count patients": "MATCH (p:Patient) WHERE p.status = 'remission' RETURN count(p) as patient_count",
            "patient status": "MATCH (p:Patient) RETURN p.status, count(p) as count",
            "remission status": "MATCH (p:Patient) WHERE p.status = 'remission' RETURN count(p) as in_remission"
        }
        
        # Check for basic pattern matches first
        lower_query = user_query.lower()
        if "patients" in lower_query and "remission" in lower_query:
            self.logger.info("Using template query for remission count")
            return basic_templates["remission status"]
            
        # If no template matches, try AI generation
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.user_prompt}\n\n"
            f"User Query: {user_query}\n"
            "Generate a Cypher query that can be used to retrieve relevant information from the Neo4j database."
        )

        try:
            cypher_query = self.ai_client.generate_query("", prompt)
            if cypher_query:
                cleaned_query = cypher_query.strip()
                self.logger.info(f"Generated Cypher query: {cleaned_query}")
                return cleaned_query
            else:
                self.logger.warning("AI query generation failed, using fallback")
                return basic_templates.get("patient status", "MATCH (p:Patient) RETURN p.status, count(p)")
        except Exception as e:
            self.logger.error(f"Error generating Cypher query: {e}")
            return basic_templates.get("patient status", "MATCH (p:Patient) RETURN p.status, count(p)")

    def generate_fallback_cypher_query(self, user_query):
        """
        Generate a fallback Cypher query if no predefined query matches.

        Args:
            user_query (str): The user's query.

        Returns:
            str: A fallback Cypher query or empty string on failure.
        """
        context_summary = self.create_context_summary()
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.user_prompt}\n\n"
            f"{context_summary}\n\n"
            f"User Query: {user_query}\n"
            "No suitable predefined query was found. Please generate a fallback Cypher query that best attempts to retrieve relevant information given the context."
        )
        try:
            fallback_query = self.ai_client.generate_query("", prompt).strip()
        except Exception as e:
            self.logger.error(f"Error generating fallback Cypher query: {e}")
            fallback_query = ""
        return fallback_query

    def generate_external_answer(self, user_query):
        """
        Generate an external answer for queries unrelated to the internal knowledge base.

        Args:
            user_query (str): The user's query.

        Returns:
            str: External answer text.
        """
        prompt = (
            f"{self.system_prompt}\n"
            f"{self.user_prompt}\n\n"
            "The user's query is external. Provide a helpful, factual external answer.\n\n"
            f"User Query: {user_query}\n"
        )
        try:
            external_answer = self.ai_client.generate_query("", prompt).strip()
        except Exception as e:
            self.logger.error(f"Error generating external answer: {e}")
            external_answer = "I'm sorry, I couldn't provide an external answer at this time."
        return external_answer

    def execute_cypher_query(self, cypher_query):
        """
        Execute the given Cypher query against the Neo4j database.

        Args:
            cypher_query (str): The Cypher query string.

        Returns:
            list: List of result records (dict) or an empty list on error.
        """
        if not self.neo4j_helper:
            self.logger.error("Neo4j helper is not initialized.")
            return []
        cleaned_query = re.sub(r"(?i)^cypher query:?\s*", "", cypher_query.strip())
        try:
            results = self.neo4j_helper.run_query(cleaned_query)
            self.logger.info("Cypher query executed successfully.")
        except Exception as e:
            self.logger.error(f"Error executing Cypher query: {e}")
            results = []
        return results

    def extract_keywords(self, text: str) -> set:
        """
        Extract keywords from user query by removing stopwords and punctuation.

        Args:
            text (str): Input text from the user.

        Returns:
            set: Set of keywords extracted from the user query.
        """
        stopwords = {
            'the','in','with','is','of','a','an','what','list','all','to','for','and','or','on','by','from','how','many','are'
        }
        words = re.findall(r'\b\w+\b', text.lower())
        return {w for w in words if w not in stopwords}

    def match_predefined_query(self, user_query) -> dict:
        """
        Attempt to match the user's query to a predefined query based on keyword overlap.

        Args:
            user_query (str): User's natural language query.

        Returns:
            dict: Matched query dictionary or an empty dict if no match.
        """
        if not self.predefined_queries:
            self.logger.warning("No predefined queries available.")
            return {}

        user_keywords = self.extract_keywords(user_query)
        best_match = None
        best_overlap = 0

        for q_data in self.predefined_queries.values():
            if not isinstance(q_data, dict) or 'keywords' not in q_data:
                continue
            q_keywords = {kw.lower() for kw in q_data.get('keywords', [])}
            overlap = len(user_keywords.intersection(q_keywords))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = q_data

        MIN_OVERLAP = 1
        if best_match and best_overlap >= MIN_OVERLAP:
            self.logger.info(f"Best match found with keyword overlap {best_overlap}")
            return best_match

        self.logger.info("No suitable match found based on keyword overlap.")
        return {}

    def results_to_markdown_table(self, results):
        """
        Convert the query results (list of dicts) into a Markdown table.

        Args:
            results (list): Query result records.

        Returns:
            str: A Markdown formatted table or "No results found." if empty.
        """
        if not results:
            return "No results found."

        headers = list(results[0].keys())
        table_md = "| " + " | ".join(headers) + " |\n"
        table_md += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for record in results:
            row_data = [str(record[h]) for h in headers]
            table_md += "| " + " | ".join(row_data) + " |\n"
        return table_md.strip()

    def analyze_results(self, table_md):
        """
        Analyze the given Markdown table of results using AI with retry logic.
        """
        self.logger.debug(f"Attempting to analyze results:\n{table_md}")
        
        # First try quick basic analysis
        basic_analysis = self._generate_basic_analysis(table_md)
        
        # Get retry settings
        max_attempts = getattr(self, 'max_retries', 5)  # Default to 5 if not set
        backoff = getattr(self, 'backoff_factor', 10)   # Default to 10 if not set
        
        # Try AI analysis with retries
        for attempt in range(max_attempts):
            try:
                prompt = (
                    f"Analyze this medical trial data concisely:\n\n"
                    f"{table_md}\n\n"
                    "Focus on: \n"
                    "1. Treatment effectiveness\n"
                    "2. Key patterns\n"
                    "3. Success rates\n"
                    "Provide a brief, clear analysis."
                )

                analysis = self.ai_client.generate_query("", prompt)
                if analysis and len(analysis.strip()) > 0:
                    return f"{analysis.strip()}\n\n{basic_analysis}"
                    
                wait_time = backoff * (2 ** attempt)
                self.logger.warning(f"Empty analysis on attempt {attempt + 1}, waiting {wait_time}s")
                time.sleep(wait_time)
                
            except Exception as e:
                self.logger.error(f"Analysis attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    wait_time = backoff * (2 ** attempt)
                    self.logger.info(f"Retrying after {wait_time}s")
                    time.sleep(wait_time)
                continue
        
        # If all AI attempts fail, return basic analysis
        self.logger.warning("All AI analysis attempts failed, using basic analysis")
        return basic_analysis

    def _generate_basic_analysis(self, table_md: str) -> str:
        """Generate quick statistical analysis of the results."""
        try:
            # Parse table data
            lines = [l for l in table_md.split('\n') if l.strip()]
            if len(lines) < 3:
                return "No results to analyze."

            # Count remission cases
            remission_count = table_md.lower().count('yes')
            no_remission_count = table_md.lower().count('no')
            total_cases = remission_count + no_remission_count

            # Calculate percentages
            remission_rate = (remission_count / total_cases * 100) if total_cases > 0 else 0
            
            analysis = [
                f"Total cases analyzed: {total_cases}",
                f"Patients in remission: {remission_count} ({remission_rate:.1f}%)",
                f"Patients not in remission: {no_remission_count}",
            ]

            return "\n".join(analysis)
            
        except Exception as e:
            self.logger.error(f"Error in basic analysis: {e}")
            return "Basic statistical analysis failed."

    def update_neo4j_connection(self, uri, user, password):
        """
        Update the Neo4j connection parameters and reinitialize the Neo4jHelper.

        Args:
            uri (str): The Neo4j URI.
            user (str): The Neo4j username.
            password (str): The Neo4j password.
        """
        try:
            self.neo4j_helper = Neo4jHelper(uri=uri, user=user, password=password)
            self.logger.info("Neo4j connection updated successfully.")
        except Exception as e:
            self.logger.error(f"Failed to update Neo4j connection: {e}")
            raise

    def process_query(self, user_input: str) -> dict:
        self.logger.info(f"Processing query: {user_input}")
        try:
            query_type = self.determine_query_type(user_input)
            self.logger.debug(f"Query type determined as: {query_type}")

            if query_type == 'external':
                self.logger.debug("Handling external query")
                external_answer = self.generate_external_answer(user_input)
                self.logger.debug(f"External answer generated: {external_answer[:100]}...")
                return {
                    'response': external_answer,
                    'results': None,
                    'analysis': None
                }

            self.logger.debug("Handling internal query")
            cypher_query = self.generate_cypher_query(user_input)
            
            if not cypher_query:
                self.logger.warning("Failed to generate Cypher query")
                return {
                    'response': "I'm sorry, I couldn't generate a query for your request.",
                    'results': None,
                    'analysis': None
                }

            self.logger.info(f"Generated Cypher Query: {cypher_query}")

            matched_query = self.match_predefined_query(user_input)
            if not matched_query:
                # No suitable predefined match found, use fallback
                fallback_query = self.generate_fallback_cypher_query(user_input)
                if not fallback_query:
                    return {
                        'response': "I couldn't generate a fallback query for your request.",
                        'results': None,
                        'analysis': None
                    }

                results = self.execute_cypher_query(fallback_query)
                table_md = self.results_to_markdown_table(results)
                analysis = self.analyze_results(table_md)
                response = f"**Query Results (Fallback):**\n\n{table_md}\n\n**Analysis:**\n\n{analysis}"
                return {
                    'response': response,
                    'results': results,
                    'analysis': analysis
                }
            else:
                # Match found, execute matched query
                cypher_to_execute = matched_query.get('query', '')
                if not cypher_to_execute:
                    return {
                        'response': "The matched query does not contain a valid Cypher statement.",
                        'results': None,
                        'analysis': None
                    }

                results = self.execute_cypher_query(cypher_to_execute)
                table_md = self.results_to_markdown_table(results)
                analysis = self.analyze_results(table_md)
                response = f"""**Query Results:**\n\n{table_md}\n\n**Analysis:**\n\n{analysis}"""
                return {
                    'response': response,
                    'results': results,
                    'analysis': analysis
                }

        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                'response': "An unexpected error occurred while processing your query.",
                'results': None,
                'analysis': None
            }
