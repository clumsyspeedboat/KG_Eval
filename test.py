import configparser
import logging
import json
from pathlib import Path
from neo4j import GraphDatabase, basic_auth
import requests
from typing import Dict, List
import re
import time
import difflib

DEBUG = True
VERBOSE = False

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)
    if DEBUG:
        logger.setLevel(logging.DEBUG)
        print("[DEBUG] Logging initialized with DEBUG level")
    return logger

def load_config(config_path: str) -> configparser.ConfigParser:
    if DEBUG:
        print(f"[DEBUG] Loading configuration from {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

class Neo4jHelper:
    def __init__(self, uri, user, password):
        if DEBUG:
            print(f"[DEBUG] Initializing Neo4j connection to {uri}")
        self.driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
        
    def close(self):
        self.driver.close()

    def run_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        if DEBUG:
            print(f"[DEBUG] Executing Neo4j query:\n{query}")
            if parameters:
                print(f"[DEBUG] Query parameters: {parameters}")
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

class LLMAgent:
    def __init__(self, config: configparser.ConfigParser):
        if DEBUG:
            print("[DEBUG] Initializing LLM Agent")
        self.llama_url = config.get('llama', 'llama_api_url')
        self.llama_key = config.get('llama', 'llama_api_key')
        self.timeout = config.getint('llama', 'timeout')
        self.retries = config.getint('llama', 'retries')
        self.backoff = config.getfloat('llama', 'backoff')
        self.max_cypher_attempts = config.getint('llama', 'max_cypher_attempts')
        self.context = self.load_context_files()

    def load_context_files(self) -> Dict:
        if DEBUG:
            print("[DEBUG] Loading context files")
        base_path = Path('resources/demo')
        context = {}
        # Using two consolidated files for a database-agnostic schema
        for filename in ['nodes.json', 'relationships.json']:
            file_path = base_path / filename
            if DEBUG:
                print(f"[DEBUG] Loading {file_path}")
            if not file_path.exists():
                raise FileNotFoundError(f"Context file not found: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                context[filename] = json.load(f)
            if VERBOSE:
                print(f"[VERBOSE] Loaded {filename}:\n{json.dumps(context[filename], indent=2)}")
        return context

    def _call_llm(self, prompt: str) -> Dict:
        if DEBUG:
            print(f"[DEBUG] Calling LLM with prompt (first 5000 chars): {prompt[:5000]}...")
        attempts = 0
        while attempts < self.retries:
            try:
                response = requests.post(
                    self.llama_url,
                    headers={
                        'Authorization': f'Bearer {self.llama_key}',
                        'Content-Type': 'application/json'
                    },
                    json={'prompt': prompt},
                    timeout=self.timeout
                )
                if DEBUG:
                    print(f"[DEBUG] LLM response status: {response.status_code}")
                if response.status_code != 200:
                    raise ValueError(f"LLM API Error: {response.status_code}")
                return response.json()
            except Exception as e:
                attempts += 1
                if DEBUG:
                    print(f"[DEBUG] LLM call failed (attempt {attempts}): {str(e)}")
                time.sleep(self.backoff * (2 ** (attempts - 1)))
        return {}

    def analyze_query(self, query: str) -> Dict:
        if DEBUG:
            print(f"[DEBUG] Analyzing query: {query}")
        prompt = f"""
You are an expert in converting natural language queries to graph database queries.
Here is the database schema:
Nodes: {json.dumps(self.context.get('nodes.json', {}), indent=2)}
Relationships: {json.dumps(self.context.get('relationships.json', {}), indent=2)}

Analyze the following natural language query and extract:
- Node types involved
- Relationship types involved
- Any constraints (e.g., specific names or values)

Query: "{query}"

Provide the output as a JSON with keys: nodes, relationships, constraints.
"""
        response = self._call_llm(prompt)
        if not response or response.get('response') == '_USE_RULES_':
            if DEBUG:
                print("[DEBUG] Falling back to rule-based analysis")
            return self._rule_based_analysis(query)
        try:
            analysis = json.loads(response.get('response', '{}'))
            if DEBUG:
                print(f"[DEBUG] LLM analysis: {analysis}")
            return analysis
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to parse LLM response: {str(e)}. Using fallback.")
            return self._rule_based_analysis(query)

    def _rule_based_analysis(self, query: str) -> Dict:
        """
        A generic fallback that tokenizes the query (after removing common stop words)
        and uses fuzzy matching to determine node types and relationship types.
        Additionally, it extracts proper noun phrases and assigns them as constraints.
        """
        analysis = {"nodes": [], "relationships": [], "constraints": {}}
        stop_words = {'which', 'who', 'what', 'are', 'the', 'of', 'with', 'in', 'on', 'a', 'an', 'to'}
        query_tokens = [token.lower() for token in re.findall(r'\w+', query) if token.lower() not in stop_words]
        
        # Fuzzy match node types from nodes.json
        nodes_context = self.context.get("nodes.json", {})
        for node_type in nodes_context:
            if node_type.lower() == "metadata":
                continue
            if difflib.get_close_matches(node_type.lower(), query_tokens, cutoff=0.8):
                analysis["nodes"].append(node_type)
                
        # Fuzzy match relationship types from relationships.json
        rel_context = self.context.get("relationships.json", {})
        for rel_type in rel_context:
            if rel_type.lower() == "metadata":
                continue
            normalized = rel_type.replace("_", " ").lower()
            if difflib.get_close_matches(normalized, query_tokens, cutoff=0.8):
                analysis["relationships"].append(rel_type)
        
        # Generic extraction of proper noun phrases (optionally with titles)
        proper_nouns = re.findall(r'\b(?:Dr\.?\s+|Mr\.?\s+|Ms\.?\s+)?[A-Z][a-zA-Z]+\b', query)
        # Remove duplicates while preserving original casing
        proper_nouns = list({pn.lower(): pn for pn in proper_nouns}.values())
        
        for pn in proper_nouns:
            # Skip if already used as a constraint value
            if any(pn.lower() == v.lower() for v in analysis["constraints"].values()):
                continue
            if len(analysis["nodes"]) == 1:
                analysis["constraints"][analysis["nodes"][0]] = pn
            elif pn.lower().startswith("dr"):
                if "Doctor" in analysis["nodes"]:
                    analysis["constraints"]["Doctor"] = pn
                elif analysis["nodes"]:
                    analysis["constraints"][analysis["nodes"][0]] = pn
            else:
                if analysis["nodes"]:
                    analysis["constraints"][analysis["nodes"][0]] = pn
        if DEBUG:
            print(f"[DEBUG] Rule-based analysis (generic): {analysis}")
        return analysis

    def build_query_context(self, analysis: Dict) -> Dict:
        if DEBUG:
            print("[DEBUG] Building query context")
        query_context = {
            'matched_nodes': self.context.get('nodes.json', {}),
            'matched_relationships': self.context.get('relationships.json', {}),
            'analysis': analysis
        }
        if DEBUG:
            print(f"[DEBUG] Using nodes: {list(query_context['matched_nodes'].keys())}")
            print(f"[DEBUG] Using relationships: {list(query_context['matched_relationships'].keys())}")
        return query_context

    def _clean_query(self, candidate: str) -> str:
        """Remove markdown formatting fences from candidate query."""
        candidate = candidate.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            # Remove the first line if it starts with "```"
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            # Remove the last line if it starts with "```"
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
        return candidate

    def generate_cypher_query(self, query: str, context: Dict, neo4j_helper: Neo4jHelper) -> str:
        """
        Iteratively generate a Cypher query using the LLM and verify it by executing it.
        If execution fails, include the error message in the next prompt for refinement.
        """
        attempts = 0
        extra_info = ""
        while attempts < self.max_cypher_attempts:
            prompt = f"""
You are an expert in generating Cypher queries for Neo4j based on natural language queries.
The analysis of the query is:
{json.dumps(context.get('analysis', {}), indent=2)}

The database schema is:
Nodes: {json.dumps(context.get('matched_nodes', {}), indent=2)}
Relationships: {json.dumps(context.get('matched_relationships', {}), indent=2)}

The original natural language query is:
"{query}"

{("Additional context: " + extra_info) if extra_info else ""}

Based on the above, generate a complete and correct Cypher query that would answer the query.
Ensure that you include all necessary MATCH patterns and WHERE constraints.
Output only the Cypher query.
"""
            response = self._call_llm(prompt)
            candidate = response.get('response', '').strip() if response.get('response') else ""
            candidate = self._clean_query(candidate)
            if DEBUG:
                print(f"[DEBUG] Attempt {attempts+1}: Generated Cypher query:\n{candidate}")
            # Try executing the candidate query to verify it
            try:
                _ = neo4j_helper.run_query(candidate)
                if DEBUG:
                    print(f"[DEBUG] Cypher query executed successfully.")
                return candidate
            except Exception as e:
                extra_info = f"Error encountered when executing the query: {str(e)}. Please refine the query."
                if DEBUG:
                    print(f"[DEBUG] Attempt {attempts+1} failed with error: {str(e)}")
            attempts += 1
        raise Exception("Failed to generate a valid Cypher query after several attempts.")

    def format_results(self, query: str, results: List[Dict]) -> str:
        if DEBUG:
            print("[DEBUG] Formatting results with LLM")
        prompt = f"""
You are a graph database interpreter. Here is the executed Cypher query:
{query}

The raw results are:
{json.dumps(results, indent=2)}

Provide a concise and insightful summary of the results, including:
- A clear interpretation of what the query was intended to retrieve,
- Key findings from the results,
- Potential next steps or missing elements.

Output the summary as plain text.
"""
        response = self._call_llm(prompt)
        if not response or response.get('response') == '_USE_RULES_':
            if DEBUG:
                print("[DEBUG] LLM failed to format results, using fallback summary")
            return f"Query executed, but no detailed summary available. Raw results: {json.dumps(results, indent=2)}"
        formatted_response = response.get('response', '').strip()
        if DEBUG:
            print(f"[DEBUG] Formatted response: {formatted_response}")
        return formatted_response

class QueryResult:
    def __init__(self, natural_query: str, analysis: Dict, cypher_query: str, results: List[Dict], formatted_response: str):
        self.natural_query = natural_query
        self.analysis = analysis
        self.cypher_query = cypher_query
        self.results = results
        self.formatted_response = formatted_response
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "natural_query": self.natural_query,
            "analysis": self.analysis,
            "cypher_query": self.cypher_query,
            "results": self.results,
            "formatted_response": self.formatted_response
        }

def save_results(results: List[Dict], filepath: str = 'resources/demo/demo_results.json'):
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({"queries": results}, f, indent=2, ensure_ascii=False)
        if DEBUG:
            print(f"[DEBUG] Results saved to {filepath}")
    except Exception as e:
        print(f"[ERROR] Failed to save results: {e}")

def process_query(query: str, neo4j_helper: Neo4jHelper, llm_agent: LLMAgent, logger: logging.Logger) -> Dict:
    if DEBUG:
        print("\n" + "=" * 80)
        print(f"[DEBUG] Processing query: {query}")
        print("=" * 80)
    try:
        analysis = llm_agent.analyze_query(query)
        context = llm_agent.build_query_context(analysis)
        cypher_query = llm_agent.generate_cypher_query(query, context, neo4j_helper)
        results = neo4j_helper.run_query(cypher_query)
        formatted_results = llm_agent.format_results(query, results)
        query_result = QueryResult(query, analysis, cypher_query, results, formatted_results)
        print("\nResults:")
        print(formatted_results)
        print("\n" + "=" * 80 + "\n")
        return query_result.to_dict()
    except Exception as e:
        logger.error(f"Error in query processing pipeline: {str(e)}")
        if DEBUG:
            import traceback
            print(f"[DEBUG] Full error traceback:\n{traceback.format_exc()}")
        raise

def main():
    if DEBUG:
        print("[DEBUG] Starting main execution")
    logger = setup_logging()
    config = load_config('config.ini')
    neo4j_helper = Neo4jHelper(
        config.get('demo_neo4j', 'uri'),
        config.get('demo_neo4j', 'user'),
        config.get('demo_neo4j', 'password')
    )
    llm_agent = LLMAgent(config)
    test_queries = [
        "Which doctor recommends Beta Blockers?",
        "Who are the patients treated with Beta Blockers?",
        "Which patients are diagnosed with diseases treated by Dr. Bennett?",
        "What are the ages and genders of all patients diagnosed with Asthma?",
        "Which treatments are most commonly recommended by doctors?",
        "Which diseases are most commonly diagnosed among patients?"
    ]
    all_results = []
    for i, query in enumerate(test_queries, 1):
        if DEBUG:
            print(f"\n[DEBUG] Processing query {i}/{len(test_queries)}")
        try:
            result = process_query(query, neo4j_helper, llm_agent, logger)
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error processing query '{query}': {str(e)}")
    save_results(all_results)
    if DEBUG:
        print("[DEBUG] Main execution completed")
    neo4j_helper.close()

if __name__ == "__main__":
    main()
