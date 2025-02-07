from neo4j import GraphDatabase
import logging
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import json
from rapidfuzz import fuzz, process
import re

import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class Neo4jAgent:
    """Handles all Neo4j database operations and query management"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.logger = logging.getLogger(__name__)
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.cached_queries = self._load_cached_queries()
        self.logger.info("Loading predefined queries and mappings...")
        self.query_files = self._load_all_query_files()
        self.logger.info(f"Loaded {len(self.query_files)} query patterns")
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.custom_stopwords = {
            'how', 'many', 'what', 'which', 'show', 'list', 'find',
            'get', 'give', 'display', 'tell', 'would', 'could'
        }
        self.stopwords.update(self.custom_stopwords)

    def initialize(self) -> bool:
        """Initialize Neo4j connection and load queries"""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info("Successfully connected to Neo4j")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j connection: {e}")
            return False

    def _load_cached_queries(self) -> Dict:
        """Load predefined queries from query files"""
        try:
            query_path = Path(__file__).parent.parent / "resources" / "queries" / "query_set1.json"
            with open(query_path, 'r', encoding='utf-8') as f:
                return json.load(f).get("queries", {})
        except Exception as e:
            self.logger.error(f"Error loading cached queries: {e}")
            return {}

    def _load_all_query_files(self) -> List[Dict]:
        """Load all query files from resources/queries directory"""
        queries = []
        query_dir = Path(__file__).parent.parent / 'resources' / 'queries'
        
        if not query_dir.exists():
            self.logger.error(f"Query directory not found: {query_dir}")
            return []

        query_files = list(query_dir.glob('*.json'))
        for file in query_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'queries' in data:
                        # Process each query pattern
                        for pattern in data['queries']:
                            if isinstance(pattern, str):
                                # Convert string pattern to dict format
                                query_dict = {
                                    'name': pattern,
                                    'description': pattern.replace('_', ' ').lower(),
                                    'keywords': pattern.lower().split('_'),
                                    'query': data['queries'].get(pattern, '')
                                }
                                queries.append(query_dict)
                            elif isinstance(pattern, dict):
                                queries.append(pattern)
                            
                        self.logger.debug(f"[DEBUG] Loaded {len(queries)} patterns from {file}")
            except Exception as e:
                self.logger.error(f"Error loading query file {file}: {str(e)}")

        return queries

    def update_connection(self, uri: str, user: str, password: str) -> bool:
        """Update Neo4j connection parameters"""
        try:
            if self.driver:
                self.driver.close()
            self.uri = uri
            self.user = user
            self.password = password
            return self.initialize()
        except Exception as e:
            self.logger.error(f"Failed to update connection: {e}")
            return False

    def execute_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """Execute a Cypher query and return results"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record.data() for record in result]
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            raise

    def _extract_cypher_query(self, pattern: Dict) -> str:
        """Extract Cypher query from pattern with proper handling of nested structures"""
        self.logger.debug(f"[DEBUG] Extracting Cypher query from pattern: {pattern.get('name', 'unnamed')}")
        
        # Handle direct query string
        if isinstance(pattern.get('query'), str):
            return pattern['query']
            
        # Handle nested query object
        query_obj = pattern.get('query', {})
        if isinstance(query_obj, dict):
            # Try to get the actual query string
            if 'query' in query_obj:
                return query_obj['query']
            # Check if cypher key exists (alternative format)
            elif 'cypher' in query_obj:
                return query_obj['cypher']
                
        # Handle cypher key at pattern level
        if 'cypher' in pattern:
            return pattern['cypher']
            
        self.logger.debug("[DEBUG] No valid Cypher query found in pattern")
        return ""

    def _preprocess_text(self, text: str) -> Tuple[List[str], Set[str]]:
        """
        Preprocess text with advanced tokenization and lemmatization
        Returns: (tokenized_text, query_type_indicators)
        """
        # Convert to lowercase and tokenize
        tokens = word_tokenize(text.lower())
        
        # Detect query type indicators
        query_indicators = {
            'count': {'how', 'many', 'count', 'number', 'total'},
            'list': {'what', 'list', 'show', 'display', 'which'},
            'describe': {'describe', 'explain', 'tell'}
        }
        
        query_types = set()
        for token in tokens:
            for qtype, indicators in query_indicators.items():
                if token in indicators:
                    query_types.add(qtype)
        
        # Remove stopwords and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self.stopwords and len(token) > 1
        ]
        
        return tokens, query_types

    def _extract_keywords(self, text: str) -> Dict[str, any]:
        """Enhanced keyword extraction with query type detection"""
        if not text:
            return {'keywords': set(), 'query_types': set(), 'tokens': []}
            
        # Preprocess text
        tokens, query_types = self._preprocess_text(text)
        
        # Extract keywords and their variations
        keywords = set(tokens)
        variations = set()
        
        for word in keywords:
            # Add original form
            variations.add(word)
            
            # Add lemmatized forms
            variations.add(self.lemmatizer.lemmatize(word, 'n'))  # noun form
            variations.add(self.lemmatizer.lemmatize(word, 'v'))  # verb form
            
            # Add plural/singular forms
            if word.endswith('s'):
                variations.add(word[:-1])
            else:
                variations.add(word + 's')
        
        return {
            'keywords': variations,
            'query_types': query_types,
            'tokens': tokens
        }

    def find_matching_query(self, user_query: str) -> Optional[Dict]:
        """Enhanced query matching with query type awareness"""
        self.logger.debug(f"[DEBUG] Starting find_matching_query with: {user_query}")
        
        if not self.query_files:
            self.logger.debug("[DEBUG] No query files found")
            return {}

        try:
            extracted_info = self._extract_keywords(user_query)
            user_keywords = extracted_info['keywords']
            query_types = extracted_info['query_types']
            tokens = extracted_info['tokens']
            
            self.logger.debug(f"[DEBUG] Extracted keywords: {user_keywords}")
            self.logger.debug(f"[DEBUG] Query types: {query_types}")
            self.logger.debug(f"[DEBUG] Tokens: {tokens}")

            best_match = None
            best_score = 0
            match_details = []

            # Add specific patterns to boost for count queries
            count_query_patterns = {
                'patient', 'patients', 'remissionpatient', 'remissionpatients'
            }

            for pattern in self.query_files:
                # Preserve original pattern name for display
                original_name = pattern.get('name') if isinstance(pattern, dict) else str(pattern)
                
                if not isinstance(pattern, dict):
                    pattern = {
                        'name': original_name,  # Store original name
                        'description': original_name.replace('_', ' ').lower(),
                        'keywords': original_name.lower().split('_'),
                        'query': self.cached_queries.get(str(pattern), '')
                    }

                # Extract query first to validate pattern
                cypher_query = self._extract_cypher_query(pattern)
                if not cypher_query:
                    continue

                # Preprocess pattern text
                pattern_text = f"{pattern.get('name', '')} {pattern.get('description', '')}"
                pattern_info = self._extract_keywords(pattern_text)
                pattern_keywords = pattern_info['keywords']

                # Calculate various similarity scores
                name_score = fuzz.token_set_ratio(
                    ' '.join(tokens),
                    pattern.get('name', '').replace('_', ' ').lower()
                )
                
                pattern_name = pattern.get('name', '').lower()
                
                # Apply specific scoring rules
                score_multiplier = 1.0
                
                # Boost patterns that match expected output type
                if 'count' in query_types:
                    # Higher boost for patterns containing both "patient" and query keywords
                    if any(p in pattern_name for p in count_query_patterns):
                        score_multiplier *= 2.0
                    # Lower boost for patterns missing patient context
                    else:
                        score_multiplier *= 0.5

                # Calculate keyword overlap with weighted scoring
                keyword_matches = []
                for kw in user_keywords:
                    matches = [
                        fuzz.ratio(kw, p_kw.lower()) 
                        for p_kw in pattern.get('keywords', [])
                    ]
                    if matches:
                        keyword_matches.append(max(matches))
                
                keyword_score = (sum(keyword_matches) / len(keyword_matches)) if keyword_matches else 0

                # Final weighted score calculation
                combined_score = (
                    0.3 * name_score +
                    0.5 * keyword_score + 
                    0.2 * (100 if pattern_name in count_query_patterns else 0)
                ) * score_multiplier

                self.logger.debug(
                    f"[DEBUG] Pattern '{pattern.get('name')}' scores:"
                    f" name={name_score:.2f}, keyword={keyword_score:.2f},"
                    f" multiplier={score_multiplier:.2f}, final={combined_score:.2f}"
                )

                match_details.append({
                    'query_name': pattern.get('name', 'Unknown'),
                    'score': combined_score,
                    'cypher': cypher_query
                })

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = {
                        **pattern,
                        'query': cypher_query,
                        'display_name': original_name,  # Add display name
                        'percentage_score': min(100, combined_score)  # Convert to percentage
                    }

            if best_score >= 30:
                result = {
                    'matched_query': best_match,
                    'match_score': min(100, best_score),  # Cap at 100%
                    'display_name': best_match.get('display_name', best_match.get('name', '')),
                    'considered_matches': [
                        {
                            **match,
                            'score': min(100, match['score'])  # Convert all scores to percentages
                        }
                        for match in sorted(match_details, key=lambda x: x['score'], reverse=True)[:3]
                    ]
                }
                self.logger.debug(f"[DEBUG] Found match: {result['display_name']} with score {result['match_score']}%")
                return result

            self.logger.debug("[DEBUG] No matching query found")
            return {}

        except Exception as e:
            self.logger.error(f"[ERROR] Error in find_matching_query: {str(e)}")
            self.logger.exception(e)
            return {}

    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()

