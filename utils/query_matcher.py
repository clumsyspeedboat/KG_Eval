# utils/query_matcher.py

import difflib
import logging
import re
from collections import defaultdict
from rapidfuzz import fuzz, process  # Uncomment this import

logger = logging.getLogger(__name__)

def extract_keywords(text: str) -> set:
    """
    Extract keywords from a given text by removing stopwords and punctuation.

    Args:
        text (str): The input text from which to extract keywords.

    Returns:
        set: A set of extracted keywords.
    """
    # Define a set of common stopwords
    stopwords = {
        'how', 'many', 'are', 'the', 'in', 'with', 'is', 'of', 'a', 'an',
        'what', 'list', 'all', 'to', 'for', 'and', 'or', 'on', 'by', 'from',
        # Add metaproteomics-specific stopwords
        'protein', 'peptide', 'sequence', 'mass', 'spectra', 'abundance',
        'identification', 'quantification'
    }
    # Use regex to find words, excluding punctuation
    words = re.findall(r'\b\w+\b', text.lower())
    # Remove stopwords and return unique keywords
    keywords = set(word for word in words if word not in stopwords)
    return keywords

def find_best_query_match_enhanced(user_query: str, predefined_queries: list) -> dict:
    """
    Enhanced method to find the best matching predefined query based on keyword overlap and similarity.

    Args:
        user_query (str): The user's natural language query.
        predefined_queries (list): List of predefined query dictionaries.

    Returns:
        dict: The best matching query dictionary or an empty dict if no match is found.
    """
    try:
        if not predefined_queries:
            logger.warning("No predefined queries available for matching.")
            return {}

        # Extract and filter valid predefined queries
        valid_queries = [
            q for q in predefined_queries
            if isinstance(q, dict) and 'description' in q and isinstance(q['description'], str)
        ]

        if not valid_queries:
            logger.warning("No valid predefined queries with 'description' found.")
            return {}

        user_keywords = extract_keywords(user_query)
        best_match = None
        max_overlap = 0
        highest_similarity = 0.0

        for q in valid_queries:
            description = q['description']
            desc_keywords = extract_keywords(description)
            overlap = len(user_keywords.intersection(desc_keywords))

            # Calculate similarity ratio
            similarity = difflib.SequenceMatcher(None, user_query.lower(), description.lower()).ratio()

            # Update best match based on overlap and similarity
            if overlap > max_overlap or (overlap == max_overlap and similarity > highest_similarity):
                best_match = q
                max_overlap = overlap
                highest_similarity = similarity

        # Define a minimum threshold for overlap and similarity to consider a match
        MIN_OVERLAP = 1  # At least one keyword overlap
        MIN_SIMILARITY = 0.3  # Similarity ratio threshold

        if best_match and max_overlap >= MIN_OVERLAP and highest_similarity >= MIN_SIMILARITY:
            logger.info(f"Best match found: '{best_match['description']}' with overlap {max_overlap} and similarity {highest_similarity:.2f}")
            return best_match

        logger.info("No suitable match found based on enhanced matching criteria.")
        return {}
    except Exception as e:
        logger.error(f"Error in enhanced query matching: {e}")
        return {}

def find_best_query_match_rapidfuzz(user_query: str, predefined_queries: list, threshold: int = 40) -> dict:
    """
    Find the best matching predefined query using RapidFuzz for better performance and flexibility.

    Args:
        user_query (str): The user's natural language query.
        predefined_queries (list): List of predefined query dictionaries.
        threshold (int): Minimum score to consider a match (0-100).

    Returns:
        dict: The best matching query dictionary or an empty dict if no match is found.
    """
    try:
        if not predefined_queries:
            logger.warning("No predefined queries available for matching.")
            return {}

        # Extract and filter valid predefined queries
        valid_queries = [
            q for q in predefined_queries
            if isinstance(q, dict) and 'description' in q and isinstance(q['description'], str)
        ]

        if not valid_queries:
            logger.warning("No valid predefined queries with 'description' found.")
            return {}

        descriptions = [q['description'] for q in valid_queries]
        matches = process.extract(user_query, descriptions, scorer=fuzz.token_set_ratio, limit=5)

        # Filter matches based on threshold
        filtered_matches = [match for match in matches if match[1] >= threshold]

        if not filtered_matches:
            logger.info("No matches found above the threshold.")
            return {}

        # Select the best match
        best_match_description, best_match_score, _ = filtered_matches[0]
        best_match = next(q for q in valid_queries if q['description'] == best_match_description)

        logger.info(f"Best match found: '{best_match_description}' with score {best_match_score}")
        return best_match

    except Exception as e:
        logger.error(f"Error finding best query match with RapidFuzz: {e}")
        return {}

def find_best_query_match_with_tags(user_query: str, predefined_queries: list) -> dict:
    """
    Find the best matching predefined query based on keyword tags.

    Args:
        user_query (str): The user's natural language query.
        predefined_queries (list): List of predefined query dictionaries.

    Returns:
        dict: The best matching query dictionary or an empty dict if no match is found.
    """
    try:
        if not predefined_queries:
            logger.warning("No predefined queries available for matching.")
            return {}

        # Extract and filter valid predefined queries
        valid_queries = [
            q for q in predefined_queries
            if isinstance(q, dict) and 'tags' in q and isinstance(q['tags'], list)
        ]

        if not valid_queries:
            logger.warning("No predefined queries with 'tags' found.")
            return {}

        user_keywords = extract_keywords(user_query)
        best_match = None
        max_overlap = 0

        for q in valid_queries:
            tags = set(map(str.lower, q['tags']))
            overlap = len(user_keywords.intersection(tags))

            if overlap > max_overlap:
                best_match = q
                max_overlap = overlap

        # Define a minimum overlap to consider a match
        MIN_OVERLAP = 1  # At least one tag overlap

        if best_match and max_overlap >= MIN_OVERLAP:
            logger.info(f"Best match based on tags: '{best_match['description']}' with tag overlap {max_overlap}")
            return best_match

        logger.info("No suitable match found based on tag overlap.")
        return {}
    except Exception as e:
        logger.error(f"Error in tag-based query matching: {e}")
        return {}
