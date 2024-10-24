# utils/query_matcher.py

import difflib
import logging

logger = logging.getLogger(__name__)

def find_best_query_match(user_query, predefined_queries, cutoff=0.6):
    """
    Find the best matching predefined query based on the user's input.

    Args:
        user_query (str): The user's natural language query.
        predefined_queries (dict): A dictionary of predefined queries.
        cutoff (float): The similarity threshold for matching.

    Returns:
        dict or None: The best matching query data or None if no good match is found.
    """
    descriptions = [v['description'] for v in predefined_queries['queries'].values()]
    matches = difflib.get_close_matches(user_query, descriptions, n=1, cutoff=0.5)
    if matches:
        best_match_description = matches[0]
        for query_data in predefined_queries['queries'].values():
            if query_data['description'] == best_match_description:
                return query_data
    return None
