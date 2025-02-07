# utils/json_loader.py
import json
import logging
from pathlib import Path
from typing import Union, Dict, List

logger = logging.getLogger(__name__)

def load_json_file(filepath: str) -> Union[Dict, List]:
    """
    Load data from a JSON file.

    Args:
        filepath (str): Path to the JSON file

    Returns:
        Union[Dict, List]: Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If invalid JSON
    """
    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded: {Path(filepath).name}")
        return data
    except FileNotFoundError as e:
        logger.error(f"File not found: {filepath}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        raise
