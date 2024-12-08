# utils/context_loader.py

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_context(folder_path: str, as_dict: bool = False) -> list:
    """
    Load all JSON files from a folder.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        as_dict (bool): If True, return a list of dictionaries.

    Returns:
        list: List of data from JSON files.
    """
    try:
        folder = Path(folder_path)
        data = []
        for file in folder.glob("*.json"):
            with open(file, 'r', encoding='utf-8-sig') as f:
                content = json.load(f)
                if as_dict:
                    data.extend(content)
                else:
                    data.append(content)
            logger.info(f"Loaded {file.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading context from {folder_path}: {e}")
        raise
