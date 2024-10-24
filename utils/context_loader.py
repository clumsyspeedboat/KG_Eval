# utils/context_loader.py

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_context(folder_path, as_dict=False, max_chars=5000):
    """
    Load and combine contents from all JSON files in the specified folder.

    Args:
        folder_path (str or Path): Path to the folder containing JSON files.
        as_dict (bool): Return data as a combined dictionary if True.
        max_chars (int): Maximum number of characters to include in the context.

    Returns:
        str or dict: Combined content extracted from all JSON files.
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"The folder {folder_path} does not exist or is not a directory.")

    combined_data = {} if as_dict else ""
    for file_path in folder.glob("*.json"):
        try:
            with file_path.open('r', encoding='utf-8-sig') as f:
                data = json.load(f)
                if as_dict:
                    combined_data.update(data)
                else:
                    # Extract key information instead of full data
                    key_info = extract_key_info(data)
                    combined_data += f"### {file_path.stem}\n{key_info}\n"
            logger.info(f"Loaded {file_path.name}")
        except Exception as e:
            logger.error(f"Failed to read {file_path.name}: {e}")

    # Truncate combined_data if it exceeds max_chars
    if not as_dict and len(combined_data) > max_chars:
        combined_data = combined_data[:max_chars] + "\n[Context truncated due to size]"
    return combined_data

def extract_key_info(data):
    """
    Extract key information from the data to include in the context.

    Args:
        data (list or dict): The JSON data loaded from the file.

    Returns:
        str: A string representation of the key information.
    """
    if isinstance(data, list):
        key_info = ""
        for item in data:
            if isinstance(item, dict):
                # Include selected keys from each item
                key_info += ", ".join(f"{k}: {v}" for k, v in item.items()) + "\n"
            else:
                key_info += str(item) + "\n"
        return key_info
    elif isinstance(data, dict):
        # Include selected keys from the dict
        return ", ".join(f"{k}: {v}" for k, v in data.items())
    else:
        return str(data)
