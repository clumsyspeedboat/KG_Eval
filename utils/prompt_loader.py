# utils/prompt_loader.py

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def load_agent_prompts(prompts_path: str) -> dict:
    """
    Load agent prompts from a JSON file.

    Args:
        prompts_path (str): Path to the agent_prompts.json file.

    Returns:
        dict: Dictionary of agent prompts.
    """
    try:
        with open(prompts_path, 'r', encoding='utf-8-sig') as f:
            prompts = json.load(f)
        logger.info("Agent prompts loaded successfully.")
        return prompts
    except FileNotFoundError as e:
        logger.error(f"Agent prompts file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from agent prompts file: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading agent prompts: {e}")
        raise
