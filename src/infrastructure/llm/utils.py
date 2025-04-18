"""
Utility functions for LLM extraction.

This module provides helper functions used by various LLM client implementations.
"""
import json
import logging
import re
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def extract_json_from_response(text: str) -> Dict[str, Any]:
    """
    Extract JSON from LLM response text.
    
    Handles different formats that LLMs might return, including:
    - Plain JSON
    - JSON within markdown code blocks
    - JSON with leading/trailing text
    
    Args:
        text (str): The raw response text from an LLM.
        
    Returns:
        Dict[str, Any]: Extracted JSON data as a dictionary.
    """
    # First, try to extract JSON from markdown code blocks
    json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(json_pattern, text)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # If no code block, try to find JSON-like content directly
        # Look for content between curly braces
        json_pattern = r"(\{[\s\S]*\})"
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1).strip()
        else:
            # If still not found, use the whole text (assuming it might be valid JSON)
            json_str = text.strip()
    
    # Try to parse the extracted content as JSON
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from text: {e}")
        logger.debug(f"Problematic text: {text}")
        
        # Attempt to clean the text and try again
        clean_json_str = clean_json_string(json_str)
        try:
            return json.loads(clean_json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON even after cleaning")
            # Return a basic error structure if all attempts fail
            return {"error": "Failed to parse response as JSON", "raw_text": text}

def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string to make it more likely to parse successfully.
    
    Args:
        json_str (str): The potentially invalid JSON string.
        
    Returns:
        str: Cleaned JSON string.
    """
    # Replace single quotes with double quotes
    cleaned = json_str.replace("'", '"')
    
    # Fix common issues with trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix unquoted keys
    def quote_keys(match):
        key = match.group(1)
        return f'"{key}":'
    
    cleaned = re.sub(r'(\w+):', quote_keys, cleaned)
    
    return cleaned
