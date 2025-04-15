"""
Simplified LLM extractor module for Calendar Analysis.
This version only uses Ollama for personnel extraction from calendar events.
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

# Import from our simplified Ollama client
from functions.llm_extraction.ollama_client import is_ollama_ready, extract_personnel

# Import settings
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

def is_llm_ready() -> bool:
    """
    Check if the LLM service is ready to use.
    
    Returns:
        bool: True if the service is ready, False otherwise
    """
    return is_ollama_ready()

def process_events_with_llm(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process calendar events with LLM to extract personnel names.
    
    Args:
        events_df: DataFrame containing calendar events with 'summary' column
        
    Returns:
        pd.DataFrame: DataFrame with added 'extracted_personnel' column
    """
    if not is_llm_ready():
        logger.warning("Ollama service not available, skipping extraction")
        # Add an empty extraction column
        events_df['extracted_personnel'] = [["Unknown"]] * len(events_df)
        return events_df
    
    # Load the list of canonical personnel names
    # This is used to help the LLM identify known personnel
    try:
        from functions import config_manager
        _, _, canonical_names = config_manager.load_personnel_config()
    except Exception as e:
        logger.error(f"Error loading personnel config: {e}")
        canonical_names = []
    
    # Process each event
    extracted_personnel_list = []
    for idx, row in events_df.iterrows():
        summary = row.get('summary', '')
        
        if not summary:
            extracted_personnel_list.append(["Unknown"])
            continue
        
        try:
            # Extract personnel using our simplified Ollama client
            personnel = extract_personnel(summary, canonical_names)
            extracted_personnel_list.append(personnel)
        except Exception as e:
            logger.error(f"Error extracting personnel for event {idx}: {e}")
            extracted_personnel_list.append(["Unknown_Error"])
    
    # Add the extracted personnel to the DataFrame
    events_df['extracted_personnel'] = extracted_personnel_list
    
    return events_df
