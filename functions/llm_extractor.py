"""
Optimized LLM extractor module for Calendar Analysis.
This version uses direct parallel processing optimized for dual RTX 3090s in NVLINK configuration.
"""
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

# Import from our simplified Ollama client for service check
# from functions.llm_extraction.ollama_client import is_ollama_ready # Old import
from src.infrastructure.llm.factory import LLMClientFactory # New import

# Import our sequential processor for event-by-event processing
from functions.llm_extraction.sequential_processor import SequentialProcessor

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
    Optimized for dual RTX 3090s in NVLINK configuration using direct parallel processing.
    
    Args:
        events_df: DataFrame containing calendar events with 'summary' column
        
    Returns:
        pd.DataFrame: DataFrame with added 'extracted_personnel' column
    """
    client = LLMClientFactory.get_client() # Get client
    if not client or not client.is_available(): # Check client and availability
        logger.warning("Ollama service not available, skipping extraction")
        # Add an empty extraction column
        events_df['extracted_personnel'] = [["Unknown"]] * len(events_df)
        return events_df
    
    # Load the list of canonical personnel names
    # This is used to help the LLM identify known personnel
    try:
        from functions import config_manager
        canonical_names = config_manager.get_all_personnel()
        logger.info(f"Loaded {len(canonical_names)} personnel names from config")
    except Exception as e:
        logger.error(f"Error loading personnel config: {e}")
        canonical_names = []
    
    try:
        # Create progress callback for Streamlit
        def update_progress(completed, total):
            if 'st' in globals():
                import streamlit as st
                progress = completed / total if total > 0 else 0
                st.session_state['progress'] = progress
                st.session_state['progress_text'] = f"{completed}/{total} events processed"
                  # Create the sequential processor for one-by-one event processing
        logger.info("Starting sequential processing of calendar events")
        processor = SequentialProcessor(progress_callback=update_progress)
        
        # Process the dataframe sequentially, one event at a time
        start_time = pd.Timestamp.now()
        result_df = processor.process_dataframe(events_df, summary_col="summary", canonical_names=canonical_names)
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()
        
        logger.info(f"Sequential processing complete in {elapsed:.2f} seconds")
        logger.info(f"Processed {len(events_df)} events with LLM extraction")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error in direct parallel LLM extraction: {e}")
        # Add an empty extraction column as fallback
        events_df['extracted_personnel'] = [["Unknown_Error"]] * len(events_df)
        return events_df
