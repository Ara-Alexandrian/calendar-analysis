"""
Smart extractor module that leverages the optimized batch processor for efficient LLM extraction.
This module provides an enhanced interface for extracting personnel information from calendar events
using optimal distribution between MCP (RTX 4090) and Ollama (dual RTX 3090s) servers.
"""
import logging
import time
import pandas as pd
import streamlit as st
from typing import List, Dict, Any, Optional

from config import settings
from functions import config_manager
from .client import get_llm_client, is_llm_ready
from .smart_batch import BatchProcessor

# Configure logging
logger = logging.getLogger(__name__)

def run_smart_extraction(df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
    """
    Run optimized LLM extraction using the smart batch processor.
    This function leverages both the RTX 4090 (MCP) and dual RTX 3090s (Ollama)
    to efficiently process calendar events.
    
    Args:
        df: DataFrame with a 'summary' column containing event summaries
        progress_callback: Optional callback function for progress updates
        
    Returns:
        DataFrame with an added 'extracted_personnel' column
    """
    if not is_llm_ready():
        logger.error("LLM client is not available. Cannot perform extraction.")
        if st:
            st.error("LLM service is not available. Please check your Ollama and MCP server connections.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        return df
        
    if 'summary' not in df.columns:
        logger.error("'summary' column not found in DataFrame. Cannot extract.")
        if st:
            st.error("Input data is missing the 'summary' column.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        return df
        
    # Get canonical names from configuration
    _, _, canonical_names = config_manager.get_personnel_details()
    
    if not canonical_names:
        logger.error("Cannot run extraction: No canonical names found in configuration.")
        if st:
            st.error("Personnel configuration is empty. Please configure personnel in the Admin page.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        return df
        
    # Create a streamlit progress indicator if available
    progress_bar = None
    if st:
        st.info(f"Starting optimized LLM extraction for {len(df)} events using dual GPU setup...")
        progress_bar = st.progress(0, text="Initializing extraction...")
        
    # Define progress callback for streamlit updates
    def update_progress(completed, total):
        if progress_bar:
            progress_percent = min(1.0, completed / max(1, total))
            progress_bar.progress(progress_percent, text=f"Processing event {completed}/{total}...")
            
    # Use the provided callback or our streamlit-based one
    callback = progress_callback or update_progress
    
    # Initialize the smart batch processor with optimized settings for dual GPU setup
    # Default batch size is 8, but the processor will auto-tune based on performance
    processor = BatchProcessor(
        batch_size=getattr(settings, "LLM_BATCH_SIZE", 8),
        max_workers=getattr(settings, "LLM_MAX_WORKERS", 12),
        progress_callback=callback
    )
    
    start_time = time.time()
    
    try:
        # Process the dataframe with smart batching
        result_df = processor.process_dataframe(
            df, 
            summary_col="summary",
            canonical_names=canonical_names
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Smart extraction completed in {elapsed_time:.2f} seconds")
        
        if progress_bar:
            progress_bar.progress(1.0, text="Extraction Complete!")
            st.success(f"Extraction completed in {elapsed_time:.2f} seconds")
            
        return result_df
        
    except Exception as e:
        logger.error(f"Error during smart extraction: {e}")
        if st:
            st.error(f"Extraction failed: {e}")
            
        # Fall back to the traditional extraction if smart extraction fails
        logger.info("Falling back to traditional extraction method")
        if st:
            st.warning("Falling back to traditional extraction method")
            
        from .extractor import run_llm_extraction_parallel
        return run_llm_extraction_parallel(df)
