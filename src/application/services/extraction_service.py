"""
Calendar event extraction service.

This module provides the service for extracting information from calendar events
using LLM technology.
"""
import json
import logging
import time
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import streamlit as st

from src.infrastructure.llm import get_llm_client
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class ExtractionService:
    """
    Service for extracting information from calendar events.
    
    This service handles the extraction of personnel names and event types
    from calendar event descriptions using LLM technology.
    """
    
    def __init__(self, canonical_names: Optional[List[str]] = None):
        """
        Initialize the extraction service.
        
        Args:
            canonical_names (Optional[List[str]]): List of canonical personnel names.
                                                 If None, loads from configuration.
        """
        self.llm_client = get_llm_client()
        self.canonical_names = canonical_names or self._load_canonical_names()
        
        # Define valid event types from settings
        self.valid_event_types = getattr(settings, "VALID_EVENT_TYPES", [])
        
    def _load_canonical_names(self) -> List[str]:
        """
        Load canonical names from configuration.
        
        Returns:
            List[str]: List of canonical personnel names.
        """
        try:
            from functions import config_manager
            personnel_config = config_manager.load_personnel_config()
            return [person["name"] for person in personnel_config.get("personnel", [])]
        except Exception as e:
            logger.error(f"Error loading canonical names: {e}")
            return []
    
    def extract_from_single_event(self, event_description: str) -> Dict[str, Any]:
        """
        Extract information from a single calendar event.
        
        Args:
            event_description (str): The calendar event description.
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted information.
        """
        if not event_description or not isinstance(event_description, str) or len(event_description.strip()) == 0:
            logger.debug("Skipping extraction for empty or non-string event description.")
            return {"personnel": ["Unknown"], "event_type": "Unknown"}

        if not self.llm_client:
            logger.error("LLM client is None in extract_from_single_event.")
            return {"personnel": ["Unknown_Error"], "event_type": "Unknown_Error"}
        
        # Construct the prompt
        prompt = self._build_extraction_prompt(event_description)
        
        # Call the LLM
        try:
            logger.debug(f"Attempting LLM extraction for event: '{event_description[:50]}...'")
            start_time = time.time()
            
            response = self.llm_client.extract_all(event_description)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"LLM extraction completed in {elapsed_time:.2f} seconds")
            
            # Process and validate the response
            personnel = response.get("personnel", [])
            event_type = response.get("event_type", "Unknown")
            
            # Additional validation can be added here
            
            return {"personnel": personnel, "event_type": event_type}
        except Exception as e:
            logger.error(f"Error during LLM extraction: {e}")
            return {"personnel": ["Error"], "event_type": "Error", "error": str(e)}
    
    def extract_from_dataframe(self, df: pd.DataFrame, description_column: str) -> pd.DataFrame:
        """
        Extract information from all events in a dataframe.
        
        Args:
            df (pd.DataFrame): Dataframe containing calendar events.
            description_column (str): Name of the column containing event descriptions.
            
        Returns:
            pd.DataFrame: Dataframe with additional columns for extracted information.
        """
        if description_column not in df.columns:
            logger.error(f"Column {description_column} not found in dataframe.")
            return df
        
        # Create progress tracking
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        
        # Initialize new columns
        df["extracted_personnel"] = None
        df["extracted_event_type"] = None
        
        # Determine batch size and max workers based on settings
        batch_size = getattr(settings, "LLM_BATCH_SIZE", 10)
        max_workers = getattr(settings, "LLM_MAX_WORKERS", 2)
        
        # Get the event descriptions to process
        events = df[description_column].fillna("").tolist()
        total_events = len(events)
        processed_events = 0
        
        # Batch processing with ThreadPoolExecutor
        logger.info(f"Starting extraction for {total_events} events with {max_workers} workers")
        
        result_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks in batches
            for i in range(0, total_events, batch_size):
                batch = events[i:i+batch_size]
                futures = {executor.submit(self.extract_from_single_event, event): idx 
                          for idx, event in enumerate(batch, start=i)}
                
                # Process completed tasks
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        result_data.append((idx, result))
                    except Exception as e:
                        logger.error(f"Error processing event at index {idx}: {e}")
                        result_data.append((idx, {"personnel": ["Error"], "event_type": "Error"}))
                    
                    # Update progress
                    processed_events += 1
                    progress_bar.progress(min(processed_events / total_events, 1.0))
            
        # Sort results by index and update dataframe
        result_data.sort(key=lambda x: x[0])
        
        for idx, result in result_data:
            if idx < len(df):
                df.at[idx, "extracted_personnel"] = json.dumps(result.get("personnel", []))
                df.at[idx, "extracted_event_type"] = result.get("event_type", "Unknown")
        
        # Clean up progress display
        progress_container.empty()
        
        logger.info(f"Completed extraction for {total_events} events")
        return df
    
    def _build_extraction_prompt(self, event_description: str) -> str:
        """
        Build the prompt for LLM extraction.
        
        Args:
            event_description (str): The calendar event description.
            
        Returns:
            str: The formatted prompt for the LLM.
        """
        return f"""
        Your task is to analyze the provided calendar event summary and extract two pieces of information:
        1.  **Personnel Names:** Identify ONLY the canonical physicist names from the provided list that are mentioned or clearly referenced in the summary. Consider variations (initials, last names) but map them back EXACTLY to a name in the list. If multiple names are found, include all. If none are found, use an empty list `[]`.
        2.  **Event Type:** Identify the primary service or event type described in the summary. Choose the BEST match ONLY from the 'Valid Event Types' list provided below. If no suitable match is found in the list, return "Unknown".

        Known Canonical Physicist Names:
        {json.dumps(self.canonical_names, indent=2)}

        Valid Event Types (Choose ONLY from this list):
        {json.dumps(self.valid_event_types, indent=2)}

        Event Summary:
        "{event_description}"

        IMPORTANT: Respond ONLY with a single, valid JSON object containing two keys: "personnel" (a JSON array of strings) and "event_type" (a JSON string).

        Examples of CORRECT responses:
        {{"personnel": [], "event_type": "Unknown"}}
        {{"personnel": ["D. Solis"], "event_type": "GK Tx SBRT"}}
        {{"personnel": ["C. Chu", "Wood"], "event_type": "SpaceOAR VUE"}}
        {{"personnel": ["G. Pitcher", "E. Chorniak"], "event_type": "WH HDR CYL"}}

        Examples of INCORRECT responses (DO NOT USE THESE FORMATS):
        ["D. Solis"] # Incorrect: Not a JSON object
        {{"names": ["D. Solis"]}} # Incorrect: Wrong key name
        "Post GK" # Incorrect: Not a JSON object

        Absolutely NO surrounding text or explanations. Your entire response must be JUST the JSON object.
        """
