# enhanced_admin_functions.py
"""
Enhanced admin functions to support multiple personnel assignments.
"""
import streamlit as st
import logging
import pandas as pd
import json
import re
from config import settings
from functions import db as db_manager

logger = logging.getLogger(__name__)

def save_multi_personnel_assignment(entry_id, personnel_list, event_type):
    """
    Save a manual assignment with multiple personnel.
    
    Args:
        entry_id: Unique identifier for the calendar entry
        personnel_list: List of personnel names to assign (or single name)
        event_type: The event type to assign
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Handle both list and single string inputs
        if isinstance(personnel_list, str):
            # Single string - convert to list
            personnel_list = [personnel_list]
        elif not isinstance(personnel_list, list):
            # If not a string or list, convert to string then list
            personnel_list = [str(personnel_list)]
        
        # Filter out empty strings and None values
        personnel_list = [p for p in personnel_list if p and str(p).strip()]
        
        # Default to Unknown if list is empty after filtering
        if not personnel_list:
            personnel_list = ["Unknown"]
        
        # Log the assignment
        logger.info(f"Multi-personnel assignment: Entry ID: {entry_id}, Personnel: {personnel_list}, Event Type: {event_type}")
        
        # Save to database if enabled
        if settings.DB_ENABLED:
            conn = db_manager.get_db_connection()
            if conn:
                # Format personnel as a JSON array string
                personnel_json = json.dumps(personnel_list)
                
                # Update the database
                with conn.cursor() as cursor:
                    query = f"""
                    UPDATE {settings.DB_TABLE_PROCESSED_DATA}
                    SET personnel = %s::jsonb,
                        extracted_event_type = %s,
                        processing_status = 'manual_assigned'
                    WHERE uid = %s
                    """
                    cursor.execute(query, (personnel_json, event_type, entry_id))
                
                conn.commit()
                conn.close()
                logger.info(f"Database updated for entry {entry_id}")
                
                # Also update session state if available
                update_session_state_multi_personnel(entry_id, personnel_list, event_type)
                
                return True
        else:
            # If no database, update only in session state
            return update_session_state_multi_personnel(entry_id, personnel_list, event_type)
    
    except Exception as e:
        logger.error(f"Error saving multi-personnel assignment: {e}")
        return False

def update_session_state_multi_personnel(entry_id, personnel_list, event_type):
    """
    Update entry in session state dataframes with multiple personnel.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check different possible dataframes in session state
        dataframes_to_check = [
            'normalized_df',
            'llm_processed_df',
            'analysis_ready_df'
        ]
        
        updated = False
        
        for df_name in dataframes_to_check:
            if df_name in st.session_state and st.session_state[df_name] is not None:
                df = st.session_state[df_name]
                
                # Find the row with matching entry_id
                mask = df['uid'] == entry_id
                if mask.any():
                    # Update the values
                    df.loc[mask, 'extracted_personnel'] = [personnel_list]  # Wrap in list for consistency
                    df.loc[mask, 'extracted_event_type'] = event_type
                    
                    # If analysis_ready_df, also update assigned_personnel
                    if df_name == 'analysis_ready_df' and 'assigned_personnel' in df.columns:
                        df.loc[mask, 'assigned_personnel'] = personnel_list
                    
                    updated = True
                    logger.info(f"Updated entry {entry_id} in session state {df_name}")
        
        return updated
    
    except Exception as e:
        logger.error(f"Error updating session state data with multiple personnel: {e}")
        return False

def parse_personnel_pattern(summary):
    """
    Parse personnel patterns like "CS/SS Unity" to extract personnel initials.
    
    Args:
        summary: The calendar event summary text
        
    Returns:
        list: List of personnel initials found or empty list if none found
    """
    # Regular expression to find patterns like "CS/SS", "CS/KD/JW", etc.
    pattern = r'\b([A-Z]{1,3})\/([A-Z]{1,3}(?:\/[A-Z]{1,3})*)\b'
    
    matches = re.findall(pattern, summary)
    
    if not matches:
        return []
    
    initials = []
    for match in matches:
        # First initial from the match
        initials.append(match[0])
        
        # Remaining initials split by '/'
        remaining = match[1].split('/')
        initials.extend(remaining)
    
    return initials

def match_initials_to_personnel(initials, canonical_names):
    """
    Match initials to full personnel names from the canonical list.
    
    Args:
        initials: List of initials to match
        canonical_names: List of canonical personnel names
        
    Returns:
        list: List of matched personnel names or empty list if no matches
    """
    matched_personnel = []
    
    for initial in initials:
        for name in canonical_names:
            # Extract initials from the name (assuming format "First Last")
            name_parts = name.split()
            if len(name_parts) >= 2:
                first_initial = name_parts[0][0].upper()
                last_initial = name_parts[-1][0].upper()
                
                # Check if initials match (both letters must be present in the initial)
                if len(initial) == 2 and first_initial == initial[0] and last_initial == initial[1]:
                    matched_personnel.append(name)
                    break
    
    return matched_personnel

def extract_personnel_from_pattern(summary, canonical_names):
    """
    Extract personnel from patterns like "CS/SS Unity" using canonical names.
    
    Args:
        summary: The calendar event summary text
        canonical_names: List of canonical personnel names
        
    Returns:
        list: List of personnel names extracted from the pattern
    """
    initials = parse_personnel_pattern(summary)
    
    if not initials:
        return []
    
    return match_initials_to_personnel(initials, canonical_names)
