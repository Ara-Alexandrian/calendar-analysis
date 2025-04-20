"""
Admin-specific utility functions for the admin dashboard.
"""
import streamlit as st
import logging
import pandas as pd
import json
from config import settings
from functions import db as db_manager

logger = logging.getLogger(__name__)

def save_manual_assignment(entry_id, personnel, event_type):
    """
    Save a manual assignment made by an administrator.
    
    Args:
        entry_id: Unique identifier for the calendar entry
        personnel: The personnel name(s) to assign. Can be a single string or a list of strings.
        event_type: The event type to assign
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert personnel to a list if it's a single string
        personnel_list = personnel if isinstance(personnel, list) else [personnel]
        
        # Filter out empty strings or None values
        personnel_list = [p for p in personnel_list if p and p.strip()]
        
        # Use ["Unknown"] if the list is empty after filtering
        if not personnel_list:
            personnel_list = ["Unknown"]
        
        # Log the assignment
        logger.info(f"Manual assignment by admin '{st.session_state.admin_username}': Entry ID: {entry_id}, Personnel: {personnel_list}, Event Type: {event_type}")
        
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
                update_session_state_data(entry_id, personnel, event_type)
                
                return True
        else:
            # If no database, update only in session state
            return update_session_state_data(entry_id, personnel, event_type)
    
    except Exception as e:
        logger.error(f"Error saving manual assignment: {e}")
        return False

def update_session_state_data(entry_id, personnel, event_type):
    """
    Update entry in session state dataframes.
    
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
                    df.loc[mask, 'extracted_personnel'] = [[personnel]]
                    df.loc[mask, 'extracted_event_type'] = event_type
                    
                    # If analysis_ready_df, also update assigned_personnel
                    if df_name == 'analysis_ready_df' and 'assigned_personnel' in df.columns:
                        df.loc[mask, 'assigned_personnel'] = personnel
                    
                    updated = True
                    logger.info(f"Updated entry {entry_id} in session state {df_name}")
        
        return updated
    
    except Exception as e:
        logger.error(f"Error updating session state data: {e}")
        return False

def get_admin_audit_log():
    """
    Get a log of manual assignments made by administrators.
    
    Returns:
        pd.DataFrame: A dataframe of admin actions
    """
    try:
        if settings.DB_ENABLED:
            conn = db_manager.get_db_connection()
            if conn:
                query = f"""
                SELECT 
                    uid, 
                    summary, 
                    extracted_personnel, 
                    extracted_event_type,
                    modified_by,
                    last_modified
                FROM {settings.DB_TABLE_PROCESSED_DATA}
                WHERE processing_status = 'manual_assigned'
                ORDER BY last_modified DESC
                LIMIT 100
                """
                
                df = pd.read_sql(query, conn)
                conn.close()
                
                return df
        
        # If no database or no results, return empty dataframe
        return pd.DataFrame()
    
    except Exception as e:
        logger.error(f"Error getting admin audit log: {e}")
        return pd.DataFrame()

def reprocess_with_llm(entry_ids):
    """
    Reprocess selected entries with LLM.
    
    Args:
        entry_ids: List of entry IDs to reprocess
        
    Returns:
        bool: True if successful, False otherwise
    """
    # This is a placeholder for future implementation
    logger.info(f"Reprocess with LLM requested for entries: {entry_ids}")
    return False
