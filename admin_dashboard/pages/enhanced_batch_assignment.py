# enhanced_batch_assignment.py
"""
Enhanced batch assignment function that supports multiple personnel selection.
This can be integrated into the _manual_assignment.py file.
"""
import streamlit as st
import pandas as pd
import time
import logging
from typing import List, Dict, Any
import json
import sys
import os

# Add parent directory to path to allow importing from project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import required modules - adjust these imports as needed
from config import settings
from functions import db as db_manager
from admin_dashboard.utils.admin_functions import save_manual_assignment

# Set up logging
logger = logging.getLogger(__name__)

def show_enhanced_batch_assignment(df, personnel_data, valid_event_types):
    """
    Allow batch assignment of multiple personnel and event types based on patterns.
    
    This enhanced version supports selecting multiple personnel for batch assignment.
    
    Args:
        df: DataFrame containing calendar entries
        personnel_data: Dictionary containing personnel information
        valid_event_types: List of valid event types
    """
    st.subheader("Enhanced Batch Assignment")
    st.markdown("Assign multiple personnel and event types to multiple entries at once based on patterns.")
    
    # Search pattern input
    pattern = st.text_input(
        "Search Pattern",
        help="Enter a search pattern to find matching entries (e.g., 'GK SIM', 'Dr. Smith')",
        key="enhanced_batch_pattern"
    )
    
    if not pattern:
        st.info("Enter a search pattern to find matching entries.")
        return
    
    # Find matching entries
    matching_df = df[df['summary'].str.contains(pattern, case=False, na=False)].copy()
    
    if matching_df.empty:
        st.warning(f"No entries found matching the pattern '{pattern}'.")
        return
    
    st.write(f"Found {len(matching_df)} entries matching the pattern '{pattern}'.")
    
    # Display matching entries
    with st.expander("View Matching Entries", expanded=True):
        display_cols = ['uid', 'summary', 'start_time', 'extracted_personnel', 'extracted_event_type']
        st.dataframe(matching_df[display_cols].head(10), use_container_width=True)
        
        if len(matching_df) > 10:
            st.info(f"Showing 10 of {len(matching_df)} matching entries.")
    
    # Batch assignment controls
    st.markdown("### Batch Assignment")
    col1, col2 = st.columns(2)
    
    with col1:
        # Multi-personnel selection
        personnel_options = personnel_data["canonical_names"]
        batch_personnel = st.multiselect(
            "Assign Personnel (select multiple if needed)",
            options=personnel_options,
            key="enhanced_batch_personnel"
        )
    
    with col2:
        # Event type selection
        event_type_options = [""] + valid_event_types
        batch_event_type = st.selectbox(
            "Assign Event Type",
            event_type_options,
            key="enhanced_batch_event_type"
        )
    
    # Preview changes
    if batch_personnel or batch_event_type:
        st.markdown("### Preview Changes")
        
        if batch_personnel:
            personnel_display = ", ".join(batch_personnel) if batch_personnel else "None"
            st.write(f"Personnel will be changed to: **{personnel_display}**")
        else:
            st.write("Personnel will remain unchanged")
            
        if batch_event_type:
            st.write(f"Event type will be changed to: **{batch_event_type}**")
        else:
            st.write("Event type will remain unchanged")
        
        # Confirm button
        if st.button("Apply Batch Assignment", key="enhanced_batch_confirm"):
            try:
                if not batch_personnel and not batch_event_type:
                    st.warning("Please select either personnel or event type to assign.")
                    return
                
                success_count = 0
                fail_count = 0
                total_count = len(matching_df)
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each matching entry
                for i, (idx, row) in enumerate(matching_df.iterrows()):
                    entry_id = row['uid']
                    
                    # Determine what to update
                    current_personnel_list = row['extracted_personnel'] if isinstance(row['extracted_personnel'], list) else ['Unknown']
                    current_personnel_str = ", ".join(current_personnel_list) if current_personnel_list else "Unknown"
                    
                    # Use the multi-selected personnel or keep current
                    personnel_to_assign = batch_personnel if batch_personnel else current_personnel_list
                    event_type_to_assign = batch_event_type or row['extracted_event_type']
                    
                    # Skip if nothing is changing
                    if (not batch_personnel and event_type_to_assign == row['extracted_event_type']):
                        progress = (i + 1) / total_count
                        progress_bar.progress(progress)
                        status_text.text(f"Skipping {i+1}/{total_count} (no change)...")
                        continue
                    
                    # Save the assignment with enhanced multi-personnel support
                    if save_multi_personnel_assignment(entry_id, personnel_to_assign, event_type_to_assign):
                        success_count += 1
                    else:
                        fail_count += 1
                    
                    # Update progress
                    progress = (i + 1) / total_count
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{total_count} entries...")
                
                # Final status
                progress_bar.empty()
                status_text.empty()
                if success_count > 0:
                    st.success(f"Batch assignment completed. Successfully updated {success_count} of {total_count} entries.")
                if fail_count > 0:
                    st.warning(f"Batch assignment completed. Failed to update {fail_count} of {total_count} entries.")
                if success_count == 0 and fail_count == 0:
                    st.info("Batch assignment completed. No entries required updates.")
                
                time.sleep(1)
                st.rerun()  # Refresh the page to show updated data
                
            except Exception as e:
                st.error(f"Error applying batch assignment: {e}")
                logger.error(f"Error in batch assignment: {e}")
    else:
        st.info("Select personnel or event type to assign to matching entries.")

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

# Usage example (for demonstration only):
# if __name__ == "__main__":
#     # You can test the function here if needed
#     pass
