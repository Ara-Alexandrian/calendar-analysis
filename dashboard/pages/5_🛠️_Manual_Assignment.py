"""
Manual assignment page for the admin dashboard.
Allows administrators to manually assign personnel and event types to calendar entries.
Also allows adding new event types on the fly.
"""
import streamlit as st
import pandas as pd
import logging
import time
import json # Import json for handling personnel list
import re # Import re for parsing settings.py
import ast # Import ast for safely parsing dictionary string
import os # Import os for path joining
from typing import List, Tuple # Import List and Tuple for type hinting
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import settings
from functions import db as db_manager
from functions import config_manager
from functions.llm_extraction.normalizer import normalize_extracted_personnel
from dashboard.utils.admin_functions import save_manual_assignment # Updated import path
from src.application.services.extraction_service import ExtractionService # Import ExtractionService

logger = logging.getLogger(__name__)

def add_new_event_type(new_type_name: str):
    """
    Adds a new event type to the EVENT_TYPE_MAPPING in config/settings.py.
    """
    if not new_type_name or not new_type_name.strip():
        st.warning("Please enter a valid event type name.")
        return False

    new_type_name = new_type_name.strip()
    new_key = new_type_name.lower()
    settings_py_path = os.path.join(settings.CONFIG_DIR, 'settings.py') # Define path locally

    try:
        with open(settings_py_path, 'r') as f: # Use local variable
            settings_content = f.read()

        # Find the EVENT_TYPE_MAPPING dictionary definition
        match = re.search(r"EVENT_TYPE_MAPPING\s*=\s*({.*?})", settings_content, re.DOTALL | re.MULTILINE)
        if not match:
            st.error("Could not find EVENT_TYPE_MAPPING dictionary in settings.py.")
            return False

        dict_str = match.group(1)
        # Use ast.literal_eval for safe parsing
        try:
            event_mapping = ast.literal_eval(dict_str)
        except (ValueError, SyntaxError) as e:
            st.error(f"Error parsing EVENT_TYPE_MAPPING in settings.py: {e}")
            logger.error(f"AST parsing error for EVENT_TYPE_MAPPING: {e}\nContent: {dict_str}")
            # Fallback or stricter regex might be needed if comments interfere
            st.info("Manual editing of settings.py might be required if parsing fails.")
            return False


        if new_key in event_mapping:
            st.warning(f"Event type key '{new_key}' (derived from '{new_type_name}') already exists.")
            return False

        # Add the new mapping
        event_mapping[new_key] = new_type_name

        # Reconstruct the dictionary string carefully, preserving formatting as much as possible
        # This is basic, might not perfectly match original style but should be functional
        new_dict_lines = ["{"]
        for k, v in sorted(event_mapping.items()): # Sort for consistency
             # Escape quotes within strings if necessary
            k_repr = repr(k)
            v_repr = repr(v)
            new_dict_lines.append(f"    {k_repr}: {v_repr},")
        new_dict_lines.append("}")
        new_dict_str = "\n".join(new_dict_lines)

        # Replace the old dictionary string with the new one
        new_settings_content = settings_content.replace(dict_str, new_dict_str, 1)

        with open(settings_py_path, 'w') as f: # Use local variable
            f.write(new_settings_content)

        # Clear the event types from session state to force reload
        if 'event_types_cache' in st.session_state:
            del st.session_state.event_types_cache
        
        st.success(f"Successfully added event type '{new_type_name}'.")
        # Clear relevant cache if Streamlit caching is used for settings
        # st.cache_data.clear() # Example if caching was used
        time.sleep(0.5) # Give time for file write
        return True

    except FileNotFoundError:
        st.error(f"Could not find settings file at: {settings_py_path}") # Use local variable
        return False
    except Exception as e:
        st.error(f"An error occurred while adding the event type: {e}")
        logger.error(f"Error adding event type '{new_type_name}': {e}")
        return False

def show_manual_assignment_page():
    """
    Display the manual assignment interface for administrators.
    """
    st.title("✏️ Manual Assignment")
    st.markdown("Manually assign personnel and event types to calendar entries.")

    # --- Add New Event Type Section ---
    with st.expander("Add New Event Type"):
        new_event_type_name = st.text_input("New Event Type Name", key="new_event_type_input")
        if st.button("Add Event Type", key="add_event_type_button"):
            if add_new_event_type(new_event_type_name):
                # No need to clear session state here, rerun handles it.
                st.rerun() # Rerun to reload event types
            else:
                # Keep input field populated if add failed
                pass
    # --- End Add New Event Type Section ---


    # Load data
    df_data = load_data_for_manual_assignment()
    if df_data is None or df_data.empty:
        st.warning("No data available for manual assignment. Please upload and process data first.")
        return

    # Get valid personnel and event types
    personnel_data = load_personnel_data()
    valid_event_types = load_event_types() # This will now load unique types

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Unassigned Entries", "All Entries", "Batch Assignment"])

    with tab1:
        show_unassigned_entries(df_data, personnel_data, valid_event_types)

    with tab2:
        show_all_entries(df_data, personnel_data, valid_event_types)

    with tab3:
        show_batch_assignment(df_data, personnel_data, valid_event_types)

def load_data_for_manual_assignment():
    """
    Load calendar data for manual assignment.
    Prioritizes loading from database if enabled, otherwise from session state.
    """
    try:
        if settings.DB_ENABLED:
            # Load from database
            # st.info("Loading data from database...") # Less verbose
            conn = db_manager.get_db_connection()
            if conn:
                query = f"""
                SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA}
                ORDER BY
                    CASE
                        WHEN personnel = 'Unknown' OR personnel = 'Unknown_Error' THEN 1
                        ELSE 2
                    END,
                    start_time DESC
                """
                df = pd.read_sql(query, conn)
                conn.close()

                # Map DB columns to expected UI columns
                if 'personnel' in df.columns:
                    df['extracted_personnel'] = df['personnel'].apply(lambda x: [x] if isinstance(x, str) else x)
                if 'event_type' in df.columns:
                    df['extracted_event_type'] = df['event_type']

                if not df.empty:
                    # Convert string representation of lists to actual lists
                    # Be more robust against malformed strings
                    def safe_eval(cell):
                        if isinstance(cell, list):
                            return cell
                        try:
                            # Only evaluate if it looks like a list string
                            if isinstance(cell, str) and cell.startswith('[') and cell.endswith(']'):
                                return ast.literal_eval(cell)
                            elif isinstance(cell, str): # Handle single string personnel
                                return [cell]
                        except (ValueError, SyntaxError):
                            logger.warning(f"Could not evaluate personnel string: {cell}")
                            return ['Unknown_Error'] # Mark as error if eval fails
                        return cell # Return original if not string or list-like string

                    df['extracted_personnel'] = df['extracted_personnel'].apply(safe_eval)
                    # Ensure it's always a list, even if None initially
                    df['extracted_personnel'] = df['extracted_personnel'].apply(lambda x: x if isinstance(x, list) else ['Unknown'])


                    return df

        # If database not enabled or no data found, try session state
        if 'normalized_df' in st.session_state and st.session_state.normalized_df is not None:
            return st.session_state.normalized_df

        # Try other session state variables
        if 'llm_processed_df' in st.session_state and st.session_state.llm_processed_df is not None:
            return st.session_state.llm_processed_df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        logger.error(f"Error loading data for manual assignment: {e}")

    return None

def load_personnel_data():
    """
    Load personnel data from configuration.
    """
    try:
        personnel_by_role, role_mapping, canonical_names = config_manager.get_personnel_details()

        # Convert to a format suitable for dropdowns
        personnel_data = {
            "canonical_names": sorted(list(set(canonical_names))), # Ensure unique and sorted
            "by_role": personnel_by_role,
            "role_mapping": role_mapping
        }

        return personnel_data
    except Exception as e:
        st.error(f"Error loading personnel data: {e}")
        logger.error(f"Error loading personnel data: {e}")
        return {"canonical_names": [], "by_role": {}, "role_mapping": {}}

def load_event_types():
    """
    Load unique, sorted valid event types from settings.
    """
    try:
        # Check if we've already cached the event types in this session
        if 'event_types_cache' not in st.session_state:
            # Get event types from settings, ensure uniqueness and sort
            # Reload settings module in case it was changed by add_new_event_type
            import importlib
            importlib.reload(settings)
            all_event_types = list(settings.EVENT_TYPE_MAPPING.values())
            unique_event_types = sorted(list(set(all_event_types)))
            # Cache the result in session state
            st.session_state.event_types_cache = unique_event_types
            return unique_event_types
        else:
            # Return cached event types
            return st.session_state.event_types_cache
    except Exception as e:
        st.error(f"Error loading event types: {e}")
        logger.error(f"Error loading event types: {e}")
        return []

def show_unassigned_entries(df, personnel_data, valid_event_types):
    """
    Display and allow editing of unassigned entries.
    """
    st.subheader("Unassigned Entries")
    st.markdown("These entries have been marked as 'Unknown' or 'Unknown_Error' and need manual assignment.")

    # Filter for unassigned entries
    unassigned_df = df[
        df['extracted_personnel'].apply(lambda x: isinstance(x, list) and (
            'Unknown' in x or 'Unknown_Error' in x
        )) |
        (df['extracted_event_type'] == 'Unknown') |
        (df['extracted_event_type'] == 'Unknown_Error')
    ].copy()

    if unassigned_df.empty:
        st.success("No unassigned entries found! All entries have personnel and event types assigned.")
        return

    st.write(f"Found {len(unassigned_df)} unassigned entries.")

    # Add filtering and sorting options
    st.markdown("---")
    st.markdown("### Filter and Sort")
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search summaries", "", key="unassigned_search")
    with col2:
        date_range = st.date_input(
            "Date Range",
            value=[],
            help="Filter entries by date range",
            key="unassigned_date_range"
        )

    # Apply filters
    filtered_unassigned_df = unassigned_df.copy()

    # Apply search filter
    if search_term:
        filtered_unassigned_df = filtered_unassigned_df[filtered_unassigned_df['summary'].str.contains(search_term, case=False, na=False)]

    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_unassigned_df = filtered_unassigned_df[
            (pd.to_datetime(filtered_unassigned_df['start_time']).dt.date >= start_date) &
            (pd.to_datetime(filtered_unassigned_df['start_time']).dt.date <= end_date)
        ]

    st.write(f"Showing {len(filtered_unassigned_df)} entries after filtering.")

    # Sorting options
    sort_options = {
        "Date (Newest First)": ("start_time", False),
        "Date (Oldest First)": ("start_time", True),
        "Summary (A-Z)": ("summary", True),
        "Summary (Z-A)": ("summary", False),
    }
    selected_sort = st.selectbox("Sort by", list(sort_options.keys()), key="unassigned_sort")
    sort_column, ascending = sort_options[selected_sort]
    filtered_unassigned_df = filtered_unassigned_df.sort_values(by=sort_column, ascending=ascending)

    st.markdown("---")

    # Batch Retry Button for filtered unassigned entries
    if not filtered_unassigned_df.empty:
        if st.button(f"Batch Retry LLM for {len(filtered_unassigned_df)} Filtered Entries"):
            batch_retry_entries(filtered_unassigned_df['uid'].tolist())

    st.markdown("---")

    # Pagination
    items_per_page = st.slider("Entries per page", min_value=5, max_value=50, value=10, step=5, key="unassigned_slider")
    page_count = (len(filtered_unassigned_df) + items_per_page - 1) // items_per_page

    if 'unassigned_page' not in st.session_state:
        st.session_state.unassigned_page = 0

    # Reset page if filters change resulting page count
    if st.session_state.unassigned_page >= page_count:
        st.session_state.unassigned_page = 0


    # Page navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.unassigned_page <= 0, key="unassigned_prev"):
            st.session_state.unassigned_page -= 1
            st.rerun()

    with col2:
        st.write(f"Page {st.session_state.unassigned_page + 1} of {max(1, page_count)}")

    with col3:
        if st.button("Next ➡️", disabled=st.session_state.unassigned_page >= page_count - 1, key="unassigned_next"):
            st.session_state.unassigned_page += 1
            st.rerun()

    # Display entries for current page
    start_idx = st.session_state.unassigned_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_unassigned_df))
    page_df = filtered_unassigned_df.iloc[start_idx:end_idx].copy()

    # Function to handle assignment
    def handle_assignment(entry_uid, personnel, event_type):
        try:
            # entry_id = page_df.loc[entry_index]['uid'] # Use loc with index
            # original_summary = page_df.loc[entry_index]['summary']

            if personnel and event_type:
                # Save assignment
                success = save_manual_assignment(entry_uid, personnel, event_type)

                if success:
                    st.success(f"Successfully assigned personnel '{personnel}' and event type '{event_type}' to entry {entry_uid}.")
                    # Note: Updating the original unassigned_df might be complex with filtering/sorting.
                    # For simplicity, we'll rely on a rerun to refresh the data.
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(f"Failed to save assignment for entry {entry_uid}. Please try again.")
            else:
                st.warning("Please select both personnel and event type.")
        except Exception as e:
            st.error(f"Error assigning personnel for entry {entry_uid}: {e}")
            logger.error(f"Error in handle_assignment for {entry_uid}: {e}")

    # Display each entry with assignment controls
    for i, (idx, row) in enumerate(page_df.iterrows()): # Use original index idx
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            entry_uid = row.get('uid', 'N/A')

            with col1:
                st.markdown(f"**Entry ID:** {entry_uid}")
                st.markdown(f"**Summary:** {row.get('summary', 'N/A')}")
                st.markdown(f"**Date/Time:** {row.get('start_time', 'N/A')}")

                # Current assignments
                current_personnel_list = row.get('extracted_personnel', ['Unknown'])
                if not isinstance(current_personnel_list, list): # Handle potential non-list data
                     current_personnel_list = ['Unknown_Error']
                current_personnel = "Unknown" if current_personnel_list == ['Unknown'] else ", ".join(current_personnel_list)

                current_event_type = row.get('extracted_event_type', 'Unknown')
                st.markdown(f"**Current Personnel:** {current_personnel}")
                st.markdown(f"**Current Event Type:** {current_event_type}")

            with col2:
                # Assignment controls
                st.subheader("Assign")

                # Personnel selection - using multiselect
                personnel_options = personnel_data["canonical_names"]
                selected_personnel = st.multiselect(
                    "Personnel (select multiple if needed)",
                    options=personnel_options,
                    key=f"personnel_{entry_uid}" # Use UID for unique key
                )

                # Event type selection
                event_type_options = [""] + valid_event_types
                selected_event_type = st.selectbox(
                    "Event Type",
                    event_type_options,
                    key=f"event_type_{entry_uid}" # Use UID for unique key
                )

                # Save button
                if st.button("Save Assignment", key=f"save_{entry_uid}"): # Use UID for unique key
                    handle_assignment(entry_uid, selected_personnel, selected_event_type)

                # LLM retry button
                if st.button("Retry LLM", key=f"retry_{entry_uid}"): # Use UID for unique key
                    # Call the retry function for this single entry
                    retry_single_entry(entry_uid)

def retry_single_entry(entry_uid: str):
    """
    Retry LLM extraction for a single entry and update the database.
    Returns True if successful, False otherwise.
    """
    try:
        st.info(f"Retrying LLM extraction for entry UID: {entry_uid}")

        # Load the specific entry from the database
        conn = db_manager.get_db_connection()
        if not conn:
            st.error("Database connection failed.")
            return False

        query = f"SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE uid = %s"
        entry_df = pd.read_sql(query, conn, params=(entry_uid,))
        conn.close()

        if entry_df.empty:
            st.warning(f"Entry with UID {entry_uid} not found.")
            return False

        entry_data = entry_df.iloc[0]
        event_description = entry_data.get('summary', '')

        if not event_description:
            st.warning(f"Entry with UID {entry_uid} has no summary to process.")
            return False

        # Perform LLM extraction
        extraction_service = ExtractionService()
        extracted_info = extraction_service.extract_from_single_event(event_description)

        new_personnel = extracted_info.get("personnel", ["Unknown_Error"])
        new_event_type = extracted_info.get("event_type", "Unknown_Error")

        # Update the database with the new information
        conn = db_manager.get_db_connection()
        if not conn:
            st.error("Database connection failed.")
            return False

        cursor = conn.cursor()
        update_query = f"""
        UPDATE {settings.DB_TABLE_PROCESSED_DATA}
        SET personnel = %s, event_type = %s, processing_status = %s
        WHERE uid = %s
        """
        # Update status based on result
        new_status = 'extracted' if new_event_type != 'Unknown_Error' and new_personnel != ['Unknown_Error'] else 'error'
        cursor.execute(update_query, (json.dumps(new_personnel), new_event_type, new_status, entry_uid))
        conn.commit()
        cursor.close()
        conn.close()

        st.success(f"Successfully retried and updated entry {entry_uid}.")
        return True # Indicate success

    except Exception as e:
        st.error(f"Error retrying entry {entry_uid}: {e}")
        logger.error(f"Error in retry_single_entry for UID {entry_uid}: {e}")
        return False # Indicate failure

def batch_retry_entries(entry_uids: List[str]):
    """
    Retry LLM extraction for a batch of entries and update the database.
    """
    st.info(f"Starting batch retry for {len(entry_uids)} entries...")
    success_count = 0
    fail_count = 0
    total_count = len(entry_uids)

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Use ThreadPoolExecutor for potentially faster batch processing
    # Note: Consider DB connection pooling if this becomes a bottleneck
    max_workers = settings.LLM_MAX_WORKERS # Reuse setting
    results = []

    with st.spinner(f"Processing {total_count} entries with LLM..."):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map UIDs to futures
            future_to_uid = {executor.submit(retry_single_entry_processing, uid): uid for uid in entry_uids}

            for i, future in enumerate(as_completed(future_to_uid)):
                uid = future_to_uid[future]
                try:
                    result = future.result() # result is (success_flag, uid) or None if error in processing func
                    if result and result[0]: # Check if success_flag is True
                        success_count += 1
                    else:
                        fail_count += 1
                    results.append(result) # Store result if needed later
                except Exception as exc:
                    logger.error(f"Batch retry generated an exception for UID {uid}: {exc}")
                    fail_count += 1
                    results.append((False, uid)) # Mark as failed

                # Update progress
                progress = (i + 1) / total_count
                progress_bar.progress(progress)
                status_text.text(f"Processing {i+1}/{total_count} entries...")

    # Final status
    progress_bar.empty()
    status_text.empty()
    if success_count > 0:
        st.success(f"Batch retry completed. Successfully updated {success_count} of {total_count} entries.")
    if fail_count > 0:
        st.warning(f"Batch retry completed. Failed to update {fail_count} of {total_count} entries. Check logs for details.")
    if success_count == 0 and fail_count == 0:
         st.info("Batch retry completed. No entries required processing or updates.")


    time.sleep(1)
    st.rerun() # Refresh the page to show updated data

def retry_single_entry_processing(entry_uid: str) -> Tuple[bool, str]:
    """
    Helper function for batch processing: Retries LLM and updates DB for one entry.
    Returns (success_flag, entry_uid)
    """
    try:
        # Load the specific entry from the database
        conn = db_manager.get_db_connection()
        if not conn: return (False, entry_uid) # Fail early

        query = f"SELECT summary FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE uid = %s"
        # Use cursor directly for single value fetch
        cursor = conn.cursor()
        cursor.execute(query, (entry_uid,))
        result = cursor.fetchone()
        cursor.close()
        conn.close() # Close connection after fetch

        if not result:
            logger.warning(f"Entry with UID {entry_uid} not found during batch processing.")
            return (False, entry_uid)

        event_description = result[0]
        if not event_description:
            logger.warning(f"Entry with UID {entry_uid} has no summary during batch processing.")
            return (False, entry_uid) # Consider this success or failure? False for now.

        # Perform LLM extraction
        # Initialize service here to ensure thread safety if needed, or pass client if safe
        extraction_service = ExtractionService()
        extracted_info = extraction_service.extract_from_single_event(event_description)

        new_personnel = extracted_info.get("personnel", ["Unknown_Error"])
        new_event_type = extracted_info.get("event_type", "Unknown_Error")

        # Update the database with the new information
        conn = db_manager.get_db_connection()
        if not conn: return (False, entry_uid) # Fail early

        cursor = conn.cursor()
        update_query = f"""
        UPDATE {settings.DB_TABLE_PROCESSED_DATA}
        SET personnel = %s, extracted_event_type = %s, processing_status = %s
        WHERE uid = %s
        """
        new_status = 'extracted' if new_event_type != 'Unknown_Error' and new_personnel != ['Unknown_Error'] else 'error'
        cursor.execute(update_query, (json.dumps(new_personnel), new_event_type, new_status, entry_uid))
        conn.commit()
        cursor.close()
        conn.close() # Close connection after update

        return (True, entry_uid) # Indicate success

    except Exception as e:
        logger.error(f"Error processing entry {entry_uid} in batch thread: {e}")
        # Ensure connection is closed if open
        try:
            if conn and not conn.closed:
                conn.close()
        except: pass
        return (False, entry_uid) # Indicate failure


def show_all_entries(df, personnel_data, valid_event_types):
    """
    Display and allow editing of all entries.
    """
    st.subheader("All Entries")
    st.markdown("View and edit assignments for all calendar entries.")

    # Search and filter options
    search_term = st.text_input("Search summaries", "")

    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_personnel = st.multiselect(
            "Filter by Personnel",
            options=["Unknown"] + personnel_data["canonical_names"],
            default=[]
        )

    with col2:
        selected_event_types = st.multiselect(
            "Filter by Event Type",
            options=["Unknown"] + valid_event_types, # Uses unique list now
            default=[]
        )

    with col3:
        date_range = st.date_input(
            "Date Range",
            value=[],
            help="Filter entries by date range"
        )

    # Apply filters
    filtered_df = df.copy()

    # Apply search filter
    if search_term:
        filtered_df = filtered_df[filtered_df['summary'].str.contains(search_term, case=False, na=False)]

    # Apply personnel filter
    if selected_personnel:
        if "Unknown" in selected_personnel:
            # Include both Unknown and specific personnel
            other_personnel = [p for p in selected_personnel if p != "Unknown"]
            personnel_filter = filtered_df['extracted_personnel'].apply(lambda x: isinstance(x, list) and ('Unknown' in x or 'Unknown_Error' in x))
            if other_personnel:
                 personnel_filter |= filtered_df['extracted_personnel'].apply(lambda x: isinstance(x, list) and any(p in x for p in other_personnel))
            filtered_df = filtered_df[personnel_filter]

        else:
            # Only include specific personnel
            filtered_df = filtered_df[
                filtered_df['extracted_personnel'].apply(lambda x: isinstance(x, list) and
                    any(p in x for p in selected_personnel)
                )
            ]

    # Apply event type filter
    if selected_event_types:
        if "Unknown" in selected_event_types:
            # Include both Unknown and specific event types
            other_event_types = [et for et in selected_event_types if et != "Unknown"]
            event_filter = filtered_df['extracted_event_type'].isin(['Unknown', 'Unknown_Error'])
            if other_event_types:
                 event_filter |= filtered_df['extracted_event_type'].isin(other_event_types)
            filtered_df = filtered_df[event_filter]
        else:
            # Only include specific event types
            filtered_df = filtered_df[filtered_df['extracted_event_type'].isin(selected_event_types)]    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['start_time']).dt.date >= start_date) &
            (pd.to_datetime(filtered_df['start_time']).dt.date <= end_date)
        ]

    # Show results count
    st.write(f"Found {len(filtered_df)} entries matching your filters.")

    # --- >>> ADD THIS SECTION <<< ---
    # Add Batch Re-Extract Button for Filtered Entries
    if not filtered_df.empty:
        st.markdown("---") # Optional separator
        if st.button(f"Batch Re-Extract LLM for {len(filtered_df)} Filtered Entries", key="all_batch_retry"):
            uids_to_retry = filtered_df['uid'].tolist()
            if uids_to_retry:
                # Call the existing batch retry function with the filtered UIDs
                batch_retry_entries(uids_to_retry)
            else:
                st.warning("No UIDs found in the filtered data to retry.")
        st.markdown("---") # Optional separator
    # --- >>> END OF ADDED SECTION <<< ---

    # Display in data editor with pagination
    if not filtered_df.empty:
        items_per_page = st.slider("Entries per page", min_value=5, max_value=50, value=10, step=5, key="all_slider")
        page_count = (len(filtered_df) + items_per_page - 1) // items_per_page

        if 'all_page' not in st.session_state:
            st.session_state.all_page = 0

        # Reset page if filters change resulting page count
        if st.session_state.all_page >= page_count:
            st.session_state.all_page = 0

        # Page navigation
        col1, col2, col3 = st.columns([1, 3, 1])
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.all_page <= 0, key="all_prev"):
                st.session_state.all_page -= 1
                st.rerun()

        with col2:
            st.write(f"Page {st.session_state.all_page + 1} of {max(1, page_count)}")

        with col3:
            if st.button("Next ➡️", disabled=st.session_state.all_page >= page_count - 1, key="all_next"):
                st.session_state.all_page += 1
                st.rerun()

        # Display entries for current page
        start_idx = st.session_state.all_page * items_per_page
        end_idx = min(start_idx + items_per_page, len(filtered_df))
        page_df = filtered_df.iloc[start_idx:end_idx].copy()

        # Simplified display for this page
        display_cols = ['uid', 'summary', 'start_time', 'extracted_personnel', 'extracted_event_type']
        st.dataframe(page_df[display_cols], use_container_width=True)

        # Select entry for editing
        edit_options = [""] + page_df['uid'].tolist()
        selected_uid = st.selectbox(
            "Select entry to edit",
            options=edit_options,
            format_func=lambda x: f"{x} - {page_df[page_df['uid'] == x]['summary'].iloc[0][:50]}..." if x else "Select...",
            key="all_edit_select"
        )

        if selected_uid:
            selected_row = page_df[page_df['uid'] == selected_uid].iloc[0]

            st.markdown("### Edit Selected Entry")
            st.markdown(f"**Summary:** {selected_row['summary']}")

            col1, col2 = st.columns(2)

            with col1:
                # Personnel selection - modified to use multiselect
                current_personnel_list = selected_row['extracted_personnel']
                if not isinstance(current_personnel_list, list):
                    current_personnel_list = ['Unknown']
                personnel_options = personnel_data["canonical_names"]
                new_personnel = st.multiselect(
                    "Personnel (select multiple if needed)",
                    options=personnel_options,
                    default=[p for p in current_personnel_list if p in personnel_options and p != 'Unknown' and p != 'Unknown_Error'],
                    key="selected_personnel"
                )

            with col2:
                # Event type selection
                current_event_type = selected_row['extracted_event_type']
                event_type_options = ["Unknown"] + valid_event_types
                new_event_type = st.selectbox(
                    "Event Type",
                    options=event_type_options,
                    index=event_type_options.index(current_event_type) if current_event_type in event_type_options else 0,
                    key="selected_event_type"
                )

            # Save button
            if st.button("Save Changes", key="all_save_changes"):
                try:
                    # Only save if something changed
                    current_personnel_set = set(p for p in current_personnel_list if p != 'Unknown' and p != 'Unknown_Error')
                    new_personnel_set = set(new_personnel)
                    
                    if new_personnel_set != current_personnel_set or new_event_type != current_event_type:
                        success = save_manual_assignment(selected_uid, new_personnel, new_event_type)

                        if success:
                            st.success(f"Successfully updated assignment for entry {selected_uid}.")
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"Failed to save changes for entry {selected_uid}. Please try again.")
                    else:
                        st.info("No changes detected.")
                except Exception as e:
                    st.error(f"Error saving changes for entry {selected_uid}: {e}")
                    logger.error(f"Error saving changes for {selected_uid}: {e}")

def show_batch_assignment(df, personnel_data, valid_event_types):
    """
    Allow batch assignment of personnel and event types based on patterns.
    """
    st.subheader("Batch Assignment")
    st.markdown("Assign personnel and event types to multiple entries at once based on patterns.")

    # Search pattern input
    pattern = st.text_input(
        "Search Pattern",
        help="Enter a search pattern to find matching entries (e.g., 'GK SIM', 'Dr. Smith')"
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
        # Personnel selection - modified to use multiselect
        personnel_options = personnel_data["canonical_names"]
        batch_personnel = st.multiselect(
            "Assign Personnel (select multiple if needed)",
            options=personnel_options,
            key="batch_personnel"
        )

    with col2:
        # Event type selection
        event_type_options = [""] + valid_event_types
        batch_event_type = st.selectbox(
            "Assign Event Type",
            event_type_options,
            key="batch_event_type"
        )

    # Preview changes
    if batch_personnel or batch_event_type:
        st.markdown("### Preview Changes")

        if batch_personnel:
            personnel_display = ", ".join(batch_personnel)
            st.write(f"Personnel will be changed to: **{personnel_display}**")
        else:
            st.write("Personnel will remain unchanged")

        if batch_event_type:
            st.write(f"Event type will be changed to: **{batch_event_type}**")
        else:
            st.write("Event type will remain unchanged")

        # Confirm button
        if st.button("Apply Batch Assignment", key="batch_confirm"):
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
                    # Handle potential non-list personnel data
                    current_personnel_list = row['extracted_personnel'] if isinstance(row['extracted_personnel'], list) else ['Unknown']
                    
                    # Use the multi-selected personnel or keep current
                    personnel_to_assign = batch_personnel if batch_personnel else current_personnel_list
                    event_type_to_assign = batch_event_type or row['extracted_event_type']

                    # Skip if nothing is changing
                    if (not batch_personnel and event_type_to_assign == row['extracted_event_type']):
                         progress = (i + 1) / total_count
                         progress_bar.progress(progress)
                         status_text.text(f"Skipping {i+1}/{total_count} (no change)...")
                         continue

                    # Save the assignment
                    if save_manual_assignment(entry_id, personnel_to_assign, event_type_to_assign):
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
                st.rerun() # Refresh the page to show updated data

            except Exception as e:
                st.error(f"Error applying batch assignment: {e}")
                logger.error(f"Error in batch assignment: {e}")
    else:
        st.info("Select personnel or event type to assign to matching entries.")
