"""
Manual assignment page for the admin dashboard.
Allows administrators to manually assign personnel and event types to calendar entries.
"""
import streamlit as st
import pandas as pd
import logging
import time
from config import settings
from functions import db as db_manager
from functions import config_manager
from functions.llm_extraction.normalizer import normalize_extracted_personnel
from admin_dashboard.utils.admin_functions import save_manual_assignment

logger = logging.getLogger(__name__)

def show_manual_assignment_page():
    """
    Display the manual assignment interface for administrators.
    """
    st.title("✏️ Manual Assignment")
    st.markdown("Manually assign personnel and event types to calendar entries.")
    
    # Load data
    df_data = load_data_for_manual_assignment()
    if df_data is None or df_data.empty:
        st.warning("No data available for manual assignment. Please upload and process data first.")
        return
    
    # Get valid personnel and event types
    personnel_data = load_personnel_data()
    valid_event_types = load_event_types()
    
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
            st.info("Loading data from database...")
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
                    try:
                        df['extracted_personnel'] = df['extracted_personnel'].apply(eval)
                    except:
                        pass
                    
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
            "canonical_names": canonical_names,
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
    Load valid event types from settings.
    """
    try:
        # Get event types from settings
        valid_event_types = list(settings.EVENT_TYPE_MAPPING.values())
        return valid_event_types
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
    
    # Pagination
    items_per_page = st.slider("Entries per page", min_value=5, max_value=50, value=10, step=5)
    page_count = (len(unassigned_df) + items_per_page - 1) // items_per_page
    
    if 'unassigned_page' not in st.session_state:
        st.session_state.unassigned_page = 0
        
    # Page navigation
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.unassigned_page <= 0):
            st.session_state.unassigned_page -= 1
            st.rerun()
            
    with col2:
        st.write(f"Page {st.session_state.unassigned_page + 1} of {max(1, page_count)}")
        
    with col3:
        if st.button("Next ➡️", disabled=st.session_state.unassigned_page >= page_count - 1):
            st.session_state.unassigned_page += 1
            st.rerun()
    
    # Display entries for current page
    start_idx = st.session_state.unassigned_page * items_per_page
    end_idx = min(start_idx + items_per_page, len(unassigned_df))
    page_df = unassigned_df.iloc[start_idx:end_idx].copy()
    
    # Function to handle assignment
    def handle_assignment(entry_index, personnel, event_type):
        try:
            entry_id = page_df.iloc[entry_index]['uid']
            original_summary = page_df.iloc[entry_index]['summary']
            
            if personnel and event_type:
                # Save assignment
                success = save_manual_assignment(entry_id, personnel, event_type)
                
                if success:
                    st.success(f"Successfully assigned personnel '{personnel}' and event type '{event_type}' to entry.")
                    # Update the dataframe
                    page_df.at[entry_index, 'extracted_personnel'] = [personnel]
                    page_df.at[entry_index, 'extracted_event_type'] = event_type
                    unassigned_df.iloc[start_idx + entry_index] = page_df.iloc[entry_index]
                      # Refresh the display after a short delay
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to save assignment. Please try again.")
            else:
                st.warning("Please select both personnel and event type.")
        except Exception as e:
            st.error(f"Error assigning personnel: {e}")
            logger.error(f"Error in handle_assignment: {e}")
    
    # Display each entry with assignment controls
    for i, (idx, row) in enumerate(page_df.iterrows()):
        with st.container():
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Entry ID:** {row.get('uid', 'N/A')}")
                st.markdown(f"**Summary:** {row.get('summary', 'N/A')}")
                st.markdown(f"**Date/Time:** {row.get('start_time', 'N/A')}")
                
                # Current assignments
                current_personnel = "Unknown" if row['extracted_personnel'] == ['Unknown'] else ", ".join(row['extracted_personnel'])
                current_event_type = row.get('extracted_event_type', 'Unknown')
                
                st.markdown(f"**Current Personnel:** {current_personnel}")
                st.markdown(f"**Current Event Type:** {current_event_type}")
            
            with col2:
                # Assignment controls
                st.subheader("Assign")
                
                # Personnel selection
                personnel_options = [""] + personnel_data["canonical_names"]
                selected_personnel = st.selectbox(
                    "Personnel", 
                    personnel_options,
                    key=f"personnel_{i}"
                )
                
                # Event type selection
                event_type_options = [""] + valid_event_types
                selected_event_type = st.selectbox(
                    "Event Type", 
                    event_type_options,
                    key=f"event_type_{i}"
                )
                
                # Save button
                if st.button("Save Assignment", key=f"save_{i}"):
                    handle_assignment(i, selected_personnel, selected_event_type)
                
                # LLM retry button
                if st.button("Retry LLM", key=f"retry_{i}"):
                    st.info("LLM retry functionality will be implemented in the next version.")

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
            options=["Unknown"] + valid_event_types,
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
            if other_personnel:
                filtered_df = filtered_df[
                    filtered_df['extracted_personnel'].apply(lambda x: 
                        (isinstance(x, list) and ('Unknown' in x or 'Unknown_Error' in x)) or 
                        any(p in x for p in other_personnel if isinstance(x, list))
                    )
                ]
            else:
                filtered_df = filtered_df[
                    filtered_df['extracted_personnel'].apply(lambda x: 
                        isinstance(x, list) and ('Unknown' in x or 'Unknown_Error' in x)
                    )
                ]
        else:
            # Only include specific personnel
            filtered_df = filtered_df[
                filtered_df['extracted_personnel'].apply(lambda x: 
                    any(p in x for p in selected_personnel if isinstance(x, list))
                )
            ]
    
    # Apply event type filter
    if selected_event_types:
        if "Unknown" in selected_event_types:
            # Include both Unknown and specific event types
            other_event_types = [et for et in selected_event_types if et != "Unknown"]
            if other_event_types:
                filtered_df = filtered_df[
                    (filtered_df['extracted_event_type'].isin(other_event_types)) | 
                    (filtered_df['extracted_event_type'] == 'Unknown') | 
                    (filtered_df['extracted_event_type'] == 'Unknown_Error')
                ]
            else:
                filtered_df = filtered_df[
                    (filtered_df['extracted_event_type'] == 'Unknown') | 
                    (filtered_df['extracted_event_type'] == 'Unknown_Error')
                ]
        else:
            # Only include specific event types
            filtered_df = filtered_df[filtered_df['extracted_event_type'].isin(selected_event_types)]
    
    # Apply date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (pd.to_datetime(filtered_df['start_time']).dt.date >= start_date) & 
            (pd.to_datetime(filtered_df['start_time']).dt.date <= end_date)
        ]
    
    # Show results count
    st.write(f"Found {len(filtered_df)} entries matching your filters.")
    
    # Display in data editor with pagination
    if not filtered_df.empty:
        items_per_page = st.slider("Entries per page", min_value=5, max_value=50, value=10, step=5, key="all_slider")
        page_count = (len(filtered_df) + items_per_page - 1) // items_per_page
        
        if 'all_page' not in st.session_state:
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
        selected_uid = st.selectbox(
            "Select entry to edit",
            options=page_df['uid'].tolist(),
            format_func=lambda x: f"{x} - {page_df[page_df['uid'] == x]['summary'].iloc[0][:50]}..."
        )
        
        if selected_uid:
            selected_row = page_df[page_df['uid'] == selected_uid].iloc[0]
            
            st.markdown("### Edit Selected Entry")
            st.markdown(f"**Summary:** {selected_row['summary']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Personnel selection
                current_personnel = selected_row['extracted_personnel'][0] if isinstance(selected_row['extracted_personnel'], list) and len(selected_row['extracted_personnel']) > 0 else "Unknown"
                personnel_options = ["Unknown"] + personnel_data["canonical_names"]
                new_personnel = st.selectbox(
                    "Personnel", 
                    options=personnel_options,
                    index=personnel_options.index(current_personnel) if current_personnel in personnel_options else 0,
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
            if st.button("Save Changes"):
                try:
                    # Only save if something changed
                    if new_personnel != current_personnel or new_event_type != current_event_type:
                        success = save_manual_assignment(selected_uid, new_personnel, new_event_type)
                        
                        if success:
                            st.success(f"Successfully updated assignment for entry.")
                            # Update the dataframes
                            idx = page_df[page_df['uid'] == selected_uid].index[0]
                            page_df.at[idx, 'extracted_personnel'] = [new_personnel]
                            page_df.at[idx, 'extracted_event_type'] = new_event_type
                            
                            # Refresh after a short delay
                            time.sleep(0.5)
                            st.experimental_rerun()
                        else:
                            st.error("Failed to save changes. Please try again.")
                    else:
                        st.info("No changes detected.")
                except Exception as e:
                    st.error(f"Error saving changes: {e}")
                    logger.error(f"Error saving changes: {e}")

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
        # Personnel selection
        personnel_options = [""] + personnel_data["canonical_names"]
        batch_personnel = st.selectbox(
            "Assign Personnel", 
            personnel_options,
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
            st.write(f"Personnel will be changed to: **{batch_personnel}**")
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
                total_count = len(matching_df)
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process each matching entry
                for i, (idx, row) in enumerate(matching_df.iterrows()):
                    entry_id = row['uid']
                    
                    # Determine what to update
                    personnel_to_assign = batch_personnel or (row['extracted_personnel'][0] if isinstance(row['extracted_personnel'], list) and len(row['extracted_personnel']) > 0 else "Unknown")
                    event_type_to_assign = batch_event_type or row['extracted_event_type']
                    
                    # Skip if both are unknown
                    if personnel_to_assign == "Unknown" and event_type_to_assign == "Unknown":
                        continue
                    
                    # Save the assignment
                    if save_manual_assignment(entry_id, personnel_to_assign, event_type_to_assign):
                        success_count += 1
                    
                    # Update progress
                    progress = (i + 1) / total_count
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {i+1}/{total_count} entries...")
                
                # Final status
                if success_count > 0:
                    st.success(f"Successfully updated {success_count} of {total_count} entries.")
                    
                    # Refresh after a short delay
                    time.sleep(1)
                    st.experimental_rerun()
                else:
                    st.error("Failed to update any entries. Please try again.")
                
            except Exception as e:
                st.error(f"Error applying batch assignment: {e}")
                logger.error(f"Error in batch assignment: {e}")
    else:
        st.info("Select personnel or event type to assign to matching entries.")
