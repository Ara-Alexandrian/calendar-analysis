# fixed_manual_assignment.py
"""
Fixed version of the manual assignment page for the admin dashboard.
This file fixes indentation issues and properly supports multiple personnel assignment.
"""
import streamlit as st
import pandas as pd
import logging
import time
import json
import re
import ast
import os
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import settings
from functions import db as db_manager
from functions import config_manager
from functions.llm_extraction.normalizer import normalize_extracted_personnel
from admin_dashboard.utils.admin_functions import save_manual_assignment
from src.application.services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)

# This function fixes the batch assignment functionality to support multiple personnel
def fixed_show_batch_assignment(df, personnel_data, valid_event_types):
    """
    Allow batch assignment of personnel and event types based on patterns.
    This version properly supports multiple personnel assignment.
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
        # Personnel selection - using multiselect for multiple personnel assignment
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
            # Show all selected personnel in the preview
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
                st.rerun()  # Refresh the page to show updated data

            except Exception as e:
                st.error(f"Error applying batch assignment: {e}")
                logger.error(f"Error in batch assignment: {e}")
    else:
        st.info("Select personnel or event type to assign to matching entries.")
