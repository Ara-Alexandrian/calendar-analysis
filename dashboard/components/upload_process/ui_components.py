"""
UI components for the Upload & Process page.
Contains Streamlit UI elements and rendering functions.
"""
import streamlit as st
import pandas as pd
import logging
import time
import hashlib
import uuid

logger = logging.getLogger(__name__)

def render_file_uploader():
    """Render the file uploader component and handle uploaded files."""
    uploaded_file = st.file_uploader("Choose a calendar.json file", type="json", key="uploader")
    
    if uploaded_file is not None:
        # Store filename and content in session state immediately upon upload
        st.session_state.uploaded_filename = uploaded_file.name
        
        # Read content - important to do it here before the file buffer might close
        try:
            file_content_bytes = uploaded_file.getvalue()  # Read as bytes
            st.session_state.uploaded_file_content = file_content_bytes  # Store bytes in session
            
            # Calculate hash for later use
            st.session_state.uploaded_file_hash = hashlib.sha256(file_content_bytes).hexdigest()
            
            return uploaded_file, file_content_bytes
        except Exception as e:
            logger.error(f"Error reading uploaded file {uploaded_file.name}: {e}", exc_info=True)
            st.error(f"An error occurred reading the uploaded file: {e}")
            st.session_state.raw_df = None
            st.session_state.uploaded_filename = None
            
    return None, None

def display_file_info(raw_df):
    """Display information about the uploaded file."""
    if raw_df is not None:
        st.success(f"Successfully loaded '{st.session_state.uploaded_filename}'. Found {len(raw_df)} raw events.")
        st.dataframe(raw_df.head())
        return True
    return False

def render_processing_button(config_ok, llm_ok):
    """Render the processing button with appropriate state."""
    # Only enable button if config is okay
    process_button_disabled = not config_ok or not llm_ok
    tooltip_text = "Personnel configuration missing. Check Admin page." if not config_ok else "Run the full data processing pipeline"
    
    if not llm_ok:
        tooltip_text = "LLM is required but not available. Check the error message."
    
    return st.button("Process Data", key="process_button", disabled=process_button_disabled, help=tooltip_text)

def display_progress_bar(label="Progress"):
    """Create and return a progress bar."""
    return st.progress(0, text=label)

def update_progress_bar(progress_bar, progress, text):
    """Update the progress bar value and text."""
    progress_bar.progress(progress, text=text)

def display_processing_summary(df, stage_name):
    """Display a summary of the processed data at various stages."""
    if df is not None and not df.empty:
        st.write(f"{stage_name} finished: {len(df)} events.")
        return True
    st.warning(f"No data available after {stage_name.lower()} stage.")
    return False

def display_sample_data(df, columns=None, title="Sample Data"):
    """Display a sample of the dataframe with optional column selection."""
    if df is not None and not df.empty:
        st.write(title)
        display_cols = columns if columns else df.columns
        st.dataframe(df[display_cols].head())
        return True
    return False

def render_llm_status(llm_enabled, is_llm_ready_fn):
    """Render LLM status information and controls."""
    llm_ok = True
    
    if llm_enabled:
        if not is_llm_ready_fn():
            st.warning("LLM is configured but not reachable. Name extraction will be skipped or may fail.")
            
            # Add a checkbox option to proceed without LLM
            st.session_state.skip_llm = st.checkbox(
                "Proceed without LLM extraction (all personnel will be marked as 'Unknown')",
                value=True,
                key="skip_llm_checkbox"
            )
            
            if not st.session_state.skip_llm:
                llm_ok = False  # Only block processing if user unchecks the box
    
    return llm_ok

def render_resume_redo_buttons(batch_id, batch_status, db_enabled):
    """Render resume and redo buttons with appropriate state."""
    if batch_status is None:
        return
    
    total_in_batch = batch_status.get('total_events', 0)
    extracted_count = batch_status.get('extracted', 0)
    assigned_count = batch_status.get('assigned', 0)
    error_count = batch_status.get('error', 0)
    processing_count = batch_status.get('processing', 0)
    
    # Events are extracted but not all are assigned
    needs_assignment = extracted_count > 0 and assigned_count < total_in_batch
    # Or there are errors
    has_errors = error_count > 0
    # Or there's active processing
    is_processing = processing_count > 0
    
    resume_disabled = not (needs_assignment or has_errors or is_processing)
    
    if resume_disabled:
        if extracted_count == total_in_batch and assigned_count == 0:
            # Special case: All events extracted but none assigned
            resume_tooltip = "Resume to complete normalization and assignment phase"
            resume_disabled = False
        else:
            resume_tooltip = "No partial processing detected"
    else:
        resume_tooltip = "Continue processing from where you left off"
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Resume Analysis", disabled=resume_disabled, help=resume_tooltip):
            st.session_state.processing_mode = "Resume Processing"
            st.session_state.current_batch_id = batch_id
            st.session_state.processing_triggered = True
            if not db_enabled:
                st.success("Resume processing triggered!")
            # Force page rerun to trigger the processing
            st.rerun()
    
    with col2:
        if st.button("🔄 Redo Analysis", help="Start fresh with current LLM settings"):
            st.session_state.processing_mode = "Start Fresh"
            st.session_state.current_batch_id = batch_id
            st.session_state.processing_triggered = True
            if not db_enabled:
                st.success("Redo processing triggered!")
            # Force page rerun to trigger the processing
            st.rerun()

def render_processing_mode_selection(batch_id, batch_status):
    """Render the processing mode selection radio buttons."""
    if batch_status is None:
        st.error("Failed to retrieve batch status. Defaulting to Start Fresh.")
        st.session_state.processing_mode = "Start Fresh"
        return
    
    total_in_batch = batch_status.get('total_events', 0)
    if total_in_batch > 0:
        st.write(f"Status for Batch {batch_id}:")
        st.json(batch_status)  # Show the status counts
        
        processed_count = batch_status.get('extracted', 0) + batch_status.get('assigned', 0)
        # Offer choice only if partially processed (excluding errors from partial check)
        if 0 < total_in_batch and (processed_count < total_in_batch or batch_status.get('error', 0) == total_in_batch):
            # Use a container to hold the radio button
            choice_container = st.container()
            with choice_container:
                processing_mode_choice = st.radio(
                    "Choose processing mode:",
                    ("Resume Processing", "Start Fresh (Reprocess All)"),
                    key="resume_radio",
                    index=0 if st.session_state.processing_mode == "Resume Processing" else 1,
                    help="Resume: Process only events not yet processed. Start Fresh: Reprocess all events."
                )
                st.session_state.processing_mode = processing_mode_choice  # Update session state immediately
        elif total_in_batch > 0:  # If total > 0 and the above condition is false, it means processing is complete
            st.info("Previous processing appears complete. Defaulting to Start Fresh.")
            st.session_state.processing_mode = "Start Fresh"
    else:  # Corresponds to 'if total_in_batch > 0'
        st.info("No previous processing data found for this batch. Starting fresh.")
        st.session_state.processing_mode = "Start Fresh"
