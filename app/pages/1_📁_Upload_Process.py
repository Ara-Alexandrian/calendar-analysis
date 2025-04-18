# pages/1_📁_Upload_Process.py
"""
Upload & Process page for the Calendar Analysis application.
Handles uploading calendar data and processing it through the pipeline.
"""
import streamlit as st
import pandas as pd
import logging
import time
import importlib
import sys, os

# Ensure project root is in path (if running page directly)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the refactored components
from app.components.upload_process import ui_components, db_operations, llm_processing, data_processing

# Import other required modules
from functions import data_processor
from functions.llm_extraction.ollama_client import get_ollama_client
from functions.llm_extraction.normalizer import normalize_extracted_personnel
from functions.llm_extraction.client import is_llm_ready
import functions.llm_extraction
from functions import config_manager
from config import settings
from functions import db as db_manager

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Upload & Process", layout="wide")

st.markdown("# 1. Upload & Process Calendar Data")
st.markdown("Upload your `calendar.json` file below and click 'Process Data' to run the analysis pipeline.")

# Initialize session state variables
if 'processing_triggered' not in st.session_state:
    st.session_state.processing_triggered = False

if 'processing_mode' not in st.session_state:
    st.session_state.processing_mode = "Start Fresh"

if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False

# --- File Upload ---
uploaded_file, file_content = ui_components.render_file_uploader()

if uploaded_file is not None:
    # Load raw data
    raw_df = data_processing.load_and_process_data(data_processor, st.session_state.uploaded_file_content, st.session_state.uploaded_filename)
    
    if raw_df is not None:
        # Display info about the loaded file
        ui_components.display_file_info(raw_df)
        
        # Check if file was processed before (for Resume/Redo options)
        is_duplicate_file = False
        current_db_batch_id = None
        
        # Log database status for debugging
        logger.info(f"DB_ENABLED in settings: {settings.DB_ENABLED}")
        st.write(f"Database enabled: {'Yes' if settings.DB_ENABLED else 'No'}")
        
        # Try to check for previous processing
        try:
            if settings.DB_ENABLED:
                logger.info("Checking if file exists in database...")
                success, current_db_batch_id, is_duplicate = db_operations.check_file_in_database(
                    db_manager, 
                    st.session_state.uploaded_filename, 
                    st.session_state.uploaded_file_content
                )
                
                if is_duplicate and current_db_batch_id:
                    st.info(f"This file has been processed before (Batch ID: {current_db_batch_id}).")
                    try:
                        batch_status = db_manager.check_batch_status(current_db_batch_id)
                        logger.info(f"Batch status: {batch_status}")
                        if batch_status is not None:
                            total_in_batch = batch_status.get('total_events', 0)
                            processed_count = batch_status.get('extracted', 0) + batch_status.get('assigned', 0)
                            
                            # Show batch status
                            st.write("Current processing status:")
                            st.json(batch_status)
                            
                            # Try to get LLM info if available
                            try:
                                llm_info = db_manager.get_batch_llm_info(current_db_batch_id)
                                if llm_info:
                                    st.info(f"Previous analysis used LLM provider: {llm_info.get('provider', 'Unknown')}, Model: {llm_info.get('model', 'Unknown')}")
                            except Exception as llm_info_err:
                                logger.error(f"Error getting LLM info: {llm_info_err}")
                                st.warning("Could not retrieve LLM information for this batch.")
                            
                            # Show Resume/Redo buttons
                            ui_components.render_resume_redo_buttons(current_db_batch_id, batch_status, settings.DB_ENABLED)
                            
                    except Exception as status_err:
                        logger.error(f"Error checking batch status: {status_err}")
                        st.error(f"Could not check batch status: {status_err}")
            else:
                # For testing: Show buttons even when DB is disabled
                st.warning("Database is disabled. Adding Resume/Redo buttons for testing.")
                
                # Create a temporary batch ID for testing
                import hashlib
                temp_batch_id = f"temp_{hashlib.md5(file_content).hexdigest()[:8]}"
                
                # Mock batch status for testing
                mock_batch_status = {
                    'total_events': len(raw_df),
                    'extracted': 0,
                    'assigned': 0,
                    'error': 0,
                    'processing': 0
                }
                
                # Show Resume/Redo buttons for testing
                ui_components.render_resume_redo_buttons(temp_batch_id, mock_batch_status, False)
                
        except Exception as e:
            logger.error(f"Unexpected error in file processing check: {e}", exc_info=True)
            st.error(f"Error checking file status: {e}")
        
        # Reset processing flags if a new file is uploaded unless Resume/Redo was explicitly chosen
        if not st.session_state.get('processing_triggered', False):
            st.session_state.data_processed = False
            st.session_state.preprocessed_df = None
            st.session_state.llm_processed_df = None
            st.session_state.normalized_df = None
            st.session_state.analysis_ready_df = None
            st.session_state.processing_mode = "Start Fresh"  # Default for new uploads

# --- Processing Button ---
if st.session_state.get('raw_df') is not None:
    st.markdown("---")
    st.subheader("Processing Pipeline")

    # Load personnel config directly from config_manager
    try:
        _, _, canonical_names = config_manager.get_personnel_details()
        st.session_state['canonical_names'] = canonical_names  # Store in session state for later use
        if not canonical_names:
            st.warning("Personnel configuration is loaded but empty. Processing may yield poor results.")
            config_ok = False
        else:
            config_ok = True
            st.info(f"Found {len(canonical_names)} personnel in configuration.")
    except Exception as e:
        st.warning(f"Personnel configuration could not be loaded: {e}. Processing may yield poor results.")
        config_ok = True  # Still allow processing even if config can't be loaded

    # Check LLM status
    llm_enabled, llm_ok = llm_processing.check_llm_status(settings, is_llm_ready, functions.llm_extraction)

    # Render processing button
    if ui_components.render_processing_button(config_ok, llm_ok):
        if st.session_state.raw_df is not None:
            logger.info("Starting data processing pipeline...")
            start_time = time.time()
            with st.spinner("Processing data... This may take a while for the initial steps."):
                try:
                    # --- Ensure DB Schema Exists BEFORE Processing ---
                    if settings.DB_ENABLED:
                        if not db_operations.ensure_database_schema(db_manager):
                            st.stop()
                    # ------------------------------------------------

                    # 0. Save Calendar File Info to DB (if enabled) and check for duplicates
                    is_duplicate_file = False  # Default
                    current_db_batch_id = None  # Default
                    if settings.DB_ENABLED:
                        st.write("Registering uploaded file in database...")
                        current_db_batch_id, is_duplicate_file = db_operations.save_file_to_database(
                            db_manager,
                            st.session_state.uploaded_filename,
                            st.session_state.uploaded_file_content,
                            settings
                        )

                    # --- Resume/Start Fresh Logic (Now inside button press, after DB check) ---
                    # Check if processing was triggered by Resume/Redo buttons
                    if st.session_state.get('processing_triggered', False):
                        logger.info(f"Processing triggered by button: {st.session_state.processing_mode}")
                        # Clear the triggered flag after acknowledging it
                        st.session_state.processing_triggered = False
                    # Otherwise, initialize processing_mode if it doesn't exist or if it's a new file
                    elif 'processing_mode' not in st.session_state or not is_duplicate_file:
                        st.session_state.processing_mode = "Start Fresh"
                    
                    # If we're not triggered by a button and it's a duplicate file, we might need to offer a choice
                    if not st.session_state.get('processing_triggered', False) and is_duplicate_file and current_db_batch_id:
                        st.warning(f"This file (Batch ID: {current_db_batch_id}) seems to have been processed before.")
                        batch_status = db_manager.check_batch_status(current_db_batch_id)
                        ui_components.render_processing_mode_selection(current_db_batch_id, batch_status)

                    logger.info(f"Processing Mode selected: {st.session_state.processing_mode}")  # Log the selected mode

                    # 1. Preprocess Data
                    preprocessed_df = data_processing.preprocess_data(data_processor, st.session_state.raw_df)
                    if preprocessed_df is None:
                        st.stop()

                    # 2. LLM Extraction
                    if llm_enabled and llm_ok:
                        st.write("Step 2: Extracting personnel names from calendar events...")
                        try:
                            st.info("Processing calendar entries with LLM extraction...")
                            llm_client = get_ollama_client()  # Get the client
                            if not llm_client:
                                raise Exception("Failed to get LLM Client.")

                            canonical_names_list = st.session_state.get('canonical_names', [])
                            
                            # Process events with LLM
                            llm_processed_df = llm_processing.process_events_with_llm(
                                preprocessed_df,
                                llm_client,
                                canonical_names_list,
                                db_manager,
                                current_db_batch_id,
                                settings
                            )
                            
                            st.session_state.llm_processed_df = llm_processed_df
                            
                        except Exception as e:
                            logger.error(f"LLM extraction loop or final fetch failed: {e}", exc_info=True)
                            st.error(f"Extraction failed: {e}. Using Unknown for all events.")
                            # Create placeholder with Unknown values
                            temp_df = preprocessed_df.copy()
                            temp_df['extracted_personnel'] = [['Unknown']] * len(temp_df)
                            temp_df['extracted_event_type'] = ['Unknown'] * len(temp_df)
                            st.session_state.llm_processed_df = temp_df
                            llm_processed_df = temp_df  # Ensure llm_processed_df is assigned
                    else:
                        st.warning("Skipping LLM extraction (disabled or not ready). Assigning 'Unknown'.")
                        # Create placeholder columns if skipping LLM
                        temp_df = preprocessed_df.copy()
                        temp_df['extracted_personnel'] = [['Unknown']] * len(temp_df)
                        temp_df['extracted_event_type'] = ['Unknown'] * len(temp_df)
                        st.session_state.llm_processed_df = temp_df  # Use preprocessed df as base
                        llm_processed_df = temp_df  # Ensure llm_processed_df is assigned

                    # 3. Normalize Extracted Personnel Names (Event type already normalized in step 2)
                    normalized_df = data_processing.normalize_personnel(
                        normalize_extracted_personnel,
                        llm_processed_df
                    )
                    
                    if normalized_df is None:
                        st.stop()

                    # Display sample of normalized data (ensure normalized_df exists)
                    if normalized_df is not None and not normalized_df.empty:
                        st.write("Sample of data after normalization:")
                        display_cols = [col for col in ['summary', 'start_time', 'duration_hours', 'extracted_personnel', 'extracted_event_type', 'assigned_personnel'] if col in normalized_df.columns]
                        st.dataframe(normalized_df[display_cols].head())
                    else:
                        st.warning("No data available to display after normalization step.")

                    # 4. Explode by Personnel for Analysis
                    analysis_ready_df = data_processing.explode_data_for_analysis(
                        data_processor,
                        normalized_df
                    )
                    
                    if analysis_ready_df is None:
                        st.stop()

                    # Mark processing as complete and ensure data is available for analysis
                    st.session_state.data_processed = True

                    # Store analysis data in session state
                    current_batch_key = data_processing.store_analysis_data(
                        analysis_ready_df,
                        st.session_state.uploaded_filename
                    )

                    # 5. Save processed data to PostgreSQL database (if enabled)
                    if settings.DB_ENABLED:
                        st.write("Step 5: Saving processed data to PostgreSQL database...")
                        db_operations.save_analysis_data(db_manager, analysis_ready_df)

                    elapsed_time = time.time() - start_time
                    st.success(f"Processing complete in {elapsed_time:.1f} seconds! You can now proceed to the Analysis page.")
                    logger.info(f"Total processing pipeline completed in {elapsed_time:.1f} seconds")

                except Exception as e:
                    logger.error(f"Processing pipeline error: {e}", exc_info=True)
                    st.error(f"An error occurred during processing: {e}")
                    if isinstance(e, IndexError) and "index out of bounds" in str(e):
                        st.info("This error might indicate that the LLM is not available or is not responding correctly. Check your LLM configuration.")
                    st.session_state.data_processed = False

# --- Processing Status ---
if st.session_state.get('data_processed', False):
    st.success("✅ Data has been processed and is ready for analysis!")
    
    # Show a link to the Analysis page
    st.markdown("### Next Steps")
    st.markdown("Proceed to the Analysis page to explore your data.")
    if st.button("Go to Analysis Page"):
        st.switch_page("pages/2_📊_Analysis.py")
