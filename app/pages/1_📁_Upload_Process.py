# pages/1_📁_Upload_Process.py
import streamlit as st
import pandas as pd
import logging
import time
import importlib

# Ensure project root is in path (if running page directly)
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import and reload modules to ensure latest changes
from functions import data_processor
import functions.llm_extraction as llm_extraction
importlib.reload(llm_extraction)  # Force reload to get latest changes
from functions import config_manager # Needed to check config status
from config import settings # For LLM provider check
import functions.db_manager as db_manager
importlib.reload(db_manager)  # Force reload the module to get latest changes

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Upload & Process", layout="wide") # Config for this page

st.markdown("# 1. Upload & Process Calendar Data")
st.markdown("Upload your `calendar.json` file below and click 'Process Data' to run the analysis pipeline.")

# --- File Upload ---
uploaded_file = st.file_uploader("Choose a calendar.json file", type="json", key="uploader")

if uploaded_file is not None:
    # Store filename and content in session state immediately upon upload
    st.session_state.uploaded_filename = uploaded_file.name
    # Read content - important to do it here before the file buffer might close
    try:
        file_content = uploaded_file.getvalue()
        
        # Save the raw file to database if enabled
        from functions import db_manager
        if settings.DB_ENABLED:
            success, batch_id, is_duplicate = db_manager.save_calendar_file_to_db(
                filename=uploaded_file.name,
                file_content=file_content
            )
            
            if success:
                st.session_state.current_batch_id = batch_id
                if is_duplicate:
                    st.info(f"This file has been uploaded before. Existing data will be used to avoid duplicates.")
        
        # Load raw data using the modified function
        raw_df = data_processor.load_raw_calendar_data(file_content, uploaded_file.name)

        if raw_df is not None:
            st.session_state.raw_df = raw_df
            st.success(f"Successfully loaded '{uploaded_file.name}'. Found {len(raw_df)} raw events.")
            st.dataframe(raw_df.head())
            # Reset processing flags if a new file is uploaded
            st.session_state.data_processed = False
            st.session_state.preprocessed_df = None
            st.session_state.llm_processed_df = None
            st.session_state.normalized_df = None
            st.session_state.analysis_ready_df = None
        else:
            st.error(f"Failed to load or parse '{uploaded_file.name}'. Check file format and logs.")
            st.session_state.raw_df = None # Clear invalid data
            st.session_state.uploaded_filename = None

    except Exception as e:
        st.error(f"An error occurred reading the uploaded file: {e}")
        logger.error(f"Error reading uploaded file {uploaded_file.name}: {e}", exc_info=True)
        st.session_state.raw_df = None
        st.session_state.uploaded_filename = None

# --- Processing Button ---
if st.session_state.get('raw_df') is not None:
    st.markdown("---")
    st.subheader("Processing Pipeline")

    # Check if personnel config is loaded
    if not st.session_state.get('canonical_names'):
         st.warning("Personnel configuration is not loaded or empty. Please check the Admin page. Processing may yield poor results.")
         config_ok = False
    else:
        config_ok = True

    # Check LLM status if provider is configured
    llm_ok = True
    llm_enabled = settings.LLM_PROVIDER and settings.LLM_PROVIDER.lower() != 'none'
    if llm_enabled:
        if not llm_extraction.is_llm_ready():
            if not llm_extraction.OLLAMA_AVAILABLE:
                st.error("""
                **Ollama library is not installed.** This is required for LLM extraction.
                
                Run this command in your terminal to install it:
                ```
                pip install ollama
                ```
                
                Then restart the application. You can still proceed with processing, but all personnel will be marked as 'Unknown'.
                """)
            else:
                st.warning(f"LLM ({settings.LLM_PROVIDER}) is configured but not reachable at {settings.OLLAMA_BASE_URL}. Name extraction will be skipped or may fail.")
            
            # Add a checkbox option to proceed without LLM
            st.session_state.skip_llm = st.checkbox("Proceed without LLM extraction (all personnel will be marked as 'Unknown')", 
                                                  value=True, 
                                                  key="skip_llm_checkbox")
            
            if not st.session_state.skip_llm:
                llm_ok = False  # Only block processing if user unchecks the box
            else:
                llm_ok = True   # Allow processing with warnings if user wants to proceed

    # Only enable button if config is okay (or allow proceed with warning?)
    # Disable button if config is absolutely required and missing.
    process_button_disabled = not config_ok or not llm_ok  # Disable if config is missing or LLM is required but not available
    tooltip_text = "Personnel configuration missing. Check Admin page." if not config_ok else "Run the full data processing pipeline"
    if not llm_ok:
        tooltip_text = "LLM is required but not available. Check the error message."

    if st.button("Process Data", key="process_button", disabled=process_button_disabled, help=tooltip_text):
        if st.session_state.raw_df is not None:
            logger.info("Starting data processing pipeline...")
            start_time = time.time()
            with st.spinner("Processing data... This may take a while for the initial steps."):
                try:
                    # 1. Preprocess Data
                    st.write("Step 1: Preprocessing data (dates, duration)...")
                    preprocessed_df = data_processor.preprocess_data(st.session_state.raw_df)
                    if preprocessed_df is None or preprocessed_df.empty:
                        st.error("Preprocessing failed or resulted in empty data. Check logs.")
                        st.stop()
                    st.session_state.preprocessed_df = preprocessed_df
                    st.write(f"Preprocessing done: {len(preprocessed_df)} events remaining.")
                    logger.info(f"Preprocessing complete: {len(preprocessed_df)} events.")

                    # Generate a unique batch ID for this processing job
                    import uuid
                    batch_id = f"batch_{uuid.uuid4().hex[:8]}_{int(time.time())}"
                    st.session_state.current_batch_id = batch_id
                      # 2. LLM Extraction - Using Smart GPU-Optimized Processing
                    if llm_enabled and llm_ok: 
                        st.write("Step 2: Extracting personnel names using optimized dual-GPU LLM extraction...")
                        progress_container = st.empty()
                        with progress_container.container():
                            with st.spinner("Optimized LLM extraction in progress - leveraging dual RTX 3090s and RTX 4090..."):
                                try:
                                    # Try the new smart extractor first
                                    llm_processed_df = llm_extraction.run_smart_extraction(preprocessed_df)
                                    st.session_state.llm_processed_df = llm_processed_df
                                except Exception as e:
                                    logger.warning(f"Smart extraction failed: {e}. Falling back to standard extraction.")
                                    st.warning("Optimized extraction encountered an issue. Falling back to standard extraction.")
                                    # Fall back to sequential extraction if smart extraction fails
                                    llm_processed_df = llm_extraction.run_llm_extraction_sequential(preprocessed_df)
                                    st.session_state.llm_processed_df = llm_processed_df
                        st.write("LLM extraction finished.")
                        logger.info("LLM extraction complete.")
                    else:
                        st.warning("Skipping LLM extraction (disabled or not ready). Assigning 'Unknown'.")
                        # Create placeholder columns if skipping LLM
                        preprocessed_df['extracted_personnel'] = [['Unknown']] * len(preprocessed_df)
                        st.session_state.llm_processed_df = preprocessed_df # Use preprocessed df as base

                    # 3. Normalize Extracted Personnel Names
                    st.write("Step 3: Normalizing extracted names...")
                    normalized_df = llm_extraction.normalize_extracted_personnel(st.session_state.llm_processed_df)
                    st.session_state.normalized_df = normalized_df
                    st.write("Normalization finished.")
                    logger.info("Normalization complete.")

                    # Display sample of normalized data
                    st.write("Sample of data after normalization:")
                    st.dataframe(normalized_df[['summary', 'start_time', 'duration_hours', 'extracted_personnel', 'assigned_personnel']].head())

                    # 4. Explode by Personnel for Analysis
                    st.write("Step 4: Preparing data for analysis (exploding by personnel)...")
                    # Use the 'assigned_personnel' column created by normalization
                    analysis_ready_df = data_processor.explode_by_personnel(normalized_df, personnel_col='assigned_personnel')
                    if analysis_ready_df is None or analysis_ready_df.empty:
                        st.error("Exploding data failed or resulted in empty data. Check logs.")
                        st.stop()
                    st.session_state.analysis_ready_df = analysis_ready_df
                    st.write(f"Data ready for analysis: {len(analysis_ready_df)} rows after exploding.")
                    logger.info(f"Explosion complete: {len(analysis_ready_df)} analysis rows.")

                    # Mark processing as complete
                    st.session_state.data_processed = True
                    
                    # 5. Save processed data to PostgreSQL database (if enabled)
                    if settings.DB_ENABLED:
                        from functions import db_manager
                        st.write("Step 5: Saving processed data to PostgreSQL database...")
                        try:
                            db_save_success = db_manager.save_processed_data_to_db(analysis_ready_df)
                            if db_save_success:
                                st.success(f"Successfully saved {len(analysis_ready_df)} records to PostgreSQL database at {settings.DB_HOST}.")
                                logger.info(f"Successfully saved {len(analysis_ready_df)} records to database.")
                            else:
                                st.warning("Failed to save data to PostgreSQL database. Check logs for details.")
                                logger.warning("Database save operation returned False.")
                        except Exception as db_e:
                            st.error(f"Error saving to database: {db_e}")
                            logger.error(f"Database save error: {db_e}", exc_info=True)
                    
                    end_time = time.time()
                    st.success(f"Pipeline finished successfully in {end_time - start_time:.2f} seconds! Navigate to the 'Analysis' page.")
                    logger.info(f"Processing pipeline finished in {end_time - start_time:.2f} seconds.")

                    # Optional: Show unknown assignments count
                    unknown_df = analysis_ready_df[analysis_ready_df['personnel'].isin(['Unknown', 'Unknown_Error'])]
                    if not unknown_df.empty:
                        st.warning(f"Found {len(unknown_df)} event assignments marked as 'Unknown' or 'Unknown_Error' after processing.")

                except Exception as e:
                    st.error(f"An error occurred during processing: {e}")
                    logger.error(f"Pipeline error: {e}", exc_info=True)
                    st.session_state.data_processed = False # Mark as failed
        else:
            st.warning("No raw data loaded to process.")
else:
    st.info("Upload a `calendar.json` file to enable processing.")

# Display status at the end too
st.markdown("---")
st.subheader("Current Status")
if st.session_state.get('data_processed') and st.session_state.get('analysis_ready_df') is not None:
     st.success(f"Data from '{st.session_state.uploaded_filename}' is processed ({len(st.session_state.analysis_ready_df)} analysis rows).")
elif st.session_state.get('raw_df') is not None:
     st.info(f"Data from '{st.session_state.uploaded_filename}' loaded but not processed.")
else:
     st.warning("No data loaded.")