# pages/1_📁_Upload_Process.py
import streamlit as st
import pandas as pd
import logging
import time

# Ensure project root is in path (if running page directly)
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functions import data_processor
from functions import llm_extractor
from functions import config_manager # Needed to check config status
from config import settings # For LLM provider check

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
        if not llm_extractor.is_llm_ready():
             st.warning(f"LLM ({settings.LLM_PROVIDER}) is configured but not reachable at {settings.OLLAMA_BASE_URL}. Name extraction will be skipped or may fail.")
             # llm_ok = False # Allow processing without LLM for now, will assign 'Unknown'

    # Only enable button if config is okay (or allow proceed with warning?)
    # Disable button if config is absolutely required and missing.
    process_button_disabled = not config_ok # Disable if config is missing
    tooltip_text = "Personnel configuration missing. Check Admin page." if not config_ok else "Run the full data processing pipeline"

    if st.button("Process Data", key="process_button", disabled=process_button_disabled, help=tooltip_text):
        if st.session_state.raw_df is not None:
            logger.info("Starting data processing pipeline...")
            start_time = time.time()
            with st.spinner("Processing data... This may take a while, especially LLM extraction."):
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

                    # 2. LLM Extraction (if enabled and ready)
                    if llm_enabled and llm_ok: # Only run if enabled and client check passed initially
                         st.write("Step 2: Extracting personnel names using LLM...")
                         llm_processed_df = llm_extractor.run_llm_extraction_parallel(preprocessed_df)
                         st.session_state.llm_processed_df = llm_processed_df
                         st.write("LLM extraction finished.")
                         logger.info("LLM extraction complete.")
                    else:
                         st.warning("Skipping LLM extraction (disabled or not ready). Assigning 'Unknown'.")
                         # Create placeholder columns if skipping LLM
                         preprocessed_df['extracted_personnel'] = [['Unknown']] * len(preprocessed_df)
                         st.session_state.llm_processed_df = preprocessed_df # Use preprocessed df as base
                         logger.info("Skipped LLM extraction.")


                    # 3. Normalize Extracted Personnel Names
                    st.write("Step 3: Normalizing extracted names...")
                    normalized_df = llm_extractor.normalize_extracted_personnel(st.session_state.llm_processed_df)
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