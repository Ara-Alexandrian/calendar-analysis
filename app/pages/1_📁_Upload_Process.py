"""
Replacement Upload & Process page that uses the direct extractor
for guaranteed console output during LLM processing.
"""
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
# Specific imports for LLM processing
# from functions.llm_extraction.sequential_processor import SequentialProcessor # <-- Bypass this
from functions.llm_extraction.extractor import _extract_single_physicist_llm # <-- Import directly
from functions.llm_extraction.ollama_client import get_ollama_client # <-- Correct path for client getter
from functions.llm_extraction.normalizer import normalize_extracted_personnel, normalize_event_type # Re-confirming this import
from functions.llm_extraction.client import is_llm_ready # Keep this for the readiness check
# Import the package itself if needed for constants like OLLAMA_AVAILABLE
import functions.llm_extraction
from functions import config_manager # Needed to check config status
from config import settings # For LLM provider check, and EVENT_TYPE_MAPPING
import functions.db_manager as db_manager
# No need to reload db_manager usually unless actively developing it
# importlib.reload(db_manager)

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
        file_content_bytes = uploaded_file.getvalue() # Read as bytes
        st.session_state.uploaded_file_content = file_content_bytes # Store bytes in session

        # Calculate hash here for later use
        import hashlib
        st.session_state.uploaded_file_hash = hashlib.sha256(file_content_bytes).hexdigest()

        # Load raw data using the modified function (pass bytes)
        raw_df = data_processor.load_raw_calendar_data(file_content_bytes, uploaded_file.name)

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

    # Check LLM status if provider is configured
    llm_ok = True
    llm_enabled = settings.LLM_PROVIDER and settings.LLM_PROVIDER.lower() != 'none'
    if llm_enabled:
        # Check if OLLAMA_AVAILABLE constant exists before using it
        ollama_lib_available = getattr(functions.llm_extraction, 'OLLAMA_AVAILABLE', False)
        if not is_llm_ready():
            if not ollama_lib_available:
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

    # Only enable button if config is okay
    process_button_disabled = not config_ok or not llm_ok
    tooltip_text = "Personnel configuration missing. Check Admin page." if not config_ok else "Run the full data processing pipeline"
    if not llm_ok:
        tooltip_text = "LLM is required but not available. Check the error message."

    if st.button("Process Data", key="process_button", disabled=process_button_disabled, help=tooltip_text):
        if st.session_state.raw_df is not None:
            logger.info("Starting data processing pipeline...")
            start_time = time.time()
            with st.spinner("Processing data... This may take a while for the initial steps."):
                try:
                    # 0. Save Calendar File Info to DB (if enabled) - Moved here
                    if settings.DB_ENABLED:
                        st.write("Registering uploaded file in database...")
                        try:
                            success, batch_id, is_duplicate = db_manager.save_calendar_file_to_db(
                                filename=st.session_state.uploaded_filename,
                                file_content=st.session_state.uploaded_file_content # Use bytes from session
                            )
                            if success:
                                st.session_state.current_batch_id = batch_id # Store the batch_id from DB save
                                if is_duplicate:
                                    st.info(f"File '{st.session_state.uploaded_filename}' has been processed before (Batch ID: {batch_id}). Re-processing.")
                                else:
                                    st.info(f"File '{st.session_state.uploaded_filename}' registered with Batch ID: {batch_id}")
                            else:
                                st.error("Failed to register file in database. Processing will continue but might lack DB tracking.")
                                # Assign a temporary batch ID if DB save fails
                                import uuid
                                st.session_state.current_batch_id = f"temp_batch_{uuid.uuid4().hex[:8]}"
                        except Exception as db_save_e:
                            st.error(f"Error registering file in database: {db_save_e}")
                            logger.error(f"Error in save_calendar_file_to_db: {db_save_e}", exc_info=True)
                            # Assign a temporary batch ID if DB save fails
                            import uuid
                            st.session_state.current_batch_id = f"temp_batch_{uuid.uuid4().hex[:8]}"

                    # 1. Preprocess Data
                    st.write("Step 1: Preprocessing data (dates, duration)...")
                    preprocessed_df = data_processor.preprocess_data(st.session_state.raw_df)
                    if preprocessed_df is None or preprocessed_df.empty:
                        st.error("Preprocessing failed or resulted in empty data. Check logs.")
                        st.stop()
                    st.session_state.preprocessed_df = preprocessed_df
                    st.write(f"Preprocessing done: {len(preprocessed_df)} events remaining.")
                    logger.info(f"Preprocessing complete: {len(preprocessed_df)} events.")

                    # Batch ID is now set earlier during DB save or fallback

                    # 2. LLM Extraction
                    if llm_enabled and llm_ok:
                        st.write("Step 2: Extracting personnel names from calendar events (Simplified Loop)...")
                        try:
                            st.info("Processing calendar entries with simplified LLM extraction loop...")
                            llm_client = get_ollama_client() # Get the client
                            if not llm_client:
                                raise Exception("Failed to get LLM Client.")
                            
                            summaries = preprocessed_df['summary'].tolist()
                            canonical_names_list = st.session_state.get('canonical_names', [])
                            personnel_results = [] # List for personnel names
                            event_type_results = [] # List for event types
                            total_summaries = len(summaries)
                            extraction_errors = 0

                            st_progress_bar = st.progress(0, text="Starting simplified extraction...")

                            for i, summary in enumerate(summaries):
                                try:
                                    logger.info(f"Simple Loop - Processing event {i+1}/{total_summaries}...")
                                    # Directly call the core extraction logic (no retry here for simplicity)
                                    # Unpack the tuple result
                                    personnel_result, event_type_result = _extract_single_physicist_llm(summary, llm_client, canonical_names_list)
                                    # Normalize the event type *before* appending
                                    normalized_event_type_str = normalize_event_type(event_type_result, settings.EVENT_TYPE_MAPPING)
                                    personnel_results.append(personnel_result) # Append personnel list
                                    event_type_results.append(normalized_event_type_str) # Append normalized type
                                    logger.info(f"Simple Loop - Event {i+1} result: Personnel={personnel_result}, RawEventType='{event_type_result}', NormalizedEventType='{normalized_event_type_str}'")
                                except Exception as loop_exc:
                                     logger.error(f"Simple Loop - Error processing event {i+1}: {loop_exc}", exc_info=True)
                                     personnel_results.append(["Unknown_Error"])
                                     event_type_results.append("Unknown") # Append normalized unknown
                                     extraction_errors += 1

                                # Update progress
                                progress_percent = int(((i + 1) / total_summaries) * 100)
                                st_progress_bar.progress(progress_percent, text=f"Extracting event {i+1}/{total_summaries}...")
                                time.sleep(0.1) # Small delay

                            st_progress_bar.progress(100, text="Simplified extraction complete.")
                            
                            # Create the dataframe with results
                            llm_processed_df = preprocessed_df.copy()
                            llm_processed_df['extracted_personnel'] = personnel_results
                            # Use a clearer name for the column holding the *normalized* event type
                            llm_processed_df['event_type'] = event_type_results
                            st.session_state.llm_processed_df = llm_processed_df

                            st.success(f"Simplified LLM extraction finished. Errors: {extraction_errors}")
                            logger.info(f"Simplified LLM extraction complete. Errors: {extraction_errors}")

                        except Exception as e:
                            logger.error(f"Simplified LLM extraction loop failed: {e}", exc_info=True) # Log full traceback
                            st.error(f"Extraction failed: {e}. Using Unknown for all events.")
                            # Create placeholder with Unknown values
                            temp_df = preprocessed_df.copy()
                            temp_df['extracted_personnel'] = [['Unknown']] * len(temp_df)
                            st.session_state.llm_processed_df = temp_df
                            llm_processed_df = temp_df # Ensure llm_processed_df is assigned
                    else:
                        st.warning("Skipping LLM extraction (disabled or not ready). Assigning 'Unknown'.")
                        # Create placeholder columns if skipping LLM
                        temp_df = preprocessed_df.copy()
                        temp_df['extracted_personnel'] = [['Unknown']] * len(temp_df)
                        st.session_state.llm_processed_df = temp_df # Use preprocessed df as base
                        llm_processed_df = temp_df # Ensure llm_processed_df is assigned

                    # 3. Normalize Extracted Personnel Names (Event type already normalized in step 2)
                    st.write("Step 3: Normalizing extracted personnel names...")

                    # Ensure llm_processed_df is a DataFrame before proceeding
                    if not isinstance(llm_processed_df, pd.DataFrame):
                        logger.error(f"llm_processed_df is not a DataFrame after Step 2, it's a {type(llm_processed_df)}. Falling back.")
                        st.error(f"Internal error: LLM processing step did not return a DataFrame. Using default values.")
                        # Fallback: Create DataFrame with 'Unknown' assigned personnel
                        temp_df = preprocessed_df.copy()
                        temp_df['extracted_personnel'] = [['Unknown']] * len(temp_df)
                        temp_df['assigned_personnel'] = [['Unknown']] * len(temp_df)
                        st.session_state.llm_processed_df = temp_df # Store the fallback df
                        st.session_state.normalized_df = temp_df
                        normalized_df = temp_df # Ensure normalized_df is assigned
                    else:
                        # Proceed with normalization
                        try:
                            # Call the correctly imported function
                            normalized_df = normalize_extracted_personnel(llm_processed_df)
                            st.session_state.normalized_df = normalized_df
                            st.write("Normalization finished.")
                            logger.info("Normalization complete.")
                        except Exception as e:
                            logger.error(f"Normalization error: {e}", exc_info=True)
                            st.error(f"Error during normalization: {e}. Using default values.")
                            # Fallback: Create a simple normalized_df with assigned_personnel column
                            temp_df = llm_processed_df.copy() # Use the df from LLM step
                            temp_df['assigned_personnel'] = [['Unknown']] * len(temp_df)
                            st.session_state.normalized_df = temp_df
                            normalized_df = temp_df # Ensure normalized_df is assigned

                    # Display sample of normalized data (ensure normalized_df exists)
                    if normalized_df is not None and not normalized_df.empty:
                         st.write("Sample of data after normalization:")
                         # Update display columns to show the normalized 'event_type'
                         display_cols = [col for col in ['summary', 'start_time', 'duration_hours', 'extracted_personnel', 'event_type', 'assigned_personnel'] if col in normalized_df.columns]
                         st.dataframe(normalized_df[display_cols].head())
                    else:
                         st.warning("No data available to display after normalization step.")


                    # 4. Explode by Personnel for Analysis
                    st.write("Step 4: Preparing data for analysis (exploding by personnel)...")
                    # Use the 'assigned_personnel' column created by normalization
                    # Use normalized_df which is guaranteed to exist at this point
                    if normalized_df is not None and 'assigned_personnel' in normalized_df.columns:
                        analysis_ready_df = data_processor.explode_by_personnel(normalized_df, personnel_col='assigned_personnel')
                        if analysis_ready_df is None or analysis_ready_df.empty:
                            st.error("Exploding data failed or resulted in empty data. Check logs.")
                            st.stop()
                        st.session_state.analysis_ready_df = analysis_ready_df
                        st.write(f"Data ready for analysis: {len(analysis_ready_df)} rows after exploding.")
                        logger.info(f"Explosion complete: {len(analysis_ready_df)} analysis rows.")
                    else:
                        st.error("Cannot prepare data for analysis: 'assigned_personnel' column missing after normalization.")
                        st.stop()

                    # Mark processing as complete and ensure data is available for analysis
                    st.session_state.data_processed = True

                    # Always store the analysis dataframe in session state for the Analysis page
                    if 'analysis_data' not in st.session_state:
                        st.session_state.analysis_data = {}

                    # Store the current batch with a timestamp for the Analysis page
                    current_batch_key = f"batch_{int(time.time())}" # Use timestamp as part of key
                    st.session_state.analysis_data[current_batch_key] = {
                        'df': analysis_ready_df,
                        'timestamp': time.time(),
                        'source': st.session_state.uploaded_filename,
                        'processed_count': len(analysis_ready_df)
                    }

                    # Store the most recent batch ID for easy access
                    st.session_state.most_recent_batch = current_batch_key

                    # 5. Save processed data to PostgreSQL database (if enabled)
                    if settings.DB_ENABLED:
                        # from functions import db_manager # Already imported
                        st.write("Step 5: Saving processed data to PostgreSQL database...")
                        try:
                            # Use the new batch saving function with progress indicators
                            batch_size = 500  # Save in batches of 500 records
                            total_records = len(analysis_ready_df)
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()

                            # Save in batches with visual feedback
                            status_text.write("Starting database save operation in batches...")

                            # Use the batch ID generated earlier
                            current_db_batch_id = st.session_state.get('current_batch_id', current_batch_key) # Use generated batch_id

                            db_save_success, saved_count = db_manager.save_processed_data_in_batches(
                                analysis_ready_df,
                                batch_id=current_db_batch_id,
                                batch_size=batch_size
                            )

                            # Update progress bar to 100% when complete
                            progress_bar.progress(1.0)

                            if db_save_success:
                                st.success(f"Successfully saved {saved_count}/{total_records} records to PostgreSQL database at {settings.DB_HOST}.")
                                logger.info(f"Successfully saved {saved_count}/{total_records} records to database.")

                                # Mark the calendar file as processed only if saving was successful
                                if current_db_batch_id and hasattr(db_manager, 'mark_calendar_file_as_processed'):
                                    db_manager.mark_calendar_file_as_processed(current_db_batch_id)
                            else:
                                st.warning(f"Database save incomplete: Only saved {saved_count}/{total_records} records. Check logs for details.")
                                logger.warning(f"Database save operation incomplete: {saved_count}/{total_records} records saved.")
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
