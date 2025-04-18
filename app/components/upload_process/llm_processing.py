"""
LLM processing operations for the Upload & Process page.
Contains functions for extracting information using LLM.
"""
import streamlit as st
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

def process_events_with_llm(preprocessed_df, llm_client, canonical_names, db_manager, 
                           current_batch_id, settings, intermediate_save_batch_size=50):
    """
    Process events using LLM to extract personnel names and event types.
    
    Args:
        preprocessed_df: Preprocessed DataFrame
        llm_client: LLM client instance
        canonical_names: List of canonical personnel names
        db_manager: Database manager module
        current_batch_id: Current batch ID
        settings: Application settings
        intermediate_save_batch_size: Size of intermediate batches to save
        
    Returns:
        DataFrame: Processed DataFrame with LLM extraction results
    """
    from functions.llm_extraction.extractor import _extract_single_physicist_llm
    from functions.llm_extraction.normalizer import normalize_event_type
    
    summaries = preprocessed_df['summary'].tolist()
    total_summaries = len(summaries)
    extraction_errors = 0
    processed_batch_data = []  # Temp list to hold batch results
    
    st_progress_bar = st.progress(0, text="Starting LLM extraction...")
    
    for i, summary in enumerate(summaries):
        event_data = preprocessed_df.iloc[i].to_dict()  # Get original event data
        try:
            logger.info(f"Processing event {i+1}/{total_summaries}...")
            personnel_result, event_type_result = _extract_single_physicist_llm(summary, llm_client, canonical_names)
            normalized_event_type_str = normalize_event_type(event_type_result, settings.EVENT_TYPE_MAPPING)
            
            # Add extracted data to the event dictionary
            event_data['extracted_personnel'] = personnel_result
            event_data['extracted_event_type'] = normalized_event_type_str
            event_data['processing_status'] = 'extracted'  # Mark status
            
            logger.info(f"Event {i+1} result: Personnel={personnel_result}, RawEventType='{event_type_result}', NormalizedEventType='{normalized_event_type_str}'")
        except Exception as e:
            logger.error(f"Error processing event {i+1}: {e}", exc_info=True)
            event_data['extracted_personnel'] = ["Unknown_Error"]
            event_data['extracted_event_type'] = "Unknown"
            event_data['processing_status'] = 'error'  # Mark status
            extraction_errors += 1
        
        processed_batch_data.append(event_data)  # Add result to batch list
        
        # --- Incremental Save Logic ---
        is_final_event = (i + 1) == total_summaries
        batch_ready = len(processed_batch_data) >= intermediate_save_batch_size
        logger.info(f"Event {i+1}/{total_summaries}: Checking incremental save. DB_ENABLED={settings.DB_ENABLED}, Batch size={len(processed_batch_data)} (Ready={batch_ready}), Is final={is_final_event}")
        
        if settings.DB_ENABLED and (batch_ready or is_final_event):
            logger.info(f"Triggering intermediate save for {len(processed_batch_data)} records (Event {i+1}/{total_summaries})...")
            try:
                batch_df_to_save = pd.DataFrame(processed_batch_data)
                save_success = db_manager.save_partial_processed_data(batch_df_to_save, batch_id=current_batch_id)
                if save_success:
                    logger.info(f"Successfully saved intermediate batch of {len(processed_batch_data)} records for batch {current_batch_id}.")
                    st.info(f"Saved intermediate batch ({len(processed_batch_data)} events) to DB...")
                    processed_batch_data = []  # Clear the batch list
                else:
                    logger.error(f"Failed to save intermediate batch for batch {current_batch_id}.")
                    st.warning("Failed to save intermediate batch to DB.")
            except Exception as e:
                logger.error(f"Error saving intermediate batch: {e}", exc_info=True)
                st.error(f"Error saving intermediate batch: {e}")
        
        # Update progress
        progress_percent = int(((i + 1) / total_summaries) * 100)
        st_progress_bar.progress(progress_percent / 100, text=f"Extracting event {i+1}/{total_summaries}...")
    
    st_progress_bar.progress(1.0, text="LLM extraction complete.")
    st.success(f"LLM extraction finished. Errors: {extraction_errors}")
    logger.info(f"LLM extraction complete. Errors: {extraction_errors}")
    
    # Fetch processed data from database or create a fallback DataFrame
    if settings.DB_ENABLED:
        logger.info(f"Fetching processed data from DB for batch {current_batch_id} to create final llm_processed_df")
        llm_processed_df = db_manager.get_processed_events_by_batch(current_batch_id)
        if llm_processed_df is None or llm_processed_df.empty:
            logger.error(f"Failed to fetch back processed data for batch {current_batch_id} after loop. Normalization might fail.")
            # Fallback if fetch fails
            llm_processed_df = preprocessed_df.copy()
            llm_processed_df['extracted_personnel'] = [['Unknown']] * len(llm_processed_df)
            llm_processed_df['extracted_event_type'] = ['Unknown'] * len(llm_processed_df)
        else:
            # Ensure the 'extracted_personnel' column is list type if loaded from DB
            if 'extracted_personnel' in llm_processed_df.columns:
                llm_processed_df['extracted_personnel'] = llm_processed_df['extracted_personnel'].apply(
                    lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else ['Unknown']
                )
            logger.info(f"Successfully fetched {len(llm_processed_df)} records from DB for final llm_processed_df.")
    else:
        # If DB not enabled, we need to build llm_processed_df from the loop results
        logger.warning("DB not enabled, llm_processed_df might be incomplete if process was interrupted.")
        llm_processed_df = preprocessed_df.copy()  # Placeholder
        llm_processed_df['extracted_personnel'] = [['Unknown']] * len(llm_processed_df)
        llm_processed_df['extracted_event_type'] = ['Unknown'] * len(llm_processed_df)
    
    # DEBUG: Inspect DataFrame after LLM extraction
    if llm_processed_df is not None and not llm_processed_df.empty:
        logger.info(f"DataFrame columns after LLM extraction step: {llm_processed_df.columns.tolist()}")
        logger.info(f"First 5 rows of 'extracted_event_type' after LLM:\n{llm_processed_df['extracted_event_type'].head().to_string()}")
    else:
        logger.warning("llm_processed_df is None or empty after LLM extraction step.")
    
    return llm_processed_df

def check_llm_status(settings, is_llm_ready_fn, functions_module):
    """
    Check if LLM is available and configured.
    
    Args:
        settings: Application settings
        is_llm_ready_fn: Function to check if LLM is ready
        functions_module: Functions module containing LLM constants
        
    Returns:
        tuple: (llm_enabled, llm_ok)
    """
    llm_enabled = settings.LLM_PROVIDER and settings.LLM_PROVIDER.lower() != 'none'
    llm_ok = True
    
    if llm_enabled:
        # Check if OLLAMA_AVAILABLE constant exists before using it
        ollama_lib_available = getattr(functions_module, 'OLLAMA_AVAILABLE', False)
        if not is_llm_ready_fn():
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
            st.session_state.skip_llm = st.checkbox(
                "Proceed without LLM extraction (all personnel will be marked as 'Unknown')",
                value=True,
                key="skip_llm_checkbox"
            )

            if not st.session_state.skip_llm:
                llm_ok = False  # Only block processing if user unchecks the box
    
    return llm_enabled, llm_ok
