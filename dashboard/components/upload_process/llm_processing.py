"""
LLM processing operations for the Upload & Process page.
Contains functions for extracting information using LLM.
"""
import streamlit as st
import pandas as pd
import logging
import time
import json # Added for saving results

# Import the new service
from src.application.services.extraction_service import ExtractionService

logger = logging.getLogger(__name__)

def process_events_with_llm(preprocessed_df, llm_client, canonical_names, db_manager,
                           current_batch_id, settings, intermediate_save_batch_size=50):
    """
    Process events using LLM to extract personnel names and event types using ExtractionService.

    Args:
        preprocessed_df: Preprocessed DataFrame
        llm_client: LLM client instance (Note: No longer directly used, service handles client)
        canonical_names: List of canonical personnel names
        db_manager: Database manager module
        current_batch_id: Current batch ID
        settings: Application settings
        intermediate_save_batch_size: Size of intermediate batches to save (Note: Service handles batching internally)

    Returns:
        DataFrame: Processed DataFrame with LLM extraction results
    """
    # Instantiate the ExtractionService
    # Pass canonical_names if available, otherwise service loads them
    extraction_service = ExtractionService(canonical_names=canonical_names)

    if not extraction_service.llm_client:
        st.error("LLM Client could not be initialized by the Extraction Service.")
        # Fallback: Return original df with error columns
        llm_processed_df = preprocessed_df.copy()
        llm_processed_df['extracted_personnel'] = json.dumps(["Unknown_Error"])
        llm_processed_df['extracted_event_type'] = "Unknown_Error"
        return llm_processed_df

    try:
        logger.info(f"Starting LLM extraction using ExtractionService for batch {current_batch_id}...")
        start_time = time.time()

        # Use the service's dataframe processing method
        # The service handles parallel processing and progress updates internally (via Streamlit elements)
        llm_processed_df = extraction_service.extract_from_dataframe(
            preprocessed_df.copy(), # Pass a copy to avoid modifying original
            description_column='summary'
        )

        elapsed_time = time.time() - start_time
        logger.info(f"ExtractionService completed processing in {elapsed_time:.2f} seconds.")
        st.success(f"LLM extraction finished in {elapsed_time:.2f} seconds.")

        # --- Save results (if DB enabled) ---
        # The service returns the df with results; saving happens after
        if settings.DB_ENABLED and llm_processed_df is not None and not llm_processed_df.empty:
            logger.info(f"Saving {len(llm_processed_df)} processed records to DB for batch {current_batch_id}...")
            try:
                # Add batch_id and status before saving
                llm_processed_df['batch_id'] = current_batch_id
                # Determine status based on results (simplified check)
                def determine_status(row):
                    personnel = json.loads(row.get('extracted_personnel', '[]'))
                    event_type = row.get('extracted_event_type', 'Unknown')
                    if "Error" in personnel or event_type == "Error":
                        return "error"
                    elif "Unknown" in personnel or event_type == "Unknown":
                        return "extracted_unknown" # More specific status
                    else:
                        return "extracted"
                llm_processed_df['processing_status'] = llm_processed_df.apply(determine_status, axis=1)

                save_success = db_manager.save_partial_processed_data(llm_processed_df, batch_id=current_batch_id)
                if save_success:
                    logger.info(f"Successfully saved final batch of {len(llm_processed_df)} records for batch {current_batch_id}.")
                else:
                    logger.error(f"Failed to save final batch for batch {current_batch_id}.")
                    st.warning("Failed to save final batch results to DB.")
            except Exception as e:
                logger.error(f"Error saving final batch: {e}", exc_info=True)
                st.error(f"Error saving final batch results: {e}")
        elif not settings.DB_ENABLED:
             logger.info("DB not enabled, skipping final save.")
        # ------------------------------------

    except Exception as e:
        logger.error(f"Error during ExtractionService processing: {e}", exc_info=True)
        st.error(f"An error occurred during LLM extraction: {e}")
        # Fallback: Return original df with error columns
        llm_processed_df = preprocessed_df.copy()
        llm_processed_df['extracted_personnel'] = json.dumps(["Unknown_Error"])
        llm_processed_df['extracted_event_type'] = "Unknown_Error"

    # DEBUG: Inspect DataFrame after LLM extraction
    if llm_processed_df is not None and not llm_processed_df.empty:
        logger.info(f"DataFrame columns after LLM extraction step: {llm_processed_df.columns.tolist()}")
        # Ensure columns exist before logging head
        if 'extracted_event_type' in llm_processed_df.columns:
            logger.info(f"First 5 rows of 'extracted_event_type' after LLM:\n{llm_processed_df['extracted_event_type'].head().to_string()}")
        if 'extracted_personnel' in llm_processed_df.columns:
             logger.info(f"First 5 rows of 'extracted_personnel' after LLM:\n{llm_processed_df['extracted_personnel'].head().to_string()}")
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
