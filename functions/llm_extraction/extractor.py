# functions/llm_extraction/extractor.py
"""
Core LLM extraction functionality for extracting personnel names from calendar events.
"""
import logging
import json
from json.decoder import JSONDecodeError
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import streamlit as st
import threading

# Import from config and other modules
from config import settings
from functions import config_manager
from .client import get_llm_client, is_llm_ready, restart_ollama_server
from .utils import get_persistent_progress_bar

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    from .utils import SimpleTqdm as tqdm

# Configure logging
logger = logging.getLogger(__name__)

def _extract_single_physicist_llm(summary: str, llm_client, canonical_names: list[str]) -> tuple[list[str], str]:
    """
    Internal function to extract names and event type for a single summary using the LLM.
    Returns tuple: (list of personnel names, event_type string) on success,
    or (["Unknown"], "Unknown") / (["Unknown_Error"], "Unknown_Error") on failure.
    """
    if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
        logger.debug("Skipping extraction for empty or non-string summary.")
        return ["Unknown"], "Unknown" # Return tuple

    if not llm_client:
        logger.error("LLM client is None in _extract_single_physicist_llm.")
        return ["Unknown_Error"], "Unknown_Error" # Return tuple

    # Define the list of valid event types provided by the user (comprehensive list)
    valid_event_types = [
        "Barrigel", "Barrigel OLOL OR", "BR 4D", "BR 4D BH", "BR 4D poss BH", "BR 4D RTP",
        "BR 4D RTP BH", "BR BH", "BR BH 4D", "BR GK SIM", "BR RTP 4D", "BR RTP UNITY",
        "BR Seed Implant", "BR UNITY CT", "BR UNITY RTP", "BR UNITY RTP 4D", "BR Volume Study",
        "BR2 SBRT", "BR3 SBRT", "BRG SEED IMPLANT", "CC Chart Rounds", "COV 4D", "COV BH",
        "COV HDR", "COV HDR T&R", "COV RTP BH", "COV SBRT", "CV1 SBRT", "CV2 SBRT",
        "Framed GK Tx", "GK Remote Plan Review", "GK SIM", "GK SIM/Tx", "GK SBRT Tx",
        "GK Tx", "GK Tx SBRT", "GON 4D", "GON BH", "GON Pluvicto", "GON SBRT", "GON Xofigo",
        "HAM 4D", "HAM 4D RTP", "Ham 4D poss BH", "Ham BH", "Ham SBRT", "Hammond SBRT",
        "HOU 4D", "HOU BH", "HOU BH 4D", "HOU SBRT", "Post GK", "Post GK MRI",
        "Resident's MR-in-Radiotherapy Workshop", "Seed Implant", "SpaceOAR Classic",
        "SpaceOAR Vue", "WH BH", "WH Breast Compression", "WH CT/RTP HDR", "WH HDR",
        "WH HDR CYL", "WH HDR RTP CYL", "WH HDR T&O", "WH RTP BH", "WH SBRT",
        "CS UNITY", "CS/SS UNITY", "DN UNITY", "JV UNITY", "JV Unity", "SS UNITY",
        "SS/CS Unity", "SS/DN UNITY"
    ]

    # Construct the prompt
    prompt = f"""
    Your task is to analyze the provided calendar event summary and extract two pieces of information:
    1.  **Personnel Names:** Identify ONLY the canonical physicist names from the provided list that are mentioned or clearly referenced in the summary. Consider variations (initials, last names) but map them back EXACTLY to a name in the list. If multiple names are found, include all. If none are found, use an empty list `[]`.
    2.  **Event Type:** Identify the primary service or event type described in the summary. Choose the BEST match ONLY from the 'Valid Event Types' list provided below. If no suitable match is found in the list, return "Unknown".

    Known Canonical Physicist Names:
    {json.dumps(canonical_names, indent=2)}

    Valid Event Types (Choose ONLY from this list):
    {json.dumps(valid_event_types, indent=2)}

    Event Summary:
    "{summary}"

    IMPORTANT: Respond ONLY with a single, valid JSON object containing two keys: "personnel" (a JSON array of strings) and "event_type" (a JSON string).

    Examples of CORRECT responses:
    {{"personnel": [], "event_type": "Unknown"}}
    {{"personnel": ["D. Solis"], "event_type": "GK Tx SBRT"}}
    {{"personnel": ["C. Chu", "Wood"], "event_type": "SpaceOAR VUE"}}
    {{"personnel": ["G. Pitcher", "E. Chorniak"], "event_type": "WH HDR CYL"}}

    Examples of INCORRECT responses (DO NOT USE THESE FORMATS):
    ["D. Solis"] # Incorrect: Not a JSON object
    {{"names": ["D. Solis"]}} # Incorrect: Wrong key name
    "Post GK" # Incorrect: Not a JSON object

    Absolutely NO surrounding text or explanations. Your entire response must be JUST the JSON object.
    """

    # --- Start LLM Call ---
    logger.debug(f"Attempting LLM extraction for summary: '{summary[:50]}...'")
    logger.debug(f"Prompt sent to LLM:\n{prompt}") # Log the full prompt for debugging

    try:
        start_time = time.time()
        timeout_seconds = 30

        response = llm_client.chat(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an assistant that extracts specific personnel names and the event type from text based on a provided list and context. You output ONLY a valid JSON object with keys 'personnel' (list of strings) and 'event_type' (string)."},
                {"role": "user", "content": prompt}
            ],
            format='json',
            options={'temperature': 0.1, 'timeout': timeout_seconds}
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Ollama API call completed in {elapsed_time:.2f} seconds.")
        logger.debug(f"Raw LLM response object: {response}") # Log the full response object

        content = response.get('message', {}).get('content', '')
        logger.debug(f"Extracted LLM response content: {content}") # Log the content string

        if not content:
             logger.warning(f"LLM returned empty content for summary: '{summary[:50]}...'")
             return ["Unknown"], "Unknown"

        # --- Parse LLM Response ---
        extracted_personnel = []
        extracted_event_type = "Unknown"
        try:
            extracted_data = json.loads(content)
            logger.debug(f"Successfully parsed JSON content: {extracted_data}")

            if isinstance(extracted_data, dict):
                personnel_list = extracted_data.get('personnel', [])
                if isinstance(personnel_list, list):
                    extracted_personnel = [item for item in personnel_list if isinstance(item, str)]
                    logger.debug(f"Extracted personnel list: {extracted_personnel}")
                else:
                    logger.warning(f"LLM response 'personnel' key was not a list: {personnel_list}")
                    extracted_personnel = []

                event_type_str = extracted_data.get('event_type', 'Unknown')
                if isinstance(event_type_str, str) and event_type_str.strip():
                    extracted_event_type = event_type_str.strip()
                    logger.debug(f"Extracted event type: '{extracted_event_type}'")
                else:
                    logger.warning(f"LLM response 'event_type' key was not a valid string: {event_type_str}")
                    extracted_event_type = "Unknown"
            else:
                logger.warning(f"LLM (JSON mode) returned unexpected format (not dict): {content[:100]}")

        except JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON object response from LLM for '{summary[:30]}...': {content[:100]}. Error: {str(json_err)}")
            return ["Unknown_Error"], "Unknown_Error"
        except Exception as e:
             logger.error(f"Error processing LLM response content structure: {e}\nResponse content: {content[:100]}")
             return ["Unknown_Error"], "Unknown_Error"

        # --- Validate Extracted Data ---
        validated_names = [name for name in extracted_personnel if name in canonical_names]
        logger.debug(f"Validated personnel names against canonical list: {validated_names}")

        # Validate event type against the provided list
        if extracted_event_type not in valid_event_types:
            logger.warning(f"Extracted event type '{extracted_event_type}' not in valid list for summary '{summary[:50]}...'. Setting to Unknown.")
            validated_event_type = "Unknown"
        else:
            validated_event_type = extracted_event_type

        if not validated_names:
            logger.debug(f"LLM found no known physicists in: '{summary[:30]}...'")
            return ["Unknown"], validated_event_type # Return "Unknown" personnel, but keep validated event type
        else:
            logger.debug(f"LLM validated {validated_names} and event type '{validated_event_type}' in '{summary[:30]}...'")
            return validated_names, validated_event_type

    except Exception as e:
        logger.error(f"Error calling Ollama API or processing response for summary '{summary[:30]}...': {e}", exc_info=True) # Log full traceback
        return ["Unknown_Error"], "Unknown_Error" # Indicate an error occurred

# --- Remaining functions (extract_personnel_with_llm, run_llm_extraction_parallel, etc.) remain unchanged ---
# ... (Keep the rest of the file content as it was) ...

def extract_personnel_with_llm(summary: str, llm_client, canonical_names: list[str], retry_count=3) -> tuple[list[str], str]:
    """
    Extract personnel and event type from event data with improved error handling and retry logic.
    Returns a tuple: (personnel_list, event_type_string)
    """
    if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
        return ["Unknown"], "Unknown" # Return tuple for consistency

    try:
        result = _extract_single_physicist_llm(summary, llm_client, canonical_names)
        return result
    except Exception as e:
        # Log the error
        logger.error(f"LLM extraction error: {str(e)}")

        # Retry logic
        if retry_count > 0:
            logger.info(f"Retrying extraction for '{summary[:30]}...'. Attempts left: {retry_count-1}")
            time.sleep(2)  # Backoff before retry
            return extract_personnel_with_llm(summary, llm_client, canonical_names, retry_count-1)

        # Return fallback after all retries fail
        logger.error(f"All retry attempts failed for '{summary[:30]}...'")
        return ["Unknown_Error"], "Unknown_Error" # Return tuple for error

def run_llm_extraction_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs LLM extraction (personnel + event type) in parallel on the 'summary' column.
    If parallel processing fails, will fall back to sequential processing.

    Args:
        df: DataFrame with a 'summary' column.

    Returns:
        DataFrame with new 'extracted_personnel' and 'extracted_event_type' columns.
    """
    if not is_llm_ready():
        st.error("LLM client is not available. Cannot perform extraction.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        df['extracted_event_type'] = ["Unknown_Error"] * len(df)
        return df

    if 'summary' not in df.columns:
        logger.error("'summary' column not found in DataFrame. Cannot extract.")
        st.error("Input data is missing the 'summary' column.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        df['extracted_event_type'] = ["Unknown_Error"] * len(df)
        return df

    df_copy = df.copy()
    df_copy['summary'] = df_copy['summary'].fillna('') # Ensure no NaN summaries

    summaries = df_copy['summary'].tolist()
    personnel_results = [None] * len(summaries) # Initialize results lists
    event_type_results = [None] * len(summaries)

    llm_client = get_llm_client() # Get cached client
    _, _, canonical_names = config_manager.get_personnel_details() # Get current names

    if not canonical_names:
        logger.error("Cannot run LLM extraction: No canonical names found in configuration.")
        st.error("Personnel configuration is empty. Please configure personnel in the Admin page.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        df['extracted_event_type'] = ["Unknown_Error"] * len(df)
        return df

    start_time = time.time()
    logger.info(f"Starting parallel LLM extraction for {len(summaries)} summaries using {settings.LLM_MAX_WORKERS} workers...")
    st.info(f"Starting LLM name extraction for {len(summaries)} events using model '{settings.LLM_MODEL}'...")

    # Streamlit progress bar and persistent terminal progress bar
    progress_bar = st.progress(0, text="Initializing LLM extraction...")
    term_progress = get_persistent_progress_bar(len(summaries), "LLM Extraction")
    total_summaries = len(summaries)
    completed_count = 0

    # Test extraction with a single item first to validate the LLM connectivity
    logger.info("Testing LLM extraction with a single item first...")
    test_summary = summaries[0] if summaries else "Test event"
    try:
        test_result = extract_personnel_with_llm(test_summary, llm_client, canonical_names)
        logger.info(f"Test extraction successful: {test_result}")
        term_progress.update(1)  # Update the terminal progress bar
    except Exception as e:
        logger.error(f"Test extraction failed: {e}. Will try sequential mode with more robust error handling.")
        st.warning("Initial test of LLM connection failed. Using more cautious sequential processing mode.")
        term_progress.close()  # Close the terminal progress bar
        # Switch to sequential mode immediately if test fails
        try:
            return run_llm_extraction_sequential(df)
        except Exception as e2:
            logger.error(f"Sequential extraction also failed: {e2}")
            st.error(f"LLM extraction failed in both parallel and sequential modes. Check Ollama server connection.")
            df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
            return df

    try:
        with ThreadPoolExecutor(max_workers=settings.LLM_MAX_WORKERS) as executor:
            # Map futures to original index
            future_to_index = {
                executor.submit(extract_personnel_with_llm, summary, llm_client, canonical_names): i
                for i, summary in enumerate(summaries)
            }

            # Add timeout to detect stuck processes
            timeout_start = time.time()
            timeout_limit = 300  # 5 minutes timeout for the entire process

            for future in as_completed(future_to_index): # Use as_completed directly
                if time.time() - timeout_start > timeout_limit:
                    logger.error("Processing timeout: Exceeded 5 minutes.")
                    st.error("Processing timeout: Please try again or check the logs.")
                    break

                index = future_to_index[future]
                try:
                    personnel_list, event_type = future.result() # Unpack tuple
                    personnel_results[index] = personnel_list
                    event_type_results[index] = event_type
                except Exception as exc:
                    logger.error(f"Summary index {index} ('{summaries[index][:50]}...') generated an exception: {exc}")
                    personnel_results[index] = ["Unknown_Error"] # Mark both as error
                    event_type_results[index] = "Unknown_Error"

                # Update progress bar
                completed_count += 1
                term_progress.update(1)  # Update the terminal progress bar
                progress_percent = int((completed_count / total_summaries) * 100)
                progress_bar.progress(progress_percent, text=f"Processing event {completed_count}/{total_summaries}...")

            # Check if all results were processed correctly
            missing_personnel = [i for i, r in enumerate(personnel_results) if r is None]
            missing_event_type = [i for i, r in enumerate(event_type_results) if r is None]
            missing_indices = sorted(list(set(missing_personnel) | set(missing_event_type))) # Combine unique indices

            if missing_indices:
                logger.warning(f"{len(missing_indices)} events were not processed correctly in parallel mode. Processing them sequentially.")
                # Process missing results sequentially
                for i in missing_indices:
                    try:
                        personnel_list, event_type = extract_personnel_with_llm(summaries[i], llm_client, canonical_names) # Unpack tuple
                        personnel_results[i] = personnel_list
                        event_type_results[i] = event_type
                        # Update progress bar (ensure completed_count reflects actual completions)
                        # Note: completed_count might be slightly off if only one part failed, but okay for progress indication
                        if i not in future_to_index: # Only increment if it wasn't processed at all before
                             completed_count += 1
                        term_progress.update(1)  # Update the terminal progress bar
                        progress_percent = int((completed_count / total_summaries) * 100)
                        progress_bar.progress(progress_percent, text=f"Processing event {completed_count}/{total_summaries}...")
                    except Exception as e:
                        logger.error(f"Sequential fallback failed for index {i}: {e}")
                        personnel_results[i] = ["Unknown_Error"] # Mark both as error
                        event_type_results[i] = "Unknown_Error"

        progress_bar.progress(100, text="LLM Extraction Complete!")
        term_progress.close()  # Close the terminal progress bar
        end_time = time.time()
        logger.info(f"LLM extraction finished in {end_time - start_time:.2f} seconds.")
        st.success(f"LLM extraction finished in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"An error occurred during the parallel LLM execution: {e}")
        st.warning(f"Parallel processing error: {e}. Switching to sequential processing.")
        term_progress.close()  # Close the terminal progress bar

        # If parallel execution fails completely, fall back to sequential processing
        try:
            return run_llm_extraction_sequential(df)
        except Exception as e2:
            logger.error(f"Sequential extraction also failed: {e2}")
            st.error(f"LLM extraction failed in both parallel and sequential modes. Check Ollama server connection.")
            # Fill remaining results with error marker
            df_copy['extracted_personnel'] = [["Unknown_Error"]] * len(df_copy)
            return df_copy

    # Ensure all results are set
    for i in range(len(summaries)):
        if personnel_results[i] is None:
            personnel_results[i] = ["Unknown_Error"]
        if event_type_results[i] is None:
            event_type_results[i] = "Unknown_Error"

    df_copy['extracted_personnel'] = personnel_results
    df_copy['extracted_event_type'] = event_type_results
    return df_copy

# --- Keep ultra_basic_extraction, run_llm_extraction_sequential, run_llm_extraction_background ---
# ... (rest of the file content) ...

import os
import sys
import time
import json
import logging
import streamlit as st

# Ensure path includes the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if (PROJECT_ROOT not in sys.path):
    sys.path.insert(0, PROJECT_ROOT)

from config import settings
from functions import config_manager
from .client import get_llm_client, is_llm_ready

logger = logging.getLogger(__name__)

def ultra_basic_extraction(df):
    """
    Minimal extraction function with no fancy handling, just basic calls.
    """
    st.warning("Using ultra-basic extraction method")

    # Copy dataframe to avoid modifying the original
    df_copy = df.copy()

    # Get names from config
    _, _, canonical_names = config_manager.get_personnel_details()
    if not canonical_names:
        logger.error("Cannot run ultra_basic_extraction: No canonical names found.")
        st.error("Personnel configuration is empty.")
        df_copy['extracted_personnel'] = [["Unknown_Error"]] * len(df_copy)
        return df_copy

    # Get LLM client
    client = get_llm_client()
    if not client:
        logger.error("Cannot run ultra_basic_extraction: Failed to get LLM client.")
        st.error("Failed to connect to LLM.")
        df_copy['extracted_personnel'] = [["Unknown_Error"]] * len(df_copy)
        return df_copy

    # List to store results
    results = []
    total_rows = len(df_copy)

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text(f"Starting processing of {total_rows} events...")

    # Process each row
    for i, row_tuple in enumerate(df_copy.iterrows()):
        index, data = row_tuple # Get index and data from tuple
        percent_complete = int(((i + 1) / total_rows) * 100)
        status_text.text(f"Processing event {i+1} of {total_rows}...")

        summary = data.get('summary', '')
        logger.info(f"[UltraBasic] Processing event {i+1}/{total_rows} (Index: {index}): {summary[:50]}...")

        if not summary or len(summary.strip()) == 0:
            logger.warning(f"[UltraBasic] Skipping event {i+1} due to empty summary.")
            results.append(["Unknown"])
            progress_bar.progress(percent_complete)
            continue

        try:
            prompt = f"""
            Extract physicist names from this summary: "{summary}"
            Only return names from this list: {json.dumps(canonical_names)}
            Format as JSON array. If none found, return empty array.
            """

            logger.debug(f"[UltraBasic] Calling LLM for event {i+1}...")
            response = client.chat(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Extract names, respond with only a JSON array."},
                    {"role": "user", "content": prompt}
                ],
                format="json",
                options={'timeout': 60} # Add a timeout
            )
            logger.debug(f"[UltraBasic] LLM call complete for event {i+1}.")

            content = response.get('message', {}).get('content', '')
            logger.info(f"[UltraBasic] LLM response for event {i+1}: {content[:100]}")

            if not content:
                 logger.warning(f"[UltraBasic] Empty content received for event {i+1}")
                 results.append(["Unknown"])
                 progress_bar.progress(percent_complete)
                 time.sleep(0.2) # Small delay even on empty content
                 continue

            try:
                extracted_data = json.loads(content)
                if isinstance(extracted_data, list):
                    extracted_names = [name for name in extracted_data if name in canonical_names]
                    if extracted_names:
                        results.append(extracted_names)
                        logger.info(f"[UltraBasic] Found names for event {i+1}: {extracted_names}")
                    else:
                        results.append(["Unknown"])
                        logger.info(f"[UltraBasic] No known names found for event {i+1}.")
                elif isinstance(extracted_data, dict):
                    found_name = False
                    for key, value in extracted_data.items():
                        if isinstance(value, str) and value in canonical_names:
                            results.append([value])
                            logger.info(f"[UltraBasic] Found name in dict for event {i+1}: {value}")
                            found_name = True
                            break
                    if not found_name:
                        results.append(["Unknown"])
                        logger.info(f"[UltraBasic] No known names found in dict for event {i+1}.")
                else:
                    results.append(["Unknown"])
                    logger.warning(f"[UltraBasic] Unexpected data type from JSON parse for event {i+1}: {type(extracted_data)}")
            except json.JSONDecodeError as parse_error:
                logger.error(f"[UltraBasic] Error parsing JSON response for event {i+1}: {parse_error}. Response: {content[:100]}")
                # Attempt regex fallback as in the main function
                name_matches = re.findall(r'"([^"]+)"', content)
                if name_matches:
                    validated_names = [name for name in name_matches if name in canonical_names]
                    if validated_names:
                        logger.info(f"[UltraBasic] Regex fallback extracted for event {i+1}: {validated_names}")
                        results.append(validated_names)
                    else:
                        logger.info(f"[UltraBasic] Regex fallback found no known names for event {i+1}.")
                        results.append(["Unknown"])
                else:
                    results.append(["Unknown_Error"])
            except Exception as parse_error:
                logger.error(f"[UltraBasic] Unexpected error parsing response for event {i+1}: {parse_error}")
                results.append(["Unknown_Error"])

            time.sleep(0.5) # Keep the delay

        except Exception as e:
            logger.error(f"[UltraBasic] Error during LLM call or processing for event {i+1}: {e}", exc_info=True)
            results.append(["Unknown_Error"])
            time.sleep(1)  # Longer delay after error
            # Try to get a new client instance in case the old one is stuck
            logger.warning("[UltraBasic] Attempting to refresh LLM client after error.")
            client = get_llm_client()
            if not client:
                 logger.error("[UltraBasic] Failed to refresh LLM client. Stopping extraction.")
                 st.error("LLM connection lost. Processing stopped.")
                 # Fill remaining results with error
                 remaining_count = total_rows - len(results)
                 results.extend([["Unknown_Error"]] * remaining_count)
                 break # Exit the loop if client cannot be refreshed

        # Update progress bar at the end of the loop iteration
        progress_bar.progress(percent_complete)

    # Ensure results list matches dataframe length
    if len(results) < total_rows:
        logger.warning(f"[UltraBasic] Results list length ({len(results)}) doesn't match DataFrame length ({total_rows}). Appending errors.")
        results.extend([["Unknown_Error"]] * (total_rows - len(results)))
    elif len(results) > total_rows:
        logger.warning(f"[UltraBasic] Results list length ({len(results)}) exceeds DataFrame length ({total_rows}). Truncating.")
        results = results[:total_rows]

    progress_bar.progress(100)
    status_text.text(f"Processing complete! Processed {len(results)} events.")

    df_copy['extracted_personnel'] = results
    logger.info(f"[UltraBasic] Finished extraction. Returning DataFrame with {len(df_copy)} rows.")

    return df_copy

def run_llm_extraction_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs LLM extraction sequentially on the 'summary' column of the DataFrame.
    This is a more reliable but slower alternative to the parallel version.

    Args:
        df: DataFrame with a 'summary' column.

    Returns:
        DataFrame with a new 'extracted_personnel' column containing lists of names
        or error strings ('Unknown', 'Unknown_Error').
    """
    if not is_llm_ready():
        logger.warning("LLM client is not available. Cannot perform extraction.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df) # Add column indicating failure
        return df

    if 'summary' not in df.columns:
        logger.error("'summary' column not found in DataFrame. Cannot extract.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        return df

    # Process sequentially using our new sequential processor
    try:
        # Get canonical names for validation
        _, _, canonical_names = config_manager.get_personnel_details()

        # Initialize sequential processor for reliable event-by-event processing
        from .sequential_processor import SequentialProcessor
        processor = SequentialProcessor()

        # Log start of sequential processing
        logger.info(f"Starting sequential processing of {len(df)} events...")

        # Process the dataframe with sequential processor
        start_time = time.time()
        result_df = processor.process_dataframe(df, summary_col="summary", canonical_names=canonical_names)
        elapsed = time.time() - start_time

        logger.info(f"Sequential processing complete in {elapsed:.2f} seconds")
        return result_df

    except Exception as e:
        logger.error(f"Error in sequential extraction: {e}")
        # Add an empty extraction column as fallback
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        return df

def run_llm_extraction_background(df: pd.DataFrame, batch_id: str) -> bool:
    """
    Runs LLM extraction in the background, saving partial results to the database.
    This is a simplified version that processes items sequentially for reliability.

    Args:
        df: DataFrame containing a 'summary' column
        batch_id: Unique identifier for this processing batch

    Returns:
        bool: True if processing started successfully
    """
    try:
        if not is_llm_ready():
            logger.error("LLM client is not available. Cannot perform background extraction.")
            return False

        if 'summary' not in df.columns:
            logger.error("'summary' column not found in DataFrame. Cannot extract.")
            return False

        df_copy = df.copy()
        df_copy['summary'] = df_copy['summary'].fillna('') # Ensure no NaN summaries

        summaries = df_copy['summary'].tolist()

        llm_client = get_llm_client() # Get cached client
        _, _, canonical_names = config_manager.get_personnel_details() # Get current names

        if not canonical_names:
            logger.error("Cannot run background LLM extraction: No canonical names found in configuration.")
            return False

        # Import here to avoid circular imports
        from functions import db_manager

        # First save the preprocessed data to database with 'processing' status (if DB enabled)
        if settings.DB_ENABLED:
            df_copy['processing_status'] = 'processing'
            db_manager.save_partial_processed_data(df_copy, batch_id)
        else:
            # Initialize the background_processed_data in session state for non-DB mode
            if not hasattr(st.session_state, 'background_processed_data'):
                st.session_state.background_processed_data = {}

            # Initialize with empty status
            st.session_state.background_processed_data[batch_id] = {
                'status': 'in_progress',
                'progress': 0.0,
                'message': f"Starting processing of {len(summaries)} events",
                'timestamp': time.time(),
                'total_events': len(summaries),
                'processed_events': 0
            }

        # Start background thread for processing
        def process_in_background():
            logger.info(f"Starting background LLM extraction for batch {batch_id} with {len(summaries)} summaries")
              # Create a persistent terminal progress bar for the background process that's clearly visible
            if TQDM_AVAILABLE:
                term_progress = tqdm(total=len(summaries), desc=f"Batch {batch_id}", position=0, leave=True,
                                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            else:
                term_progress = get_persistent_progress_bar(len(summaries), f"Background LLM Extraction (Batch {batch_id})")

            print(f"\n\033[1m>>> Processing {len(summaries)} items with batch ID {batch_id} <<<\033[0m")

            # Get a fresh LLM client for this thread
            thread_llm_client = get_llm_client()
            _, _, thread_canonical_names = config_manager.get_personnel_details()

            if not thread_llm_client or not thread_canonical_names:
                logger.error(f"Thread failed to get LLM client or canonical names for batch {batch_id}")
                term_progress.close()
                return

            # Process in small batches with simple sequential processing
            batch_size = 5  # Process 5 summaries at a time
            save_frequency = 10  # Save results to DB every 10 items

            # Track progress
            processed_count = 0
            error_count = 0

            # Process one summary at a time for reliability
            for i, summary in enumerate(summaries):
                try:
                    logger.info(f"Processing summary {i+1}/{len(summaries)}: '{summary[:50]}...'")

                    # Refresh the client connection every 100 items
                    if i > 0 and i % 100 == 0:
                        logger.info(f"Refreshing LLM client after processing {i} items...")
                        thread_llm_client = get_llm_client()

                    # Extract personnel and event type from the summary
                    personnel_list, event_type = extract_personnel_with_llm(summary, thread_llm_client, thread_canonical_names)
                    df_copy.loc[i, 'extracted_personnel'] = personnel_list
                    df_copy.loc[i, 'extracted_event_type'] = event_type # Assign event type
                    df_copy.loc[i, 'processing_status'] = 'extracted'

                    # Save progress every 10 items
                    if i > 0 and i % save_frequency == 0:
                        if settings.DB_ENABLED:
                            batch_df = df_copy.iloc[max(0, i - save_frequency):i+1].copy()
                            db_manager.save_partial_processed_data(batch_df, batch_id)
                            logger.info(f"Saved progress for items {max(0, i - save_frequency)}-{i}.")                    # Update progress tracking with more visible output
                    processed_count += 1
                    term_progress.update(1)
                    progress = processed_count / len(summaries)

                    # Print direct status updates that will be visible in terminal
                    if processed_count % 5 == 0 or processed_count == 1:
                        print(f"✓ Processed {processed_count}/{len(summaries)} ({progress*100:.2f}%) - Last result: {str(result)[:50]}...") # Corrected variable name

                    logger.info(f"Progress: {processed_count}/{len(summaries)} ({progress*100:.2f}%) - Result: {personnel_list}, {event_type}") # Corrected variable name

                    # Add a small delay to avoid overwhelming the server
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error processing summary {i+1}/{len(summaries)}: {e}")
                    df_copy.loc[i, 'extracted_personnel'] = ["Unknown_Error"]
                    df_copy.loc[i, 'extracted_event_type'] = "Unknown_Error" # Assign error state
                    error_count += 1

                    # Refresh the client after multiple errors
                    if error_count > 5:
                        logger.warning(f"Encountered {error_count} errors. Refreshing LLM client...")
                        thread_llm_client = get_llm_client()
                        error_count = 0
                        time.sleep(2)  # Longer delay after multiple errors
                    else:
                        time.sleep(0.5)  # Brief delay after an error

            # After all individual processing is done, save final results
            if settings.DB_ENABLED:
                logger.info(f"Saving final results for batch {batch_id}")
                db_manager.save_partial_processed_data(df_copy, batch_id)

            # Import normalizer module for the next step
            from .normalizer import normalize_extracted_personnel

            # Normalize the extracted personnel
            logger.info(f"Normalizing extracted names for batch {batch_id}")
            try:
                normalized_df = normalize_extracted_personnel(df_copy)

                # Save final normalized results
                if settings.DB_ENABLED:
                    db_manager.save_partial_processed_data(normalized_df, batch_id)

                    # Explode by personnel for final analysis-ready data
                    from functions import data_processor
                    logger.info(f"Exploding by personnel for batch {batch_id}")
                    analysis_df = data_processor.explode_by_personnel(normalized_df, personnel_col='assigned_personnel')
                    db_manager.save_processed_data_to_db(analysis_df, batch_id)

                    # Mark the calendar file as processed
                    db_manager.mark_calendar_file_as_processed(batch_id)
                else:
                    # Store results in session state for non-DB mode
                    from functions import data_processor
                    analysis_df = data_processor.explode_by_personnel(normalized_df, personnel_col='assigned_personnel')

                    if hasattr(st.session_state, 'background_processed_data'):
                        st.session_state.background_processed_data[batch_id] = {
                            'normalized_df': normalized_df,
                            'analysis_df': analysis_df,
                            'status': 'complete',
                            'progress': 1.0,
                            'message': f"Processing complete: {len(analysis_df)} events processed",
                            'timestamp': time.time(),
                            'total_events': len(summaries),
                            'processed_events': len(summaries)
                        }

                logger.info(f"Background processing complete for batch {batch_id}")

            except Exception as e:
                logger.error(f"Error during final processing steps for batch {batch_id}: {e}")

            term_progress.close()  # Close the terminal progress bar when complete

        # Start the processing thread
        background_thread = threading.Thread(
            target=process_in_background,
            daemon=True  # Make thread a daemon so it doesn't block app shutdown
        )
        background_thread.start()

        logger.info(f"Background processing thread started for batch {batch_id}")
        return True

    except Exception as e:
        logger.error(f"Error setting up background LLM extraction: {e}")
        return False
