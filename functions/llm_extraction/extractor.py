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

def _extract_single_physicist_llm(summary: str, llm_client, canonical_names: list[str]) -> list[str] | str:
    """
    Internal function to extract names for a single summary using the LLM.
    Returns list of names on success, or "Unknown" / "Unknown_Error" on failure.
    """
    if not llm_client:
        # Should not happen if called correctly, but safety check
        return "Unknown_Error"
    if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
        logger.debug("Skipping extraction for empty or non-string summary.")
        return ["Unknown"] # Return list for consistency

    # Construct the prompt
    prompt = f"""
    Your task is to identify physicist names from the provided calendar event summary.
    You are given a specific list of known canonical physicist names.
    Analyze the summary and identify ONLY the canonical names from the list below that are mentioned or clearly referenced in the summary.
    Consider variations in names (e.g., initials, last names only, common misspellings if obvious) but map them back EXACTLY to a name present in the canonical list.
    Do not guess or include names not on the list. If multiple physicists are mentioned, include all of them.
    If no physicists from the list are clearly identified, return an empty list.

    Known Canonical Physicist Names:
    {json.dumps(canonical_names, indent=2)}

    Event Summary:
    "{summary}"

    Respond ONLY with a valid JSON list containing the identified canonical names found in the summary.
    Example Response: ["Name1", "Name2"]
    Example Response (none found): []
    Do not add explanations or surrounding text. ONLY the JSON list.
    """

    try:
        # Add timeout to prevent hanging indefinitely
        start_time = time.time()
        
        # Set a timeout for the request
        timeout_seconds = 30  # 30 second timeout
        
        # Log that we're about to call the API
        logger.debug(f"Calling Ollama API for summary: '{summary[:30]}...'")
        
        response = llm_client.chat(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an assistant that extracts specific names from text based on a provided list and outputs ONLY a valid JSON list containing those names."},
                {"role": "user", "content": prompt}
            ],
            format='json', # Request JSON format
            options={'temperature': 0.1, 'timeout': timeout_seconds} # Add timeout and low temperature for deterministic output
        )
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Ollama API call completed in {elapsed_time:.2f} seconds")

        content = response.get('message', {}).get('content', '')
        if not content:
             logger.warning(f"LLM returned empty content for summary: '{summary[:50]}...'")
             return ["Unknown"] # Treat empty content as no names found

        # --- Enhanced JSON Parsing Logic ---
        extracted_names = []
        try:
            # New preprocessing step to handle malformed responses that look like arrays inside curly braces
            # E.g., '{"D. Solis"}' or '{"G. Pitcher, E. Chorniak, Wood"}'
            if content.startswith('{') and ('"' in content or "'" in content):
                # Check if it looks like a malformed "array as dict" response
                if ':' not in content or ((':' in content) and (',' in content and content.find(',') < content.find(':'))):
                    # Replace curly braces with square brackets
                    fixed_content = content.replace('{', '[').replace('}', ']')
                    logger.info(f"Fixed malformed JSON response: {content[:50]}... to {fixed_content[:50]}...")
                    
                    # Additional fixes for malformed JSON lists
                    # 1. Try to identify and fix broken string format without commas
                    if fixed_content.count('"') >= 2 and ',' not in fixed_content:
                        # Handle case where there's a single name without proper commas: ["D. Solis"              ]
                        name_match = re.search(r'"([^"]+)"', fixed_content)
                        if name_match:
                            name = name_match.group(1).strip()
                            fixed_content = f'["{name}"]'
                            logger.info(f"Reformatted single name response to: {fixed_content}")
                    
                    # 2. Handle comments and trailing text
                    if '//' in fixed_content:
                        # Remove everything after //
                        fixed_content = fixed_content.split('//')[0].strip()
                        if fixed_content.endswith(','):
                            fixed_content = fixed_content[:-1]  # Remove trailing comma
                        if not fixed_content.endswith(']'):
                            fixed_content += ']'  # Add closing bracket if needed
                        logger.info(f"Removed comments from JSON: {fixed_content}")
                    
                    # 3. Fix unquoted items in list
                    if re.search(r',\s*[A-Za-z][^",\]]*[,\]]', fixed_content):
                        # Find unquoted names like ['"G. Pitcher", E. Chorniak, Wood, King]'
                        fixed_parts = []
                        in_quotes = False
                        current_part = ""
                        
                        for char in fixed_content:
                            if char == '"' and (not current_part.endswith('\\')):
                                in_quotes = not in_quotes
                            
                            current_part += char
                            
                            if not in_quotes and char in ',]':
                                if current_part.strip().startswith(','):
                                    current_part = current_part.strip()[1:].strip()
                                
                                # If we have a non-quoted item
                                if current_part.strip() and not (current_part.strip().startswith('"') and current_part.strip().endswith('"')):
                                    # Not already quoted and not just punctuation
                                    if re.search(r'[A-Za-z]', current_part):
                                        part = current_part.strip()
                                        if part.endswith(',') or part.endswith(']'):
                                            part = part[:-1].strip()
                                        fixed_parts.append(f'"{part}"')
                                        if char == ',':
                                            fixed_parts.append(',')
                                        elif char == ']':
                                            fixed_parts.append(']')
                                    else:
                                        fixed_parts.append(current_part)
                                else:
                                    fixed_parts.append(current_part)
                                
                                current_part = ""
                        
                        # Build the corrected JSON
                        if fixed_parts:
                            fixed_content = ''.join(fixed_parts)
                            if not fixed_content.startswith('['):
                                fixed_content = '[' + fixed_content
                            if not fixed_content.endswith(']'):
                                fixed_content += ']'
                            logger.info(f"Fixed unquoted items: {fixed_content}")
                    
                    content = fixed_content

            try:
                extracted_data = json.loads(content)
            except json.JSONDecodeError as json_err:
                # Last resort fallback for very malformed responses
                # Try to extract names directly using regex
                logger.warning(f"JSON decode error: {json_err}. Attempting regex fallback.")
                
                # Special handling for unterminated strings
                if 'Unterminated string' in str(json_err) and content.startswith('['):
                    # Try to fix the unterminated string by adding a closing quote and bracket if needed
                    try:
                        # First try direct content extraction - this works with many of the error cases
                        if content.startswith('["') and not content.endswith('"]'):
                            # Extract content between brackets, ignoring quotes issues
                            inner_content = content[2:].rstrip(']').strip()
                            # For case ["D. Solis" - extract just the name
                            if '"' in inner_content:
                                parts = inner_content.split('"', 1)[0].strip()
                                if parts:  # Make sure we got something
                                    logger.info(f"Simple unterminated string fix: {parts}")
                                    return [parts]
                        
                        # More complex parsing for multi-name cases
                        # Extract all content between the square brackets
                        bracket_content = re.search(r'\[(.*?)(?:\]|$)', content)
                        if bracket_content:
                            # Get the content and ensure proper format with quotes
                            names_text = bracket_content.group(1).strip()
                            
                            # Handle missing closing quotes in strings like ["D. Solis"...
                            if names_text.startswith('"') and '",' not in names_text and '"]' not in names_text:
                                name = re.search(r'"([^"]+)', names_text)
                                if name:
                                    extracted = name.group(1).strip()
                                    logger.info(f"Fixed unclosed quote in single name: {extracted}")
                                    return [extracted]
                            
                            # Handle multiple names in a list with comma separation
                            # Split by commas, strip each part, and put in proper JSON format
                            name_parts = [part.strip() for part in names_text.split(',')]
                            # Remove any trailing comments or text after //
                            name_parts = [part.split('//')[0].strip() for part in name_parts]
                            # Remove quotes from parts that have them
                            name_parts = [part.strip('"') for part in name_parts]
                            # Filter empty parts
                            name_parts = [part for part in name_parts if part]
                            
                            # Remove any text in parentheses that appears after names like "(Note:..."
                            clean_parts = []
                            for part in name_parts:
                                if "(" in part:
                                    clean_parts.append(part.split("(")[0].strip())
                                else:
                                    clean_parts.append(part)
                            
                            logger.info(f"Unterminated string fallback extracted: {clean_parts}")
                            return clean_parts
                    except Exception as e:
                        logger.error(f"Error in unterminated string handling: {e}")
                
                # Standard regex fallback for quoted strings
                name_matches = re.findall(r'"([^"]+)"', content)
                if name_matches:
                    logger.info(f"Regex fallback extracted: {name_matches}")
                    return name_matches
                else:
                    # Last attempt: extract anything that looks like a name between commas
                    comma_separated = re.split(r'[\[\],]', content)
                    potential_names = [item.strip() for item in comma_separated if item.strip()]
                    if potential_names:
                        logger.info(f"Last-resort comma split extracted: {potential_names}")
                        return potential_names
                    
                    logger.error(f"Failed to decode JSON response from LLM for '{summary[:30]}...': {content[:100]}. Error: {str(json_err)}")
                    return ["Unknown_Error"]  # Indicate a structural error from LLM

            # 1. Ideal case: Is it directly a list?
            if isinstance(extracted_data, list):
                # Validate items are strings
                extracted_names = [item for item in extracted_data if isinstance(item, str)]
                logger.debug(f"LLM returned a direct list for '{summary[:30]}...': {extracted_names}")

            # 2. Common case: Is it a dictionary containing a list under common keys?
            elif isinstance(extracted_data, dict):
                logger.debug(f"LLM returned dict for '{summary[:30]}...': {content[:100]}")
                found_list = None
                possible_keys = ['names', 'physicists', 'identified_names', 'canonical_names', 'result', 'output']
                for key in possible_keys:
                    if key in extracted_data and isinstance(extracted_data.get(key), list):
                        potential_list = extracted_data[key]
                        # Validate items are strings
                        found_list = [item for item in potential_list if isinstance(item, str)]
                        logger.debug(f"Found list under key '{key}'")
                        break # Found it

                # Special case: Handle single key-value pairs where value is a name
                # Example: {"A": "Wood"} - extract "Wood" as a name
                if found_list is None and len(extracted_data) <= 3:  # Only for small dicts
                    for key, value in extracted_data.items():
                        if isinstance(value, str) and value:  # If value is a non-empty string
                            # Check if value is likely a name (in canonical list or has letter characters)
                            if value in canonical_names or re.search(r'[A-Za-z]', value):
                                found_list = [value]
                                logger.info(f"Extracted name from simple key-value pair: {value}")
                                break

                # Fallback: check *any* value that is a list of strings
                if found_list is None:
                    for value in extracted_data.values():
                        if isinstance(value, list):
                            potential_list = value
                            # Validate items are strings
                            string_list = [item for item in potential_list if isinstance(item, str)]
                            if string_list: # Use the first non-empty list of strings found
                                found_list = string_list
                                logger.debug("Found list as a dictionary value (unknown key)")
                                break

                if found_list is not None:
                    extracted_names = found_list
                else:
                    logger.warning(f"LLM returned dict, but failed to extract expected list structure: {content[:100]}")
                    extracted_names = [] # Treat as no names found

            # 3. Handle unexpected format
            else:
                 logger.warning(f"LLM (JSON mode) returned unexpected format (not list or dict): {content[:100]}")
                 extracted_names = [] # Treat as no names found

        except JSONDecodeError as json_err:
            logger.error(f"Failed to decode JSON response from LLM for '{summary[:30]}...': {content[:100]}. Error: {str(json_err)}")
            return "Unknown_Error" # Indicate a structural error from LLM
        except Exception as e:
             logger.error(f"Error processing LLM response content structure: {e}\nResponse content: {content[:100]}")
             return "Unknown_Error"

        # --- Validation Against Canonical List ---
        validated_names = [name for name in extracted_names if name in canonical_names]

        if not validated_names:
            logger.debug(f"LLM found no known physicists in: '{summary[:30]}...' (Raw response: {content[:50]}...)")
            return ["Unknown"] # No known names found or returned list was empty
        else:
            logger.debug(f"LLM validated {validated_names} in '{summary[:30]}...'")
            return validated_names # Return list of validated canonical names

    except Exception as e:
        # Catch Ollama client errors / network errors etc. during the API call
        logger.error(f"Error calling Ollama API for summary '{summary[:30]}...': {e}")
        # Indicate an error occurred during the call
        return "Unknown_Error"

def extract_personnel_with_llm(summary: str, llm_client, canonical_names: list[str], retry_count=3) -> list[str] | str:
    """
    Extract personnel from event data with improved error handling and retry logic
    """
    if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
        return ["Unknown"]  # Return list for consistency

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
        return ["Unknown_Error"]

def run_llm_extraction_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs LLM extraction in parallel on the 'summary' column of the DataFrame.
    If parallel processing fails, will fall back to sequential processing.

    Args:
        df: DataFrame with a 'summary' column.

    Returns:
        DataFrame with a new 'extracted_personnel' column containing lists of names
        or error strings ('Unknown', 'Unknown_Error').
    """
    if not is_llm_ready():
        st.error("LLM client is not available. Cannot perform extraction.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df) # Add column indicating failure
        return df

    if 'summary' not in df.columns:
        logger.error("'summary' column not found in DataFrame. Cannot extract.")
        st.error("Input data is missing the 'summary' column.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
        return df

    df_copy = df.copy()
    df_copy['summary'] = df_copy['summary'].fillna('') # Ensure no NaN summaries

    summaries = df_copy['summary'].tolist()
    results = [None] * len(summaries) # Initialize results list

    llm_client = get_llm_client() # Get cached client
    _, _, canonical_names = config_manager.get_personnel_details() # Get current names

    if not canonical_names:
        logger.error("Cannot run LLM extraction: No canonical names found in configuration.")
        st.error("Personnel configuration is empty. Please configure personnel in the Admin page.")
        df['extracted_personnel'] = [["Unknown_Error"]] * len(df)
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
            
            # Add timeout to prevent hanging indefinitely
            active_futures = list(future_to_index.keys())
            timeout_occurred = False
            
            # Monitor the first few futures to detect if processing is stalled
            monitoring_count = min(5, len(active_futures)) if len(summaries) > 5 else 1
            
            # Set an overall timeout for checking if parallel processing is working
            parallel_check_timeout = 60  # 1 minute to check if parallel processing works
            parallel_check_start = time.time()
            
            # Check if any futures complete in the given timeout
            completed_futures = as_completed(active_futures[:monitoring_count], timeout=parallel_check_timeout)
            
            try:
                # Get at least one future to validate parallel processing works
                first_future = next(completed_futures)
                idx = future_to_index[first_future]
                result = first_future.result()
                results[idx] = result
                completed_count += 1
                term_progress.update(1)  # Update the terminal progress bar
                
                # If we got here, at least one future completed successfully, proceed with parallel
                logger.info("Parallel processing verified working, continuing with parallel extraction")
            except TimeoutError:
                # If no futures complete within timeout, switch to sequential
                logger.warning(f"No futures completed within {parallel_check_timeout}s timeout. Switching to sequential mode.")
                st.warning("Parallel processing appears to be stalled. Switching to sequential processing mode.")
                timeout_occurred = True
            except Exception as e:
                logger.error(f"Error processing first batch of futures: {e}")
                st.warning("Error in parallel processing. Switching to sequential mode for reliability.")
                timeout_occurred = True
                
            # If timeout occurred or other problem detected, switch to sequential
            if timeout_occurred:
                # Cancel all pending futures
                for future in active_futures:
                    future.cancel()
                
                term_progress.close()  # Close the terminal progress bar
                # Switch to sequential mode
                return run_llm_extraction_sequential(df)
            
            # Continue with parallel processing for the remaining futures
            
            # Add timeout to detect stuck processes
            timeout_start = time.time()
            timeout_limit = 300  # 5 minutes timeout for the entire process

            for future in as_completed(active_futures):
                if time.time() - timeout_start > timeout_limit:
                    logger.error("Processing timeout: Exceeded 5 minutes.")
                    st.error("Processing timeout: Please try again or check the logs.")
                    break

                index = future_to_index[future]
                try:
                    result = future.result()  # Get the list of names or error string
                    results[index] = result
                except Exception as exc:
                    logger.error(f"Summary index {index} ('{summaries[index][:50]}...') generated an exception: {exc}")
                    results[index] = ["Unknown_Error"]  # Mark as error, ensure it's a list

                # Update progress bar
                completed_count += 1
                term_progress.update(1)  # Update the terminal progress bar
                progress_percent = int((completed_count / total_summaries) * 100)
                progress_bar.progress(progress_percent, text=f"Processing event {completed_count}/{total_summaries}...")

            # Check if all results were processed
            missing_results = [i for i, r in enumerate(results) if r is None]
            if missing_results:
                logger.warning(f"{len(missing_results)} events were not processed in parallel mode. Processing them sequentially.")
                # Process missing results sequentially
                for i in missing_results:
                    try:
                        results[i] = extract_personnel_with_llm(summaries[i], llm_client, canonical_names)
                        # Update progress bar
                        completed_count += 1
                        term_progress.update(1)  # Update the terminal progress bar
                        progress_percent = int((completed_count / total_summaries) * 100)
                        progress_bar.progress(progress_percent, text=f"Processing event {completed_count}/{total_summaries}...")
                    except Exception as e:
                        logger.error(f"Sequential fallback failed for index {i}: {e}")
                        results[i] = ["Unknown_Error"]

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
    for i in range(len(results)):
        if results[i] is None:
            results[i] = ["Unknown_Error"]
            
    df_copy['extracted_personnel'] = results
    return df_copy

def run_llm_extraction_sequential(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs LLM extraction sequentially (one at a time) as a fallback when parallel extraction fails.
    This is slower but more reliable when there are connection issues.
    """
    logger.info("Starting sequential LLM extraction as fallback method...")
    st.info("Processing events one at a time for reliability (this may take longer)...")
    
    df_copy = df.copy()
    df_copy['summary'] = df_copy['summary'].fillna('') # Ensure no NaN summaries
    summaries = df_copy['summary'].tolist()
    
    # Get client and names
    llm_client = get_llm_client()
    _, _, canonical_names = config_manager.get_personnel_details()
    
    # Create a progress bar and persistent terminal progress bar
    progress_bar = st.progress(0, text="Starting sequential extraction...")
    term_progress = get_persistent_progress_bar(len(summaries), "Sequential LLM Extraction")
    
    # Initialize results list with Unknown placeholders
    results = [["Unknown"]] * len(summaries)
    
    start_time = time.time()
    successful = 0
    errors = 0
    
    # Process each summary one at a time with individual timeouts
    for i, summary in enumerate(summaries):
        try:
            # Check if we need to restart Ollama server (every 100 items)
            if i > 0 and i % 100 == 0:
                logger.info(f"Processed {i} items, checking if Ollama server needs to be restarted...")
                # First try refreshing the client connection
                llm_client = get_llm_client()
                
                # If we've had multiple errors, restart the Ollama server
                if errors > 5:
                    logger.warning(f"Detected {errors} errors in the last batch. Attempting to restart Ollama server...")
                    # Try to restart the Ollama server
                    restart_success = restart_ollama_server()
                    if restart_success:
                        logger.info("Ollama server restarted successfully. Continuing extraction...")
                        llm_client = get_llm_client()  # Get a fresh client
                        errors = 0  # Reset error count
                        st.warning("Ollama server was automatically restarted to improve performance")
                    else:
                        logger.error("Failed to restart Ollama server. Continuing with current client...")
                
            # Process with a more conservative timeout and retry mechanism
            result = extract_personnel_with_llm(summary, llm_client, canonical_names, retry_count=1)
            results[i] = result
            successful += 1
            
            # Log progress occasionally
            if i % 50 == 0 or i == len(summaries) - 1:
                logger.info(f"Sequential extraction progress: {i+1}/{len(summaries)} items processed")
                
            # Update progress bars
            term_progress.update(1)  # Update the terminal progress bar
            progress_percent = int(((i + 1) / len(summaries)) * 100)
            progress_bar.progress(progress_percent, text=f"Processing event {i+1}/{len(summaries)}...")
            
        except Exception as e:
            logger.error(f"Error in sequential extraction for index {i}: {e}")
            results[i] = ["Unknown_Error"]
            errors += 1
            
            # If we see too many consecutive errors, refresh the client
            if errors > 10:
                logger.warning("Too many consecutive errors, refreshing LLM client")
                try:
                    llm_client = get_llm_client()
                    errors = 0  # Reset error count
                except Exception as client_error:
                    logger.error(f"Failed to refresh client: {client_error}")
                    
    end_time = time.time()
    elapsed = end_time - start_time
    
    logger.info(f"Sequential extraction completed in {elapsed:.2f} seconds. {successful} successful, {errors} errors.")
    st.success(f"LLM extraction completed. {successful} events processed successfully, {errors} errors.")
    
    term_progress.close()  # Close the terminal progress bar
    df_copy['extracted_personnel'] = results
    return df_copy

def run_llm_extraction_background(df: pd.DataFrame, batch_id: str) -> bool:
    """
    Runs LLM extraction in the background, saving partial results to the database.
    This version doesn't use Streamlit progress bars and is designed to be run
    in a separate thread.

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
            
            # Create a persistent terminal progress bar for the background process
            term_progress = get_persistent_progress_bar(len(summaries), f"Background LLM Extraction (Batch {batch_id})")
            
            # Update LLM client and get latest canonical names when thread starts
            # This ensures we're not using stale connections
            thread_llm_client = get_llm_client()
            _, _, thread_canonical_names = config_manager.get_personnel_details()
            
            if not thread_llm_client or not thread_canonical_names:
                logger.error(f"Thread failed to get LLM client or canonical names for batch {batch_id}")
                term_progress.close()
                return
                
            # Process in batches to save intermediate results
            batch_size = 10  # Process 10 summaries at a time
            
            for i in range(0, len(summaries), batch_size):
                # Check if we need to reload the data from the database (if DB enabled)
                # This happens if the thread has been running a while and the app was restarted
                if settings.DB_ENABLED and i > 0 and i % 50 == 0:
                    try:
                        # Check if the calendar file still exists in the database
                        filename, file_content, processed = db_manager.get_calendar_file_by_batch_id(batch_id)
                        if not filename or processed:
                            logger.info(f"Calendar file for batch {batch_id} marked as processed or no longer exists. Stopping background process.")
                            term_progress.close()
                            return
                        
                        # Reload state from database
                        latest_status = db_manager.get_latest_processing_status(batch_id)
                        if latest_status['status'] == 'complete':
                            logger.info(f"Batch {batch_id} already marked as complete in database. Stopping background process.")
                            term_progress.close()
                            return
                    except Exception as e:
                        logger.error(f"Error checking calendar file status: {e}")
                        # Continue processing anyway
                
                batch_indices = range(i, min(i+batch_size, len(summaries)))
                batch_summaries = [summaries[j] for j in batch_indices]
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(summaries)-1)//batch_size + 1} for batch_id {batch_id}")
                
                # Process this batch of summaries
                error_count = 0
                batch_results = []
                
                # Check if we need to restart the Ollama server
                # Restart after processing 100 events or if we have consecutive errors
                if i > 0 and (i % 100 == 0 or error_count > 5):
                    logger.info(f"Checking if Ollama server needs restart after processing {i} events...")
                    # Try to restart the Ollama server
                    restart_success = restart_ollama_server()
                    if restart_success:
                        thread_llm_client = get_llm_client()  # Get a fresh client after restart
                        error_count = 0  # Reset error count
                        logger.info(f"Successfully restarted Ollama server at batch position {i}")
                    else:
                        logger.warning(f"Failed to restart Ollama server, continuing with current client")
                
                for summary in batch_summaries:
                    try:
                        result = extract_personnel_with_llm(summary, thread_llm_client, thread_canonical_names)
                        batch_results.append(result)
                        term_progress.update(1)  # Update terminal progress
                    except Exception as exc:
                        logger.error(f"Error processing summary '{summary[:30]}...': {exc}")
                        batch_results.append(["Unknown_Error"])
                        error_count += 1
                        term_progress.update(1)  # Still update progress even on error
                        # Reload LLM client if we got an error
                        thread_llm_client = get_llm_client()
                
                # Update the dataframe with results for this batch
                for idx, j in enumerate(batch_indices):
                    df_copy.loc[j, 'extracted_personnel'] = batch_results[idx]
                
                # Save partial results to database (if DB enabled)
                if settings.DB_ENABLED:
                    batch_df = df_copy.iloc[batch_indices].copy()
                    db_manager.save_partial_processed_data(batch_df, batch_id)
                else:
                    # Update progress in session state for non-DB mode
                    processed_count = min(i + batch_size, len(summaries))
                    progress = processed_count / len(summaries)
                    
                    if hasattr(st.session_state, 'background_processed_data') and batch_id in st.session_state.background_processed_data:
                        st.session_state.background_processed_data[batch_id].update({
                            'progress': progress,
                            'message': f"Processed {processed_count}/{len(summaries)} events ({progress*100:.1f}%)",
                            'timestamp': time.time(),
                            'processed_events': processed_count
                        })
                
                # Sleep briefly to avoid overloading
                time.sleep(0.1)
            
            # Import normalizer module for the next step
            from .normalizer import normalize_extracted_personnel
            
            # Normalize the extracted personnel
            logger.info(f"Normalizing extracted names for batch {batch_id}")
            normalized_df = normalize_extracted_personnel(df_copy)
            
            # Save final normalized results (if DB enabled)
            if settings.DB_ENABLED:
                db_manager.save_partial_processed_data(normalized_df, batch_id)
                
                # Explode by personnel for final analysis-ready data
                from functions import data_processor
                logger.info(f"Exploding by personnel for batch {batch_id}")
                try:
                    analysis_df = data_processor.explode_by_personnel(normalized_df, personnel_col='assigned_personnel')
                    db_manager.save_processed_data_to_db(analysis_df, batch_id)
                    logger.info(f"Background processing complete for batch {batch_id}")
                    
                    # Mark the calendar file as processed
                    db_manager.mark_calendar_file_as_processed(batch_id)
                except Exception as e:
                    logger.error(f"Error in final explode/save for batch {batch_id}: {e}")
                    
            # If DB is disabled, store results in a global variable that can be accessed by the Analysis page
            else:
                # Store the final normalized and exploded data in a global variable
                from functions import data_processor
                logger.info(f"Exploding by personnel for batch {batch_id}")
                analysis_df = data_processor.explode_by_personnel(normalized_df, personnel_col='assigned_personnel')
                
                # Store in session state-like global variable
                if not hasattr(st.session_state, 'background_processed_data'):
                    st.session_state.background_processed_data = {}
                
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
                
                logger.info(f"Background processing complete for batch {batch_id} (stored in session state)")
            
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