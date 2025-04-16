"""
Simplified extractor implementation with improved console output and progress bar display.
"""
import os
import sys
import time
import json
import logging
import re
import pandas as pd
import streamlit as st

# Ensure path includes the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if (PROJECT_ROOT not in sys.path):
    sys.path.insert(0, PROJECT_ROOT)

from config import settings
from functions import config_manager
from .client import get_llm_client, is_llm_ready

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    from .utils import SimpleTqdm as tqdm

def improved_extraction(df):
    """
    An improved extraction function that shows detailed console output and updates progress bars.
    """
    # Display a prominent message in the console
    print(f"\n\033[1m>>> STARTING IMPROVED EXTRACTION: Processing {len(df)} calendar events <<<\033[0m")
    
    # Copy dataframe to avoid modifying the original
    df_copy = df.copy()

    # Get names from config
    _, _, canonical_names = config_manager.get_personnel_details()
    if not canonical_names:
        logger.error("Cannot run improved_extraction: No canonical names found.")
        st.error("Personnel configuration is empty.")
        df_copy['extracted_personnel'] = [["Unknown_Error"]] * len(df_copy)
        return df_copy

    # Get LLM client
    client = get_llm_client()
    if not client:
        logger.error("Cannot run improved_extraction: Failed to get LLM client.")
        st.error("Failed to connect to LLM.")
        df_copy['extracted_personnel'] = [["Unknown_Error"]] * len(df_copy)
        return df_copy

    # List to store results
    results = []
    total_rows = len(df_copy)

    # Create a Streamlit progress bar and status text
    progress_bar = st.progress(0, text="Initializing extraction...")
    status_container = st.container()
    with status_container:
        st.write("Extraction progress:")
        status_text = st.empty()
        results_display = st.empty()

    # Create terminal progress bar
    if TQDM_AVAILABLE:
        term_progress = tqdm(total=total_rows, desc="LLM Extraction", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        # Use a custom progress bar from the utils module
        from .utils import get_persistent_progress_bar
        term_progress = get_persistent_progress_bar(total_rows, "LLM Extraction")

    # Start timing
    start_time = time.time()
    
    # Store last 5 results for display
    last_results = []

    # Process each row
    for i, row_tuple in enumerate(df_copy.iterrows()):
        index, data = row_tuple  # Get index and data from tuple
        percent_complete = (i + 1) / total_rows
        progress_bar.progress(percent_complete, text=f"Processing event {i+1} of {total_rows}...")

        # Get the summary from the current row
        summary = data.get('summary', '')
        
        # Skip empty summaries
        if not summary or len(summary.strip()) == 0:
            logger.info(f"Skipping event {i+1}/{total_rows} due to empty summary")
            results.append(["Unknown"])
            term_progress.update(1)
            continue

        # Print direct output to console for regular progress updates
        if i % 5 == 0 or i == 0 or i == total_rows - 1:
            print(f"Processing {i+1}/{total_rows} ({percent_complete*100:.1f}%): '{summary[:50]}...'")
        
        try:
            status_text.write(f"**Event {i+1}/{total_rows}:** \"{summary[:50]}...\"")
            
            # Format the prompt
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

            IMPORTANT: Respond ONLY with a valid JSON array. Do not use any other JSON format.
            Correct format examples:
            []
            ["D. Solis"]
            ["C. Chu", "D. Solis"]
            """

            # Call the LLM with the prompt
            logger.info(f"Calling Ollama API for event {i+1}/{total_rows}")
            response = client.chat(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant that extracts specific names from text based on a provided list and outputs ONLY a valid JSON list containing those names."},
                    {"role": "user", "content": prompt}
                ],
                format="json",
                options={'temperature': 0.1, 'timeout': 30}
            )

            # Process the response
            content = response.get('message', {}).get('content', '')
            
            # Log the raw response for debugging
            logger.info(f"LLM response for event {i+1}/{total_rows}: {content[:100]}")
            
            # Display the response in the console
            print(f"✓ Response: {content[:100]}")

            # Try to parse the JSON response
            try:
                extracted_data = json.loads(content)
                
                # Handle different response formats
                if isinstance(extracted_data, list):
                    # Great! We got a list directly
                    extracted_names = [name for name in extracted_data if name in canonical_names]
                    if extracted_names:
                        results.append(extracted_names)
                    else:
                        results.append(["Unknown"])
                        
                elif isinstance(extracted_data, dict):
                    # Try to find a list in the dictionary
                    found_list = None
                    
                    # Check common keys that might contain the list
                    possible_keys = ['names', 'physicists', 'identified_names', 'canonical_names', 'result', 'output']
                    for key in possible_keys:
                        if key in extracted_data and isinstance(extracted_data.get(key), list):
                            potential_list = extracted_data[key]
                            found_list = [item for item in potential_list if isinstance(item, str) and item in canonical_names]
                            break
                    
                    # If we found valid names, use them
                    if found_list:
                        results.append(found_list)
                    else:
                        results.append(["Unknown"])
                else:
                    # Unexpected data type
                    results.append(["Unknown"])
                    
            except json.JSONDecodeError:
                # Try to extract names using regex as a fallback
                name_matches = re.findall(r'"([^"]+)"', content)
                validated_names = [name for name in name_matches if name in canonical_names]
                
                if validated_names:
                    results.append(validated_names)
                    logger.info(f"Regex fallback extracted names: {validated_names}")
                else:
                    results.append(["Unknown"])
                    
            except Exception as e:
                logger.error(f"Error processing LLM response: {e}")
                results.append(["Unknown_Error"])

            # Update the result display
            current_result = results[-1]
            result_text = f"**Event {i+1}:** '{summary[:40]}...' → **Result:** {current_result}"
            
            # Keep only the last 5 results for display
            last_results.append(result_text)
            if len(last_results) > 5:
                last_results.pop(0)
                
            # Update the results display
            results_display.markdown("\\n".join(last_results))
            
            # Show direct console output for the result
            if current_result[0] != "Unknown":
                print(f"✅ Found: {current_result} in event {i+1}")
            else:
                print(f"⚠️ No known personnel found in event {i+1}")
                
        except Exception as e:
            logger.error(f"Error during extraction for event {i+1}/{total_rows}: {e}")
            print(f"❌ Error processing event {i+1}: {str(e)[:100]}")
            results.append(["Unknown_Error"])
            time.sleep(1)  # Add a delay after errors
            
            # Try to refresh the client after errors
            error_count = sum(1 for r in results[-5:] if r == ["Unknown_Error"])
            if error_count >= 3:  # If we've had several errors in a row
                print("Multiple errors detected. Refreshing LLM client...")
                client = get_llm_client()

        # Update progress bars
        term_progress.update(1)
        
        # Add a small delay between requests to avoid overwhelming the API
        time.sleep(0.1)

    # Close the terminal progress bar
    term_progress.close()
    
    # Calculate elapsed time and stats
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / total_rows if total_rows else 0
    
    # Complete the progress bar
    progress_bar.progress(1.0, text=f"Extraction complete! Processed {total_rows} events in {elapsed_time:.2f}s")
    
    # Print final status
    print(f"\n\033[1m>>> EXTRACTION COMPLETE: Processed {total_rows} events in {elapsed_time:.2f}s ({avg_time:.2f}s per event) <<<\033[0m")
    
    # Add the results to the dataframe
    df_copy['extracted_personnel'] = results
    
    # Display final stats in Streamlit
    st.success(f"Extraction complete! Processed {total_rows} events in {elapsed_time:.2f} seconds.")
    
    return df_copy
