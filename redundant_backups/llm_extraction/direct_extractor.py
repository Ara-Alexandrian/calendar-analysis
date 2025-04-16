"""
Direct Ollama extractor with guaranteed console output.
This implementation uses direct httpx requests with explicit print statements
to ensure that Ollama responses are visible in the console.
"""
import os
import sys
import time
import json
import logging
import httpx  # Added missing httpx import
from pprint import pprint
import pandas as pd
import streamlit as st
import re
from json.decoder import JSONDecodeError

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from config import settings

# Set up logging
logger = logging.getLogger(__name__)

def extract_with_direct_output(df, progress_bar=None, status_text=None, results_display=None):
    """
    Process calendar events with direct calls to Ollama API.
    Provides detailed error handling and streaming response processing.
    """
    # Convert df to a copy to avoid SettingWithCopyWarning
    df_copy = df.copy()
    
    # Prepare the result container
    results = []
    last_results = []
    
    # Get total items to process for progress tracking
    total = len(df_copy)
      # Get the canonical names list from configuration
    try:
        from functions import config_manager
        # Try to load using our new function
        canonical_names = config_manager.get_all_personnel()
    except (AttributeError, ImportError) as e:
        # Fallback method: load the personnel directly from config file
        print(f"Warning: {e}. Falling back to direct loading of personnel config...")
        import json
        from config import settings
        try:
            with open(settings.PERSONNEL_CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            canonical_names = list(config_data.keys())
            print(f"Successfully loaded {len(canonical_names)} personnel names directly.")
        except Exception as load_error:
            print(f"Error loading personnel: {load_error}")
            canonical_names = []
    
    try:
        print(f"Starting extraction with direct output for {total} events...")
        
        for i, row in enumerate(df_copy.iterrows()):
            # Get summary from the row
            summary = row[1].get('summary', '')
            if not summary or not isinstance(summary, str):
                print(f"[{i+1}/{total}] Empty or invalid summary, skipping")
                results.append(["Unknown"])
                continue
                
            # Update progress bar and status
            if progress_bar:
                progress = (i+1) / total
                progress_bar.progress(progress, f"Processing {i+1}/{total}: {progress*100:.1f}%")
            if status_text:
                status_text.write(f"**Processing:** {i+1}/{total} - \"{summary[:50]}...\"")
            
            # Print what we're processing with clear formatting
            print(f"\n{'-'*80}")
            print(f"[{i+1}/{total}] Processing: \"{summary[:100]}\"")
              # Build the prompt
            import json  # Ensure json is imported here for use in the f-string
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
            
            # Import settings at the module level to ensure it's available throughout the function
            # The from config import settings line at the top of the file should be used instead
            payload = {
                "model": settings.LLM_MODEL,  # Use the globally imported settings
                "messages": [
                    {"role": "system", "content": "You are an assistant that extracts specific names from text based on a provided list and outputs ONLY a valid JSON list containing those names."},
                    {"role": "user", "content": prompt}
                ],
                "format": "json"
            }
            
            # Initialize variables for retry mechanism
            max_retries = 3
            retry_count = 0
            extracted_names = []
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Make a direct API call with httpx to handle streaming JSON
                    print(f"Sending request to Ollama API (attempt {retry_count+1}/{max_retries})...")
                    start_time = time.time()
                    
                    # Use a direct httpx call with stream=True to handle streaming responses
                    complete_content = ""
                    last_chunk = {}
                    
                    # Increase timeout for later retry attempts
                    timeout_seconds = 45.0 + (retry_count * 15.0)
                    
                    with httpx.stream(
                        "POST",
                        f"{settings.OLLAMA_BASE_URL}/api/chat",
                        json=payload,
                        timeout=timeout_seconds  # Increasing timeout for reliability
                    ) as response:
                        # Print status
                        print(f"Response initiated: Status {response.status_code}")
                        
                        if response.status_code != 200:
                            print(f"Error: Received non-200 status code: {response.status_code}")
                            raise Exception(f"API returned status code {response.status_code}")
                        
                        # Process each chunk from the streaming response
                        print("\nReceiving streaming response:")
                        for chunk in response.iter_text():
                            print(f"CHUNK: {chunk}")  # Print the raw chunk
                            
                            try:
                                # Each chunk should be a separate JSON object
                                chunk_data = json.loads(chunk)
                                
                                # Extract content from the chunk
                                if 'message' in chunk_data and 'content' in chunk_data['message']:
                                    chunk_content = chunk_data['message']['content']
                                    complete_content += chunk_content
                                    print(f"Content added: '{chunk_content}'")
                                
                                # Check if this is the final chunk with done=true
                                if chunk_data.get('done') == True:
                                    print("Received final chunk (done=true)")
                                
                                # Keep track of the last complete chunk
                                last_chunk = chunk_data
                            except json.JSONDecodeError:
                                print(f"Invalid JSON chunk: {chunk}")
                    
                    elapsed_time = time.time() - start_time
                    
                    # Print the response status and timing
                    print(f"Response completed: Status {response.status_code} in {elapsed_time:.2f}s")
                      # Print the reconstructed complete content
                    print("\nComplete reconstructed content:")
                    print(complete_content)
                    
                    # Check if response is just emojis or other non-standard content
                    has_only_special_chars = all(not c.isalnum() and not c.isspace() for c in complete_content.strip())
                    if has_only_special_chars and complete_content.strip():
                        print(f"Warning: Response contains only emojis or special characters: {complete_content}")
                        print("Using empty result due to invalid response format")
                        extracted_names = []
                        success = True  # Mark as success to avoid retries with same result
                        continue
                    
                    # Directly look for known physicist names in the complete content
                    # This is a more direct approach that doesn't rely on JSON parsing
                    print("\nDirectly searching for known names in response:")
                    direct_matches = []
                    for name in canonical_names:
                        # Look for exact matches of canonical names in the response
                        if name in complete_content:
                            direct_matches.append(name)
                            print(f"Found direct match: '{name}'")
                    
                    # If we found direct matches, use them
                    if direct_matches:
                        # Use these names directly
                        print(f"Using {len(direct_matches)} directly matched names: {direct_matches}")
                        extracted_names = direct_matches
                        success = True
                    else:
                        # If no direct matches, try to parse JSON
                        try:
                            # Try to extract a JSON list from the content
                            import re
                            json_match = re.search(r'\[\s*(?:"[^"]*"\s*,?\s*)*\]', complete_content)
                            if json_match:
                                json_str = json_match.group(0)
                                print(f"Extracted JSON string: {json_str}")
                                try:
                                    extracted_list = json.loads(json_str)
                                    if isinstance(extracted_list, list):
                                        extracted_names = extracted_list
                                        print(f"Extracted {len(extracted_names)} names from JSON: {extracted_names}")
                                        success = True
                                except json.JSONDecodeError:
                                    print(f"Failed to parse extracted JSON: {json_str}")
                        except Exception as parsing_e:
                            print(f"Error parsing response content: {parsing_e}")
                    
                    # If we were able to extract names, consider this successful
                    if success:
                        break
                
                except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError) as timeout_error:
                    print(f"Timeout error: {timeout_error}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying ({retry_count}/{max_retries})...")
                        time.sleep(2)  # Wait before retrying
                    else:
                        print(f"Max retries reached. Using empty result.")
                        extracted_names = []
                
                except Exception as e:
                    print(f"Error during API call: {e}")
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying ({retry_count}/{max_retries})...")
                        time.sleep(2)  # Wait before retrying
                    else:
                        print(f"Max retries reached. Using empty result.")
                        extracted_names = []
            
            # Regardless of success or failure, add results
            results.append(extracted_names)
            print(f"Final result for event {i+1}: {extracted_names}")
            
            # Update the display with the last 5 results
            if results_display is not None:
                last_result = results[-1]
                result_text = f"**Event {i+1}:** '{summary[:40]}...' → **Result:** {last_result}"
                last_results.append(result_text)
                if len(last_results) > 5:
                    last_results.pop(0)
                results_display.markdown("\n".join(last_results))            # Add a delay between requests to avoid overwhelming the server
            # Increase the delay to give Ollama more time to recover between requests
            time.sleep(2)  # Increased from 0.5 to 2 seconds
        
        print(f"Extraction complete. Processed {len(results)} events.")
        
        # Instead of returning just a list of results, add it to the DataFrame
        df_copy['extracted_personnel'] = results
        print(f"Created DataFrame with {len(df_copy)} rows and extracted_personnel column")
        return df_copy
            
    except Exception as e:
        import traceback
        print(f"Error in extract_with_direct_output: {e}")
        traceback.print_exc()
        
        # In case of error, still return a proper DataFrame with Unknown values
        if 'df_copy' in locals() and isinstance(df_copy, pd.DataFrame):
            df_copy['extracted_personnel'] = [['Unknown']] * len(df_copy) 
            return df_copy
        else:
            # If df_copy isn't available, return an empty DataFrame with the right structure
            return pd.DataFrame({'extracted_personnel': [['Unknown']]})
