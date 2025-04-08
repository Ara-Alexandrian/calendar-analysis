# functions/llm_extractor.py
import logging
import json
from json.decoder import JSONDecodeError
import ollama
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
import streamlit as st # Import Streamlit for caching

# Use static settings
from config import settings
# Use config manager for dynamic personnel data access
from functions import config_manager

# Configure logging (ensure it's set up in streamlit_app.py or main entry point)
logger = logging.getLogger(__name__)

# --- LLM Client Setup ---
# Use Streamlit's caching for the client to avoid re-initializing on every run
@st.cache_resource # Cache the client resource
def get_llm_client():
    """Initializes and returns the Ollama client, or None if unavailable."""
    if settings.LLM_PROVIDER == "ollama":
        try:
            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            # Perform a quick check to see if the client can connect
            client.list() # Throws error if server not reachable
            logger.info(f"Successfully connected to Ollama server at {settings.OLLAMA_BASE_URL}")
            return client
        except ImportError:
            logger.error("Ollama library not installed. Please install with 'pip install ollama'")
            st.error("Ollama library not found. Please install it (`pip install ollama`). LLM features disabled.", icon="🚨")
            return None
        except Exception as e:
            logger.error(f"Error initializing or connecting to Ollama client at {settings.OLLAMA_BASE_URL}: {e}")
            st.error(f"Could not connect to Ollama server at {settings.OLLAMA_BASE_URL}. Check if it's running. LLM features disabled.", icon="🚨")
            return None
    else:
        logger.warning(f"LLM_PROVIDER is not 'ollama' (current: {settings.LLM_PROVIDER}). Ollama client not created.")
        return None

# Function to check if LLM is ready (can be called from UI)
def is_llm_ready():
    client = get_llm_client()
    return client is not None

# --- Single Extraction Function (Worker for Parallel Execution) ---
def _extract_single_physicist_llm(summary: str, llm_client, canonical_names: list[str]) -> list[str] | str:
    """
    Internal function to extract names for a single summary using the LLM.
    Returns list of names on success, or "Unknown" / "Unknown_Error" on failure.
    """
    if not llm_client:
        # Should not happen if called correctly, but safety check
        return "Unknown_Error"
    if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
        # logger.debug("Skipping extraction for empty or non-string summary.")
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
        response = llm_client.chat(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an assistant that extracts specific names from text based on a provided list and outputs ONLY a valid JSON list containing those names."},
                {"role": "user", "content": prompt}
            ],
            format='json', # Request JSON format
            options={'temperature': 0.1} # Low temperature for deterministic output
        )

        content = response.get('message', {}).get('content', '')
        if not content:
             # logger.warning(f"LLM returned empty content for summary: '{summary[:50]}...'")
             return ["Unknown"] # Treat empty content as no names found

        # --- Enhanced JSON Parsing Logic ---
        extracted_names = []
        try:
            extracted_data = json.loads(content)

            # 1. Ideal case: Is it directly a list?
            if isinstance(extracted_data, list):
                # Validate items are strings
                extracted_names = [item for item in extracted_data if isinstance(item, str)]
                # logger.debug(f"LLM returned a direct list for '{summary[:50]}...'")

            # 2. Common case: Is it a dictionary containing a list under common keys?
            elif isinstance(extracted_data, dict):
                # logger.debug(f"LLM returned dict for '{summary[:50]}...': {content}")
                found_list = None
                possible_keys = ['names', 'physicists', 'identified_names', 'canonical_names', 'result', 'output']
                for key in possible_keys:
                    if key in extracted_data and isinstance(extracted_data.get(key), list):
                        potential_list = extracted_data[key]
                        # Validate items are strings
                        found_list = [item for item in potential_list if isinstance(item, str)]
                        # logger.debug(f"Found list under key '{key}'")
                        break # Found it

                # Fallback: check *any* value that is a list of strings
                if found_list is None:
                    for value in extracted_data.values():
                        if isinstance(value, list):
                            potential_list = value
                            # Validate items are strings
                            string_list = [item for item in potential_list if isinstance(item, str)]
                            if string_list: # Use the first non-empty list of strings found
                                found_list = string_list
                                # logger.debug("Found list as a dictionary value (unknown key)")
                                break

                if found_list is not None:
                    extracted_names = found_list
                else:
                    # logger.warning(f"LLM returned dict, but failed to extract expected list structure: {content}")
                    extracted_names = [] # Treat as no names found

            # 3. Handle unexpected format
            else:
                 # logger.warning(f"LLM (JSON mode) returned unexpected format (not list or dict): {content}")
                 extracted_names = [] # Treat as no names found

        except JSONDecodeError:
            # logger.error(f"Failed to decode JSON response from LLM for '{summary[:50]}...': {content}")
            return "Unknown_Error" # Indicate a structural error from LLM
        except Exception as e:
             # logger.error(f"Error processing LLM response content structure: {e}\nResponse content: {content}")
             return "Unknown_Error"

        # --- Validation Against Canonical List ---
        validated_names = [name for name in extracted_names if name in canonical_names]

        if not validated_names:
            # logger.debug(f"LLM found no known physicists in: '{summary[:50]}...' (Raw response: {content})")
            return ["Unknown"] # No known names found or returned list was empty
        else:
            # logger.debug(f"LLM validated {validated_names} in '{summary[:50]}...'")
            return validated_names # Return list of validated canonical names

    except Exception as e:
        # Catch Ollama client errors / network errors etc. during the API call
        # logger.error(f"Error calling Ollama API for summary '{summary[:50]}...': {e}")
        # Indicate an error occurred during the call
        return "Unknown_Error"

# --- Parallel Extraction Orchestration ---
def run_llm_extraction_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs LLM extraction in parallel on the 'summary' column of the DataFrame.

    Args:
        df: DataFrame with a 'summary' column.

    Returns:
        DataFrame with a new 'extracted_personnel' column containing lists of names
        or error strings ('Unknown', 'Unknown_Error').
    """
    if not is_llm_ready():
        st.error("LLM client is not available. Cannot perform extraction.")
        df['extracted_personnel'] = "Unknown_Error" # Add column indicating failure
        return df

    if 'summary' not in df.columns:
        logger.error("'summary' column not found in DataFrame. Cannot extract.")
        st.error("Input data is missing the 'summary' column.")
        df['extracted_personnel'] = "Unknown_Error"
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
        df['extracted_personnel'] = "Unknown_Error"
        return df

    start_time = time.time()
    logger.info(f"Starting parallel LLM extraction for {len(summaries)} summaries using {settings.LLM_MAX_WORKERS} workers...")
    st.info(f"Starting LLM name extraction for {len(summaries)} events using model '{settings.LLM_MODEL}'...")

    # Streamlit progress bar
    progress_bar = st.progress(0, text="Initializing LLM extraction...")
    total_summaries = len(summaries)
    completed_count = 0

    try:
        with ThreadPoolExecutor(max_workers=settings.LLM_MAX_WORKERS) as executor:
            # Map futures to original index
            future_to_index = {
                executor.submit(_extract_single_physicist_llm, summary, llm_client, canonical_names): i
                for i, summary in enumerate(summaries)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result() # Get the list of names or error string
                    results[index] = result
                except Exception as exc:
                    # Log error for the specific summary
                    logger.error(f"Summary index {index} ('{summaries[index][:50]}...') generated an exception in thread future: {exc}")
                    results[index] = "Unknown_Error" # Mark as error

                # Update progress bar
                completed_count += 1
                progress_percent = int((completed_count / total_summaries) * 100)
                progress_bar.progress(progress_percent, text=f"Processing event {completed_count}/{total_summaries}...")

        progress_bar.progress(100, text="LLM Extraction Complete!")
        end_time = time.time()
        logger.info(f"LLM extraction finished in {end_time - start_time:.2f} seconds.")
        st.success(f"LLM extraction finished in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        logger.error(f"An error occurred during the parallel LLM execution setup: {e}")
        st.error(f"A critical error occurred during LLM processing: {e}")
        # Fill remaining results with error marker if needed
        for i in range(len(results)):
            if results[i] is None:
                results[i] = "Unknown_Error"

    df_copy['extracted_personnel'] = results
    return df_copy


# --- Normalization Function ---
# This function takes the raw output from the LLM ('extracted_personnel')
# and maps it using the VARIATION_MAP, returning the final 'assigned_personnel' list.
def normalize_extracted_personnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'extracted_personnel' column (output from LLM)
    using the variation map to create the 'assigned_personnel' column.

    Handles lists, 'Unknown', and 'Unknown_Error'.
    """
    if 'extracted_personnel' not in df.columns:
        logger.warning("Column 'extracted_personnel' not found for normalization. Skipping.")
        # Add placeholder if analysis expects it
        df['assigned_personnel'] = [['Unknown']] * len(df)
        return df

    variation_map = config_manager.get_variation_map()
    df_copy = df.copy()

    def normalize(extracted_item):
        if isinstance(extracted_item, list):
            normalized_list = set() # Use set to handle duplicates from LLM
            for item in extracted_item:
                 # LLM should return canonical names directly now, but map just in case
                 # Or handle variations if prompt asked for variations
                 # Assuming LLM returns names matching canonical list:
                 if item in config_manager.get_canonical_names():
                     normalized_list.add(item)
                 # Add mapping logic here if LLM returns variations instead
                 # elif item.lower() in variation_map:
                 #     normalized_list.add(variation_map[item.lower()])
                 elif item != "Unknown": # Log unexpected items not in canonical list
                      logger.debug(f"LLM returned name '{item}' not in current canonical list. Ignoring.")

            if not normalized_list:
                return ["Unknown"] # Return list containing 'Unknown'
            return sorted(list(normalized_list)) # Return sorted list of unique canonical names

        elif isinstance(extracted_item, str) and extracted_item.startswith("Unknown"):
            # Keep "Unknown" or "Unknown_Error" as is, but ensure it's in a list
            return [extracted_item]
        else:
            # Handle unexpected types, treat as Unknown
            logger.warning(f"Unexpected data type in 'extracted_personnel': {type(extracted_item)}. Treating as Unknown.")
            return ["Unknown"] # Return list containing 'Unknown'

    df_copy['assigned_personnel'] = df_copy['extracted_personnel'].apply(normalize)

    # Log value counts for debugging
    logger.info("Counts of assigned personnel lists (first element shown for brevity if list):")
    try:
        # Show counts of the list representation or the string if not list
        counts = df_copy['assigned_personnel'].apply(lambda x: str(x) if isinstance(x, list) else x).value_counts()
        logger.info("\n" + counts.to_string(max_rows=50))
    except Exception as e:
        logger.warning(f"Could not generate value counts for assigned_personnel: {e}")


    return df_copy