# llm_extractor.py
import logging
import json
from json.decoder import JSONDecodeError
import config
import ollama # Ensure ollama is imported if client setup is here

# Configure logging (ensure it's set up)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- LLM Client Setup --- (Keep your existing setup here)
llm_client = None
if config.LLM_PROVIDER == "ollama":
    try:
        import ollama
        llm_client = ollama.Client(host=config.OLLAMA_BASE_URL)
        # Optional: Connection check can go here
        # llm_client.list()
        # logging.info(f"Successfully connected...")
    except ImportError:
        logging.error("Ollama library not installed. Please install with 'pip install ollama'")
        llm_client = None # Ensure it's None on error
    except Exception as e:
        logging.error(f"Error initializing Ollama client: {e}")
        llm_client = None # Ensure it's None on error
else:
    logging.warning(f"LLM_PROVIDER is set to '{config.LLM_PROVIDER}', not 'ollama'. No Ollama client created.")
    llm_client = None


# --- Extraction Function (UPDATED) ---

def extract_physicists_llm(summary: str) -> list[str]:
    """
    Uses the configured Ollama LLM to extract canonical physicist names from a summary string,
    handling various JSON output formats from the LLM.

    Args:
        summary: The calendar event summary string.

    Returns:
        A list of identified canonical physicist names, or ["Unknown"] if none are found
        or an error occurs.
    """
    if not llm_client:
        # Log error only once or less frequently if needed to avoid spamming logs
        # logging.error(f"Ollama client not initialized (host: {config.OLLAMA_BASE_URL}). Cannot extract names.")
        return ["Unknown"]
    if not summary or not isinstance(summary, str):
        logging.debug("Skipping extraction for empty or non-string summary.")
        return ["Unknown"]

    # Construct the prompt (Consider minor tweaks for clarity)
    prompt = f"""
    Your task is to identify physicist names from the provided calendar event summary.
    You are given a specific list of known canonical physicist names.
    Analyze the summary and identify ONLY the canonical names from the list below that are mentioned or clearly referenced in the summary.
    Consider variations in names (e.g., initials, last names only) but map them back to a name present in the canonical list.
    Do not guess or include names not on the list. If multiple physicists are mentioned, include all of them.
    If no physicists from the list are clearly identified, return an empty list.

    Known Canonical Physicist Names:
    {json.dumps(config.CANONICAL_NAMES, indent=2)}

    Event Summary:
    "{summary}"

    Respond ONLY with a valid JSON structure containing the identified canonical names.
    The ideal response is a JSON list like ["Name1", "Name2"] or [].
    If you must use a dictionary, use a key like "names" for the list, e.g., {{"names": ["Name1", "Name2"]}}.
    Do not add explanations or surrounding text.
    """

    extracted_names = [] # Initialize empty list

    try:
        response = llm_client.chat(
            model=config.LLM_MODEL,
            messages=[
                # System prompt can sometimes help reinforce instructions
                {"role": "system", "content": "You are an assistant that extracts specific names from text based on a provided list and outputs ONLY valid JSON containing those names."},
                {"role": "user", "content": prompt}
            ],
            format='json', # Request JSON format
            options={'temperature': 0.1} # Keep temperature low
        )

        content = response.get('message', {}).get('content', '')
        if not content:
             logging.warning(f"LLM returned empty content for summary: '{summary}'")
             return ["Unknown"] # Treat empty content as failure

        # --- Enhanced JSON Parsing Logic ---
        try:
            extracted_data = json.loads(content)

            # 1. Ideal case: Is it directly a list?
            if isinstance(extracted_data, list):
                extracted_names = extracted_data
                logging.debug(f"LLM returned a direct list for '{summary}'")

            # 2. Common case: Is it a dictionary containing a list?
            elif isinstance(extracted_data, dict):
                logging.debug(f"LLM returned dict for '{summary}': {content}")
                found_list = None
                # Check common keys first
                possible_keys = ['names', 'physicists', 'identified_names', 'canonical_names', 'identifiedCanonicalNames', 'result', 'output']
                for key in possible_keys:
                    if key in extracted_data and isinstance(extracted_data.get(key), list):
                        found_list = extracted_data[key]
                        logging.debug(f"Found list under key '{key}'")
                        break # Found it

                # Fallback: If no known key worked, check *any* value that is a list
                if found_list is None:
                    for value in extracted_data.values():
                        if isinstance(value, list):
                            found_list = value
                            logging.debug("Found list as a dictionary value (unknown key)")
                            break # Use the first list found

                # Fallback: Handle the "{'Name1': true, 'Name2': true}" case
                if found_list is None:
                     # Check if all values are boolean (or maybe just True)
                     all_bools = all(isinstance(v, bool) for v in extracted_data.values())
                     # Or specifically check for True if that's the pattern
                     # all_true = all(v is True for v in extracted_data.values())
                     if extracted_data and all_bools: # Ensure dict not empty
                         found_list = list(extracted_data.keys()) # Assume keys are the names
                         logging.debug("Found names as keys in a boolean dictionary")

                if found_list is not None:
                    extracted_names = found_list
                else:
                    # If it's a dictionary but we couldn't find a list or expected structure
                    logging.warning(f"LLM returned dict, but failed to extract expected list structure: {content}")
                    # Keep extracted_names as []

            # 3. Handle unexpected format
            else:
                 logging.warning(f"LLM (JSON mode) returned unexpected format (not list or dict): {content}")
                 # Keep extracted_names as []

        except JSONDecodeError:
            logging.error(f"Failed to decode JSON response from LLM (model: {config.LLM_MODEL}): {content}")
            return ["Unknown"] # Treat JSON decode error as failure
        except Exception as e:
             logging.error(f"Error processing LLM response content structure: {e}\nResponse content: {content}")
             return ["Unknown"] # Treat other parsing errors as failure

        # --- Validation ---
        # Ensure extracted names are strings before validation
        validated_names = [str(name) for name in extracted_names if str(name) in config.CANONICAL_NAMES]

        if not validated_names:
            if not extracted_names:
                 # LLM correctly identified no names OR parsing failed to find any list
                 logging.debug(f"LLM/Parsing found no known physicists in: '{summary}' (Raw response: {content})")
            else:
                 # LLM returned names, but none matched the canonical list after filtering
                 logging.warning(f"LLM (model: {config.LLM_MODEL}) returned names not in canonical list for '{summary}': {extracted_names}. Filtered. (Raw: {content})")
            # Return ["Unknown"] if validation is empty, simplifying downstream logic
            return ["Unknown"]
        else:
            logging.debug(f"LLM (model: {config.LLM_MODEL}) successfully identified & validated {validated_names} in '{summary}'")
            return validated_names

    except Exception as e:
        # Catch Ollama client errors / network errors etc.
        logging.error(f"Error calling Ollama API (model: {config.LLM_MODEL}, host: {config.OLLAMA_BASE_URL}) for summary '{summary}': {e}")
        # Consider adding more specific error handling (e.g., connection errors) if needed
        return ["Unknown"]