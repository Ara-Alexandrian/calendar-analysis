# functions/llm_extraction/normalizer.py
"""
Functions for normalizing extracted personnel names from LLM responses.
"""
import logging
import pandas as pd
from functions import config_manager

# Configure logging
logger = logging.getLogger(__name__)


def normalize_event_type(event_type_str: str | None, mapping: dict) -> str:
    """
    Normalizes a raw event type string using a predefined mapping.

    Args:
        event_type_str: The raw event type string extracted by the LLM, or None.
        mapping: A dictionary mapping lowercase raw strings to standardized names.

    Returns:
        The standardized event type name, or the original string if no mapping is found
        or if the input is invalid. Returns "Unknown" if input is None or empty after stripping.
    """
    if not isinstance(event_type_str, str) or not event_type_str:
        logger.debug(f"Invalid or empty event_type_str received: {event_type_str}. Returning 'Unknown'.")
        return "Unknown"

    cleaned_str = event_type_str.strip().lower()
    if not cleaned_str:
        logger.debug(f"Event type string became empty after cleaning: '{event_type_str}'. Returning 'Unknown'.")
        return "Unknown"

    normalized_value = mapping.get(cleaned_str)

    if normalized_value:
        logger.debug(f"Normalized '{event_type_str}' (cleaned: '{cleaned_str}') to '{normalized_value}'")
        return normalized_value
    else:
        # Return the original (but stripped) string if no mapping found,
        # potentially capitalizing it for consistency if desired, but let's keep it simple for now.
        logger.warning(f"No mapping found for event type '{cleaned_str}' (original: '{event_type_str}'). Returning cleaned version.")
        # Consider returning event_type_str.strip().title() or similar if capitalization is desired for unmapped types
        return cleaned_str


def normalize_extracted_personnel(df) -> pd.DataFrame:
    """
    Normalizes the 'extracted_personnel' column (output from LLM)
    using the variation map to create the 'assigned_personnel' column.

    Handles lists, 'Unknown', and 'Unknown_Error'.

    Args:
        df: DataFrame containing an 'extracted_personnel' column with lists of names
            or special markers like 'Unknown' or 'Unknown_Error'

    Returns:
        DataFrame with a new 'assigned_personnel' column containing normalized names
    """
    # Handle the case when a list is passed instead of a DataFrame
    if not isinstance(df, pd.DataFrame):
        logger.error(f"Expected DataFrame but received {type(df)}. Converting to DataFrame.")
        try:
            # Try to convert list to DataFrame if possible
            if isinstance(df, list):
                # Create a basic DataFrame with the list as 'extracted_personnel' column
                df = pd.DataFrame({'extracted_personnel': df})
            else:
                # For other unexpected types, create an empty DataFrame with placeholder
                logger.error(f"Could not convert {type(df)} to DataFrame. Creating empty DataFrame.")
                df = pd.DataFrame({'extracted_personnel': [['Unknown']]})
        except Exception as e:
            logger.error(f"Error converting to DataFrame: {e}")
            df = pd.DataFrame({'extracted_personnel': [['Unknown']]})

    if 'extracted_personnel' not in df.columns:
        logger.warning("Column 'extracted_personnel' not found for normalization. Skipping.")
        # Add placeholder if analysis expects it
        df['assigned_personnel'] = [['Unknown']] * len(df)
        return df

    # Get canonical names and variations map from configuration
    # Use force_reload=True to ensure we have the latest config
    personnel_config, variation_map, canonical_names = config_manager.load_personnel_config(force_reload=True)
    if not variation_map:
        logger.error("Variation map is empty. Cannot normalize names.")
        # Return original column or a default if it doesn't exist
        if 'extracted_personnel' in df.columns:
             df['assigned_personnel'] = df['extracted_personnel'] # Pass through if map empty
        else:
             df['assigned_personnel'] = [['Unknown']] * len(df) # Default if col missing
        return df

    df_copy = df.copy()
    logger.debug(f"Starting normalization. Variation map loaded with {len(variation_map)} entries.") # Log map size

    def normalize(extracted_item):
        """Normalize a single extracted_personnel item (list or string)."""
        # Initialize empty list for normalized names
        normalized_list = []

        # Handle special string cases like "Unknown" or "Unknown_Error"
        if isinstance(extracted_item, str):
            logger.warning(f"Normalizer expected list but got string: '{extracted_item}'")
            if extracted_item == "Unknown" or extracted_item == "Unknown_Error":
                return ["Unknown"]
            else:
                # Try to treat single string as a name
                item_lower = extracted_item.lower() # Use lowercase for map lookup
                if extracted_item in canonical_names:
                    logger.debug(f"Normalized single string '{extracted_item}' to canonical: '{extracted_item}'")
                    return [extracted_item]
                elif item_lower in variation_map:
                    canonical_name = variation_map[item_lower]
                    logger.debug(f"Normalized single string '{extracted_item}' via variation map to: '{canonical_name}'")
                    return [canonical_name]
                else:
                    logger.warning(f"Single string '{extracted_item}' not found in canonical names or variations.")
                    return ["Unknown"]

        # Normal case: list of extracted names
        elif isinstance(extracted_item, list):
            normalized_set = set()  # Use set to handle duplicates from LLM
            logger.debug(f"Normalizing list: {extracted_item}") # Log the input list
            for item in extracted_item:
                if not isinstance(item, str) or not item.strip():
                    logger.debug(f"Skipping non-string or empty item in list: '{item}'")
                    continue  # Skip non-string or empty items

                item_clean = item.strip()
                item_lower = item_clean.lower()

                # Check if name is already canonical
                if item_clean in canonical_names:
                    logger.debug(f"Item '{item_clean}' is already canonical.")
                    normalized_set.add(item_clean)
                    continue

                # Check if it's a known variation (using lowercase)
                if item_lower in variation_map:
                    canonical_name = variation_map[item_lower]
                    logger.debug(f"Mapped variation '{item_clean}' (lowercase: '{item_lower}') to canonical: '{canonical_name}'")
                    normalized_set.add(canonical_name)
                    continue

                # Log if no match found, unless it's a known non-match marker
                if item_clean not in ["Unknown", "Unknown_Error"]:
                    logger.info(f"Name '{item_clean}' not found in canonical names or variations map (lowercase key: '{item_lower}').")

            # Convert set back to list
            normalized_list = list(normalized_set)
            logger.debug(f"Normalization result for list {extracted_item}: {normalized_list}")

            # If we didn't find any names, return "Unknown"
            if not normalized_list:
                logger.debug(f"List {extracted_item} resulted in empty normalized list, returning ['Unknown']")
                return ["Unknown"]
            return normalized_list

        # Fallback for unexpected types
        else:
            logger.warning(f"Unexpected type in extracted_personnel: {type(extracted_item)}. Value: {extracted_item}")
            return ["Unknown"]

    # Apply normalization to each row
    try:
        df_copy['assigned_personnel'] = df_copy['extracted_personnel'].apply(normalize)
        logger.info(f"Normalization applied to {len(df_copy)} entries.")
    except Exception as e:
        logger.error(f"Error during normalization .apply() process: {e}", exc_info=True) # Log full traceback
        df_copy['assigned_personnel'] = [['Unknown_Error']] * len(df_copy) # Mark as error

    return df_copy
