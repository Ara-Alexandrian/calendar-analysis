# functions/llm_extraction/normalizer.py
"""
Functions for normalizing extracted personnel names from LLM responses.
"""
import logging
import pandas as pd
from functions import config_manager

# Configure logging
logger = logging.getLogger(__name__)

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

    variation_map = config_manager.get_variation_map()
    df_copy = df.copy()

    def normalize(extracted_item):
        """Normalize a single extracted_personnel item (list or string)."""
        if isinstance(extracted_item, list):
            normalized_list = set()  # Use set to handle duplicates from LLM
            for item in extracted_item:
                 # LLM should return canonical names directly now, but map just in case
                 # Or handle variations if prompt asked for variations
                 # Assuming LLM returns names matching canonical list:
                 if item in config_manager.get_canonical_names():
                     normalized_list.add(item)
                 # Add mapping logic here if LLM returns variations instead
                 # elif item.lower() in variation_map:
                 #     normalized_list.add(variation_map[item.lower()])
                 elif item != "Unknown":  # Log unexpected items not in canonical list
                      logger.debug(f"LLM returned name '{item}' not in current canonical list. Ignoring.")

            if not normalized_list:
                return ["Unknown"]  # Return list containing 'Unknown'
            return sorted(list(normalized_list))  # Return sorted list of unique canonical names

        elif isinstance(extracted_item, str) and extracted_item.startswith("Unknown"):
            # Keep "Unknown" or "Unknown_Error" as is, but ensure it's in a list
            return [extracted_item]
        else:
            # Handle unexpected types, treat as Unknown
            logger.warning(f"Unexpected data type in 'extracted_personnel': {type(extracted_item)}. Treating as Unknown.")
            return ["Unknown"]  # Return list containing 'Unknown'

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