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

    # Get canonical names and variations map from configuration
    canonical_names = config_manager.get_canonical_names()
    variation_map = config_manager.get_variation_map()
    df_copy = df.copy()

    def normalize(extracted_item):
        """Normalize a single extracted_personnel item (list or string)."""
        # Initialize empty list for normalized names
        normalized_list = []
        
        # Handle special string cases like "Unknown" or "Unknown_Error"
        if isinstance(extracted_item, str):
            logger.warning(f"Expected list but got string: {extracted_item}")
            if extracted_item == "Unknown" or extracted_item == "Unknown_Error":
                return ["Unknown"]
            else:
                # Try to treat single string as a name
                if extracted_item in canonical_names:
                    return [extracted_item]
                # Check if it's in variation map
                elif extracted_item in variation_map:
                    return [variation_map[extracted_item]]
                else:
                    return ["Unknown"]
                    
        # Normal case: list of extracted names
        elif isinstance(extracted_item, list):
            normalized_set = set()  # Use set to handle duplicates from LLM
            for item in extracted_item:
                if not isinstance(item, str):
                    continue  # Skip non-string items
                    
                # Check if name is already canonical
                if item in canonical_names:
                    normalized_set.add(item)
                    continue
                    
                # Check if it's a known variation
                if item in variation_map:
                    normalized_set.add(variation_map[item])
                    continue
                    
                # No match found, leave as is if not "Unknown"
                if item != "Unknown_Error":
                    logger.info(f"Name '{item}' not found in canonical names or variations")
            
            # Convert set back to list
            normalized_list = list(normalized_set)
            
            # If we didn't find any names, return "Unknown"
            if not normalized_list:
                return ["Unknown"]
            return normalized_list
            
        # Fallback for unexpected types
        else:
            logger.warning(f"Unexpected type in extracted_personnel: {type(extracted_item)}")
            return ["Unknown"]

    # Apply normalization to each row
    try:
        df_copy['assigned_personnel'] = df_copy['extracted_personnel'].apply(normalize)
        logger.info(f"Normalized {len(df_copy)} personnel entries")
    except Exception as e:
        logger.error(f"Error during normalization process: {e}")
        df_copy['assigned_personnel'] = [['Unknown']] * len(df_copy)
    
    return df_copy
