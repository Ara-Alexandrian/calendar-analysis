# functions/data_processor.py
import pandas as pd
import json
import re
import logging
import os
from datetime import datetime
import io # Import io for BytesIO

# Use static settings
from config import settings
# Use config manager for dynamic personnel data access
from functions import config_manager

# Setup logger for this module
logger = logging.getLogger(__name__)

# parse_datetime remains the same as your original good version...
def parse_datetime(dt_str):
    # ... (keep your existing robust parse_datetime function here) ...
    if not dt_str: return pd.NaT # Handle None or empty strings
    # ... (rest of your parsing logic) ...
    # Make sure it handles the vDDDTypes format correctly
    # Example handling for vDDDTypes string:
    if isinstance(dt_str, str):
        # Try ISO 8601 format with timezone first (most robust)
        try:
            # Handles 'YYYY-MM-DD HH:MM:SS+ZZ:ZZ', 'YYYY-MM-DDTHH:MM:SSZ', etc.
            dt = pd.to_datetime(dt_str, errors='coerce', utc=True)
            if pd.notna(dt): return dt
        except Exception:
             pass # Try other formats

        # Handle formats like '20241016T180652Z' (common in iCal)
        try:
            dt = pd.to_datetime(dt_str, format='%Y%m%dT%H%M%SZ', errors='coerce', utc=True)
            if pd.notna(dt): return dt
        except (ValueError, TypeError):
            pass # Try next format

        # Handle vDDDTypes format by extracting the datetime part
        # Regex improved to handle optional T separator and optional timezone Z
        match = re.search(r"(\d{4}-\d{2}-\d{2}[ T]?\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)", dt_str)
        if match:
            extracted_str = match.group(1).replace('T', ' ') # Replace T with space for pandas
            try:
                # Parse the extracted string, trying common variations
                dt = pd.to_datetime(extracted_str, errors='coerce', utc=True)
                if pd.notna(dt): return dt
            except Exception as e:
                 logger.debug(f"Could not parse extracted datetime string '{extracted_str}' from '{dt_str}': {e}")
                 pass # Could not parse extracted string

    elif isinstance(dt_str, dict) and 'dt' in dt_str: # Handle potential nested structure
        # Recursively call parse_datetime on the nested value
        return parse_datetime(dt_str['dt'])

    # If all parsing attempts fail
    logger.warning(f"Could not parse datetime string: {dt_str}")
    return pd.NaT


# MODIFIED: Accepts file_content (bytes or string) instead of source_path
def load_raw_calendar_data(file_content, filename="Uploaded File") -> pd.DataFrame | None:
    """Loads raw calendar event data from uploaded file content (JSON)."""
    logger.info(f"Attempting to load raw calendar data from: {filename}")

    if not file_content:
        logger.error(f"Error: No file content provided for {filename}.")
        return None
    try:
        # If content is bytes, decode it first
        if isinstance(file_content, bytes):
            content_str = file_content.decode('utf-8')
        elif isinstance(file_content, str):
            content_str = file_content
        else:
             logger.error(f"Error: Invalid file content type: {type(file_content)}. Expected bytes or str.")
             return None

        if not content_str.strip():
             logger.error(f"Error: Uploaded file '{filename}' is empty.")
             return None

        # Now try to parse the content string
        data = json.loads(content_str)
        if not isinstance(data, list): # Expecting a list of events
             logger.error(f"Error: JSON content in '{filename}' does not contain a list of events.")
             return None

        df = pd.DataFrame(data)
        logger.info(f"Successfully loaded data from '{filename}'. Found {len(df)} events.")
        return df

    except UnicodeDecodeError as e:
        logger.error(f"Error decoding file content from '{filename}'. Ensure it's UTF-8 encoded: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error: Could not decode JSON from '{filename}'. Invalid JSON: {e}")
        return None
    except ValueError as e:
         logger.error(f"Error reading JSON into DataFrame from '{filename}'. Check format/types. Error: {e}")
         return None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading the data from '{filename}': {e}")
        return None

# preprocess_data remains largely the same...
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parses timestamps, calculates duration, cleans summary, prepares data."""
    if df is None or df.empty:
        logger.warning("Input DataFrame is empty or None. Skipping preprocessing.")
        # Return empty df with expected cols for consistency downstream
        return pd.DataFrame(columns=['uid', 'summary', 'start_time', 'end_time',
                                      'duration_minutes', 'duration_hours',
                                      'description', 'location', 'status'])

    logger.info(f"Starting preprocessing for {len(df)} events.")

    # Check required columns
    required_cols = ['start', 'end', 'summary', 'uid']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Input DataFrame missing required columns: {missing_cols}. Cannot preprocess fully.")
        # Add missing cols as NaN? Or return empty? Let's try adding them.
        for col in missing_cols:
            df[col] = None # Or pd.NA
        # Proceed with caution or return empty depending on severity
        # For now, proceed but log the error.
        # return pd.DataFrame(...) # Safer option

    df_processed = df.copy()

    # 1. Parse Timestamps
    logger.info("Parsing start and end timestamps...")
    # Ensure 'start'/'end' exist before applying parse_datetime
    if 'start' in df_processed.columns:
        df_processed['start_time'] = df_processed['start'].apply(parse_datetime)
    else:
        df_processed['start_time'] = pd.NaT
    if 'end' in df_processed.columns:
        df_processed['end_time'] = df_processed['end'].apply(parse_datetime)
    else:
         df_processed['end_time'] = pd.NaT

    # Report parsing issues
    start_nulls = df_processed['start_time'].isnull().sum()
    end_nulls = df_processed['end_time'].isnull().sum()
    if start_nulls > 0:
        logger.warning(f"Could not parse 'start' time for {start_nulls}/{len(df_processed)} events.")
    if end_nulls > 0:
        logger.warning(f"Could not parse 'end' time for {end_nulls}/{len(df_processed)} events.")

    # Drop rows where essential time parsing failed (start or end)
    original_count = len(df_processed)
    df_processed.dropna(subset=['start_time', 'end_time'], inplace=True)
    dropped_count = original_count - len(df_processed)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} rows due to missing/unparseable start or end times.")
        if len(df_processed) == 0:
            logger.error("All rows dropped due to time parsing errors. Check date formats in source JSON.")
            # Return empty df with expected columns
            return pd.DataFrame(columns=df.columns.tolist() + ['start_time', 'end_time', 'duration_minutes', 'duration_hours'])


    # Ensure timezone consistency (convert all to UTC if not already)
    # This is crucial for accurate duration calculation
    try:
        if df_processed['start_time'].dt.tz is None:
             df_processed['start_time'] = df_processed['start_time'].dt.tz_localize('UTC')
        else:
             df_processed['start_time'] = df_processed['start_time'].dt.tz_convert('UTC')

        if df_processed['end_time'].dt.tz is None:
             df_processed['end_time'] = df_processed['end_time'].dt.tz_localize('UTC')
        else:
             df_processed['end_time'] = df_processed['end_time'].dt.tz_convert('UTC')
    except Exception as e:
        logger.error(f"Error converting timestamps to UTC: {e}. Duration calculation may be incorrect.")
        # Decide how to handle: proceed with potential errors or stop? Stop is safer.
        return pd.DataFrame(columns=df_processed.columns.tolist() + ['duration_minutes', 'duration_hours'])


    # 2. Calculate Duration
    logger.info("Calculating event durations...")
    df_processed['duration'] = df_processed['end_time'] - df_processed['start_time']
    df_processed['duration_minutes'] = df_processed['duration'].dt.total_seconds() / 60.0
    df_processed['duration_hours'] = df_processed['duration_minutes'] / 60.0

    # Filter out negative or zero durations
    original_count = len(df_processed)
    min_duration = settings.DEFAULT_MIN_EVENT_DURATION_MINUTES # Use config
    invalid_duration_mask = df_processed['duration_minutes'] < min_duration

    invalid_duration_count = invalid_duration_mask.sum()
    if invalid_duration_count > 0:
        logging.warning(f"Found {invalid_duration_count} events with duration < {min_duration} minutes.")
        # logging.debug(...) # Optionally log details
        df_processed = df_processed[~invalid_duration_mask] # Keep rows with valid duration
        logging.info(f"Dropped {invalid_duration_count} rows with duration < {min_duration} minutes.")
        if len(df_processed) == 0:
            logging.error(f"All remaining rows dropped due to duration filter (< {min_duration} mins).")
            return pd.DataFrame(columns=df_processed.columns.tolist())

    # 3. Basic Summary Cleaning
    if 'summary' in df_processed.columns:
        df_processed['summary'] = df_processed['summary'].astype(str).fillna('').str.strip()
    else:
        df_processed['summary'] = '' # Ensure column exists

    # 4. Select and Rename Columns
    columns_to_keep = [
        'uid', 'summary', 'start_time', 'end_time',
        'duration_minutes', 'duration_hours',
        # Add others present in the original JSON that might be useful
        'description', 'location', 'status', #'categories', 'created', 'last_modified'
    ]
    # Ensure only existing columns are selected
    final_cols = [col for col in columns_to_keep if col in df_processed.columns]
    # Add essential calculated columns if they aren't in final_cols yet
    for col in ['start_time', 'end_time', 'duration_minutes', 'duration_hours']:
         if col not in final_cols and col in df_processed.columns:
              final_cols.append(col)

    df_final = df_processed[final_cols].copy()

    logger.info(f"Preprocessing complete. {len(df_final)} events remaining.")
    return df_final

# explode_by_physicist remains the same conceptually
# It should now use 'assigned_personnel' which is the output after LLM + normalization
def explode_by_personnel(df: pd.DataFrame, personnel_col='assigned_personnel') -> pd.DataFrame:
    """
    Explodes the DataFrame based on a list column containing assigned personnel.
    Renames the column to 'personnel'. Handles potential non-list entries gracefully.
    """
    output_col = 'personnel' # Final column name after exploding

    if personnel_col not in df.columns:
        logger.error(f"Cannot explode DataFrame: '{personnel_col}' column not found.")
        # Add the column as 'Unknown' to prevent downstream errors if needed
        # df[output_col] = 'Unknown'
        # Alternatively, return the df as is, analysis functions should handle missing col
        return df

    df_copy = df.copy() # Work on a copy

    # --- Robustly handle different data types in the personnel column ---
    def clean_and_ensure_list(item):
        if isinstance(item, list):
            # Filter out non-strings or empty strings within the list
            cleaned_list = [str(p).strip() for p in item if isinstance(p, str) and str(p).strip()]
            # If list becomes empty after cleaning, represent as 'Unknown'
            return cleaned_list if cleaned_list else ['Unknown']
        elif isinstance(item, str):
             # Handle comma-separated strings, json strings representing lists, etc.
             item_stripped = item.strip()
             if not item_stripped:
                 return ['Unknown']
             # Check if it looks like a JSON list string
             if item_stripped.startswith('[') and item_stripped.endswith(']'):
                 try:
                     parsed_list = json.loads(item_stripped)
                     if isinstance(parsed_list, list):
                         # Recursively clean the parsed list
                         return clean_and_ensure_list(parsed_list)
                 except json.JSONDecodeError:
                     # Not valid JSON list, treat as single item (or split by comma?)
                     pass # Fall through to single item handling
             # Split comma-separated strings (assuming simple comma separation)
             if ',' in item_stripped:
                  potential_list = [p.strip() for p in item_stripped.split(',') if p.strip()]
                  if len(potential_list) > 1:
                       return potential_list
             # Otherwise, treat as a single name/item
             return [item_stripped]
        elif pd.isna(item):
            return ["Unknown"]
        else:
            # Wrap other single items (like numbers if they somehow occur) in a list
            return [str(item)]

    df_copy[personnel_col] = df_copy[personnel_col].apply(clean_and_ensure_list)
    # --- End robust handling ---

    try:
        # Explode the DataFrame using the cleaned list column
        df_exploded = df_copy.explode(personnel_col, ignore_index=True)

        # Rename the exploded column
        df_exploded.rename(columns={personnel_col: output_col}, inplace=True)

        # Ensure the final 'personnel' column is string type and handle potential NaNs from explode
        df_exploded[output_col] = df_exploded[output_col].fillna('Unknown').astype(str)

        logger.info(f"DataFrame exploded by personnel. Row count changed from {len(df)} to {len(df_exploded)}.")
        return df_exploded
    except Exception as e:
         logger.error(f"Error during DataFrame explosion based on '{personnel_col}': {e}")
         # Return the original dataframe, maybe add the 'personnel' column as Unknown
         df[output_col] = 'Unknown' # Add default column
         return df


# Saving/Loading processed data might be repurposed for Export/Import in Streamlit if needed
def save_processed_data_for_export(df, file_path=settings.PROCESSED_EXPORT_PATH):
    """Saves the DataFrame (with extracted personnel) to JSON for export."""
    logger.info(f"Attempting to save processed data for export ({len(df)} events) to: {file_path}")
    if df is None or df.empty:
        logger.warning("Attempted to save an empty or None DataFrame. Skipping export.")
        return False
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert DataFrame to JSON format (list of records)
        # Handle Timestamps and Timedeltas for JSON serialization
        df_copy = df.copy()
        for col in df_copy.select_dtypes(include=['datetime64[ns, UTC]', 'datetime64[ns]']).columns:
             df_copy[col] = df_copy[col].isoformat()
        for col in df_copy.select_dtypes(include=['timedelta64[ns]']).columns:
             df_copy[col] = df_copy[col].total_seconds() # Save timedelta as seconds

        # Use pandas to_json
        df_copy.to_json(file_path, orient='records', indent=4, date_format='iso')

        logger.info(f"Successfully saved exported data to {file_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving exported data to {file_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during saving exported data: {e}")
        return False

# Loading is less relevant for internal flow now, might be for importing exported data
# def load_processed_data_from_export(file_path=settings.PROCESSED_EXPORT_PATH): ...