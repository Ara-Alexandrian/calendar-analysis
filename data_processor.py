# data_processor.py
"""
Functions for loading and preprocessing calendar data.
"""
import pandas as pd
import json
import re
import logging
from datetime import datetime

# Configure basic logging if not already configured elsewhere
# (It's good practice for modules to be able to log independently)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_datetime(dt_str):
    """Handles different datetime formats found in the JSON."""
    if not dt_str: return pd.NaT # Handle None or empty strings

    if isinstance(dt_str, str):
        # Try ISO 8601 format with timezone first (most robust)
        try:
            # Handles 'YYYY-MM-DD HH:MM:SS+ZZ:ZZ', 'YYYY-MM-DDTHH:MM:SSZ', etc.
            dt = pd.to_datetime(dt_str, errors='coerce', utc=True)
            if pd.notna(dt): return dt
        except Exception:
             # This format might be ambiguous, log a warning if specific parsing fails later
            pass

        # Handle formats like '20241016T180652Z' (common in iCal)
        try:
            dt = pd.to_datetime(dt_str, format='%Y%m%dT%H%M%SZ', errors='coerce', utc=True)
            if pd.notna(dt): return dt
        except (ValueError, TypeError): # Be more specific on expected errors
            pass # Try next format

        # Handle vDDDTypes format by extracting the datetime part
        # Example: "vDDDTypes(2023-06-15 19:14:00+00:00, Parameters({}))"
        # Regex improved to handle optional T separator and optional timezone Z
        match = re.search(r"(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)?)", dt_str)
        if match:
            extracted_str = match.group(1)
            try:
                # Parse the extracted string, trying common variations
                dt = pd.to_datetime(extracted_str, errors='coerce', utc=True)
                if pd.notna(dt): return dt
            except Exception as e:
                 # Log if extraction succeeded but parsing failed
                 logging.debug(f"Could not parse extracted datetime string '{extracted_str}' from '{dt_str}': {e}")
                 pass # Could not parse extracted string

    elif isinstance(dt_str, dict) and 'dt' in dt_str: # Handle potential nested structure
        # Recursively call parse_datetime on the nested value
        return parse_datetime(dt_str['dt'])

    # If all parsing attempts fail
    logging.warning(f"Could not parse datetime string: {dt_str}")
    return pd.NaT


def load_data(json_file_path: str) -> pd.DataFrame | None:
    """Loads calendar data from a JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            # Handle potential empty file or invalid JSON structure
            content = f.read()
            if not content:
                logging.error(f"Error: JSON file is empty: {json_file_path}")
                return None
            data = json.loads(content)
            if not isinstance(data, list): # Expecting a list of events
                 logging.error(f"Error: JSON file does not contain a list of events: {json_file_path}")
                 return None
        df = pd.DataFrame(data)
        logging.info(f"Successfully loaded data from {json_file_path}. Found {len(df)} events.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: JSON file not found at {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error: Could not decode JSON from {json_file_path}. Invalid JSON: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading the data from {json_file_path}: {e}")
        return None


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parses timestamps, calculates duration, and prepares data."""
    if df is None or df.empty:
        logging.warning("Input DataFrame is empty or None. Skipping preprocessing.")
        # Return an empty DataFrame with expected columns for consistency downstream
        return pd.DataFrame(columns=['uid', 'summary', 'start', 'end', 'description', 'location', 'status', 'categories', 'created', 'last_modified', 'dtstamp', 'start_time', 'end_time', 'duration_minutes'])

    logging.info(f"Starting preprocessing for {len(df)} events.")

    # Ensure required columns exist for processing time; others are optional
    required_cols = ['start', 'end', 'summary', 'uid']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Input DataFrame missing required columns for processing: {missing_cols}. Cannot proceed.")
        # Return an empty DataFrame if essential columns are missing
        return pd.DataFrame(columns=df.columns.tolist() + ['start_time', 'end_time', 'duration_minutes'])


    # 1. Parse Timestamps
    df['start_time'] = df['start'].apply(parse_datetime)
    df['end_time'] = df['end'].apply(parse_datetime)

    # Report parsing issues
    start_nulls = df['start_time'].isnull().sum()
    end_nulls = df['end_time'].isnull().sum()
    total_rows = len(df)
    if start_nulls > 0:
        logging.warning(f"Could not parse 'start' time for {start_nulls}/{total_rows} events.")
    if end_nulls > 0:
        logging.warning(f"Could not parse 'end' time for {end_nulls}/{total_rows} events.")

    # Drop rows where essential time parsing failed (start or end)
    original_count = len(df)
    df.dropna(subset=['start_time', 'end_time'], inplace=True)
    dropped_count = original_count - len(df)
    if dropped_count > 0:
        logging.info(f"Dropped {dropped_count} rows due to missing/unparseable start or end times.")
        if len(df) == 0:
            logging.error("All rows dropped due to time parsing errors. Check date formats in JSON.")
            return pd.DataFrame(columns=df.columns.tolist() + ['duration_minutes']) # Keep original columns + duration

    # Ensure timezone consistency (parse_datetime should handle this by converting to UTC)
    # Add checks just in case parsing logic changes or fails silently
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']) or df['start_time'].dt.tz is None:
         logging.warning("Start times are not timezone-aware UTC after parsing. This may indicate a problem.")
         # Attempt to force UTC localization if needed, though parse_datetime should prevent this state
         # df['start_time'] = df['start_time'].dt.tz_localize('UTC', ambiguous='raise', nonexistent='raise')

    if not pd.api.types.is_datetime64_any_dtype(df['end_time']) or df['end_time'].dt.tz is None:
         logging.warning("End times are not timezone-aware UTC after parsing. This may indicate a problem.")
         # df['end_time'] = df['end_time'].dt.tz_localize('UTC', ambiguous='raise', nonexistent='raise')


    # 2. Calculate Duration
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

    # Filter out negative or zero durations
    original_count = len(df)
    invalid_duration_mask = df['duration_minutes'] <= 0
    invalid_duration_count = invalid_duration_mask.sum()

    if invalid_duration_count > 0:
        logging.warning(f"Found {invalid_duration_count} events with non-positive duration (end <= start).")
        # Optionally log details of invalid events
        # logging.debug("Events with invalid duration:\n" + df[invalid_duration_mask][['uid', 'start_time', 'end_time', 'summary']].to_string())
        df = df[~invalid_duration_mask] # Keep rows where duration > 0
        logging.info(f"Dropped {invalid_duration_count} rows with non-positive duration.")
        if len(df) == 0:
            logging.error("All remaining rows dropped due to non-positive duration.")
            # Return empty df with expected columns
            return pd.DataFrame(columns=original_count.columns.tolist())


    logging.info(f"Preprocessing complete. {len(df)} events remaining for analysis.")
    return df


def explode_by_physicist(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explodes the DataFrame based on the 'assigned_physicists' list column.
    Renames the column to 'physicist'. Handles potential non-list entries gracefully.
    """
    if 'assigned_physicists' not in df.columns:
        logging.error("Cannot explode DataFrame: 'assigned_physicists' column not found.")
        # Add the column as empty lists to prevent downstream errors, or handle differently
        df['assigned_physicists'] = [[] for _ in range(len(df))] # Example: Add empty list
        # return df # Or return original df if preferred

    # Ensure the column contains lists, converting non-lists (like 'Unknown' string) if necessary
    def ensure_list(x):
        if isinstance(x, list):
            return x
        elif pd.isna(x):
            return ["Unknown"] # Or [] if preferred for NaNs
        else:
            return [x] # Wrap single items in a list

    df['assigned_physicists'] = df['assigned_physicists'].apply(ensure_list)

    try:
        df_exploded = df.explode('assigned_physicists', ignore_index=True) # ignore_index resets index
        df_exploded.rename(columns={'assigned_physicists': 'physicist'}, inplace=True)
        # Ensure 'physicist' column is string type after explode
        df_exploded['physicist'] = df_exploded['physicist'].astype(str)
        logging.info(f"DataFrame exploded by physicist. Row count changed from {len(df)} to {len(df_exploded)}.")
        return df_exploded
    except Exception as e:
         logging.error(f"Error during DataFrame explosion: {e}")
         # Return the original dataframe or an empty one depending on desired handling
         return df # Return original df to allow potential partial analysis