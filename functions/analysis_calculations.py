# functions/analysis_calculations.py
import pandas as pd
import logging
from functions import config_manager # To get role, pct etc.

logger = logging.getLogger(__name__)

def filter_data(df: pd.DataFrame, start_date=None, end_date=None, selected_personnel=None, selected_roles=None, selected_event_types=None):
    """Filters the exploded DataFrame based on UI selections."""
    if df is None or df.empty:
        return pd.DataFrame()

    df_filtered = df.copy()

    # Ensure datetime columns are timezone-aware (UTC)
    if 'start_time' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['start_time']):
         if df_filtered['start_time'].dt.tz is None:
            df_filtered['start_time'] = df_filtered['start_time'].dt.tz_localize('UTC')
         else:
            df_filtered['start_time'] = df_filtered['start_time'].dt.tz_convert('UTC')
    else:
        logger.warning("Filtering skipped: 'start_time' column missing or not datetime.")
        return df # Return original if time filtering not possible

    # 1. Filter by Date Range
    if start_date:
        start_date_tz = pd.to_datetime(start_date).tz_localize('UTC') # Make tz aware
        df_filtered = df_filtered[df_filtered['start_time'] >= start_date_tz]
    if end_date:
        # End date is inclusive, so filter up to the end of that day
        end_date_tz = pd.to_datetime(end_date).tz_localize('UTC') + pd.Timedelta(days=1)
        df_filtered = df_filtered[df_filtered['start_time'] < end_date_tz]

    # 2. Filter by Personnel
    # Ensure 'personnel' column exists
    if 'personnel' not in df_filtered.columns:
         logger.warning("Filtering skipped: 'personnel' column missing.")
    elif selected_personnel: # If list is not empty
        df_filtered = df_filtered[df_filtered['personnel'].isin(selected_personnel)]    # 3. Filter by Role
    if selected_roles: # If list is not empty
        # Check if role column exists in the dataframe
        if 'role' in df_filtered.columns:
            # Use existing role column
            df_filtered = df_filtered[df_filtered['role'].isin(selected_roles)]
        else:
            # Map personnel names to roles if role column doesn't exist
            personnel_roles = {name: config_manager.get_role(name) for name in df_filtered['personnel'].unique()}
            df_filtered['role'] = df_filtered['personnel'].map(personnel_roles)
            df_filtered = df_filtered[df_filtered['role'].isin(selected_roles)]
            # Optionally drop the temporary 'role' column if not needed later
            # df_filtered = df_filtered.drop(columns=['role'])

    # 4. Filter by Event Type
    if selected_event_types: # If list is not empty
        if 'event_type' in df_filtered.columns:
            df_filtered = df_filtered[df_filtered['event_type'].isin(selected_event_types)]
        else:
            logger.warning("Filtering by event type skipped: 'event_type' column missing.")


    logger.info(f"Data filtered: {len(df_filtered)} rows remaining.")
    return df_filtered


def calculate_workload_summary(df_filtered: pd.DataFrame, group_by_event_type: bool = False):
    """
    Calculates workload metrics (events, duration) grouped by personnel,
    optionally also grouping by event_type.
    """
    required_cols = ['personnel', 'duration_hours', 'uid'] # uid needed for event count
    if group_by_event_type:
        required_cols.append('event_type')

    if df_filtered is None or df_filtered.empty:
        logger.warning("Input DataFrame is empty. Returning empty summary.")
        cols = ['personnel', 'role', 'clinical_pct', 'total_events', 'total_duration_hours', 'avg_duration_hours']
        if group_by_event_type:
            cols.insert(1, 'event_type') # Add event_type after personnel
        return pd.DataFrame(columns=cols)

    # Check for required columns
    missing_cols = [col for col in required_cols if col not in df_filtered.columns]
    if missing_cols:
         logger.error(f"Cannot calculate workload: Missing required columns: {missing_cols}")
         cols = ['personnel', 'role', 'clinical_pct', 'total_events', 'total_duration_hours', 'avg_duration_hours']
         if group_by_event_type:
             cols.insert(1, 'event_type')
         return pd.DataFrame(columns=cols)

    # Define grouping columns
    grouping_cols = ['personnel']
    if group_by_event_type:
        grouping_cols.append('event_type')

    # Group and aggregate
    workload = df_filtered.groupby(grouping_cols, observed=True).agg( # observed=True is good practice
        total_events=('uid', 'nunique'), # Count unique event IDs within the group
        total_duration_hours=('duration_hours', 'sum')
    ).reset_index()

    # Calculate average duration safely, handling potential division by zero
    workload['avg_duration_hours'] = workload.apply(
        lambda row: row['total_duration_hours'] / row['total_events'] if row['total_events'] > 0 else 0,
        axis=1
    )

    # Add role and clinical percentage from config
    workload['role'] = workload['personnel'].apply(config_manager.get_role)
    # Provide a default value (e.g., 0) if clinical_pct is None or NaN
    workload['clinical_pct'] = workload['personnel'].apply(
        lambda p: config_manager.get_clinical_pct(p) if pd.notna(config_manager.get_clinical_pct(p)) else 0
    )

    # Reorder columns for clarity
    final_cols = ['personnel']
    if group_by_event_type:
        final_cols.append('event_type')
    final_cols.extend(['role', 'clinical_pct', 'total_events', 'total_duration_hours', 'avg_duration_hours'])
    workload = workload[final_cols]

    # Sort by personnel, then event type (if applicable), then duration
    sort_cols = ['personnel']
    ascending_list = [True] # Start with ascending for personnel
    if group_by_event_type:
        sort_cols.append('event_type')
        ascending_list.append(True) # Ascending for event type
    sort_cols.append('total_duration_hours')
    ascending_list.append(False) # Descending for duration

    # Ensure the lengths match before sorting
    if len(sort_cols) == len(ascending_list):
        workload = workload.sort_values(by=sort_cols, ascending=ascending_list).reset_index(drop=True)
    else:
        # Fallback sort if lengths mismatch (shouldn't happen with this logic)
        logger.error(f"Mismatch between sort columns ({len(sort_cols)}) and ascending list ({len(ascending_list)}). Falling back to default sort.")
        workload = workload.sort_values(by='total_duration_hours', ascending=False).reset_index(drop=True)


    return workload

def get_unknown_assignments(df_exploded: pd.DataFrame):
    """Extracts rows where personnel assignment is 'Unknown' or 'Unknown_Error'."""
    if df_exploded is None or df_exploded.empty or 'personnel' not in df_exploded.columns:
        return pd.DataFrame()

    unknown_mask = df_exploded['personnel'].isin(['Unknown', 'Unknown_Error'])
    return df_exploded[unknown_mask]
