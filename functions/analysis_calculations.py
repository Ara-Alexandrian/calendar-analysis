# functions/analysis_calculations.py
import pandas as pd
import logging
from functions import config_manager # To get role, pct etc.

logger = logging.getLogger(__name__)

def filter_data(df: pd.DataFrame, start_date=None, end_date=None, selected_personnel=None, selected_roles=None):
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
        df_filtered = df_filtered[df_filtered['personnel'].isin(selected_personnel)]

    # 3. Filter by Role
    if selected_roles: # If list is not empty
        # Map personnel names to roles
        personnel_roles = {name: config_manager.get_role(name) for name in df_filtered['personnel'].unique()}
        df_filtered['role'] = df_filtered['personnel'].map(personnel_roles)
        df_filtered = df_filtered[df_filtered['role'].isin(selected_roles)]
        # Optionally drop the temporary 'role' column if not needed later
        # df_filtered = df_filtered.drop(columns=['role'])

    logger.info(f"Data filtered: {len(df_filtered)} rows remaining.")
    return df_filtered


def calculate_workload_summary(df_filtered: pd.DataFrame):
    """Calculates workload metrics (events, duration) grouped by personnel."""
    if df_filtered is None or df_filtered.empty:
        return pd.DataFrame(columns=['personnel', 'role', 'clinical_pct', 'total_events', 'total_duration_hours', 'avg_duration_hours'])

    if 'personnel' not in df_filtered.columns or 'duration_hours' not in df_filtered.columns:
         logger.error("Cannot calculate workload: Missing 'personnel' or 'duration_hours' column.")
         return pd.DataFrame(columns=['personnel', 'role', 'clinical_pct', 'total_events', 'total_duration_hours', 'avg_duration_hours'])

    # Group by personnel and aggregate
    workload = df_filtered.groupby('personnel').agg(
        total_events=('uid', 'count'), # Assuming 'uid' uniquely identifies events before explode
        total_duration_hours=('duration_hours', 'sum')
    ).reset_index()

    # Calculate average duration
    workload['avg_duration_hours'] = workload['total_duration_hours'] / workload['total_events']

    # Add role and clinical percentage from config
    workload['role'] = workload['personnel'].apply(config_manager.get_role)
    workload['clinical_pct'] = workload['personnel'].apply(config_manager.get_clinical_pct)

    # Reorder columns for clarity
    workload = workload[[
        'personnel', 'role', 'clinical_pct',
        'total_events', 'total_duration_hours', 'avg_duration_hours'
    ]]

    # Sort by total duration by default
    workload = workload.sort_values(by='total_duration_hours', ascending=False).reset_index(drop=True)

    return workload

def get_unknown_assignments(df_exploded: pd.DataFrame):
    """Extracts rows where personnel assignment is 'Unknown' or 'Unknown_Error'."""
    if df_exploded is None or df_exploded.empty or 'personnel' not in df_exploded.columns:
        return pd.DataFrame()

    unknown_mask = df_exploded['personnel'].isin(['Unknown', 'Unknown_Error'])
    return df_exploded[unknown_mask]