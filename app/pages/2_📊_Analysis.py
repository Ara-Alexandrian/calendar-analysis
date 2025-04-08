# pages/2_📊_Analysis.py
import streamlit as st
import pandas as pd
import logging
import datetime

# Ensure project root is in path
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functions import analysis_calculations as ac
from functions import visualization_plotly as viz
from functions import config_manager # To get personnel list, roles

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Analysis", layout="wide")

st.markdown("# 2. Analyze Processed Data")
st.markdown("Filter the processed data and view workload summaries and visualizations.")

# --- Check if data is ready ---
if 'analysis_ready_df' not in st.session_state or st.session_state.analysis_ready_df is None or st.session_state.analysis_ready_df.empty:
    st.warning("No processed data available for analysis. Please upload and process data on the 'Upload & Process' page first.", icon="⚠️")
    st.stop() # Stop execution of this page

df_analysis = st.session_state.analysis_ready_df
logger.info(f"Analysis page loaded with {len(df_analysis)} rows.")

# --- Sidebar Filters ---
st.sidebar.header("Analysis Filters")

# Get filter options from data and config
all_personnel = sorted(df_analysis['personnel'].unique())
all_roles = sorted(list(set(config_manager.get_role(p) for p in all_personnel)))
min_date = df_analysis['start_time'].min().date() if not df_analysis.empty else datetime.date.today()
max_date = df_analysis['start_time'].max().date() if not df_analysis.empty else datetime.date.today()

# Date Range Filter
selected_start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date, key="start_date")
selected_end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date, key="end_date")

# Personnel Filter
# Add 'All' option implicitly by leaving the list empty
default_personnel = [] # Default to all
selected_personnel = st.sidebar.multiselect(
    "Select Personnel (leave blank for all)",
    options=all_personnel,
    default=default_personnel,
    key="personnel_filter"
)

# Role Filter
default_roles = [] # Default to all
selected_roles = st.sidebar.multiselect(
    "Select Roles (leave blank for all)",
    options=all_roles,
    default=default_roles,
    key="role_filter"
)

# Apply Filters Button (Optional - could filter on change, but button is safer for large data)
apply_filters = st.sidebar.button("Apply Filters", key="apply_filters_btn")

# --- Apply Filters ---
# We can filter whenever the button is pressed OR always filter on widget change
# For simplicity here, let's filter on change (remove the button if preferred)
# Note: For very large data, filtering on change might feel sluggish.

# Always apply filters based on current widget state
try:
    df_filtered = ac.filter_data(
        df_analysis,
        start_date=selected_start_date,
        end_date=selected_end_date,
        selected_personnel=selected_personnel if selected_personnel else None, # Pass None if empty
        selected_roles=selected_roles if selected_roles else None
    )
    logger.info(f"Filters applied. {len(df_filtered)} rows remaining.")
except Exception as e:
    st.error(f"Error applying filters: {e}")
    logger.error(f"Filter error: {e}", exc_info=True)
    st.stop()

# --- Display Analysis Results ---
st.markdown("---")
st.header("Analysis Results")

if df_filtered.empty:
    st.warning("No data matches the selected filters.")
else:
    st.info(f"Displaying results for **{len(df_filtered)}** event assignments within the selected filters.")

    # Calculate Workload Summary
    try:
        workload_summary_df = ac.calculate_workload_summary(df_filtered)
        logger.info("Workload summary calculated.")
    except Exception as e:
        st.error(f"Error calculating workload summary: {e}")
        logger.error(f"Workload calculation error: {e}", exc_info=True)
        workload_summary_df = pd.DataFrame() # Ensure it's an empty df

    if not workload_summary_df.empty:
        st.subheader("Workload Summary Table")
        st.dataframe(workload_summary_df.style.format({
            'clinical_pct': '{:.1%}',
            'total_duration_hours': '{:.1f}',
            'avg_duration_hours': '{:.1f}'
        }))

        # Download Button for summary
        csv_summary = workload_summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
           label="Download Summary as CSV",
           data=csv_summary,
           file_name=f'workload_summary_{selected_start_date}_to_{selected_end_date}.csv',
           mime='text/csv',
           key="download_summary"
        )

        # Visualizations
        st.subheader("Workload Visualizations")
        col1, col2 = st.columns(2)

        with col1:
             st.plotly_chart(viz.plot_workload_duration_plotly(workload_summary_df), use_container_width=True)

        with col2:
             st.plotly_chart(viz.plot_workload_events_plotly(workload_summary_df), use_container_width=True)

        # Heatmap (optional, can be slow for very large datasets)
        st.subheader("Event Frequency Heatmap")
        try:
             heatmap_fig = viz.plot_daily_hourly_heatmap(df_filtered)
             st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception as e:
             st.error(f"Could not generate heatmap: {e}")
             logger.error(f"Heatmap generation error: {e}", exc_info=True)


    else:
        st.info("No workload summary data generated for the selected filters (perhaps only 'Unknown' assignments remain).")


    # Display Raw Filtered Data (optional, maybe behind expander)
    with st.expander("View Filtered Raw Data Details"):
         st.dataframe(df_filtered)
         # Download Button for filtered data
         csv_filtered = df_filtered.to_csv(index=False).encode('utf-8')
         st.download_button(
            label="Download Filtered Data as CSV",
            data=csv_filtered,
            file_name=f'filtered_data_{selected_start_date}_to_{selected_end_date}.csv',
            mime='text/csv',
            key="download_filtered"
         )

# Display Unknown Assignments Summary
st.markdown("---")
st.header("Unassigned Events")
unknown_df = ac.get_unknown_assignments(df_analysis) # Get unknowns from original analysis df

if not unknown_df.empty:
     st.warning(f"Found {len(unknown_df)} event assignments marked as 'Unknown' or 'Unknown_Error' in the full dataset.")
     with st.expander("Show Unassigned Event Summaries"):
          st.dataframe(unknown_df[['summary', 'start_time', 'personnel']].head(50)) # Show first 50
else:
     st.success("No 'Unknown' or 'Unknown_Error' assignments found in the dataset.")

logger.info("Analysis page processing complete.")