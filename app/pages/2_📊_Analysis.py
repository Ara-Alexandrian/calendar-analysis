# pages/2_📊_Analysis.py
import streamlit as st
import pandas as pd
import logging
import datetime
import time
from config import settings

# Ensure project root is in path
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functions import analysis_calculations as ac
from functions import visualization_plotly as viz
from functions import config_manager # To get personnel list, roles
from functions import db_manager # For real-time database queries
from functions import llm_extractor # For checking background processing status

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Analysis", layout="wide")

st.markdown("# 2. Analyze Processed Data")
st.markdown("Filter the processed data and view workload summaries and visualizations.")

# --- Real-time data loading options ---
if 'last_refresh_time' not in st.session_state:
    st.session_state.last_refresh_time = time.time()
    
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False
    
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 30  # Default refresh interval in seconds

# Function to load the most up-to-date data
def load_latest_data():
    """Loads the most up-to-date data from either session state or database"""
    data_source = st.session_state.get('data_source', 'session')
    
    # Check for background processing data first
    if st.session_state.get('llm_background_processing', False):
        batch_id = st.session_state.get('current_batch_id')
        
        if batch_id:
            if settings.DB_ENABLED:
                # Try to load the latest processed data from database
                try:
                    # Get data from the database for this batch
                    df = db_manager.get_processed_events_by_batch(batch_id)
                    if not df.empty:
                        logger.info(f"Loaded {len(df)} background processed records from database")
                        return df
                except Exception as e:
                    logger.error(f"Error loading background processed data from database: {e}")
            else:
                # If DB not enabled, check session state background data
                if hasattr(st.session_state, 'background_processed_data') and batch_id in st.session_state.background_processed_data:
                    background_data = st.session_state.background_processed_data[batch_id]
                    if 'analysis_df' in background_data and not background_data['analysis_df'].empty:
                        logger.info(f"Loaded {len(background_data['analysis_df'])} records from background processing session state")
                        return background_data['analysis_df']
    
    if data_source == 'database' and settings.DB_ENABLED:
        # Try to load from database
        try:
            # Get data from the last 90 days by default (can be filtered later)
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=90)
            df = db_manager.get_processed_events(start_date=start_date, end_date=end_date, limit=10000)
            if not df.empty:
                logger.info(f"Loaded {len(df)} records from database in real-time")
                return df
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            st.error("Could not load data from database. Using session data instead.")
    
    # Fall back to session state
    if 'analysis_ready_df' in st.session_state and st.session_state.analysis_ready_df is not None:
        return st.session_state.analysis_ready_df
    
    return None

# Sidebar controls for real-time updates
with st.sidebar:
    st.header("Real-time Data Options")
    
    # Data source selector
    data_source_options = ["Session Data", "Database (Real-time)"]
    selected_source = st.radio(
        "Data Source",
        data_source_options,
        index=0 if st.session_state.get('data_source', 'session') == 'session' else 1,
        help="Choose whether to use data from the current session or query the database in real-time"
    )
    st.session_state.data_source = 'session' if selected_source == "Session Data" else 'database'
    
    # Auto-refresh toggle
    st.session_state.auto_refresh = st.toggle(
        "Auto-refresh",
        value=st.session_state.auto_refresh,
        help="Automatically refresh data at the specified interval"
    )
    
    # Refresh interval slider (only shown when auto-refresh is enabled)
    if st.session_state.auto_refresh:
        st.session_state.refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=5,
            max_value=300,
            value=st.session_state.refresh_interval,
            step=5
        )
        
        # Display time until next refresh
        time_since_refresh = time.time() - st.session_state.last_refresh_time
        time_until_refresh = max(0, st.session_state.refresh_interval - time_since_refresh)
        st.caption(f"Next refresh in {int(time_until_refresh)} seconds")
    
    # Manual refresh button
    if st.button("Refresh Now", key="manual_refresh"):
        st.session_state.last_refresh_time = time.time()
        st.rerun()

# Auto-refresh logic
if st.session_state.auto_refresh:
    time_since_refresh = time.time() - st.session_state.last_refresh_time
    if time_since_refresh > st.session_state.refresh_interval:
        st.session_state.last_refresh_time = time.time()
        st.rerun()

# --- Load data ---
df_analysis = load_latest_data()

# --- Check if data is ready ---
if df_analysis is None or df_analysis.empty:
    st.warning("No processed data available for analysis. Please upload and process data on the 'Upload & Process' page first.", icon="⚠️")
    st.stop() # Stop execution of this page

logger.info(f"Analysis page loaded with {len(df_analysis)} rows.")

# --- Sidebar Filters ---
st.sidebar.header("Analysis Filters")

# Get filter options from data and config
all_personnel = sorted(df_analysis['personnel'].unique())
all_roles = sorted(list(set(config_manager.get_role(p) for p in all_personnel)))
min_date = df_analysis['start_time'].min().date() if not df_analysis.empty else datetime.date.today()
max_date = df_analysis['start_time'].max().date() if not df_analysis.empty else datetime.date.today()

# Date Range Filter with Presets
filter_preset = st.sidebar.selectbox(
    "Date Range Preset", 
    ["Custom", "Last 7 Days", "Last 30 Days", "Last Quarter", "Year to Date"]
)

today = datetime.datetime.now().date()

if filter_preset == "Custom":
    selected_start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date, key="start_date")
    selected_end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date, key="end_date")
else:
    # Logic to set date ranges based on presets
    if filter_preset == "Last 7 Days":
        selected_start_date = today - datetime.timedelta(days=7)
        selected_end_date = today
    elif filter_preset == "Last 30 Days":
        selected_start_date = today - datetime.timedelta(days=30)
        selected_end_date = today
    elif filter_preset == "Last Quarter":
        selected_start_date = today - datetime.timedelta(days=90)
        selected_end_date = today
    elif filter_preset == "Year to Date":
        selected_start_date = datetime.datetime(today.year, 1, 1).date()
        selected_end_date = today
    
    st.sidebar.info(f"Date Range: {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}")

# Personnel Filter with Search
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

# Apply Filters Button
apply_filters = st.sidebar.button("Apply Filters", key="apply_filters_btn")

# Show active filters summary
if selected_personnel:
    st.sidebar.caption(f"Showing events for {len(selected_personnel)} personnel")
else:
    st.sidebar.caption(f"Showing events for all {len(all_personnel)} personnel")

# --- Apply Filters ---
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

# --- Display Analysis Results with Tabs ---
st.markdown("---")
st.header("Analysis Results")

# Display background processing status if active
if st.session_state.get('llm_background_processing', False):
    st.info("📊 **Background LLM processing is active.** The data will refresh automatically as more events are processed.")
    
    # Get the current batch ID
    batch_id = st.session_state.get('current_batch_id')
    
    if batch_id:
        if settings.DB_ENABLED:
            # Get processing status from database
            status = db_manager.get_latest_processing_status(batch_id)
            
            # Create a progress bar for background processing
            if status['status'] == 'in_progress':
                st.progress(status['pct_complete']/100, text=f"Processing: {status['pct_complete']:.1f}% complete")
                st.caption(f"Processing details: {status['extracted']} extracted, {status['processing']} remaining")
                
                # Force refresh every 10 seconds during active processing
                st.session_state.auto_refresh = True
                st.session_state.refresh_interval = min(st.session_state.refresh_interval, 10)
                
            elif status['status'] == 'complete':
                st.success("✅ Background processing complete! All data is now available for analysis.")
                
                # When complete, refresh one more time to ensure we have the latest data
                if st.session_state.get('data_source') != 'database':
                    st.session_state.data_source = 'database'
                    st.rerun()
        else:
            # Check session state for background processing status when DB is disabled
            if hasattr(st.session_state, 'background_processed_data') and batch_id in st.session_state.background_processed_data:
                background_data = st.session_state.background_processed_data[batch_id]
                
                if background_data.get('status') == 'complete':
                    st.success("✅ Background processing complete! All data is now available for analysis.")
                else:
                    # Calculate progress if available
                    if 'progress' in background_data:
                        progress = background_data['progress']
                        st.progress(progress, text=f"Processing: {progress*100:.1f}% complete")
                    
                    if 'message' in background_data:
                        st.caption(background_data['message'])
                    
                    # Force refresh every 10 seconds during active processing
                    st.session_state.auto_refresh = True
                    st.session_state.refresh_interval = min(st.session_state.refresh_interval, 10)
            else:
                st.warning("Processing status not available. Auto-refresh enabled to check progress.")
                st.session_state.auto_refresh = True
                st.session_state.refresh_interval = 10

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

    # Create tabs for better organization
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Personnel Analysis", "Time Distribution", "Raw Data"])

    with tab1:
        st.subheader("Quick Insights")
        
        # Key metrics in columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Events", len(df_filtered))
            st.metric("Total Personnel", df_filtered['personnel'].nunique())
        with col2:
            avg_duration = df_filtered['duration_hours'].mean()
            st.metric("Average Duration (hrs)", f"{avg_duration:.2f}")
            
            # Find the busiest day
            busiest_day = df_filtered.groupby(df_filtered['start_time'].dt.date)['uid'].count().idxmax()
            st.metric("Busiest Day", busiest_day.strftime('%Y-%m-%d'))
        
        # Weekly distribution chart
        st.subheader("Weekly Event Distribution")
        try:
            # Create a new weekly distribution chart if available
            weekly_fig = viz.plot_daily_hourly_heatmap(df_filtered)
            st.plotly_chart(weekly_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate weekly distribution: {e}")
            logger.error(f"Weekly distribution error: {e}", exc_info=True)

    with tab2:
        st.subheader("Personnel Workload Analysis")
        
        if not workload_summary_df.empty:
            # Display workload visualization
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(viz.plot_workload_duration_plotly(workload_summary_df), use_container_width=True)
            with col2:
                st.plotly_chart(viz.plot_workload_events_plotly(workload_summary_df), use_container_width=True)
            
            # Show workload table
            st.subheader("Detailed Personnel Workload")
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
        else:
            st.info("No workload summary data generated for the selected filters (perhaps only 'Unknown' assignments remain).")

    with tab3:
        st.subheader("Time Distribution Analysis")
        
        # New personnel heatmap visualization
        st.subheader("Event Distribution by Day and Hour")
        try:
            heatmap_fig = viz.create_personnel_heatmap(df_filtered)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate day/hour heatmap: {e}")
            logger.error(f"Heatmap generation error: {e}", exc_info=True)
        
        # Add additional time distribution charts if available
        try:
            # Day of week distribution
            dow_fig = viz.plot_daily_hourly_heatmap(df_filtered)
            st.plotly_chart(dow_fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not generate time distribution charts: {e}")
            logger.error(f"Time distribution error: {e}", exc_info=True)

    with tab4:
        st.subheader("Raw Data")
        st.dataframe(df_filtered)
        
        # Export options
        export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
        
        if st.button("Export Data"):
            if export_format == "CSV":
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="calendar_analysis.csv",
                    mime="text/csv",
                    key="download_csv"
                )
            elif export_format == "Excel":
                # Use BytesIO for Excel export
                buffer = pd.io.excel.BytesIO()
                df_filtered.to_excel(buffer, index=False)
                st.download_button(
                    label="Download Excel",
                    data=buffer.getvalue(),
                    file_name="calendar_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_excel"
                )
            elif export_format == "JSON":
                json_data = df_filtered.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="calendar_analysis.json",
                    mime="application/json",
                    key="download_json"
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