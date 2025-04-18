"""
Enhanced analysis page for the admin dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta
from config import settings
from functions import db as db_manager
from functions import visualization_plotly, analysis_calculations

logger = logging.getLogger(__name__)

def show_analysis_page():
    """
    Display the enhanced analysis page for administrators.
    """
    st.title("📈 Enhanced Analysis")
    st.markdown("Advanced data analysis and visualization tools for administrators.")
    
    # Load data
    df_data = load_data_for_analysis()
    
    if df_data is None or df_data.empty:
        st.warning("No data available for analysis. Please process some calendar data first.")
        return
      # Enhanced analysis options
    st.sidebar.markdown("## Analysis Options")
    
    # Date range selection with presets
    with st.sidebar.expander("Date Range", expanded=True):
        # Get min and max dates from data
        if 'start_time' in df_data.columns:
            try:
                min_date = pd.to_datetime(df_data['start_time']).min().date()
                max_date = pd.to_datetime(df_data['start_time']).max().date()
            except:
                min_date = datetime.now().date() - timedelta(days=365)
                max_date = datetime.now().date()
        else:
            min_date = datetime.now().date() - timedelta(days=365)
            max_date = datetime.now().date()
        
        # Create preset date ranges
        current_year = datetime.now().year
        current_date = datetime.now().date()
        
        # Define preset date ranges
        presets = {
            "Custom Range": None,  # Will be handled separately
            "Last 30 Days": (current_date - timedelta(days=30), current_date),
            "Last 90 Days": (current_date - timedelta(days=90), current_date),
            "Year to Date": (datetime(current_year, 1, 1).date(), current_date),
            f"Q1 {current_year}": (datetime(current_year, 1, 1).date(), datetime(current_year, 3, 31).date()),
            f"Q2 {current_year}": (datetime(current_year, 4, 1).date(), datetime(current_year, 6, 30).date()),
            f"Q3 {current_year}": (datetime(current_year, 7, 1).date(), datetime(current_year, 9, 30).date()),
            f"Q4 {current_year}": (datetime(current_year, 10, 1).date(), datetime(current_year, 12, 31).date()),
            f"Full Year {current_year}": (datetime(current_year, 1, 1).date(), datetime(current_year, 12, 31).date()),
            f"Full Year {current_year-1}": (datetime(current_year-1, 1, 1).date(), datetime(current_year-1, 12, 31).date()),
        }
        
        # Preset selection dropdown
        selected_preset = st.selectbox(
            "Date Range Preset", 
            options=list(presets.keys()),
            index=0
        )
        
        # Handle date range selection based on preset
        if selected_preset == "Custom Range":
            # Default to last 30 days for custom range
            default_start = max_date - timedelta(days=30)
            default_end = max_date
            
            # Custom date range picker
            date_range = st.date_input(
                "Select Custom Date Range",
                value=(default_start, default_end),
                min_value=min_date,
                max_value=max_date
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = default_start, default_end
        else:
            # Use preset date range
            start_date, end_date = presets[selected_preset]
            # Show the selected date range
            st.info(f"Selected date range: {start_date.strftime('%b %d, %Y')} to {end_date.strftime('%b %d, %Y')}")
        
        # Filter data based on selected date range
        try:
            df_filtered = df_data[
                (pd.to_datetime(df_data['start_time']).dt.date >= start_date) & 
                (pd.to_datetime(df_data['start_time']).dt.date <= end_date)
            ]
            # Show count of filtered events
            st.caption(f"Showing {len(df_filtered)} events out of {len(df_data)} total events")
        except Exception as e:
            logger.error(f"Error filtering by date range: {e}")
            st.warning("Unable to filter by date range. Using all data.")
            df_filtered = df_data
    
    # Personnel selection
    logger.info(f"df_filtered columns: {df_filtered.columns.tolist()}")
    logger.info(f"Sample data from df_filtered: {df_filtered.head().to_dict()}")

    with st.sidebar.expander("Personnel", expanded=True):
        if 'assigned_personnel' in df_filtered.columns:
            logger.info("Using 'assigned_personnel' column for personnel selection.")
            personnel_options = sorted(df_filtered['assigned_personnel'].unique())
        elif 'extracted_personnel' in df_filtered.columns:
            logger.info("Using 'extracted_personnel' column for personnel selection.")
            # Extract unique personnel from the lists
            personnel_lists = df_filtered['extracted_personnel'].dropna()
            all_personnel = []
            for p_list in personnel_lists:
                if isinstance(p_list, list):
                    all_personnel.extend(p_list)
                elif isinstance(p_list, str) and p_list.startswith('[') and p_list.endswith(']'):
                    # Try to eval string representation of list
                    try:
                        p_list = eval(p_list)
                        all_personnel.extend(p_list)
                    except Exception as e:
                        logger.error(f"Error evaluating personnel list: {e}")
            personnel_options = sorted(set(all_personnel))
        else:
            logger.warning("No personnel columns found in the DataFrame.")
            personnel_options = []
        
        selected_personnel = st.multiselect(
            "Select Personnel",
            options=personnel_options,
            default=personnel_options if len(personnel_options) < 5 else []
        )
        
        if selected_personnel:
            # Filter data by selected personnel
            if 'assigned_personnel' in df_filtered.columns:
                df_filtered = df_filtered[df_filtered['assigned_personnel'].isin(selected_personnel)]
            elif 'extracted_personnel' in df_filtered.columns:
                # Filter based on extracted_personnel lists
                mask = df_filtered['extracted_personnel'].apply(
                    lambda x: isinstance(x, list) and any(p in x for p in selected_personnel)
                )
                df_filtered = df_filtered[mask]
    
    # Event type selection with exclusion
    with st.sidebar.expander("Event Types", expanded=True):
        all_event_types = load_event_types_from_db()
        if all_event_types:
            excluded_event_types = st.multiselect(
                "Exclude Event Types",
                options=all_event_types,
                default=[et for et in all_event_types if "Unity" in et]
            )

            # Filter data by excluding selected event types
            if 'extracted_event_type' in df_filtered.columns:
                df_filtered = df_filtered[~df_filtered['extracted_event_type'].isin(excluded_event_types)]
        else:
            st.warning("No event types found in the database.")
    
    # Export options
    with st.sidebar.expander("Export", expanded=False):
        if st.button("Export Data (CSV)"):
            # Convert to CSV for download
            csv = df_filtered.to_csv(index=False)
            
            # Create download button
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"calendar_analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if st.button("Export Charts (PNG)"):
            st.info("Chart export functionality will be available in a future update")
    
    # Display analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Event Analysis", "Personnel Workload", "Advanced Metrics"])
    
    with tab1:
        show_analysis_overview(df_filtered)
    
    with tab2:
        show_event_analysis(df_filtered)
    
    with tab3:
        show_personnel_workload(df_filtered)
    
    with tab4:
        show_advanced_metrics(df_filtered)

def load_data_for_analysis():
    """
    Load data from the database or session state for analysis.
    """
    try:
        if settings.DB_ENABLED:
            # Load from database
            conn = db_manager.get_db_connection()
            if conn:
                query = f"""
                SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA}
                ORDER BY start_time DESC
                """
                df = pd.read_sql(query, conn)
                conn.close()
                
                if not df.empty:
                    # Extract personnel data from raw_data if available
                    try:
                        # First, check if raw_data contains extracted_personnel
                        if 'raw_data' in df.columns:
                            logger.info("Checking raw_data for extracted_personnel information")
                            df['extracted_personnel'] = df['raw_data'].apply(
                                lambda x: eval(x.get('extracted_personnel', '[]')) if isinstance(x, dict) else []
                            )
                        # If that fails or doesn't exist, use the personnel column if available
                        elif 'personnel' in df.columns and 'extracted_personnel' not in df.columns:
                            logger.info("Using personnel column as extracted_personnel")
                            # Convert single values to lists for consistency
                            df['extracted_personnel'] = df['personnel'].apply(lambda x: [x] if x else [])
                    except Exception as e:
                        logger.warning(f"Error extracting personnel data: {e}")
                        # Fallback to simple personnel column
                        if 'personnel' in df.columns and 'extracted_personnel' not in df.columns:
                            df['extracted_personnel'] = df['personnel'].apply(lambda x: [x] if x else [])
                    
                    # Ensure start_time and end_time are datetime
                    try:
                        df['start_time'] = pd.to_datetime(df['start_time'])
                        df['end_time'] = pd.to_datetime(df['end_time'])
                    except:
                        pass
                    
                    return df
        
        # If database not enabled or no data found, try session state
        if 'analysis_ready_df' in st.session_state and st.session_state.analysis_ready_df is not None:
            return st.session_state.analysis_ready_df
        
        # Try other dataframes in session state
        for df_name in ['normalized_df', 'llm_processed_df']:
            if df_name in st.session_state and st.session_state[df_name] is not None:
                df = st.session_state[df_name]
                
                # Check if we need to explode by personnel
                if 'extracted_personnel' in df.columns and 'assigned_personnel' not in df.columns:
                    try:
                        # Normalize the data for analysis
                        from functions.data_processor import explode_by_personnel
                        df = explode_by_personnel(df, personnel_col='extracted_personnel')
                        return df
                    except Exception as e:
                        logger.error(f"Error exploding by personnel: {e}")
                
                return df
    
    except Exception as e:
        logger.error(f"Error loading data for analysis: {e}")
        st.error(f"Error loading data: {e}")
    
    return None

def load_event_types_from_db():
    """
    Load all event types from the database and map them using EVENT_TYPE_MAPPING.
    """
    try:
        conn = db_manager.get_db_connection()
        if conn:
            logger.info("Database connection established successfully.")
            query = f"SELECT DISTINCT extracted_event_type FROM {settings.DB_TABLE_PROCESSED_DATA}"
            event_types_df = pd.read_sql(query, conn)
            conn.close()
            logger.info(f"Retrieved {len(event_types_df)} event types from the database.")
            # Map event types using EVENT_TYPE_MAPPING
            mapped_event_types = [settings.EVENT_TYPE_MAPPING.get(et.lower(), et) for et in event_types_df['extracted_event_type'].dropna()]
            return sorted(set(mapped_event_types))
        else:
            logger.error("Failed to establish database connection.")
    except Exception as e:
        logger.error(f"Error loading event types from database: {e}")
        return []

def show_analysis_overview(df):
    """
    Display an overview of the analyzed data.
    """
    st.subheader("Analysis Overview")
    
    # Key metrics for the selected data
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_events = len(df)
        st.metric("Total Events", total_events)
    
    with col2:
        if 'duration_minutes' in df.columns:
            total_hours = df['duration_minutes'].sum() / 60
            st.metric("Total Hours", f"{total_hours:.1f}")
        else:
            try:
                # Calculate duration from start and end time
                df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
                total_hours = df['duration_minutes'].sum() / 60
                st.metric("Total Hours", f"{total_hours:.1f}")
            except:
                st.metric("Total Hours", "N/A")
    
    with col3:
        unique_personnel = 0
        if 'assigned_personnel' in df.columns:
            unique_personnel = df['assigned_personnel'].nunique()
        elif 'extracted_personnel' in df.columns:
            # Count unique personnel across all lists
            all_personnel = []
            for p_list in df['extracted_personnel'].dropna():
                if isinstance(p_list, list):
                    all_personnel.extend(p_list)
            unique_personnel = len(set(all_personnel))
        st.metric("Unique Personnel", unique_personnel)
    
    # Overview charts
    if not df.empty:
        try:
            # Event distribution by event type
            if 'extracted_event_type' in df.columns:
                # Show event distribution by type
                # fig = visualization_plotly.plot_event_type_distribution(df) # Function does not exist
                # st.plotly_chart(fig, use_container_width=True) # Commented out as function is missing
                st.write("*(Event type distribution chart removed - function missing)*") # Placeholder message
            
            # Events by day of week
            try:
                if 'start_time' in df.columns:
                    df['day_of_week'] = df['start_time'].dt.day_name()
                    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    day_counts = df['day_of_week'].value_counts().reindex(days_order).fillna(0)
                    
                    fig = px.bar(
                        x=day_counts.index, 
                        y=day_counts.values,
                        labels={'x': 'Day of Week', 'y': 'Number of Events'},
                        title='Events by Day of Week',
                        color=day_counts.values,
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(xaxis_title="Day of Week", yaxis_title="Number of Events")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                logger.error(f"Error creating day of week chart: {e}")
        
        except Exception as e:
            logger.error(f"Error creating overview charts: {e}")
            st.error("Error creating charts. See logs for details.")

def show_event_analysis(df):
    """
    Display event analysis charts and metrics.
    """
    st.subheader("Event Analysis")
    
    if df.empty:
        st.warning("No data available for event analysis.")
        return
    
    # Event type distribution
    if 'extracted_event_type' in df.columns:
        # Get top event types
        event_counts = df['extracted_event_type'].value_counts().head(10)
        
        fig = px.pie(
            values=event_counts.values,
            names=event_counts.index,
            title='Top Event Types',
            hole=0.4
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Events over time
    if 'start_time' in df.columns:
        try:
            # Group by date
            df['date'] = df['start_time'].dt.date
            date_counts = df.groupby('date').size().reset_index(name='count')
            
            fig = px.line(
                date_counts,
                x='date',
                y='count',
                title='Events Over Time',
                markers=True
            )
            
            fig.update_layout(xaxis_title="Date", yaxis_title="Number of Events")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating events over time chart: {e}")
    
    # Event duration distribution
    if 'duration_minutes' in df.columns:
        try:
            # Create duration bins
            duration_hours = df['duration_minutes'] / 60
            
            fig = px.histogram(
                duration_hours,
                title='Event Duration Distribution',
                labels={'value': 'Duration (hours)'},
                nbins=20
            )
            
            fig.update_layout(xaxis_title="Duration (hours)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating duration distribution chart: {e}")

def show_personnel_workload(df):
    """
    Display personnel workload analysis.
    """
    st.subheader("Personnel Workload Analysis")
    
    if df.empty:
        st.warning("No data available for personnel workload analysis.")
        return
    
    # Personnel workload data preparation
    if 'assigned_personnel' in df.columns:
        personnel_col = 'assigned_personnel'
    elif 'extracted_personnel' in df.columns:
        personnel_col = 'extracted_personnel'
    else:
        st.warning("No personnel data found for workload analysis.")
        return
    
    if personnel_col == 'extracted_personnel' and df[personnel_col].apply(lambda x: isinstance(x, list)).any():
        # For list-type personnel data, we need to explode the DataFrame
        try:
            # Get unique personnel
            all_personnel = []
            for p_list in df[personnel_col].dropna():
                if isinstance(p_list, list):
                    all_personnel.extend(p_list)
            unique_personnel = sorted(set(all_personnel))
            
            # Show workload for each person
            workload_data = []
            
            for person in unique_personnel:
                # Filter events where this person is mentioned
                person_events = df[df[personnel_col].apply(lambda x: isinstance(x, list) and person in x)]
                total_events = len(person_events)
                
                if 'duration_minutes' in person_events.columns:
                    total_hours = person_events['duration_minutes'].sum() / 60
                else:
                    total_hours = person_events.shape[0]  # Count events if duration not available
                
                workload_data.append({
                    'Personnel': person,
                    'Events': total_events,
                    'Hours': total_hours
                })
            
            workload_df = pd.DataFrame(workload_data)
            workload_df = workload_df.sort_values('Hours', ascending=False)
            
            # Show chart for top personnel
            if not workload_df.empty:
                top_personnel = workload_df.head(10)
                
                fig = px.bar(
                    top_personnel,
                    x='Personnel',
                    y='Hours',
                    title='Top Personnel by Hours',
                    color='Hours',
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(xaxis_title="Personnel", yaxis_title="Total Hours")
                st.plotly_chart(fig, use_container_width=True)
                
                # Show workload data table
                st.subheader("Personnel Workload Details")
                st.dataframe(workload_df, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating personnel workload analysis: {e}")
            st.error(f"Error in workload analysis: {e}")
    else:
        # For single-value personnel column
        try:
            personnel_workload = df.groupby(personnel_col).agg({
                'id': 'count',
                'duration_minutes': 'sum' if 'duration_minutes' in df.columns else 'count'
            }).reset_index()
            
            personnel_workload.columns = [personnel_col, 'Events', 'Hours' if 'duration_minutes' in df.columns else 'Events']
            
            if 'duration_minutes' in df.columns:
                personnel_workload['Hours'] = personnel_workload['Hours'] / 60
            
            personnel_workload = personnel_workload.sort_values('Hours' if 'duration_minutes' in df.columns else 'Events', ascending=False)
            
            # Show chart for top personnel
            top_personnel = personnel_workload.head(10)
            
            fig = px.bar(
                top_personnel,
                x=personnel_col,
                y='Hours' if 'duration_minutes' in df.columns else 'Events',
                title=f'Top Personnel by {"Hours" if "duration_minutes" in df.columns else "Events"}',
                color='Hours' if 'duration_minutes' in df.columns else 'Events',
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(xaxis_title="Personnel", yaxis_title=f"Total {'Hours' if 'duration_minutes' in df.columns else 'Events'}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show workload data table
            st.subheader("Personnel Workload Details")
            st.dataframe(personnel_workload, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating personnel workload analysis: {e}")
            st.error(f"Error in workload analysis: {e}")

def show_advanced_metrics(df):
    """
    Display advanced analytics metrics.
    """
    st.subheader("Advanced Metrics")
    
    if df.empty:
        st.warning("No data available for advanced metrics.")
        return
    
    # Advanced metrics options
    metric_type = st.selectbox(
        "Select Metric Type",
        options=["Workload Distribution", "Time Allocation", "Event Patterns"]
    )
    
    if metric_type == "Workload Distribution":
        show_workload_distribution(df)
    elif metric_type == "Time Allocation":
        show_time_allocation(df)
    elif metric_type == "Event Patterns":
        show_event_patterns(df)

def show_workload_distribution(df):
    """
    Show workload distribution metrics.
    """
    st.write("Workload distribution analysis helps identify imbalances in work allocation.")
    
    # Workload distribution metrics
    if 'extracted_personnel' in df.columns or 'assigned_personnel' in df.columns:
        try:
            if 'extracted_personnel' in df.columns and df['extracted_personnel'].apply(lambda x: isinstance(x, list)).any():
                personnel_col = 'extracted_personnel'
                
                # Get unique personnel
                all_personnel = []
                for p_list in df[personnel_col].dropna():
                    if isinstance(p_list, list):
                        all_personnel.extend(p_list)
                unique_personnel = sorted(set(all_personnel))
                
                # Calculate workload distribution
                workload_data = []
                
                for person in unique_personnel:
                    # Filter events where this person is mentioned
                    person_events = df[df[personnel_col].apply(lambda x: isinstance(x, list) and person in x)]
                    
                    if 'extracted_event_type' in person_events.columns:
                        # Group by event type
                        event_type_counts = {}
                        for idx, row in person_events.iterrows():
                            event_type = row['extracted_event_type']
                            if event_type not in event_type_counts:
                                event_type_counts[event_type] = 0
                            event_type_counts[event_type] += 1
                        
                        # Add to workload data
                        for event_type, count in event_type_counts.items():
                            workload_data.append({
                                'Personnel': person,
                                'Event Type': event_type,
                                'Count': count
                            })
                
                workload_df = pd.DataFrame(workload_data)
                
                if not workload_df.empty:
                    # Create heatmap of personnel vs event type
                    pivot_df = workload_df.pivot_table(
                        index='Personnel',
                        columns='Event Type',
                        values='Count',
                        aggfunc='sum',
                        fill_value=0
                    )
                    
                    # Select top personnel and event types to avoid overcrowding
                    top_personnel = pivot_df.sum(axis=1).sort_values(ascending=False).head(10).index
                    top_event_types = pivot_df.sum(axis=0).sort_values(ascending=False).head(10).index
                    
                    pivot_df = pivot_df.loc[top_personnel, top_event_types]
                    
                    fig = px.imshow(
                        pivot_df,
                        labels=dict(x="Event Type", y="Personnel", color="Event Count"),
                        title="Workload Distribution by Personnel and Event Type",
                        aspect="auto",
                        color_continuous_scale="Viridis"
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display fairness metrics (Commented out due to missing calculate_gini_coefficient)
                    # col1, col2 = st.columns(2)
                    
                    # with col1:
                    #     # Calculate Gini coefficient for workload
                    #     personnel_totals = workload_df.groupby('Personnel')['Count'].sum().sort_values(ascending=False)
                    #     gini = analysis_calculations.calculate_gini_coefficient(personnel_totals.values)
                        
                    #     st.metric(
                    #         "Workload Inequality (Gini)",
                    #         f"{gini:.2f}",
                    #         delta_color="inverse",
                    #         help="0 means perfect equality, 1 means complete inequality"
                    #     )
                    
                    # with col2:
                    #     # Calculate coefficient of variation (standard deviation / mean)
                    #     cv = personnel_totals.std() / personnel_totals.mean() if personnel_totals.mean() > 0 else 0
                        
                    #     st.metric(
                    #         "Workload Variation (CV)",
                    #         f"{cv:.2f}",
                    #         delta_color="inverse",
                    #         help="Lower is better - measures the spread of workload"
                    #     )
                    st.write("*(Fairness metrics removed - calculation function missing)*") # Placeholder
                    
                    # Show raw data
                    with st.expander("View Raw Workload Data"):
                        st.dataframe(pivot_df)
            
            else:
                # For single-value personnel column
                personnel_col = 'assigned_personnel' if 'assigned_personnel' in df.columns else 'extracted_personnel'
                
                if 'extracted_event_type' in df.columns:
                    # Create pivot table
                    pivot_df = pd.pivot_table(
                        df,
                        index=personnel_col,
                        columns='extracted_event_type',
                        values='id',
                        aggfunc='count',
                        fill_value=0
                    )
                    
                    # Select top personnel and event types to avoid overcrowding
                    top_personnel = pivot_df.sum(axis=1).sort_values(ascending=False).head(10).index
                    top_event_types = pivot_df.sum(axis=0).sort_values(ascending=False).head(10).index
                    
                    pivot_df = pivot_df.loc[top_personnel, top_event_types]
                    
                    fig = px.imshow(
                        pivot_df,
                        labels=dict(x="Event Type", y="Personnel", color="Event Count"),
                        title="Workload Distribution by Personnel and Event Type",
                        aspect="auto",
                        color_continuous_scale="Viridis"
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display fairness metrics (Commented out due to missing calculate_gini_coefficient)
                    # col1, col2 = st.columns(2)
                    
                    # with col1:
                    #     # Calculate Gini coefficient for workload
                    #     personnel_totals = df.groupby(personnel_col).size().sort_values(ascending=False)
                    #     gini = analysis_calculations.calculate_gini_coefficient(personnel_totals.values)
                        
                    #     st.metric(
                    #         "Workload Inequality (Gini)",
                    #         f"{gini:.2f}",
                    #         delta_color="inverse",
                    #         help="0 means perfect equality, 1 means complete inequality"
                    #     )
                    
                    # with col2:
                    #     # Calculate coefficient of variation (standard deviation / mean)
                    #     cv = personnel_totals.std() / personnel_totals.mean() if personnel_totals.mean() > 0 else 0
                        
                    #     st.metric(
                    #         "Workload Variation (CV)",
                    #         f"{cv:.2f}",
                    #         delta_color="inverse",
                    #         help="Lower is better - measures the spread of workload"
                    #     )
                    st.write("*(Fairness metrics removed - calculation function missing)*") # Placeholder
                    
                    # Show raw data
                    with st.expander("View Raw Workload Data"):
                        st.dataframe(pivot_df)
        
        except Exception as e:
            logger.error(f"Error creating workload distribution metrics: {e}")
            st.error(f"Error in workload distribution analysis: {e}")
    else:
        st.warning("Personnel data is required for workload distribution analysis.")

def show_time_allocation(df):
    """
    Show time allocation metrics.
    """
    st.write("Time allocation analysis helps understand how time is distributed across different activities.")
    
    # Time allocation metrics
    if 'start_time' in df.columns and 'extracted_event_type' in df.columns:
        try:
            # Add hour of day
            df['hour_of_day'] = df['start_time'].dt.hour
            
            # Create heatmap of hour of day vs day of week
            df['day_of_week'] = df['start_time'].dt.day_name()
            
            # Define day order
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            # Create pivot table
            pivot_df = pd.pivot_table(
                df,
                index='day_of_week',
                columns='hour_of_day',
                values='id',
                aggfunc='count',
                fill_value=0
            )
            
            # Reorder days
            pivot_df = pivot_df.reindex(day_order)
            
            # Create heatmap
            fig = px.imshow(
                pivot_df,
                labels=dict(x="Hour of Day", y="Day of Week", color="Event Count"),
                title="Event Distribution by Day and Hour",
                aspect="auto",
                color_continuous_scale="Viridis",
                x=list(range(24))  # Force x-axis to show all hours
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show time allocation by event type
            if 'duration_minutes' in df.columns:
                # Calculate time spent by event type
                time_by_event = df.groupby('extracted_event_type')['duration_minutes'].sum().reset_index()
                time_by_event['hours'] = time_by_event['duration_minutes'] / 60
                time_by_event = time_by_event.sort_values('hours', ascending=False)
                
                top_events = time_by_event.head(10)
                
                fig = px.pie(
                    top_events,
                    values='hours',
                    names='extracted_event_type',
                    title='Time Allocation by Event Type',
                    hole=0.4
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data
                with st.expander("View Time Allocation Data"):
                    st.dataframe(time_by_event)
        
        except Exception as e:
            logger.error(f"Error creating time allocation metrics: {e}")
            st.error(f"Error in time allocation analysis: {e}")
    else:
        st.warning("Event timing data is required for time allocation analysis.")

def show_event_patterns(df):
    """
    Show event pattern metrics.
    """
    st.write("Event pattern analysis helps identify recurring patterns and trends in calendar data.")
    
    # Event pattern metrics
    if 'start_time' in df.columns:
        try:
            # Calculate event frequency metrics
            df['month'] = df['start_time'].dt.month
            df['quarter'] = df['start_time'].dt.quarter
            df['year'] = df['start_time'].dt.year
            
            # Events by month
            events_by_month = df.groupby('month').size().reindex(range(1, 13), fill_value=0)
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            fig = px.bar(
                x=month_names,
                y=events_by_month.values,
                labels={'x': 'Month', 'y': 'Number of Events'},
                title='Events by Month',
                color=events_by_month.values,
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Check if we have enough date range for trend analysis
            if df['start_time'].dt.date.nunique() > 30:
                # Create daily event counts
                df['date'] = df['start_time'].dt.date
                daily_counts = df.groupby('date').size()
                
                # Calculate moving average
                window_size = 7  # 7-day moving average
                if len(daily_counts) > window_size:
                    moving_avg = daily_counts.rolling(window=window_size).mean()
                    
                    # Create trend chart
                    trend_df = pd.DataFrame({
                        'date': daily_counts.index,
                        'events': daily_counts.values,
                        'trend': moving_avg.values
                    }).dropna()
                    
                    fig = px.line(
                        trend_df,
                        x='date',
                        y=['events', 'trend'],
                        labels={'date': 'Date', 'value': 'Number of Events', 'variable': 'Series'},
                        title='Event Trend Analysis',
                        color_discrete_map={'events': 'lightblue', 'trend': 'darkblue'}
                    )
                    
                    fig.update_layout(legend_title_text='')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Pattern detection - find repeating event patterns
            if 'summary' in df.columns:
                # Find most common event summaries
                common_summaries = df['summary'].value_counts().head(5)
                
                st.subheader("Recurring Event Patterns")
                
                for summary, count in common_summaries.items():
                    st.write(f"**{summary}** - occurs {count} times")
                    
                    # Get events with this summary
                    recurring_events = df[df['summary'] == summary]
                    
                    # Show on which days they typically occur
                    if len(recurring_events) > 1:
                        day_counts = recurring_events['day_of_week'].value_counts()
                        
                        fig = px.bar(
                            x=day_counts.index,
                            y=day_counts.values,
                            labels={'x': 'Day of Week', 'y': 'Count'},
                            title=f"Typical Days for '{summary[:30]}...'",
                            color=day_counts.values,
                            color_continuous_scale='Viridis'
                        )
                        
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            logger.error(f"Error creating event pattern metrics: {e}")
            st.error(f"Error in event pattern analysis: {e}")
    else:
        st.warning("Event timing data is required for pattern analysis.")
