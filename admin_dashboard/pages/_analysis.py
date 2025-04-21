"""
Enhanced analysis page for the admin dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import logging
from datetime import datetime, timedelta
import json # Import json
from config import settings
from functions import db as db_manager
from functions import visualization_plotly, analysis_calculations
from functions.data_processor import explode_by_personnel # Import the explosion function

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
            f"Q1 {current_year-2}": (datetime(current_year-2, 1, 1).date(), datetime(current_year-2, 3, 31).date()),
            f"Q2 {current_year-2}": (datetime(current_year-2, 4, 1).date(), datetime(current_year-2, 6, 30).date()),
            f"Q3 {current_year-2}": (datetime(current_year-2, 7, 1).date(), datetime(current_year-2, 9, 30).date()),
            f"Q4 {current_year-2}": (datetime(current_year-2, 10, 1).date(), datetime(current_year-2, 12, 31).date()),
            f"Q1 {current_year-1}": (datetime(current_year-1, 1, 1).date(), datetime(current_year-1, 3, 31).date()),
            f"Q2 {current_year-1}": (datetime(current_year-1, 4, 1).date(), datetime(current_year-1, 6, 30).date()),
            f"Q3 {current_year-1}": (datetime(current_year-1, 7, 1).date(), datetime(current_year-1, 9, 30).date()),
            f"Q4 {current_year-1}": (datetime(current_year-1, 10, 1).date(), datetime(current_year-1, 12, 31).date()),
            f"Q1 {current_year}": (datetime(current_year, 1, 1).date(), datetime(current_year, 3, 31).date()),
            f"Q2 {current_year}": (datetime(current_year, 4, 1).date(), datetime(current_year, 6, 30).date()),
            f"Q3 {current_year}": (datetime(current_year, 7, 1).date(), datetime(current_year, 9, 30).date()),
            f"Q4 {current_year}": (datetime(current_year, 10, 1).date(), datetime(current_year, 12, 31).date()),
            f"Full Year {current_year}": (datetime(current_year, 1, 1).date(), datetime(current_year, 12, 31).date()),
            f"Full Year {current_year-1}": (datetime(current_year-1, 1, 1).date(), datetime(current_year-1, 12, 31).date()),
            f"Full Year {current_year-2}": (datetime(current_year-2, 1, 1).date(), datetime(current_year-2, 12, 31).date())
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
        # Use the comprehensive list of personnel stored in _all_personnel_options
        if '_all_personnel_options' in df_filtered.columns and df_filtered['_all_personnel_options'].iloc[0]:
             personnel_options = df_filtered['_all_personnel_options'].iloc[0]
             logger.info(f"Using _all_personnel_options for personnel selection. Found {len(personnel_options)} options.")
        elif 'assigned_personnel' in df_filtered.columns:
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
            logger.warning("No personnel columns or _all_personnel_options found in the DataFrame.")
            personnel_options = []

        selected_personnel = st.multiselect(
            "Select Personnel",
            options=personnel_options,
            default=[] # Removed default selection
        )

        # Add Role selection filter
        all_roles = sorted(df_filtered['role'].dropna().unique().tolist()) if 'role' in df_filtered.columns else []
        selected_roles = st.multiselect(
            "Select Roles",
            options=all_roles,
            default=[] # Removed default selection
        )

        if selected_personnel or selected_roles:
            # Start with the filtered data from date range
            df_combined_filter = df_filtered.copy()

            # Filter by selected personnel if any are selected
            if selected_personnel:
                if 'assigned_personnel' in df_combined_filter.columns:
                    df_combined_filter = df_combined_filter[df_combined_filter['assigned_personnel'].isin(selected_personnel)]
                elif 'extracted_personnel' in df_combined_filter.columns:
                    # Filter based on extracted_personnel lists
                    mask = df_combined_filter['extracted_personnel'].apply(
                        lambda x: isinstance(x, list) and any(p in x for p in selected_personnel)
                    )
                    df_combined_filter = df_combined_filter[mask]
                else:
                    # If no personnel column to filter on, and personnel were selected,
                    # this means no events will match.
                    df_combined_filter = df_combined_filter.head(0) # Return empty DataFrame


            # Filter by selected roles if any are selected
            if selected_roles and 'role' in df_combined_filter.columns:
                 df_combined_filter = df_combined_filter[df_combined_filter['role'].isin(selected_roles)]
            elif selected_roles and 'role' not in df_combined_filter.columns:
                 # If roles were selected but no 'role' column exists, filter everything out
                 df_combined_filter = df_combined_filter.head(0)


            # Update df_filtered to the combined filtered result
            df_filtered = df_combined_filter


    # Event type selection with exclusion
    with st.sidebar.expander("Event Types", expanded=True):
        all_event_types = load_event_types_from_db()
        if all_event_types:
            excluded_event_types = st.multiselect(
                "Exclude Event Types",
                options=all_event_types,
                default=[] # Removed default exclusion
            )

            # Filter data by excluding selected event types
            if 'extracted_event_type' in df_filtered.columns:
                df_filtered = df_filtered[~df_filtered['extracted_event_type'].isin(excluded_event_types)]
        else:
            st.warning("No event types found in the database.")

    # --- Perform Explosion Once After Filtering ---
    personnel_col_for_explosion = None
    if 'assigned_personnel' in df_filtered.columns:
        personnel_col_for_explosion = 'assigned_personnel'
    elif 'extracted_personnel' in df_filtered.columns:
        personnel_col_for_explosion = 'extracted_personnel'

    df_exploded = df_filtered # Default to filtered if no explosion needed or possible

    if personnel_col_for_explosion:
        # Always attempt to explode if a personnel column is identified.
        # The explode_by_personnel function handles cases where the column
        # might not contain lists or is empty.
        logger.info(f"Exploding DataFrame based on '{personnel_col_for_explosion}'.")
        try:
            df_exploded = explode_by_personnel(df_filtered.copy(), personnel_col=personnel_col_for_explosion)
        except Exception as e:
            logger.error(f"Error during DataFrame explosion based on '{personnel_col_for_explosion}': {e}")
            st.error("An error occurred preparing data for analysis tabs. See logs.")
            # Fallback to filtered data if explosion fails
            df_exploded = df_filtered
            # Ensure a 'personnel' column exists even on fallback
            if 'personnel' not in df_exploded.columns:
                 df_exploded['personnel'] = 'Unknown'


    # Ensure 'personnel' column exists in the final df passed to tabs, default to 'Unknown' if necessary
    # This check is still needed in case personnel_col_for_explosion was None
    if 'personnel' not in df_exploded.columns:
        logger.warning("Adding default 'personnel' column as 'Unknown' before passing to tabs.")
        df_exploded['personnel'] = 'Unknown'

    # --- End Explosion Logic ---


    # Export options (Now uses df_exploded)
    with st.sidebar.expander("Export", expanded=False):
        if st.button("Export Data (CSV)"):
            # Convert to CSV for download
            csv = df_exploded.to_csv(index=False) # Use exploded data for export

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
        show_analysis_overview(df_exploded) # Pass exploded data

    with tab2:
        show_event_analysis(df_exploded) # Pass exploded data
    with tab3:
        show_personnel_workload(df_exploded) # Pass exploded data

    with tab4:
        show_advanced_metrics(df_exploded) # Pass exploded data

def load_data_for_analysis():
    """
    Load data from the database or session state for analysis.
    """
    try:
        df = None # Initialize df to None
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

        # If database not enabled or no data found, try session state
        if df is None and 'analysis_ready_df' in st.session_state and st.session_state.analysis_ready_df is not None:
            df = st.session_state.analysis_ready_df

        if df is None: # If still no data, try other dataframes in session state
             for df_name in ['normalized_df', 'llm_processed_df']:
                if df_name in st.session_state and st.session_state[df_name] is not None:
                    df = st.session_state[df_name]

                    # Check if we need to explode by personnel
                    if 'extracted_personnel' in df.columns and 'assigned_personnel' not in df.columns:
                        try:
                            # Normalize the data for analysis
                            from functions.data_processor import explode_by_personnel
                            df = explode_by_personnel(df, personnel_col='extracted_personnel')
                        except Exception as e:
                            logger.error(f"Error exploding by personnel: {e}")
                    break # Stop after finding the first available dataframe


        if df is not None and not df.empty:
            # After loading data, get all personnel from personnel_config.json
            try:
                personnel_config_path = settings.PERSONNEL_CONFIG_JSON_PATH
                with open(personnel_config_path, 'r') as f:
                    personnel_data = json.load(f)
                all_personnel_from_config = list(personnel_data.keys())
                logger.info(f"Loaded {len(all_personnel_from_config)} personnel from config.")

                # Combine personnel from data and config for a comprehensive list
                personnel_in_data = []
                if 'assigned_personnel' in df.columns:
                    personnel_in_data.extend(df['assigned_personnel'].dropna().unique().tolist())
                if 'extracted_personnel' in df.columns:
                     for p_list in df['extracted_personnel'].dropna():
                        if isinstance(p_list, list):
                            all_personnel_in_list = eval(p_list) if isinstance(p_list, str) else p_list
                            personnel_in_data.extend(all_personnel_in_list)


                combined_personnel_list = sorted(list(set(all_personnel_from_config + personnel_in_data)))
                df['_all_personnel_options'] = [combined_personnel_list] * len(df) # Store for later use

            except Exception as e:
                logger.warning(f"Could not load personnel from config or combine lists: {e}")
                # Fallback to just personnel in data if config loading fails
                personnel_in_data = []
                if 'assigned_personnel' in df.columns:
                    personnel_in_data.extend(df['assigned_personnel'].dropna().unique().tolist())
                if 'extracted_personnel' in df.columns:
                     for p_list in df['extracted_personnel'].dropna():
                        if isinstance(p_list, list):
                            all_personnel_in_list = eval(p_list) if isinstance(p_list, str) else p_list
                            personnel_in_data.extend(all_personnel_in_list)
                df['_all_personnel_options'] = [sorted(list(set(personnel_in_data)))] * len(df)

        return df # Return the dataframe even if empty or personnel loading failed


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

def show_event_analysis(df, selected_personnel=None):
    """
    Display event analysis charts and metrics.
    If personnel are selected, show individual analyses as well.
    
    Args:
        df: DataFrame containing event data
        selected_personnel: List of selected personnel names (optional)
    """
    st.subheader("Event Analysis")

    if df.empty:
        st.warning("No data available for event analysis.")
        return

    # Determine if we're working with multiple personnel
    has_personnel_selection = selected_personnel is not None and len(selected_personnel) > 0
    personnel_col_to_use = 'personnel'

    # --- OVERALL ANALYSIS ---
    
    # Only show the overall analysis header if we also have individual personnel selected
    if has_personnel_selection:
        st.markdown("### Overall Event Analysis")
        st.caption("Analysis for all events in the filtered data")
    
    # Event type distribution
    if 'extracted_event_type' in df.columns:
        # Get event type counts
        event_counts = df['extracted_event_type'].value_counts()

        fig = px.pie(
            values=event_counts.values,
            names=event_counts.index,
            title='Overall Event Type Distribution',
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
                title='Overall Events Over Time',
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
                title='Overall Event Duration Distribution',
                labels={'value': 'Duration (hours)'},
                nbins=20
            )

            fig.update_layout(xaxis_title="Duration (hours)", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error creating duration distribution chart: {e}")
    
    # --- SELECTED PERSONNEL ANALYSIS ---
    
    # If specific personnel are selected, add individual and combined analysis
    if has_personnel_selection:
        # First, add a combined analysis for all selected personnel together
        st.markdown("---")
        st.markdown(f"### Combined Analysis for Selected Personnel")
        st.caption(f"Analysis for: {', '.join(selected_personnel)}")
        
        # Filter for events matching any selected personnel
        df_selected = df[df[personnel_col_to_use].isin(selected_personnel)]
        
        if df_selected.empty:
            st.info(f"No events found for the selected personnel.")
        else:
            # Event type distribution for selected personnel combined
            if 'extracted_event_type' in df_selected.columns:
                event_counts = df_selected['extracted_event_type'].value_counts()
                
                fig = px.pie(
                    values=event_counts.values,
                    names=event_counts.index,
                    title=f'Event Type Distribution for Selected Personnel',
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Events over time for selected personnel combined
            if 'date' not in df_selected.columns and 'start_time' in df_selected.columns:
                df_selected['date'] = df_selected['start_time'].dt.date
            
            if 'date' in df_selected.columns:
                try:
                    date_counts = df_selected.groupby('date').size().reset_index(name='count')
                    
                    fig = px.line(
                        date_counts,
                        x='date',
                        y='count',
                        title='Events Over Time for Selected Personnel',
                        markers=True
                    )
                    fig.update_layout(xaxis_title="Date", yaxis_title="Number of Events")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating events over time chart for selected personnel: {e}")
            
            # Event duration distribution for selected personnel combined
            if 'duration_minutes' in df_selected.columns:
                try:
                    duration_hours = df_selected['duration_minutes'] / 60
                    
                    fig = px.histogram(
                        duration_hours,
                        title='Event Duration Distribution for Selected Personnel',
                        labels={'value': 'Duration (hours)'},
                        nbins=20
                    )
                    fig.update_layout(xaxis_title="Duration (hours)", yaxis_title="Count")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    logger.error(f"Error creating duration distribution chart for selected personnel: {e}")
        
        # Now add individual analysis for each personnel in expandable sections
        st.markdown("---")
        st.markdown("### Individual Personnel Analysis")
        
        for person in selected_personnel:
            with st.expander(f"Analysis for {person}", expanded=False):
                # Filter for just this person
                df_person = df[df[personnel_col_to_use] == person]
                
                if df_person.empty:
                    st.info(f"No events found for {person}.")
                    continue
                
                # Show summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Events", len(df_person))
                with col2:
                    if 'duration_minutes' in df_person.columns:
                        total_hours = df_person['duration_minutes'].sum() / 60
                        st.metric("Total Hours", f"{total_hours:.1f}")
                  # Event type distribution for individual
                if 'extracted_event_type' in df_person.columns:
                    # Debug: Print event types for A. Alexandrian to identify issues
                    if person == 'A. Alexandrian':
                        st.write("### Debug: Original Event Types for A. Alexandrian")
                        event_types_debug = df_person['extracted_event_type'].value_counts()
                        st.write(event_types_debug)
                    
                    event_counts = df_person['extracted_event_type'].value_counts()
                    
                    fig = px.pie(
                        values=event_counts.values,
                        names=event_counts.index,
                        title=f'Event Type Distribution for {person}',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Events over time for individual
                if 'date' not in df_person.columns and 'start_time' in df_person.columns:
                    df_person['date'] = df_person['start_time'].dt.date
                
                if 'date' in df_person.columns:
                    try:
                        date_counts = df_person.groupby('date').size().reset_index(name='count')
                        
                        fig = px.line(
                            date_counts,
                            x='date',
                            y='count',
                            title=f'Events Over Time for {person}',
                            markers=True
                        )
                        fig.update_layout(xaxis_title="Date", yaxis_title="Number of Events")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error creating events over time chart for {person}: {e}")
                
                # Event duration distribution for individual
                if 'duration_minutes' in df_person.columns and len(df_person) > 1:
                    try:
                        duration_hours = df_person['duration_minutes'] / 60
                        
                        fig = px.histogram(
                            duration_hours,
                            title=f'Event Duration Distribution for {person}',
                            labels={'value': 'Duration (hours)'},
                            nbins=min(20, len(df_person))
                        )
                        fig.update_layout(xaxis_title="Duration (hours)", yaxis_title="Count")
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        logger.error(f"Error creating duration distribution chart for {person}: {e}")

def show_personnel_workload(df, selected_personnel=None):
    """
    Display personnel workload analysis.
    
    Args:
        df: DataFrame containing event data
        selected_personnel: List of selected personnel names (optional)
    """
    st.subheader("Personnel Workload Analysis")

    if df.empty:
        st.warning("No data available for personnel workload analysis.")
        return

    # Assume input df is already exploded and has a 'personnel' column
    personnel_col_to_use = 'personnel'

    if personnel_col_to_use not in df.columns:
        st.warning(f"Expected 'personnel' column not found in the input data for workload analysis.")
        return

    try:
        # Get unique personnel from the 'personnel' column
        unique_personnel = sorted(df[personnel_col_to_use].dropna().unique())

        # Show workload for each person
        workload_data = []

        for person in unique_personnel:
            # Filter events where this person is mentioned in the final personnel column
            person_events = df[df[personnel_col_to_use] == person]
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
            top_personnel = workload_df.head(settings.PLOT_PERSONNEL_LIMIT) # Use the setting for limit

            fig = px.bar(
                top_personnel,
                x='Personnel',
                y='Hours',
                title=f'Top {settings.PLOT_PERSONNEL_LIMIT} Personnel by Hours',
                color='Hours',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show chart for top personnel by event count
            top_personnel_events = workload_df.sort_values('Events', ascending=False).head(settings.PLOT_PERSONNEL_LIMIT)

            fig_events = px.bar(
                top_personnel_events,
                x='Personnel',
                y='Events',
                title=f'Top {settings.PLOT_PERSONNEL_LIMIT} Personnel by Event Count',
                color='Events',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig_events, use_container_width=True)

        # --- Add Event Type Distribution for Selected Personnel ---
        if selected_personnel: # Only show if personnel are selected in the sidebar
            st.subheader("Event Type Distribution for Selected Personnel")

            # Filter the exploded data for the selected personnel
            df_selected_personnel = df[df[personnel_col_to_use].isin(selected_personnel)]

            if not df_selected_personnel.empty and 'extracted_event_type' in df_selected_personnel.columns:
                event_counts_selected = df_selected_personnel['extracted_event_type'].value_counts()

                fig_personnel_events = px.pie(
                    values=event_counts_selected.values,
                    names=event_counts_selected.index,
                    title=f'Event Type Distribution for {", ".join(selected_personnel)}',
                    hole=0.4
                )
                st.plotly_chart(fig_personnel_events, use_container_width=True)
            elif selected_personnel:
                 st.info(f"No events found for the selected personnel: {', '.join(selected_personnel)}")
        # --- End Added Section ---


    except Exception as e:
        logger.error(f"Error calculating or displaying workload: {e}")
        st.error("Error calculating workload. See logs for details.")


def show_advanced_metrics(df):
    """
    Display advanced analysis metrics and charts.
    """
    st.subheader("Advanced Metrics")

    if df.empty:
        st.warning("No data available for advanced metrics.")
        return

    metric_type = st.selectbox(
        "Select Metric Type",
        ["Workload Distribution", "Time Allocation", "Event Patterns"]
    )

    if metric_type == "Workload Distribution":
        show_workload_distribution(df)
    elif metric_type == "Time Allocation":
        show_time_allocation(df)
    elif metric_type == "Event Patterns":
        show_event_patterns(df)


def show_workload_distribution(df):
    """
    Display workload distribution by personnel and time.
    """
    st.subheader("Workload Distribution")

    if df.empty:
        st.warning("No data available for workload distribution.")
        return

    # Assume input df is already exploded and has a 'personnel' column
    personnel_col_to_use = 'personnel'

    if personnel_col_to_use not in df.columns:
        st.warning(f"Expected 'personnel' column not found in the input data for workload distribution.")
        return

    # Ensure duration_minutes and start_time exist
    if 'duration_minutes' not in df.columns or 'start_time' not in df.columns:
         try:
            # Calculate duration if missing (use input df)
            df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
         except Exception as e:
            logger.error(f"Error calculating duration_minutes in workload distribution: {e}")
            st.warning("Duration or start time data missing/invalid for workload distribution.")
            return

    # Aggregate data by personnel and day (use input df)
    try:
        df['date'] = df['start_time'].dt.date
        # Debugging: Check the type and content of the personnel column before groupby
        logger.info(f"Type of df['{personnel_col_to_use}'] before groupby in show_workload_distribution: {type(df[personnel_col_to_use])}")
        logger.info(f"Sample of df['{personnel_col_to_use}'] before groupby in show_workload_distribution: {df[personnel_col_to_use].head().tolist()}")

        # Group by personnel and date
        workload_by_day_personnel = df.groupby([df[personnel_col_to_use], df['date']])['duration_minutes'].sum().reset_index()
        workload_by_day_personnel.rename(columns={personnel_col_to_use: 'personnel'}, inplace=True) # Rename the personnel column after groupby
        workload_by_day_personnel['duration_hours'] = workload_by_day_personnel['duration_minutes'] / 60

        # Create a pivot table for the heatmap
        pivot_df = workload_by_day_personnel.pivot_table(
            index='date',
            columns='personnel', # Use 'personnel' directly
            values='duration_hours',
            fill_value=0
        )

        # Sort columns by total hours for better visualization
        sorted_personnel = pivot_df.sum().sort_values(ascending=False).index.tolist()
        pivot_df = pivot_df[sorted_personnel]

        # Create heatmap
        fig = px.imshow(
            pivot_df.T, # Transpose to have personnel on y-axis and date on x-axis
            labels=dict(x="Date", y="Personnel", color="Hours"),
            x=pivot_df.index,
            y=pivot_df.columns,
            title="Workload Heatmap (Hours per Day per Personnel)",
            aspect="auto",
            color_continuous_scale="Viridis"
        )

        fig.update_layout(xaxis_nticks=20) # Adjust number of date ticks
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error creating workload heatmap: {e}")
        st.error("Error generating workload heatmap. See logs for details.")

    # Alternative view: Events per day per personnel (use input df)
    try:
        # Ensure date column exists (might have been added above)
        if 'date' not in df.columns:
             df['date'] = df['start_time'].dt.date

        events_by_day_personnel = df.groupby(['personnel', 'date']).size().reset_index(name='event_count') # Use 'personnel' directly

        # Create a pivot table for the heatmap
        pivot_df_events = events_by_day_personnel.pivot_table(
            index='date',
            columns='personnel', # Use 'personnel' directly
            values='event_count',
            fill_value=0
        )

        # Sort columns by total events
        sorted_personnel_events = pivot_df_events.sum().sort_values(ascending=False).index.tolist()
        pivot_df_events = pivot_df_events[sorted_personnel_events]


        # Create heatmap for events
        fig_events = px.imshow(
            pivot_df_events.T, # Transpose
            labels=dict(x="Date", y="Personnel", color="Event Count"),
            x=pivot_df_events.index,
            y=pivot_df_events.columns,
            title="Event Count Heatmap (Events per Day per Personnel)",
            aspect="auto",
            color_continuous_scale="Plasma"
        )

        fig_events.update_layout(xaxis_nticks=20)
        st.plotly_chart(fig_events, use_container_width=True)

    except Exception as e:
        logger.error(f"Error creating event count heatmap: {e}")
        st.error("Error generating event count heatmap. See logs for details.")


def show_time_allocation(df):
    """
    Display time allocation analysis by hour of day and event type.
    """
    st.subheader("Time Allocation")

    if df.empty:
        st.warning("No data available for time allocation analysis.")
        return

    # Ensure duration_minutes and start_time exist
    if 'duration_minutes' not in df.columns or 'start_time' not in df.columns:
         try:
            df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
         except:
            st.warning("Duration or start time data missing for time allocation.")
            return

    # Time allocation by hour of day
    try:
        df['hour_of_day'] = df['start_time'].dt.hour
        time_by_hour = df.groupby('hour_of_day')['duration_minutes'].sum().reset_index()
        time_by_hour['duration_hours'] = time_by_hour['duration_minutes'] / 60

        fig = px.bar(
            time_by_hour,
            x='hour_of_day',
            y='duration_hours',
            title='Total Hours by Hour of Day',
            labels={'hour_of_day': 'Hour of Day', 'duration_hours': 'Total Hours'},
            color='duration_hours',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(xaxis=dict(tickmode='linear', dtick=1)) # Ensure all hours are shown
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error creating time by hour chart: {e}")
        st.error("Error generating time by hour chart. See logs for details.")

    # Time allocation by event type (Top 10)
    try:
        time_by_event = df.groupby('extracted_event_type')['duration_minutes'].sum().reset_index()
        time_by_event = time_by_event.sort_values('duration_minutes', ascending=False).head(10)
        time_by_event['duration_hours'] = time_by_event['duration_minutes'] / 60

        fig = px.pie(
            time_by_event,
            values='duration_hours',
            names='extracted_event_type',
            title='Top 10 Event Types by Total Hours',
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error creating time by event type chart: {e}")
        st.error("Error generating time by event type chart. See logs for details.")


def show_event_patterns(df):
    """
    Display analysis of event patterns, like recurring events or trends.
    """
    st.subheader("Event Patterns")

    if df.empty:
        st.warning("No data available for event pattern analysis.")
        return

    # Ensure start_time exists
    if 'start_time' not in df.columns:
        st.warning("Start time data missing for event pattern analysis.")
        return

    # Monthly event trends
    try:
        df['month'] = df['start_time'].dt.to_period('M').astype(str)
        monthly_counts = df['month'].value_counts().sort_index().reset_index(name='count')
        monthly_counts.rename(columns={'index': 'month'}, inplace=True)

        fig = px.line(
            monthly_counts,
            x='month',
            y='count',
            title='Monthly Event Count Trend',
            labels={'month': 'Month', 'count': 'Number of Events'},
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        logger.error(f"Error creating monthly event trend chart: {e}")
        st.error("Error generating monthly event trend chart. See logs for details.")

    # Daily event patterns (e.g., recurring events)
    try:
        st.subheader("Daily Event Patterns")
        st.info("This section can be used to identify recurring events or daily patterns.")

        # Example: Find events that occur on the same day and time frequently
        # This is a simplified example; a real implementation might need more sophisticated logic
        if 'extracted_event_type' in df.columns:
            df['time_of_day'] = df['start_time'].dt.strftime('%H:%M')
            df['day_of_week'] = df['start_time'].dt.day_name()

            # Group by event type, day of week, and time of day to find recurring patterns
            recurring_patterns = df.groupby(['extracted_event_type', 'day_of_week', 'time_of_day']).size().reset_index(name='count')
            recurring_patterns = recurring_patterns.sort_values('count', ascending=False)

            st.write("Most frequent event patterns (Event Type, Day of Week, Time of Day):")
            st.dataframe(recurring_patterns.head(10)) # Show top 10 recurring patterns

            # You could add more visualizations here, e.g., heatmap of events by hour and day of week
            # Example: Heatmap of event counts by hour and day of week
            event_counts_heatmap = df.groupby(['day_of_week', df['start_time'].dt.hour]).size().unstack(fill_value=0)
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            event_counts_heatmap = event_counts_heatmap.reindex(days_order)

            fig_heatmap = px.imshow(
                event_counts_heatmap,
                labels=dict(x="Hour of Day", y="Day of Week", color="Number of Events"),
                x=event_counts_heatmap.columns,
                y=event_counts_heatmap.index,
                title="Event Count Heatmap by Day of Week and Hour of Day",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)


    except Exception as e:
        logger.error(f"Error analyzing daily event patterns: {e}")
        st.error("Error analyzing daily event patterns. See logs for details.")
