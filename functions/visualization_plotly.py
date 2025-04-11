# functions/visualization_plotly.py
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging
from config import settings # For default plot settings

logger = logging.getLogger(__name__)

def plot_workload_duration_plotly(workload_df: pd.DataFrame, limit=settings.PLOT_PERSONNEL_LIMIT):
    """Generates an interactive Plotly bar chart for total duration per personnel."""
    if workload_df is None or workload_df.empty:
        logger.warning("Workload data is empty. Skipping duration plot.")
        return go.Figure() # Return empty figure

    if 'total_duration_hours' not in workload_df.columns or 'personnel' not in workload_df.columns:
        logger.error("Workload DataFrame missing required columns for duration plot.")
        return go.Figure()

    # Sort and limit data for plotting
    plot_data = workload_df.sort_values('total_duration_hours', ascending=False).head(limit)

    fig = px.bar(plot_data,
                 y='personnel', # Plotly uses 'y' for categories on horizontal bars
                 x='total_duration_hours',
                 orientation='h', # Horizontal bar chart
                 title=f"{settings.PLOT_TITLE} - Total Duration (Top {limit})",
                 labels={'personnel': settings.PLOT_Y_LABEL,
                         'total_duration_hours': settings.PLOT_X_LABEL_HOURS},
                 color='total_duration_hours', # Color bars by duration
                 color_continuous_scale=px.colors.sequential.Viridis, # Color scale
                 text='total_duration_hours' # Display value on bar
                )

    fig.update_layout(
        yaxis={'categoryorder':'total ascending'}, # Keep the sorted order
        xaxis_title=settings.PLOT_X_LABEL_HOURS,
        yaxis_title=settings.PLOT_Y_LABEL,
        height=max(400, len(plot_data) * 35), # Adjust height dynamically
        coloraxis_showscale=False # Hide the color scale bar if desired
    )
    fig.update_traces(texttemplate='%{text:.1f}h', textposition='outside') # Format text labels

    return fig

def plot_workload_events_plotly(workload_df: pd.DataFrame, limit=settings.PLOT_PERSONNEL_LIMIT):
    """Generates an interactive Plotly bar chart for total events per personnel."""
    if workload_df is None or workload_df.empty:
        logger.warning("Workload data is empty. Skipping event count plot.")
        return go.Figure()

    if 'total_events' not in workload_df.columns or 'personnel' not in workload_df.columns:
         logger.error("Workload DataFrame missing required columns for event count plot.")
         return go.Figure()

    # Sort and limit data for plotting
    plot_data = workload_df.sort_values('total_events', ascending=False).head(limit)

    fig = px.bar(plot_data,
                 y='personnel',
                 x='total_events',
                 orientation='h',
                 title=f"{settings.PLOT_TITLE} - Total Events (Top {limit})",
                 labels={'personnel': settings.PLOT_Y_LABEL,
                         'total_events': settings.PLOT_X_LABEL_EVENTS},
                 color='total_events',
                 color_continuous_scale=px.colors.sequential.Magma,
                 text='total_events'
                )

    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        xaxis_title=settings.PLOT_X_LABEL_EVENTS,
        yaxis_title=settings.PLOT_Y_LABEL,
        height=max(400, len(plot_data) * 35),
        coloraxis_showscale=False
    )
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')

    return fig

def plot_daily_hourly_heatmap(df_filtered: pd.DataFrame):
    """Generates a heatmap showing event counts by hour and day of the week."""
    if df_filtered is None or df_filtered.empty or 'start_time' not in df_filtered.columns:
        logger.warning("Cannot generate heatmap: Data missing or 'start_time' column absent.")
        return go.Figure()

    df_plot = df_filtered.copy()
    # Ensure start_time is datetime
    df_plot['start_time'] = pd.to_datetime(df_plot['start_time'])
    # Extract hour and day of week
    df_plot['hour'] = df_plot['start_time'].dt.hour
    df_plot['day_of_week'] = df_plot['start_time'].dt.day_name()

    # Order days of the week
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_plot['day_of_week'] = pd.Categorical(df_plot['day_of_week'], categories=day_order, ordered=True)

    # Create pivot table for heatmap
    heatmap_data = df_plot.pivot_table(index='day_of_week', columns='hour', values='uid', aggfunc='count', fill_value=0)
    # Ensure all hours 0-23 are present
    heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)

    fig = px.imshow(heatmap_data,
                    labels=dict(x="Hour of Day", y="Day of Week", color="Number of Events"),
                    x=heatmap_data.columns,
                    y=heatmap_data.index,
                    text_auto=True, # Show counts on cells
                    aspect="auto", # Adjust aspect ratio
                    color_continuous_scale="YlGnBu", # Choose a color scale
                    title="Event Frequency by Day and Hour"
                   )
    fig.update_xaxes(side="bottom", tickmode='linear', dtick=1) # Show all hours clearly
    fig.update_layout(
         height=500,
         xaxis_title="Hour of Day (0-23)",
         yaxis_title="Day of Week"
     )

    return fig

def create_personnel_heatmap(df):
    """
    Create a heatmap showing personnel workload by day of week and hour
    
    Args:
        df (pandas.DataFrame): DataFrame with event data including start_time
        
    Returns:
        plotly.graph_objects.Figure: Heatmap visualization
    """
    import plotly.express as px
    import pandas as pd
    
    # Extract day of week and hour
    df = df.copy()
    
    # Ensure the start_time is a datetime object
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
        df['start_time'] = pd.to_datetime(df['start_time'])
    
    df['day_of_week'] = df['start_time'].dt.day_name()
    df['hour'] = df['start_time'].dt.hour
    
    # Aggregate data
    heatmap_data = df.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
    
    # Create pivot table
    pivot_data = heatmap_data.pivot(index='day_of_week', columns='hour', values='count')
    
    # Fill NaN values with 0
    pivot_data = pivot_data.fillna(0)
    
    # Sort days of week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(days_order)
    
    # Create heatmap
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Hour of Day", y="Day of Week", color="Event Count"),
        x=[f"{h}:00" for h in range(24) if h in pivot_data.columns],
        y=[day for day in days_order if day in pivot_data.index],
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        title="Event Distribution by Day and Hour",
        height=500
    )
    
    return fig