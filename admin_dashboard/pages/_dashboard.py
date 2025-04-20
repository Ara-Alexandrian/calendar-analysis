"""
Dashboard overview page for the admin dashboard.
Displays key statistics and system information.
"""
import streamlit as st
import pandas as pd
import datetime
import logging
import plotly.express as px
from config import settings
from functions import db as db_manager
from functions.llm_extraction.client import is_llm_ready

logger = logging.getLogger(__name__)

def show_dashboard_page():
    """
    Display the main dashboard overview page.
    """
    st.title("📊 Admin Dashboard")
    st.markdown("Welcome to the Calendar Analyzer administrative dashboard. Monitor system health and view key metrics.")
    
    # Get system health and statistics
    system_health = check_system_health()
    statistics = get_system_statistics()
    
    # Layout in two columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        show_key_metrics(statistics)
        show_recent_activity()
    
    with col2:
        show_system_health(system_health)
        show_quick_actions()

def check_system_health():
    """
    Check the health of various system components.
    """
    health = {
        "database": {
            "status": "unknown",
            "message": "Not checked"
        },
        "llm": {
            "status": "unknown",
            "message": "Not checked"
        },
        "file_system": {
            "status": "unknown",
            "message": "Not checked"
        }
    }
    
    # Check database connection
    if settings.DB_ENABLED:
        try:
            conn = db_manager.get_db_connection()
            if conn:
                health["database"]["status"] = "ok"
                health["database"]["message"] = "Connected successfully"
                conn.close()
            else:
                health["database"]["status"] = "error"
                health["database"]["message"] = "Failed to connect"
        except Exception as e:
            health["database"]["status"] = "error"
            health["database"]["message"] = f"Error: {str(e)[:50]}"
    else:
        health["database"]["status"] = "disabled"
        health["database"]["message"] = "Database is disabled in settings"
    
    # Check LLM connectivity
    try:
        llm_status = is_llm_ready()
        if llm_status:
            health["llm"]["status"] = "ok"
            health["llm"]["message"] = f"Connected to {settings.LLM_PROVIDER} ({settings.LLM_MODEL})"
        else:
            health["llm"]["status"] = "error"
            health["llm"]["message"] = "LLM service is not responding"
    except Exception as e:
        health["llm"]["status"] = "error"
        health["llm"]["message"] = f"Error: {str(e)[:50]}"
    
    # Check file system
    try:
        import os
        if os.path.exists(settings.OUTPUT_DIR) and os.access(settings.OUTPUT_DIR, os.W_OK):
            health["file_system"]["status"] = "ok"
            health["file_system"]["message"] = "Output directory is accessible"
        else:
            health["file_system"]["status"] = "error"
            health["file_system"]["message"] = "Output directory is not accessible"
    except Exception as e:
        health["file_system"]["status"] = "error"
        health["file_system"]["message"] = f"Error: {str(e)[:50]}"
    
    return health

def get_system_statistics():
    """
    Get system statistics and metrics.
    """
    statistics = {
        "total_events": 0,
        "events_processed": 0,
        "unknown_events": 0,
        "unknown_personnel": 0,
        "admin_edits": 0,
        "recent_uploads": 0
    }
    
    # Get statistics from database if enabled
    if settings.DB_ENABLED:
        try:
            conn = db_manager.get_db_connection()
            if conn:
                # Total events
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {settings.DB_TABLE_PROCESSED_DATA}")
                    statistics["total_events"] = cursor.fetchone()[0]
                
                # Events with valid processing
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE processing_status = 'processed'")
                    statistics["events_processed"] = cursor.fetchone()[0]
                
                # Events with unknown event type
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE extracted_event_type IN ('Unknown', 'Unknown_Error')")
                    statistics["unknown_events"] = cursor.fetchone()[0]
                  # Events with unknown personnel
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE personnel = 'Unknown' OR personnel = 'Unknown_Error'")
                    statistics["unknown_personnel"] = cursor.fetchone()[0]
                
                # Admin edited events
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE processing_status = 'manual_assigned'")
                    statistics["admin_edits"] = cursor.fetchone()[0]
                
                # Recent uploads (last 7 days)
                with conn.cursor() as cursor:
                    cursor.execute(f"SELECT COUNT(*) FROM {settings.DB_TABLE_CALENDAR_FILES} WHERE upload_date > NOW() - INTERVAL '7 days'")
                    statistics["recent_uploads"] = cursor.fetchone()[0]
                
                conn.close()
        except Exception as e:
            logger.error(f"Error fetching system statistics: {e}")
    
    # If DB is not enabled, try to get statistics from session state
    else:
        if 'normalized_df' in st.session_state and st.session_state.normalized_df is not None:
            df = st.session_state.normalized_df
            statistics["total_events"] = len(df)
            statistics["events_processed"] = len(df[df['processing_status'] == 'processed']) if 'processing_status' in df.columns else 0
            statistics["unknown_events"] = len(df[df['extracted_event_type'].isin(['Unknown', 'Unknown_Error'])]) if 'extracted_event_type' in df.columns else 0
            
            if 'extracted_personnel' in df.columns:
                unknown_personnel_mask = df['extracted_personnel'].apply(lambda x: isinstance(x, list) and ('Unknown' in x or 'Unknown_Error' in x))
                statistics["unknown_personnel"] = unknown_personnel_mask.sum()
    
    return statistics

def show_key_metrics(statistics):
    """
    Display key metrics in a visual dashboard.
    """
    st.subheader("Key Metrics")
    
    # Display metrics in a grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Events", statistics["total_events"])
        
    with col2:
        st.metric("Processed Events", statistics["events_processed"])
        
    with col3:
        st.metric("Admin Edits", statistics["admin_edits"])
    
    # Second row of metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Unknown Event Types", statistics["unknown_events"], 
                 delta=-statistics["admin_edits"] if statistics["admin_edits"] > 0 else None,
                 delta_color="inverse")
        
    with col2:
        st.metric("Unknown Personnel", statistics["unknown_personnel"],
                 delta=-statistics["admin_edits"] if statistics["admin_edits"] > 0 else None,
                 delta_color="inverse")
        
    with col3:
        st.metric("Recent Uploads", statistics["recent_uploads"])
    
    # Create a simple chart if data is available
    if statistics["total_events"] > 0:
        # Create sample data for chart
        chart_data = {
            "Category": ["Processed", "Unknown Event", "Unknown Personnel", "Admin Edited"],
            "Count": [
                statistics["events_processed"],
                statistics["unknown_events"],
                statistics["unknown_personnel"],
                statistics["admin_edits"]
            ]
        }
        df_chart = pd.DataFrame(chart_data)
        
        # Create a horizontal bar chart
        fig = px.bar(df_chart, x="Count", y="Category", orientation='h',
                     title="Event Processing Status",
                     color="Category",
                     color_discrete_map={
                         "Processed": "#28a745",
                         "Unknown Event": "#ffc107",
                         "Unknown Personnel": "#ffc107",
                         "Admin Edited": "#17a2b8"
                     })
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def show_system_health(health):
    """
    Display system health information.
    """
    st.subheader("System Health")
    
    # Create a status container
    for component, status in health.items():
        status_color = {
            "ok": "🟢",
            "warning": "🟠",
            "error": "🔴",
            "unknown": "⚪",
            "disabled": "⚫"
        }.get(status["status"], "⚪")
        
        st.markdown(f"{status_color} **{component.title()}**: {status['message']}")
    
    # Additional system information
    with st.expander("System Details", expanded=False):
        st.markdown("**Application Information**")
        st.markdown(f"- Project Root: `{settings.PROJECT_ROOT}`")
        st.markdown(f"- Config Directory: `{settings.CONFIG_DIR}`")
        st.markdown(f"- Output Directory: `{settings.OUTPUT_DIR}`")
        st.markdown(f"- Data Directory: `{settings.DATA_DIR}`")
        
        st.markdown("**LLM Configuration**")
        st.markdown(f"- Provider: `{settings.LLM_PROVIDER}`")
        st.markdown(f"- Model: `{settings.LLM_MODEL}`")
        st.markdown(f"- Base URL: `{settings.OLLAMA_BASE_URL}`")
        st.markdown(f"- Max Workers: `{settings.LLM_MAX_WORKERS}`")
        
        if settings.DB_ENABLED:
            st.markdown("**Database Configuration**")
            st.markdown(f"- Host: `{settings.DB_HOST}`")
            st.markdown(f"- Database: `{settings.DB_NAME}`")
            st.markdown(f"- User: `{settings.DB_USER}`")
            
        # Application runtime
        import psutil
        import os
        process = psutil.Process(os.getpid())
        st.markdown("**Runtime Information**")
        st.markdown(f"- Memory Usage: `{process.memory_info().rss / 1024 / 1024:.2f} MB`")
        st.markdown(f"- CPU Usage: `{process.cpu_percent()} %`")
        st.markdown(f"- Threads: `{process.num_threads()}`")
        st.markdown(f"- Process ID: `{process.pid}`")

def show_quick_actions():
    """
    Display quick action buttons for common admin tasks.
    """
    st.subheader("Quick Actions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restart LLM", use_container_width=True):
            try:
                from functions.llm_extraction.client import restart_ollama_server
                result = restart_ollama_server()
                if result:
                    st.success("LLM service restarted successfully")
                else:
                    st.error("Failed to restart LLM service")
            except Exception as e:
                st.error(f"Error restarting LLM service: {e}")
    
    with col2:
        if st.button("📊 Generate Reports", use_container_width=True):
            st.info("Report generation will be available in a future update")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✏️ Manual Assignment", use_container_width=True, key="manual_assignment_button"):
            st.session_state.admin_current_page = "manual_assignment"
            st.rerun()

    with col2:
        if st.button("⚙️ Settings", use_container_width=True, key="settings_button"):
            st.session_state.admin_current_page = "settings"
            st.rerun()

def show_recent_activity():
    """
    Display recent activity in the system.
    """
    st.subheader("Recent Activity")
    
    # Try to get recent activity from database
    recent_activity = []
    
    if settings.DB_ENABLED:
        try:
            conn = db_manager.get_db_connection()
            if conn:                # Get recent calendar uploads
                with conn.cursor() as cursor:
                    # Note: 'username' column doesn't exist in calendar_files table based on schema.py
                    # Using a placeholder 'System' for the actor.
                    cursor.execute(f"""
                    SELECT 'Calendar Upload' as activity_type, filename, upload_date, 'System' as actor
                    FROM {settings.DB_TABLE_CALENDAR_FILES}
                    ORDER BY upload_date DESC
                    LIMIT 5
                    """)
                    for row in cursor.fetchall():
                        recent_activity.append({
                            "type": row[0],
                            "description": f"Uploaded file: {row[1]}",
                            "timestamp": row[2],
                            "actor": row[3]
                        })
                
                # Get recent manual assignments
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                    SELECT 'Manual Assignment' as activity_type, summary, processing_date, personnel
                    FROM {settings.DB_TABLE_PROCESSED_DATA}
                    WHERE processing_status = 'manual_assigned'
                    ORDER BY processing_date DESC
                    LIMIT 5
""")
                    for row in cursor.fetchall():
                        recent_activity.append({
                            "type": row[0],
                            "description": f"Edited entry: {row[1][:50]}...",
                            "timestamp": row[2],
                            "actor": row[3]
                        })
                
                conn.close()
                
                # Sort by timestamp
                recent_activity.sort(key=lambda x: x["timestamp"], reverse=True)
        except Exception as e:
            logger.error(f"Error fetching recent activity: {e}")
    
    # If no activity or no database, show placeholder
    if not recent_activity:
        st.info("No recent activity found")
    else:
        # Display activity in a table
        for activity in recent_activity[:5]:  # Show up to 5 recent activities
            with st.container():
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    if activity["type"] == "Calendar Upload":
                        st.markdown("📅")
                    elif activity["type"] == "Manual Assignment":
                        st.markdown("✏️")
                    else:
                        st.markdown("🔍")
                
                with col2:
                    st.markdown(f"**{activity['type']}**")
                    st.markdown(activity["description"])
                    st.markdown(f"By {activity['actor']} - {activity['timestamp'].strftime('%Y-%m-%d %H:%M') if isinstance(activity['timestamp'], datetime.datetime) else activity['timestamp']}")
                
                st.markdown("---")
