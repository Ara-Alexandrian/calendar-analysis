"""
Calendar Workload Analyzer - Admin Dashboard
Main entry point for the administrator dashboard with enhanced capabilities.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path to ensure all imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import Streamlit
import streamlit as st

# Import configuration
from config import settings

# Set page configuration before any other Streamlit commands
st.set_page_config(
    page_title="Calendar Analyzer Admin Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import auth utilities
from admin_dashboard.utils.auth import check_authentication, show_login_page

# Configure logging
logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting Calendar Workload Analyzer Admin Dashboard")

# Apply custom CSS
def load_css():
    with open(os.path.join(os.path.dirname(__file__), "static", "css", "style.css"), "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Check authentication
if not check_authentication():
    show_login_page()
else:    # Import sidebar component after authentication is confirmed
    from admin_dashboard.components.sidebar import show_sidebar
    
    # Display sidebar
    show_sidebar()
    
    # Initialize current page in session state if it doesn't exist
    if 'admin_current_page' not in st.session_state:
        st.session_state.admin_current_page = "dashboard"
    # Import page components
    from admin_dashboard.pages._dashboard import show_dashboard_page
    from admin_dashboard.pages._analysis import show_analysis_page # Corrected import path
    from admin_dashboard.pages._manual_assignment import show_manual_assignment_page
    from admin_dashboard.pages._settings import show_settings_page
    
    # Show the appropriate page based on selection
    current_page = st.session_state.admin_current_page
    
    if current_page == "dashboard":
        show_dashboard_page()
    elif current_page == "analysis":
        show_analysis_page()
    elif current_page == "manual_assignment":
        show_manual_assignment_page()
    elif current_page == "settings":
        show_settings_page()
    else:
        # Default to dashboard
        show_dashboard_page()

    # Footer
    st.markdown("---")
    st.markdown("© 2025 Calendar Workload Analyzer - Admin Dashboard")
