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

# Hide the default Streamlit sidebar navigation
st.config.set_option('browser.showSidebarNav', False)

# Custom CSS to hide the default sidebar navigation elements
st.markdown("""
<style>
section[data-testid="stSidebar"] > div:first-child > div:nth-child(2) {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

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
    
    # Try to import page components with proper error handling
    try:
        from admin_dashboard.pages._dashboard import show_dashboard_page
        from admin_dashboard.pages._analysis import show_analysis_page
        # Handle manual assignment page separately
        try:
            from admin_dashboard.pages._manual_assignment import show_manual_assignment_page
        except ImportError as e:
            logger.error(f"Error importing manual assignment page: {e}")
            def show_manual_assignment_page():
                st.title("Manual Assignment (Unavailable)")
                st.error("The manual assignment page could not be loaded due to an error.")
                st.info("Please check the logs for more information.")
                
        from admin_dashboard.pages._settings import show_settings_page
    except ImportError as e:
        logger.error(f"Error importing page components: {e}")
        st.error("Some page components could not be loaded. Please check logs for more information.")
    
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
