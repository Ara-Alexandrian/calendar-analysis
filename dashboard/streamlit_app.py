"""
Calendar Workload Analyzer - Main Streamlit Application
This is the main entry point for the Streamlit application.
"""
# Standard library imports
import sys
import os
import logging
import traceback
from pathlib import Path

# First, set up the Python path before any other imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now import streamlit
import streamlit as st

# CRITICAL: Set page configuration FIRST, before any other st.* calls
st.set_page_config(
    page_title="Calendar Workload Analyzer",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now try to load settings
try:
    from config import settings
except Exception as e:
    # Create default settings
    class DefaultSettings:
        APP_TITLE = "Calendar Workload Analyzer"
        LOGGING_LEVEL = logging.INFO
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        LOG_FILE = "calendar_analysis.log"
        PROJECT_ROOT = project_root
        OUTPUT_DIR = os.path.join(project_root, "output")
        OLLAMA_BASE_URL = "http://localhost:11434"
        LLM_PROVIDER = "ollama"
        LLM_MODEL = "mistral:latest"
        DB_ENABLED = False
    
    settings = DefaultSettings()
    st.warning("Using default settings due to import error. Some features may be limited.")
    st.error(f"Error importing modules: {e}")
    st.code(traceback.format_exc())

# Configure logging
# Ensure the output directory exists
log_dir = os.path.dirname(settings.LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE, encoding='utf-8'), # Explicitly set UTF-8 encoding
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Starting Calendar Workload Analyzer application")

# Initialize database if DB_ENABLED
if settings.DB_ENABLED:
    try:
        from functions import initialize_db
        logger.info("Initializing database with baseline configuration...")
        initialize_db.initialize_database()
    except Exception as db_init_error:
        logger.error(f"Error during database initialization: {db_init_error}")
        st.error("Database initialization failed. Check the logs for details.")

# Initialize session state variables for persistent storage across pages
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None
if 'llm_processed_df' not in st.session_state:
    st.session_state.llm_processed_df = None
if 'normalized_df' not in st.session_state:
    st.session_state.normalized_df = None
if 'analysis_ready_df' not in st.session_state:
    st.session_state.analysis_ready_df = None
if 'current_batch_id' not in st.session_state:
    st.session_state.current_batch_id = None

# Main app header
st.title(f"📅 {settings.APP_TITLE}")

# Welcome message on the main page
st.markdown("""
## Welcome to the Calendar Workload Analyzer

This application helps you analyze calendar data to understand workload distribution across your team.

### Getting Started:
1. 📁 Navigate to the **Upload & Process** page to upload your calendar data
2. 📊 Use the **Analysis** page to explore the processed data
3. ⚙️ Visit the **Admin** page to configure personnel settings
4. 🔍 Use the **Database Viewer** if you want to explore the raw data (when database is enabled)

### Current Configuration:
- LLM Provider: {0}
- LLM Model: {1}
- Database Enabled: {2}
""".format(
    settings.LLM_PROVIDER, 
    settings.LLM_MODEL,
    "Yes" if settings.DB_ENABLED else "No"
))

# Display system information
with st.expander("System Information"):
    st.write(f"Project Root: {settings.PROJECT_ROOT}")
    st.write(f"Output Directory: {settings.OUTPUT_DIR}")
    st.write(f"Ollama Base URL: {settings.OLLAMA_BASE_URL}")
    
    # Check if Ollama is available
    try:
        from functions.llm_extraction.ollama_client import is_ollama_ready
        ollama_status = is_ollama_ready()
        st.write(f"Ollama Connection: {'✅ Connected' if ollama_status else '❌ Not Connected'}")
    except Exception as e:
        st.write(f"Ollama Connection: ❌ Error checking connection - {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2025 Calendar Workload Analyzer")
