# streamlit_app.py
import streamlit as st
import logging
import os

# --- Path Setup & Early Init ---
# Add project root to sys.path BEFORE importing local modules
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level to get to the root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# --- End Path Setup ---

from config import settings # Import static settings
from functions import config_manager # Import manager to load config early if needed

# --- Configure logging ---
# Ensure log directory exists
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
# Clear existing handlers if any (important for Streamlit re-runs)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Set up console handler with UTF-8 encoding
console_handler = logging.StreamHandler()
# Set encoding for FileHandler to UTF-8 explicitly
file_handler = logging.FileHandler(settings.LOG_FILE, encoding='utf-8')

# Configure formatter
formatter = logging.Formatter(settings.LOG_FORMAT)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Set up basic config with these handlers
logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)
# --- End Logging Config ---

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title=settings.APP_TITLE,
    page_icon="📅",
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
# Check and initialize keys needed across pages if they don't exist
# Raw data from upload
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
# Data after basic preprocessing (dates, duration)
if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None
# Data after LLM extraction (includes 'extracted_personnel')
if 'llm_processed_df' not in st.session_state:
    st.session_state.llm_processed_df = None
# Data after normalization (includes 'assigned_personnel' list)
if 'normalized_df' not in st.session_state:
    st.session_state.normalized_df = None
# Final data, exploded by personnel, ready for analysis
if 'analysis_ready_df' not in st.session_state:
    st.session_state.analysis_ready_df = None
# Flag to indicate if data has been successfully processed
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
# Store the filename of the uploaded file
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None
# Store personnel config (loaded once or when refreshed)
if 'personnel_config' not in st.session_state:
     # Load initial config into session state
     p_config, v_map, c_names = config_manager.load_personnel_config()
     st.session_state.personnel_config = p_config
     st.session_state.variation_map = v_map
     st.session_state.canonical_names = c_names
     logger.info("Initial personnel config loaded into session state.")


# --- Main App Display ---
st.title(f"📅 {settings.APP_TITLE}")

st.markdown("""
Welcome to the Calendar Workload Analyzer.

Use the sidebar to navigate between pages:
1.  **📁 Upload & Process:** Upload your `calendar.json` file and run the processing pipeline (including LLM extraction).
2.  **📊 Analysis:** Explore the processed data with interactive filters and visualizations.
3.  **⚙️ Admin:** View and manage the personnel configuration used for analysis.

**Status:**
""")

# Display current data status
if st.session_state.data_processed and st.session_state.analysis_ready_df is not None:
     st.success(f"Data from '{st.session_state.uploaded_filename}' is processed and ready for analysis ({len(st.session_state.analysis_ready_df)} analysis rows). Navigate to the 'Analysis' page.")
elif st.session_state.raw_df is not None:
     st.warning(f"Raw data from '{st.session_state.uploaded_filename}' is loaded but not yet processed. Go to the 'Upload & Process' page.")
else:
     st.info("No data loaded. Please go to the 'Upload & Process' page to begin.")

# Optionally display loaded personnel count
st.markdown(f"**Personnel Configured:** {len(st.session_state.canonical_names)} names loaded.")

st.sidebar.success("Select a page above.")

logger.info("Streamlit main page executed.")

# Note: The actual logic for each page is in the files within the 'pages/' directory.
# Streamlit automatically discovers and runs them based on the file structure.