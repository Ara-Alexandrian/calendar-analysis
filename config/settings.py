# config/settings.py
import os
import logging
import sys

# --- Core Path Configuration ---
# Get project root assuming settings.py is in config/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to path if it's not already there
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Define output and data directories relative to the project root
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data') # For example data if needed
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')

# --- Input/Output File Paths ---
PERSONNEL_CONFIG_JSON_PATH = os.path.join(CONFIG_DIR, 'personnel_config.json') # Dynamic config
# Raw data is uploaded, processed data stored in session state, maybe allow export later
PROCESSED_EXPORT_PATH = os.path.join(OUTPUT_DIR, 'processed_events_export.json') # Optional export
LOG_FILE = os.path.join(OUTPUT_DIR, 'calendar_analysis.log')

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LLM Configuration ---
LLM_PROVIDER = "ollama" # Options: "ollama", None
OLLAMA_BASE_URL = "http://192.168.1.5:11434" # Required if LLM_PROVIDER is "ollama"
LLM_MODEL = "llama3.1:8b"

# --- Performance Configuration ---
LLM_MAX_WORKERS = 10 # Parallel threads for LLM extraction

# --- Logging Configuration ---
LOGGING_LEVEL_STR = "INFO" # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOGGING_LEVEL = getattr(logging, LOGGING_LEVEL_STR.upper(), logging.INFO)
LOG_FORMAT = '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

# --- Analysis Parameters (Defaults - can be overridden in UI) ---
DEFAULT_MIN_EVENT_DURATION_MINUTES = 5

# --- Plotting Configuration (Defaults) ---
PLOT_TITLE = "Personnel Workload Summary"
PLOT_X_LABEL_HOURS = "Total Duration (Hours)"
PLOT_X_LABEL_EVENTS = "Total Event Count"
PLOT_Y_LABEL = "Personnel"
PLOT_PERSONNEL_LIMIT = 25

# --- UI Configuration ---
APP_TITLE = "Calendar Workload Analyzer"