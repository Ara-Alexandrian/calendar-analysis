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

# --- Application Settings ---
APP_TITLE = "Calendar Workload Analyzer"
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOGGING_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# --- LLM Configuration (Simplified) ---
# Ollama is the only supported provider in this simplified version
LLM_PROVIDER = "ollama"
# Model name to use with Ollama
LLM_MODEL = os.environ.get("LLM_MODEL", "mistral:latest")
# Ollama server URL
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.5:11434")

# Extraction performance settings
LLM_MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "3"))

# --- Visualization Settings ---
# Maximum number of personnel to display in workload plots
PLOT_PERSONNEL_LIMIT = int(os.environ.get("PLOT_PERSONNEL_LIMIT", "20"))
# Title prefix for visualization plots
PLOT_TITLE = os.environ.get("PLOT_TITLE", "Calendar Workload Analysis")
# Labels for visualization axes
PLOT_Y_LABEL = os.environ.get("PLOT_Y_LABEL", "Personnel")
PLOT_X_LABEL_HOURS = os.environ.get("PLOT_X_LABEL_HOURS", "Duration (Hours)")
PLOT_X_LABEL_EVENTS = os.environ.get("PLOT_X_LABEL_EVENTS", "Number of Events")

# --- Database Configuration ---
DB_ENABLED = os.environ.get("DB_ENABLED", "True").lower() in ["true", "1", "yes"]
DB_HOST = os.environ.get("DB_HOST", "192.168.1.50")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "spc_calendar")
DB_USER = os.environ.get("DB_USER", "spc_physics")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "!Physics314")
DB_TABLE_PROCESSED_DATA = os.environ.get("DB_TABLE_PROCESSED_DATA", "processed_events")
DB_TABLE_PERSONNEL = os.environ.get("DB_TABLE_PERSONNEL", "personnel_config")
DB_TABLE_CALENDAR_FILES = os.environ.get("DB_TABLE_CALENDAR_FILES", "calendar_files")
DB_TABLE_PROCESSING_STATUS = os.environ.get("DB_TABLE_PROCESSING_STATUS", "processing_status")

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEFAULT_MIN_EVENT_DURATION_MINUTES = 0.5  # 30 seconds as the minimum event duration
