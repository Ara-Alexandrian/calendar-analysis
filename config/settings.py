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

# --- LLM Configuration ---
# LLM provider can be "ollama" or "mcp" 
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
# Model name to use with the LLM provider
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3")
# Ollama-specific settings (Unraid server with 2x3090 NVLink)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://192.168.1.5:11434")
# MCP-specific settings (Local PC with 4090)
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8000")
MCP_REQUEST_TIMEOUT = int(os.environ.get("MCP_REQUEST_TIMEOUT", "60"))
MCP_MODEL_NAME = os.environ.get("MCP_MODEL_NAME", "microsoft/phi-2")  # Default model for MCP server

# --- Smart Router Configuration ---
# Enable multi-provider support for using both Ollama and MCP simultaneously
ENABLE_MULTI_PROVIDER = os.environ.get("ENABLE_MULTI_PROVIDER", "True").lower() in ["true", "1", "yes"]
# Enable the smart router for automatic provider selection
USE_SMART_ROUTER = os.environ.get("USE_SMART_ROUTER", "True").lower() in ["true", "1", "yes"]
# Router strategy - options: "balanced", "performance", "reliability"
ROUTER_STRATEGY = os.environ.get("ROUTER_STRATEGY", "balanced")
# Large model threshold in billions of parameters - models above this use Ollama with more VRAM
LARGE_MODEL_THRESHOLD = int(os.environ.get("LARGE_MODEL_THRESHOLD", "13"))

# Extraction performance settings
LLM_MAX_WORKERS = int(os.environ.get("LLM_MAX_WORKERS", "3"))

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Dependencies Information ---
# This section documents the dependencies required for different features
# Required dependencies:
#   - streamlit: Web UI framework
#   - pandas: Data processing
#   - plotly: Visualization
#
# Optional dependencies:
#   - ollama: Required for LLM extraction of personnel names (pip install ollama)
#   - tqdm: Optional for progress bars in CLI mode
#   - psycopg2-binary: Required for PostgreSQL database connection

# --- Database Configuration ---
DB_ENABLED = True  # Enable database persistence
DB_HOST = "192.168.1.50"
DB_PORT = "5432"
DB_NAME = "postgres"  # Changed to use the default PostgreSQL database that always exists
DB_USER = "spc_physics"
DB_PASSWORD = "!Physics314"  # Make sure there are no hidden/whitespace characters
DB_TABLE_PROCESSED_DATA = "processed_events"
DB_TABLE_PERSONNEL = "personnel_config"
DB_TABLE_CALENDAR_FILES = "calendar_files"  # Adding this setting for consistency

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