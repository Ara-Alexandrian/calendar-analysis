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
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma3:27b") # Changed default model
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

# --- Event Type Normalization ---
# Dictionary to map raw extracted event types (lowercase) to standardized names
# Based *only* on the comprehensive list provided to the LLM prompt.
EVENT_TYPE_MAPPING = {
    "barrigel": "Barrigel",
    "barrigel olol or": "Barrigel OLOL OR",
    "br 4d": "BR 4D",
    "br 4d bh": "BR 4D BH",
    "br 4d poss bh": "BR 4d poss BH",
    "br 4d rtp": "BR 4D RTP",
    "br 4d rtp bh": "BR 4D RTP BH",
    "br bh": "BR BH",
    "br bh 4d": "BR BH 4D",
    "br gk sim": "BR GK SIM",
    "br rtp 4d": "BR RTP 4D",
    "br rtp unity": "Unity Coverage", # Mapped to Unity Coverage
    "br seed implant": "BR Seed Implant",
    "br unity ct": "Unity Coverage", # Mapped to Unity Coverage
    "br unity rtp": "Unity Coverage", # Mapped to Unity Coverage
    "br unity rtp 4d": "Unity Coverage", # Mapped to Unity Coverage
    "br volume study": "BR Volume Study",
    "br2 sbrt": "BR2 SBRT",
    "br3 sbrt": "BR3 SBRT",
    "brg seed implant": "BRG SEED IMPLANT",
    "cc chart rounds": "CC Chart Rounds",
    "cov 4d": "COV 4D",
    "cov bh": "COV BH",
    "cov hdr": "COV HDR",
    "cov hdr t&r": "COV HDR T&R",
    "cov rtp bh": "COV RTP BH",
    "cov sbrt": "COV SBRT",
    "cv1 sbrt": "CV1 SBRT",
    "cv2 sbrt": "CV2 SBRT",
    "framed gk tx": "Framed GK Tx",
    "gk mri": "GK MRI",
    "gk remote plan review": "GK Remote Plan Review",
    "gk sim": "GK SIM",
    "gk sim/tx": "GK SIM/Tx",
    "gk sbrt tx": "GK SBRT Tx",
    "gk tx": "GK Tx",
    "gk tx sbrt": "GK Tx SBRT",
    "gon 4d": "GON 4D",
    "gon bh": "GON BH",
    "gon pluvicto": "GON Pluvicto",
    "gon sbrt": "GON SBRT",
    "gon xofigo": "GON Xofigo",
    "ham 4d": "HAM 4D", # Using HAM consistently
    "ham 4d rtp": "HAM 4D RTP",
    "ham 4d poss bh": "Ham 4D poss BH", # Keeping original casing as provided
    "ham bh": "Ham BH",
    "ham sbrt": "Ham SBRT",
    "hammond sbrt": "Hammond SBRT",
    "hou 4d": "HOU 4D",
    "hou bh": "HOU BH",
    "hou bh 4d": "HOU BH 4D",
    "hou sbrt": "HOU SBRT",
    "post gk": "Post GK",
    "post gk mri": "Post GK MRI",
    "resident's mr-in-radiotherapy workshop": "Resident's MR-in-Radiotherapy Workshop",
    "seed implant": "Seed Implant",
    "spaceoar classic": "SpaceOAR Classic",
    "spaceoar vue": "SpaceOAR Vue",
    "wh bh": "WH BH",
    "wh breast compression": "WH Breast Compression",
    "wh ct/rtp hdr": "WH CT/RTP HDR",
    "wh hdr": "WH HDR",
    "wh hdr cyl": "WH HDR CYL",
    "wh hdr rtp cyl": "WH HDR RTP CYL",    "wh hdr t&o": "WH HDR T&O",
    "wh rtp bh": "WH RTP BH",
    "wh sbrt": "WH SBRT",
    # --- Updated Unity Mappings to a single category ---
    "cs unity": "Unity Coverage",
    "cs/ss unity": "Unity Coverage",
    "dn unity": "Unity Coverage",
    "jv unity": "Unity Coverage",
    "ss unity": "Unity Coverage",
    "ss/cs unity": "Unity Coverage",
    "ss/dn unity": "Unity Coverage",
    # Add common variations if the LLM extracts them explicitly
    "unity": "Unity Coverage", # Generic mapping if LLM just outputs "Unity"
    "ss/jv unity": "Unity Coverage", # Example: add other combinations as needed
    # --------------------------------------------------
    "gk mri": "GK MRI", # Added missing event type
    # New event types added to match extractor.py and logs
    "br rtp sbrt bh": "BR RTP SBRT BH",
    "br rtp bh": "BR RTP BH",
    "wh ct/rtp hdr cyl": "WH CT/RTP HDR CYL",
    "ham 4d ct rtp": "Ham 4d CT RTP", # Keeping casing for now, normalize if needed
    "gon 4d rtp bh": "GON 4D RTP BH",
    "wh hdr cyl 4d": "WH HDR CYL 4D",    "br sbrt": "BR SBRT", # Added from previous log warning
    "gk framed tx": "GK Framed Tx", # Added variation of "Framed GK Tx"
    "unknown": "Unknown", # Ensure Unknown maps to Unknown
    # Additional mappings from previous log warnings not explicitly in the provided list:
    "wh hdr cyl 4:5": "WH HDR CYL", # Map specific variation to general type
    "ham 4d ct rtp": "HAM 4D RTP", # Standardize HAM capitalization if necessary
    "hou bh 4d": "HOU BH 4D", # Ensure this is present and correctly cased
    "br 2 sbrt": "BR2 SBRT", # Map BR 2 to BR2
    # --- Added mapping for "post GK coverage" variation ---
    "post gk coverage": "Post GK",
    # --------------------------------------------------
    # Add any other raw types you see in the WARNING logs here
}

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
