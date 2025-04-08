# config.py
"""
Configuration settings for the calendar analysis project.
"""
import logging

# --- File Paths ---
# Assume ICS file path might still be needed elsewhere, or handled via argparse
# ICS_FILE_PATH = "/path/to/your/calendar.ics" # Example if needed

# Add the missing JSON_FILE_PATH - adjust the actual path as needed
# This might be used for saving processed data, LLM outputs, or final results.
JSON_FILE_PATH = "" # Example path

# Define output directory (useful for plots, reports)
OUTPUT_DIR = "./output"

# --- LLM Configuration ---
# Set to None if not using LLM features
LLM_PROVIDER = "ollama" # or "openai", "anthropic", None
OLLAMA_BASE_URL = "http://192.168.1.5:11434" # Required if LLM_PROVIDER is "ollama"
LLM_MODEL = "llama3.1:8b" # Model name (e.g., "gpt-4o", "claude-3-opus-20240229", "llama3.1:8b")
# Add API Keys if needed (using environment variables is recommended for security)
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# --- Performance Configuration ---
# Number of parallel threads to use for LLM extraction (if applicable)
LLM_MAX_WORKERS = 8  # Adjust based on performance and server capacity

# --- Personnel Configuration ---
# Renamed from PHYSICISTS_CONFIG for clarity
# Added 'role' to distinguish between roles if needed for analysis
PERSONNEL_CONFIG = {
    # Physicists
    "A. Alexandrian": {"role": "physicist", "clinical_pct": 0.80, "variations": ["A. Alexandrian", "Alexandrian"]},
    "D. Perrin": {"role": "physicist", "clinical_pct": 0.95, "variations": ["D. Perrin", "Perrin"]},
    "D. Neck": {"role": "physicist", "clinical_pct": 0.95, "variations": ["D. Neck", "Neck"]},
    "D. Solis": {"role": "physicist", "clinical_pct": 0.60, "variations": ["D. Solis", "Solis"]},
    "C. Chu": {"role": "physicist", "clinical_pct": 0.95, "variations": ["C. Chu", "Chu"]},
    "B. Smith": {"role": "physicist", "clinical_pct": 0.95, "variations": ["B. Smith", "Smith"]}, # Ambiguous if just "Smith"
    "A. McGuffey": {"role": "physicist", "clinical_pct": 0.80, "variations": ["A. McGuffey", "McGuffey"]},
    "E. Chorniak": {"role": "physicist", "clinical_pct": 0.95, "variations": ["E. Chorniak", "Chorniak"]},
    "G. Pitcher": {"role": "physicist", "clinical_pct": 0.50, "variations": ["G. Pitcher", "Pitcher"]},
    "S. Stathakis": {"role": "physicist", "clinical_pct": 0.40, "variations": ["S. Stathakis", "Stathakis"]},
    "C. Schneider": {"role": "physicist", "clinical_pct": 0.80, "variations": ["C. Schneider", "Schneider"]},
    "R. Guidry": {"role": "physicist", "clinical_pct": 0.95, "variations": ["R. Guidry", "Guidry"]},
    "J. Voss": {"role": "physicist", "clinical_pct": 0.95, "variations": ["J. Voss", "Voss"]},
    "A. Husain": {"role": "physicist", "clinical_pct": 0.95, "variations": ["A. Husain", "Husain"]},
    "J. Chen": {"role": "physicist", "clinical_pct": 0.95, "variations": ["J. Chen", "Chen"]},
    # Physicians (as noted by user)
    "Wood": {"role": "physician", "clinical_pct": 0.95, "variations": ["Wood"]}, # Assuming clinical_pct is still relevant/correct
    "King": {"role": "physician", "clinical_pct": 0.95, "variations": ["King", "King III"]},
    "Kovtun": {"role": "physician", "clinical_pct": 0.95, "variations": ["Kovtun"]}, # User identified as Physician
    "Wang": {"role": "physician", "clinical_pct": 0.95, "variations": ["Wang"]},
    "Elson": {"role": "physician", "clinical_pct": 0.95, "variations": ["Elson"]},
    "Castle": {"role": "physician", "clinical_pct": 0.95, "variations": ["Castle"]}, # User identified as Physician
    "Hymel": {"role": "physician", "clinical_pct": 0.95, "variations": ["Hymel"]},
    "Henkelmann": {"role": "physician", "clinical_pct": 0.95, "variations": ["Henkelmann"]},
    "Bermudez": {"role": "physician", "clinical_pct": 0.95, "variations": ["Bermudez"]},
}

# --- Derived Configuration (automatically generated) ---

# Map all variations (lowercase) to the canonical personnel name
VARIATION_MAP = {}
for name, config_data in PERSONNEL_CONFIG.items():
    if not isinstance(config_data, dict):
        logging.warning(f"Invalid config for {name}. Skipping.")
        continue
    for var in config_data.get("variations", []):
        lower_var = var.lower()
        if lower_var in VARIATION_MAP:
            logging.warning(f"Duplicate variation '{var}' mapped to '{name}'. It was previously mapped to '{VARIATION_MAP[lower_var]}'. Overwriting.")
        VARIATION_MAP[lower_var] = name

# List of canonical names for validation or iteration
CANONICAL_NAMES = list(PERSONNEL_CONFIG.keys())

# Function to safely get clinical percentage
def get_clinical_pct(personnel_name):
    """Retrieves the clinical percentage for a given canonical personnel name."""
    config_data = PERSONNEL_CONFIG.get(personnel_name, {})
    if isinstance(config_data, dict):
        return config_data.get('clinical_pct', None)
    return None # Return None if the name is somehow misconfigured or not found

# Function to safely get role
def get_role(personnel_name):
    """Retrieves the role for a given canonical personnel name."""
    config_data = PERSONNEL_CONFIG.get(personnel_name, {})
    if isinstance(config_data, dict):
        return config_data.get('role', 'Unknown') # Default to 'Unknown' if not specified
    return 'Unknown'

# --- Logging Configuration ---
LOGGING_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
LOG_FILE = './output/calendar_analysis.log' # Example log file path

# --- Analysis Parameters ---
# Define the date range for analysis (inclusive)
# Set to None to analyze all events in the ICS file
# Example: START_DATE = "2024-01-01"
# Example: END_DATE = "2024-12-31"
START_DATE = None
END_DATE = None

# Define categories and rules (example structure, adapt as needed)
# This might replace or supplement regex rules previously in config.yaml
# CATEGORY_RULES = {
#     "Meetings": ["meeting", "sync", "1:1"],
#     "Clinical Prep": ["contour", "plan check", "chart check"],
#     "Treatment": ["tx", "treatment delivery"],
#     # Add more categories and keywords/regex patterns
# }

# Minimum event duration in minutes to be considered in analysis (e.g., ignore 5-min reminders)
MIN_EVENT_DURATION_MINUTES = 10

# --- Plotting Configuration ---
PLOT_OUTPUT_PATH = "./output/workload_summary.png"
PLOT_TITLE = "Personnel Workload Summary"
PLOT_X_LABEL_HOURS = "Total Duration (Hours)"
PLOT_X_LABEL_EVENTS = "Total Event Count"
PLOT_Y_LABEL = "Personnel"