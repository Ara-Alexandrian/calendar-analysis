# config.py
"""
Configuration settings for the calendar analysis project.
"""
import os # Keep os import if you might use env vars for other things

# --- File Paths ---
JSON_FILE_PATH = 'calendar.json'

# --- Physicist Configuration ---
# (Keep PHYSICISTS_CONFIG as it was)
PHYSICISTS_CONFIG = {
    "A. Alexandrian": {"clinical_pct": 0.80, "variations": ["A. Alexandrian", "Alexandrian"]},
    "D. Perrin": {"clinical_pct": 0.95, "variations": ["D. Perrin", "Perrin"]},
    "D. Neck": {"clinical_pct": 0.95, "variations": ["D. Neck", "Neck"]},
    "D. Solis": {"clinical_pct": 0.60, "variations": ["D. Solis", "Solis"]},
    "C. Chu": {"clinical_pct": 0.95, "variations": ["C. Chu", "Chu"]},
    "B. Smith": {"clinical_pct": 0.95, "variations": ["B. Smith", "Smith"]}, # Ambiguous if just "Smith"
    "A. McGuffey": {"clinical_pct": 0.80, "variations": ["A. McGuffey", "McGuffey"]},
    "E. Chorniak": {"clinical_pct": 0.95, "variations": ["E. Chorniak", "Chorniak"]},
    "G. Pitcher": {"clinical_pct": 0.50, "variations": ["G. Pitcher", "Pitcher"]},
    "S. Stathakis": {"clinical_pct": 0.40, "variations": ["S. Stathakis", "Stathakis"]},
    "C. Schneider": {"clinical_pct": 0.80, "variations": ["C. Schneider", "Schneider"]},
    "R. Guidry": {"clinical_pct": 0.95, "variations": ["R. Guidry", "Guidry"]},
    "J. Voss": {"clinical_pct": 0.95, "variations": ["J. Voss", "Voss"]},
    "A. Husain": {"clinical_pct": 0.95, "variations": ["A. Husain", "Husain"]}, # Estimate clinical %
    "J. Chen": {"clinical_pct": 0.95, "variations": ["J. Chen", "Chen"]},       # Estimate clinical %
    "Wood": {"clinical_pct": 0.95, "variations": ["Wood"]},
    "King": {"clinical_pct": 0.95, "variations": ["King", "King III"]},
    "Kovtun": {"clinical_pct": 0.95, "variations": ["Kovtun"]},
    "Wang": {"clinical_pct": 0.95, "variations": ["Wang"]},
    "Elson": {"clinical_pct": 0.95, "variations": ["Elson"]},
    "Castle": {"clinical_pct": 0.95, "variations": ["Castle"]},
    "Hymel": {"clinical_pct": 0.95, "variations": ["Hymel"]},
    "Henkelmann": {"clinical_pct": 0.95, "variations": ["Henkelmann"]},
    "Bermudez": {"clinical_pct": 0.95, "variations": ["Bermudez"]},
}


# --- LLM Configuration ---
LLM_PROVIDER = "ollama"  # Changed provider
OLLAMA_BASE_URL = "http://192.168.1.5:11434" # Your Ollama server URL (use http)
# Choose the model name exactly as listed by 'ollama list'
LLM_MODEL = "llama3.1:8b" # Recommended starting model
# Alternative models you could try: "mistral:latest", "nous-hermes2-mixtral:latest"

# --- Derived Configuration ---

# Build a reverse map from any variation (lowercase) to the canonical name
VARIATION_MAP = {}
for name, config_data in PHYSICISTS_CONFIG.items():
    for var in config_data.get("variations", []):
        VARIATION_MAP[var.lower()] = name

# List of canonical names for the LLM prompt
CANONICAL_NAMES = list(PHYSICISTS_CONFIG.keys())

# Function to get clinical percentage for a canonical name
def get_clinical_pct(physicist_name):
    """Safely retrieves the clinical percentage for a given physicist."""
    return PHYSICISTS_CONFIG.get(physicist_name, {}).get('clinical_pct', None)