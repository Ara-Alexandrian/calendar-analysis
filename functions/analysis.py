import json
import pandas as pd
import re
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
JSON_FILE_PATH = 'calendar.json'

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
# Re-generate VARIATION_MAP automatically in the script
    # Add other known names if applicable (e.g., from the summaries)
    "Wood": {"clinical_pct": 0.95, "variations": ["Wood"]}, # Assuming Wood is ~95% clinical
    "King": {"clinical_pct": 0.95, "variations": ["King", "King III"]}, # Assuming King is ~95% clinical
    "Kovtun": {"clinical_pct": 0.95, "variations": ["Kovtun"]}, # Assuming Kovtun is ~95% clinical
    "Wang": {"clinical_pct": 0.95, "variations": ["Wang"]}, # Assuming Wang is ~95% clinical
    "Elson": {"clinical_pct": 0.95, "variations": ["Elson"]}, # Assuming Elson is ~95% clinical
    "Castle": {"clinical_pct": 0.95, "variations": ["Castle"]}, # Assuming Castle is ~95% clinical
    "Hymel": {"clinical_pct": 0.95, "variations": ["Hymel"]}, # Assuming Hymel is ~95% clinical
    "Henkelmann": {"clinical_pct": 0.95, "variations": ["Henkelmann"]},# Assuming Henkelmann is ~95% clinical
    "Bermudez": {"clinical_pct": 0.95, "variations": ["Bermudez"]},# Assuming Bermudez is ~95% clinical
    # Example of how to handle someone not explicitly listed initially
    # "T. Hagan": {"clinical_pct": 0.95, "variations": ["T. Hagan", "Hagan"]},
    # "G. Debevec": {"clinical_pct": 0.95, "variations": ["G. Debevec", "Debevec"]}

}

# Build a reverse map from variation to canonical name
VARIATION_MAP = {}
for name, config in PHYSICISTS_CONFIG.items():
    for var in config["variations"]:
        VARIATION_MAP[var.lower()] = name # Use lower case for matching

# --- Helper Functions ---

def parse_datetime(dt_str):
    """Handles different datetime formats found in the JSON."""
    if isinstance(dt_str, str):
        # Handle formats like '2024-10-16 18:00:00+00:00'
        try:
            return pd.to_datetime(dt_str, errors='coerce')
        except Exception:
            # Handle formats like '20241016T180652Z'
            try:
                 return pd.to_datetime(dt_str, format='%Y%m%dT%H%M%SZ', errors='coerce', utc=True)
            except Exception:
                 # Handle vDDDTypes format by extracting the datetime part
                 match = re.search(r"(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})", dt_str)
                 if match:
                     try:
                        return pd.to_datetime(match.group(1), errors='coerce', utc=True)
                     except Exception:
                        return pd.NaT
                 else:
                     return pd.NaT

    elif isinstance(dt_str, dict) and 'dt' in dt_str: # Handle potential nested structure if exists
         return parse_datetime(dt_str['dt'])
    return pd.NaT


def extract_physicists(summary):
    """Extracts physicist names/initials from the summary string."""
    found_physicists = set()
    
    # Split by '//' and check the last few segments
    segments = [s.strip() for s in summary.split('//') if s.strip()]
    
    # Check the last 3 segments (most likely place for names)
    segments_to_check = segments[-3:] 
    
    for segment in segments_to_check:
        # Clean up potential noise like '?'
        cleaned_segment = segment.replace('?', '').strip()
        
        # Attempt direct match using the variation map
        if cleaned_segment.lower() in VARIATION_MAP:
            found_physicists.add(VARIATION_MAP[cleaned_segment.lower()])
            continue # Found a match, move to next segment if any

        # Try matching initials like "A. Alexandrian" or single last names
        # Be careful with single initials as they are ambiguous
        potential_names = re.findall(r'([A-Z]\.\s*[A-Z][a-zA-Z]+(?:\s*[A-Z][a-zA-Z]+)?|[A-Z][a-zA-Z]+(?:(?:\s|,|&)[A-Z][a-zA-Z]+)*)', cleaned_segment)
        # Example potential_names for "Kovtun // D. Perrin" -> ['Kovtun', 'D. Perrin']
        
        for name in potential_names:
            name = name.strip()
            if name.lower() in VARIATION_MAP:
                 found_physicists.add(VARIATION_MAP[name.lower()])
            # Handle cases like "King III" -> map to "King"
            elif name.replace(' III', '').lower() in VARIATION_MAP:
                 found_physicists.add(VARIATION_MAP[name.replace(' III', '').lower()])


    if not found_physicists:
        # Fallback: Check the entire summary if segments failed
         for var, canonical_name in VARIATION_MAP.items():
              # Use regex to find variations as whole words to avoid partial matches
              # (e.g., finding 'King' in 'Parking')
              if re.search(r'\b' + re.escape(var) + r'\b', summary, re.IGNORECASE):
                   found_physicists.add(canonical_name)
                   
    return list(found_physicists) if found_physicists else ["Unknown"]

# --- Main Processing ---

# 1. Load Data
try:
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except FileNotFoundError:
    print(f"Error: JSON file not found at {JSON_FILE_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {JSON_FILE_PATH}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred loading the data: {e}")
    exit()

# 2. Parse Timestamps and Calculate Duration
df['start_time'] = df['start'].apply(parse_datetime)
df['end_time'] = df['end'].apply(parse_datetime)

# Drop rows where time parsing failed
df.dropna(subset=['start_time', 'end_time'], inplace=True)

# Ensure timezone consistency (convert all to UTC for calculation)
df['start_time'] = df['start_time'].dt.tz_convert('UTC')
df['end_time'] = df['end_time'].dt.tz_convert('UTC')


df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

# Filter out negative or zero durations which indicate errors
df = df[df['duration_minutes'] > 0]

# 3. Extract Physicists
df['assigned_physicists'] = df['summary'].apply(extract_physicists)

# 4. Explode DataFrame for multi-physicist events
#    Each row with multiple physicists becomes multiple rows, one for each.
#    This allows easy grouping but means durations are counted per physicist involved.
df_exploded = df.explode('assigned_physicists')
df_exploded.rename(columns={'assigned_physicists': 'physicist'}, inplace=True)

# --- Analysis ---

# Filter out 'Unknown' assignments for workload analysis
df_assigned = df_exploded[df_exploded['physicist'] != "Unknown"].copy()

# Calculate workload metrics
workload = df_assigned.groupby('physicist').agg(
    total_events=('uid', 'count'),
    total_duration_minutes=('duration_minutes', 'sum')
).reset_index()

# Convert total duration to hours for readability
workload['total_duration_hours'] = workload['total_duration_minutes'] / 60

# Add clinical percentage context
workload['clinical_pct'] = workload['physicist'].map(lambda name: PHYSICISTS_CONFIG.get(name, {}).get('clinical_pct', None))

# Calculate expected clinical hours (based on total hours in analysis period - ROUGH ESTIMATE)
# NOTE: This is a VERY rough estimate. A proper calculation needs the *total* working hours
# for the period covered by the calendar data, which isn't available here.
# Let's assume a standard work week for simplicity in demonstrating the concept.
total_analysis_duration_days = (df['end_time'].max() - df['start_time'].min()).days
if total_analysis_duration_days == 0: total_analysis_duration_days = 1 # Avoid division by zero for short periods
# Assuming ~8 hours/day clinical availability for a 100% clinical person over the period
# This needs significant refinement based on actual work patterns.
# For now, let's just show the assigned hours vs clinical %
# estimated_total_available_hours = total_analysis_duration_days * 8
# workload['expected_clinical_hours'] = workload['clinical_pct'] * estimated_total_available_hours
# workload['workload_vs_expected(%)'] = (workload['total_duration_hours'] / workload['expected_clinical_hours']) * 100

workload = workload.sort_values(by='total_duration_hours', ascending=False)


# --- Output ---
print("--- Physicist Workload Summary ---")
print(workload[['physicist', 'clinical_pct', 'total_events', 'total_duration_hours']].round(2))

print("\n--- Events Assigned to 'Unknown' ---")
unknown_events = df_exploded[df_exploded['physicist'] == "Unknown"]
if not unknown_events.empty:
    print(f"Found {len(unknown_events)} events with unknown physicist assignment.")
    print(unknown_events[['uid', 'summary']].head()) # Print first few
else:
    print("No events found with unknown physicist assignment.")

# --- Visualization ---
plt.figure(figsize=(12, 8))
sns.barplot(data=workload, x='total_duration_hours', y='physicist', palette='viridis')
plt.title('Total Assigned Clinical Duration per Physicist (Hours)')
plt.xlabel('Total Duration (Hours)')
plt.ylabel('Physicist')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
sns.barplot(data=workload.sort_values('total_events', ascending=False), x='total_events', y='physicist', palette='magma')
plt.title('Total Number of Assigned Events per Physicist')
plt.xlabel('Number of Events')
plt.ylabel('Physicist')
plt.tight_layout()
plt.show()

# Optional: Plot workload relative to clinical % (if a good baseline is established)
# This requires a meaningful way to calculate 'expected hours'.
# Example (needs refinement on expected hours calculation):
# workload_filtered = workload.dropna(subset=['expected_clinical_hours'])
# if not workload_filtered.empty:
#     plt.figure(figsize=(12, 8))
#     sns.barplot(data=workload_filtered.sort_values('workload_vs_expected(%)', ascending=False),
#                 x='workload_vs_expected(%)', y='physicist', palette='coolwarm')
#     plt.title('Assigned Workload vs. Expected Clinical Availability (%)')
#     plt.xlabel('Workload / Expected (%)')
#     plt.ylabel('Physicist')
#     plt.axvline(100, color='grey', linestyle='--', label='100% Expected')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()