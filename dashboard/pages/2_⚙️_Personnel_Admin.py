# pages/3_⚙️_Admin.py
import streamlit as st
import pandas as pd
import logging
import json # For validating variations input

# Ensure project root is in path
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functions import config_manager
from config import settings # Import settings for database configuration

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Admin - Personnel Config", layout="wide")

st.markdown("# 3. Admin: Manage Personnel Configuration")
st.markdown("View, edit, add, or remove personnel entries used for analysis.")
st.markdown(f"Configuration is loaded from and saved to: `{config_manager.settings.PERSONNEL_CONFIG_JSON_PATH}`")

# --- Load Current Config ---
# Use force_reload=True if we want changes made here to reflect immediately
# without needing a full app restart after saving.
try:
     personnel_config, variation_map, canonical_names = config_manager.load_personnel_config(force_reload=True)
     # Store a working copy in session state to track edits before saving
     if 'admin_personnel_config' not in st.session_state or st.button("Reload Config from File", key="reload_cfg"):
         st.session_state.admin_personnel_config = personnel_config.copy() # Work on a copy
         st.success("Personnel configuration loaded/reloaded.")
     working_config = st.session_state.admin_personnel_config
except Exception as e:
     st.error(f"Failed to load personnel configuration: {e}")
     logger.error(f"Admin page failed to load config: {e}", exc_info=True)
     st.stop()

# --- Display/Edit Existing Personnel using st.data_editor ---
st.subheader("Edit Existing Personnel")

if not working_config:
    st.info("No personnel configured yet. Use the 'Add New Personnel' section below.")
else:
    # Convert the config dict to a list of dicts suitable for DataFrame/data_editor
    config_list = []
    for name, details in working_config.items():
        config_list.append({
            'Canonical Name': name,
            'Role': details.get('role', ''),
            'Clinical %': details.get('clinical_pct', 0.0),
            # Variations need careful handling - display as string, edit needs validation
            'Variations (JSON list)': json.dumps(details.get('variations', [])),
        })

    # Create DataFrame for editing
    df_config = pd.DataFrame(config_list)

    st.info("Edit values directly in the table below. Variations must be entered as a valid JSON list of strings (e.g., `[\"Variation1\", \"V2\"]`).")

    # Use st.data_editor for interactive editing
    edited_df = st.data_editor(
        df_config,
        key="personnel_editor",
        num_rows="dynamic", # Allow adding/deleting rows
        use_container_width=True,
        # Configure columns for better editing experience
        column_config={
            "Canonical Name": st.column_config.TextColumn(required=True, help="The main identifier for the person."),
            "Role": st.column_config.SelectboxColumn("Role", options=["physicist", "physician", "admin", "other", ""], help="Assign a role.", default=""),
            "Clinical %": st.column_config.NumberColumn("Clinical %", min_value=0.0, max_value=1.0, step=0.05, format="%.2f", help="Clinical time percentage (0.0 to 1.0)."),
            "Variations (JSON list)": st.column_config.TextColumn(required=True, help="Enter names/aliases as a JSON list: [\"Name\", \"Alias\"]", default="[]"),
        },
        # Hide pandas index
        hide_index=True,
    )

    # --- Process Edits and Save ---
    if st.button("Save Changes", key="save_changes"):
        # Convert edited DataFrame back to the dictionary format
        new_config_dict = {}
        validation_errors = []
        processed_names = set()

        for index, row in edited_df.iterrows():
            name = row['Canonical Name']
            if not name or not isinstance(name, str) or not name.strip():
                 validation_errors.append(f"Row {index+1}: Canonical Name cannot be empty.")
                 continue
            name = name.strip() # Use stripped name

            # Check for duplicate canonical names introduced during editing
            if name in processed_names:
                validation_errors.append(f"Row {index+1}: Duplicate Canonical Name '{name}'. Names must be unique.")
                continue
            processed_names.add(name)

            role = row['Role'] if row['Role'] else None # Handle empty selection
            clinical_pct = row['Clinical %']

            # Validate variations JSON
            variations_str = row['Variations (JSON list)']
            variations = []
            try:
                parsed_variations = json.loads(variations_str)
                if not isinstance(parsed_variations, list):
                    raise ValueError("Input must be a JSON list.")
                # Ensure all items in list are strings
                variations = [str(v).strip() for v in parsed_variations if isinstance(v, (str, int, float)) and str(v).strip()] # Allow numbers convert to str
                if not variations: # If list is empty or only contained empty strings
                     variations = [name] # Default variation to the name itself if none provided
                     st.toast(f"Warning: No valid variations provided for '{name}'. Defaulting variation to canonical name.", icon="⚠️")

            except Exception as json_e:
                validation_errors.append(f"Row {index+1} ('{name}'): Invalid JSON format for Variations. Error: {json_e}")
                variations = [name] # Default variation on error

            # Build the entry for the new config dict
            new_config_dict[name] = {
                'role': role,
                'clinical_pct': clinical_pct,
                'variations': variations
            }

        # If validation passed, save the new config
        if not validation_errors:
            try:
                save_success = config_manager.save_personnel_config(new_config_dict)
                if save_success:
                    st.success("Personnel configuration saved successfully!")
                    
                    # Also save to PostgreSQL database if enabled
                    if settings.DB_ENABLED:
                        from functions import db as db_manager # Correct import for db package
                        from config import settings
                        try:
                            db_save_success = db_manager.save_personnel_config_to_db(new_config_dict)
                            if db_save_success:
                                st.success(f"Also saved personnel configuration to PostgreSQL database at {settings.DB_HOST}.")
                                logger.info(f"Successfully saved personnel configuration to database.")
                            else:
                                st.warning("Failed to save personnel configuration to PostgreSQL database. Check logs for details.")
                                logger.warning("Database personnel save operation returned False.")
                        except Exception as db_e:
                            st.error(f"Error saving personnel to database: {db_e}")
                            logger.error(f"Database personnel save error: {db_e}", exc_info=True)
                    
                    # Update the working copy in session state to reflect saved changes
                    st.session_state.admin_personnel_config = new_config_dict.copy()
                    # Force reload of config in session state for other pages (might need refresh)
                    p_config, v_map, c_names = config_manager.load_personnel_config(force_reload=True)
                    st.session_state.personnel_config = p_config
                    st.session_state.variation_map = v_map
                    st.session_state.canonical_names = c_names
                    st.rerun() # Rerun page to reflect saved state
                else:
                    st.error("Failed to save configuration to file. Check logs.")
            except Exception as save_e:
                st.error(f"An error occurred while saving: {save_e}")
                logger.error(f"Error saving personnel config: {save_e}", exc_info=True)
        else:
            st.error("Validation Errors Found:")
            for error in validation_errors:
                st.error(f"- {error}")


# --- Display Derived Maps (Read-only) ---
st.markdown("---")
st.subheader("Derived Configuration (Read-Only)")
col1, col2 = st.columns(2)
with col1:
    st.write("**Canonical Names List:**")
    st.json(canonical_names, expanded=False)
with col2:
     st.write("**Variation Map (Lowercase Variation -> Canonical Name):**")
     st.json(variation_map, expanded=False)

# Reminder Note
st.markdown("---")
st.info("💡 **Note:** After saving changes, the application might need a moment to update. Reloading the config should apply changes for subsequent processing runs.")

# --- LLM Configuration Section ---
st.markdown("---")
st.subheader("🤖 LLM Model Configuration")
st.markdown("Configure the Large Language Model (LLM) settings for extraction tasks.")

# Add section to select Ollama model
st.markdown("### Ollama Model Selection")
st.markdown("Choose which model to use for LLM extraction tasks. Smaller models are faster but may be less accurate.")

# Import the function to get available models
from functions.llm_extraction.ollama_client import get_available_models, is_ollama_ready

# Check if Ollama is available
ollama_available = is_ollama_ready()

if not ollama_available:
    st.error("⚠️ Ollama service is not available. Please check that the Ollama server is running and properly configured.")
else:
    st.success("✅ Ollama service is available and responding.")
    
    # Get current model from settings
    current_model = settings.LLM_MODEL
    st.write(f"**Current model:** {current_model}")
      # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.warning("Could not retrieve available models from Ollama server. Using default models instead.")
        # Add default models based on the list you provided
        model_options = [
            current_model,
            "mistral:latest",
            "mistral-openorca:latest",
            "llama3.1:8b",
            "starcoder2:3b",
            "deepseek-coder:33b",
            "codellama:70b-code",
            "deepcoder:latest"
        ]
    else:
        st.write(f"Found {len(available_models)} available models.")
        model_options = available_models
    
    # Model selection dropdown
    selected_model = st.selectbox(
        "Select Ollama Model",
        options=model_options,
        index=model_options.index(current_model) if current_model in model_options else 0,
        help="Select which Ollama model to use for extraction tasks. Smaller models are faster but may be less accurate."
    )
    
    # Update model selection if changed
    if selected_model != current_model:
        if st.button("Apply Model Change"):
            # In a real implementation, we would update the settings
            # Here we're using session state as a temporary solution
            if 'llm_model' not in st.session_state:
                st.session_state.llm_model = current_model
                
            st.session_state.llm_model = selected_model
            
            # Update the LLM_MODEL setting
            # This is temporary - will reset when app restarts
            settings.LLM_MODEL = selected_model
            
            st.success(f"Model changed to {selected_model}. This change is temporary and will reset when the application restarts.")
            st.info("To make this change permanent, you'll need to update the LLM_MODEL in your environment variables or settings.py file.")

# Display additional LLM settings
st.markdown("### Other LLM Settings")
st.write(f"**Base URL:** {settings.OLLAMA_BASE_URL}")

# Add configurable Max Workers setting
current_max_workers = settings.LLM_MAX_WORKERS
st.write("**Max Workers:** Number of parallel extraction tasks. Higher values may improve speed but require more resources.")

# Create a number input for Max Workers
new_max_workers = st.number_input(
    "Max Workers",
    min_value=1,
    max_value=16,  # Set a reasonable upper limit
    value=current_max_workers,
    step=1,
    help="Number of concurrent LLM extraction operations. Increase for faster processing if your hardware supports it. Recommended values: 3-8 for large models, 8-16 for small models."
)

# Button to apply the change
if new_max_workers != current_max_workers:
    if st.button("Apply Max Workers Change"):
        # Update the setting in memory
        settings.LLM_MAX_WORKERS = new_max_workers
        
        # Store in session state to maintain during session
        if 'llm_max_workers' not in st.session_state:
            st.session_state.llm_max_workers = current_max_workers
        
        st.session_state.llm_max_workers = new_max_workers
        
        st.success(f"Max Workers changed from {current_max_workers} to {new_max_workers}. This change is temporary and will reset when the application restarts.")
        st.info("To make this change permanent, update the LLM_MAX_WORKERS in your environment variables or settings.py file.")

# Performance recommendations
st.markdown("#### 🚀 Performance Recommendations")
st.markdown("""
- **Small models** (1-5GB): Can typically use 8-16 workers
- **Medium models** (5-15GB): Can typically use 4-8 workers
- **Large models** (15GB+): Recommended to use 1-4 workers
- Increase workers gradually and monitor performance
- Reduce workers if you encounter memory errors or slow responses
""")

# Add detailed hardware-specific recommendations
st.markdown("---")
st.subheader("💻 Hardware-Optimized LLM Guidelines")
st.markdown("Recommendations for Threadripper 5955WX + 256GB RAM + Dual RTX 3090 Setup")

# Create a table with model-specific recommendations
guidelines_data = {
    "Model": [
        "mistral:latest (4.1GB)", 
        "llama3.1:8b (4.7GB)", 
        "starcoder2:3b (1.7GB)",
        "deepseek-coder:33b (19GB)",
        "deepcoder:latest (9.0GB)",
        "codellama:70b-code (38GB)",
        "qwen2.5:32b (19GB)",
        "llama3.3:70b (42GB)",
        "mixtral:latest (26GB)",
        "gemma3:27b (17GB)"
    ],
    "Max Workers": [
        "12-16", 
        "10-14", 
        "14-16",
        "3-5",
        "6-10",
        "1-3",
        "3-5",
        "1-2",
        "2-4",
        "3-6"
    ],
    "Speed": [
        "Very Fast", 
        "Fast", 
        "Very Fast",
        "Moderate",
        "Fast",
        "Slow",
        "Moderate",
        "Very Slow",
        "Slow",
        "Moderate"
    ],
    "Extraction Quality": [
        "Good", 
        "Good", 
        "Moderate",
        "Very Good",
        "Good",
        "Excellent",
        "Very Good",
        "Excellent",
        "Very Good",
        "Very Good"
    ],
    "Notes": [
        "Great balance of speed and accuracy", 
        "Good for general extraction tasks", 
        "Best for quick testing, less accurate",
        "Good for complex extractions",
        "Good for code-related extractions",
        "High quality but very resource intensive",
        "Well-balanced for extraction tasks",
        "Highest quality but slowest performance",
        "Good for handling complex context",
        "Current default, balanced performance"
    ]
}

import pandas as pd
guidelines_df = pd.DataFrame(guidelines_data)
st.table(guidelines_df)

st.markdown("""
### Hardware Utilization Tips

- **Dual RTX 3090 (24GB each)**: 
  - For models under 20GB, you can run entirely on a single GPU
  - For 40GB+ models, utilizes both GPUs with NVIDIA NVLink
  - Optimal batch size increases with smaller models

- **256GB RAM**: 
  - Allows aggressive worker count with small/medium models
  - Can handle multiple model instances in memory simultaneously
  - Enables high worker count (12-16) with small models

- **Threadripper 5955WX**:
  - 32 threads enable excellent parallel processing
  - Can handle high worker counts without CPU bottlenecks
  - Allows for fast preprocessing while models run on GPUs

- **Optimization Strategy**:
  - For small/medium models (≤10GB): Maximize workers (10-16)
  - For large models (10-25GB): Moderate workers (3-8)
  - For very large models (>25GB): Limited workers (1-4)
  - Balance between speed and accuracy based on your task requirements
""")

logger.info("Admin page processed.")
