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

logger.info("Admin page processed.")

# Reminder Note
st.markdown("---")
st.info("💡 **Note:** After saving changes, the application might need a moment to update. Reloading the config should apply changes for subsequent processing runs.")