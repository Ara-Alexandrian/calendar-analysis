"""
Settings page for the admin dashboard.
"""
import streamlit as st
import os
import json
import logging
from config import settings
from dashboard.utils.auth import load_admin_config, hash_password # Updated import path
from src.infrastructure.llm.factory import LLMClientFactory # Import the factory

logger = logging.getLogger(__name__)

def show_settings_page():
    """
    Display the settings page for administrators.
    """
    st.title("⚙️ Dashboard Settings")
    st.markdown("Configure administrator dashboard settings and preferences.")

    # Tabs for different settings categories
    tab1, tab2, tab3 = st.tabs(["User Management", "Display Settings", "System Configuration"])

    with tab1:
        show_user_management()

    with tab2:
        show_display_settings()

    with tab3:
        show_system_configuration()

def show_user_management():
    """
    Display user management settings.
    """
    st.subheader("User Management")
    st.markdown("Manage administrator accounts.")

    # Load current admin configuration
    admin_config = load_admin_config()
    admins = admin_config.get("admins", [])

    # Display current users
    st.markdown("### Current Administrators")

    if not admins:
        st.info("No administrators configured. Using default admin account.")
    else:
        # Create a table of administrators
        admin_data = []
        for admin in admins:
            admin_data.append({
                "Username": admin.get("username", ""),
                "Role": admin.get("role", "administrator")
            })

        st.table(admin_data)

    # Add new administrator
    st.markdown("### Add New Administrator")

    with st.form("add_admin_form"):
        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        new_role = st.selectbox("Role", ["administrator", "analyst"])

        submit_button = st.form_submit_button("Add Administrator")

        if submit_button:
            if not new_username or not new_password:
                st.error("Username and password are required.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif any(admin.get("username") == new_username for admin in admins):
                st.error(f"Administrator with username '{new_username}' already exists.")
            else:
                # Add new administrator
                try:
                    new_admin = {
                        "username": new_username,
                        "password_hash": hash_password(new_password),
                        "role": new_role
                    }

                    admins.append(new_admin)
                    admin_config["admins"] = admins

                    # Save the updated configuration
                    admin_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                   "config", "admin_config.json")

                    os.makedirs(os.path.dirname(admin_config_path), exist_ok=True)
                    with open(admin_config_path, 'w') as f:
                        json.dump(admin_config, f, indent=4)

                    st.success(f"Administrator '{new_username}' added successfully.")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error adding administrator: {e}")
                    st.error(f"Error adding administrator: {e}")

    # Remove administrator
    st.markdown("### Remove Administrator")

    if admins:
        admin_to_remove = st.selectbox(
            "Select Administrator to Remove",
            options=[admin.get("username") for admin in admins]
        )

        if st.button("Remove Administrator"):
            # Assuming admin_username is stored in session_state after login
            current_user = st.session_state.get("admin_username", None)
            if admin_to_remove == current_user:
                st.error("You cannot remove your own account.")
            else:
                try:
                    # Remove the administrator
                    admin_config["admins"] = [admin for admin in admins if admin.get("username") != admin_to_remove]

                    # Save the updated configuration
                    admin_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                                   "config", "admin_config.json")

                    with open(admin_config_path, 'w') as f:
                        json.dump(admin_config, f, indent=4)

                    st.success(f"Administrator '{admin_to_remove}' removed successfully.")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error removing administrator: {e}")
                    st.error(f"Error removing administrator: {e}")

def show_display_settings():
    """
    Display appearance and display settings.
    """
    st.subheader("Display Settings")
    st.markdown("Configure appearance and display options for the admin dashboard.")

    # Theme settings
    st.markdown("### Theme Settings")

    # Initialize session state for settings if not exists
    if 'admin_theme' not in st.session_state:
        st.session_state.admin_theme = "light"
    if 'admin_color_scheme' not in st.session_state:
        st.session_state.admin_color_scheme = "blue"
    if 'admin_sidebar_collapsed' not in st.session_state:
        st.session_state.admin_sidebar_collapsed = False

    # Theme selector
    selected_theme = st.radio(
        "Dashboard Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state.admin_theme == "light" else 1,
        horizontal=True
    )

    if selected_theme == "Light" and st.session_state.admin_theme != "light":
        st.session_state.admin_theme = "light"
        st.success("Theme updated to Light mode. Changes will apply on next page load.")
    elif selected_theme == "Dark" and st.session_state.admin_theme != "dark":
        st.session_state.admin_theme = "dark"
        st.success("Theme updated to Dark mode. Changes will apply on next page load.")

    # Color scheme selector
    selected_color = st.selectbox(
        "Color Scheme",
        options=["Blue", "Green", "Purple", "Orange"],
        index=["blue", "green", "purple", "orange"].index(st.session_state.admin_color_scheme)
    )

    if selected_color.lower() != st.session_state.admin_color_scheme:
        st.session_state.admin_color_scheme = selected_color.lower()
        st.success(f"Color scheme updated to {selected_color}. Changes will apply on next page load.")

    # Sidebar collapse setting
    sidebar_collapsed = st.checkbox(
        "Start with sidebar collapsed",
        value=st.session_state.admin_sidebar_collapsed
    )

    if sidebar_collapsed != st.session_state.admin_sidebar_collapsed:
        st.session_state.admin_sidebar_collapsed = sidebar_collapsed
        st.success(f"Sidebar collapse setting updated. Changes will apply on next page load.")

    # Chart settings
    st.markdown("### Chart Settings")

    # Default chart type
    chart_types = ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"]
    selected_chart = st.selectbox(
        "Default Chart Type",
        options=chart_types,
        index=chart_types.index(st.session_state.get('default_chart_type', "Bar Chart"))
    )

    if selected_chart != st.session_state.get('default_chart_type', "Bar Chart"):
        st.session_state.default_chart_type = selected_chart
        st.success(f"Default chart type updated to {selected_chart}.")

    # Default color palette
    color_palettes = ["Plotly", "Viridis", "Blues", "Reds", "Greens"]
    selected_palette = st.selectbox(
        "Default Color Palette",
        options=color_palettes,
        index=color_palettes.index(st.session_state.get('default_color_palette', "Plotly"))
    )

    if selected_palette != st.session_state.get('default_color_palette', "Plotly"):
        st.session_state.default_color_palette = selected_palette
        st.success(f"Default color palette updated to {selected_palette}.")

    # Save settings button
    if st.button("Save Display Settings"):
        st.success("Display settings saved. Some changes will apply on next page load.")

def show_system_configuration():
    """
    Display system configuration settings.
    """
    st.subheader("System Configuration")
    st.markdown("Configure system behavior and integration settings.")

    # Session timeout setting
    st.markdown("### Session Settings")

    # Load current admin configuration
    admin_config = load_admin_config()
    current_timeout = admin_config.get("session_timeout_minutes", 60)

    # Session timeout slider
    new_timeout = st.slider(
        "Session Timeout (minutes)",
        min_value=5,
        max_value=240,
        value=current_timeout,
        step=5
    )

    if new_timeout != current_timeout:
        try:
            # Update the configuration
            admin_config["session_timeout_minutes"] = new_timeout

            # Save the updated configuration
            admin_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                           "config", "admin_config.json")

            with open(admin_config_path, 'w') as f:
                json.dump(admin_config, f, indent=4)

            st.success(f"Session timeout updated to {new_timeout} minutes.")
        except Exception as e:
            logger.error(f"Error updating session timeout: {e}")
            st.error(f"Error updating session timeout: {e}")

    # Database settings
    st.markdown("### Database Settings")

    # Display current database settings (read-only)
    st.markdown("**Current Database Configuration**")
    st.markdown(f"- **Host**: {settings.DB_HOST}")
    st.markdown(f"- **Database**: {settings.DB_NAME}")
    st.markdown(f"- **Username**: {settings.DB_USER}")
    st.markdown(f"- **Enabled**: {'Yes' if settings.DB_ENABLED else 'No'}")

    st.markdown("*Note: Database settings can only be changed in the application configuration files.*")

    # LLM settings
    st.markdown("### LLM Settings")

    # Display current LLM settings (read-only)
    st.markdown("**Current LLM Configuration**")
    st.markdown(f"- **Provider**: {settings.LLM_PROVIDER}")
    st.markdown(f"- **Model**: {settings.LLM_MODEL}")
    st.markdown(f"- **Base URL**: {settings.OLLAMA_BASE_URL}")

    st.markdown("*Note: LLM settings can only be changed in the application configuration files.*")

    # Restart LLM button
    if st.button("Restart LLM Service"):
        try:
            # from functions.llm_extraction.client import restart_ollama_server # Old import
            client = LLMClientFactory.get_client() # Get client
            if client and hasattr(client, 'restart_server'): # Check if client exists and has method
                result = client.restart_server() # Call method on client
                if result:
                    st.success("LLM service restarted successfully.")
                else:
                    st.error("Failed to restart LLM service (client method returned False)")
            elif client:
                st.error("LLM client does not support restarting.")
            else:
                st.error("Failed to restart LLM service.") # Changed error message
        except Exception as e:
            logger.error(f"Error restarting LLM service: {e}")
            st.error(f"Error restarting LLM service: {e}")

    # Log file viewer
    st.markdown("### Log Files")

    # Display log file path
    st.markdown(f"**Log File**: {settings.LOG_FILE}")

    # Button to view recent logs
    if st.button("View Recent Logs"):
        try:
            with open(settings.LOG_FILE, 'r') as f:
                # Get the last 100 lines of the log file
                log_lines = f.readlines()[-100:]
                log_content = ''.join(log_lines)

                # Display the log content
                st.text_area("Recent Log Entries", log_content, height=300)
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
            st.error(f"Error reading log file: {e}")
