"""
Authentication utilities for the admin dashboard.
"""
import os
import json
import hashlib
import time
import streamlit as st
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Path to the admin configuration file
ADMIN_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                "config", "admin_config.json")

def hash_password(password):
    """
    Simple password hashing function.
    In a production environment, use a more secure method like bcrypt.
    """
    return hashlib.sha256(password.encode()).hexdigest()

def load_admin_config():
    """
    Load admin configuration from JSON file.
    Creates a default configuration if file doesn't exist.
    """
    default_config = {
        "admins": [
            {
                "username": "admin",
                "password_hash": hash_password("admin123"),  # Default password
                "role": "administrator"
            }
        ],
        "session_timeout_minutes": 60  # 1 hour timeout
    }
    
    try:
        if os.path.exists(ADMIN_CONFIG_PATH):
            with open(ADMIN_CONFIG_PATH, 'r') as f:
                return json.load(f)
        else:
            # Create default configuration
            os.makedirs(os.path.dirname(ADMIN_CONFIG_PATH), exist_ok=True)
            with open(ADMIN_CONFIG_PATH, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Created default admin configuration at {ADMIN_CONFIG_PATH}")
            return default_config
    except Exception as e:
        logger.error(f"Error loading admin configuration: {e}")
        return default_config

def verify_credentials(username, password):
    """
    Verify admin credentials against stored configuration.
    """
    config = load_admin_config()
    password_hash = hash_password(password)
    
    for admin in config.get("admins", []):
        if admin.get("username") == username and admin.get("password_hash") == password_hash:
            return True, admin.get("role", "administrator")
    
    return False, None

def check_authentication():
    """
    Check if user is authenticated.
    Returns True if authenticated, False otherwise.
    """
    # Initialize session state variables if they don't exist
    if 'admin_authenticated' not in st.session_state:
        st.session_state.admin_authenticated = False
    if 'admin_username' not in st.session_state:
        st.session_state.admin_username = None
    if 'admin_role' not in st.session_state:
        st.session_state.admin_role = None
    if 'admin_login_time' not in st.session_state:
        st.session_state.admin_login_time = None
    
    # Check for session timeout
    if st.session_state.admin_authenticated and st.session_state.admin_login_time:
        config = load_admin_config()
        timeout_minutes = config.get("session_timeout_minutes", 60)
        current_time = time.time()
        elapsed_minutes = (current_time - st.session_state.admin_login_time) / 60
        
        if elapsed_minutes > timeout_minutes:
            logger.info(f"Admin session timed out after {elapsed_minutes:.1f} minutes")
            logout_user()
            return False
    
    return st.session_state.admin_authenticated

def login_user(username, password):
    """
    Attempt to log in a user.
    Returns True if login successful, False otherwise.
    """
    is_valid, role = verify_credentials(username, password)
    
    if is_valid:
        st.session_state.admin_authenticated = True
        st.session_state.admin_username = username
        st.session_state.admin_role = role
        st.session_state.admin_login_time = time.time()
        logger.info(f"Admin login successful: {username} ({role})")
        return True
    else:
        logger.warning(f"Failed login attempt for username: {username}")
        return False

def logout_user():
    """
    Log out the current user by clearing session state.
    """
    if st.session_state.admin_authenticated:
        logger.info(f"Admin logout: {st.session_state.admin_username}")
    
    st.session_state.admin_authenticated = False
    st.session_state.admin_username = None
    st.session_state.admin_role = None
    st.session_state.admin_login_time = None

def show_login_page():
    """
    Display the login page.
    This is a simplified version - the actual login page will be in its own file.
    """
    st.title("⚙️ Admin Dashboard Login")
    
    with st.form("admin_login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Log In")
        
        if submit_button:
            if login_user(username, password):
                st.success("Login successful! Redirecting...")
                st.rerun()
            else:
                st.error("Invalid username or password")
                
    st.markdown("---")
    st.markdown("© 2025 Calendar Workload Analyzer - Admin Dashboard")
