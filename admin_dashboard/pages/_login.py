"""
Login page for the admin dashboard.
"""
import streamlit as st
import logging
from admin_dashboard.utils.auth import login_user

logger = logging.getLogger(__name__)

def show_login_page():
    """
    Display the admin login page with enhanced styling.
    """
    st.title("⚙️ Calendar Analyzer Admin Dashboard")
    
    # Container for login form with custom styling
    login_container = st.container()
    
    with login_container:
        st.markdown("### Administrator Login")
        st.markdown("Please enter your credentials to access the admin dashboard.")
        
        # Login form
        with st.form("admin_login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                submit_button = st.form_submit_button("Log In", use_container_width=True)
            
            if submit_button:
                if username and password:                    
                    if login_user(username, password):
                        st.success("Login successful! Redirecting...")
                        st.session_state.remember_admin = remember_me
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
    
    # Add some information about the admin dashboard
    with st.expander("About the Admin Dashboard"):
        st.markdown("""
        The Calendar Analyzer Admin Dashboard provides advanced capabilities for administrators:
        
        - **Enhanced Analytics**: Access comprehensive data analysis views
        - **Manual Data Management**: Assign personnel and event types to calendar entries
        - **System Administration**: Configure application settings and manage users
        
        For assistance, please contact your system administrator.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("© 2025 Calendar Workload Analyzer - Admin Dashboard")
