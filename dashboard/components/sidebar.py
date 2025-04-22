"""
Sidebar component for the admin dashboard.
"""
import streamlit as st
import logging
from admin_dashboard.utils.auth import logout_user

logger = logging.getLogger(__name__)

def show_sidebar():
    """
    Display the sidebar navigation for the admin dashboard.
    """
    with st.sidebar:
        st.title("Admin Navigation")
        
        # Display logged in user info
        if st.session_state.admin_username:
            st.write(f"Logged in as: **{st.session_state.admin_username}**")
            st.write(f"Role: **{st.session_state.admin_role}**")
        
        st.markdown("---")
        
        # Navigation menu
        st.subheader("Menu")
        
        if st.sidebar.button("📊 Dashboard", use_container_width=True):
            st.session_state.admin_current_page = "dashboard"
            st.rerun()
            
        if st.sidebar.button("📈 Analysis", use_container_width=True):
            st.session_state.admin_current_page = "analysis"
            st.rerun()
            
        if st.sidebar.button("✏️ Manual Assignment", use_container_width=True):
            st.session_state.admin_current_page = "manual_assignment"
            st.rerun()
            
        if st.sidebar.button("⚙️ Settings", use_container_width=True):
            st.session_state.admin_current_page = "settings"
            st.rerun()
            
        st.markdown("---")
        
        # Logout button
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout_user()
            st.rerun()
            
        # Return to main app
        if st.sidebar.button("🏠 Return to Main App", use_container_width=True):
            # We'll handle this in the main app.py file
            pass
            
        # Footer
        st.markdown("---")
        st.markdown("v1.0.0")
        st.markdown("© 2025 Calendar Analyzer")
