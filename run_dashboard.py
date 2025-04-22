"""
Calendar Workload Analyzer - Application Launcher
This script properly sets up the Python path and runs the Streamlit application.
"""
import os
import sys
import subprocess
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.absolute()

# Ensure the project root is in the Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added {project_root} to Python path")

# Define the path to the Streamlit app
streamlit_app_path = os.path.join(project_root, "dashboard", "streamlit_app.py")

# Run the Streamlit application
print("Starting Calendar Workload Analyzer...")
print(f"Using application file: {streamlit_app_path}")
subprocess.run(["streamlit", "run", streamlit_app_path], check=True)
