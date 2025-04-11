# functions/llm_extraction/client.py
"""
Client module for handling Ollama LLM connections and server management.
"""
import logging
import time
import streamlit as st
import subprocess
import platform
from urllib.parse import urlparse
import requests

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Import settings from the main config
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

@st.cache_resource  # Cache the client resource
def get_llm_client():
    """
    Initializes and returns the Ollama client, or None if unavailable.
    Uses Streamlit caching to avoid reinitialization on every call.
    """
    if not OLLAMA_AVAILABLE:
        error_msg = "Ollama library not installed. Please install with 'pip install ollama'"
        logger.error(error_msg)
        st.error(f"{error_msg}\n\nRun this command in your terminal:\n```\npip install ollama\n```\n\nThen restart the application.", icon="🚨")
        return None
        
    if settings.LLM_PROVIDER == "ollama":
        try:
            client = ollama.Client(host=settings.OLLAMA_BASE_URL)
            # Perform a quick check to see if the client can connect
            client.list()  # Throws error if server not reachable
            logger.info(f"Successfully connected to Ollama server at {settings.OLLAMA_BASE_URL}")
            return client
        except Exception as e:
            logger.error(f"Error initializing or connecting to Ollama client at {settings.OLLAMA_BASE_URL}: {e}")
            st.error(f"Could not connect to Ollama server at {settings.OLLAMA_BASE_URL}. Check if it's running. LLM features disabled.", icon="🚨")
            return None
    else:
        logger.warning(f"LLM_PROVIDER is not 'ollama' (current: {settings.LLM_PROVIDER}). Ollama client not created.")
        return None


def restart_ollama_server():
    """
    Attempts to restart the Ollama server to resolve memory issues.
    Returns True if successful, False otherwise.
    """
    logger.warning("Attempting to restart Ollama server due to performance degradation...")
    try:
        # For local installs, try sending restart command
        # Determine the command based on the platform
        if platform.system() == "Windows":
            # For Windows - try to stop and start the service
            try:
                # Check if it's running via the API endpoint
                # Parse the base URL to get host and port
                parsed_url = urlparse(settings.OLLAMA_BASE_URL)
                host = parsed_url.hostname or "localhost"
                port = parsed_url.port or 11434
                
                # Try to gracefully stop the server by sending a request
                shutdown_url = f"http://{host}:{port}/api/shutdown"
                requests.post(shutdown_url, timeout=5)
                logger.info("Sent shutdown request to Ollama server")
                time.sleep(3)  # Wait for the server to shutdown
                
                # For Windows, try to restart by launching a new process
                # Attempt to start Ollama with the serve command
                subprocess.Popen(["ollama", "serve"], 
                                shell=True, 
                                creationflags=subprocess.CREATE_NO_WINDOW)
                logger.info("Started new Ollama server process")
                time.sleep(5)  # Give the server time to start
                
                # Verify the server is responding
                check_url = f"http://{host}:{port}/api/health"
                response = requests.get(check_url, timeout=5)
                if response.status_code == 200:
                    logger.info("Ollama server restarted successfully")
                    # Clear client cache to force reconnection
                    get_llm_client.clear()
                    return True
            except Exception as e:
                logger.error(f"Failed to restart Ollama using Windows-specific method: {e}")
                
            # If the above fails, try a more direct approach
            try:
                # Kill any running ollama processes 
                subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], shell=True)
                time.sleep(2)
                # Start a new ollama process
                subprocess.Popen(["ollama", "serve"], 
                                shell=True, 
                                creationflags=subprocess.CREATE_NO_WINDOW)
                logger.info("Forcefully restarted Ollama server")
                time.sleep(5)  # Give it time to start
                return True
            except Exception as e:
                logger.error(f"Failed to forcefully restart Ollama on Windows: {e}")
        
        elif platform.system() == "Linux":
            # For Linux - try systemctl if installed as a service
            try:
                subprocess.run(["systemctl", "restart", "ollama"], check=True)
                logger.info("Restarted Ollama via systemctl")
                time.sleep(5)  # Give it time to restart
                return True
            except Exception as e:
                logger.error(f"Failed to restart Ollama via systemctl: {e}")
                
                # Try direct process management if systemctl fails
                try:
                    subprocess.run(["pkill", "ollama"])
                    time.sleep(2)
                    subprocess.Popen(["ollama", "serve"])
                    logger.info("Restarted Ollama via process management")
                    time.sleep(5)  # Give it time to start
                    return True
                except Exception as e2:
                    logger.error(f"Failed to restart Ollama via process management: {e2}")
        
        elif platform.system() == "Darwin":  # macOS
            # For macOS - try stopping and starting
            try:
                subprocess.run(["pkill", "ollama"])
                time.sleep(2)
                subprocess.Popen(["ollama", "serve"])
                logger.info("Restarted Ollama on macOS")
                time.sleep(5)  # Give it time to start
                return True
            except Exception as e:
                logger.error(f"Failed to restart Ollama on macOS: {e}")
                
        # If we reach here, none of the platform-specific methods worked
        # Try to send a restart command to the API directly as a last resort
        try:
            # Parse the base URL to get host and port
            parsed_url = urlparse(settings.OLLAMA_BASE_URL)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 11434
            
            # Try to restart via the API
            restart_url = f"http://{host}:{port}/api/restart"
            requests.post(restart_url, timeout=5)
            logger.info("Sent restart command to Ollama API")
            time.sleep(5)  # Give it time to restart
            
            # Verify the server is responding
            check_url = f"http://{host}:{port}/api/health"
            response = requests.get(check_url, timeout=5)
            if response.status_code == 200:
                logger.info("Ollama API restarted successfully")
                # Clear client cache to force reconnection
                get_llm_client.clear()
                return True
        except Exception as e:
            logger.error(f"Failed to restart Ollama via API: {e}")
        
        # If all methods fail, we can't restart
        logger.error("All methods to restart Ollama failed")
        return False
        
    except Exception as e:
        logger.error(f"General error trying to restart Ollama server: {e}")
        return False


def is_llm_ready():
    """
    Check if the LLM client is available and ready to use.
    Returns True if the client is available, False otherwise.
    """
    client = get_llm_client()
    return client is not None