"""
Simplified LLM client module - Ollama-only version for the simplified branch.
"""
import logging
import streamlit as st
import requests

# Import settings
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Check if ollama library is available
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

def is_ollama_ready():
    """
    Simple check to verify if Ollama server is available and responding.
    
    Returns:
        bool: True if Ollama server is ready, False otherwise
    """
    if not OLLAMA_AVAILABLE:
        logger.error("Ollama library not installed")
        return False
    
    try:
        # Direct simple HTTP connectivity test
        base_url = settings.OLLAMA_BASE_URL.rstrip('/')
        check_url = f"{base_url}/"
        
        logger.debug(f"Testing Ollama connection at: {check_url}")
        
        response = requests.get(check_url, timeout=5)
        if response.status_code == 200:
            logger.info(f"Ollama server is responding at {check_url}")
            return True
        else:
            logger.error(f"Ollama server returned status code {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error checking Ollama server: {e}")
        return False

@st.cache_resource  # Cache the client resource
def get_ollama_client():
    """
    Initializes and returns the Ollama client.
    Uses Streamlit caching to avoid reinitialization on every call.
    
    Returns:
        ollama.Client or None: Initialized client or None if not available
    """
    if not OLLAMA_AVAILABLE:
        error_msg = "Ollama library not installed. Please install with 'pip install ollama'"
        logger.error(error_msg)
        st.error(f"{error_msg}\n\nRun this command in your terminal:\n```\npip install ollama\n```\n\nThen restart the application.", icon="🚨")
        return None
            
    try:
        # Get Ollama base URL from settings
        base_url = settings.OLLAMA_BASE_URL
        
        # Create and test the client with a simple request
        logger.debug(f"Initializing Ollama client with URL: {base_url}")
        client = ollama.Client(host=base_url)
        
        # Simple connection test without accessing model information
        try:
            response = requests.get(f"{base_url.rstrip('/')}/", timeout=5)
            if response.status_code == 200:
                logger.info(f"Successfully connected to Ollama server at {base_url}")
                return client
            else:
                logger.error(f"Ollama server returned status code {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating Ollama client: {e}")
        return None

def extract_personnel(summary, canonical_names=None):
    """
    Extract personnel names from event summary using the Ollama LLM.
    
    Args:
        summary: The calendar event summary text
        canonical_names: List of known canonical names to look for
        
    Returns:
        list: Extracted personnel names or ["Unknown"] if extraction fails
    """
    client = get_ollama_client()
    if not client:
        logger.warning("Ollama client unavailable, returning Unknown")
        return ["Unknown"]
    
    try:
        # Use a simple prompt for extraction
        canonical_list = ", ".join(canonical_names) if canonical_names else "N/A"
        
        prompt = f"""
        Extract personnel names from this calendar event summary: "{summary}"
        
        Known personnel names: {canonical_list}
        
        Return ONLY the names found in the summary, one per line.
        If no names are found, return "Unknown".
        """
          # Use the Ollama client with chat API for better model compatibility
        response = client.chat(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "Extract personnel names from calendar events."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.1}
        )
        
        if not response or not response.get('message', {}).get('content'):
            logger.warning("Empty response from Ollama")
            return ["Unknown"]
        
        # Process the response from chat API
        names = response['message']['content'].strip().split('\n')
        # Filter out empty lines and generic responses
        names = [name.strip() for name in names if name.strip() and name.strip().lower() not in ['unknown', 'n/a', 'none']]
        
        if not names:
            return ["Unknown"]
        
        return names
        
    except Exception as e:
        logger.error(f"Error extracting personnel with Ollama: {e}")
        return ["Unknown_Error"]
