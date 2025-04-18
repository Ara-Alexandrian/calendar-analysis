"""
Ollama client implementation for the Calendar Analysis application.

This module provides a concrete implementation of the BaseLLMClient
interface for the Ollama LLM provider.
"""
import logging
import time
from typing import Dict, List, Optional, Any, Union

import requests
from urllib.parse import urlparse
import streamlit as st

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from src.infrastructure.llm.base_client import BaseLLMClient
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class OllamaClient(BaseLLMClient):
    """
    Ollama client implementation of the BaseLLMClient interface.
    
    Handles communication with the Ollama server for LLM operations.
    """
    
    def __init__(self, host: Optional[str] = None):
        """
        Initialize the Ollama client.
        
        Args:
            host (Optional[str]): The host URL for the Ollama server.
                                  If None, uses the URL from settings.
        """
        self.host = host or settings.OLLAMA_BASE_URL
        self.client = None
        self.model_name = getattr(settings, "OLLAMA_MODEL_NAME", "llama3")
        
        if OLLAMA_AVAILABLE:
            try:
                self.client = ollama.Client(host=self.host)
                # Quick check to see if the client can connect
                self.client.list()
                logger.info(f"Successfully connected to Ollama server at {self.host}")
            except Exception as e:
                logger.error(f"Error initializing or connecting to Ollama client at {self.host}: {e}")
                self.client = None
        else:
            logger.error("Ollama library not installed. Please install with 'pip install ollama'")
    
    def is_available(self) -> bool:
        """
        Check if the Ollama service is available and ready.
        
        Returns:
            bool: True if the service is available, False otherwise.
        """
        if not OLLAMA_AVAILABLE or self.client is None:
            return False
        
        try:
            self.client.list()
            return True
        except Exception as e:
            logger.error(f"Ollama service is not available: {e}")
            return False
    
    def extract_personnel(self, event_description: str) -> Dict[str, Any]:
        """
        Extract personnel information from an event description using Ollama.
        
        Args:
            event_description (str): The calendar event description text.
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted personnel information.
        """
        if not self.is_available():
            logger.error("Ollama service not available for personnel extraction")
            return {"error": "LLM service not available"}
        
        try:
            prompt = f"""Extract all personnel information from this calendar event description.
                      Return JSON format with 'organizer' and 'attendees' fields.
                      
                      Event description:
                      {event_description}
                      
                      Output JSON only with no explanation:"""
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            # Process the response to extract JSON
            from src.infrastructure.llm.utils import extract_json_from_response
            result = extract_json_from_response(response["response"])
            return result
        except Exception as e:
            logger.error(f"Error during personnel extraction: {e}")
            return {"error": str(e)}
    
    def extract_event_type(self, event_description: str) -> Dict[str, Any]:
        """
        Extract event type information from an event description using Ollama.
        
        Args:
            event_description (str): The calendar event description text.
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted event type information.
        """
        if not self.is_available():
            logger.error("Ollama service not available for event type extraction")
            return {"error": "LLM service not available"}
        
        try:
            prompt = f"""Categorize this calendar event into one of the following types:
                      Meeting, Training, Interview, Admin, Break, Travel, or Other.
                      
                      Event description:
                      {event_description}
                      
                      Output JSON only with a 'event_type' field and no explanation:"""
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            # Process the response to extract JSON
            from src.infrastructure.llm.utils import extract_json_from_response
            result = extract_json_from_response(response["response"])
            return result
        except Exception as e:
            logger.error(f"Error during event type extraction: {e}")
            return {"error": str(e)}
    
    def extract_all(self, event_description: str) -> Dict[str, Any]:
        """
        Extract all relevant information from an event description using Ollama.
        
        Args:
            event_description (str): The calendar event description text.
            
        Returns:
            Dict[str, Any]: Dictionary containing all extracted information.
        """
        if not self.is_available():
            logger.error("Ollama service not available for information extraction")
            return {"error": "LLM service not available"}
        
        try:
            prompt = f"""Extract all relevant information from this calendar event.
                      Return JSON format with 'organizer', 'attendees', and 'event_type' fields.
                      Event types should be one of: Meeting, Training, Interview, Admin, Break, Travel, or Other.
                      
                      Event description:
                      {event_description}
                      
                      Output JSON only with no explanation:"""
            
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={"temperature": 0.1}
            )
            
            # Process the response to extract JSON
            from src.infrastructure.llm.utils import extract_json_from_response
            result = extract_json_from_response(response["response"])
            return result
        except Exception as e:
            logger.error(f"Error during information extraction: {e}")
            return {"error": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded Ollama model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        if not self.is_available():
            logger.error("Ollama service not available for model information")
            return {"error": "LLM service not available"}
        
        try:
            models = self.client.list()
            current_model = next((m for m in models["models"] if m["name"] == self.model_name), None)
            
            if current_model:
                return {
                    "name": current_model["name"],
                    "size": current_model.get("size", "Unknown"),
                    "modified_at": current_model.get("modified_at", "Unknown"),
                    "provider": "Ollama"
                }
            else:
                return {
                    "name": self.model_name,
                    "status": "Not loaded",
                    "provider": "Ollama"
                }
        except Exception as e:
            logger.error(f"Error getting model information: {e}")
            return {"error": str(e)}
    
    def restart_server(self) -> bool:
        """
        Attempts to restart the Ollama server to resolve memory issues.
        
        Returns:
            bool: True if successful, False otherwise.
        """
        logger.warning("Attempting to restart Ollama server due to performance degradation...")
        try:
            # Parse the base URL to get host and port
            parsed_url = urlparse(self.host)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 11434
            
            # Try to gracefully stop the server by sending a request
            shutdown_url = f"http://{host}:{port}/api/shutdown"
            requests.post(shutdown_url, timeout=5)
            logger.info("Sent shutdown request to Ollama server")
            time.sleep(3)  # Wait for the server to shutdown
            
            # Wait for the server to come back
            for attempt in range(10):
                try:
                    test_url = f"http://{host}:{port}/api/version"
                    response = requests.get(test_url, timeout=2)
                    if response.status_code == 200:
                        logger.info(f"Ollama server is back online after restart: {response.json()}")
                        
                        # Recreate the client
                        self.client = ollama.Client(host=self.host)
                        return True
                except Exception:
                    logger.info(f"Waiting for Ollama server to restart... (attempt {attempt+1}/10)")
                    time.sleep(2)
            
            logger.error("Ollama server did not restart successfully")
            return False
        except Exception as e:
            logger.error(f"Error restarting Ollama server: {e}")
            return False
