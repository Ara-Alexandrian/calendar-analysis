"""
MCP client module for connecting to Model Context Protocol (MCP) server.
This provides an interface similar to the Ollama client for easy integration.
"""
import httpx
import json
import logging
from typing import Dict, List, Any, Optional
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class MCPClient:
    """Client for interacting with an MCP server."""
      def __init__(self, base_url: str = None):
        """
        Initialize the MCP client.
        
        Args:
            base_url: The base URL of the MCP server. If None, uses the value from settings.
        """
        self.base_url = base_url or getattr(settings, "MCP_SERVER_URL", "http://localhost:8000")
        self.client = httpx.Client(timeout=getattr(settings, "MCP_REQUEST_TIMEOUT", 60))
        logger.info(f"Initialized MCP client with base URL: {self.base_url}")
        
    def chat(self, 
             model: str = None, 
             messages: List[Dict[str, str]] = None, 
             format: str = None, 
             options: Dict[str, Any] = None):
        """
        Send a chat request to the MCP server.
        
        Args:
            model: The model to use (ignored in MCP server, included for compatibility)
            messages: List of message dictionaries with 'role' and 'content'
            format: Optional format for the response (ignored in MCP server)
            options: Additional options for the request
            
        Returns:
            The response from the MCP server
        """
        try:
            temperature = options.get('temperature', 0.7) if options else 0.7
            max_tokens = options.get('max_tokens', 1000) if options else 1000
            
            request_data = {
                "model": model or "mcp-default",
                "messages": messages or [],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            logger.debug(f"Sending chat request to MCP server: {json.dumps(request_data)}")
            
            response = self.client.post(
                f"{self.base_url}/v1/chat",
                json=request_data
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"MCP server returned error: {response.status_code}, {response.text}")
                return {"error": f"MCP server error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error calling MCP server: {str(e)}")
            return {"error": f"MCP client error: {str(e)}"}
    
    def extract_personnel(self, summary: str, canonical_names: List[str]) -> List[str]:
        """
        Extract personnel names from a calendar event summary using the MCP server.
        This is a specialized endpoint for the calendar analysis project.
        
        Args:
            summary: The calendar event summary text
            canonical_names: List of known canonical names to look for
            
        Returns:
            List of extracted personnel names
        """
        try:
            request_data = {
                "summary": summary,
                "canonical_names": canonical_names
            }
            
            logger.debug(f"Sending personnel extraction request to MCP server")
            
            response = self.client.post(
                f"{self.base_url}/v1/extract_personnel",
                json=request_data,
                timeout=getattr(settings, "MCP_REQUEST_TIMEOUT", 60)
            )
            
            if response.status_code == 200:
                result = response.json()
                extracted_personnel = result.get("extracted_personnel", [])
                return extracted_personnel
            else:
                logger.error(f"MCP server returned error: {response.status_code}, {response.text}")
                return ["Unknown_Error"]
                
        except Exception as e:
            logger.error(f"Error calling MCP server personnel extraction: {str(e)}")
            return ["Unknown_Error"]

def get_mcp_client():
    """
    Get an MCP client instance.
    
    Returns:
        An initialized MCP client
    """
    try:
        return MCPClient()
    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {str(e)}")
        return None

def is_mcp_ready():
    """
    Check if the MCP server is available and ready.
    
    Returns:
        bool: True if the MCP server is ready, False otherwise
    """
    try:
        client = MCPClient()
        base_url = client.base_url
        response = httpx.get(f"{base_url}/", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"MCP server not ready: {str(e)}")
        return False
