"""
Factory for creating LLM client instances.

This module provides a factory class for creating and managing LLM client instances
based on application configuration.
"""
import logging
from typing import Optional, Dict, Any

import streamlit as st

from src.infrastructure.llm.base_client import BaseLLMClient
from src.infrastructure.llm.ollama_client import OllamaClient
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class LLMClientFactory:
    """
    Factory for creating and managing LLM client instances.
    
    Provides methods to create and retrieve the appropriate LLM client
    based on the application settings.
    """
    
    @staticmethod
    @st.cache_resource  # Cache the client resource
    def get_client() -> Optional[BaseLLMClient]:
        """
        Get or create an LLM client based on application settings.
        
        Uses Streamlit caching to avoid reinitialization on every call.
        
        Returns:
            Optional[BaseLLMClient]: The appropriate LLM client instance,
                                    or None if no client could be created.
        """
        llm_provider = getattr(settings, "LLM_PROVIDER", "ollama").lower()
        
        if llm_provider == "ollama":
            client = OllamaClient()
            if client.is_available():
                logger.info(f"Successfully initialized Ollama client at {settings.OLLAMA_BASE_URL}")
                return client
            else:
                error_msg = f"Could not connect to Ollama server at {settings.OLLAMA_BASE_URL}. Check if it's running."
                logger.error(error_msg)
                st.error(f"{error_msg} LLM features disabled.", icon="🚨")
                return None
        elif llm_provider == "mcp":
            # Future implementation for MCP client
            logger.warning("MCP client is not yet implemented in the refactored architecture.")
            st.warning("MCP client is not yet implemented in the refactored architecture.", icon="⚠️")
            return None
        else:
            logger.warning(f"Unknown LLM_PROVIDER: {llm_provider}. No LLM client created.")
            st.error(f"Unknown LLM provider type: {llm_provider}. Please set a valid LLM_PROVIDER in settings.", icon="🚨")
            return None
    
    @staticmethod
    def get_available_providers() -> Dict[str, str]:
        """
        Get a dictionary of available LLM providers.
        
        Returns:
            Dict[str, str]: Dictionary mapping provider IDs to display names.
        """
        return {
            "ollama": "Ollama (Local)",
            "mcp": "Model Context Protocol (MCP)"
        }
