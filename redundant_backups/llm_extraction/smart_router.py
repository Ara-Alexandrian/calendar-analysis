"""
Simplified direct provider module replacing the complex smart router.
This module directly returns the configured LLM client without any routing logic.
"""

import logging
from typing import Dict, Any, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class SmartRouter:
    """
    Simplified router that always returns the default LLM client.
    This replaces the complex smart routing logic with a simple direct approach.
    """
    
    def __init__(self):
        logger.info("Initializing simplified direct LLM provider")
    
    def get_best_provider(self, **kwargs) -> Tuple[str, Any]:
        """
        Get the default LLM provider - no smart routing.
        
        Args:
            **kwargs: Any parameters (ignored in this simplified version)
            
        Returns:
            tuple: ('default', client) where client is the LLM client instance
        """
        from .client import get_llm_client
        
        # Always use the default client
        logger.debug("Using default LLM client (no smart routing)")
        client = get_llm_client()
        
        return 'default', client
    
    def route_request(self, **kwargs) -> str:
        """
        Route a request - simplified to always return 'default'
        
        Args:
            **kwargs: Any parameters (ignored in this simplified version)
            
        Returns:
            str: Always returns 'default'
        """
        return 'default'
        
    def extract_personnel(self, summary: str, canonical_names=None):
        """
        Extract personnel names from a calendar event summary.
        This method directly implements extraction to avoid circular imports.
        
        Args:
            summary: The calendar event summary text
            canonical_names: List of known canonical names to look for
            
        Returns:
            list: Extracted personnel names or ["Unknown"] if extraction fails
        """
        import json
        import time
        from .client import get_llm_client
        
        # Skip empty summaries
        if not summary or not isinstance(summary, str) or len(summary.strip()) == 0:
            logger.debug("Skipping extraction for empty or non-string summary in smart_router")
            return ["Unknown"]
        
        client = get_llm_client()
        if not client:
            logger.warning("LLM client unavailable in smart router, returning Unknown")
            return ["Unknown"]
            
        # IMPORTANT: Instead of calling back to extractor.py, we'll directly use ollama_client
        # This breaks the circular dependency
        from .ollama_client import extract_personnel as ollama_extract
        
        try:
            # Call the ollama_client function directly
            return ollama_extract(summary, canonical_names)
            
        except Exception as e:
            logger.error(f"Error in SmartRouter.extract_personnel: {e}")
            return ["Unknown_Error"]

# Singleton router instance
_router_instance = None

def get_router() -> SmartRouter:
    """Get the singleton router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = SmartRouter()
    return _router_instance
