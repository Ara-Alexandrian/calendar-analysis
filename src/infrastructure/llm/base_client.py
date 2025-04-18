"""
Base LLM client interface for the Calendar Analysis application.

This module defines the abstract base class for all LLM client implementations,
ensuring a consistent interface regardless of the underlying LLM provider.
"""
from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Any, Union

# Configure logging
logger = logging.getLogger(__name__)

class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All LLM provider implementations should inherit from this class
    and implement its abstract methods.
    """
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM service is available and ready.
        
        Returns:
            bool: True if the service is available, False otherwise.
        """
        pass
    
    @abstractmethod
    def extract_personnel(self, event_description: str) -> Dict[str, Any]:
        """
        Extract personnel information from an event description.
        
        Args:
            event_description (str): The calendar event description text.
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted personnel information.
        """
        pass
    
    @abstractmethod
    def extract_event_type(self, event_description: str) -> Dict[str, Any]:
        """
        Extract event type information from an event description.
        
        Args:
            event_description (str): The calendar event description text.
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted event type information.
        """
        pass
    
    @abstractmethod
    def extract_all(self, event_description: str) -> Dict[str, Any]:
        """
        Extract all relevant information from an event description.
        
        Args:
            event_description (str): The calendar event description text.
            
        Returns:
            Dict[str, Any]: Dictionary containing all extracted information.
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dict[str, Any]: Dictionary containing model information.
        """
        pass
