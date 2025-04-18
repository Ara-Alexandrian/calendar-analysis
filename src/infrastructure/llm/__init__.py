"""
LLM module for the Calendar Analysis application.

This package provides LLM client implementations and utilities
for extracting information from calendar events.
"""

from src.infrastructure.llm.base_client import BaseLLMClient
from src.infrastructure.llm.ollama_client import OllamaClient
from src.infrastructure.llm.factory import LLMClientFactory

# Provide convenient access to the factory method
get_llm_client = LLMClientFactory.get_client
