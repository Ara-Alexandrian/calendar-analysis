# functions/llm_extraction/__init__.py
"""
LLM extraction package for extracting personnel information from calendar events.
This module maintains the same interface as the original llm_extractor.py for backward compatibility.
"""

import logging
from .client import get_llm_client, is_llm_ready, restart_ollama_server
from .extractor import (
    extract_personnel_with_llm, 
    run_llm_extraction_parallel,
    ultra_basic_extraction, 
    run_llm_extraction_background
)
from .normalizer import normalize_extracted_personnel

# Import the smart extractor if available
try:
    from .smart_extractor import run_smart_extraction
    SMART_EXTRACTION_AVAILABLE = True
except ImportError:
    # Create a fallback function that raises ImportError when called
    def run_smart_extraction(*args, **kwargs):
        raise ImportError("Smart extraction module is not available")
    SMART_EXTRACTION_AVAILABLE = False

# Provide the run_llm_extraction_sequential function as an alias to ultra_basic_extraction
run_llm_extraction_sequential = ultra_basic_extraction

# Configure logging
logger = logging.getLogger(__name__)

# Re-export all the public functions to maintain the same interface
__all__ = [
    'get_llm_client',
    'is_llm_ready',
    'restart_ollama_server',
    'extract_personnel_with_llm',
    'run_llm_extraction_parallel',
    'run_llm_extraction_sequential',
    'normalize_extracted_personnel',
    'run_llm_extraction_background'
]

# Constants to be available at the package level
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Import the SimpleTqdm from utils
    from .utils import SimpleTqdm as tqdm