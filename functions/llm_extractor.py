# functions/llm_extractor.py
"""
LLM extraction module for extracting personnel information from calendar events.
DEPRECATED: This module is kept for backward compatibility.
Please use the functions from the llm_extraction package instead.
"""

import logging
import sys
import warnings

# Configure logging
logger = logging.getLogger(__name__)

# Show a deprecation warning
warnings.warn(
    "This module is deprecated and will be removed in a future version. "
    "Please use 'from functions.llm_extraction import ...' instead.",
    DeprecationWarning, 
    stacklevel=2
)

# Import and re-export all the functions from the llm_extraction package
from functions.llm_extraction import (
    # Client functions
    get_llm_client,
    is_llm_ready,
    restart_ollama_server,
    
    # Extraction functions
    extract_personnel_with_llm,
    run_llm_extraction_parallel,
    run_llm_extraction_sequential,
    run_llm_extraction_background,
    
    # Normalization functions
    normalize_extracted_personnel,
    
    # Constants
    OLLAMA_AVAILABLE,
    TQDM_AVAILABLE
)

# Export all imported names
__all__ = [
    'get_llm_client',
    'is_llm_ready',
    'restart_ollama_server',
    'extract_personnel_with_llm',
    'run_llm_extraction_parallel',
    'run_llm_extraction_sequential',
    'run_llm_extraction_background',
    'normalize_extracted_personnel',
    'OLLAMA_AVAILABLE',
    'TQDM_AVAILABLE'
]

logger.info("llm_extractor.py is deprecated. Using the new modular implementation from llm_extraction package.")