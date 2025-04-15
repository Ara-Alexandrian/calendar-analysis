"""
LLM Extraction package for Calendar Analysis.
Simplified version that only uses Ollama.
"""
# Import necessary functions from our simplified client and make them available at the package level
from .ollama_client import is_ollama_ready, get_ollama_client, extract_personnel, OLLAMA_AVAILABLE
from .extractor import (
    run_llm_extraction_parallel,
    run_llm_extraction_sequential,
    extract_personnel_with_llm
)
from .smart_extractor import run_smart_extraction
from .normalizer import normalize_extracted_personnel
from .client import is_llm_ready

# Define what's available when importing from this package
__all__ = [
    'is_ollama_ready',
    'get_ollama_client', 
    'extract_personnel',
    'OLLAMA_AVAILABLE',
    'run_llm_extraction_parallel',
    'run_llm_extraction_sequential',
    'run_smart_extraction',
    'normalize_extracted_personnel',
    'extract_personnel_with_llm'
]
