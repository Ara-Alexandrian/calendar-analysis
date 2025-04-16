# functions/llm_extraction/smart_router.py
"""
Stub implementation of the smart router that was previously used.
This file exists to maintain compatibility with existing code.
"""
import logging

# Configure logging
logger = logging.getLogger(__name__)

def get_router(*args, **kwargs):
    """
    Stub function for the removed smart router.
    
    Returns:
        None: Always returns None as this functionality has been removed.
    """
    logger.warning("Smart router functionality has been removed. Using direct LLM extraction instead.")
    return None
