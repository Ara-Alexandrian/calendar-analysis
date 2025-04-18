"""
Case-insensitive and fuzzy matching for event types to handle human-written variations in calendar entries.
This module enhances the event type validation in the LLM extraction process.
"""
import logging
from rapidfuzz import process, fuzz
from typing import List, Optional, Dict, Tuple
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

def get_best_event_type_match(extracted_type: str, 
                             threshold: int = 85) -> str:
    """
    Find the best match for an extracted event type using case-insensitive matching
    and fuzzy string matching as a fallback.
    
    Args:
        extracted_type: The event type string extracted by the LLM
        threshold: Minimum similarity score (0-100) to accept a fuzzy match
        
    Returns:
        The matched standardized event type or "Unknown" if no good match found
    """
    if not extracted_type or not isinstance(extracted_type, str):
        logger.warning(f"Invalid event type format: {extracted_type}")
        return "Unknown"
    
    # Clean the input
    extracted_type = extracted_type.strip()
    if not extracted_type:
        return "Unknown"
    
    # Get the valid event types from settings
    valid_event_types = list(settings.EVENT_TYPE_MAPPING.values())
    
    # 1. Try direct match first (case-sensitive)
    if extracted_type in valid_event_types:
        logger.debug(f"Direct match found for '{extracted_type}'")
        return extracted_type
    
    # 2. Try case-insensitive match
    extracted_lower = extracted_type.lower()
    for valid_type in valid_event_types:
        if valid_type.lower() == extracted_lower:
            logger.debug(f"Case-insensitive match: '{extracted_type}' → '{valid_type}'")
            return valid_type
    
    # 3. Check if it's in the EVENT_TYPE_MAPPING directly (handles variant spellings)
    normalized = settings.EVENT_TYPE_MAPPING.get(extracted_lower)
    if normalized:
        logger.debug(f"Found in mapping: '{extracted_type}' → '{normalized}'")
        return normalized
    
    # 4. Try fuzzy matching
    try:
        # Use token sort ratio to handle word order differences
        best_match, score = process.extractOne(
            extracted_type, 
            valid_event_types,
            scorer=fuzz.token_sort_ratio
        )
        
        if score >= threshold:
            logger.info(f"Fuzzy match: '{extracted_type}' → '{best_match}' (score: {score})")
            return best_match
        else:
            logger.warning(f"No good fuzzy match for '{extracted_type}' (best: '{best_match}', score: {score})")
    except Exception as e:
        logger.error(f"Error in fuzzy matching: {e}")
    
    # 5. Fallback to tokenized partial matching for complex cases
    # This helps with cases like "BR 4D RTP BH" vs "BR 4D BH"
    extracted_tokens = set(extracted_lower.split())
    best_match = None
    best_token_overlap = 0
    
    for valid_type in valid_event_types:
        valid_tokens = set(valid_type.lower().split())
        common_tokens = extracted_tokens.intersection(valid_tokens)
        
        # Consider a match if there's significant token overlap
        if len(common_tokens) > best_token_overlap and len(common_tokens) >= 2:
            best_token_overlap = len(common_tokens)
            best_match = valid_type
    
    if best_match and best_token_overlap >= 2:
        logger.info(f"Token match: '{extracted_type}' → '{best_match}' (common tokens: {best_token_overlap})")
        return best_match
    
    return "Unknown"
