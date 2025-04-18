"""
Case-insensitive validation for event types.
This improved validation logic ensures that case variations like 'RTp' vs 'RTP' 
are properly matched to their canonical forms.
"""
import logging
from config import settings

logger = logging.getLogger(__name__)

def validate_event_type(extracted_event_type, summary=None):
    """
    Validate an extracted event type against the list of valid event types using
    case-insensitive matching.
    
    Args:
        extracted_event_type: The event type string returned by the LLM
        summary: Optional event summary for logging purposes
        
    Returns:
        str: The validated event type with proper capitalization or "Unknown"
    """
    if not extracted_event_type or not isinstance(extracted_event_type, str):
        return "Unknown"
        
    # Get valid event types from settings.py - single source of truth
    valid_event_types = list(settings.EVENT_TYPE_MAPPING.values())
    
    # Validate event type against the provided list - using case-insensitive matching
    validated_event_type = "Unknown"
    for valid_type in valid_event_types:
        if extracted_event_type.lower() == valid_type.lower():
            validated_event_type = valid_type  # Use the official casing
            break
    
    # Log if the event type was not found
    if validated_event_type == "Unknown" and summary:
        logger.warning(f"Extracted event type '{extracted_event_type}' not in valid list for summary '{summary[:50]}...'. Setting to Unknown.")
    
    return validated_event_type
