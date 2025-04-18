# Case-insensitive validation function to replace the current validation code in extractor.py

# Replace the current validation code (around line 175-190) with this function:

# --- Validate Extracted Data ---
validated_names = []
for name in extracted_personnel:
    # Check for exact match first
    if name in canonical_names:
        validated_names.append(name)
        continue
        
    # Otherwise check for case-insensitive match
    for canonical_name in canonical_names:
        if name.lower() == canonical_name.lower():
            validated_names.append(canonical_name)  # Use the canonical version
            break

logger.debug(f"Validated personnel names against canonical list: {validated_names}")

# Validate event type against the provided list - using case-insensitive matching
validated_event_type = "Unknown"
for valid_type in valid_event_types:
    if extracted_event_type.lower() == valid_type.lower():
        validated_event_type = valid_type  # Use the official casing
        break
        
if validated_event_type == "Unknown":
    logger.warning(f"Extracted event type '{extracted_event_type}' not in valid list for summary '{summary[:50]}...'. Setting to Unknown.")

if not validated_names:
    logger.debug(f"LLM found no known physicists in: '{summary[:30]}...'")
    return ["Unknown"], validated_event_type # Return "Unknown" personnel, but keep validated event type
else:
    logger.debug(f"LLM validated {validated_names} and event type '{validated_event_type}' in '{summary[:30]}...'")
    return validated_names, validated_event_type
