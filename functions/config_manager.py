# functions/config_manager.py
import json
import logging
import os
from config import settings # Import static settings to get path

logger = logging.getLogger(__name__)

# Global cache for personnel config and derived maps to avoid redundant processing
_personnel_config_cache = None
_variation_map_cache = None
_canonical_names_cache = None

def load_personnel_config(force_reload=False):
    """Loads personnel configuration from JSON, caches it, and derives maps."""
    global _personnel_config_cache, _variation_map_cache, _canonical_names_cache

    if not force_reload and _personnel_config_cache is not None:
        return _personnel_config_cache, _variation_map_cache, _canonical_names_cache

    filepath = settings.PERSONNEL_CONFIG_JSON_PATH
    logger.info(f"Loading personnel configuration from: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
            if not isinstance(config_data, dict):
                logger.error(f"Invalid format in {filepath}. Expected a JSON object (dictionary).")
                return {}, {}, []
            _personnel_config_cache = config_data
            _variation_map_cache, _canonical_names_cache = _derive_maps(config_data)
            logger.info(f"Successfully loaded and processed {len(_canonical_names_cache)} personnel entries.")
            return _personnel_config_cache, _variation_map_cache, _canonical_names_cache

    except FileNotFoundError:
        logger.error(f"Personnel configuration file not found: {filepath}")
        return {}, {}, []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return {}, {}, []
    except Exception as e:
        logger.error(f"An unexpected error occurred loading personnel config: {e}")
        return {}, {}, []

def save_personnel_config(config_data):
    """Saves the personnel configuration dictionary to JSON."""
    global _personnel_config_cache, _variation_map_cache, _canonical_names_cache
    filepath = settings.PERSONNEL_CONFIG_JSON_PATH
    logger.info(f"Attempting to save personnel configuration to: {filepath}")
    try:
        if not isinstance(config_data, dict):
             logger.error("Invalid data type provided for saving. Expected a dictionary.")
             return False

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)

        logger.info(f"Successfully saved personnel configuration to {filepath}")
        # Clear cache to force reload on next access
        _personnel_config_cache = None
        _variation_map_cache = None
        _canonical_names_cache = None
        return True
    except IOError as e:
        logger.error(f"IOError saving personnel configuration to {filepath}: {e}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred saving personnel config: {e}")
        return False

def _derive_maps(personnel_config):
    """Derives the variation map and canonical names list from the config."""
    variation_map = {}
    canonical_names = list(personnel_config.keys())
    log_entries = set() # To avoid duplicate warnings per session

    for canonical_name, config_data in personnel_config.items():
        if isinstance(config_data, dict) and "variations" in config_data:
            if isinstance(config_data["variations"], list):
                for variation in config_data["variations"]:
                    if not isinstance(variation, str):
                        if (canonical_name, 'non_string') not in log_entries:
                             logger.warning(f"Non-string variation found for '{canonical_name}': {variation}. Skipping.")
                             log_entries.add((canonical_name, 'non_string'))
                        continue

                    lower_var = variation.lower().strip()
                    if not lower_var:
                         if (canonical_name, 'empty_string') not in log_entries:
                            logger.warning(f"Empty variation string found for '{canonical_name}'. Skipping.")
                            log_entries.add((canonical_name, 'empty_string'))
                         continue

                    if lower_var in variation_map:
                        # Log a warning only if the mapping changes
                        if variation_map[lower_var] != canonical_name:
                            warning_key = ('duplicate', lower_var)
                            if warning_key not in log_entries:
                                logger.warning(
                                    f"Duplicate variation '{variation}' (lowercase: '{lower_var}') mapped to '{canonical_name}'. "
                                    f"It was previously mapped to '{variation_map[lower_var]}'. Overwriting."
                                )
                                log_entries.add(warning_key)
                    variation_map[lower_var] = canonical_name
            else:
                 if (canonical_name, 'invalid_variations') not in log_entries:
                    logger.warning(f"Personnel config for '{canonical_name}' has 'variations' but it's not a list. Skipping.")
                    log_entries.add((canonical_name, 'invalid_variations'))
        else:
            if (canonical_name, 'missing_variations') not in log_entries:
                logger.warning(f"Personnel config for '{canonical_name}' is invalid or missing 'variations'. Skipping.")
                log_entries.add((canonical_name, 'missing_variations'))

    return variation_map, canonical_names

def get_personnel_details():
    """Convenience function to get all cached config details."""
    return load_personnel_config() # Ensures loading if cache is empty

def get_canonical_names():
    """Returns the cached list of canonical names."""
    _, _, names = load_personnel_config()
    return names

def get_variation_map():
    """Returns the cached variation map."""
    _, vmap, _ = load_personnel_config()
    return vmap

def get_personnel_config_dict():
     """Returns the cached personnel config dictionary."""
     config_dict, _, _ = load_personnel_config()
     return config_dict

def get_role(personnel_name):
    """Retrieves the role for a given canonical personnel name."""
    config_dict = get_personnel_config_dict()
    return config_dict.get(personnel_name, {}).get('role', 'Unknown')

def get_clinical_pct(personnel_name):
    """Retrieves the clinical percentage for a given canonical personnel name."""
    config_dict = get_personnel_config_dict()
    # Return None if not found or not a valid number
    pct = config_dict.get(personnel_name, {}).get('clinical_pct')
    if isinstance(pct, (int, float)):
        return pct
    return None