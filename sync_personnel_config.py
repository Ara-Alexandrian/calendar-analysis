# sync_personnel_config.py
"""
Utility script to synchronize personnel configuration between JSON file and database.
This helps ensure that personnel data is properly stored in both places.
"""
import os
import sys
import json
import logging

# Add project root to path to ensure all imports work
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import configuration - using the new settings file
from config import settings_new as settings

# Configure logging
logging.basicConfig(
    level=settings.LOGGING_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler(settings.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Import required functions
from functions.db.personnel_ops import save_personnel_config_to_db, load_personnel_config_from_db
from functions.config_manager import load_personnel_config, save_personnel_config
from functions.db.connection import get_db_connection
from functions.db.schema import ensure_tables_exist

def read_json_config():
    """Read the personnel configuration from JSON file."""
    try:
        config_dict, _, _ = load_personnel_config(force_reload=True)
        return config_dict
    except Exception as e:
        logger.error(f"Error reading personnel config JSON: {e}")
        return {}

def sync_config_to_db():
    """Synchronize personnel configuration from JSON file to database."""
    if not settings.DB_ENABLED:
        logger.error("Database is not enabled in settings. Cannot sync.")
        return False
        
    # Ensure database tables exist
    if not ensure_tables_exist():
        logger.error("Failed to ensure database tables exist.")
        return False
        
    # Read config from JSON
    config_dict = read_json_config()
    if not config_dict:
        logger.error("No personnel configuration found in JSON file.")
        return False
        
    # Save config to database
    success = save_personnel_config_to_db(config_dict)
    if success:
        logger.info(f"Successfully synchronized {len(config_dict)} personnel entries to database")
    else:
        logger.error("Failed to save personnel config to database")
    
    return success

def sync_db_to_json():
    """Synchronize personnel configuration from database to JSON file."""
    if not settings.DB_ENABLED:
        logger.error("Database is not enabled in settings. Cannot sync.")
        return False
        
    # Load config from database
    config_dict = load_personnel_config_from_db()
    if not config_dict:
        logger.error("No personnel configuration found in database or could not access database.")
        return False
        
    # Save config to JSON file
    success = save_personnel_config(config_dict)
    if success:
        logger.info(f"Successfully synchronized {len(config_dict)} personnel entries to JSON file")
    else:
        logger.error("Failed to save personnel config to JSON file")
    
    return success

def verify_config_integrity():
    """Verify that personnel configuration in JSON and database match."""
    if not settings.DB_ENABLED:
        logger.error("Database is not enabled in settings. Cannot verify.")
        return False
        
    # Read config from both sources
    json_config = read_json_config()
    db_config = load_personnel_config_from_db()
    
    if not json_config and not db_config:
        logger.error("No personnel configuration found in either JSON or database.")
        return False
        
    if not json_config:
        logger.warning("No personnel configuration found in JSON file.")
        return False
        
    if not db_config:
        logger.warning("No personnel configuration found in database or could not access database.")
        return False
    
    # Compare configurations
    json_names = set(json_config.keys())
    db_names = set(db_config.keys())
    
    missing_in_db = json_names - db_names
    missing_in_json = db_names - json_names
    common_names = json_names.intersection(db_names)
    
    if missing_in_db:
        logger.warning(f"Names in JSON but missing in DB: {missing_in_db}")
    
    if missing_in_json:
        logger.warning(f"Names in DB but missing in JSON: {missing_in_json}")
    
    # Check for differences in common entries
    differences = []
    for name in common_names:
        json_entry = json_config[name]
        db_entry = db_config[name]
        
        # Compare role
        if json_entry.get('role') != db_entry.get('role'):
            differences.append(f"{name}: role differs - JSON: {json_entry.get('role')}, DB: {db_entry.get('role')}")
            
        # Compare clinical_pct - allowing for small floating point differences
        json_pct = json_entry.get('clinical_pct', 0)
        db_pct = db_entry.get('clinical_pct', 0)
        if abs(json_pct - db_pct) > 0.001:  # Small epsilon for floating point comparison
            differences.append(f"{name}: clinical_pct differs - JSON: {json_pct}, DB: {db_pct}")
        
        # Compare variations - convert to sets for order-insensitive comparison
        json_variations = set(json_entry.get('variations', []))
        db_variations = set(db_entry.get('variations', []))
        if json_variations != db_variations:
            only_in_json = json_variations - db_variations
            only_in_db = db_variations - json_variations
            diff_msg = f"{name}: variations differ"
            if only_in_json:
                diff_msg += f" - Only in JSON: {only_in_json}"
            if only_in_db:
                diff_msg += f" - Only in DB: {only_in_db}"
            differences.append(diff_msg)
    
    if differences:
        logger.warning(f"Found {len(differences)} differences between JSON and DB configurations:")
        for diff in differences:
            logger.warning(f"  {diff}")
        return False
    
    logger.info(f"Personnel configurations match between JSON and database ({len(common_names)} entries)")
    return True

def show_db_config():
    """Display the current personnel configuration in the database."""
    if not settings.DB_ENABLED:
        print("Database is not enabled in settings.")
        return
        
    db_config = load_personnel_config_from_db()
    if not db_config:
        print("No personnel configuration found in database or could not access database.")
        return
        
    print("\nPersonnel Configuration in Database:")
    print("====================================")
    for name, details in sorted(db_config.items()):
        print(f"\nName: {name}")
        print(f"  Role: {details.get('role', 'Not specified')}")
        print(f"  Clinical %: {details.get('clinical_pct', 'Not specified')}")
        print(f"  Variations: {', '.join(details.get('variations', ['None']))}")
    
    print(f"\nTotal: {len(db_config)} personnel entries")

def show_json_config():
    """Display the current personnel configuration in the JSON file."""
    json_config, _, _ = load_personnel_config(force_reload=True)
    if not json_config:
        print("No personnel configuration found in JSON file.")
        return
        
    print("\nPersonnel Configuration in JSON file:")
    print("====================================")
    for name, details in sorted(json_config.items()):
        print(f"\nName: {name}")
        print(f"  Role: {details.get('role', 'Not specified')}")
        print(f"  Clinical %: {details.get('clinical_pct', 'Not specified')}")
        print(f"  Variations: {', '.join(details.get('variations', ['None']))}")
    
    print(f"\nTotal: {len(json_config)} personnel entries")

def main():
    """Main function to execute the script based on command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Synchronize personnel configuration between JSON and database.')
    parser.add_argument('--json-to-db', action='store_true', help='Sync from JSON file to database')
    parser.add_argument('--db-to-json', action='store_true', help='Sync from database to JSON file')
    parser.add_argument('--verify', action='store_true', help='Verify that JSON and database configurations match')
    parser.add_argument('--show-db', action='store_true', help='Show personnel configuration in database')
    parser.add_argument('--show-json', action='store_true', help='Show personnel configuration in JSON file')
    
    args = parser.parse_args()
    
    # Default action if no arguments provided
    if not any(vars(args).values()):
        print("No action specified. Use --help to see available options.")
        show_json_config()
        if settings.DB_ENABLED:
            print("\n")
            show_db_config()
        return
    
    if args.json_to_db:
        success = sync_config_to_db()
        print(f"Sync from JSON to database: {'Successful' if success else 'Failed'}")
    
    if args.db_to_json:
        success = sync_db_to_json()
        print(f"Sync from database to JSON: {'Successful' if success else 'Failed'}")
    
    if args.verify:
        success = verify_config_integrity()
        print(f"Configuration verification: {'Passed' if success else 'Failed'}")
    
    if args.show_db:
        show_db_config()
    
    if args.show_json:
        show_json_config()

if __name__ == "__main__":
    main()
