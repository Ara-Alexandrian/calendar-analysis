"""
Database initialization utilities for Calendar Analysis application.
Creates baseline configuration in PostgreSQL upon startup.
"""
import logging
import json
from config import settings
from . import db_manager

logger = logging.getLogger(__name__)

# Default personnel configuration to use when no configuration exists
DEFAULT_PERSONNEL_CONFIG = {
    "D. Smith": {
        "role": "physicist",
        "clinical_pct": 0.8,
        "variations": ["Smith", "David S.", "D. Smith", "David Smith"]
    },
    "J. Johnson": {
        "role": "physicist",
        "clinical_pct": 0.75,
        "variations": ["Johnson", "Jane J.", "J. Johnson", "Jane Johnson"]
    },
    "M. Richards": {
        "role": "physicist",
        "clinical_pct": 0.9,
        "variations": ["Richards", "Michael R.", "M. Richards", "Michael Richards"]
    },
    "A. Patel": {
        "role": "physician",
        "clinical_pct": 0.6,
        "variations": ["Patel", "Dr. Patel", "A. Patel", "Amanda Patel"]
    }
}

def initialize_database():
    """
    Initializes the database with required tables and baseline configuration.
    Should be called during application startup.
    """
    # Make sure database and tables exist
    conn = db_manager.get_db_connection()
    if not conn:
        logger.warning("Could not establish database connection for initialization.")
        return False
    
    conn.close()  # Close initial connection after database creation
    
    # Create all tables
    db_manager.ensure_tables_exist()
    
    # Initialize baseline personnel configuration if none exists
    initialize_personnel_config()
    
    logger.info("Database initialization complete.")
    return True

def initialize_personnel_config():
    """
    Checks if personnel configuration exists in database, and if not,
    initializes with default values.
    """
    conn = db_manager.get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            # Check if personnel table has any data
            cursor.execute(f"SELECT COUNT(*) FROM {settings.DB_TABLE_PERSONNEL}")
            count = cursor.fetchone()[0]
            
            if count == 0:
                logger.info("No personnel configuration found in database. Creating baseline configuration.")
                # Insert default personnel configuration
                for name, details in DEFAULT_PERSONNEL_CONFIG.items():
                    cursor.execute(
                        f"""
                        INSERT INTO {settings.DB_TABLE_PERSONNEL}
                        (canonical_name, role, clinical_pct, variations)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            name,
                            details.get('role', ''),
                            details.get('clinical_pct', 0.0),
                            json.dumps(details.get('variations', [])),
                        )
                    )
                conn.commit()
                logger.info(f"Created baseline personnel configuration with {len(DEFAULT_PERSONNEL_CONFIG)} entries.")
                return True
            else:
                logger.info(f"Personnel configuration already exists in database with {count} entries.")
                return False
                
    except Exception as e:
        logger.error(f"Error initializing personnel configuration: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
