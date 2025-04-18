# functions/db/personnel_ops.py
import logging
import json
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from config import settings
from .connection import get_db_connection
from .schema import ensure_tables_exist

logger = logging.getLogger(__name__)

def save_personnel_config_to_db(config_dict):
    """
    Saves the personnel configuration to the PostgreSQL database.

    Args:
        config_dict: Dictionary containing personnel configuration.

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    if not config_dict:
        logger.warning("No personnel configuration to save to database.")
        return False

    conn = get_db_connection()
    if not conn:
        return False

    try:
        # Ensure the tables exist first
        ensure_tables_exist()

        rows_to_insert = []
        for name, details in config_dict.items():
            # Convert variations list to JSON string for PostgreSQL JSONB format
            variations_json = json.dumps(details.get('variations', []))
            rows_to_insert.append((
                name,
                details.get('role', ''),
                details.get('clinical_pct', 0.0),
                variations_json
            ))

        with conn.cursor() as cursor:
            # Use INSERT ... ON CONFLICT DO UPDATE to handle existing names
            sql_query = sql.SQL("""
                INSERT INTO {} (canonical_name, role, clinical_pct, variations)
                VALUES %s
                ON CONFLICT (canonical_name) DO UPDATE SET
                    role = EXCLUDED.role,
                    clinical_pct = EXCLUDED.clinical_pct,
                    variations = EXCLUDED.variations,
                    last_updated = CURRENT_TIMESTAMP;
            """).format(sql.Identifier(settings.DB_TABLE_PERSONNEL))

            execute_values(cursor, sql_query, rows_to_insert)
            conn.commit()

        logger.info(f"Successfully saved/updated {len(rows_to_insert)} personnel entries to database")
        return True

    except Exception as e:
        logger.error(f"Error saving personnel config to database: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def load_personnel_config_from_db():
    """
    Loads personnel configuration from the PostgreSQL database.

    Returns:
        dict: Personnel configuration dictionary, or None if loading fails or table doesn't exist.
    """
    conn = get_db_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cursor:
            # Check if the table exists before trying to select
            cursor.execute("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables
                   WHERE table_schema = %s AND table_name = %s
                )
            """, ('public', settings.DB_TABLE_PERSONNEL)) # Assuming public schema

            if not cursor.fetchone()[0]:
                logger.warning(f"Personnel table '{settings.DB_TABLE_PERSONNEL}' does not exist. Cannot load config.")
                return None # Return None, not an empty dict, to indicate absence

            # Fetch the personnel config
            cursor.execute(sql.SQL("SELECT canonical_name, role, clinical_pct, variations FROM {}")
                           .format(sql.Identifier(settings.DB_TABLE_PERSONNEL)))
            rows = cursor.fetchall()

            # Convert to dictionary format
            config_dict = {}
            for name, role, clinical_pct, variations_json in rows:
                # variations_json might be None or a JSON string
                variations_list = []
                if variations_json:
                    try:
                        variations_list = json.loads(variations_json) if isinstance(variations_json, str) else variations_json
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode variations JSON for {name}: {variations_json}")
                        variations_list = [] # Default to empty list on error

                config_dict[name] = {
                    'role': role,
                    'clinical_pct': clinical_pct,
                    'variations': variations_list # Store as Python list
                }

            logger.info(f"Successfully loaded {len(config_dict)} personnel entries from database")
            return config_dict

    except psycopg2.Error as db_e:
         logger.error(f"Database error loading personnel config: {db_e}", exc_info=True)
         return None
    except Exception as e:
        logger.error(f"Unexpected error loading personnel config from database: {e}", exc_info=True)
        return None
    finally:
        if conn:
            conn.close()
