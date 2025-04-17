# functions/db/schema.py
import logging
import psycopg2
from psycopg2 import sql
from config import settings
from .connection import get_db_connection # Import from sibling module

logger = logging.getLogger(__name__)

def ensure_tables_exist():
    """
    Ensures that the necessary database tables exist, creating them if they don't.
    Also ensures necessary columns exist in existing tables.
    """
    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cursor:
            # Create processed events table if it doesn't exist
            cursor.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    uid TEXT UNIQUE,
                    summary TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    duration_hours FLOAT,
                    personnel TEXT,
                    role TEXT,
                    clinical_pct FLOAT,
                    extracted_event_type TEXT,
                    raw_data JSONB,
                    batch_id TEXT,
                    processing_status TEXT,
                    calendar_file_id INTEGER,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)))
            logger.info(f"Ensured table '{settings.DB_TABLE_PROCESSED_DATA}' exists.")

            # --- Check and Add 'extracted_event_type' column ---
            column_exists = False
            try:
                cursor.execute(sql.SQL("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
                """), (settings.DB_TABLE_PROCESSED_DATA, 'extracted_event_type'))
                column_exists = cursor.fetchone() is not None
                logger.info(f"Checked information_schema: 'extracted_event_type' column exists? {column_exists}")
            except Exception as check_e:
                 logger.warning(f"Could not check information_schema for 'extracted_event_type' column: {check_e}")
                 # Proceed with ALTER TABLE anyway as a fallback

            if not column_exists:
                logger.info(f"'extracted_event_type' column not found, attempting to add it...")
                try:
                    # Use ALTER TABLE ADD COLUMN IF NOT EXISTS as before
                    cursor.execute(sql.SQL("""
                        ALTER TABLE {}
                        ADD COLUMN IF NOT EXISTS extracted_event_type TEXT;
                    """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)))
                    conn.commit() # Commit the ALTER TABLE immediately
                    logger.info(f"Executed ALTER TABLE to add 'extracted_event_type'.")
                except Exception as alter_e:
                    # Log warning but don't necessarily stop the whole process.
                    # The command is IF NOT EXISTS, so failure might not be critical if the column is already there.
                    # Rely on the final commit/rollback for the overall function.
                    logger.warning(f"Could not add 'extracted_event_type' column via ALTER TABLE (may require manual check or run): {alter_e}")
                    # Removed explicit rollback here to simplify transaction handling.
            # --- End Check and Add ---


            # Create personnel config table if it doesn't exist
            cursor.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    canonical_name TEXT UNIQUE,
                    role TEXT,
                    clinical_pct FLOAT,
                    variations JSONB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """).format(sql.Identifier(settings.DB_TABLE_PERSONNEL)))
            logger.info(f"Ensured table '{settings.DB_TABLE_PERSONNEL}' exists.")

            # Create calendar files table to store raw JSON files
            cursor.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    filename TEXT,
                    file_content JSONB,
                    file_hash TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    batch_id TEXT,
                    is_current BOOLEAN DEFAULT FALSE
                )
            """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)))
            logger.info(f"Ensured table '{settings.DB_TABLE_CALENDAR_FILES}' exists.")

            # Add an index on the uid field for faster lookups
            cursor.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS idx_processed_events_uid
                ON {} (uid);
            """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)))
            logger.info(f"Ensured index 'idx_processed_events_uid' exists.")

            # Add an index on file_hash for faster duplicate checks in calendar_files
            cursor.execute(sql.SQL("""
                CREATE INDEX IF NOT EXISTS idx_calendar_files_hash
                ON {} (file_hash);
            """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)))
            logger.info(f"Ensured index 'idx_calendar_files_hash' exists.")

            # Create a function to enforce the constraint that only one file can be current
            cursor.execute("""
                CREATE OR REPLACE FUNCTION enforce_single_current_file()
                RETURNS TRIGGER AS $$
                BEGIN
                    IF NEW.is_current = TRUE THEN
                        UPDATE calendar_files SET is_current = FALSE WHERE id != NEW.id;
                    END IF;
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            logger.info("Ensured function 'enforce_single_current_file' exists.")

            # Try to create the trigger if it doesn't exist
            try:
                cursor.execute(sql.SQL("""
                    CREATE TRIGGER ensure_single_current_file
                    BEFORE INSERT OR UPDATE ON {}
                    FOR EACH ROW
                    WHEN (NEW.is_current = TRUE)
                    EXECUTE FUNCTION enforce_single_current_file();
                """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)))
                logger.info("Created trigger 'ensure_single_current_file'.")
            except psycopg2.errors.DuplicateObject:
                # If the trigger already exists, this is expected, log info and continue
                logger.info("Trigger 'ensure_single_current_file' already exists.")
                conn.rollback() # Rollback the failed trigger creation attempt
                # Need to re-establish transaction state if rollback occurred
                # For simplicity, we might just let the commit happen or handle state more carefully
            except Exception as e:
                # Log other unexpected errors during trigger creation
                logger.warning(f"Could not create trigger 'ensure_single_current_file': {e}")
                conn.rollback() # Rollback on other errors too

            conn.commit()
            logger.info("Database schema setup/verification complete.")
            return True
    except Exception as e:
        logger.error(f"Error during database schema setup: {e}", exc_info=True)
        if conn:
             try:
                 conn.rollback()
             except Exception as rb_e:
                 logger.error(f"Rollback failed during schema setup error handling: {rb_e}")
        return False
    finally:
        if conn:
            conn.close()

def clear_database_tables():
    """
    Clears all data from the main data tables (processed_events).
    Does NOT delete the tables themselves.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not settings.DB_ENABLED:
        logger.warning("Database persistence is disabled. Cannot clear tables.")
        return False

    conn = get_db_connection()
    if not conn:
        logger.error("Failed to connect to database for clearing tables.")
        return False

    table_name = settings.DB_TABLE_PROCESSED_DATA
    logger.info(f"Attempting to clear table: {table_name}")

    try:
        with conn.cursor() as cursor:
            # Clear the processed events table using DELETE FROM
            logger.warning(f"Executing DELETE FROM {table_name}")
            cursor.execute(sql.SQL("DELETE FROM {}").format(
                sql.Identifier(table_name)
            ))
            deleted_rows = cursor.rowcount
            logger.info(f"DELETE command executed. Rows affected: {deleted_rows}")

            # Reset the sequence for the primary key if it's a SERIAL type
            seq_name = f"{table_name}_id_seq"
            logger.info(f"Attempting to reset sequence: {seq_name}")
            try:
                cursor.execute(sql.SQL("ALTER SEQUENCE {sequence_name} RESTART WITH 1").format(
                    sequence_name=sql.Identifier(seq_name)
                ))
                logger.info(f"Sequence {seq_name} reset successfully.")
            except psycopg2.Error as seq_e:
                logger.warning(f"Could not reset sequence {seq_name}. Error code: {seq_e.pgcode}. Message: {seq_e.pgerror}. This might be expected if the sequence name differs or doesn't exist.")
                conn.rollback() # Rollback this specific error

            # Optionally, clear or reset other related tables if needed
            # We might want to reset the 'processed' flag on calendar files later.
            # logger.info(f"Attempting to reset flags in {settings.DB_TABLE_CALENDAR_FILES}")
            # try:
            #     cursor.execute(sql.SQL("UPDATE {} SET processed = FALSE, is_current = FALSE").format(
            #         sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)
            #     ))
            #     logger.info(f"Flags reset in {settings.DB_TABLE_CALENDAR_FILES}")
            # except Exception as flag_e:
            #     logger.error(f"Failed to reset flags in {settings.DB_TABLE_CALENDAR_FILES}: {flag_e}")
            #     conn.rollback() # Rollback if flag reset fails

            logger.info(f"Committing transaction for table clear: {table_name}")
            conn.commit()
            logger.info(f"Successfully cleared table {table_name}. Rows deleted: {deleted_rows}")
            return True

    except psycopg2.Error as db_e:
        logger.error(f"Database error clearing table {table_name}. Error code: {db_e.pgcode}. Message: {db_e.pgerror}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Unexpected error clearing table {table_name}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            logger.info(f"Closing connection after attempting to clear {table_name}")
            conn.close()
