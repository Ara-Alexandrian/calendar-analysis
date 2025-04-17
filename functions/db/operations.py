# functions/db/operations.py
import logging
import pandas as pd
import psycopg2
import json
import datetime
import hashlib
import uuid
import time
from psycopg2 import sql
from psycopg2.extras import execute_values
from sqlalchemy.sql import text # For SQLAlchemy text queries
from config import settings
from .connection import get_db_connection, get_sqlalchemy_engine # Import engine getter
from .schema import ensure_tables_exist
# Import config_manager carefully to avoid circular dependencies if it uses db operations
# If config_manager needs db ops, consider restructuring or passing db functions as args.
# For now, assume direct import is okay for get_role/get_clinical_pct.
from functions import config_manager


logger = logging.getLogger(__name__)

def save_processed_data_to_db(df, batch_id=None):
    """
    Saves the processed and exploded data to the PostgreSQL database.

    Args:
        df: DataFrame containing processed event data with personnel assignments.
        batch_id: Optional batch ID to associate with this data

    Returns:
        bool: True if saved successfully, False otherwise.
    """
    if df is None or df.empty:
        logger.warning("No data to save to database.")
        return False

    conn = get_db_connection()
    if not conn:
        return False

    try:
        # Ensure required columns exist
        required_columns = ['uid', 'summary', 'start_time', 'end_time', 'duration_hours', 'personnel']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"Required column '{col}' not found in DataFrame.")
                return False

        # Ensure the tables and columns exist
        ensure_tables_exist() # This now also checks/adds extracted_event_type column

        # Prepare data for insertion
        rows_to_insert = []
        for _, row in df.iterrows():
            personnel = row['personnel']
            # Ensure personnel is a single string
            if isinstance(personnel, list) and len(personnel) > 0:
                 personnel = str(personnel[0])
            elif not isinstance(personnel, str):
                 personnel = str(personnel) # Convert other types just in case

            role = config_manager.get_role(personnel)
            clinical_pct = config_manager.get_clinical_pct(personnel)
            summary = str(row['summary']) if pd.notna(row['summary']) else None
            uid = str(row['uid']) if pd.notna(row['uid']) else None

            # Prepare raw_data dict, ensuring basic JSON compatibility
            row_dict_serializable = {}
            for k, v in row.to_dict().items():
                is_array_like = isinstance(v, (list, tuple)) or hasattr(v, 'ndim')
                if is_array_like:
                    row_dict_serializable[k] = str(v)
                elif pd.isna(v):
                    row_dict_serializable[k] = None
                elif isinstance(v, (datetime.datetime, pd.Timestamp)):
                    row_dict_serializable[k] = v.isoformat() if v else None
                elif isinstance(v, (list, dict, str, int, float, bool, type(None))):
                    try:
                        json.dumps(v)
                        row_dict_serializable[k] = v
                    except TypeError:
                        row_dict_serializable[k] = str(v)
                else:
                    row_dict_serializable[k] = str(v)

            # Manually serialize the dictionary to a JSON string
            try:
                raw_data_json_str = json.dumps(row_dict_serializable)
            except TypeError as json_err:
                logger.error(f"JSON serialization error for uid {uid}: {json_err}. Falling back.")
                fallback_dict = {k_fb: str(v_fb) for k_fb, v_fb in row_dict_serializable.items()}
                raw_data_json_str = json.dumps(fallback_dict)

            processing_status = row.get('processing_status', 'assigned')
            extracted_event_type = row.get('extracted_event_type', None)

            rows_to_insert.append((
                uid, summary, row['start_time'], row['end_time'], row['duration_hours'],
                personnel, role, clinical_pct, extracted_event_type,
                raw_data_json_str, batch_id, processing_status
            ))

        # Insert data with ON CONFLICT DO UPDATE
        with conn.cursor() as cursor:
            sql_query = sql.SQL("""
                INSERT INTO {}
                (uid, summary, start_time, end_time, duration_hours, personnel,
                 role, clinical_pct, extracted_event_type, raw_data, batch_id, processing_status)
                VALUES %s
                ON CONFLICT (uid) DO UPDATE SET
                    personnel = EXCLUDED.personnel,
                    extracted_event_type = EXCLUDED.extracted_event_type,
                    role = EXCLUDED.role,
                    clinical_pct = EXCLUDED.clinical_pct,
                    raw_data = EXCLUDED.raw_data,
                    batch_id = EXCLUDED.batch_id,
                    processing_status = EXCLUDED.processing_status,
                    processing_date = CURRENT_TIMESTAMP
            """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA))

            execute_values(cursor, sql_query, rows_to_insert)
            conn.commit()

        logger.info(f"Successfully saved/updated {len(rows_to_insert)} rows to database table {settings.DB_TABLE_PROCESSED_DATA}")

        # If we have a batch_id, mark the calendar file as processed
        if batch_id:
            mark_calendar_file_as_processed(batch_id)

        return True

    except Exception as e:
        logger.error(f"Error saving data to database: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


def save_processed_data_in_batches(df, batch_id=None, batch_size=100):
    """
    Saves processed data to the database in smaller batches.
    """
    if df is None or df.empty:
        logger.warning("No data to save to database.")
        return False, 0

    total_rows = len(df)
    saved_rows = 0
    failed_batches = []

    try:
        # Process and save data in batches
        for start_idx in range(0, total_rows, batch_size):
            end_idx = min(start_idx + batch_size, total_rows)
            batch_df = df.iloc[start_idx:end_idx].copy()

            try:
                # Save this batch using the main save function
                batch_success = save_processed_data_to_db(batch_df, batch_id)

                if batch_success:
                    saved_rows += len(batch_df)
                    logger.info(f"Saved batch {start_idx//batch_size + 1}: {saved_rows}/{total_rows} records")
                else:
                    error_msg = f"Failed to save batch starting at record {start_idx}"
                    logger.error(error_msg)
                    failed_batches.append((start_idx, end_idx, error_msg))
            except Exception as e:
                error_msg = f"Exception while saving batch starting at record {start_idx}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                failed_batches.append((start_idx, end_idx, error_msg))
                continue # Continue with next batch despite errors

        success = saved_rows > 0
        if success:
            logger.info(f"Successfully saved {saved_rows}/{total_rows} records in batches")
            if failed_batches:
                logger.warning(f"There were {len(failed_batches)} failed batches. Check logs for details.")
        else:
            logger.error("Failed to save any records to database")

        return success, saved_rows

    except Exception as e:
        logger.error(f"Error in batch saving process: {str(e)}", exc_info=True)
        return False, saved_rows


def save_partial_processed_data(df, batch_id):
    """
    Saves partial processed data to the database with a batch identifier.
    Used for background processing to track which events are being processed.

    Args:
        df: DataFrame containing partially processed event data
        batch_id: Unique identifier for this processing batch

    Returns:
        bool: True if saved successfully, False otherwise
    """
    logger.info(f"--- ENTERING save_partial_processed_data for batch {batch_id} ---")

    if df is None or df.empty:
        logger.warning(f"Batch {batch_id}: No partial data provided to save.")
        return False

    logger.info(f"Batch {batch_id}: Input df shape: {df.shape}. Columns: {df.columns.tolist()}")

    conn = None
    try:
        conn = get_db_connection()
        if not conn:
            logger.error(f"Batch {batch_id}: Failed to get DB connection.")
            return False
        logger.info(f"Batch {batch_id}: DB connection obtained.")

        # Schema is assumed to be correct here; ensure_tables_exist() should be called earlier.
        # if not ensure_tables_exist():
        #      logger.error(f"Batch {batch_id}: Failed to ensure database schema exists. Aborting save.")
        #      return False

        rows_to_insert = []
        logger.info(f"Batch {batch_id}: Preparing rows...")

        for idx, row in df.iterrows():
            try:
                uid = row.get('uid') if pd.notna(row.get('uid')) else f"{batch_id}_{idx}"
                summary = str(row.get('summary', '')) if pd.notna(row.get('summary')) else ''
                start_time = row.get('start_time') if pd.notna(row.get('start_time')) else None
                end_time = row.get('end_time') if pd.notna(row.get('end_time')) else None
                duration_hours = float(row.get('duration_hours', 0.0)) if pd.notna(row.get('duration_hours')) else 0.0

                personnel_list = row.get('extracted_personnel', ['Unknown'])
                if isinstance(personnel_list, list) and len(personnel_list) > 0:
                    personnel = str(personnel_list[0])
                elif isinstance(personnel_list, str):
                    personnel = personnel_list
                elif isinstance(personnel_list, (int, float)):
                    personnel = str(personnel_list)
                else:
                    personnel = "Unknown"

                role = config_manager.get_role(personnel)
                clinical_pct = config_manager.get_clinical_pct(personnel)
                extracted_event_type = str(row.get('extracted_event_type', 'Unknown'))
                processing_status = str(row.get('processing_status', 'extracted'))

                row_dict_serializable = {}
                for k, v in row.to_dict().items():
                    is_array_like = isinstance(v, (list, tuple)) or hasattr(v, 'ndim')
                    if is_array_like:
                        row_dict_serializable[k] = str(v)
                    elif pd.isna(v):
                        row_dict_serializable[k] = None
                    elif isinstance(v, (datetime.datetime, pd.Timestamp)):
                        row_dict_serializable[k] = v.isoformat() if v else None
                    elif isinstance(v, (list, dict, str, int, float, bool, type(None))):
                        try:
                            json.dumps(v)
                            row_dict_serializable[k] = v
                        except TypeError:
                            row_dict_serializable[k] = str(v)
                    else:
                        row_dict_serializable[k] = str(v)

                try:
                    raw_data_json_str = json.dumps(row_dict_serializable)
                except TypeError as json_err:
                    logger.error(f"Batch {batch_id}, Row {idx}: Could not serialize row data to JSON: {json_err}. Falling back.")
                    fallback_dict = {k_fb: str(v_fb) for k_fb, v_fb in row_dict_serializable.items()}
                    raw_data_json_str = json.dumps(fallback_dict)

                rows_to_insert.append((
                    uid, summary, start_time, end_time, duration_hours,
                    personnel, role, clinical_pct, extracted_event_type,
                    raw_data_json_str, batch_id, processing_status
                ))
            except Exception as row_proc_e:
                logger.error(f"Batch {batch_id}: Error processing row index {idx}: {row_proc_e}", exc_info=True)

        logger.info(f"Batch {batch_id}: Prepared {len(rows_to_insert)} rows for insertion.")

        if not rows_to_insert:
             logger.warning(f"Batch {batch_id}: No valid rows prepared for insertion. Skipping DB operation.")
             return True

        with conn.cursor() as cursor:
            sql_query = sql.SQL("""
                INSERT INTO {}
                (uid, summary, start_time, end_time, duration_hours, personnel,
                 role, clinical_pct, extracted_event_type, raw_data, batch_id, processing_status)
                VALUES %s
                ON CONFLICT (uid) DO UPDATE SET
                    personnel = EXCLUDED.personnel,
                    extracted_event_type = EXCLUDED.extracted_event_type,
                    role = EXCLUDED.role,
                    clinical_pct = EXCLUDED.clinical_pct,
                    raw_data = EXCLUDED.raw_data,
                    batch_id = EXCLUDED.batch_id,
                    processing_status = EXCLUDED.processing_status,
                    processing_date = CURRENT_TIMESTAMP
            """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA))

            logger.info(f"Batch {batch_id}: Executing execute_values with {len(rows_to_insert)} rows...")
            execute_values(cursor, sql_query, rows_to_insert)
            inserted_rows = cursor.rowcount
            logger.info(f"Batch {batch_id}: execute_values completed. Rows affected reported by cursor: {inserted_rows}")

            logger.info(f"Batch {batch_id}: Committing transaction...")
            conn.commit()
            logger.info(f"Batch {batch_id}: Commit successful.")

        logger.info(f"--- EXITING save_partial_processed_data for batch {batch_id} (Success) ---")
        return True

    except psycopg2.Error as db_e:
        logger.error(f"Batch {batch_id}: Database error saving partial data. Code: {db_e.pgcode}. Msg: {db_e.pgerror}", exc_info=True)
        if conn: conn.rollback()
        logger.info(f"--- EXITING save_partial_processed_data for batch {batch_id} (DB Error) ---")
        return False
    except Exception as e:
        logger.error(f"Batch {batch_id}: Unexpected error saving partial data: {e}", exc_info=True)
        if conn: conn.rollback()
        logger.info(f"--- EXITING save_partial_processed_data for batch {batch_id} (Unexpected Error) ---")
        return False
    finally:
        if conn:
            logger.info(f"Batch {batch_id}: Closing connection.")
            conn.close()


def get_processed_events(start_date=None, end_date=None, personnel=None, limit=1000):
    """
    Retrieves processed events from the database with optional filtering using SQLAlchemy.
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logger.error("Could not get SQLAlchemy engine to retrieve processed events.")
        return pd.DataFrame()

    try:
        # Use SQLAlchemy text for parameter binding (:param_name)
        sql_string = f"SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE 1=1"
        params = {'limit_val': limit} # Use dictionary for named parameters

        if start_date:
            sql_string += " AND start_time >= :start_date"
            params['start_date'] = start_date

        if end_date:
            sql_string += " AND end_time <= :end_date"
            params['end_date'] = end_date

        if personnel:
            # Handle potential list of personnel if needed, though current usage seems single
            if isinstance(personnel, list):
                 sql_string += " AND personnel = ANY(:personnel)" # Use ANY for list
                 params['personnel'] = personnel
            else:
                 sql_string += " AND personnel = :personnel"
                 params['personnel'] = personnel

        sql_string += " ORDER BY start_time DESC LIMIT :limit_val"

        # --- Temporarily simplified query for debugging ---
        sql_string = f"SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA} ORDER BY processing_date DESC LIMIT :limit_val"
        params = {'limit_val': limit}
        logger.info(f"Executing simplified query: {sql_string} with limit: {limit}")
        # --- End temporary query ---

        # Original query logic (commented out for debugging)
        # sql_string = f"SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE 1=1"
        # params = {'limit_val': limit} # Use dictionary for named parameters
        #
        # if start_date:
        #     sql_string += " AND start_time >= :start_date"
        #     params['start_date'] = start_date
        #
        # if end_date:
        #     sql_string += " AND end_time <= :end_date"
        #     params['end_date'] = end_date
        #
        # if personnel:
        #     # Handle potential list of personnel if needed, though current usage seems single
        #     if isinstance(personnel, list):
        #          sql_string += " AND personnel = ANY(:personnel)" # Use ANY for list
        #          params['personnel'] = personnel
        #     else:
        #          sql_string += " AND personnel = :personnel"
        #          params['personnel'] = personnel
        #
        # sql_string += " ORDER BY start_time DESC LIMIT :limit_val"

        # Use the SQLAlchemy engine with pandas
        # Pass parameters using the 'params' argument
        df = pd.read_sql_query(text(sql_string), engine, params=params)
        logger.info(f"Retrieved {len(df)} events from database using SQLAlchemy.")
        return df

    except Exception as e:
        logger.error(f"Error retrieving events from database using SQLAlchemy: {e}", exc_info=True)
        return pd.DataFrame()
    # Engine connection is managed by pandas/SQLAlchemy, no explicit close needed here


def save_calendar_file_to_db(filename, file_content):
    """
    Saves a calendar JSON file to the database, keeping only one file as "current".
    """
    conn = get_db_connection()
    if not conn:
        return False, None, False

    try:
        ensure_tables_exist()

        if isinstance(file_content, bytes):
            content_str = file_content.decode('utf-8')
        else:
            content_str = str(file_content)

        file_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()

        logger.info(f"Checking for duplicate file with hash: {file_hash[:10]}...")
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("""
                SELECT batch_id, processed FROM {}
                WHERE file_hash = %s
                ORDER BY upload_date DESC
                LIMIT 1
            """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)), (file_hash,))
            existing = cursor.fetchone()
        logger.info(f"Duplicate check complete. Found: {existing is not None}")

        if existing:
            existing_batch_id, _ = existing
            logger.info(f"Duplicate calendar file detected with hash {file_hash[:10]}... Batch ID: {existing_batch_id}")
            return True, existing_batch_id, True

        try:
            json_content = json.loads(content_str)
        except json.JSONDecodeError:
            logger.error("Failed to parse calendar file as JSON")
            return False, None, False

        batch_id = f"batch_{uuid.uuid4().hex[:8]}_{int(time.time())}"

        logger.info(f"Updating existing 'is_current=TRUE' flags...")
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("UPDATE {} SET is_current = FALSE WHERE is_current = TRUE")
                           .format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)))
            logger.info(f"Update complete. Inserting new record for '{filename}'...")
            cursor.execute(sql.SQL("""
                INSERT INTO {}
                (filename, file_content, file_hash, batch_id, processed, is_current)
                VALUES (%s, %s, %s, %s, FALSE, TRUE)
                RETURNING id
            """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)),
            (filename, json.dumps(json_content), file_hash, batch_id))
            file_id = cursor.fetchone()[0]
            logger.info(f"Insert complete. New file ID: {file_id}. Committing...")
            conn.commit()
            logger.info(f"Commit complete.")

        logger.info(f"Successfully saved calendar file '{filename}' (ID: {file_id})")
        return True, batch_id, False

    except Exception as e:
        logger.error(f"Error saving calendar file to database: {e}", exc_info=True)
        if conn: conn.rollback()
        return False, None, False
    finally:
        if conn: conn.close()


def mark_calendar_file_as_processed(batch_id):
    """
    Marks a calendar file as processed in the database.
    """
    conn = get_db_connection()
    if not conn: return False

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("UPDATE {} SET processed = TRUE WHERE batch_id = %s")
                           .format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)), (batch_id,))
            conn.commit()
            logger.info(f"Marked calendar file with batch ID {batch_id} as processed")
            return True
    except Exception as e:
        logger.error(f"Error marking calendar file as processed: {e}", exc_info=True)
        if conn: conn.rollback()
        return False
    finally:
        if conn: conn.close()


def get_calendar_file_by_batch_id(batch_id):
    """
    Retrieves a calendar file content from the database by batch ID.
    """
    conn = get_db_connection()
    if not conn: return None, None, None

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("""
                SELECT filename, file_content, processed
                FROM {} WHERE batch_id = %s LIMIT 1
            """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)), (batch_id,))
            result = cursor.fetchone()
            if result:
                logger.info(f"Retrieved calendar file with batch ID {batch_id}")
                return result # (filename, file_content, processed)
            else:
                logger.warning(f"No calendar file found with batch ID {batch_id}")
                return None, None, None
    except Exception as e:
        logger.error(f"Error retrieving calendar file by batch ID: {e}", exc_info=True)
        return None, None, None
    finally:
        if conn: conn.close()


def get_pending_calendar_files():
    """
    Retrieves calendar files that have not been processed yet.
    """
    conn = get_db_connection()
    if not conn: return []

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("""
                SELECT batch_id, filename, file_content
                FROM {} WHERE processed = FALSE ORDER BY upload_date ASC
            """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)))
            results = cursor.fetchall()
            logger.info(f"Found {len(results)} pending calendar files")
            return results
    except Exception as e:
        logger.error(f"Error retrieving pending calendar files: {e}", exc_info=True)
        return []
    finally:
        if conn: conn.close()


def get_current_calendar_file():
    """
    Retrieves the current active calendar file from the database.
    """
    conn = get_db_connection()
    if not conn: return None, None, None, None

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("""
                SELECT id, batch_id, filename, file_content
                FROM {} WHERE is_current = TRUE LIMIT 1
            """).format(sql.Identifier(settings.DB_TABLE_CALENDAR_FILES)))
            result = cursor.fetchone()
            if result:
                logger.info(f"Retrieved current calendar file: ID {result[0]}, Batch {result[1]}")
                return result # (file_id, batch_id, filename, file_content)
            else:
                logger.warning("No current calendar file found")
                return None, None, None, None
    except Exception as e:
        logger.error(f"Error retrieving current calendar file: {e}", exc_info=True)
        return None, None, None, None
    finally:
        if conn: conn.close()


def count_unique_events_in_database():
    """
    Counts the number of unique calendar events currently in the database.
    """
    conn = get_db_connection()
    if not conn: return 0

    try:
        with conn.cursor() as cursor:
            cursor.execute(sql.SQL("SELECT COUNT(DISTINCT uid) FROM {}")
                           .format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)))
            result = cursor.fetchone()
            count = result[0] if result else 0
            logger.info(f"Found {count} unique calendar events in database")
            return count
    except Exception as e:
        logger.error(f"Error counting unique events: {e}", exc_info=True)
        return 0
    finally:
        if conn: conn.close()


def get_processed_events_by_batch(batch_id):
    """
    Retrieves processed events from the database for a specific batch using SQLAlchemy.
    """
    engine = get_sqlalchemy_engine()
    if not engine:
        logger.error(f"Could not get SQLAlchemy engine to retrieve events for batch {batch_id}")
        return pd.DataFrame()

    try:
        # Use SQLAlchemy text query with named parameters
        sql_string = f"SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE batch_id = :batch_id"
        params = {'batch_id': batch_id}

        # Use the SQLAlchemy engine with pandas
        df = pd.read_sql_query(text(sql_string), engine, params=params)

        if not df.empty:
            logger.info(f"Retrieved {len(df)} events for batch {batch_id} using SQLAlchemy")
        else:
            logger.warning(f"No events found for batch {batch_id}")
        return df
    except Exception as e:
        logger.error(f"Error retrieving events for batch {batch_id}: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        if conn: conn.close()
