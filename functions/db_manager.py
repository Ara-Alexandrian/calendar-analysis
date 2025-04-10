# functions/db_manager.py
import logging
import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from config import settings

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.
    Returns None if DB_ENABLED is False or if connection fails.
    """
    if not settings.DB_ENABLED:
        logger.info("Database persistence is disabled in settings.")
        return None
    
    try:
        # Parse special characters in password properly by URL encoding
        import urllib.parse
        encoded_password = urllib.parse.quote_plus(settings.DB_PASSWORD)
        
        # Try using connection string format to handle special characters
        conn_string = f"postgresql://{settings.DB_USER}:{encoded_password}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        conn = psycopg2.connect(conn_string)
        
        logger.info(f"Successfully connected to PostgreSQL at {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
        return conn
    except Exception as e:
        # Try alternative connection method if the first one fails
        try:
            conn = psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database=settings.DB_NAME,
                user=settings.DB_USER,
                password=settings.DB_PASSWORD
            )
            logger.info(f"Successfully connected to PostgreSQL using direct parameters")
            return conn
        except Exception as e2:
            logger.error(f"Failed to connect to PostgreSQL with both methods: {e}, {e2}")
            
            # Fall back to in-memory processing when database connection fails
            logger.warning("Falling back to in-memory processing due to database connection failure")
            return None

def ensure_tables_exist():
    """
    Ensures that the necessary database tables exist, creating them if they don't.
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            # Create processed events table if it doesn't exist
            cursor.execute("""
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
                    raw_data JSONB,
                    batch_id TEXT,
                    processing_status TEXT,
                    calendar_file_id INTEGER,
                    processing_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """.format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)))
            
            # Create personnel config table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    canonical_name TEXT UNIQUE,
                    role TEXT,
                    clinical_pct FLOAT,
                    variations JSONB,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """.format(sql.Identifier(settings.DB_TABLE_PERSONNEL)))
            
            # Create calendar files table to store raw JSON files - now keeps only the most recent file
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS calendar_files (
                    id SERIAL PRIMARY KEY,
                    filename TEXT,
                    file_content JSONB,
                    file_hash TEXT,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE,
                    batch_id TEXT,
                    is_current BOOLEAN DEFAULT TRUE,
                    CONSTRAINT unique_current_file CHECK (
                        (is_current = FALSE) OR 
                        (is_current = TRUE AND (SELECT COUNT(*) FROM calendar_files WHERE is_current = TRUE) <= 1)
                    )
                )
            """)
            
            # Add an index on the uid field for faster lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_processed_events_uid 
                ON {} (uid)
            """.format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)))
            
            conn.commit()
            logger.info("Database tables created or already exist")
            return True
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

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
        
        # Create the tables if they don't exist
        ensure_tables_exist()
        
        # Prepare data for insertion
        # For each row, determine the role and clinical_pct from the personnel config
        from functions import config_manager
        
        # Convert DataFrame to list of tuples
        rows_to_insert = []
        for _, row in df.iterrows():
            personnel = row['personnel']
            role = config_manager.get_role(personnel)
            clinical_pct = config_manager.get_clinical_pct(personnel)
            
            # Convert to Python native types and handle None values
            summary = str(row['summary']) if pd.notna(row['summary']) else None
            uid = str(row['uid']) if pd.notna(row['uid']) else None
            
            # Convert the row to a dict for JSON storage (excluding large objects)
            row_dict = row.to_dict()
            for key in ['start_time', 'end_time']:  # Remove datetime objects
                if key in row_dict:
                    row_dict[key] = str(row_dict[key])
            
            # Get processing status if it exists
            processing_status = row.get('processing_status', 'assigned')
            
            rows_to_insert.append((
                uid,
                summary,
                row['start_time'],
                row['end_time'],
                row['duration_hours'],
                personnel,
                role,
                clinical_pct,
                row_dict,  # Store the full row as JSON
                batch_id,
                processing_status
            ))
        
        # Insert data with ON CONFLICT DO UPDATE to handle duplicates
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                f"""
                INSERT INTO {settings.DB_TABLE_PROCESSED_DATA} 
                (uid, summary, start_time, end_time, duration_hours, personnel, 
                role, clinical_pct, raw_data, batch_id, processing_status)
                VALUES %s
                ON CONFLICT (uid) DO UPDATE SET
                    personnel = EXCLUDED.personnel,
                    role = EXCLUDED.role,
                    clinical_pct = EXCLUDED.clinical_pct,
                    raw_data = EXCLUDED.raw_data,
                    batch_id = EXCLUDED.batch_id,
                    processing_status = EXCLUDED.processing_status,
                    processing_date = CURRENT_TIMESTAMP
                """,
                rows_to_insert
            )
            conn.commit()
            
        logger.info(f"Successfully saved {len(rows_to_insert)} rows to database table {settings.DB_TABLE_PROCESSED_DATA}")
        
        # If we have a batch_id, mark the calendar file as processed
        if batch_id:
            mark_calendar_file_as_processed(batch_id)
            
        return True
        
    except Exception as e:
        logger.error(f"Error saving data to database: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

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
        # Create the tables if they don't exist
        ensure_tables_exist()
        
        # Prepare data for insertion
        rows_to_insert = []
        for name, details in config_dict.items():
            rows_to_insert.append((
                name,
                details.get('role', ''),
                details.get('clinical_pct', 0.0),
                details.get('variations', [])
            ))
        
        with conn.cursor() as cursor:
            # First, delete existing records (we'll replace them)
            cursor.execute(f"DELETE FROM {settings.DB_TABLE_PERSONNEL}")
            
            # Then insert new records
            execute_values(
                cursor,
                f"""
                INSERT INTO {settings.DB_TABLE_PERSONNEL} 
                (canonical_name, role, clinical_pct, variations)
                VALUES %s
                """,
                rows_to_insert
            )
            conn.commit()
            
        logger.info(f"Successfully saved {len(rows_to_insert)} personnel entries to database")
        return True
        
    except Exception as e:
        logger.error(f"Error saving personnel config to database: {e}")
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
        dict: Personnel configuration dictionary, or None if loading fails.
    """
    conn = get_db_connection()
    if not conn:
        return None
    
    try:
        with conn.cursor() as cursor:
            # Check if the table exists
            cursor.execute("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_name = %s
                )
            """, (settings.DB_TABLE_PERSONNEL,))
            
            if not cursor.fetchone()[0]:
                logger.warning(f"Table {settings.DB_TABLE_PERSONNEL} does not exist yet.")
                return None
            
            # Fetch the personnel config
            cursor.execute(f"SELECT canonical_name, role, clinical_pct, variations FROM {settings.DB_TABLE_PERSONNEL}")
            rows = cursor.fetchall()
            
            # Convert to dictionary format
            config_dict = {}
            for name, role, clinical_pct, variations in rows:
                config_dict[name] = {
                    'role': role,
                    'clinical_pct': clinical_pct,
                    'variations': variations
                }
            
            logger.info(f"Successfully loaded {len(config_dict)} personnel entries from database")
            return config_dict
            
    except Exception as e:
        logger.error(f"Error loading personnel config from database: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_processed_events(start_date=None, end_date=None, personnel=None, limit=1000):
    """
    Retrieves processed events from the database with optional filtering.
    
    Args:
        start_date: Optional start date filter (datetime or string)
        end_date: Optional end date filter (datetime or string)
        personnel: Optional personnel name filter
        limit: Maximum number of records to return
        
    Returns:
        pandas.DataFrame: DataFrame containing the retrieved events, or empty DataFrame if retrieval fails.
    """
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = f"SELECT * FROM {settings.DB_TABLE_PROCESSED_DATA} WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND start_time >= %s"
            params.append(start_date)
        
        if end_date:
            query += " AND end_time <= %s"
            params.append(end_date)
        
        if personnel:
            query += " AND personnel = %s"
            params.append(personnel)
        
        query += f" ORDER BY start_time DESC LIMIT {limit}"
        
        # Execute query and fetch results into DataFrame
        df = pd.read_sql_query(query, conn, params=params)
        logger.info(f"Retrieved {len(df)} events from database")
        return df
        
    except Exception as e:
        logger.error(f"Error retrieving events from database: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

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
    if df is None or df.empty:
        logger.warning(f"No partial data to save for batch {batch_id}")
        return False
    
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        # Ensure tables exist
        ensure_tables_exist()
        
        # Add batch_id column if needed
        with conn.cursor() as cursor:
            cursor.execute(f"""
                ALTER TABLE {settings.DB_TABLE_PROCESSED_DATA} 
                ADD COLUMN IF NOT EXISTS batch_id TEXT,
                ADD COLUMN IF NOT EXISTS processing_status TEXT
            """)
            conn.commit()
        
        # Prepare data for insertion, similar to save_processed_data_to_db
        from functions import config_manager
        
        rows_to_insert = []
        for _, row in df.iterrows():
            personnel = row.get('personnel', 'Unknown')
            # For partial processing, the personnel might be in extracted_personnel
            if 'personnel' not in row and 'extracted_personnel' in row:
                # Handle both string and list formats from LLM extraction
                extracted = row['extracted_personnel']
                if isinstance(extracted, list) and len(extracted) > 0:
                    personnel = extracted[0]
                elif isinstance(extracted, str):
                    personnel = extracted
            
            role = config_manager.get_role(personnel)
            clinical_pct = config_manager.get_clinical_pct(personnel)
            
            summary = str(row['summary']) if pd.notna(row.get('summary')) else None
            uid = str(row['uid']) if pd.notna(row.get('uid')) else None
            
            # Convert the row to a dict for JSON storage
            row_dict = row.to_dict()
            for key in ['start_time', 'end_time']:
                if key in row_dict and row_dict[key] is not None:
                    row_dict[key] = str(row_dict[key])
            
            # Determine processing status
            status = 'processing'
            if 'extracted_personnel' in row and row['extracted_personnel'] != []:
                status = 'extracted'
            if 'assigned_personnel' in row:
                status = 'assigned'
            
            rows_to_insert.append((
                uid,
                summary,
                row.get('start_time'),
                row.get('end_time'),
                row.get('duration_hours'),
                personnel,
                role,
                clinical_pct,
                row_dict,
                batch_id,
                status
            ))
        
        # Insert or update data
        with conn.cursor() as cursor:
            execute_values(
                cursor,
                f"""
                INSERT INTO {settings.DB_TABLE_PROCESSED_DATA} 
                (uid, summary, start_time, end_time, duration_hours, personnel, role, 
                clinical_pct, raw_data, batch_id, processing_status)
                VALUES %s
                ON CONFLICT (uid) DO UPDATE SET
                    personnel = EXCLUDED.personnel,
                    raw_data = EXCLUDED.raw_data,
                    processing_status = EXCLUDED.processing_status
                """,
                rows_to_insert
            )
            conn.commit()
            
        logger.info(f"Successfully saved {len(rows_to_insert)} partial rows for batch {batch_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving partial data to database: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_latest_processing_status(batch_id=None):
    """
    Get the status of the latest processing batch.
    
    Args:
        batch_id: Optional batch ID to filter by
        
    Returns:
        dict: Processing status information including completion percentage
    """
    conn = get_db_connection()
    if not conn:
        return {"status": "unknown", "message": "Database not available"}
    
    try:
        with conn.cursor() as cursor:
            if (batch_id):
                # Get status for specific batch
                cursor.execute(f"""
                    SELECT 
                        processing_status, 
                        COUNT(*) as count
                    FROM {settings.DB_TABLE_PROCESSED_DATA}
                    WHERE batch_id = %s
                    GROUP BY processing_status
                """, (batch_id,))
            else:
                # Get status for latest batch
                cursor.execute(f"""
                    WITH latest_batch AS (
                        SELECT batch_id
                        FROM {settings.DB_TABLE_PROCESSED_DATA}
                        WHERE batch_id IS NOT NULL
                        ORDER BY processing_date DESC
                        LIMIT 1
                    )
                    SELECT 
                        processing_status, 
                        COUNT(*) as count
                    FROM {settings.DB_TABLE_PROCESSED_DATA}
                    WHERE batch_id = (SELECT batch_id FROM latest_batch)
                    GROUP BY processing_status
                """)
            
            results = cursor.fetchall()
            
            if not results:
                return {"status": "none", "message": "No processing batches found"}
            
            # Calculate completion statistics
            status_counts = {status: count for status, count in results}
            total = sum(status_counts.values())
            processing = status_counts.get('processing', 0)
            extracted = status_counts.get('extracted', 0)
            assigned = status_counts.get('assigned', 0)
            
            pct_complete = ((extracted + assigned) / total * 100) if total > 0 else 0
            
            # Determine overall status
            if processing == 0 and total > 0:
                status = "complete"
                message = f"Processing complete: {total} events processed"
            elif total > 0:
                status = "in_progress"
                message = f"Processing in progress: {pct_complete:.1f}% complete ({processing} remaining)"
            else:
                status = "unknown"
                message = "Unknown processing status"
            
            return {
                "status": status,
                "message": message,
                "total": total,
                "processing": processing,
                "extracted": extracted,
                "assigned": assigned,
                "pct_complete": pct_complete,
                "batch_id": batch_id
            }
            
    except Exception as e:
        logger.error(f"Error getting processing status: {e}")
        return {"status": "error", "message": f"Error: {str(e)}"}
    finally:
        if conn:
            conn.close()

def save_calendar_file_to_db(filename, file_content):
    """
    Saves a calendar JSON file to the database, keeping only one file as "current".
    
    Args:
        filename: The name of the uploaded file
        file_content: The raw content of the JSON file (bytes or string)
        
    Returns:
        tuple: (success, batch_id, is_duplicate) where:
            - success (bool): True if saved successfully
            - batch_id (str): Batch ID for processing (new or existing)
            - is_duplicate (bool): True if this exact file was already processed
    """
    import hashlib
    import json
    import uuid
    import time
    
    conn = get_db_connection()
    if not conn:
        return False, None, False
    
    try:
        # Ensure tables exist
        ensure_tables_exist()
        
        # Calculate hash of file content to identify duplicates
        if isinstance(file_content, bytes):
            content_str = file_content.decode('utf-8')
        else:
            content_str = str(file_content)
            
        file_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()
        
        # Check if this exact file has been processed before
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT batch_id, processed FROM calendar_files
                WHERE file_hash = %s
                ORDER BY upload_date DESC
                LIMIT 1
            """, (file_hash,))
            
            existing = cursor.fetchone()
            
            if existing:
                existing_batch_id, is_processed = existing
                logger.info(f"Duplicate calendar file detected with hash {file_hash[:10]}...")
                
                # Return existing batch ID if already processed
                return True, existing_batch_id, True
        
        # Parse JSON content to store as JSONB in PostgreSQL
        try:
            json_content = json.loads(content_str)
        except json.JSONDecodeError:
            logger.error("Failed to parse calendar file as JSON")
            return False, None, False
        
        # Generate a unique batch ID for this upload
        batch_id = f"batch_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # Before inserting, mark all existing calendar files as not current
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE calendar_files
                SET is_current = FALSE
            """)
            
            # Now insert the new file as the current one
            cursor.execute("""
                INSERT INTO calendar_files
                (filename, file_content, file_hash, batch_id, processed, is_current)
                VALUES (%s, %s, %s, %s, FALSE, TRUE)
                RETURNING id
            """, (filename, json.dumps(json_content), file_hash, batch_id))
            
            file_id = cursor.fetchone()[0]
            conn.commit()
            
        logger.info(f"Successfully saved calendar file '{filename}' to database with ID {file_id} and set as current file")
        return True, batch_id, False
        
    except Exception as e:
        logger.error(f"Error saving calendar file to database: {e}")
        if conn:
            conn.rollback()
        return False, None, False
    finally:
        if conn:
            conn.close()

def mark_calendar_file_as_processed(batch_id):
    """
    Marks a calendar file as processed in the database.
    
    Args:
        batch_id: The batch ID associated with the file
        
    Returns:
        bool: True if marked successfully, False otherwise
    """
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE calendar_files
                SET processed = TRUE
                WHERE batch_id = %s
            """, (batch_id,))
            
            conn.commit()
            logger.info(f"Marked calendar file with batch ID {batch_id} as processed")
            return True
            
    except Exception as e:
        logger.error(f"Error marking calendar file as processed: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()

def get_calendar_file_by_batch_id(batch_id):
    """
    Retrieves a calendar file content from the database by batch ID.
    
    Args:
        batch_id: The batch ID associated with the file
        
    Returns:
        tuple: (filename, file_content, processed) or (None, None, None) if not found
    """
    conn = get_db_connection()
    if not conn:
        return None, None, None
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT filename, file_content, processed 
                FROM calendar_files
                WHERE batch_id = %s
                LIMIT 1
            """, (batch_id,))
            
            result = cursor.fetchone()
            
            if result:
                filename, file_content, processed = result
                logger.info(f"Retrieved calendar file '{filename}' with batch ID {batch_id}")
                return filename, file_content, processed
            else:
                logger.warning(f"No calendar file found with batch ID {batch_id}")
                return None, None, None
            
    except Exception as e:
        logger.error(f"Error retrieving calendar file: {e}")
        return None, None, None
    finally:
        if conn:
            conn.close()

def get_pending_calendar_files():
    """
    Retrieves calendar files that have not been processed yet.
    
    Returns:
        list: List of tuples (batch_id, filename, file_content)
    """
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT batch_id, filename, file_content
                FROM calendar_files
                WHERE processed = FALSE
                ORDER BY upload_date ASC
            """)
            
            results = cursor.fetchall()
            
            if results:
                logger.info(f"Found {len(results)} pending calendar files to process")
                return results
            else:
                logger.info("No pending calendar files to process")
                return []
            
    except Exception as e:
        logger.error(f"Error retrieving pending calendar files: {e}")
        return []
    finally:
        if conn:
            conn.close()

def get_current_calendar_file():
    """
    Retrieves the current active calendar file from the database.
    This is the most recently uploaded file marked as 'is_current'.
    
    Returns:
        tuple: (file_id, batch_id, filename, file_content) or (None, None, None, None) if not found
    """
    conn = get_db_connection()
    if not conn:
        return None, None, None, None
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, batch_id, filename, file_content
                FROM calendar_files
                WHERE is_current = TRUE
                LIMIT 1
            """)
            
            result = cursor.fetchone()
            
            if result:
                file_id, batch_id, filename, file_content = result
                logger.info(f"Retrieved current calendar file: '{filename}' (ID: {file_id}, Batch: {batch_id})")
                return file_id, batch_id, filename, file_content
            else:
                logger.warning("No current calendar file found in database")
                return None, None, None, None
            
    except Exception as e:
        logger.error(f"Error retrieving current calendar file: {e}")
        return None, None, None, None
    finally:
        if conn:
            conn.close()

def count_unique_events_in_database():
    """
    Counts the number of unique calendar events currently in the database.
    
    Returns:
        int: Count of unique events or 0 if there was an error
    """
    conn = get_db_connection()
    if not conn:
        return 0
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT COUNT(DISTINCT uid) 
                FROM {settings.DB_TABLE_PROCESSED_DATA}
            """)
            
            result = cursor.fetchone()
            count = result[0] if result else 0
            
            logger.info(f"Found {count} unique calendar events in database")
            return count
            
    except Exception as e:
        logger.error(f"Error counting unique events: {e}")
        return 0
    finally:
        if conn:
            conn.close()