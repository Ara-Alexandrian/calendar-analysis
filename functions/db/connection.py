# functions/db/connection.py
import logging
import logging
import psycopg2
import time
import urllib.parse
from sqlalchemy import create_engine, exc as sqlalchemy_exc
from config import settings

logger = logging.getLogger(__name__)

def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.
    Returns None if DB_ENABLED is False or if connection fails.

    Creates the database if it doesn't exist.
    """
    if not settings.DB_ENABLED:
        logger.info("Database persistence is disabled in settings.")
        return None

    # First try to connect to default database to check if our target database exists
    try:
        # Connect to 'postgres' default database to check if our database exists
        default_conn = psycopg2.connect(
            host=settings.DB_HOST,
            port=settings.DB_PORT,
            database='postgres',  # Default database that should exist
            user=settings.DB_USER,
            password=settings.DB_PASSWORD
        )
        default_conn.autocommit = True  # Required for creating database

        # Check if our database exists
        cursor = default_conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings.DB_NAME.lower(),))
        db_exists = cursor.fetchone()

        if not db_exists:
            logger.warning(f"Database '{settings.DB_NAME}' does not exist. Creating it now...")
            try:
                # Create the database if it doesn't exist
                cursor.execute(f"CREATE DATABASE {settings.DB_NAME}")
                logger.info(f"Database '{settings.DB_NAME}' created successfully.")
            except Exception as create_e:
                logger.error(f"Failed to create database: {create_e}")

        cursor.close()
        default_conn.close()
    except Exception as check_e:
        logger.error(f"Error checking/creating database: {check_e}")

    # Now try to connect to our target database
    try:
        # Parse special characters in password properly by URL encoding
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
            logger.error(f"Failed to connect to PostgreSQL database '{settings.DB_NAME}': {e2}")

            # Fall back to in-memory processing when database connection fails
            logger.warning("Falling back to in-memory processing due to database connection failure")
            return None

def get_db_connection_with_retry(max_retries=3, retry_delay=2):
    """
    Attempts to connect to the database with retry logic.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Seconds to wait between retries

    Returns:
        Connection object or None if all attempts fail
    """
    retry_count = 0
    last_error = None

    while retry_count < max_retries:
        try:
            conn = get_db_connection()
            if conn:
                # Test the connection with a simple query
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    cursor.fetchone()
                logger.info(f"Successfully connected to database after {retry_count + 1} attempts")
                return conn
        except Exception as e:
            last_error = e
            retry_count += 1
            logger.warning(f"Database connection attempt {retry_count} failed: {e}")
            if retry_count < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    logger.error(f"Failed to connect to database after {max_retries} attempts: {last_error}")
    return None


def get_sqlalchemy_engine():
    """
    Creates and returns a SQLAlchemy engine for the PostgreSQL database.
    Returns None if DB_ENABLED is False or if connection fails.
    """
    if not settings.DB_ENABLED:
        logger.info("Database persistence is disabled in settings.")
        return None

    try:
        # Ensure the database exists using the raw connection logic first
        # (SQLAlchemy doesn't typically create databases)
        conn_check = get_db_connection()
        if conn_check:
            conn_check.close() # Close the check connection
        else:
            # If get_db_connection failed (which includes creation attempt), engine will likely fail too
            logger.error("Initial DB connection/creation check failed. Cannot create SQLAlchemy engine.")
            return None

        # Construct the SQLAlchemy connection string (DSN)
        # Format: postgresql+psycopg2://user:password@host:port/database
        encoded_password = urllib.parse.quote_plus(settings.DB_PASSWORD)
        db_url = f"postgresql+psycopg2://{settings.DB_USER}:{encoded_password}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

        # Create the engine
        # Consider adding pool settings for production: pool_size, max_overflow, etc.
        engine = create_engine(db_url, echo=False) # echo=True for debugging SQL

        # Test the connection
        try:
            with engine.connect() as connection:
                logger.info(f"Successfully created SQLAlchemy engine and connected to PostgreSQL at {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
            return engine
        except sqlalchemy_exc.OperationalError as op_err:
            logger.error(f"SQLAlchemy engine failed to connect: {op_err}")
            return None

    except Exception as e:
        logger.error(f"Failed to create SQLAlchemy engine: {e}", exc_info=True)
        return None
