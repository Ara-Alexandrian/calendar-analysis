# functions/db/__init__.py

"""
Database interaction package.

This package centralizes all database operations for the calendar analysis application.
It provides functions for connecting to the database, managing the schema,
and performing CRUD operations on various data tables.
"""

# Import key functions from submodules to make them directly accessible
from .connection import get_db_connection, get_db_connection_with_retry
from .schema import ensure_tables_exist, clear_database_tables
from .operations import (
    save_processed_data_to_db,
    save_processed_data_in_batches,
    save_partial_processed_data,
    get_processed_events,
    save_calendar_file_to_db,
    mark_calendar_file_as_processed,
    get_calendar_file_by_batch_id,
    get_pending_calendar_files,
    get_current_calendar_file,
    count_unique_events_in_database,
    get_processed_events_by_batch
)
from .personnel_ops import (
    save_personnel_config_to_db,
    load_personnel_config_from_db
)
from .status_ops import get_latest_processing_status, check_batch_status # Add check_batch_status

# Define __all__ for explicit public API (optional but good practice)
__all__ = [
    # Connection functions
    'get_db_connection',
    'get_db_connection_with_retry',

    # Schema functions
    'ensure_tables_exist',
    'clear_database_tables',

    # General operations
    'save_processed_data_to_db',
    'save_processed_data_in_batches',
    'save_partial_processed_data',
    'get_processed_events',
    'save_calendar_file_to_db',
    'mark_calendar_file_as_processed',
    'get_calendar_file_by_batch_id',
    'get_pending_calendar_files',
    'get_current_calendar_file',
    'count_unique_events_in_database',
    'get_processed_events_by_batch',

    # Personnel operations
    'save_personnel_config_to_db',
    'load_personnel_config_from_db',

    # Status operations
    'get_latest_processing_status',
    'check_batch_status', # Add check_batch_status to __all__
]
