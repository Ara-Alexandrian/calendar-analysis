# pages/4_🔍_Database_Viewer.py
import streamlit as st
import pandas as pd
import logging
import psycopg2
import json
from config import settings
from sqlalchemy import create_engine, text

# Ensure project root is in path
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from functions import db as db_manager # Corrected import

logger = logging.getLogger(__name__)

st.set_page_config(page_title="Database Viewer", layout="wide")

st.markdown("# 4. Database Viewer")
st.markdown("View and inspect database tables to troubleshoot data persistence issues.")

# First, check if DB is enabled and connection is working
if not settings.DB_ENABLED:
    st.error("Database is disabled in settings. Please enable it by setting DB_ENABLED = True in the settings.py file.")
    st.stop()

# Try to establish a connection
conn = db_manager.get_db_connection()
if not conn:
    st.error("Unable to connect to the PostgreSQL database. Please check your connection settings and make sure the database server is running.")

    # Show the current settings
    st.subheader("Current Database Settings")
    st.code(f"""
    DB_HOST = "{settings.DB_HOST}"
    DB_PORT = "{settings.DB_PORT}"
    DB_NAME = "{settings.DB_NAME}"
    DB_USER = "{settings.DB_USER}"
    DB_PASSWORD = "{'*' * len(settings.DB_PASSWORD)}"  # Masked for security
    DB_TABLE_PROCESSED_DATA = "{settings.DB_TABLE_PROCESSED_DATA}"
    DB_TABLE_PERSONNEL = "{settings.DB_TABLE_PERSONNEL}"
    """)

    st.subheader("Troubleshooting Tips")
    st.markdown("""
    1. Make sure the PostgreSQL server is running at the specified host and port
    2. Verify that the database exists and the user has access permissions
    3. Check for any special characters in the password that might need escaping
    4. Ensure the psycopg2-binary package is installed (`pip install psycopg2-binary`)
    """)

    # Add a button to create the database if it doesn't exist
    if st.button("Create Database"):
        try:
            # Connect to the default 'postgres' database to create our application database
            import psycopg2
            conn_create = psycopg2.connect(
                host=settings.DB_HOST,
                port=settings.DB_PORT,
                database="postgres",  # Connect to default DB
                user=settings.DB_USER,
                password=settings.DB_PASSWORD
            )
            conn_create.autocommit = True  # Required for CREATE DATABASE

            with conn_create.cursor() as cursor:
                # Check if database already exists
                cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{settings.DB_NAME}'")
                exists = cursor.fetchone()

                if not exists:
                    # Create the database
                    cursor.execute(f"CREATE DATABASE {settings.DB_NAME}")
                    st.success(f"Database '{settings.DB_NAME}' created successfully! Please refresh the page.")
                else:
                    st.info(f"Database '{settings.DB_NAME}' already exists.")

            conn_create.close()
        except Exception as e:
            st.error(f"Failed to create database: {e}")

    # Add a button to install psycopg2 if it's not already installed
    if st.button("Install Required Packages"):
        st.info("Installing psycopg2-binary and other required packages...")
        result = os.system("pip install psycopg2-binary")
        if result == 0:
            st.success("Packages installed successfully! Please restart the application.")
        else:
            st.error("Failed to install packages. Please install manually: 'pip install psycopg2-binary'")

    st.stop()

# Create SQLAlchemy engine for pandas operations
try:
    import urllib.parse
    encoded_password = urllib.parse.quote_plus(settings.DB_PASSWORD)
    engine = create_engine(f"postgresql://{settings.DB_USER}:{encoded_password}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    logger.info("SQLAlchemy engine created successfully")
except Exception as e:
    st.error(f"Failed to create SQLAlchemy engine: {e}")
    st.warning("Falling back to direct database connection (may show pandas warnings)")
    engine = None

# Function to get list of tables in the database
def get_all_tables():
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            tables = [row[0] for row in cursor.fetchall()]
            return tables
    except Exception as e:
        st.error(f"Error fetching tables: {e}")
        return []

# Function to get column information for a table
def get_table_columns(table_name):
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            columns = cursor.fetchall()
            return columns
    except Exception as e:
        st.error(f"Error fetching column info: {e}")
        return []

# Function to count rows in a table
def count_table_rows(table_name):
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}") # Use f-string carefully, table name is validated
            count = cursor.fetchone()[0]
            return count
    except Exception as e:
        st.error(f"Error counting rows: {e}")
        return 0

# Function to query data from a table with custom SQL
def execute_custom_query(query, params=None):
    try:
        if engine is not None:
            # Use SQLAlchemy engine if available
            if params:
                # For parameterized queries with SQLAlchemy
                query_obj = text(query)
                df = pd.read_sql_query(query_obj, engine, params=params)
            else:
                df = pd.read_sql_query(query, engine)
        else:
            # Fall back to direct connection if engine creation failed
            df = pd.read_sql_query(query, conn, params=params)
        return df
    except Exception as e:
        st.error(f"Error executing query: {e}")
        return pd.DataFrame()

# Function to fetch table data with pagination
def get_table_data(table_name, limit=100, offset=0, where_clause=""):
    try:
        query = f"SELECT * FROM {table_name}" # Use f-string carefully, table name is validated
        if where_clause:
            query += f" WHERE {where_clause}" # Use f-string carefully, column name is validated
        query += f" LIMIT {limit} OFFSET {offset}"

        if engine is not None:
            # Use SQLAlchemy engine if available
            df = pd.read_sql_query(query, engine)
        else:
            # Fall back to direct connection
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Get list of tables
tables = get_all_tables()

if not tables:
    st.warning("No tables found in the database. You may need to upload and process data first.")

    # Button to ensure tables exist
    if st.button("Create database tables"):
        success = db_manager.ensure_tables_exist()
        if success:
            st.success("Database tables were successfully created!")
            st.rerun()
        else:
            st.error("Failed to create database tables. Check the logs for more information.")
else:
    # Database information
    st.subheader("Database Information")
    st.info(f"Connected to PostgreSQL database '{settings.DB_NAME}' at {settings.DB_HOST}:{settings.DB_PORT} as user '{settings.DB_USER}'")
    st.success(f"Found {len(tables)} tables in the database")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Table Explorer", "Custom Query", "Status Dashboard"])

    with tab1:
        # Table selector
        selected_table = st.selectbox("Select Table", tables)

        if selected_table:
            # Table information
            row_count = count_table_rows(selected_table)
            st.info(f"Table '{selected_table}' contains {row_count} rows")

            # Show table schema
            st.subheader("Table Schema")
            columns = get_table_columns(selected_table)
            if columns:
                schema_df = pd.DataFrame(columns, columns=["Column", "Data Type", "Nullable"])
                st.dataframe(schema_df)

            # Data pagination controls
            st.subheader("Table Data")

            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
            with col2:
                page_number = st.number_input("Page", min_value=1, value=1, step=1)
                offset = (page_number - 1) * page_size
            with col3:
                # Optional simple filter
                filter_column = st.selectbox("Filter by column", ["None"] + [col[0] for col in columns])

            if filter_column != "None":
                filter_value = st.text_input("Filter value (exact match)")
                # Basic validation/escaping might be needed here for security if values can contain quotes
                where_clause = f"{filter_column} = '{filter_value}'" if filter_value else ""
            else:
                where_clause = ""

            # Fetch and display data
            table_data = get_table_data(selected_table, limit=page_size, offset=offset, where_clause=where_clause)

            # Handle special column formatting
            for col in table_data.columns:
                # Format JSONB columns for better display
                if col == 'raw_data' or col == 'variations' or col == 'file_content':
                    try:
                        # Convert JSON strings to readable format
                        table_data[col] = table_data[col].apply(
                            lambda x: json.dumps(x, indent=2) if isinstance(x, dict) else str(x)[:100] + '...' if x else None
                        )
                    except:
                        pass

            st.dataframe(table_data, use_container_width=True)

            # Download button for the table data
            csv_data = table_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download data as CSV",
                data=csv_data,
                file_name=f"{selected_table}_data.csv",
                mime="text/csv"
            )

            # Pagination controls
            st.caption(f"Showing rows {offset+1} to {min(offset+page_size, row_count)} of {row_count}")

            if row_count > page_size:
                col1_page, col2_page = st.columns(2)
                with col1_page:
                    if st.button("Previous Page", disabled=(page_number <= 1)):
                        # This needs state management to work correctly across reruns
                        st.warning("Pagination buttons require state management (not fully implemented here).")
                with col2_page:
                    if st.button("Next Page", disabled=(offset + page_size >= row_count)):
                        st.warning("Pagination buttons require state management (not fully implemented here).")

    with tab2:
        st.subheader("Custom SQL Query")
        st.warning("Be careful with custom queries - avoid modifying data unless you know what you're doing")

        query = st.text_area("SQL Query", height=150,
                            placeholder="Example: SELECT * FROM processed_events WHERE start_time > '2023-01-01' LIMIT 100")

        col1_query, col2_query = st.columns([1, 3])
        with col1_query:
            execute = st.button("Execute Query")
        with col2_query:
            st.markdown("**Tables**: " + ", ".join(tables))

        if execute and query:
            st.subheader("Query Results")
            results = execute_custom_query(query)

            if not results.empty:
                st.dataframe(results, use_container_width=True)

                # Download results
                csv_results = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download results as CSV",
                    data=csv_results,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            else:
                st.info("Query returned no results")

    with tab3:
        st.subheader("Database Status Dashboard")

        # Table statistics
        st.markdown("### Table Statistics")
        stats_data = []
        for table in tables:
            row_count = count_table_rows(table)
            stats_data.append({"Table": table, "Row Count": row_count})

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

        # Processing Status
        st.markdown("### Processing Status")

        # Get all batch IDs for processed data
        try:
            batch_stats = execute_custom_query(f"""
                SELECT batch_id, COUNT(*) as event_count,
                       MIN(processing_date) as start_date,
                       MAX(processing_date) as end_date
                FROM {settings.DB_TABLE_PROCESSED_DATA}
                WHERE batch_id IS NOT NULL
                GROUP BY batch_id
                ORDER BY MAX(processing_date) DESC
            """)

            if not batch_stats.empty:
                st.dataframe(batch_stats, use_container_width=True)

                # For each batch, show processing status
                selected_batch = st.selectbox("Select Batch for Details",
                                         batch_stats['batch_id'].tolist())

                if selected_batch:
                    # Get processing status by batch ID
                    status = db_manager.get_latest_processing_status(selected_batch)

                    # Display status information
                    col1_status, col2_status, col3_status = st.columns(3)
                    with col1_status:
                        st.metric("Status", status['status'].upper())
                    with col2_status:
                        st.metric("Completion", f"{status['pct_complete']:.1f}%")
                    with col3_status:
                        st.metric("Events", status['total'])

                    st.markdown(f"**Message**: {status['message']}")

                    # Show detailed status by processing_status
                    status_query = f"""
                        SELECT processing_status, COUNT(*) as count
                        FROM {settings.DB_TABLE_PROCESSED_DATA}
                        WHERE batch_id = %s
                        GROUP BY processing_status
                    """
                    # Pass params as a tuple for psycopg2 compatibility
                    status_df = execute_custom_query(status_query, params=(selected_batch,))

                    if not status_df.empty:
                        st.subheader("Detailed Status")
                        st.dataframe(status_df)
            else:
                st.info("No batch processing data found")

        except Exception as e:
            st.error(f"Error fetching batch statistics: {e}")

        # Calendar Files
        st.markdown("### Calendar Files")
        try:
            calendar_files = execute_custom_query("""
                SELECT id, filename, upload_date, processed, batch_id, is_current
                FROM calendar_files
                ORDER BY upload_date DESC
            """)

            if not calendar_files.empty:
                st.dataframe(calendar_files)
            else:
                st.info("No calendar files found in the database")
        except Exception as e:
            st.error(f"Error fetching calendar files: {e}")

        # Admin Actions Section
        st.markdown("---")
        st.markdown("### Admin Actions")

        # Initialize session state for confirmation
        if 'confirm_clear_db' not in st.session_state:
            st.session_state.confirm_clear_db = False

        clear_db_button = st.button("Clear Processed Event Data", type="secondary")

        if clear_db_button:
            st.session_state.confirm_clear_db = True

        if st.session_state.confirm_clear_db:
            st.warning("⚠️ **Are you sure you want to clear all processed event data?** This action cannot be undone.")

            col_confirm, col_cancel = st.columns(2)
            with col_confirm:
                if st.button("Yes, Clear Database", type="primary"):
                    st.session_state.confirm_clear_db = False # Reset confirmation state

                    # Call the function to clear the database
                    success = db_manager.clear_database_tables()

                    if success:
                        st.success("Database tables cleared successfully!")
                        st.rerun() # Rerun the page to reflect changes
                    else:
                        st.error("Failed to clear database tables. Check logs for details.")
            with col_cancel:
                if st.button("Cancel"):
                    st.session_state.confirm_clear_db = False
                    st.rerun() # Rerun to hide confirmation

# Close connection when done
if conn:
    conn.close()
