#!/usr/bin/env python3
"""
Debug script to check database connection and schema
"""
import logging
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"Added {PROJECT_ROOT} to Python path")

try:
    # Import settings and database manager
    from config import settings
    # from functions import db as db_manager  # Old import
    from src.infrastructure.persistence import connection, operations, schema, personnel_ops, status_ops, llm_info # New import
    import pandas as pd
    import psycopg2

    # Print database settings
    print("\n=== Database Configuration ===")
    print(f"DB_ENABLED: {settings.DB_ENABLED}")
    print(f"DB_HOST: {settings.DB_HOST}")
    print(f"DB_PORT: {settings.DB_PORT}")
    print(f"DB_NAME: {settings.DB_NAME}")
    print(f"DB_USER: {settings.DB_USER}")
    print(f"DB_PASSWORD: {'*' * len(settings.DB_PASSWORD)}")
    print(f"DB_TABLE_PROCESSED_DATA: {settings.DB_TABLE_PROCESSED_DATA}")

    # Test database connection
    print("\n=== Testing Connection ===")
    conn = connection.get_db_connection() # Updated call
    if conn:
        print("✅ Database connection successful!")
        
        # List all databases
        print("\n=== Available Databases ===")
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT datname FROM pg_database WHERE datistemplate = false;")
                databases = cursor.fetchall()
                for db in databases:
                    print(f"- {db[0]}")
        except Exception as e:
            print(f"❌ Error listing databases: {e}")
            
        # Check current database
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT current_database();")
                current_db = cursor.fetchone()[0]
                print(f"\nCurrently connected to database: {current_db}")
                
                # Check for case sensitivity issues
                if current_db.lower() != settings.DB_NAME.lower():
                    print(f"⚠️ Warning: Case mismatch! Connected to {current_db} but config specifies {settings.DB_NAME}")
        except Exception as e:
            print(f"❌ Error checking current database: {e}")
        
        # List all tables
        print("\n=== Tables in Current Database ===")
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                tables = cursor.fetchall()
                if tables:
                    for table in tables:
                        print(f"- {table[0]}")
                else:
                    print("No tables found! Tables need to be created.")
        except Exception as e:
            print(f"❌ Error listing tables: {e}")
            
        # Check if tables need to be created
        if not tables:
            print("\n=== Creating Tables ===")
            success = schema.ensure_tables_exist() # Updated call
            print(f"Table creation {'succeeded' if success else 'failed'}")

            # Check again after creation
            if success:
                conn = connection.get_db_connection()  # Updated call
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        ORDER BY table_name;
                    """)
                    tables = cursor.fetchall()
                    print("\n=== Tables After Creation ===")
                    for table in tables:
                        print(f"- {table[0]}")
                        
        conn.close()
    else:
        print("❌ Database connection failed!")
        
        # Try direct connection to see more specific error
        print("\n=== Attempting Direct Connection ===")
        try:
            import urllib.parse
            encoded_password = urllib.parse.quote_plus(settings.DB_PASSWORD)
            conn_string = f"postgresql://{settings.DB_USER}:{encoded_password}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
            print(f"Connection string (password masked): postgresql://{settings.DB_USER}:******@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
            conn = psycopg2.connect(conn_string)
            print("✅ Direct connection successful! Problem may be in db_manager.py")
            conn.close()
        except Exception as e:
            print(f"❌ Direct connection failed: {e}")
            
            # Try lowercase database name
            try:
                conn_string = f"postgresql://{settings.DB_USER}:{encoded_password}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME.lower()}"
                print(f"Trying lowercase: postgresql://{settings.DB_USER}:******@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME.lower()}")
                conn = psycopg2.connect(conn_string)
                print("✅ Connection with lowercase DB name successful! Case sensitivity issue confirmed.")
                conn.close()
            except Exception as e:
                print(f"❌ Lowercase attempt failed: {e}")
                
            # Try connection to postgres default database
            try:
                conn_string = f"postgresql://{settings.DB_USER}:{encoded_password}@{settings.DB_HOST}:{settings.DB_PORT}/postgres"
                print(f"Trying postgres default DB: postgresql://{settings.DB_USER}:******@{settings.DB_HOST}:{settings.DB_PORT}/postgres")
                conn = psycopg2.connect(conn_string)
                print("✅ Connection to default 'postgres' database successful! SPC_CALENDAR database may not exist.")
                
                # Try to create the database
                conn.autocommit = True
                with conn.cursor() as cursor:
                    try:
                        cursor.execute(f"CREATE DATABASE {settings.DB_NAME};")
                        print(f"✅ Successfully created database '{settings.DB_NAME}'")
                    except Exception as e:
                        print(f"❌ Failed to create database: {e}")
                conn.close()
            except Exception as e:
                print(f"❌ Default database connection failed: {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
