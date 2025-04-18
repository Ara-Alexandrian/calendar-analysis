"""
Database operations for the Upload & Process page.
Contains functions for interacting with the database.
"""
import streamlit as st
import pandas as pd
import logging
import time
import uuid

logger = logging.getLogger(__name__)

def check_file_in_database(db_manager, filename, file_content):
    """
    Check if a file already exists in the database.
    
    Args:
        db_manager: Database manager module
        filename: Name of the file
        file_content: Content of the file as bytes
        
    Returns:
        tuple: (success, batch_id, is_duplicate)
    """
    try:
        success, batch_id, is_duplicate = db_manager.save_calendar_file_to_db(
            filename=filename,
            file_content=file_content,
            check_only=True  # Only check, don't save yet
        )
        logger.info(f"Database check result: success={success}, batch_id={batch_id}, is_duplicate={is_duplicate}")
        return success, batch_id, is_duplicate
    except Exception as e:
        logger.error(f"Error checking file in database: {e}", exc_info=True)
        st.error(f"Database error: {e}")
        return False, None, False

def save_file_to_database(db_manager, filename, file_content, settings):
    """
    Save a file to the database and record LLM information.
    
    Args:
        db_manager: Database manager module
        filename: Name of the file
        file_content: Content of the file as bytes
        settings: Application settings module
        
    Returns:
        tuple: (batch_id, is_duplicate)
    """
    try:
        success, batch_id, is_duplicate = db_manager.save_calendar_file_to_db(
            filename=filename,
            file_content=file_content
        )
        
        if success:
            st.session_state.current_batch_id = batch_id
            
            # Save LLM provider information
            if settings.LLM_PROVIDER and settings.LLM_PROVIDER.lower() != 'none':
                llm_info = {
                    'base_url': settings.OLLAMA_BASE_URL if hasattr(settings, 'OLLAMA_BASE_URL') else None,
                    'temperature': settings.LLM_TEMPERATURE if hasattr(settings, 'LLM_TEMPERATURE') else 0.7,
                    'max_tokens': settings.LLM_MAX_TOKENS if hasattr(settings, 'LLM_MAX_TOKENS') else 1024
                }
                db_manager.store_batch_llm_info(
                    batch_id=batch_id,
                    provider=settings.LLM_PROVIDER,
                    model=settings.LLM_MODEL,
                    params=llm_info
                )
                logger.info(f"Stored LLM info for batch {batch_id}: {settings.LLM_PROVIDER}/{settings.LLM_MODEL}")
            
            if not is_duplicate:
                st.info(f"New file '{filename}' registered with Batch ID: {batch_id}")
                st.session_state.processing_mode = "Start Fresh"  # Always start fresh for new files
            
            return batch_id, is_duplicate
        else:
            st.error("Failed to register file in database. Processing will continue but might lack DB tracking.")
            # Assign a temporary batch ID if DB save fails
            temp_batch_id = f"temp_batch_{uuid.uuid4().hex[:8]}"
            st.session_state.current_batch_id = temp_batch_id
            return temp_batch_id, False
            
    except Exception as e:
        logger.error(f"Error in save_calendar_file_to_db: {e}", exc_info=True)
        st.error(f"Error registering file in database: {e}")
        
        # Assign a temporary batch ID if DB save fails
        temp_batch_id = f"temp_batch_{uuid.uuid4().hex[:8]}"
        st.session_state.current_batch_id = temp_batch_id
        return temp_batch_id, False

def ensure_database_schema(db_manager):
    """
    Ensure the database schema exists.
    
    Args:
        db_manager: Database manager module
        
    Returns:
        bool: True if schema exists or was created successfully, False otherwise
    """
    st.write("Verifying database schema...")
    if not db_manager.ensure_tables_exist():
        st.error("Failed to verify or update database schema. Aborting processing.")
        return False
    else:
        st.write("Database schema verified.")
        return True

def save_partial_processed_data(db_manager, batch_data, batch_id):
    """
    Save a batch of processed data to the database.
    
    Args:
        db_manager: Database manager module
        batch_data: List of processed event data dictionaries
        batch_id: Batch ID
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        batch_df_to_save = pd.DataFrame(batch_data)
        # Use save_partial_processed_data which handles the required columns
        save_success = db_manager.save_partial_processed_data(batch_df_to_save, batch_id=batch_id)
        if save_success:
            logger.info(f"Successfully saved intermediate batch of {len(batch_data)} records for batch {batch_id}.")
            return True
        else:
            logger.error(f"Failed to save intermediate batch for batch {batch_id}.")
            return False
    except Exception as e:
        logger.error(f"Error saving intermediate batch: {e}", exc_info=True)
        st.error(f"Error saving intermediate batch: {e}")
        return False

def get_processed_events(db_manager, batch_id):
    """
    Get processed events from the database.
    
    Args:
        db_manager: Database manager module
        batch_id: Batch ID
        
    Returns:
        DataFrame: Processed events or None if not found
    """
    try:
        logger.info(f"Fetching processed data from DB for batch {batch_id}")
        processed_df = db_manager.get_processed_events_by_batch(batch_id)
        
        if processed_df is None or processed_df.empty:
            logger.error(f"Failed to fetch back processed data for batch {batch_id}. Normalization might fail.")
            return None
        else:
            # Ensure the 'extracted_personnel' column is list type if loaded from DB
            if 'extracted_personnel' in processed_df.columns:
                processed_df['extracted_personnel'] = processed_df['extracted_personnel'].apply(
                    lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else ['Unknown']
                )
            logger.info(f"Successfully fetched {len(processed_df)} records from DB for final processing.")
            return processed_df
    except Exception as e:
        logger.error(f"Error fetching processed events: {e}", exc_info=True)
        return None

def save_analysis_data(db_manager, df, batch_size=500):
    """
    Save analysis-ready data to the database in batches.
    
    Args:
        db_manager: Database manager module
        df: DataFrame to save
        batch_size: Number of records to save in each batch
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        # Use the new batch saving function with progress indicators
        total_records = len(df)
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Save in batches with visual feedback
        for i in range(0, total_records, batch_size):
            end_idx = min(i + batch_size, total_records)
            batch = df.iloc[i:end_idx]
            
            # Log the save attempt
            logger.info(f"Saving batch {i//batch_size + 1}/{(total_records//batch_size) + 1} ({len(batch)} records)")
            
            # Update progress
            progress_percent = (end_idx / total_records)
            progress_bar.progress(progress_percent)
            status_text.text(f"Saving records {i+1} to {end_idx} of {total_records}...")
            
            try:
                # Call the database save function (implementation depends on your DB structure)
                db_manager.save_processed_data(batch)
            except Exception as batch_error:
                logger.error(f"Error saving batch {i//batch_size + 1}: {batch_error}")
                st.error(f"Error saving batch {i//batch_size + 1}: {batch_error}")
                return False
        
        # Complete progress bar and show success
        progress_bar.progress(1.0)
        status_text.text(f"Successfully saved all {total_records} records to database.")
        return True
    except Exception as e:
        logger.error(f"Error saving analysis data: {e}", exc_info=True)
        st.error(f"Database error during final save: {e}")
        return False
