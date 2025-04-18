"""
Data processing operations for the Upload & Process page.
Contains functions for processing calendar data.
"""
import streamlit as st
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

def load_and_process_data(data_processor, file_content, filename):
    """
    Load raw data from file content.
    
    Args:
        data_processor: Data processor module
        file_content: File content as bytes
        filename: Name of the file
        
    Returns:
        DataFrame: Raw calendar data or None if loading failed
    """
    try:
        raw_df = data_processor.load_raw_calendar_data(file_content, filename)
        
        if raw_df is not None:
            st.session_state.raw_df = raw_df
            return raw_df
        else:
            st.error(f"Failed to load or parse '{filename}'. Check file format and logs.")
            st.session_state.raw_df = None  # Clear invalid data
            st.session_state.uploaded_filename = None
            return None
    except Exception as e:
        logger.error(f"Error loading raw data: {e}", exc_info=True)
        st.error(f"An error occurred loading the file: {e}")
        return None

def preprocess_data(data_processor, raw_df):
    """
    Preprocess the raw calendar data.
    
    Args:
        data_processor: Data processor module
        raw_df: Raw calendar data DataFrame
        
    Returns:
        DataFrame: Preprocessed data or None if preprocessing failed
    """
    try:
        st.write("Step 1: Preprocessing data (dates, duration)...")
        preprocessed_df = data_processor.preprocess_data(raw_df)
        
        if preprocessed_df is None or preprocessed_df.empty:
            st.error("Preprocessing failed or resulted in empty data. Check logs.")
            return None
        
        st.session_state.preprocessed_df = preprocessed_df
        st.write(f"Preprocessing done: {len(preprocessed_df)} events remaining.")
        logger.info(f"Preprocessing complete: {len(preprocessed_df)} events.")
        
        return preprocessed_df
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}", exc_info=True)
        st.error(f"Preprocessing error: {e}")
        return None

def normalize_personnel(normalizer_fn, llm_processed_df):
    """
    Normalize extracted personnel names.
    
    Args:
        normalizer_fn: Function to normalize personnel names
        llm_processed_df: DataFrame with LLM-extracted personnel
        
    Returns:
        DataFrame: Normalized data or None if normalization failed
    """
    try:
        st.write("Step 3: Normalizing extracted personnel names...")
        
        # Ensure llm_processed_df is a DataFrame before proceeding
        if not isinstance(llm_processed_df, pd.DataFrame):
            logger.error(f"llm_processed_df is not a DataFrame, it's a {type(llm_processed_df)}. Falling back.")
            st.error(f"Internal error: LLM processing step did not return a DataFrame. Using default values.")
            # Fallback: Create DataFrame with 'Unknown' assigned personnel
            if 'preprocessed_df' in st.session_state and st.session_state.preprocessed_df is not None:
                temp_df = st.session_state.preprocessed_df.copy()
                temp_df['extracted_personnel'] = [['Unknown']] * len(temp_df)
                temp_df['assigned_personnel'] = [['Unknown']] * len(temp_df)
                st.session_state.normalized_df = temp_df
                return temp_df
            else:
                return None
                
        # Proceed with normalization
        normalized_df = normalizer_fn(llm_processed_df)
        st.session_state.normalized_df = normalized_df
        st.write("Normalization finished.")
        logger.info("Normalization complete.")
        
        return normalized_df
    except Exception as e:
        logger.error(f"Normalization error: {e}", exc_info=True)
        st.error(f"Error during normalization: {e}. Using default values.")
        # Fallback: Create a simple normalized_df with assigned_personnel column
        temp_df = llm_processed_df.copy()
        temp_df['assigned_personnel'] = [['Unknown']] * len(temp_df)
        st.session_state.normalized_df = temp_df
        return temp_df

def explode_data_for_analysis(data_processor, normalized_df):
    """
    Explode the normalized data by personnel for analysis.
    
    Args:
        data_processor: Data processor module
        normalized_df: Normalized DataFrame
        
    Returns:
        DataFrame: Analysis-ready data or None if explosion failed
    """
    try:
        st.write("Step 4: Preparing data for analysis (exploding by personnel)...")
        
        # Use the 'assigned_personnel' column created by normalization
        if normalized_df is not None and 'assigned_personnel' in normalized_df.columns:
            analysis_ready_df = data_processor.explode_by_personnel(normalized_df, personnel_col='assigned_personnel')
            
            if analysis_ready_df is None or analysis_ready_df.empty:
                st.error("Exploding data failed or resulted in empty data. Check logs.")
                return None
                
            st.session_state.analysis_ready_df = analysis_ready_df
            st.write(f"Data ready for analysis: {len(analysis_ready_df)} rows after exploding.")
            logger.info(f"Explosion complete: {len(analysis_ready_df)} analysis rows.")
            
            return analysis_ready_df
        else:
            st.error("Cannot prepare data for analysis: 'assigned_personnel' column missing after normalization.")
            return None
    except Exception as e:
        logger.error(f"Error exploding data: {e}", exc_info=True)
        st.error(f"Error preparing data for analysis: {e}")
        return None

def store_analysis_data(analysis_ready_df, filename):
    """
    Store the analysis-ready data in session state for use in other pages.
    
    Args:
        analysis_ready_df: Analysis-ready DataFrame
        filename: Original filename
        
    Returns:
        str: Key of the stored batch
    """
    # Always store the analysis dataframe in session state for the Analysis page
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {}
    
    # Store the current batch with a timestamp for the Analysis page
    current_batch_key = f"batch_{int(time.time())}"  # Use timestamp as part of key
    st.session_state.analysis_data[current_batch_key] = {
        'df': analysis_ready_df,
        'timestamp': time.time(),
        'source': filename,
        'processed_count': len(analysis_ready_df)
    }
    
    # Store the most recent batch ID for easy access
    st.session_state.most_recent_batch = current_batch_key
    
    return current_batch_key
