"""
Database operations for tracking LLM information used for processing.
"""
import logging
import json
from psycopg2 import sql
from .connection import get_db_connection
from .schema import ensure_tables_exist
from config import settings

logger = logging.getLogger(__name__)

def store_batch_llm_info(batch_id, provider, model, params=None):
    """
    Store information about the LLM used for processing a batch.
    
    Args:
        batch_id: The batch ID
        provider: LLM provider (e.g., 'ollama', 'openai')
        model: Model name used
        params: Optional parameters used for LLM configuration
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    if not batch_id:
        logger.error("Cannot store LLM info: No batch_id provided")
        return False
        
    conn = get_db_connection()
    if not conn:
        return False
        
    try:        # Ensure the llm_info table exists with proper constraint
        with conn.cursor() as cursor:
            # First create the table if it doesn't exist
            cursor.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    id SERIAL PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    params JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                )
            """).format(sql.Identifier("llm_info")))
            
            # Add unique constraint if it doesn't exist
            # First check if constraint exists
            cursor.execute("""
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'llm_info_batch_id_key'
                LIMIT 1
            """)
            constraint_exists = cursor.fetchone() is not None
            
            if not constraint_exists:
                try:
                    cursor.execute(sql.SQL("""
                        ALTER TABLE {} ADD CONSTRAINT llm_info_batch_id_key 
                        UNIQUE (batch_id)
                    """).format(sql.Identifier("llm_info")))
                    logger.info("Added unique constraint to llm_info table")
                except Exception as e:
                    logger.warning(f"Could not add unique constraint: {e}")
            
            # Store LLM info
            if params and not isinstance(params, str):
                params_json = json.dumps(params)
            else:
                params_json = params
                
            cursor.execute(sql.SQL("""
                INSERT INTO {} (batch_id, provider, model_name, params)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (batch_id) 
                DO UPDATE SET 
                    provider = EXCLUDED.provider,
                    model_name = EXCLUDED.model_name,
                    params = EXCLUDED.params,
                    created_at = NOW()
            """).format(sql.Identifier("llm_info")), 
            (batch_id, provider, model, params_json))
            
        conn.commit()
        logger.info(f"Successfully stored LLM info for batch {batch_id}: {provider}/{model}")
        return True
    except Exception as e:
        logger.error(f"Error storing LLM info: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_batch_llm_info(batch_id):
    """
    Retrieve information about the LLM used for a specific batch.
    
    Args:
        batch_id: The batch ID
        
    Returns:
        dict: LLM information or None if not found
    """
    if not batch_id:
        logger.error("Cannot retrieve LLM info: No batch_id provided")
        return None
        
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        with conn.cursor() as cursor:
            # Check if table exists first
            cursor.execute("""
                SELECT EXISTS (
                   SELECT FROM information_schema.tables 
                   WHERE table_name = 'llm_info'
                )
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                logger.warning("llm_info table does not exist yet")
                return None
                
            cursor.execute(sql.SQL("""
                SELECT provider, model_name, params, created_at
                FROM {} 
                WHERE batch_id = %s
                ORDER BY created_at DESC
                LIMIT 1
            """).format(sql.Identifier("llm_info")), (batch_id,))
            
            result = cursor.fetchone()
            
            if result:
                provider, model, params, created_at = result
                return {
                    'provider': provider,
                    'model': model,
                    'params': json.loads(params) if params else None,
                    'created_at': created_at.isoformat() if created_at else None
                }
            return None
    except Exception as e:
        logger.error(f"Error retrieving LLM info: {e}")
        return None
    finally:
        conn.close()
