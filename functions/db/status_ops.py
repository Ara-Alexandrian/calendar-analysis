# functions/db/status_ops.py
import logging
import psycopg2
from psycopg2 import sql
from config import settings
from .connection import get_db_connection

logger = logging.getLogger(__name__)

def get_latest_processing_status(batch_id=None):
    """
    Get the status of the latest processing batch or a specific batch.

    Args:
        batch_id: Optional batch ID to filter by. If None, gets the latest batch.

    Returns:
        dict: Processing status information including completion percentage,
              or a default status if DB is unavailable or no batches found.
    """
    conn = get_db_connection()
    if not conn:
        return {"status": "unknown", "message": "Database not available", "total": 0, "pct_complete": 0}

    try:
        with conn.cursor() as cursor:
            if batch_id:
                # Get status for specific batch
                cursor.execute(sql.SQL("""
                    SELECT processing_status, COUNT(*) as count
                    FROM {} WHERE batch_id = %s
                    GROUP BY processing_status
                """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)), (batch_id,))
                target_batch_id = batch_id
            else:
                # Find the latest batch_id first
                cursor.execute(sql.SQL("""
                    SELECT batch_id FROM {}
                    WHERE batch_id IS NOT NULL
                    ORDER BY processing_date DESC
                    LIMIT 1
                """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)))
                latest_batch_result = cursor.fetchone()

                if not latest_batch_result:
                    logger.info("No processing batches found in the database.")
                    return {"status": "none", "message": "No processing batches found", "total": 0, "pct_complete": 0}

                target_batch_id = latest_batch_result[0]
                logger.info(f"Latest batch ID found: {target_batch_id}")

                # Get status counts for the latest batch
                cursor.execute(sql.SQL("""
                    SELECT processing_status, COUNT(*) as count
                    FROM {} WHERE batch_id = %s
                    GROUP BY processing_status
                """).format(sql.Identifier(settings.DB_TABLE_PROCESSED_DATA)), (target_batch_id,))

            results = cursor.fetchall()

            if not results:
                 # This might happen if the batch exists but has no rows yet, or only null statuses
                 logger.warning(f"No status counts found for batch ID: {target_batch_id}")
                 return {"status": "starting", "message": f"Batch {target_batch_id} found but no events processed yet.", "total": 0, "pct_complete": 0, "batch_id": target_batch_id}


            # Calculate completion statistics
            status_counts = {status: count for status, count in results}
            total = sum(status_counts.values())
            # Define what counts as 'in progress' vs 'complete'
            # Assuming 'extracted' and 'assigned' mean completed steps for this status check
            processing = status_counts.get('processing', 0) # Still needs LLM extraction
            extracted = status_counts.get('extracted', 0)   # LLM done, needs assignment
            assigned = status_counts.get('assigned', 0)     # Fully processed for this stage

            # Calculate percentage based on steps potentially completed (extracted + assigned)
            completed_steps = extracted + assigned
            pct_complete = (completed_steps / total * 100) if total > 0 else 0

            # Determine overall status message
            if processing == 0 and total > 0:
                status = "complete"
                message = f"Processing complete for batch {target_batch_id}: {total} events processed."
            elif total > 0:
                status = "in_progress"
                # More detailed message about stages
                message = f"Batch {target_batch_id} in progress: {pct_complete:.1f}% steps complete ({processing} awaiting extraction, {extracted} awaiting assignment, {assigned} assigned)."
            else:
                 # Should be caught by the earlier 'no results' check, but as a fallback
                status = "unknown"
                message = f"Unknown processing status for batch {target_batch_id}."

            return {
                "status": status,
                "message": message,
                "total": total,
                "processing": processing,
                "extracted": extracted,
                "assigned": assigned,
                "pct_complete": pct_complete,
                "batch_id": target_batch_id # Return the actual batch ID checked
            }

    except psycopg2.Error as db_e:
        logger.error(f"Database error getting processing status: {db_e}", exc_info=True)
        return {"status": "error", "message": f"DB Error: {str(db_e)}", "total": 0, "pct_complete": 0}
    except Exception as e:
        logger.error(f"Unexpected error getting processing status: {e}", exc_info=True)
        return {"status": "error", "message": f"Error: {str(e)}", "total": 0, "pct_complete": 0}
    finally:
        if conn:
            conn.close()
