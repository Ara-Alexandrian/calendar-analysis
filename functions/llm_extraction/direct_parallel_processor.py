"""
Direct parallel processor optimized for powerful GPU setups.
Designed specifically for dual RTX 3090s in NVLINK configuration to handle LLM extraction
without unnecessary batching.
"""
import logging
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import settings
from .utils import get_persistent_progress_bar
from .extractor import _extract_single_physicist_llm
# Import the function to get the client, not a non-existent class
from .ollama_client import get_ollama_client

# Configure logging
logger = logging.getLogger(__name__)

class DirectParallelProcessor:
    """
    Optimized processor for direct LLM extraction taking advantage of dual RTX 3090s in NVLINK.
    Maximizes throughput for powerful GPU setups without unnecessary batching.
    """
    
    def __init__(self, max_workers=None, progress_callback=None):
        """
        Initialize the processor.
        
        Args:
            max_workers: Maximum number of concurrent workers (default: auto-configured)
            progress_callback: Optional callback function to report progress
        """
        # Auto-configure max_workers based on hardware
        if max_workers is None:
            # Take full advantage of dual RTX 3090s in NVLINK
            # 6 is a good starting point for dual RTX 3090s to avoid overwhelming them
            self.max_workers = getattr(settings, "LLM_MAX_WORKERS", 6)
        else:
            self.max_workers = max_workers
            
        self.progress_callback = progress_callback
        # Get the client instance using the function
        self.client = get_ollama_client()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Performance tracking
        self._times = []
        self._errors = 0
        
    def _extract_with_retry(self, summary, canonical_names, max_retries=2):
        """
        Extract personnel with retry logic.
        
        Args:
            summary: Event summary text
            canonical_names: List of canonical names
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of extracted personnel names
        """
        # Try extraction up to max_retries+1 times
        for attempt in range(max_retries + 1):
            try:
                result = _extract_single_physicist_llm(summary, self.client, canonical_names)
                return result
            except Exception as e:
                logger.warning(f"Extraction attempt {attempt+1} failed: {e}")
                if attempt < max_retries:
                    # Exponential backoff
                    time.sleep(1 * (2 ** attempt))
                else:
                    logger.error(f"All {max_retries+1} extraction attempts failed")
                    raise Exception(f"Extraction failed after {max_retries+1} attempts: {str(e)}")

    # Correct indentation for the method definition
    def process_dataframe(self, df, summary_col="summary", canonical_names=None):
        """
        Process a dataframe using the direct parallel processor.
        Submits all events concurrently to the LLM via a thread pool.
        
        Args:
            df: Input dataframe
            summary_col: Column containing text to process
            canonical_names: List of canonical names to extract
            
        Returns:
            Dataframe with added 'extracted_personnel' column
        """
        if not canonical_names:
            logger.error("No canonical names provided")
            return df
            
        df_copy = df.copy()
        df_copy[summary_col] = df_copy[summary_col].fillna('')
        
        summaries = df_copy[summary_col].tolist()
        total_items = len(summaries)
        
        if total_items == 0:
            logger.warning("Empty dataframe, nothing to process")
            return df_copy
            
        logger.info(f"Processing {total_items} events one by one sequentially")
        
        # Initialize results storage
        results = [None] * total_items
        
        # Process items one by one sequentially
        completed_count = 0
        progress_bar = get_persistent_progress_bar(total_items, "Direct Parallel LLM Processing")
        
        # Use ThreadPoolExecutor for parallel processing
        results = [None] * total_items # Pre-allocate results list
        futures = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            for idx, summary in enumerate(summaries):
                if not summary: # Skip empty summaries
                    results[idx] = ["Unknown"]
                    continue
                future = executor.submit(self._extract_with_retry, summary, canonical_names)
                futures[future] = idx # Map future back to original index

            # Process completed futures
            completed_count = 0
            start_time = time.time()
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    item_result = future.result()
                    results[idx] = item_result
                    logger.debug(f"Event {idx+1}: Extracted personnel: {item_result}")
                except Exception as e:
                    logger.error(f"Error processing item {idx}: {e}")
                    results[idx] = ["Unknown_Error"]
                    with self._lock:
                        self._errors += 1
                finally:
                    completed_count += 1
                    progress_bar.update(1)
                    if self.progress_callback:
                        self.progress_callback(completed_count, total_items)

        progress_bar.close()
        end_time = time.time()
        logger.info(f"Parallel processing finished in {end_time - start_time:.2f} seconds.")
        # Log performance statistics
        error_rate = (self._errors / max(1, total_items)) * 100
        logger.info(f"Error rate: {error_rate:.2f}% ({self._errors} errors)")
        
        # Ensure all results are set
        for i in range(total_items):
            if results[i] is None:
                results[i] = ["Unknown_Error"]
                
        # Add results to dataframe
        df_copy['extracted_personnel'] = results
        return df_copy
